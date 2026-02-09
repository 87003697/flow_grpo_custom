# hybrid_normal_renderer.py

"""
混合 Normal 渲染器：Voxel Normal (subs 可微) + Mesh Rendering (高质量)

使用 grid_sample_3d 进行三线性插值，对 sub_logits 可微。

梯度路径:
  - subs → center_occ / neighbor_occ → occupancy_diff → normal → pixel_normals

调用栈:
  render()  ← 线性 5 步，无分支
  ├── 1. _transform_vertices(...)       → vertices_clip, vertices_cam, vertices_batch
  ├── 2. _rasterize(...)                → rast_ctx (内部自动选单次/分块)
  ├── 3. _prepare_voxel_normals(...)    → voxel_normals
  ├── 4. _render_pixels(rast_ctx, ...)  → out_dict (内部自动选单次/分块)
  │      └── _render_one_pass(...)      ← 核心渲染，单次调一次，分块循环调
  └── 5. _downsample(out_dict)          → final

使用方法:
    renderer = Hybrid26NormalRenderer({"resolution": 512})
    outputs = renderer.render(mesh, subs, coords, extrinsics, intrinsics, voxel_resolution)
    normal = outputs.normal  # (H, W, 3)
"""

from typing import List, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from flex_gemm.ops.grid_sample import grid_sample_3d

from .diff_voxel_normal_neighbor26 import (
    _neighbor_offsets_26,
    _compute_neighbor_occupancy_soft,
    _flip_normals_to_camera,
)


# =============================================================================
# 常量
# =============================================================================

_RAST_CHUNK_RATIO = 4  # rast_chunk = chunk_size × ratio


# =============================================================================
# 辅助函数（模块级）
# =============================================================================

def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """计算 vertex normals（世界坐标系）

    Args:
        vertices: (N, 3) mesh 顶点（世界坐标）
        faces: (F, 3) 面索引

    Returns:
        v_normals: (N, 3) 每个顶点的法向量（世界坐标系）
    """
    i0 = faces[..., 0].long()  # (F,)
    i1 = faces[..., 1].long()  # (F,)
    i2 = faces[..., 2].long()  # (F,)

    v0 = vertices[i0, :]  # (F, 3)
    v1 = vertices[i1, :]  # (F, 3)
    v2 = vertices[i2, :]  # (F, 3)

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)

    v_normals = torch.zeros_like(vertices)  # (N, 3)
    v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
    v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
    v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

    v_normals = F.normalize(v_normals, dim=1, eps=1e-6)  # (N, 3)
    return v_normals


def _flip_normals_world(
    normals: Tensor,       # (K, 3)
    surface_pos: Tensor,   # (K, 3)
    extrinsics: Tensor,    # (4, 4)
) -> Tensor:
    """在世界坐标系下翻转法向量，使其与视线方向一致。

    dot(normal, view_dir) > 0 → 保持；≤ 0 → 翻转。
    """
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    cam_pos = -(R.T @ t)    # (3,)
    view_dir = surface_pos - cam_pos  # (K, 3)
    dot = (normals * view_dir).sum(dim=-1, keepdim=True)  # (K, 1)
    return torch.where(dot > 0, normals, -normals)  # (K, 3)


def compute_voxel_normal(
    coords: Tensor,          # (K, 3)
    subs: List[Any],
    ref_normal: Tensor,      # (K, 3) 世界坐标系，翻转后的参考方向
    voxel_resolution: int,
) -> Tensor:
    """计算 26-neighbor occupancy 差分法向量（对 subs 可微）

    使用 center_occ - neighbor_occ 有限差分估计 ∇occupancy，
    梯度同时流向中心和邻居的 sub_logits。

    Args:
        coords: (K, 3) 可见 voxel 坐标（整数）
        subs: 多分辨率 subdivision logits
        ref_normal: (K, 3) 参考法向量（世界坐标系，detach 后使用）
        voxel_resolution: voxel 分辨率

    Returns:
        normal: (K, 3) 可微法向量（世界坐标系）
    """
    device = coords.device
    K = coords.shape[0]
    if K == 0:
        return torch.zeros(0, 3, device=device)

    # 26 邻居偏移和权重
    offsets, dist_weights = _neighbor_offsets_26(device)  # (26, 3), (26,)
    directions = F.normalize(offsets.float(), dim=-1)     # (26, 3)
    weighted_dirs = directions * dist_weights[:, None]    # (26, 3)

    # 构造 27 个查询坐标（中心 + 26 邻居），避免分配冗余中间张量
    offsets_27 = torch.cat([
        torch.zeros(1, 3, dtype=offsets.dtype, device=device), offsets
    ], dim=0)  # (27, 3)
    all_coords = coords[:, None, :] + offsets_27[None, :, :]  # (K, 27, 3)

    # 查询 soft occupancy（对 subs 可微）
    all_occ = _compute_neighbor_occupancy_soft(
        all_coords, subs, voxel_resolution
    )  # (K, 27)

    # occupancy 差（有限差分 ≈ ∇occupancy）
    occupancy_diff = all_occ[:, :1] - all_occ[:, 1:]  # (K, 26)
    del all_occ, all_coords

    # 方向权重（无 relu，反向邻居梯度带负号）
    ref = ref_normal.detach()  # (K, 3) 不让梯度流回 vertices
    direction_weight = torch.einsum('kd,nd->kn', ref, directions)  # (K, 26)

    # 融合加权 + 矩阵乘，避免分配 (K, 26, 3) 中间张量
    combined = occupancy_diff * direction_weight  # (K, 26)
    normal = combined @ weighted_dirs              # (K, 3) = (K, 26) @ (26, 3)

    normal = F.normalize(normal, dim=-1, eps=1e-6)  # (K, 3)
    return normal


def _grid_sample_normal(
    voxel_normals: Tensor,   # (N, 3)
    coords: Tensor,          # (N, 3)
    origin: Tensor,          # (3,)
    voxel_size: float,
    voxel_resolution: int,
    vertices_batch: Tensor,  # (1, V, 3)
    rast: Tensor,            # (1, H, H, 4)
    faces: Tensor,           # (F, 3)
    rast_res: int,
) -> Tensor:
    """grid_sample_3d 渲染 normal（可被 checkpoint 包裹）

    Args:
        voxel_normals: (N, 3) 每个 voxel 的法向量
        coords: (N, 3) voxel 坐标（整数）
        origin: (3,) voxel 网格原点
        voxel_size: voxel 尺寸
        voxel_resolution: voxel 分辨率
        vertices_batch: (1, V, 3) mesh 顶点（世界坐标）
        rast: (1, H, H, 4) 光栅化结果
        faces: (F, 3) 面索引
        rast_res: 光栅化分辨率

    Returns:
        img: (1, rast_res, rast_res, 3) 渲染后的 normal 图像 [0, 1]
    """
    mask = rast[..., -1:] > 0  # (1, H, H, 1)

    xyz = dr.interpolate(vertices_batch, rast, faces)[0]  # (1, H, H, 3)
    xyz_voxel = ((xyz - origin) / voxel_size).reshape(1, -1, 3)  # (1, H*H, 3)
    del xyz

    coords_4d = torch.cat([
        torch.zeros_like(coords[:, :1]), coords
    ], dim=-1)  # (N, 4)

    voxel_shape = torch.Size([
        1, 3, voxel_resolution, voxel_resolution, voxel_resolution
    ])

    pixel_normal = grid_sample_3d(
        voxel_normals, coords_4d, voxel_shape,
        xyz_voxel, mode='trilinear'
    )
    del xyz_voxel

    img = pixel_normal.reshape(1, rast_res, rast_res, 3) * mask  # (1, H, H, 3)
    img = F.normalize(img, dim=-1, eps=1e-6)
    img = (img + 1) / 2  # → [0, 1]
    return img


def intrinsics_to_projection(
    intrinsics: Tensor, near: float, far: float
) -> Tensor:
    """OpenCV intrinsics → OpenGL perspective matrix"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


# =============================================================================
# 渲染器类
# =============================================================================

class Hybrid26NormalRenderer:
    """混合 Normal 渲染器

    使用 26-neighbor occupancy 差分计算可微法向量，
    通过 grid_sample_3d 三线性插值渲染到像素。
    对 sub_logits 端到端可微。

    rendering_options:
        resolution: 渲染分辨率
        near/far: 近远裁剪面
        ssaa: 超采样倍率
        antialias: 是否抗锯齿
        chunk_size: 统一分块大小
            - normal 计算：按 chunk_size 分块
            - 光栅化：按 chunk_size × _RAST_CHUNK_RATIO 分块
        grad_checkpoint: 是否启用 gradient checkpoint（训练时省显存）
    """

    def __init__(self, rendering_options: dict = {}, device: str = "cuda"):
        self.rendering_options = edict({
            "resolution": 512,
            "near": 0.1,
            "far": 100.0,
            "ssaa": 1,
            "antialias": True,
            "chunk_size": 50000,
            "grad_checkpoint": True,
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def render(
        self,
        mesh: Any,
        subs: List[Any],
        coords: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        voxel_resolution: int,
        return_types: List[str] = ["normal", "mask", "depth"],
    ) -> edict:
        """渲染可微法向量

        5 步流水线:
          1. 坐标变换
          2. 光栅化 + 可见性收集
          3. 可微 voxel normal
          4. 像素渲染
          5. SSAA 下采样
        """
        vertices, faces = mesh.vertices, mesh.faces
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return self._empty_result(return_types)

        # 1. 坐标变换
        vertices_clip, vertices_cam, vertices_batch = self._transform_vertices(
            vertices, extrinsics, intrinsics, return_types)

        # 2. 光栅化 + 可见性
        rast_ctx = self._rasterize(vertices_clip, faces)

        # 3. 可微 voxel normal
        voxel_normals = self._prepare_voxel_normals(
            rast_ctx.visible_ids, vertices, faces, coords,
            subs, extrinsics, voxel_resolution, return_types)

        # 4. 像素渲染
        out_dict = self._render_pixels(
            rast_ctx, vertices_clip, vertices_cam, vertices_batch,
            faces, voxel_normals, coords, voxel_resolution, return_types)

        # 5. 下采样
        return self._downsample(out_dict, return_types)

    # ------------------------------------------------------------------
    # Phase 1: 坐标变换
    # ------------------------------------------------------------------

    def _transform_vertices(self, vertices, extrinsics, intrinsics, return_types):
        """世界坐标 → clip space / camera space

        Returns:
            vertices_clip: (1, V, 4)
            vertices_cam: (1, V, 4) | None （仅 depth 需要）
            vertices_batch: (1, V, 3) 世界坐标
        """
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]

        perspective = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        full_proj = (perspective @ extrinsics).unsqueeze(0)  # (1, 4, 4)

        vertices_batch = vertices.unsqueeze(0)  # (1, V, 3)
        vertices_homo = torch.cat([
            vertices_batch, torch.ones_like(vertices_batch[..., :1])
        ], dim=-1)  # (1, V, 4)

        vertices_cam = None
        if "depth" in return_types:
            extrinsics_batch = extrinsics.unsqueeze(0)  # (1, 4, 4)
            vertices_cam = torch.bmm(
                vertices_homo, extrinsics_batch.transpose(-1, -2)
            )  # (1, V, 4)

        vertices_clip = torch.bmm(
            vertices_homo, full_proj.transpose(-1, -2)
        )  # (1, V, 4)
        del vertices_homo

        return vertices_clip, vertices_cam, vertices_batch

    # ------------------------------------------------------------------
    # Phase 2: 光栅化 + 可见性收集
    # ------------------------------------------------------------------

    def _rasterize(self, vertices_clip, faces):
        """光栅化并收集可见顶点（自动选择单次/分块）

        Returns:
            rast_ctx: edict
                .rast         (Tensor | None)  单次时有值，分块时 None
                .visible_ids  (Tensor)         可见顶点索引 (K,)
                .z_buffer     (Tensor | None)  分块时有值，单次时 None
                .is_chunked   (bool)
                .rast_res     (int)
        """
        rast_res = self.rendering_options["resolution"] * self.rendering_options["ssaa"]
        rast_chunk = self.rendering_options["chunk_size"] * _RAST_CHUNK_RATIO
        num_faces = faces.shape[0]

        if num_faces <= rast_chunk:
            # ---- 单次光栅化 ----
            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, faces, (rast_res, rast_res))
            visible_ids = self._collect_visible_vertices(rast, faces)
            return edict(
                rast=rast, visible_ids=visible_ids,
                z_buffer=None, is_chunked=False, rast_res=rast_res)

        # ---- 分块光栅化（仅收集可见性，无插值） ----
        z_buffer = torch.full(
            (1, rast_res, rast_res), float('inf'),
            device=self.device, dtype=torch.float32)
        all_vis = []

        for off in range(0, num_faces, rast_chunk):
            fc = faces[off:off + rast_chunk]
            r, _ = dr.rasterize(
                self.glctx, vertices_clip, fc, (rast_res, rast_res))

            z_filt = (r[..., 3] != 0) & (r[..., 2] < z_buffer)  # (1, H, W)
            z_buffer[z_filt] = r[z_filt][..., 2]

            # 收集本 chunk 中通过 z-test 的可见顶点
            local_ids = r[0, ..., 3][z_filt[0]].long() - 1  # chunk-local face ids
            local_ids = local_ids[local_ids >= 0].unique()
            if local_ids.numel() > 0:
                global_ids = local_ids + off  # → global face ids
                all_vis.append(faces[global_ids].flatten())

        if all_vis:
            visible_ids = torch.cat(all_vis).unique()
        else:
            visible_ids = torch.tensor([], dtype=torch.long, device=self.device)

        return edict(
            rast=None, visible_ids=visible_ids,
            z_buffer=z_buffer, is_chunked=True, rast_res=rast_res)

    def _collect_visible_vertices(self, rast, faces):
        """从光栅化结果收集可见顶点索引 → (K,)"""
        face_ids = rast[0, ..., 3].long() - 1  # (H, W)，0-indexed
        visible_face_ids = face_ids[face_ids >= 0].unique()
        if visible_face_ids.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return faces[visible_face_ids].flatten().unique()  # (K,)

    # ------------------------------------------------------------------
    # Phase 3: 可微 Voxel Normal
    # ------------------------------------------------------------------

    def _prepare_voxel_normals(self, visible_ids, vertices, faces, coords,
                                subs, extrinsics, voxel_resolution, return_types):
        """计算可见 voxel 的可微法向量 → (N, 3) 或 None"""
        N = vertices.shape[0]
        if "normal" not in return_types:
            return None

        voxel_normals = torch.zeros(N, 3, device=self.device)  # (N, 3)
        K = visible_ids.shape[0]
        if K == 0:
            return voxel_normals

        chunk_size = self.rendering_options["chunk_size"]
        use_ckpt = self.rendering_options["grad_checkpoint"]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)

        # 几何 v_normal（世界坐标系）
        v_normal_all = compute_vertex_normals(vertices, faces)  # (N, 3)
        vis_v_normal = v_normal_all[visible_ids]  # (K, 3)
        del v_normal_all

        vis_coords = coords[visible_ids]  # (K, 3)
        vis_pos = (vis_coords.float() + 0.5) * voxel_size + origin  # (K, 3)

        # 世界坐标系下翻转参考方向
        ref_normal = _flip_normals_world(
            vis_v_normal, vis_pos, extrinsics)  # (K, 3)
        del vis_v_normal

        # 分块计算 26-neighbor normal（世界坐标系）
        normal_world = self._compute_normals_chunked(
            vis_coords, subs, ref_normal,
            voxel_resolution, chunk_size, use_ckpt)  # (K, 3)
        del ref_normal

        # 变换到相机坐标系 + 翻转
        normal_cam = _flip_normals_to_camera(
            normal_world, vis_pos, extrinsics)  # (K, 3)
        del normal_world, vis_pos

        voxel_normals[visible_ids] = normal_cam
        del normal_cam
        return voxel_normals

    def _compute_normals_chunked(self, vis_coords, subs, ref_normal,
                                  voxel_resolution, chunk_size, use_ckpt):
        """分块计算法向量，可选 gradient checkpoint"""
        K = vis_coords.shape[0]
        results = []
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            if use_ckpt:
                chunk_result = checkpoint(
                    compute_voxel_normal,
                    vis_coords[start:end], subs,
                    ref_normal[start:end], voxel_resolution,
                    use_reentrant=False,
                )
            else:
                chunk_result = compute_voxel_normal(
                    vis_coords[start:end], subs,
                    ref_normal[start:end], voxel_resolution,
                )
            results.append(chunk_result)
        return torch.cat(results, dim=0)  # (K, 3)

    # ------------------------------------------------------------------
    # Phase 4: 像素渲染
    # ------------------------------------------------------------------

    def _render_pixels(self, rast_ctx, vertices_clip, vertices_cam,
                       vertices_batch, faces, voxel_normals, coords,
                       voxel_resolution, return_types):
        """像素渲染（自动选择单次/分块）"""
        if not rast_ctx.is_chunked:
            # 单次渲染，复用 Phase 2 的 rast
            return self._render_one_pass(
                rast_ctx.rast, faces, vertices_batch, vertices_clip,
                vertices_cam, voxel_normals, coords, voxel_resolution,
                rast_ctx.rast_res, return_types,
                antialias=self.rendering_options["antialias"])

        # ---- 分块渲染 + z-buffer 合并 ----
        rast_chunk = self.rendering_options["chunk_size"] * _RAST_CHUNK_RATIO
        rast_res = rast_ctx.rast_res

        out_dict = self._init_output(rast_res, return_types)
        z_buffer = torch.full(
            (1, rast_res, rast_res), float('inf'),
            device=self.device, dtype=torch.float32)

        for off in range(0, faces.shape[0], rast_chunk):
            faces_chunk = faces[off:off + rast_chunk]
            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, faces_chunk, (rast_res, rast_res))

            z_filter = (rast[..., 3] != 0) & (rast[..., 2] < z_buffer)  # (1, H, W)
            z_buffer[z_filter] = rast[z_filter][..., 2]

            imgs = self._render_one_pass(
                rast, faces_chunk, vertices_batch, vertices_clip,
                vertices_cam, voxel_normals, coords, voxel_resolution,
                rast_res, return_types, antialias=False)

            for rtype in return_types:
                if rtype in imgs:
                    out_dict[rtype][z_filter] = imgs[rtype][z_filter]

        return out_dict

    def _render_one_pass(self, rast, faces, vertices_batch, vertices_clip,
                         vertices_cam, voxel_normals, coords, voxel_resolution,
                         rast_res, return_types, antialias=True):
        """给定一个 rast 渲染所有 return_types（核心渲染函数）

        单次路径直接调用一次，分块路径在循环里反复调用。
        """
        use_ckpt = self.rendering_options["grad_checkpoint"]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)

        out = edict()
        for rtype in return_types:
            if rtype == "normal":
                out[rtype] = self._render_normal_image(
                    voxel_normals, coords, origin, voxel_size,
                    voxel_resolution, vertices_batch, rast, faces,
                    rast_res, use_ckpt)

            elif rtype == "mask":
                img = (rast[..., -1:] > 0).float()  # (1, H, H, 1)
                if antialias:
                    img = dr.antialias(img, rast, vertices_clip, faces)
                out[rtype] = img

            elif rtype == "depth":
                img = dr.interpolate(
                    vertices_cam[..., 2:3].contiguous(), rast, faces
                )[0]  # (1, H, H, 1)
                if antialias:
                    img = dr.antialias(img, rast, vertices_clip, faces)
                out[rtype] = img

        return out

    def _render_normal_image(self, voxel_normals, coords, origin, voxel_size,
                             voxel_resolution, vertices_batch, rast, faces,
                             rast_res, use_ckpt):
        """grid_sample_3d 渲染 normal，可选 gradient checkpoint"""
        if use_ckpt:
            return checkpoint(
                _grid_sample_normal,
                voxel_normals, coords, origin, voxel_size,
                voxel_resolution, vertices_batch, rast, faces, rast_res,
                use_reentrant=False,
            )
        return _grid_sample_normal(
            voxel_normals, coords, origin, voxel_size,
            voxel_resolution, vertices_batch, rast, faces, rast_res,
        )

    def _init_output(self, rast_res, return_types):
        """初始化输出字典（分块渲染用）"""
        out = edict()
        for rtype in return_types:
            if rtype == "normal":
                # 背景 normal = 0.5，与单次路径一致
                out[rtype] = torch.full(
                    (1, rast_res, rast_res, 3), 0.5,
                    device=self.device, dtype=torch.float32)
            elif rtype in ("mask", "depth"):
                out[rtype] = torch.zeros(
                    1, rast_res, rast_res, 1,
                    device=self.device, dtype=torch.float32)
        return out

    # ------------------------------------------------------------------
    # Phase 5: SSAA 下采样
    # ------------------------------------------------------------------

    def _downsample(self, out_dict, return_types):
        """SSAA 下采样 + squeeze"""
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]

        for rtype in return_types:
            if rtype not in out_dict:
                continue
            img = out_dict[rtype]
            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True
                ).squeeze(0).permute(1, 2, 0)  # (H, W, C)
            else:
                img = img.squeeze(0)  # (H, W, C)

            if img.shape[-1] == 1:
                img = img.squeeze(-1)  # (H, W)

            out_dict[rtype] = img

        return out_dict

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _empty_result(self, return_types):
        """mesh 为空时的默认返回"""
        resolution = self.rendering_options["resolution"]
        out = edict()
        for rtype in return_types:
            if rtype == "normal":
                out[rtype] = torch.full(
                    (resolution, resolution, 3), 0.5,
                    dtype=torch.float32, device=self.device)
            else:
                out[rtype] = torch.zeros(
                    (resolution, resolution),
                    dtype=torch.float32, device=self.device)
        return out
