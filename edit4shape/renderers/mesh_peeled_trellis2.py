from typing import *
from functools import partial
import logging
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from easydict import EasyDict as edict
import numpy as np
import utils3d
from trellis2.representations.mesh import Mesh, MeshWithVoxel, MeshWithPbrMaterial, TextureFilterMode, AlphaMode, TextureWrapMode
import torch.nn.functional as F
import nvdiffrast.torch as dr
from nvdiffrec_render.light import EnvironmentLight
from flex_gemm.ops.grid_sample import grid_sample_3d
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap


# =============================================================================
# 常量
# =============================================================================

# DepthPeeler 单次最大面片数。nvdiffrast 内部限制 2^24 ≈ 16.7M，
# 这里取 4M 留足安全余量。面片数超过此值时自动分 chunk 并做 per-pixel 深度归并。
_MAX_FACES_PER_CHUNK = 4_000_000


@torch.no_grad()
def recover_face_axis_and_voxel(
    faces: Tensor,          # (F, 3) int — 三角形顶点索引
    coords: Tensor,         # (N, 3) int — voxel 坐标
    voxel_resolution: int,
) -> Tuple[Tensor, Tensor]:
    """从 mesh 输出反推 per-face 的 axis_id 和 source voxel_id。

    同时支持 train=False（2 faces/quad）和 train=True（4 faces/quad + mid-point）。
    自动检测：faces 中存在 >= N 的顶点索引 → train=True。

    工作在 quad 粒度上：
    - 将连续 fpq 个 face 分组为 1 个 quad
    - 从 quad 角点坐标判定 axis（恒定维度）和 source voxel（逐维 min）
    - repeat_interleave 广播回 per-face

    Args:
        faces: (F, 3) 三角形顶点索引
        coords: (N, 3) voxel 整数坐标
        voxel_resolution: voxel 分辨率（用于 hashmap grid_size）

    Returns:
        face_axis_ids:  (F,) long, 0/1/2 = x/y/z
        face_voxel_ids: (F,) long, 源 voxel 在 coords 中的索引
    """
    device = coords.device
    N = coords.shape[0]
    F_count = faces.shape[0]

    # ---- 自动检测 train 模式 ----
    train = (faces >= N).any().item()
    fpq = 4 if train else 2                                         # faces per quad
    L = F_count // fpq                                              # quad 数

    # ---- 重组为 quad，提取 voxel 角点 ----
    quad_faces = faces.reshape(L, fpq, 3)                           # (L, fpq, 3)
    if train:
        # quad_split_train: [v0,v1,mid], [v1,v2,mid], [v2,v3,mid], [v3,v0,mid]
        # 每个三角形第 0 个顶点 = 角点 voxel（全 < N）
        corner_ids = quad_faces[:, :, 0]                            # (L, 4)
    else:
        # 第一个三角形的 3 个顶点 = quad 的 3/4 角点（全 < N）
        corner_ids = quad_faces[:, 0, :]                            # (L, 3)

    corner_coords = coords[corner_ids]                              # (L, 4or3, 3)

    # ---- Step 1: axis 判定（恒定维度） ----
    r = (corner_coords.max(dim=1).values
         - corner_coords.min(dim=1).values)                         # (L, 3)
    quad_axis = (r == 0).long().argmax(dim=1)                       # (L,)

    # ---- Step 2: 源 voxel 坐标 = 逐维 min ----
    source_coords = corner_coords.min(dim=1).values.int()           # (L, 3)

    # ---- Step 3: GPU hashmap 查找 source_coords → voxel index ----
    grid_size = torch.tensor([voxel_resolution] * 3, device=device)
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        *grid_size.tolist(),
    )
    source_key = torch.cat([
        torch.zeros(L, 1, dtype=torch.int, device=device),
        source_coords,
    ], dim=-1)                                                       # (L, 4)
    quad_voxel = _C.hashmap_lookup_3d_cuda(
        *hashmap, source_key, *grid_size.tolist()
    ).long()                                                         # (L,)

    # ---- Step 4: broadcast quad → face ----
    face_axis_ids = quad_axis.repeat_interleave(fpq)                # (F,)
    face_voxel_ids = quad_voxel.repeat_interleave(fpq)              # (F,)

    return face_axis_ids, face_voxel_ids


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -x, -y
    elif s == 1: rx, ry, rz = -torch.ones_like(x), x, -y
    elif s == 2: rx, ry, rz = x, y, torch.ones_like(x)
    elif s == 3: rx, ry, rz = x, -y, -torch.ones_like(x)
    elif s == 4: rx, ry, rz = x, torch.ones_like(x), -y
    elif s == 5: rx, ry, rz = -x, -torch.ones_like(x), -y
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = F.normalize(cube_to_dir(s, gx, gy), dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


class EnvMap:
    def __init__(self, image: torch.Tensor):
        self.image = image
        
    @property
    def _backend(self):
        if not hasattr(self, '_nvdiffrec_envlight'):
            cubemap = latlong_to_cubemap(self.image, [512, 512])
            self._nvdiffrec_envlight = EnvironmentLight(cubemap)
            self._nvdiffrec_envlight.build_mips()
        return self._nvdiffrec_envlight

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        return self._backend.shade(gb_pos, gb_normal, kd, ks, view_pos, specular)
    
    def sample(self, directions: torch.Tensor):
        return dr.texture(
            self._backend.base.unsqueeze(0),
            directions.unsqueeze(0),
            boundary_mode='cube',
        )[0]
            

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = (far + near) / (far - near)
    ret[2, 3] = 2 * near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def screen_space_ambient_occlusion(
    depth: torch.Tensor,
    normal: torch.Tensor,
    perspective: torch.Tensor,
    radius: float = 0.1,
    bias: float = 1e-6,
    samples: int = 64,
    intensity: float = 1.0,
) -> torch.Tensor:
    """
    Screen space ambient occlusion (SSAO)

    Args:
        depth (torch.Tensor): [H, W, 1] depth image
        normal (torch.Tensor): [H, W, 3] normal image
        perspective (torch.Tensor): [4, 4] camera projection matrix
        radius (float): radius of the SSAO kernel
        bias (float): bias to avoid self-occlusion
        samples (int): number of samples to use for the SSAO kernel
        intensity (float): intensity of the SSAO effect
    Returns:
        (torch.Tensor): [H, W, 1] SSAO image
    """
    device = depth.device
    H, W, _ = depth.shape
    
    fx = perspective[0, 0]
    fy = perspective[1, 1]
    cx = perspective[0, 2]
    cy = perspective[1, 2]
    
    y_grid, x_grid = torch.meshgrid(
        (torch.arange(H, device=device) + 0.5) / H * 2 - 1,
        (torch.arange(W, device=device) + 0.5) / W * 2 - 1,
        indexing='ij'
    )
    x_view = (x_grid.float() - cx) * depth[..., 0] / fx
    y_view = (y_grid.float() - cy) * depth[..., 0] / fy
    view_pos = torch.stack([x_view, y_view, depth[..., 0]], dim=-1) # [H, W, 3]
    
    depth_feat = depth.permute(2, 0, 1).unsqueeze(0)
    occlusion = torch.zeros((H, W), device=device)
    
    # start sampling
    for _ in range(samples):
        # sample normal distribution, if inside, flip the sign
        rnd_vec = torch.randn(H, W, 3, device=device)
        rnd_vec = F.normalize(rnd_vec, p=2, dim=-1)
        dot_val = torch.sum(rnd_vec * normal, dim=-1, keepdim=True)
        sample_dir = torch.sign(dot_val) * rnd_vec
        scale = torch.rand(H, W, 1, device=device)
        scale = scale * scale
        sample_pos = view_pos + sample_dir * radius * scale
        sample_z = sample_pos[..., 2]
        
        # project to screen space
        z_safe = torch.clamp(sample_pos[..., 2], min=1e-5)
        proj_u = (sample_pos[..., 0] * fx / z_safe) + cx
        proj_v = (sample_pos[..., 1] * fy / z_safe) + cy
        grid = torch.stack([proj_u, proj_v], dim=-1).unsqueeze(0)
        geo_z = F.grid_sample(depth_feat, grid, mode='nearest', padding_mode='border').squeeze()
        range_check = torch.abs(geo_z - sample_z) < radius
        is_occluded = (geo_z <= sample_z - bias) & range_check
        occlusion += is_occluded.float()
        
    f_occ = occlusion / samples * intensity
    f_occ = torch.clamp(f_occ, 0.0, 1.0)
    
    return f_occ.unsqueeze(-1)


def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
    """
    Applies ACES tone mapping curve to an HDR image tensor.
    Input:  x - HDR tensor, shape (..., 3), range [0, +inf)
    Output: LDR tensor, same shape, range [0, 1]
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # ★ 确保输入非负，防止分母为 0 或负值导致 NaN
    x = torch.clamp(x, min=0.0)
    
    # Apply the ACES fitted curve
    # 添加 eps 防止极端情况下分母过小
    eps = 1e-6
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e + eps)
    
    # Clamp to [0, 1] for display or saving
    return torch.clamp(mapped, 0.0, 1.0)


def srgb_transfer(x: torch.Tensor) -> torch.Tensor:
    """
    sRGB OETF（光电转换函数）：近零线性 + 幂律。
    梯度处处有界（最大 12.92），可安全用于训练。
    Input:  x - LDR tensor, range [0, 1]
    Output: sRGB tensor, range [0, 1]
    """
    x = torch.clamp(x, 0.0, 1.0)
    lo = 12.92 * x
    hi = 1.055 * x.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, lo, hi)
    

class MeshPeeledRenderer:
    """统一 Mesh 渲染器（DepthPeeler + 分 chunk），支持 Normal-only 和 PBR 双模式。

    Normal mode (envmap=None):
        - 单层 rasterize + 分 chunk z-buffer 合并
        - 支持 per-chunk gradient checkpoint（通过模块级 _compute_normal_layer）
        - 接口兼容原 MeshRenderer: render(mesh, ext, intr, return_types=[...])

    PBR mode (envmap provided):
        - DepthPeeler 多层渲染 + 分 chunk
        - K=1 时增量合成（低峰值显存）
        - K>1 时累积 + 排序 + 合成
        - SSAO + 环境光背景 + ACES Tonemapping
        - 接口兼容原 PbrMeshRenderer: render(mesh, ext, intr, envmap=...)

    rendering_options:
        resolution:       渲染分辨率
        near / far:       近远裁剪面
        ssaa:             超采样倍率（默认 1）
        peel_layers:      DepthPeeler 层数（默认 8，仅 PBR 模式使用）
        grad_checkpoint:  是否启用 per-layer gradient checkpoint（默认 False）
    """

    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "peel_layers": 8,
            "grad_checkpoint": False,
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _empty_normal_result(self, return_types: List[str]) -> edict:
        """Normal mode: mesh 为空时的默认返回（兼容 MeshRenderer）"""
        resolution = self.rendering_options["resolution"]
        ret = edict()
        for rtype in return_types:
            if rtype == "normal":
                ret[rtype] = torch.full(
                    (3, resolution, resolution), 1.0,
                    dtype=torch.float32, device=self.device)
            elif rtype == "mask":
                ret[rtype] = torch.zeros(
                    (resolution, resolution),
                    dtype=torch.float32, device=self.device)
            elif rtype == "depth":
                ret[rtype] = torch.zeros(
                    (resolution, resolution),
                    dtype=torch.float32, device=self.device)
        return ret

    def _empty_pbr_result(self, envmap: dict) -> edict:
        """PBR mode: mesh 为空时的默认返回"""
        resolution = self.rendering_options["resolution"]
        out_dict = edict(
            normal=torch.ones((3, resolution, resolution), dtype=torch.float32, device=self.device),
            mask=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            base_color=torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device),
            metallic=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            roughness=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            alpha=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
            clay=torch.zeros((resolution, resolution), dtype=torch.float32, device=self.device),
        )
        for k in envmap.keys():
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = torch.zeros((3, resolution, resolution), dtype=torch.float32, device=self.device)
        return out_dict

    # ------------------------------------------------------------------
    # Phase 1: 坐标变换
    # ------------------------------------------------------------------

    def _transform_vertices(self, mesh, extrinsics, intrinsics, transformation):
        """世界坐标 → clip/camera space，同时计算 rays。

        Returns:
            vertices_clip:  (1, V, 4)
            vertices_cam:   (1, V, 4)
            vertices_batch: (1, V, 3) 变换后世界坐标
            vertices_orig:  (1, V, 3) 原始世界坐标（PBR 采样用）
            rays_o:         (H, W, 3)
            rays_d:         (H, W, 3)
            perspective:    (4, 4)
            extrinsics_b:   (1, 4, 4)
        """
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        rast_res = resolution * ssaa

        rays_o, rays_d = utils3d.torch.get_image_rays(
            extrinsics, intrinsics, rast_res, rast_res)

        perspective = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        full_proj = (perspective @ extrinsics).unsqueeze(0)            # (1, 4, 4)
        extrinsics_b = extrinsics.unsqueeze(0)                        # (1, 4, 4)

        vertices = mesh.vertices.unsqueeze(0)                         # (1, V, 3)
        vertices_orig = vertices.clone()                              # (1, V, 3)
        vertices_homo = torch.cat([
            vertices, torch.ones_like(vertices[..., :1])
        ], dim=-1)                                                    # (1, V, 4)

        if transformation is not None:
            vertices_homo = torch.bmm(
                vertices_homo,
                transformation.unsqueeze(0).transpose(-1, -2))        # (1, V, 4)
            vertices = vertices_homo[..., :3].contiguous()            # (1, V, 3)

        vertices_cam = torch.bmm(
            vertices_homo, extrinsics_b.transpose(-1, -2))            # (1, V, 4)
        vertices_clip = torch.bmm(
            vertices_homo, full_proj.transpose(-1, -2))               # (1, V, 4)

        return (vertices_clip, vertices_cam, vertices,
                vertices_orig, rays_o, rays_d, perspective, extrinsics_b)

    # ------------------------------------------------------------------
    # Phase 2: Face normals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_face_normals(vertices_batch, faces):
        """计算 per-face 法向量。

        Args:
            vertices_batch: (1, V, 3)
            faces: (F, 3)
        Returns:
            face_normal: (F, 3)
        """
        v0 = vertices_batch[0, faces[:, 0], :3]  # (F, 3)
        v1 = vertices_batch[0, faces[:, 1], :3]  # (F, 3)
        v2 = vertices_batch[0, faces[:, 2], :3]  # (F, 3)
        face_normal = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        return F.normalize(face_normal, dim=1)                # (F, 3)

    # ------------------------------------------------------------------
    # Phase 3: 统一 DepthPeeler chunk 循环
    # ------------------------------------------------------------------

    def _peel_chunks(self, vertices_clip, faces, rast_res, peel_layers,
                     compute_layer_fn, use_ckpt=False):
        """统一 DepthPeeler chunk 循环（Normal / PBR 共用）。

        将 faces 按 _MAX_FACES_PER_CHUNK 分 chunk，每个 chunk 用
        DepthPeeler 逐层剥离，调用 compute_layer_fn 计算每层属性，
        以 dict-of-lists 形式泛化累积后返回。

        Args:
            vertices_clip:    (1, V, 4) clip 空间顶点
            faces:            (F, 3) 面片索引
            rast_res:         光栅化分辨率
            peel_layers:      DepthPeeler 剥离层数
            compute_layer_fn: callable(rast, rast_db, faces_chunk, off, size) -> dict
                必须包含 'alpha', 'cam_normal', 'depth'。
                可选包含任意额外 key（如 'shaded'）。
                特殊 key 'first_layer': dict，仅首层时被记录。
            use_ckpt:         是否对 compute_layer_fn 使用 gradient checkpoint

        Returns:
            accum:         dict[str, List[Tensor]]  — 包含 '_sort_depth' 及
                           compute_layer_fn 返回的所有常规字段
            fl_data_list:  List[dict] — 每个 chunk 首层属性
        """
        num_faces = faces.shape[0]
        K = (num_faces + _MAX_FACES_PER_CHUNK - 1) // _MAX_FACES_PER_CHUNK

        # if K > 1:
        #     logging.info(
        #         f"[MeshPeeledRenderer] faces={num_faces} > "
        #         f"{_MAX_FACES_PER_CHUNK}, splitting into {K} chunks")

        accum: dict = {}            # key → List[Tensor]
        fl_data_list: list = []     # List[dict]

        for chunk_idx in range(K):
            off = chunk_idx * _MAX_FACES_PER_CHUNK
            size = min(_MAX_FACES_PER_CHUNK, num_faces - off)
            faces_chunk = faces[off:off + size]                          # (F_chunk, 3)

            with dr.DepthPeeler(self.glctx, vertices_clip, faces_chunk,
                                (rast_res, rast_res)) as peeler:
                for layer_idx in range(peel_layers):
                    rast_out, rast_db = peeler.rasterize_next_layer()    # (1, H, W, 4)

                    # 提前终止：该层完全空 → 后续层也空
                    if (rast_out[0, ..., -1] == 0).all():
                        break

                    # 排序用 depth（detach + clone 避免 in-place 修改 rast_out）
                    sort_depth = rast_out[0, ..., 2].detach().clone()    # (H, W)
                    sort_depth[rast_out[0, ..., -1] == 0] = float('inf')

                    # 逐层计算（可 checkpoint）
                    if use_ckpt:
                        layer = checkpoint(
                            compute_layer_fn, rast_out, rast_db,
                            faces_chunk, off, size,
                            use_reentrant=False)
                    else:
                        layer = compute_layer_fn(
                            rast_out, rast_db, faces_chunk, off, size)

                    # 提取首层元数据
                    fl = layer.pop('first_layer', None)
                    if layer_idx == 0 and fl is not None:
                        fl['sort_depth'] = sort_depth.clone()
                        fl_data_list.append(fl)

                    # 泛化累积：_sort_depth + compute_layer_fn 返回的所有 key
                    accum.setdefault('_sort_depth', []).append(sort_depth)
                    for k, v in layer.items():
                        accum.setdefault(k, []).append(v)

        return accum, fl_data_list

    # ------------------------------------------------------------------
    # Normal 单层计算
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_one_normal_layer(
            rast, _rast_db, faces_chunk, face_offset, chunk_size,
            face_normal, vertices_batch, vertices_cam,
            rays_o, extrinsics_b, alpha_fn):
        """Normal-only 单层计算。

        签名与 _peel_chunks 的 compute_layer_fn 协议一致：
            (rast, rast_db, faces_chunk, off, size, *extra_via_closure) -> dict

        此处为 staticmethod + 所有依赖通过参数传入，可安全用于
        torch.utils.checkpoint（use_reentrant=False）。

        alpha 由外部传入的 alpha_fn(rast, face_offset) 回调计算，
        调用方在 render_normal 中根据是否有 intersect_logits 构造不同闭包。

        Returns:
            dict with keys: alpha (H,W,1), cam_normal (H,W,3), depth (H,W,1)
        """
        device = rast.device

        # ---- 世界坐标位置（用于 normal flip）----
        pos = dr.interpolate(
            vertices_batch, rast, faces_chunk
        )[0][0]                                                      # (H, W, 3)

        # ---- 相机空间深度 ----
        depth = dr.interpolate(
            vertices_cam[..., 2:3].contiguous(), rast, faces_chunk
        )[0][0]                                                      # (H, W, 1)

        # ---- Face normal 插值 ----
        face_normal_chunk = face_normal[face_offset:face_offset + chunk_size]  # (F_chunk, 3)
        chunk_fn_idx = torch.arange(
            chunk_size, dtype=torch.int, device=device
        ).unsqueeze(1).repeat(1, 3).contiguous()                     # (F_chunk, 3)
        gb_normal = dr.interpolate(
            face_normal_chunk.unsqueeze(0), rast, chunk_fn_idx
        )[0][0]                                                      # (H, W, 3)

        # ---- 翻转背面法线 ----
        gb_normal = torch.where(
            torch.sum(gb_normal * (pos - rays_o), dim=-1, keepdim=True) > 0,
            -gb_normal, gb_normal)                                   # (H, W, 3)

        # ---- 世界 → 相机空间 ----
        cam_normal = (
            extrinsics_b[..., :3, :3].reshape(1, 1, 3, 3)
            @ gb_normal.unsqueeze(-1)
        ).squeeze(-1)                                                # (H, W, 3)

        # ---- alpha（由回调决定 opaque / 半透明）----
        alpha = alpha_fn(rast, face_offset)                          # (H, W, 1)

        return {
            'alpha': alpha,            # (H, W, 1)
            'cam_normal': cam_normal,  # (H, W, 3)
            'depth': depth,            # (H, W, 1)
        }

    # ------------------------------------------------------------------
    # PBR 属性计算
    # ------------------------------------------------------------------

    def _compute_pbr_attrs(self, rast, rast_db, faces_chunk, face_offset,
                           chunk_size, mesh, vertices_orig, rast_res):
        """从 chunk 光栅化结果提取 PBR 属性，处理 face 索引重映射。

        Args:
            rast:          (1, H, W, 4) 光栅化结果（face_id 相对 chunk 内 1-indexed）
            rast_db:       (1, H, W, 4) 光栅化微分
            faces_chunk:   (F_chunk, 3) chunk 内的局部 faces
            face_offset:   chunk 在全局 faces 中的起始偏移
            chunk_size:    chunk 内面片数
            mesh:          Mesh 对象
            vertices_orig: (1, V, 3) 原始世界坐标
            rast_res:      光栅化分辨率

        Returns:
            gb_basecolor: (H, W, 3)
            gb_metallic:  (H, W, 1)
            gb_roughness: (H, W, 1)
            gb_alpha:     (H, W, 1)
        """
        if isinstance(mesh, MeshWithVoxel):
            mask = rast[..., -1:] > 0  # (1, H, W, 1)
            # ★ 用 faces_chunk，vertex 索引仍是全局的
            xyz = dr.interpolate(vertices_orig, rast, faces_chunk)[0]        # (1, H, W, 3)
            xyz = ((xyz - mesh.origin) / mesh.voxel_size).reshape(1, -1, 3) # (1, H*W, 3)
            img = grid_sample_3d(
                mesh.attrs,
                torch.cat([torch.zeros_like(mesh.coords[..., :1]), mesh.coords], dim=-1),
                mesh.voxel_shape,
                xyz,
                mode='trilinear'
            )  # (1, H*W, C)
            img = img.reshape(1, rast_res, rast_res, mesh.attrs.shape[-1]) * mask  # (1, H, W, C)
            gb_basecolor = img[0, ..., mesh.layout['base_color']]  # (H, W, 3)
            gb_metallic = img[0, ..., mesh.layout['metallic']]     # (H, W, 1)
            gb_roughness = img[0, ..., mesh.layout['roughness']]   # (H, W, 1)
            gb_alpha = img[0, ..., mesh.layout['alpha']]           # (H, W, 1)
            return gb_basecolor, gb_metallic, gb_roughness, gb_alpha

        elif isinstance(mesh, MeshWithPbrMaterial):
            tri_id = rast[0, :, :, -1:]  # (H, W, 1) chunk-local 1-indexed
            mask = tri_id > 0
            # ★ 全局 face 索引重映射
            global_tri_id = (tri_id.long() - 1 + face_offset).clamp(min=0)  # (H, W, 1)
            mid = mesh.material_ids[global_tri_id]

            # ★ UV coords 切片（chunk-local）
            uv_chunk = mesh.uv_coords[face_offset:face_offset + chunk_size]  # (F_chunk, 3, 2)
            uv_flat = uv_chunk.reshape(1, -1, 2)                            # (1, F_chunk*3, 2)
            uv_faces = torch.arange(
                chunk_size * 3, dtype=torch.int, device=self.device
            ).reshape(-1, 3)                                                 # (F_chunk, 3)
            texc, texd = dr.interpolate(
                uv_flat, rast, uv_faces,
                rast_db=rast_db, diff_attrs='all')
            # Fix problematic texture coordinates
            texc = torch.nan_to_num(texc, nan=0.0, posinf=1e3, neginf=-1e3)
            texc = torch.clamp(texc, min=-1e3, max=1e3)
            texd = torch.nan_to_num(texd, nan=0.0, posinf=1e3, neginf=-1e3)
            texd = torch.clamp(texd, min=-1e3, max=1e3)

            gb_basecolor = torch.zeros((rast_res, rast_res, 3), dtype=torch.float32, device=self.device)
            gb_metallic = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            gb_roughness = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            gb_alpha = torch.zeros((rast_res, rast_res, 1), dtype=torch.float32, device=self.device)
            for id, mat in enumerate(mesh.materials):
                mat_mask = (mid == id).float() * mask.float()
                mat_texc = texc * mat_mask
                mat_texd = texd * mat_mask

                if mat.base_color_texture is not None:
                    bc = dr.texture(
                        mat.base_color_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.base_color_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.base_color_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_basecolor += bc * mat.base_color_factor * mat_mask
                else:
                    gb_basecolor += mat.base_color_factor * mat_mask

                if mat.metallic_texture is not None:
                    m = dr.texture(
                        mat.metallic_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.metallic_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.metallic_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_metallic += m * mat.metallic_factor * mat_mask
                else:
                    gb_metallic += mat.metallic_factor * mat_mask

                if mat.roughness_texture is not None:
                    r = dr.texture(
                        mat.roughness_texture.image.unsqueeze(0),
                        mat_texc, mat_texd,
                        filter_mode='linear-mipmap-linear' if mat.roughness_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                        boundary_mode='clamp' if mat.roughness_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                    )[0]
                    gb_roughness += r * mat.roughness_factor * mat_mask
                else:
                    gb_roughness += mat.roughness_factor * mat_mask

                if mat.alpha_mode == AlphaMode.OPAQUE:
                    gb_alpha += 1.0 * mat_mask
                else:
                    if mat.alpha_texture is not None:
                        a = dr.texture(
                            mat.alpha_texture.image.unsqueeze(0),
                            mat_texc, mat_texd,
                            filter_mode='linear-mipmap-linear' if mat.alpha_texture.filter_mode == TextureFilterMode.LINEAR else 'nearest',
                            boundary_mode='clamp' if mat.alpha_texture.wrap_mode == TextureWrapMode.CLAMP_TO_EDGE else 'wrap'
                        )[0]
                        if mat.alpha_mode == AlphaMode.MASK:
                            gb_alpha += (a * mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                        elif mat.alpha_mode == AlphaMode.BLEND:
                            gb_alpha += a * mat.alpha_factor * mat_mask
                    else:
                        if mat.alpha_mode == AlphaMode.MASK:
                            gb_alpha += (mat.alpha_factor > mat.alpha_cutoff).float() * mat_mask
                        elif mat.alpha_mode == AlphaMode.BLEND:
                            gb_alpha += mat.alpha_factor * mat_mask

            return gb_basecolor, gb_metallic, gb_roughness, gb_alpha

    def _compute_one_pbr_layer(self, rast, rast_db, faces_chunk, face_offset,
                           chunk_size, face_normal, mesh,
                           vertices_batch, vertices_orig, vertices_cam,
                           rays_o, envmap, num_envmaps,
                           extrinsics_b, rast_res):
        """PBR 单层计算：法向量 + PBR 属性 + IBL shading。

        注意：含 envmap.shade() 调用，非纯 tensor op，不能 checkpoint 包裹。

        Args:
            rast:            (1, H, W, 4) 光栅化结果
            rast_db:         (1, H, W, 4) 光栅化微分
            faces_chunk:     (F_chunk, 3)
            face_offset:     chunk 偏移
            chunk_size:      chunk 面片数
            face_normal:     (F_total, 3) 全局 face normals
            mesh:            Mesh 对象
            vertices_batch:  (1, V, 3) 变换后世界坐标
            vertices_orig:   (1, V, 3) 原始世界坐标
            vertices_cam:    (1, V, 4) 相机空间坐标
            rays_o:          (H, W, 3)
            envmap:          dict of EnvMap
            num_envmaps:     int
            extrinsics_b:    (1, 4, 4)
            rast_res:        int

        Returns:
            dict with keys: gb_alpha, gb_shaded, gb_cam_normal, gb_depth,
                            out_normal, mask, base_color, metallic, roughness, alpha_attr
        """
        # ---- Pos ----
        pos = dr.interpolate(vertices_batch, rast, faces_chunk)[0][0]  # (H, W, 3)

        # ---- Camera-space depth ----
        gb_depth = dr.interpolate(
            vertices_cam[..., 2:3].contiguous(), rast, faces_chunk
        )[0][0]                                                        # (H, W, 1)

        # ---- Face normal（chunk 切片）----
        face_normal_chunk = face_normal[face_offset:face_offset + chunk_size]  # (F_chunk, 3)
        chunk_fn_idx = torch.arange(
            chunk_size, dtype=torch.int, device=self.device
        ).unsqueeze(1).repeat(1, 3).contiguous()                       # (F_chunk, 3)
        gb_normal = dr.interpolate(
            face_normal_chunk.unsqueeze(0), rast, chunk_fn_idx
        )[0][0]                                                        # (H, W, 3)
        gb_normal = torch.where(
            torch.sum(gb_normal * (pos - rays_o), dim=-1, keepdim=True) > 0,
            -gb_normal, gb_normal)                                     # (H, W, 3)
        gb_cam_normal = (extrinsics_b[..., :3, :3].reshape(1, 1, 3, 3)
                         @ gb_normal.unsqueeze(-1)).squeeze(-1)        # (H, W, 3)

        # 首层输出用
        out_normal = -gb_cam_normal * 0.5 + 0.5                        # (H, W, 3)
        mask = (rast[0, ..., -1:] > 0).float()                         # (H, W, 1)

        # ---- PBR attributes ----
        gb_basecolor, gb_metallic, gb_roughness, gb_alpha = \
            self._compute_pbr_attrs(
                rast, rast_db, faces_chunk, face_offset, chunk_size,
                mesh, vertices_orig, rast_res)

        # ---- Shading ----
        gb_basecolor_s = torch.clamp(gb_basecolor, 0.0, 1.0) ** 2.2   # (H, W, 3)
        gb_metallic_s = torch.clamp(gb_metallic, 0.0, 1.0)            # (H, W, 1)
        gb_roughness_s = torch.clamp(gb_roughness, 0.0, 1.0)          # (H, W, 1)
        gb_alpha_s = torch.clamp(gb_alpha, 0.0, 1.0)                  # (H, W, 1)
        gb_orm = torch.cat([
            torch.zeros_like(gb_metallic_s), gb_roughness_s, gb_metallic_s
        ], dim=-1)                                                     # (H, W, 3)
        gb_shaded = torch.stack([
            e.shade(
                pos.unsqueeze(0), gb_normal.unsqueeze(0),
                gb_basecolor_s.unsqueeze(0), gb_orm.unsqueeze(0),
                rays_o, specular=True,
            )[0]
            for e in envmap.values()
        ], dim=0)                                                      # (E, H, W, 3)

        return {
            # ---- 统一字段（_peel_chunks 要求）----
            'alpha': gb_alpha_s,              # (H, W, 1)
            'cam_normal': gb_cam_normal,      # (H, W, 3)
            'depth': gb_depth,                # (H, W, 1)
            # ---- PBR 专有 ----
            'shaded': gb_shaded,              # (E, H, W, 3)
            'first_layer': {                  # 首层属性（_peel_chunks layer_idx==0 时使用）
                'normal': out_normal,         # (H, W, 3)
                'mask': mask,                 # (H, W, 1)
                'base_color': gb_basecolor,   # (H, W, 3)
                'metallic': gb_metallic,      # (H, W, 1)
                'roughness': gb_roughness,    # (H, W, 1)
                'alpha_attr': gb_alpha,       # (H, W, 1)
            },
        }

    @staticmethod
    def _sort_and_composite(accum, rast_res, device):
        """per-pixel 深度排序 + front-to-back alpha composite（Normal/PBR 共用）。

        接收 _peel_chunks 返回的 accum dict，自动处理其中所有字段。
        字段维度约定：
          - 4D (T, H, W, C): 标准字段，如 alpha / cam_normal / depth
          - 5D (T, E, H, W, C): 多 envmap 字段，如 shaded

        compositing 规则：
          - alpha:                累加 w
          - depth / cam_normal:   取 max_w 对应值
          - 其余 4D/5D 字段:     加权求和 (w * value)

        Returns:
            dict  — 包含 'alpha', 'depth', 'cam_normal'，以及 accum 中
                    除 '_sort_depth' 外的其余字段（加权求和后）。
        """
        H = W = rast_res
        sort_depths = accum.get('_sort_depth', [])

        # 初始值
        result = {
            'depth': torch.full((H, W, 1), 1e10, device=device),     # (H, W, 1)
            'cam_normal': torch.zeros(H, W, 3, device=device),       # (H, W, 3)
            'alpha': torch.zeros(H, W, 1, device=device),            # (H, W, 1)
        }
        if not sort_depths:
            return result

        T = len(sort_depths)
        sort_idx = torch.stack(sort_depths).argsort(dim=0)           # (T, H, W)
        idx_1 = sort_idx.unsqueeze(-1)                               # (T, H, W, 1)

        # ---- gather 重排所有字段 ----
        sorted_fields = {}
        for key, tensors in accum.items():
            if key == '_sort_depth':
                continue
            stacked = torch.stack(tensors)
            if stacked.dim() == 4:                                   # (T, H, W, C)
                idx = idx_1.expand_as(stacked)
            elif stacked.dim() == 5:                                 # (T, E, H, W, C)
                idx = sort_idx.unsqueeze(1).unsqueeze(-1).expand_as(stacked)
            else:
                continue
            sorted_fields[key] = torch.gather(stacked, 0, idx)

        # ---- front-to-back compositing ----
        alpha = torch.zeros(H, W, 1, device=device)                 # (H, W, 1)
        max_w = torch.zeros(H, W, 1, device=device)                 # (H, W, 1)

        # 加权求和字段（排除 alpha / depth / cam_normal）
        SUM_SKIP = {'alpha', 'depth', 'cam_normal'}
        sum_fields = {k: torch.zeros_like(v[0])
                      for k, v in sorted_fields.items()
                      if k not in SUM_SKIP}                          # e.g. shaded → (E,H,W,3)

        for rank in range(T):
            w = (1 - alpha) * sorted_fields['alpha'][rank]           # (H, W, 1)
            result['depth'] = torch.where(
                w > max_w,
                sorted_fields['depth'][rank], result['depth'])       # (H, W, 1)
            result['cam_normal'] = torch.where(
                (w > max_w).expand_as(result['cam_normal']),
                sorted_fields['cam_normal'][rank],
                result['cam_normal'])                                # (H, W, 3)
            max_w = torch.maximum(max_w, w)                          # (H, W, 1)
            for k, buf in sum_fields.items():
                sum_fields[k] = buf + w * sorted_fields[k][rank]
            alpha = alpha + w                                        # (H, W, 1)

        result['alpha'] = alpha
        result.update(sum_fields)
        return result

    @staticmethod
    def _merge_first_layer(fl_data_list, rast_res, device):
        """Phase C: 跨 chunk 首层属性归并（per-pixel 选最近 chunk）

        与 PBR peeled 路径保持一致的 _merge_first_layer_depth 语义。

        Returns:
            dict: normal, mask, base_color, metallic, roughness, alpha
        """
        result = {}
        if not fl_data_list:
            return result

        if len(fl_data_list) == 1:
            fl = fl_data_list[0]
            result['normal'] = fl['normal']
            result['mask'] = fl['mask']
            result['base_color'] = fl['base_color']
            result['metallic'] = fl['metallic']
            result['roughness'] = fl['roughness']
            result['alpha'] = fl['alpha_attr']
            return result

        # 多 chunk: per-pixel 选最近 chunk
        stacked_depth = torch.stack(
            [d['sort_depth'] for d in fl_data_list])              # (K, H, W)
        closest = stacked_depth.argmin(dim=0)                     # (H, W)

        for key in ['normal', 'mask', 'base_color', 'metallic', 'roughness', 'alpha_attr']:
            stacked = torch.stack(
                [d[key] for d in fl_data_list])                   # (K, H, W, C)
            C = stacked.shape[-1]
            idx = closest.unsqueeze(-1).expand(
                -1, -1, C).unsqueeze(0)                           # (1, H, W, C)
            merged = torch.gather(stacked, 0, idx).squeeze(0)    # (H, W, C)
            out_key = 'alpha' if key == 'alpha_attr' else key
            result[out_key] = merged

        return result

    @staticmethod
    def _assemble_pbr_output(shaded, depth, normal, alpha,
                         fl_attrs, envmap, rast_res):
        """Phase D: 组装 PBR out_dict"""
        out_dict = edict()

        # 首层属性
        for k, v in fl_attrs.items():
            out_dict[k] = v

        # shaded（per envmap）
        for i, k in enumerate(envmap.keys()):
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = shaded[i]                      # (H, W, 3)

        # 内部字段（SSAO / background 用，后续清理）
        out_dict._depth = depth      # (H, W, 1)
        out_dict._normal = normal    # (H, W, 3)
        out_dict._alpha = alpha      # (H, W, 1)

        return out_dict

    # ------------------------------------------------------------------
    # Phase 4: 后处理（SSAO + Background）
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_post_effects(out_dict, perspective, rays_d, envmap, use_envmap_bg):
        """SSAO + 环境光背景"""
        # SSAO
        f_occ = screen_space_ambient_occlusion(
            out_dict._depth, out_dict._normal, perspective, intensity=1.5)
        for k in envmap.keys():
            key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[key] = out_dict[key] * (1 - f_occ)
        out_dict.clay = (1 - f_occ)

        # Background
        if use_envmap_bg:
            bg = torch.stack([e.sample(rays_d) for e in envmap.values()], dim=0)
            for i, k in enumerate(envmap.keys()):
                key = f"shaded_{k}" if k != '' else "shaded"
                out_dict[key] = out_dict[key] + (1 - out_dict._alpha) * bg[i]

        # 清理内部临时字段
        del out_dict._depth, out_dict._normal, out_dict._alpha

    # ------------------------------------------------------------------
    # Phase 5: SSAA 下采样
    # ------------------------------------------------------------------

    def _downsample(self, out_dict: edict, return_types: List[str]) -> edict:
        """SSAA 下采样 + 格式转换（Normal mode）。

        与 MeshRenderer 输出格式对齐:
            normal: (3, H, W)
            mask:   (H, W)
            depth:  (H, W)
        """
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]

        for rtype in return_types:
            if rtype not in out_dict:
                continue
            img = out_dict[rtype]                                  # (H_s, W_s, C)
            if ssaa > 1:
                img = F.interpolate(
                    img.unsqueeze(0).permute(0, 3, 1, 2),          # (1, C, H_s, W_s)
                    (resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True,
                ).squeeze(0)                                       # (C, H, W)
            else:
                img = img.permute(2, 0, 1)                         # (C, H, W)

            # squeeze: (1, H, W) → (H, W) ; (3, H, W) 保留
            if img.shape[0] == 1:
                img = img.squeeze(0)                               # (H, W)
            out_dict[rtype] = img

        return out_dict

    def _downsample_pbr(self, out_dict, envmap):
        """SSAA 下采样 + tonemapping（PBR 模式）"""
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]

        for k in list(out_dict.keys()):
            if ssaa > 1:
                out_dict[k] = F.interpolate(
                    out_dict[k].unsqueeze(0).permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode='bilinear', align_corners=False, antialias=True)
            else:
                out_dict[k] = out_dict[k].permute(2, 0, 1)
            out_dict[k] = out_dict[k].squeeze()

        # Tonemapping + sRGB
        for k in envmap.keys():
            shaded_key = f"shaded_{k}" if k != '' else "shaded"
            out_dict[shaded_key] = aces_tonemapping(out_dict[shaded_key])
            out_dict[shaded_key] = srgb_transfer(out_dict[shaded_key])

        return out_dict

    # ------------------------------------------------------------------
    # Normal mode
    # ------------------------------------------------------------------

    def render_normal(
            self,
            mesh: Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            intersect_logits: torch.Tensor,        # (N, 3) raw logits，可微
            coords: torch.Tensor,                  # (N, 3) int voxel 坐标
            voxel_resolution: int,
            return_types: List[str] = ["mask", "normal", "depth"],
            transformation: Optional[torch.Tensor] = None,
        ) -> edict:
        """Normal 渲染（DepthPeeler + 排序合成，与 PBR 走统一路径）。

        alpha = sigmoid(intersect_logits)（半透明），peel_layers 由 rendering_options 决定。

        梯度路径:
          路径 1: Loss → pixel_normal → face_normal → vertices → dual_vertices
          路径 2: Loss → alpha_compositing
                   → sigmoid(intersect_logits[voxel_id, axis_id]) → Decoder

        Args:
            mesh:              Mesh 对象
            extrinsics:        (4, 4) 相机外参
            intrinsics:        (3, 3) 相机内参
            intersect_logits:  (N, 3) raw logits，可微
            coords:            (N, 3) int voxel 坐标
            voxel_resolution:  voxel 分辨率
            return_types:      输出字段列表
            transformation:    可选模型变换
        """
        resolution = self.rendering_options["resolution"]
        ssaa = self.rendering_options["ssaa"]
        rast_res = resolution * ssaa
        use_ckpt = self.rendering_options["grad_checkpoint"]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return self._empty_normal_result(return_types)

        # Phase 1: 坐标变换
        (vertices_clip, vertices_cam, vertices_batch, _vertices_orig,
         rays_o, _rays_d, _perspective, extrinsics_b) = \
            self._transform_vertices(mesh, extrinsics, intrinsics, transformation)

        # Phase 2: Face normals
        face_normal = self._compute_face_normals(vertices_batch, mesh.faces)
        faces = mesh.faces

        # Phase 2.5: face → voxel 映射 + alpha_fn 回调
        face_axis_ids, face_voxel_ids = recover_face_axis_and_voxel(
            faces, coords, voxel_resolution)
        peel_layers = self.rendering_options["peel_layers"]

        def alpha_fn(rast, face_offset):
            fid = rast[0, ..., -1].long() - 1                    # (H, W)
            gfid = (fid + face_offset).clamp(min=0)              # (H, W)
            valid = (rast[0, ..., -1] > 0).float()               # (H, W)
            return (torch.sigmoid(
                intersect_logits[face_voxel_ids[gfid],
                                 face_axis_ids[gfid]]            # (H, W)
            ) * valid).unsqueeze(-1)                              # (H, W, 1)

        # Phase 3: 统一 DepthPeeler 渲染 + 排序合成
        compute_fn = partial(
            self._compute_one_normal_layer,
            face_normal=face_normal,
            vertices_batch=vertices_batch,
            vertices_cam=vertices_cam,
            rays_o=rays_o,
            extrinsics_b=extrinsics_b,
            alpha_fn=alpha_fn)

        accum, _fl_data_list = self._peel_chunks(
            vertices_clip, faces, rast_res,
            peel_layers=peel_layers,
            compute_layer_fn=compute_fn,
            use_ckpt=use_ckpt)

        composite = self._sort_and_composite(accum, rast_res, self.device)
        del accum
        depth = composite['depth']                                    # (H, W, 1)
        cam_normal = composite['cam_normal']                          # (H, W, 3)
        alpha = composite['alpha']                                    # (H, W, 1)

        # Phase 4: 组装输出 + SSAA 下采样
        out_dict = edict()
        if "normal" in return_types:
            out_dict["normal"] = -cam_normal * 0.5 + 0.5              # (H, W, 3) → [0,1]
        if "mask" in return_types:
            out_dict["mask"] = alpha                                   # (H, W, 1)
        if "depth" in return_types:
            out_dict["depth"] = depth                                  # (H, W, 1)

        return self._downsample(out_dict, return_types)

    # ------------------------------------------------------------------
    # PBR mode
    # ------------------------------------------------------------------

    def render_pbr(
            self,
            mesh: Mesh,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            envmap: Union['EnvMap', Dict[str, 'EnvMap']],
            use_envmap_bg: bool = False,
            transformation: Optional[torch.Tensor] = None,
        ) -> edict:
        """PBR 渲染（DepthPeeler 多层 + 分 chunk + 排序合成）。"""
        if not isinstance(envmap, dict):
            envmap = {'': envmap}

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return self._empty_pbr_result(envmap)

        rast_res = self.rendering_options["resolution"] * self.rendering_options["ssaa"]
        peel_layers = self.rendering_options["peel_layers"]
        num_envmaps = len(envmap)

        # ============ Phase 1: 坐标变换 ============
        (vertices_clip, vertices_cam, vertices_batch, vertices_orig,
         rays_o, rays_d, perspective, extrinsics_b) = \
            self._transform_vertices(mesh, extrinsics, intrinsics, transformation)

        # ============ Phase 2: Face normals ============
        face_normal = self._compute_face_normals(vertices_batch, mesh.faces)

        # ============ Phase 3: DepthPeeler 多层渲染 + 排序合成 ============
        compute_fn = partial(
            self._compute_one_pbr_layer,
            face_normal=face_normal, mesh=mesh,
            vertices_batch=vertices_batch, vertices_orig=vertices_orig,
            vertices_cam=vertices_cam, rays_o=rays_o,
            envmap=envmap, num_envmaps=num_envmaps,
            extrinsics_b=extrinsics_b, rast_res=rast_res)

        accum, fl_data_list = self._peel_chunks(
            vertices_clip, mesh.faces, rast_res,
            peel_layers, compute_fn, use_ckpt=False)

        composite = self._sort_and_composite(accum, rast_res, self.device)
        del accum
        depth = composite['depth']                                    # (H, W, 1)
        normal = composite['cam_normal']                              # (H, W, 3)
        alpha = composite['alpha']                                    # (H, W, 1)
        shaded = composite.get('shaded')                              # (E, H, W, 3) | None

        fl_attrs = self._merge_first_layer(
            fl_data_list, rast_res, self.device)
        del fl_data_list

        out_dict = self._assemble_pbr_output(
            shaded, depth, normal, alpha, fl_attrs, envmap, rast_res)

        # ============ Phase 4: 后处理（SSAO + Background） ============
        self._apply_post_effects(out_dict, perspective, rays_d, envmap, use_envmap_bg)

        # ============ Phase 5: SSAA 下采样 + Tonemapping ============
        return self._downsample_pbr(out_dict, envmap)
