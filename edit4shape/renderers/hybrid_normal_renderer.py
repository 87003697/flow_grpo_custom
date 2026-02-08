# hybrid_normal_renderer.py

"""
混合 Normal 渲染器：Voxel Normal (subs 可微) + Mesh Rendering (高质量)

梯度路径:
  - dual_vertices → vertices → v_normal → pixel_normals (几何)
  - subs → neighbor_occ → corrected_normal → pixel_normals (拓扑)

使用方法:
    renderer = Hybrid26NormalRenderer({"resolution": 512})
    outputs = renderer.render(mesh, subs, coords, extrinsics, intrinsics)
    normal = outputs.normal  # (H, W, 3)
"""

from typing import List, Tuple, Optional, Any
import torch
from torch import Tensor
import torch.nn.functional as F
import nvdiffrast.torch as dr
from easydict import EasyDict as edict

from .diff_voxel_normal_neighbor26 import (
    _neighbor_offsets_26,
    _compute_neighbor_occupancy_soft,
)


# =============================================================================
# 辅助函数
# =============================================================================

def comput_v_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """计算 vertex normals（从 MeshExtractResult 复制）
    
    Args:
        vertices: (N, 3) mesh 顶点
        faces: (F, 3) 面索引
    
    Returns:
        v_normals: (N, 3) 每个顶点的法向量
    """
    i0 = faces[..., 0].long()  # (F,)
    i1 = faces[..., 1].long()  # (F,)
    i2 = faces[..., 2].long()  # (F,)

    v0 = vertices[i0, :]  # (F, 3)
    v1 = vertices[i1, :]  # (F, 3)
    v2 = vertices[i2, :]  # (F, 3)
    
    # 计算 face normals（叉积）
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)
    
    # 累加到 vertex
    v_normals = torch.zeros_like(vertices)  # (N, 3)
    v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
    v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
    v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

    # 归一化
    v_normals = F.normalize(v_normals, dim=1, eps=1e-6)  # (N, 3)
    
    return v_normals


def compute_corrected_normal_from_subs(
    coords: Tensor,
    subs: List[Any],
    v_normal: Tensor,
    voxel_resolution: int,
    grad_shrink: float = 0.01,
) -> Tensor:
    """计算经过方向校正的可微法向量
    
    核心思想：
    1. 用 26-neighbor 计算 normal_sum（对 subs 可微）
    2. 用 v_normal（几何法向量）过滤同侧邻居，避免薄壁结构抵消
    3. 输出方向与 v_normal 一致，但对 subs 可微
    
    Args:
        coords: (K, 3) 可见顶点的 voxel 坐标
        subs: 多分辨率 subdivision logits
        v_normal: (K, 3) 几何法向量（参考方向）
        voxel_resolution: voxel 分辨率
        grad_shrink: 梯度缩放因子，稳定训练
    
    Returns:
        corrected_normal: (K, 3) 对 subs 可微，方向与 v_normal 一致
    """
    device = coords.device
    K = coords.shape[0]
    
    if K == 0:
        return torch.zeros(0, 3, device=device)
    
    # 1. 获取 26 邻居偏移和方向
    offsets, dist_weights = _neighbor_offsets_26(device)  # (26, 3), (26,)
    directions = F.normalize(offsets.float(), dim=-1)  # (26, 3)
    
    # 2. 计算邻居坐标
    neighbor_coords = coords[:, None, :] + offsets[None, :, :]  # (K, 26, 3)
    
    # 3. 查询邻居的 soft occupancy（对 subs 可微）
    neighbor_occ = _compute_neighbor_occupancy_soft(
        neighbor_coords, subs, voxel_resolution
    )  # (K, 26)
    
    # 4. 计算 missing_weight（梯度稳定技巧）
    neighbor_occ_stable = grad_shrink * neighbor_occ + (1 - grad_shrink) * neighbor_occ.detach()
    missing_weight = 1.0 - neighbor_occ_stable  # (K, 26)
    
    # 5. 用 v_normal 过滤同侧邻居（避免薄壁抵消）
    v_normal_ref = v_normal.detach()  # 不让梯度流回 vertices
    dot_with_vnormal = torch.einsum('kd,nd->kn', v_normal_ref, directions)  # (K, 26)
    same_side_weight = F.relu(dot_with_vnormal)  # (K, 26) 只保留同向的
    
    # 6. 加权累加
    weighted_dirs = directions * dist_weights[:, None]  # (26, 3)
    contribution = (
        missing_weight[:, :, None] *      # (K, 26, 1) 缺失程度（对 subs 可微）
        same_side_weight[:, :, None] *    # (K, 26, 1) 同侧权重
        weighted_dirs[None, :, :]         # (1, 26, 3) 方向
    )  # (K, 26, 3)
    
    normal_sum = contribution.sum(dim=1)  # (K, 3)
    
    # 7. 归一化
    corrected_normal = F.normalize(normal_sum, dim=-1, eps=1e-6)  # (K, 3)
    
    return corrected_normal


def intrinsics_to_projection(
    intrinsics: Tensor,
    near: float,
    far: float,
) -> Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix
    
    Args:
        intrinsics: (3, 3) OpenCV 相机内参
        near: 近裁剪面
        far: 远裁剪面
    
    Returns:
        projection: (4, 4) OpenGL 透视矩阵
    """
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
    
    结合 Mesh 几何法向量（方向正确）和 26-neighbor 法向量（对 subs 可微）
    
    Args:
        rendering_options: 渲染选项字典
        device: 设备
    """
    
    def __init__(self, rendering_options: dict = {}, device: str = "cuda"):
        self.rendering_options = edict({
            "resolution": 512,
            "near": 0.1,
            "far": 100.0,
            "ssaa": 1,
            "antialias": True,
            "vertex_chunk_size": 50000,  # 分块处理顶点
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device
    
    def render(
        self,
        mesh: Any,                      # Mesh 对象，包含 vertices, faces
        subs: List[Any],                # 多分辨率 subdivision logits
        coords: Tensor,                 # (N, 3) voxel 坐标（用于 26-neighbor 查询）
        extrinsics: Tensor,             # (4, 4) 相机外参
        intrinsics: Tensor,             # (3, 3) 相机内参
        voxel_resolution: int,          # voxel 分辨率，如 1024
        return_types: List[str] = ["normal", "mask", "depth"],
    ) -> edict:
        """渲染可微法向量
        
        Args:
            mesh: Mesh 对象，需要有 vertices (N, 3) 和 faces (F, 3)
            subs: 多分辨率 subdivision logits
            coords: (N, 3) voxel 坐标，与 mesh.vertices 一一对应
            extrinsics: (4, 4) 相机外参 (world to camera)
            intrinsics: (3, 3) 相机内参
            voxel_resolution: voxel 分辨率
            return_types: 返回类型列表，可选 "normal", "mask", "depth"
        
        Returns:
            edict:
                normal: (H, W, 3) 可微法向量图
                mask: (H, W) 前景掩码
                depth: (H, W) 深度图
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        antialias = self.rendering_options["antialias"]
        
        vertices = mesh.vertices  # (N, 3)
        faces = mesh.faces  # (F, 3)
        N = vertices.shape[0]
        
        # 空 mesh 处理
        if N == 0 or faces.shape[0] == 0:
            return self._empty_result(resolution, return_types)
        
        # 1. 计算几何 v_normal（对 vertices 可微）
        v_normal = comput_v_normals(vertices, faces)  # (N, 3)
        
        # 2. 变换到 clip space
        perspective = intrinsics_to_projection(intrinsics, near, far)
        full_proj = (perspective @ extrinsics).unsqueeze(0)  # (1, 4, 4)
        extrinsics_batch = extrinsics.unsqueeze(0)  # (1, 4, 4)
        
        vertices_batch = vertices.unsqueeze(0)  # (1, N, 3)
        vertices_homo = torch.cat([
            vertices_batch, 
            torch.ones_like(vertices_batch[..., :1])
        ], dim=-1)  # (1, N, 4)
        
        vertices_cam = torch.bmm(vertices_homo, extrinsics_batch.transpose(-1, -2))  # (1, N, 4)
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))  # (1, N, 4)
        
        # 3. 光栅化
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces, 
            (resolution * ssaa, resolution * ssaa)
        )  # (1, H, W, 4)
        
        # 4. 收集可见顶点
        visible_vertex_ids = self._collect_visible_vertices(rast, faces)  # (K,)
        K = visible_vertex_ids.shape[0]
        
        # 5. 计算可微法向量（核心）
        all_normals = torch.zeros(N, 3, device=self.device)  # (N, 3)
        
        if K > 0:
            visible_coords = coords[visible_vertex_ids]  # (K, 3)
            visible_v_normal = v_normal[visible_vertex_ids]  # (K, 3)
            
            # 变换 v_normal 到 camera space
            rot = extrinsics[:3, :3]  # (3, 3)
            visible_v_normal_cam = torch.matmul(visible_v_normal, rot.T)  # (K, 3)
            # 翻转到朝向相机
            visible_v_normal_cam = torch.where(
                visible_v_normal_cam[:, 2:3] > 0,
                -visible_v_normal_cam,
                visible_v_normal_cam
            )  # (K, 3)
            
            # 分块 + checkpoint 计算
            corrected_normal = self._compute_normals_chunked(
                visible_coords, subs, visible_v_normal_cam,
                voxel_resolution, self.rendering_options["vertex_chunk_size"]
            )  # (K, 3)
            
            all_normals[visible_vertex_ids] = corrected_normal
        
        # 6. 渲染各属性
        out_dict = edict()
        
        for rtype in return_types:
            if rtype == "normal":
                img = dr.interpolate(
                    all_normals.unsqueeze(0), rast, faces
                )[0]  # (1, H, W, 3)
                if antialias:
                    img = dr.antialias(img, rast, vertices_clip, faces)
                # 归一化（插值后可能不是单位向量）
                img = F.normalize(img, dim=-1, eps=1e-6)
                # 转换到可视化范围 [0, 1]
                img = (img + 1) / 2
                
            elif rtype == "mask":
                img = (rast[..., -1:] > 0).float()  # (1, H, W, 1)
                if antialias:
                    img = dr.antialias(img, rast, vertices_clip, faces)
                    
            elif rtype == "depth":
                img = dr.interpolate(
                    vertices_cam[..., 2:3].contiguous(), rast, faces
                )[0]  # (1, H, W, 1)
                if antialias:
                    img = dr.antialias(img, rast, vertices_clip, faces)
            
            else:
                continue
            
            # SSAA 下采样
            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2), 
                    (resolution, resolution), 
                    mode='bilinear', 
                    align_corners=False, 
                    antialias=True
                )
                img = img.squeeze(0).permute(1, 2, 0)  # (H, W, C)
            else:
                img = img.squeeze(0)  # (H, W, C)
            
            # 去掉最后一维（如果是 1）
            if img.shape[-1] == 1:
                img = img.squeeze(-1)  # (H, W)
            
            out_dict[rtype] = img
        
        return out_dict
    
    def _compute_normals_chunked(
        self,
        visible_coords: Tensor,        # (K, 3)
        subs: List[Any],
        visible_v_normal_cam: Tensor,  # (K, 3)
        voxel_resolution: int,
        chunk_size: int,
    ) -> Tensor:
        """分块 + checkpoint 计算法向量
        
        Args:
            visible_coords: (K, 3) 可见顶点坐标
            subs: 多分辨率 subdivision logits
            visible_v_normal_cam: (K, 3) camera space 几何法向量
            voxel_resolution: voxel 分辨率
            chunk_size: 每块顶点数量
        
        Returns:
            corrected_normal: (K, 3) 校正后的法向量
        """
        from torch.utils.checkpoint import checkpoint
        
        K = visible_coords.shape[0]
        results = []
        
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            chunk_result = checkpoint(
                compute_corrected_normal_from_subs,
                visible_coords[start:end],
                subs,
                visible_v_normal_cam[start:end],
                voxel_resolution,
                use_reentrant=False
            )
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)  # (K, 3)
    
    def _collect_visible_vertices(
        self, 
        rast: Tensor,  # (1, H, W, 4)
        faces: Tensor,  # (F, 3)
    ) -> Tensor:
        """从光栅化结果收集可见顶点索引
        
        Args:
            rast: 光栅化结果，rast[..., 3] 是 face_id + 1（0 表示背景）
            faces: 面索引
        
        Returns:
            visible_vertex_ids: (K,) 可见顶点索引
        """
        # rast[..., 3] 是 triangle_id，1-indexed（0 表示背景）
        face_ids = rast[0, ..., 3].long() - 1  # (H, W)
        visible_face_ids = face_ids[face_ids >= 0].unique()  # (num_visible_faces,)
        
        if visible_face_ids.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # 收集这些 face 涉及的 vertex_id
        visible_vertex_ids = faces[visible_face_ids].flatten().unique()  # (K,)
        
        return visible_vertex_ids
    
    def _empty_result(
        self, 
        resolution: int, 
        return_types: List[str]
    ) -> edict:
        """返回空结果（mesh 为空时）"""
        out_dict = edict()
        for rtype in return_types:
            if rtype == "normal":
                out_dict[rtype] = torch.full(
                    (resolution, resolution, 3), 0.5, 
                    dtype=torch.float32, device=self.device
                )
            elif rtype == "mask":
                out_dict[rtype] = torch.zeros(
                    (resolution, resolution), 
                    dtype=torch.float32, device=self.device
                )
            elif rtype == "depth":
                out_dict[rtype] = torch.zeros(
                    (resolution, resolution), 
                    dtype=torch.float32, device=self.device
                )
        return out_dict
