
"""
可微 Voxel Normal 渲染模块。

设计原则：
- 主函数端到端：渲染 + normal 计算
- 公共参数用 dataclass 封装
- 复用 o-voxel 原生 CUDA 哈希映射
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap


@dataclass
class RenderConfig:
    """渲染配置"""
    intrinsics: Tensor  # (3, 3)
    extrinsics: Tensor  # (4, 4)
    resolution: int
    voxel_size: float
    origin: Tensor  # (3,)
    grid_size: Tensor  # (3,)
    near: float = 1.0
    far: float = 100.0


def _edge_neighbor_voxel_offset(device: torch.device) -> Tensor:
    """每个轴的 4 个邻居偏移"""
    offsets = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # axis=0: YZ
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # axis=1: XZ
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # axis=2: XY
        ],
        dtype=torch.int,
        device=device,
    )  # (3, 4, 3)
    return offsets


def hard_render(coords: Tensor, config: RenderConfig) -> Tensor:
    """
    硬渲染获取 voxel_id。

    Returns:
        voxel_id: (H, W) int，-1 表示背景
    """
    import o_voxel

    coords_int = coords.int()  # (N, 3)
    positions = (coords_int.float() + 0.5) * config.voxel_size + config.origin  # (N, 3)
    attrs = torch.ones((coords_int.shape[0], 1), device=coords_int.device, dtype=positions.dtype)  # (N, 1)
    renderer = o_voxel.rasterize.VoxelRenderer({
        "resolution": config.resolution,
        "near": config.near,
        "far": config.far,
        "ssaa": 1,
    })
    render_ret = renderer.render(positions, attrs, config.voxel_size, config.extrinsics, config.intrinsics)  # dict, voxel_id: (H, W)
    voxel_id = render_ret["voxel_id"]  # (H, W)
    return voxel_id


def find_neighbor_indices_per_neighbor(
    coords: Tensor,              # (N, 3)
    neighbor_offsets: Tensor,    # (3, 4, 3)
    grid_size: Tensor,           # (3,)
) -> Tuple[Tensor, Tensor]:
    """
    使用 o-voxel 原生 CUDA 哈希映射查找邻居索引。

    Returns:
        neighbor_idx: (N, 3, 4) int
        neighbor_valid: (N, 3, 4) bool
    """
    N = coords.shape[0]
    device = coords.device
    INVALID = 0xffffffff

    hashmap = _init_hashmap(grid_size, 2 * N, device)  # (2*N,), (2*N,)
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())

    neighbor_idx_list: List[Tensor] = []
    neighbor_valid_list: List[Tensor] = []

    for axis in range(3):
        offsets = neighbor_offsets[axis]  # (4, 3)
        neighbor_coords = coords.unsqueeze(1) + offsets  # (N, 4, 3)
        neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (N * 4, 3)
        batch_zeros = torch.zeros((N * 4, 1), dtype=torch.int, device=device)  # (N * 4, 1)
        query = torch.cat([batch_zeros, neighbor_coords_flat], dim=-1)  # (N * 4, 4)
        indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())  # (N * 4,)
        indices = indices.reshape(N, 4)  # (N, 4)
        valid = (indices != INVALID)  # (N, 4)
        indices = indices.int()  # (N, 4)
        indices[~valid] = 0  # (N, 4)
        neighbor_idx_list.append(indices)
        neighbor_valid_list.append(valid)

    neighbor_idx = torch.stack(neighbor_idx_list, dim=1)  # (N, 3, 4)
    neighbor_valid = torch.stack(neighbor_valid_list, dim=1)  # (N, 3, 4)
    return neighbor_idx, neighbor_valid


def _compute_axis_normal_mean(
    neighbor_pos: Tensor,   # (N, 4, 3)
    neighbor_valid: Tensor, # (N, 4)
    dual_vertices: Tensor,  # (N, 3)
) -> Tensor:
    """对所有有效三角形的法线取均值（单轴）"""
    v0, v1, v2, v3 = neighbor_pos.unbind(dim=1)  # (N, 3) x4
    m0, m1, m2, m3 = neighbor_valid.unbind(dim=1)  # (N,) x4

    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    vertices = [v0, v1, v2, v3]
    masks = [m0, m1, m2, m3]

    all_normals: List[Tensor] = []
    all_valid: List[Tensor] = []

    for (i, j, k) in triangles:
        tri_valid = masks[i] & masks[j] & masks[k]  # (N,)
        vi, vj, vk = vertices[i], vertices[j], vertices[k]  # (N, 3) x3
        tri_normal = F.normalize(torch.cross(vj - vi, vk - vi, dim=-1), dim=-1, eps=1e-6)  # (N, 3)
        all_normals.append(tri_normal)
        all_valid.append(tri_valid)

    all_normals = torch.stack(all_normals, dim=1)  # (N, 4, 3)
    all_valid = torch.stack(all_valid, dim=1)  # (N, 4)
    masked_normals = all_normals * all_valid.unsqueeze(-1).float()  # (N, 4, 3)
    sum_normals = masked_normals.sum(dim=1)  # (N, 3)
    count = all_valid.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
    mean_normal = sum_normals / count  # (N, 3)
    has_valid = (all_valid.sum(dim=1) > 0)  # (N,)
    fallback = F.normalize(dual_vertices, dim=-1, eps=1e-6)  # (N, 3)
    axis_normal = torch.where(
        has_valid.unsqueeze(-1),
        F.normalize(mean_normal, dim=-1, eps=1e-6),
        fallback,
    )  # (N, 3)
    return axis_normal


def _compute_axis_face_normals(
    coords: Tensor,         # (N, 3)
    dual_vertices: Tensor,  # (N, 3)
    voxel_size: float,
    origin: Tensor,         # (3,)
    grid_size: Tensor,      # (3,)
) -> Tuple[Tensor, Tensor]:
    """计算每个 voxel 的 3 个轴方向 face normal（均值方案）"""
    surface_pos = (coords.float() + dual_vertices) * voxel_size + origin  # (N, 3)
    neighbor_offsets = _edge_neighbor_voxel_offset(coords.device)  # (3, 4, 3)
    neighbor_idx, neighbor_valid = find_neighbor_indices_per_neighbor(coords, neighbor_offsets, grid_size)  # (N, 3, 4), (N, 3, 4)

    axis_normals: List[Tensor] = []
    for axis in range(3):
        idx = neighbor_idx[:, axis, :]  # (N, 4)
        neighbor_pos = surface_pos[idx.clamp(min=0)]  # (N, 4, 3)
        valid = neighbor_valid[:, axis, :]  # (N, 4)
        axis_normal = _compute_axis_normal_mean(neighbor_pos, valid, dual_vertices)  # (N, 3)
        axis_normals.append(axis_normal)

    axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)
    return axis_normals, surface_pos


def _compute_occupancy_gradient(sub_logits: Tensor) -> Tensor:
    """occupancy 梯度作为法线方向"""
    occupancy = torch.sigmoid(sub_logits)  # (N, 8)
    grad_x = (occupancy[:, [1, 3, 5, 7]] - occupancy[:, [0, 2, 4, 6]]).mean(dim=1)  # (N,)
    grad_y = (occupancy[:, [2, 3, 6, 7]] - occupancy[:, [0, 1, 4, 5]]).mean(dim=1)  # (N,)
    grad_z = (occupancy[:, [4, 5, 6, 7]] - occupancy[:, [0, 1, 2, 3]]).mean(dim=1)  # (N,)
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (N, 3)
    voxel_normals = -F.normalize(gradient, dim=-1, eps=1e-6)  # (N, 3)
    return voxel_normals


def _flip_normals_to_camera(
    voxel_normals: Tensor,  # (N, 3)
    surface_pos: Tensor,    # (N, 3)
    extrinsics: Tensor,     # (4, 4)
) -> Tensor:
    """变换到 Camera Space + 用点积翻转
    
    参考 MeshRenderer：在 Camera Space 中计算 cross(e0, e1)
    顶点变换：p_cam = p_world @ R.T + t
    边向量变换：e_cam = e_world @ R.T（边是位置差，平移不影响）
    法线变换：n_cam = cross(e0_cam, e1_cam) = cross(e0 @ R.T, e1 @ R.T) = n_world @ R.T
    """
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]  # (3,)
    voxel_normals_cam = voxel_normals @ R.T  # (N, 3) 法线变换和顶点变换用同样的 R.T
    surface_pos_cam = surface_pos @ R.T + t  # (N, 3) 顶点变换
    dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N, 1)
    # MeshRenderer 的逻辑：如果 normal·v0 > 0 保持，否则翻转
    # 这使得法线最终指向远离相机的方向（和视线同向）
    voxel_normals_cam = torch.where(dot_product > 0, voxel_normals_cam, -voxel_normals_cam)  # (N, 3)
    return voxel_normals_cam


def render_normal_fdg(
    coords: Tensor,             # (N, 3)
    dual_vertices: Tensor,      # (N, 3)
    intersected_logits: Tensor, # (N, 3)
    config: RenderConfig,
) -> Tuple[Tensor, Tensor]:
    """FDG 模式：渲染 + 计算可微 normal"""
    axis_normals, surface_pos = _compute_axis_face_normals(
        coords, dual_vertices, config.voxel_size, config.origin, config.grid_size
    )  # (N, 3, 3), (N, 3)
    weights = torch.sigmoid(intersected_logits)  # (N, 3)
    weighted = (weights.unsqueeze(-1) * axis_normals).sum(dim=1)  # (N, 3)
    voxel_normals = F.normalize(weighted, dim=-1, eps=1e-6)  # (N, 3)
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    voxel_normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, config.extrinsics)  # (N, 3)
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)
    return pixel_normal, mask


def render_normal_sub(
    sub: Any,                 # feats: (N, 8), coords: (N, 4)
    config: RenderConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """单层 Sub 模式：渲染 + 计算可微 normal"""
    coords = sub.coords[:, 1:]  # (N, 3)
    voxel_normals = _compute_occupancy_gradient(sub.feats)  # (N, 3)
    surface_pos = coords.float() * config.voxel_size + config.origin  # (N, 3)
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    voxel_normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, config.extrinsics)  # (N, 3)
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)

    if target_size is not None:
        pixel_normal = F.interpolate(
            pixel_normal.permute(2, 0, 1).unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)  # (H_t, W_t, 3)
        pixel_normal = F.normalize(pixel_normal, dim=-1, eps=1e-6)  # (H_t, W_t, 3)
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0.5  # (H_t, W_t)

    return pixel_normal, mask


def render_normal_sub_multi(
    subs: List[Any],
    configs: List[RenderConfig],
    target_size: Tuple[int, int],
) -> List[Tuple[Tensor, Tensor]]:
    """多分辨率 Sub 模式"""
    results: List[Tuple[Tensor, Tensor]] = []
    for sub, config in zip(subs, configs):
        normal, mask = render_normal_sub(sub, config, target_size)  # (H, W, 3), (H, W)
        results.append((normal, mask))
    return results