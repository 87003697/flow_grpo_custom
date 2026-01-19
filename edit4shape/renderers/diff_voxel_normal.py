
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
    """渲染配置（简化接口）
    
    内部自动计算：
    - origin = [-0.5, -0.5, -0.5]
    - voxel_size = 1.0 / resolution
    - grid_size = [resolution, resolution, resolution]
    """
    extrinsic: Tensor   # (4, 4) 相机外参
    intrinsic: Tensor   # (3, 3) 相机内参
    resolution: int     # 分辨率（渲染输出 + grid_size）
    ssaa: int = 1       # 超采样抗锯齿
    near: float = 1.0
    far: float = 100.0
    
    @property
    def voxel_size(self) -> float:
        return 1.0 / self.resolution
    
    @property
    def origin(self) -> Tensor:
        return torch.tensor([-0.5, -0.5, -0.5], device=self.extrinsic.device)
    
    @property
    def grid_size(self) -> Tensor:
        r = self.resolution
        return torch.tensor([r, r, r], device=self.extrinsic.device)


def _edge_neighbor_voxel_offset(device: torch.device) -> Tensor:
    """每个轴的 8 个邻居偏移（双向：正方向 4 个 + 负方向 4 个）
    
    设计：对于每个轴，查找两个方向的面邻居
    - 正向组 [0:4]：+Y+Z / +X+Z / +X+Y 方向的单位正方形
    - 负向组 [4:8]：-Y-Z / -X-Z / -X-Y 方向的单位正方形
    """
    offsets = torch.tensor(
        [
            # axis=0: YZ 平面（法线 -X）
            # 正向：+Y+Z → cross 产生 -X
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
            # 负向：-Y-Z → cross 产生 -X（与正向一致）
             [0, 0, 0], [0, 0, -1], [0, -1, -1], [0, -1, 0]],
            # axis=1: XZ 平面（法线 -Y）
            # 正向：+X+Z → cross 产生 -Y
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
            # 负向：-X-Z → cross 产生 -Y（与正向一致）
             [0, 0, 0], [-1, 0, 0], [-1, 0, -1], [0, 0, -1]],
            # axis=2: XY 平面（法线 -Z）
            # 正向：+X+Y → cross 产生 -Z
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
            # 负向：-X-Y → cross 产生 -Z（与正向一致）
             [0, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]],
        ],
        dtype=torch.int,
        device=device,
    )  # (3, 8, 3)
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
        "ssaa": config.ssaa,
    })
    render_ret = renderer.render(positions, attrs, config.voxel_size, config.extrinsic, config.intrinsic)  # dict, voxel_id: (H, W)
    voxel_id = render_ret["voxel_id"]  # (H, W)
    return voxel_id


def find_neighbor_indices_per_neighbor(
    coords: Tensor,              # (N, 3)
    neighbor_offsets: Tensor,    # (3, K, 3) K=邻居数，双向时 K=8
    grid_size: Tensor,           # (3,)
) -> Tuple[Tensor, Tensor]:
    """
    使用 o-voxel 原生 CUDA 哈希映射查找邻居索引。

    Returns:
        neighbor_idx: (N, 3, K) int
        neighbor_valid: (N, 3, K) bool
    """
    N = coords.shape[0]
    device = coords.device
    INVALID = 0xffffffff

    hashmap = _init_hashmap(grid_size, 2 * N, device)  # (2*N,), (2*N,)
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())

    neighbor_idx_list: List[Tensor] = []
    neighbor_valid_list: List[Tensor] = []
    K = neighbor_offsets.shape[1]  # 邻居数量（双向时 K=8）

    for axis in range(3):
        offsets = neighbor_offsets[axis]  # (K, 3)
        neighbor_coords = coords.unsqueeze(1) + offsets  # (N, K, 3)
        neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (N * K, 3)
        batch_zeros = torch.zeros((N * K, 1), dtype=torch.int, device=device)  # (N * K, 1)
        query = torch.cat([batch_zeros, neighbor_coords_flat], dim=-1)  # (N * K, 4)
        indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())  # (N * K,)
        indices = indices.reshape(N, K)  # (N, K)
        valid = (indices != INVALID)  # (N, K)
        indices = indices.int()  # (N, K)
        indices[~valid] = 0  # (N, K)
        neighbor_idx_list.append(indices)
        neighbor_valid_list.append(valid)

    neighbor_idx = torch.stack(neighbor_idx_list, dim=1)  # (N, 3, K)
    neighbor_valid = torch.stack(neighbor_valid_list, dim=1)  # (N, 3, K)
    return neighbor_idx, neighbor_valid


def _compute_axis_normal_mean(
    neighbor_pos: Tensor,   # (N, 8, 3) 双向邻居
    neighbor_valid: Tensor, # (N, 8)
    dual_vertices: Tensor,  # (N, 3)
) -> Tensor:
    """对所有有效三角形的法线取均值（单轴，双向邻居版本）
    
    邻居布局：
    - [0:4]: 正向组（+方向的单位正方形 4 个角点）
    - [4:8]: 负向组（-方向的单位正方形 4 个角点）
    每组内部生成 C(4,3)=4 个三角形，共 8 个三角形
    
    注意：正向组和负向组的顶点绕序已在 _edge_neighbor_voxel_offset 中
    调整为一致，因此使用相同的 cross 计算会产生相同方向的法线。
    """
    N = neighbor_pos.shape[0]
    device = neighbor_pos.device
    
    # 三角形索引：每组 4 个点的 C(4,3)=4 种组合
    # 正向组用 [0,1,2,3]，负向组用 [4,5,6,7]
    tri_idx_base = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device=device)  # (4, 3)
    tri_idx = torch.cat([tri_idx_base, tri_idx_base + 4], dim=0)  # (8, 3) 正向+负向
    
    # 提取三角形顶点：(N, 8, 3) -> (N, 8, 3, 3) 每个三角形 3 个顶点
    vi = neighbor_pos[:, tri_idx[:, 0], :]  # (N, 8, 3)
    vj = neighbor_pos[:, tri_idx[:, 1], :]  # (N, 8, 3)
    vk = neighbor_pos[:, tri_idx[:, 2], :]  # (N, 8, 3)
    
    # 批量计算法线：cross(vj - vi, vk - vi)
    tri_normals = F.normalize(torch.cross(vj - vi, vk - vi, dim=-1), dim=-1, eps=1e-6)  # (N, 8, 3)
    
    # 三角形有效性：3 个顶点都有效
    mi = neighbor_valid[:, tri_idx[:, 0]]  # (N, 8)
    mj = neighbor_valid[:, tri_idx[:, 1]]  # (N, 8)
    mk = neighbor_valid[:, tri_idx[:, 2]]  # (N, 8)
    tri_valid = mi & mj & mk  # (N, 8)
    
    # 加权平均
    masked_normals = tri_normals * tri_valid.unsqueeze(-1).float()  # (N, 8, 3)
    sum_normals = masked_normals.sum(dim=1)  # (N, 3)
    count = tri_valid.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
    mean_normal = sum_normals / count  # (N, 3)
    
    # Fallback
    has_valid = (tri_valid.sum(dim=1) > 0)  # (N,)
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
    """计算每个 voxel 的 3 个轴方向 face normal（均值方案，双向邻居）"""
    surface_pos = (coords.float() + dual_vertices) * voxel_size + origin  # (N, 3)
    neighbor_offsets = _edge_neighbor_voxel_offset(coords.device)  # (3, 8, 3)
    neighbor_idx, neighbor_valid = find_neighbor_indices_per_neighbor(coords, neighbor_offsets, grid_size)  # (N, 3, 8), (N, 3, 8)

    axis_normals: List[Tensor] = []
    for axis in range(3):
        idx = neighbor_idx[:, axis, :]  # (N, 8)
        neighbor_pos = surface_pos[idx.clamp(min=0)]  # (N, 8, 3)
        valid = neighbor_valid[:, axis, :]  # (N, 8)
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
    voxel_normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, config.extrinsic)  # (N, 3)
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
    # 确保 feats 是 float32（可能是 float16）
    feats = sub.feats.float()  # (N, 8)
    voxel_normals = _compute_occupancy_gradient(feats)  # (N, 3)
    surface_pos = coords.float() * config.voxel_size + config.origin  # (N, 3)
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    voxel_normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, config.extrinsic)  # (N, 3)
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


def render_normal_sub_pyramid(
    subs: List[Any],
    configs: List[RenderConfig],
    target_size: Tuple[int, int],
    weights: Optional[List[float]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    金字塔融合多分辨率 Sub 渲染结果（加权平均）
    
    Args:
        subs: 多层 sub_logits
        configs: 每层的渲染配置
        target_size: 输出分辨率 (H, W)
        weights: 每层权重，None 时使用 [1, 2, 4, 8, ...]（高分辨率权重更大）
    
    Returns:
        fused_normal: (H, W, 3)
        fused_mask: (H, W)
    """
    num_layers = len(subs)
    H, W = target_size
    device = subs[0].coords.device
    
    # 默认权重：高分辨率层权重更大
    if weights is None:
        weights = [2.0 ** i for i in range(num_layers)]  # [1, 2, 4, 8]
    
    # 归一化权重
    total = sum(weights)
    weights = [w / total for w in weights]  # List[float]
    
    # 渲染每层并加权求和
    fused_normal = torch.zeros(H, W, 3, device=device)  # (H, W, 3)
    fused_mask = torch.zeros(H, W, device=device, dtype=torch.bool)  # (H, W)
    
    for w, sub, config in zip(weights, subs, configs):
        normal, mask = render_normal_sub(sub, config, target_size)  # (H, W, 3), (H, W)
        fused_normal = fused_normal + w * normal  # (H, W, 3)
        fused_mask = fused_mask | mask  # (H, W) 任意层有即为前景
    
    # 重新归一化 + 应用 mask
    fused_normal = F.normalize(fused_normal, dim=-1, eps=1e-6)  # (H, W, 3)
    fused_normal = fused_normal * fused_mask.unsqueeze(-1)  # (H, W, 3)
    
    return fused_normal, fused_mask