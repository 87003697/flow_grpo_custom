
"""
可微 Voxel Normal 渲染模块（基于 26 邻居缺失方向）

设计原则：
- 基于 26 邻居的存在/缺失来计算法向量
- 缺失邻居的方向指向物体外部，累加并归一化得到法向量
- 复用 o-voxel 原生 CUDA 哈希映射
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap


# =============================================================================
# 公共配置
# =============================================================================

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


# =============================================================================
# 公共辅助函数
# =============================================================================

def _smooth_normal_3x3(normal: Tensor, mask: Tensor) -> Tensor:
    """用 3x3 邻域平均平滑 normal（抑制锯齿）
    
    Args:
        normal: (H, W, 3)
        mask: (H, W) bool
    
    Returns:
        smoothed: (H, W, 3) 平滑后的 normal
    """
    device = normal.device
    
    # 转换为 (1, 3, H, W)
    normal_chw = normal.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    mask_chw = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # 3x3 均值核
    kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
    
    # 对每个通道做加权平均（只累加前景像素）
    smoothed_list = []
    for c in range(3):
        channel = normal_chw[:, c:c+1, :, :]  # (1, 1, H, W)
        smoothed_c = F.conv2d(channel * mask_chw, kernel, padding=1)  # (1, 1, H, W)
        smoothed_list.append(smoothed_c)
    
    # 计算有效邻居数（避免边界处除以 0）
    count = F.conv2d(mask_chw, kernel, padding=1).clamp(min=1e-6)  # (1, 1, H, W)
    
    # 归一化
    smoothed = torch.cat(smoothed_list, dim=1) / count  # (1, 3, H, W)
    smoothed = smoothed.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    
    # 重新归一化为单位向量
    smoothed = F.normalize(smoothed, dim=-1, eps=1e-6)
    
    # 只对 mask 内的像素有效
    smoothed = smoothed * mask.unsqueeze(-1)
    
    return smoothed


def _neighbor_offsets_26(device: torch.device) -> Tuple[Tensor, Tensor]:
    """生成 26 邻居的偏移量和权重
    
    26 邻居由三种类型组成：
    - 6 个面邻居（距离 1）：权重 1.0
    - 12 个边邻居（距离 √2）：权重 1/√2 ≈ 0.707
    - 8 个角邻居（距离 √3）：权重 1/√3 ≈ 0.577
    
    Returns:
        offsets: (26, 3) 邻居偏移
        weights: (26,) 权重（距离倒数）
    """
    offsets = []
    weights = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # 跳过自身
                offsets.append([dx, dy, dz])
                # 权重 = 1 / 距离，距离越近贡献越大
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                weights.append(1.0 / dist)
    
    offsets = torch.tensor(offsets, dtype=torch.int, device=device)  # (26, 3)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)  # (26,)
    return offsets, weights


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
    render_ret = renderer.render(positions, attrs, config.voxel_size, config.extrinsic, config.intrinsic)
    voxel_id = render_ret["voxel_id"]  # (H, W)
    return voxel_id


def _flip_normals_to_camera(
    voxel_normals: Tensor,  # (N, 3)
    surface_pos: Tensor,    # (N, 3)
    extrinsics: Tensor,     # (4, 4)
) -> Tensor:
    """变换到 Camera Space + 用点积翻转"""
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]  # (3,)
    voxel_normals_cam = voxel_normals @ R.T  # (N, 3)
    surface_pos_cam = surface_pos @ R.T + t  # (N, 3)
    dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N, 1)
    voxel_normals_cam = torch.where(dot_product > 0, voxel_normals_cam, -voxel_normals_cam)  # (N, 3)
    return voxel_normals_cam


# =============================================================================
# 26-Neighbor 不可导版本（用于对比）
# =============================================================================

def find_neighbor_mask_26(
    coords: Tensor,     # (N, 3) int
    grid_size: Tensor,  # (3,)
) -> Tensor:
    """使用 GPU 哈希表查询 26 邻居的存在性
    
    Returns:
        neighbor_exists: (N, 26) bool, True 表示该邻居存在
    """
    N = coords.shape[0]
    device = coords.device
    INVALID = 0xffffffff
    
    # 初始化哈希表
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())
    
    # 26 邻居偏移
    offsets, _ = _neighbor_offsets_26(device)  # (26, 3)
    
    # 计算所有邻居坐标: (N, 26, 3)
    neighbor_coords = coords[:, None, :] + offsets[None, :, :]  # (N, 26, 3)
    neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (N*26, 3)
    
    # 添加 batch 维度用于查询
    batch_zeros = torch.zeros((N * 26, 1), dtype=torch.int, device=device)  # (N*26, 1)
    query = torch.cat([batch_zeros, neighbor_coords_flat], dim=-1)  # (N*26, 4)
    
    # 查询
    indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())  # (N*26,)
    neighbor_exists = (indices != INVALID).reshape(N, 26)  # (N, 26)
    
    return neighbor_exists


def _compute_normal_from_missing_neighbors(
    coords: Tensor,     # (N, 3) int
    grid_size: Tensor,  # (3,)
) -> Tensor:
    """基于 26 邻居缺失方向计算法向量（不可导）"""
    device = coords.device
    
    offsets, weights = _neighbor_offsets_26(device)  # (26, 3), (26,)
    directions = F.normalize(offsets.float(), dim=-1, eps=1e-6)  # (26, 3)
    
    neighbor_exists = find_neighbor_mask_26(coords, grid_size)  # (N, 26)
    neighbor_missing = ~neighbor_exists  # (N, 26)
    
    weighted_directions = directions * weights[:, None]  # (26, 3)
    missing_contribution = neighbor_missing[:, :, None].float() * weighted_directions[None, :, :]  # (N, 26, 3)
    normal_sum = missing_contribution.sum(dim=1)  # (N, 3)
    normals = F.normalize(normal_sum, dim=-1, eps=1e-6)  # (N, 3)
    
    return normals


def render_normal_26neighbor(
    coords: Tensor,      # (N, 3) int
    config: RenderConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """基于 26 邻居缺失方向的法向量渲染（不可导）"""
    voxel_normals = _compute_normal_from_missing_neighbors(coords, config.grid_size)  # (N, 3)
    surface_pos = (coords.float() + 0.5) * config.voxel_size + config.origin  # (N, 3)
    
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    
    voxel_normals_cam = _flip_normals_to_camera(voxel_normals, surface_pos, config.extrinsic)  # (N, 3)
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)
    
    # 3x3 邻域平滑（抑制锯齿）
    pixel_normal = _smooth_normal_3x3(pixel_normal, mask)  # (H, W, 3)
    
    if target_size is not None:
        pixel_normal = F.interpolate(
            pixel_normal.permute(2, 0, 1).unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        pixel_normal = F.normalize(pixel_normal, dim=-1, eps=1e-6)
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0.5
    
    return pixel_normal, mask


# =============================================================================
# 多分辨率可微版本
# =============================================================================

def _expand_parent_to_child(
    parent_coords: Tensor,   # (N, 3) parent 层坐标
    parent_logits: Tensor,   # (N, 8) parent 层 logits
) -> Tuple[Tensor, Tensor, Tensor]:
    """根据 parent 层 logits 生成 child 层 voxel 坐标（通用版本）
    
    parent 层分辨率为 R，则 child 层分辨率为 2R
    
    Returns:
        coords_child: (M, 3) child 层坐标
        parent_indices: (M,) 每个 child 层 voxel 对应的 parent 索引
        corner_indices: (M,) 每个 child 层 voxel 对应的 corner 索引 (0-7)
    """
    device = parent_coords.device
    
    # 8 个 corner 的偏移
    corner_offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.int, device=device)  # (8, 3)
    
    # 哪些 corner 被占用
    occupied = parent_logits > 0  # (N, 8) bool
    
    # 使用稀疏索引
    parent_idx, corner_idx = occupied.nonzero(as_tuple=True)  # (M,), (M,)
    
    # 计算 child 层坐标：parent * 2 + offset
    base_coords = parent_coords[parent_idx] * 2  # (M, 3)
    offsets = corner_offsets[corner_idx]  # (M, 3)
    coords_child = base_coords + offsets  # (M, 3)
    
    return coords_child, parent_idx, corner_idx


def _compute_neighbor_occupancy_soft(
    neighbor_coords: Tensor,       # (K, N, 3) 目标分辨率下的查询坐标（N 可以是 26 或 27 等）
    subs: List[Any],               # [sub0, sub1, sub2, ...] 各层 sub logits
    voxel_resolution: int,         # 目标 voxel 分辨率（如 512, 1024, 1536）
) -> Tensor:
    """计算查询位置的 soft occupancy（可微，支持多分辨率）
    
    层级查找逻辑：
    - 从最高层 parent 开始查找
    - subs[-1] 的分辨率是 voxel_resolution // 2，决定目标层
    - 如果找不到，依次向更低分辨率查找
    
    Args:
        neighbor_coords: (K, N, 3) 目标分辨率下的查询坐标
        subs: 各层 sub logits
        voxel_resolution: 目标 voxel 分辨率
    
    Returns:
        occupancy: (K, N) 范围 [0, 1]，可微
    """
    device = neighbor_coords.device
    K, N = neighbor_coords.shape[0], neighbor_coords.shape[1]
    INVALID = 0xffffffff
    
    # 初始化结果
    found_mask = torch.zeros(K, N, dtype=torch.bool, device=device)  # (K, N)
    found_occupancy = torch.zeros(K, N, device=device)  # (K, N)
    
    # 从最高分辨率的 parent 开始
    for level in range(len(subs) - 1, -1, -1):
        sub = subs[level]
        sub_coords = sub.coords[:, 1:]  # (M, 3)
        sub_logits = sub.feats.float()  # (M, 8)
        M = sub_coords.shape[0]
        
        # 当前层的分辨率（根据层级动态计算）
        # subs[-1] 的分辨率是 voxel_resolution // 2
        level_resolution = voxel_resolution // (2 ** (len(subs) - level))
        child_resolution = level_resolution * 2
        
        # 邻居坐标映射到 child 层（用于计算 corner_idx）
        scale_to_child = voxel_resolution // child_resolution
        neighbor_coords_child = neighbor_coords // scale_to_child  # (K, N, 3)
        
        # 邻居坐标映射到当前层（用于查找 parent）
        scale_to_level = voxel_resolution // level_resolution
        neighbor_coords_level = neighbor_coords // scale_to_level  # (K, N, 3)
        
        # 构建当前层哈希表
        grid_size = torch.tensor([level_resolution] * 3, device=device)
        hashmap = _init_hashmap(grid_size, int(2.5 * M) + 1, device)
        sub_coords_with_batch = torch.cat([
            torch.zeros_like(sub_coords[:, :1]), sub_coords
        ], dim=-1)  # (M, 4)
        _C.hashmap_insert_3d_idx_as_val_cuda(
            *hashmap, sub_coords_with_batch, *grid_size.tolist()
        )
        
        # 查询
        query = torch.cat([
            torch.zeros((K * N, 1), dtype=torch.int, device=device),
            neighbor_coords_level.reshape(-1, 3)
        ], dim=-1)  # (K*N, 4)
        indices = _C.hashmap_lookup_3d_cuda(
            *hashmap, query, *grid_size.tolist()
        ).reshape(K, N)  # (K, N)
        
        # 找到的邻居（且之前没找到过）
        exists = (indices != INVALID)  # (K, N)
        newly_found = exists & ~found_mask  # (K, N)
        
        if newly_found.any():
            # 计算 corner_idx：用 child 层坐标
            corner_idx = (
                (neighbor_coords_child[..., 0] % 2) +
                (neighbor_coords_child[..., 1] % 2) * 2 +
                (neighbor_coords_child[..., 2] % 2) * 4
            ).long()  # (K, N)
            
            # 只处理 newly_found 的位置，节省显存
            newly_found_flat = newly_found.reshape(-1)  # (K*N,)
            newly_found_idx = newly_found_flat.nonzero(as_tuple=True)[0]  # (num_found,)
            
            indices_flat = indices.long().reshape(-1)  # (K*N,)
            corner_idx_flat = corner_idx.reshape(-1)  # (K*N,)
            
            indices_sel = indices_flat[newly_found_idx].clamp(0, M - 1)  # (num_found,)
            corner_sel = corner_idx_flat[newly_found_idx]  # (num_found,)
            
            # 获取 corner logit
            parent_logits_sel = sub_logits[indices_sel]  # (num_found, 8)
            specific_logit_sel = parent_logits_sel.gather(-1, corner_sel.unsqueeze(-1)).squeeze(-1)  # (num_found,)
            neighbor_occ_sel = torch.sigmoid(specific_logit_sel)  # (num_found,)
            
            # 更新
            found_occupancy_flat = found_occupancy.reshape(-1)  # (K*N,)
            found_occupancy_flat[newly_found_idx] = neighbor_occ_sel
            found_occupancy = found_occupancy_flat.reshape(K, N)  # (K, N)
            
            found_mask = found_mask | newly_found  # (K, N)
        
        if found_mask.all():
            break
    
    # 未找到的设为 0（完全空气）
    return found_occupancy  # (K, N)


def _compute_normal_from_soft_occupancy(
    occupancy: Tensor,  # (K, 26)
    device: torch.device,
    grad_shrink: float = 0.01,
) -> Tensor:
    """从 soft occupancy 计算法向量（纯 soft，类似 TRELLIS.2）
    
    直接使用 sigmoid 输出的 occupancy，完全可微
    - occupancy 接近 1：邻居存在，missing_weight 接近 0
    - occupancy 接近 0：邻居缺失，missing_weight 接近 1
    """
    offsets, dist_weights = _neighbor_offsets_26(device)  # (26, 3), (26,)
    directions = F.normalize(offsets.float(), dim=-1, eps=1e-6)  # (26, 3)
    
    # 稳定训练
    occupancy = grad_shrink * occupancy + (1 - grad_shrink) * occupancy.detach()
    # 纯 soft：直接用 1 - occupancy 作为 missing 权重
    missing_weight = 1.0 - occupancy  # (K, 26)
    
    # 加权累加
    weighted_dirs = directions * dist_weights[:, None]  # (26, 3)
    contribution = missing_weight[:, :, None] * weighted_dirs[None, :, :]  # (K, 26, 3)
    normal_sum = contribution.sum(dim=1)  # (K, 3)
    normals = F.normalize(normal_sum, dim=-1, eps=1e-6)  # (K, 3)
    
    return normals


def render_sub_normal_soft(
    subs: List[Any],              # [sub0, sub1, sub2, ...] 各层 sub logits
    config: RenderConfig,
    h: Any,                       # 目标层 SparseTensor
    voxel_resolution: int,        # voxel 分辨率（如 512, 1024）
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """多分辨率可微法向量渲染
    
    流程：
    1. 从 h 获取目标层坐标
    2. hard_render → 可见 voxel_id
    3. 筛选可见 voxel
    4. 计算 26 邻居 soft occupancy (查 subs)
    5. 计算法向量
    6. 翻转 + 采样到像素
    
    Args:
        subs: 各层 sub logits，subs[-1] 的分辨率是 voxel_resolution // 2
        config: 渲染配置
        h: 目标层的 SparseTensor
        voxel_resolution: voxel 分辨率
        target_size: 输出图像分辨率
        
    Returns:
        pixel_normal: (H, W, 3)
        mask: (H, W)
    """
    device = subs[0].coords.device
    
    # 步骤 1：获取目标层坐标
    coords_target = h.coords[:, 1:]  # (M, 3)
    M = coords_target.shape[0]
    print(f"目标层 ({voxel_resolution}) voxel 数量: {M}")
    
    # 步骤 2：Hard render
    voxel_id = hard_render(coords_target, config)  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    
    # 步骤 3：筛选可见 voxel
    visible_ids = voxel_id[mask]  # (num_visible_pixels,)
    unique_visible_ids = visible_ids.unique()  # (K,)
    visible_coords = coords_target[unique_visible_ids]  # (K, 3)
    K = visible_coords.shape[0]
    print(f"可见 voxel 数量: {K}")
    
    # 步骤 4：计算 26 邻居坐标
    offsets, _ = _neighbor_offsets_26(device)  # (26, 3)
    neighbor_coords = visible_coords[:, None, :] + offsets[None, :, :]  # (K, 26, 3)
    
    # 步骤 5：计算邻居 soft occupancy
    neighbor_occupancy = _compute_neighbor_occupancy_soft(
        neighbor_coords, subs, voxel_resolution
    )  # (K, 26)
    
    # 步骤 6：计算法向量
    visible_normals = _compute_normal_from_soft_occupancy(
        neighbor_occupancy, device
    )  # (K, 3)
    
    # 步骤 7：映射回所有 voxel
    all_normals = torch.zeros(M, 3, device=device)  # (M, 3)
    all_normals[unique_visible_ids] = visible_normals  # (M, 3)
    
    # 步骤 8：计算表面位置并翻转法向量
    voxel_size = 1.0 / voxel_resolution
    origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
    surface_pos = (coords_target.float() + 0.5) * voxel_size + origin  # (M, 3)
    
    all_normals_cam = _flip_normals_to_camera(
        all_normals, surface_pos, config.extrinsic
    )  # (M, 3)
    
    # 步骤 9：采样到像素
    pixel_normal = all_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)
    
    # 步骤 10：3x3 邻域平滑（抑制锯齿）
    pixel_normal = _smooth_normal_3x3(pixel_normal, mask)  # (H, W, 3)
    
    # 步骤 11：可选 resize
    if target_size is not None:
        pixel_normal = F.interpolate(
            pixel_normal.permute(2, 0, 1).unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        pixel_normal = F.normalize(pixel_normal, dim=-1, eps=1e-6)
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode="nearest",
        ).squeeze(0).squeeze(0) > 0.5
    
    return pixel_normal, mask


# =============================================================================
# 兼容接口
# =============================================================================

def render_normal_sub(
    sub: Any,                 # feats: (N, 8), coords: (N, 4)
    config: RenderConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """单层 Sub 模式：使用 26 邻居算法渲染法向量（不可导）"""
    coords = sub.coords[:, 1:]  # (N, 3)
    return render_normal_26neighbor(coords, config, target_size)
