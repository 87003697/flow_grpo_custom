"""
可微 Voxel Normal 渲染模块 — 12-Quad 版本

核心特性：
- 基于 12 条边的 quad 法线计算
- 一致性翻转避免法线相消
- Gradient Checkpointing 控制显存

改进版（相比原方案）：
- 空气邻居（面邻居不存在）的 quad 跳过，与 mesh renderer 行为一致
- 中心-邻居 crossing（tanh 版本）：crossing = tanh((logit_c - logit_n) / T)^2
- 多分辨率简单平均融合（不用 log/exp，梯度更稳定）
- 邻居 logit 多分辨率查找，所有层都有梯度
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap


# ============================================================================
# 常量定义
# ============================================================================

# 12 条边对应的 sub_logit corner 索引对
# corner 编码：(x, y, z) → idx = x + 2*y + 4*z
EDGE_CORNER_PAIRS = torch.tensor([
    # X 轴方向的 4 条边（y, z 位置不同）
    [0, 1],  # (0,0,0)-(1,0,0)
    [2, 3],  # (0,1,0)-(1,1,0)
    [4, 5],  # (0,0,1)-(1,0,1)
    [6, 7],  # (0,1,1)-(1,1,1)
    # Y 轴方向的 4 条边
    [0, 2],  # (0,0,0)-(0,1,0)
    [1, 3],  # (1,0,0)-(1,1,0)
    [4, 6],  # (0,0,1)-(0,1,1)
    [5, 7],  # (1,0,1)-(1,1,1)
    # Z 轴方向的 4 条边
    [0, 4],  # (0,0,0)-(0,0,1)
    [1, 5],  # (1,0,0)-(1,0,1)
    [2, 6],  # (0,1,0)-(0,1,1)
    [3, 7],  # (1,1,0)-(1,1,1)
], dtype=torch.long)  # (12, 2)

# 12 条边的轴归属
EDGE_TO_AXIS = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2], dtype=torch.long)  # (12,)

# 数值稳定性常量
EPS = 1e-6

# 每条边对应的 3 个邻居偏移（2 面邻居 + 1 边邻居）
# 格式：[面邻居1, 面邻居2, 边邻居]
EDGE_NEIGHBOR_OFFSETS = torch.tensor([
    # X 轴边 0: (0,0,0)-(1,0,0) → 邻居 (0,-1,0), (0,0,-1), (0,-1,-1)
    [[0, -1, 0], [0, 0, -1], [0, -1, -1]],
    # X 轴边 1: (0,1,0)-(1,1,0) → 邻居 (0,+1,0), (0,0,-1), (0,+1,-1)
    [[0, 1, 0], [0, 0, -1], [0, 1, -1]],
    # X 轴边 2: (0,0,1)-(1,0,1) → 邻居 (0,-1,0), (0,0,+1), (0,-1,+1)
    [[0, -1, 0], [0, 0, 1], [0, -1, 1]],
    # X 轴边 3: (0,1,1)-(1,1,1) → 邻居 (0,+1,0), (0,0,+1), (0,+1,+1)
    [[0, 1, 0], [0, 0, 1], [0, 1, 1]],
    # Y 轴边 4~7, Z 轴边 8~11 类似...
    # Y 轴边 4: (0,0,0)-(0,1,0)
    [[-1, 0, 0], [0, 0, -1], [-1, 0, -1]],
    # Y 轴边 5: (1,0,0)-(1,1,0)
    [[1, 0, 0], [0, 0, -1], [1, 0, -1]],
    # Y 轴边 6: (0,0,1)-(0,1,1)
    [[-1, 0, 0], [0, 0, 1], [-1, 0, 1]],
    # Y 轴边 7: (1,0,1)-(1,1,1)
    [[1, 0, 0], [0, 0, 1], [1, 0, 1]],
    # Z 轴边 8: (0,0,0)-(0,0,1)
    [[-1, 0, 0], [0, -1, 0], [-1, -1, 0]],
    # Z 轴边 9: (1,0,0)-(1,0,1)
    [[1, 0, 0], [0, -1, 0], [1, -1, 0]],
    # Z 轴边 10: (0,1,0)-(0,1,1)
    [[-1, 0, 0], [0, 1, 0], [-1, 1, 0]],
    # Z 轴边 11: (1,1,0)-(1,1,1)
    [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
], dtype=torch.int)  # (12, 3, 3)


@dataclass
class RenderConfig:
    """渲染配置"""
    extrinsic: Tensor   # (4, 4) 相机外参
    intrinsic: Tensor   # (3, 3) 相机内参
    resolution: int     # 分辨率
    ssaa: int = 1
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


# ============================================================================
# 多分辨率 Crossing Weight（logit 求和后做差版本）
# ============================================================================

def _get_logit_at_level(
    sub: Any,                 # SparseTensor
    coords: Tensor,           # (K, 3) 目标分辨率坐标（可能包含越界坐标）
    voxel_resolution: int,
    level_resolution: int,
    default_logit: float = 0.0,  # 不存在返回 0，不影响求和
) -> Tensor:
    """查找坐标在某层的 logit，不存在或越界返回 default_logit
    
    Returns:
        logit: (K,) 每个坐标对应的 logit
    """
    K = coords.shape[0]
    device = coords.device
    M = sub.feats.shape[0]
    INVALID = 0xffffffff
    
    # 检查坐标是否在有效范围内（负数或超出 voxel_resolution 都是越界）
    coords_in_bounds = (
        (coords >= 0).all(dim=-1) & 
        (coords < voxel_resolution).all(dim=-1)
    )  # (K,)
    
    scale = voxel_resolution // level_resolution
    
    # 对越界坐标 clamp 到有效范围（后面通过 valid mask 过滤）
    coords_safe = coords.clamp(min=0, max=voxel_resolution - 1)  # (K, 3)
    parent_coords = coords_safe // scale  # (K, 3)
    
    # 建立 hashmap
    grid_size = torch.tensor([level_resolution] * 3, device=device)
    hashmap = _init_hashmap(grid_size, 2 * M + 1, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap, sub.coords.int(), *grid_size.tolist()
    )
    
    # 查找 parent
    query = torch.cat([
        torch.zeros((K, 1), dtype=torch.int, device=device),
        parent_coords.int()
    ], dim=-1)  # (K, 4)
    indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())  # (K,)
    
    # 有效性：坐标不越界 且 hashmap 找到了
    valid = coords_in_bounds & (indices != INVALID)  # (K,)
    indices = indices.long().clamp(min=0, max=M - 1)  # (K,) 安全索引
    
    # 计算 corner 索引（使用安全坐标）
    child_scale = max(scale // 2, 1)
    child_coords = coords_safe // child_scale  # (K, 3)
    corner_idx = (
        (child_coords[:, 0] % 2) +
        (child_coords[:, 1] % 2) * 2 +
        (child_coords[:, 2] % 2) * 4
    ).long()  # (K,) 范围 0-7
    
    # 获取 logit（使用安全索引）
    sub_feats = sub.feats.float()  # (M, 8)
    logit = sub_feats[indices, corner_idx]  # (K,)
    
    # 无效位置用 default_logit = 0
    logit = torch.where(valid, logit, torch.full_like(logit, default_logit))
    
    return logit


def _get_multi_level_logit_sum(
    subs: List[Any],          # 所有分辨率的 SparseTensor
    coords: Tensor,           # (K, 3) 最高分辨率坐标
    voxel_resolution: int,
) -> Tensor:
    """获取坐标在所有分辨率的 logit 之和
    
    不存在的层级 logit = 0，不影响总和。
    
    Returns:
        logit_sum: (K,)
    """
    K = coords.shape[0]
    device = coords.device
    logit_sum = torch.zeros(K, device=device)  # (K,)
    
    for level, sub in enumerate(subs):
        # 当前层分辨率：subs[-1] 的分辨率是 voxel_resolution // 2
        level_resolution = voxel_resolution // (2 ** (len(subs) - level))
        
        level_logit = _get_logit_at_level(
            sub, coords, voxel_resolution, level_resolution, default_logit=0.0
        )  # (K,)
        
        logit_sum = logit_sum + level_logit  # (K,)
    
    return logit_sum


def compute_crossing_weight_soft_and(
    subs: List[Any],           # List[SparseTensor]
    center_coords: Tensor,     # (N, 3)
    neighbor_coords: Tensor,   # (N, 12, 3) 每条边的边邻居坐标
    voxel_resolution: int,
    temperature: float = 2.0,
    use_checkpoint: bool = True,
) -> Tensor:
    """多分辨率 crossing weight（logit 求和后做差版本）
    
    算法：
    1. 分别计算中心和邻居在所有分辨率的 logit 之和
    2. 做差后过 tanh 激活
    3. 不存在的层级 logit=0，自动被忽略
    
    Returns:
        crossing_weights: (N, 12) 
    """
    N = center_coords.shape[0]
    
    # 中心的多分辨率 logit 之和
    if use_checkpoint:
        center_sum = checkpoint(
            _get_multi_level_logit_sum,
            subs,
            center_coords,
            voxel_resolution,
            use_reentrant=False,
        )  # (N,)
    else:
        center_sum = _get_multi_level_logit_sum(
            subs, center_coords, voxel_resolution
        )  # (N,)
    
    # 邻居的多分辨率 logit 之和
    neighbor_flat = neighbor_coords.reshape(-1, 3)  # (N*12, 3)
    
    if use_checkpoint:
        neighbor_sum_flat = checkpoint(
            _get_multi_level_logit_sum,
            subs,
            neighbor_flat,
            voxel_resolution,
            use_reentrant=False,
        )  # (N*12,)
    else:
        neighbor_sum_flat = _get_multi_level_logit_sum(
            subs, neighbor_flat, voxel_resolution
        )  # (N*12,)
    
    neighbor_sum = neighbor_sum_flat.reshape(N, 12)  # (N, 12)
    
    # 做差 + 激活: tanh((sum_c - sum_n) / T)^2
    diff = (center_sum.unsqueeze(-1) - neighbor_sum) / temperature  # (N, 12)
    crossing = torch.tanh(diff) ** 2  # (N, 12)
    
    return crossing


# ============================================================================
# 12-Quad 法线计算
# ============================================================================

def _find_quad_neighbors(
    query_coords: Tensor,    # (M, 3) 要查询邻居的坐标（可见 voxel）
    all_coords: Tensor,      # (N, 3) 全部 voxel 坐标（用于建立 hashmap）
    grid_size: Tensor,       # (3,)
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """查找每条边的 3 个邻居
    
    Args:
        query_coords: 要查询邻居的坐标（可见 voxel）
        all_coords: 全部 voxel 坐标（用于建立 hashmap，返回的索引是全局索引）
        grid_size: grid 尺寸
    
    Returns:
        neighbor_idx: (M, 12, 3) 邻居的全局索引
        neighbor_valid: (M, 12, 3) 邻居有效性
        neighbor_coords: (M, 12, 3, 3) 邻居坐标（用于空气体心计算）
        edge_neighbor_coords: (M, 12, 3) 边邻居坐标（用于 crossing 计算）
    """
    M = query_coords.shape[0]
    N = all_coords.shape[0]
    device = query_coords.device
    INVALID = 0xffffffff
    
    edge_offsets = EDGE_NEIGHBOR_OFFSETS.to(device)  # (12, 3, 3)
    
    # 用全局坐标建立 hashmap，返回的索引是全局索引
    hashmap = _init_hashmap(grid_size, 2 * N + 1, device)
    coords_with_batch = torch.cat([
        torch.zeros_like(all_coords[:, :1]),
        all_coords
    ], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())
    
    # 计算所有邻居坐标（基于 query_coords）
    neighbor_coords = query_coords.unsqueeze(1).unsqueeze(2) + edge_offsets  # (M, 12, 3, 3)
    neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (M * 12 * 3, 3)
    
    # 批量查找
    query = torch.cat([
        torch.zeros((M * 12 * 3, 1), dtype=torch.int, device=device),
        neighbor_coords_flat.int()
    ], dim=-1)  # (M * 12 * 3, 4)
    
    indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())
    indices = indices.reshape(M, 12, 3)  # (M, 12, 3)
    
    valid = (indices != INVALID)  # (M, 12, 3)
    indices = indices.long().clamp(min=0, max=N - 1)  # (M, 12, 3) 安全索引
    
    # 边邻居坐标（每条边的第 3 个邻居，用于 crossing 计算）
    edge_neighbor_coords = neighbor_coords[:, :, 2, :]  # (M, 12, 3)
    
    return indices, valid, neighbor_coords, edge_neighbor_coords


def _compute_quad_normals(
    surface_pos: Tensor,       # (N, 3) 中心 voxel 表面位置
    neighbor_idx: Tensor,      # (N, 12, 3) 邻居索引
    neighbor_valid: Tensor,    # (N, 12, 3) 邻居有效性
    neighbor_coords: Tensor,   # (N, 12, 3, 3) 邻居坐标
    all_surface_pos: Tensor,   # (M, 3) 所有 voxel 表面位置（用于索引邻居）
    voxel_size: float,
    origin: Tensor,            # (3,)
) -> Tuple[Tensor, Tensor]:
    """计算 12 个 quad 的法线
    
    改进：空气邻居（面邻居不存在）的 quad 标记为无效，不参与聚合
    
    每个 quad = 中心 + 2 面邻居 + 1 边邻居
    法线 = cross(v_face1 - v_center, v_face2 - v_center)
    
    Returns:
        quad_normals: (N, 12, 3) 每个 quad 的法线
        quad_valid: (N, 12) 有效性（叉积模长 > EPS 且两个面邻居都存在）
    """
    N = surface_pos.shape[0]
    device = surface_pos.device
    
    # 获取邻居位置（从已有 voxel 索引）
    neighbor_pos = all_surface_pos[neighbor_idx]  # (N, 12, 3, 3)
    
    # 空气邻居使用体心位置
    # 体心 = (coord + 0.5) * voxel_size + origin
    air_pos = (neighbor_coords.float() + 0.5) * voxel_size + origin  # (N, 12, 3, 3)
    
    # 替换无效邻居为体心
    neighbor_pos = torch.where(
        neighbor_valid.unsqueeze(-1),  # (N, 12, 3, 1)
        neighbor_pos,
        air_pos
    )  # (N, 12, 3, 3)
    
    # 提取两个面邻居
    v_center = surface_pos.unsqueeze(1)           # (N, 1, 3)
    v_face1 = neighbor_pos[:, :, 0, :]            # (N, 12, 3)
    v_face2 = neighbor_pos[:, :, 1, :]            # (N, 12, 3)
    
    # 计算法线：cross(e1, e2)
    e1 = v_face1 - v_center  # (N, 12, 3)
    e2 = v_face2 - v_center  # (N, 12, 3)
    cross_result = torch.cross(e1, e2, dim=-1)  # (N, 12, 3)
    cross_norm = cross_result.norm(dim=-1, keepdim=True)  # (N, 12, 1)
    
    # 归一化（避免除以 0）
    quad_normals = cross_result / cross_norm.clamp(min=EPS)  # (N, 12, 3)
    
    # 有效性：叉积模长够大 且 两个面邻居都存在
    cross_valid = (cross_norm.squeeze(-1) > EPS)  # (N, 12)
    face_valid = neighbor_valid[:, :, :2].all(dim=-1)  # (N, 12) 两个面邻居都存在
    quad_valid = cross_valid & face_valid  # (N, 12)
    
    return quad_normals, quad_valid


# ============================================================================
# 一致性翻转加权
# ============================================================================

def consistent_weighted_normal(
    normals: Tensor,    # (N, K, 3)
    weights: Tensor,    # (N, K)
    eps: float = EPS,
) -> Tensor:
    """一致性翻转 + 加权求和
    
    1. 选择权重最大的法线作为参考
    2. 其他法线与参考反向时翻转
    3. 翻转后加权求和 + 归一化
    4. 如果加权和接近 0，fallback 到参考法线
    """
    N, K, _ = normals.shape
    device = normals.device
    
    # 权重归一化（避免全零）
    weights = weights + eps
    
    # 参考方向：权重最大
    max_idx = weights.argmax(dim=-1)  # (N,)
    ref = normals[torch.arange(N, device=device), max_idx]  # (N, 3)
    
    # 点积判断方向
    dots = (normals * ref.unsqueeze(1)).sum(dim=-1)  # (N, K)
    
    # 翻转反向法线
    flip = (dots < 0).unsqueeze(-1)  # (N, K, 1)
    aligned = torch.where(flip, -normals, normals)  # (N, K, 3)
    
    # 加权求和
    weighted = (weights.unsqueeze(-1) * aligned).sum(dim=1)  # (N, 3)
    weighted_norm = weighted.norm(dim=-1, keepdim=True)  # (N, 1)
    
    # 归一化，如果模长太小则 fallback 到参考法线
    result = torch.where(
        weighted_norm > eps,
        weighted / weighted_norm.clamp(min=eps),
        ref
    )  # (N, 3)
    
    return result


# ============================================================================
# 两级聚合
# ============================================================================

def aggregate_to_final_normal(
    quad_normals: Tensor,          # (N, 12, 3)
    quad_valid: Tensor,            # (N, 12) 叉积有效性
    crossing_weights: Tensor,      # (N, 12)
    intersected_logits: Tensor,    # (N, 3)
) -> Tensor:
    """两级加权聚合
    
    层级1：同轴 4 条边 → 1 个轴法线（crossing weight）
    层级2：3 个轴 → 最终法线（intersected_logits weight）
    
    退化 quad（叉积为 0）的权重设为 0，不参与计算
    """
    N = quad_normals.shape[0]
    device = quad_normals.device
    
    edge_to_axis = EDGE_TO_AXIS.to(device)  # (12,)
    
    # 无效 quad 权重设为 0
    combined_weights = crossing_weights * quad_valid.float()  # (N, 12)
    
    # 层级1：每个轴的 4 条边 → 1 个轴法线
    axis_normals = []
    for axis in range(3):
        edge_mask = (edge_to_axis == axis)  # (4,) True positions
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]  # (4,)
        
        normals_ax = quad_normals[:, edge_ids]    # (N, 4, 3)
        weights_ax = combined_weights[:, edge_ids]  # (N, 4)
        
        axis_normal = consistent_weighted_normal(
            normals_ax, weights_ax
        )  # (N, 3)
        axis_normals.append(axis_normal)
    
    axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)
    
    # 层级2：3 个轴 → 最终法线
    axis_weights = torch.sigmoid(intersected_logits)  # (N, 3)
    
    final_normal = consistent_weighted_normal(
        axis_normals,   # (N, 3, 3)
        axis_weights    # (N, 3)
    )  # (N, 3)
    
    return final_normal


# ============================================================================
# 渲染器类
# ============================================================================

class Quad12NormalRenderer:
    """12-Quad 法线渲染器
    
    改进版：
    - 空气邻居使用体心坐标，所有 quad 都有效
    - 中心-邻居 crossing（tanh 版本），梯度更稳定
    - 多分辨率简单平均融合
    
    使用方式：
        renderer = Quad12NormalRenderer(use_checkpoint=True, temperature=2.0)
        normal, mask = renderer.render(h, subs, config)
    """
    
    def __init__(
        self,
        use_checkpoint: bool = True,
        temperature: float = 2.0,
    ):
        self.use_checkpoint = use_checkpoint
        self.temperature = temperature
    
    # -------------------- 主入口 --------------------
    
    def render(
        self,
        h: Any,                    # SparseTensor: feats (N, 7), coords (N, 4)
        subs: List[Any],           # List[SparseTensor]
        config: RenderConfig,
        voxel_margin: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """渲染法线图
        
        Returns:
            pixel_normal: (H, W, 3)
            mask: (H, W)
        """
        # 1. 提取 FDG 数据
        coords, dual_vertices, intersected_logits = self._extract_fdg_data(h, voxel_margin)
        
        # 2. 硬渲染 + 可见性筛选
        voxel_id, mask, visible_ids = self._get_visible_voxels(coords, config)
        
        if visible_ids.numel() == 0:
            H, W = voxel_id.shape
            return torch.zeros(H, W, 3, device=h.coords.device), mask
        
        # 3. 计算可见 voxel 的法线
        voxel_normals_cam = self._compute_visible_normals(
            coords, dual_vertices, intersected_logits,
            visible_ids, subs, config
        )  # (M, 3)
        
        # 4. 采样到像素
        pixel_normal = self._sample_to_pixels(
            voxel_normals_cam, visible_ids, voxel_id, mask, coords.shape[0]
        )  # (H, W, 3)
        
        return pixel_normal, mask
    
    # -------------------- 数据提取 --------------------
    
    def _extract_fdg_data(
        self,
        h: Any,
        voxel_margin: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """提取 FDG SparseTensor 数据
        
        Returns:
            coords: (N, 3) int
            dual_vertices: (N, 3) float
            intersected_logits: (N, 3) float
        """
        coords = h.coords[:, 1:].int()  # (N, 3)
        raw_vertices = h.feats[..., 0:3]
        dual_vertices = (1 + 2 * voxel_margin) * torch.sigmoid(raw_vertices) - voxel_margin  # (N, 3)
        intersected_logits = h.feats[..., 3:6]  # (N, 3)
        return coords, dual_vertices, intersected_logits
    
    # -------------------- 可见性 --------------------
    
    def _get_visible_voxels(
        self,
        coords: Tensor,
        config: RenderConfig,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """硬渲染获取可见 voxel
        
        Returns:
            voxel_id: (H, W) int
            mask: (H, W) bool
            visible_ids: (M,) long
        """
        voxel_id = self._hard_render(coords, config)  # (H, W)
        mask = voxel_id >= 0  # (H, W)
        visible_ids = voxel_id[mask].unique()  # (M,)
        return voxel_id, mask, visible_ids
    
    # -------------------- 核心法线计算 --------------------
    
    def _compute_visible_normals(
        self,
        coords: Tensor,              # (N, 3) 全部坐标
        dual_vertices: Tensor,       # (N, 3) 全部 dual_vertices
        intersected_logits: Tensor,  # (N, 3) 全部 intersected_logits
        visible_ids: Tensor,         # (M,) 可见索引
        subs: List[Any],
        config: RenderConfig,
    ) -> Tensor:
        """计算可见 voxel 的相机空间法线
        
        Returns:
            voxel_normals_cam: (M, 3)
        """
        # 提取可见部分
        coords_vis = coords[visible_ids]                          # (M, 3)
        dual_vertices_vis = dual_vertices[visible_ids]            # (M, 3)
        intersected_logits_vis = intersected_logits[visible_ids]  # (M, 3)
        
        # 全部表面位置（邻居查找需要）
        surface_pos_all = (coords.float() + dual_vertices) * config.voxel_size + config.origin  # (N, 3)
        surface_pos_vis = surface_pos_all[visible_ids]  # (M, 3)
        
        # 查找邻居（用全局 coords 建立 hashmap，返回全局索引）
        neighbor_idx, neighbor_valid, neighbor_coords, edge_neighbor_coords = \
            _find_quad_neighbors(coords_vis, coords, config.grid_size)
        # neighbor_idx: (M, 12, 3) 全局索引，neighbor_valid: (M, 12, 3)
        # neighbor_coords: (M, 12, 3, 3), edge_neighbor_coords: (M, 12, 3)
        
        # 多分辨率 crossing weight（使用边邻居坐标）
        crossing_weights = compute_crossing_weight_soft_and(
            subs, coords_vis, edge_neighbor_coords, config.resolution,
            temperature=self.temperature,
            use_checkpoint=self.use_checkpoint,
        )  # (M, 12)
        
        # 12-Quad 法线（空气邻居用体心）
        quad_normals, quad_valid = _compute_quad_normals(
            surface_pos_vis, neighbor_idx, neighbor_valid, neighbor_coords,
            surface_pos_all, config.voxel_size, config.origin
        )  # (M, 12, 3), (M, 12)
        
        # 两级聚合（无效 quad 权重设为 0）
        voxel_normals = aggregate_to_final_normal(
            quad_normals, quad_valid, crossing_weights, intersected_logits_vis
        )  # (M, 3)
        
        # 翻转到相机空间
        voxel_normals_cam = self._flip_normals_to_camera(
            voxel_normals, surface_pos_vis, config.extrinsic
        )  # (M, 3)
        
        return voxel_normals_cam
    
    # -------------------- 像素采样 --------------------
    
    def _sample_to_pixels(
        self,
        voxel_normals: Tensor,  # (M, 3)
        visible_ids: Tensor,    # (M,)
        voxel_id: Tensor,       # (H, W)
        mask: Tensor,           # (H, W)
        num_voxels: int,        # N
    ) -> Tensor:
        """将 voxel 法线采样到像素
        
        Returns:
            pixel_normal: (H, W, 3)
        """
        device = voxel_normals.device
        
        # 建立 visible_ids -> 新索引的映射
        id_map = torch.zeros(num_voxels, dtype=torch.long, device=device)
        id_map[visible_ids] = torch.arange(len(visible_ids), device=device)
        
        # 采样
        voxel_id_mapped = id_map[voxel_id.clamp(min=0)]  # (H, W)
        pixel_normal = voxel_normals[voxel_id_mapped]    # (H, W, 3)
        pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)
        
        return pixel_normal
    
    # -------------------- 工具方法 --------------------
    
    def _hard_render(self, coords: Tensor, config: RenderConfig) -> Tensor:
        """硬渲染获取 voxel_id"""
        import o_voxel
        
        coords_int = coords.int()
        positions = (coords_int.float() + 0.5) * config.voxel_size + config.origin
        attrs = torch.ones((coords_int.shape[0], 1), device=coords.device, dtype=positions.dtype)
        
        renderer = o_voxel.rasterize.VoxelRenderer({
            "resolution": config.resolution,
            "near": config.near,
            "far": config.far,
            "ssaa": config.ssaa,
        })
        render_ret = renderer.render(
            positions, attrs, config.voxel_size,
            config.extrinsic, config.intrinsic
        )
        return render_ret["voxel_id"]
    
    def _flip_normals_to_camera(
        self,
        normals: Tensor,
        surface_pos: Tensor,
        extrinsic: Tensor,
    ) -> Tensor:
        """变换到相机空间 + 翻转"""
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        normals_cam = normals @ R.T
        pos_cam = surface_pos @ R.T + t
        dot = (normals_cam * pos_cam).sum(dim=-1, keepdim=True)
        return torch.where(dot > 0, normals_cam, -normals_cam)


# ============================================================================
# 便捷入口函数
# ============================================================================

def render_normal_12quad(
    h: Any,
    subs: List[Any],
    config: RenderConfig,
    voxel_margin: float = 0.0,
    use_checkpoint: bool = True,
    temperature: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """12-Quad 法线渲染入口
    
    Args:
        h: FDG SparseTensor，feats (N, 7)
        subs: 多分辨率 sub_logits
        config: 渲染配置
        voxel_margin: dual_vertices 边距
        use_checkpoint: 是否使用 gradient checkpointing
        temperature: 软与温度（越大越平滑）
    
    Returns:
        pixel_normal: (H, W, 3)
        mask: (H, W)
    """
    renderer = Quad12NormalRenderer(
        use_checkpoint=use_checkpoint,
        temperature=temperature,
    )
    return renderer.render(h, subs, config, voxel_margin)