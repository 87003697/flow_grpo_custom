"""
可微 Voxel Normal 渲染模块

核心思路：
- 硬渲染只决定"哪个 voxel 被击中"（voxel_id），索引操作对被索引的 tensor 是可微的
- pixel_normal = voxel_normals[voxel_id]  # voxel_id 无梯度，但 voxel_normals 有梯度

两种模式：
- FDG 模式：邻居 dual_vertices → axis_normals → intersected 加权
- Sub 模式：sub_logits → occupancy → 梯度场 → normal

设计原则：
- 每个主函数都是"端到端"的，包含渲染 + normal 计算
- 使用 o-voxel 原生 CUDA 哈希映射
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any
import torch
from torch import Tensor
import torch.nn.functional as F

# 复用 o-voxel 的 CUDA 哈希映射和渲染器
import o_voxel
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap


# ============ 公共配置 ============

@dataclass
class RenderConfig:
    """渲染配置"""
    intrinsics: Tensor      # (3, 3) 相机内参
    extrinsics: Tensor      # (4, 4) W2C 外参
    resolution: int         # 渲染分辨率
    voxel_size: float       # 体素尺寸
    origin: Tensor          # (3,) 网格原点
    grid_size: Tensor       # (3,) 网格尺寸，用于哈希映射


# ============ 邻居偏移常量 ============
# 来自 flexible_dual_grid.py，每个轴的 4 个邻居偏移
EDGE_NEIGHBOR_VOXEL_OFFSET = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # axis=0 (X): YZ 平面的 4 个邻居
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # axis=1 (Y): XZ 平面的 4 个邻居
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # axis=2 (Z): XY 平面的 4 个邻居
], dtype=torch.int)  # (3, 4, 3)


# ============ 硬渲染封装 ============

def hard_render(
    positions: Tensor,       # (N, 3) voxel 中心位置
    voxel_size: float,
    extrinsics: Tensor,      # (4, 4) W2C
    intrinsics: Tensor,      # (3, 3)
    resolution: int,
) -> Tuple[Tensor, Tensor]:
    """
    硬渲染获取 voxel_id
    
    Args:
        positions: (N, 3) voxel 中心位置（世界坐标）
        voxel_size: 体素尺寸
        extrinsics: (4, 4) W2C 相机外参
        intrinsics: (3, 3) 相机内参
        resolution: 渲染分辨率
    
    Returns:
        voxel_id: (H, W) 击中的 voxel 索引，-1 表示背景
        depth: (H, W) 深度图
    """
    renderer = o_voxel.rasterize.VoxelRenderer({
        "resolution": resolution,
        "near": 0.1,
        "far": 10.0,
        "ssaa": 1,  # 不使用 SSAA，避免 voxel_id 降采样问题
    })
    
    # 创建 dummy attrs（只需要 voxel_id）
    attrs = torch.ones(positions.shape[0], 1, device=positions.device, dtype=torch.float32)  # (N, 1)
    
    ret = renderer.render(positions, attrs, voxel_size, extrinsics, intrinsics)
    
    return ret['voxel_id'], ret['depth']  # (H, W), (H, W)


# ============ 邻居查找（使用 o-voxel 原生哈希） ============

def find_neighbor_indices(
    coords: Tensor,              # (N, 3) voxel 整数坐标
    neighbor_offsets: Tensor,    # (3, 4, 3) 每个轴的 4 个邻居偏移
    grid_size: Tensor,           # (3,) 网格尺寸
) -> Tuple[Tensor, Tensor]:
    """
    使用 o-voxel 原生 CUDA 哈希映射查找邻居索引
    
    参考代码 (flexible_dual_grid.py 第 225-236 行)
    
    Args:
        coords: (N, 3) voxel 整数坐标
        neighbor_offsets: (3, 4, 3) 每个轴的 4 个邻居偏移
        grid_size: (3,) 网格尺寸
    
    Returns:
        neighbor_idx: (N, 3, 4) 每个轴的 4 个邻居索引，无效为 -1
        axis_valid_mask: (N, 3) bool，每个轴的 4 个邻居是否都存在
    """
    N = coords.shape[0]
    device = coords.device
    
    # 确保 neighbor_offsets 在正确设备上
    neighbor_offsets = neighbor_offsets.to(device)
    grid_size = grid_size.to(device)
    
    # 构建哈希表
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())
    
    INVALID = 0xffffffff
    
    # 查找每个轴的邻居
    neighbor_idx_list = []
    axis_valid_list = []
    
    for axis in range(3):
        # 计算邻居坐标: (N, 4, 3)
        offsets = neighbor_offsets[axis]  # (4, 3)
        neighbor_coords = coords.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 4, 3)
        neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (N*4, 3)
        
        # 查询哈希表
        query = torch.cat([
            torch.zeros((N * 4, 1), dtype=torch.int, device=device),
            neighbor_coords_flat
        ], dim=-1)  # (N*4, 4)
        indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())  # (N*4,)
        indices = indices.reshape(N, 4)  # (N, 4)
        
        # 检查有效性（INVALID 表示不存在）
        valid = (indices != INVALID).all(dim=1)  # (N,)
        
        # 将无效索引替换为 0（后续会被 mask）
        # 注意：indices 是 uint32 类型，需要先转为 int64 再操作
        indices_int = indices.to(torch.int64)  # (N, 4)
        indices_safe = torch.where(indices_int == INVALID, torch.zeros_like(indices_int), indices_int)  # (N, 4)
        
        neighbor_idx_list.append(indices_safe.int())  # (N, 4)
        axis_valid_list.append(valid)  # (N,)
    
    neighbor_idx = torch.stack(neighbor_idx_list, dim=1)  # (N, 3, 4)
    axis_valid_mask = torch.stack(axis_valid_list, dim=1)  # (N, 3)
    
    return neighbor_idx, axis_valid_mask


# ============ FDG 模式内部函数 ============

def _compute_axis_face_normals(
    coords: Tensor,           # (N, 3) voxel 整数坐标
    dual_vertices: Tensor,    # (N, 3) 可微
    voxel_size: float,
    origin: Tensor,           # (3,)
    grid_size: Tensor,        # (3,)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    计算每个 voxel 的 3 个轴方向 face normal
    
    Args:
        coords: (N, 3) voxel 整数坐标
        dual_vertices: (N, 3) 顶点偏移（可微）
        voxel_size: 体素尺寸
        origin: (3,) 网格原点
        grid_size: (3,) 网格尺寸
    
    Returns:
        axis_normals: (N, 3, 3) 每个轴的 face normal
        axis_valid_mask: (N, 3) bool，每个轴的邻居是否完整
        surface_pos: (N, 3) 表面位置
    """
    device = coords.device
    origin = origin.to(device)
    
    # 计算表面位置: (N, 3)
    surface_pos = (coords.float() + dual_vertices) * voxel_size + origin
    
    # 获取邻居偏移
    neighbor_offsets = EDGE_NEIGHBOR_VOXEL_OFFSET.to(device)  # (3, 4, 3)
    
    # 查找邻居索引
    neighbor_idx, axis_valid_mask = find_neighbor_indices(
        coords, neighbor_offsets, grid_size
    )  # (N, 3, 4), (N, 3)
    
    # 计算每个轴的 face normal
    axis_normals = []
    for axis in range(3):
        # 获取邻居的 surface_pos: (N, 4, 3)
        idx = neighbor_idx[:, axis, :]  # (N, 4)
        neighbor_pos = surface_pos[idx.long()]  # (N, 4, 3)，可微！
        
        # 4 个顶点 → face normal (v0, v1, v2, v3 按顺序)
        v0 = neighbor_pos[:, 0, :]  # (N, 3)
        v1 = neighbor_pos[:, 1, :]  # (N, 3)
        v2 = neighbor_pos[:, 2, :]  # (N, 3)
        v3 = neighbor_pos[:, 3, :]  # (N, 3)
        
        # cross(v1 - v0, v3 - v0) 计算法线
        axis_normal = F.normalize(torch.cross(v1 - v0, v3 - v0, dim=-1), dim=-1, eps=1e-6)  # (N, 3)
        axis_normals.append(axis_normal)
    
    axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)
    
    return axis_normals, axis_valid_mask, surface_pos


# ============ Sub 模式内部函数 ============

def _compute_occupancy_gradient(sub_logits: Tensor) -> Tensor:
    """
    计算 occupancy 梯度作为法线方向
    
    子 voxel 索引布局 (2x2x2):
        z=0 面          z=1 面
       ┌───┬───┐      ┌───┬───┐
       │ 2 │ 3 │      │ 6 │ 7 │
       ├───┼───┤      ├───┼───┤
       │ 0 │ 1 │      │ 4 │ 5 │
       └───┴───┘      └───┴───┘
    
    Args:
        sub_logits: (N, 8) 子 voxel 的 occupancy logits
    
    Returns:
        voxel_normals: (N, 3) World Space，已归一化
    """
    occupancy = torch.sigmoid(sub_logits)  # (N, 8)
    
    # x 方向梯度：右边 - 左边 (索引 1,3,5,7 vs 0,2,4,6)
    grad_x = (occupancy[:, [1, 3, 5, 7]] - occupancy[:, [0, 2, 4, 6]]).mean(dim=1)  # (N,)
    
    # y 方向梯度：上边 - 下边 (索引 2,3,6,7 vs 0,1,4,5)
    grad_y = (occupancy[:, [2, 3, 6, 7]] - occupancy[:, [0, 1, 4, 5]]).mean(dim=1)  # (N,)
    
    # z 方向梯度：前边 - 后边 (索引 4,5,6,7 vs 0,1,2,3)
    grad_z = (occupancy[:, [4, 5, 6, 7]] - occupancy[:, [0, 1, 2, 3]]).mean(dim=1)  # (N,)
    
    # 梯度向量
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (N, 3)
    
    # 法线 = 梯度反方向（从内部指向外部）
    voxel_normals = -F.normalize(gradient, dim=-1, eps=1e-6)  # (N, 3)
    
    return voxel_normals


# ============ 公共变换函数 ============

def _flip_normals_to_camera(
    voxel_normals: Tensor,    # (N, 3) World Space
    surface_pos: Tensor,      # (N, 3) World Space
    extrinsics: Tensor,       # (4, 4) W2C
) -> Tensor:
    """
    变换到 Camera Space + 用点积翻转
    
    在 Camera Space 中，surface_pos_cam 是从相机原点指向表面的向量。
    如果 normal · pos > 0，说明法线和视线方向同向（指向远离相机），需要翻转。
    
    Args:
        voxel_normals: (N, 3) World Space 法线
        surface_pos: (N, 3) World Space 位置
        extrinsics: (4, 4) W2C 外参
    
    Returns:
        voxel_normals_cam: (N, 3) Camera Space，朝向相机
    """
    R = extrinsics[:3, :3]  # (3, 3) 旋转矩阵
    t = extrinsics[:3, 3]   # (3,) 平移向量
    
    # 变换到 Camera Space
    voxel_normals_cam = voxel_normals @ R.T  # (N, 3)
    surface_pos_cam = surface_pos @ R.T + t  # (N, 3)
    
    # 用点积判断翻转：确保法线朝向相机
    dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N, 1)
    voxel_normals_cam = torch.where(dot_product > 0, -voxel_normals_cam, voxel_normals_cam)  # (N, 3)
    
    return voxel_normals_cam


# ============ FDG 模式主函数 ============

def render_normal_fdg(
    coords: Tensor,                # (N, 3) voxel 整数坐标
    dual_vertices: Tensor,         # (N, 3) 可微
    intersected_logits: Tensor,    # (N, 3) 可微
    config: RenderConfig,
) -> Tuple[Tensor, Tensor]:
    """
    FDG 模式：渲染 + 计算可微 normal
    
    梯度流：
        dual_vertices → surface_pos → axis_normals ─┐
                                                    ├→ voxel_normals → pixel_normal → loss
        intersected_logits → sigmoid → weights ─────┘
    
    Args:
        coords: (N, 3) voxel 整数坐标
        dual_vertices: (N, 3) 顶点偏移（可微）
        intersected_logits: (N, 3) 边相交 logits（可微）
        config: RenderConfig 渲染配置
    
    Returns:
        normal: (H, W, 3) 归一化法向量，Camera Space，朝向相机
        mask: (H, W) bool，True = 前景
    """
    device = coords.device
    
    # 1. 计算 axis_normals + surface_pos
    axis_normals, axis_valid_mask, surface_pos = _compute_axis_face_normals(
        coords, dual_vertices, config.voxel_size, config.origin, config.grid_size
    )  # (N, 3, 3), (N, 3), (N, 3)
    
    # 2. intersected 加权（邻居缺失的轴权重强制为 0）
    weights = torch.sigmoid(intersected_logits)  # (N, 3)
    effective_weights = weights * axis_valid_mask.float()  # (N, 3)
    
    # 加权求和 + 归一化
    weighted = (effective_weights.unsqueeze(-1) * axis_normals).sum(dim=1)  # (N, 3)
    voxel_normals = F.normalize(weighted, dim=-1, eps=1e-6)  # (N, 3)
    
    # 3. 硬渲染 → voxel_id
    voxel_id, _ = hard_render(
        surface_pos, config.voxel_size,
        config.extrinsics, config.intrinsics, config.resolution
    )  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    
    # 4. 变换到 Camera Space + 翻转
    voxel_normals_cam = _flip_normals_to_camera(
        voxel_normals, surface_pos, config.extrinsics
    )  # (N, 3)
    
    # 5. 索引 → pixel_normal（可微！）
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0).long()]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1).float()  # (H, W, 3)
    
    return pixel_normal, mask


# ============ Sub 模式主函数 ============

def render_normal_sub(
    sub_coords: Tensor,            # (N, 3) 父 voxel 整数坐标
    sub_logits: Tensor,            # (N, 8) 子 voxel logits（可微）
    config: RenderConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    单层 Sub 模式：渲染 + 计算可微 normal
    
    Args:
        sub_coords: (N, 3) 父 voxel 整数坐标
        sub_logits: (N, 8) 子 voxel occupancy logits（可微）
        config: RenderConfig 渲染配置
        target_size: 可选，resize 目标分辨率 (H, W)
    
    Returns:
        normal: (H, W, 3) 或 (target_H, target_W, 3) Camera Space
        mask: (H, W) 或 (target_H, target_W) bool
    """
    device = sub_coords.device
    
    # 1. 计算 occupancy 梯度
    voxel_normals = _compute_occupancy_gradient(sub_logits)  # (N, 3)
    
    # 2. 计算 surface_pos（父 voxel 中心）
    origin = config.origin.to(device)
    surface_pos = (sub_coords.float() + 0.5) * config.voxel_size + origin  # (N, 3)
    
    # 3. 硬渲染 → voxel_id
    voxel_id, _ = hard_render(
        surface_pos, config.voxel_size,
        config.extrinsics, config.intrinsics, config.resolution
    )  # (H, W)
    mask = voxel_id >= 0  # (H, W)
    
    # 4. 变换到 Camera Space + 翻转
    voxel_normals_cam = _flip_normals_to_camera(
        voxel_normals, surface_pos, config.extrinsics
    )  # (N, 3)
    
    # 5. 索引 → pixel_normal
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0).long()]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1).float()  # (H, W, 3)
    
    # 6. (可选) resize + 重新归一化
    if target_size is not None:
        # normal 用双线性插值，但需要重新归一化
        pixel_normal = F.interpolate(
            pixel_normal.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # (target_H, target_W, 3)
        # 双线性插值会改变向量长度，需要重新归一化
        pixel_normal = F.normalize(pixel_normal, dim=-1, eps=1e-6)
        
        # mask 用最近邻插值
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=target_size,
            mode='nearest'
        ).squeeze() > 0.5  # (target_H, target_W)
    
    return pixel_normal, mask


def render_normal_sub_multi(
    subs: List[Any],               # List[SparseTensor]，每层 feats: (N_i, 8), coords: (N_i, 4)
    configs: List[RenderConfig],   # 每层配置（voxel_size 不同）
    target_size: Tuple[int, int],
) -> List[Tuple[Tensor, Tensor]]:
    """
    多分辨率 Sub 模式
    
    Args:
        subs: List[SparseTensor]，每层的 sub 信息
        configs: List[RenderConfig]，每层的渲染配置
        target_size: (H, W) resize 目标分辨率
    
    Returns:
        List of (normal, mask)，每层已 resize 到 target_size
    """
    results = []
    for sub, config in zip(subs, configs):
        # 从 SparseTensor 提取 coords 和 feats
        sub_coords = sub.coords[:, 1:]  # (N_i, 3) 去掉 batch_idx
        sub_logits = sub.feats  # (N_i, 8)
        
        normal, mask = render_normal_sub(sub_coords, sub_logits, config, target_size)
        results.append((normal, mask))
    
    return results


# ============ 便捷函数 ============

def normal_to_rgb(normal: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    将 Camera Space normal 转换为 RGB 可视化格式
    
    Args:
        normal: (H, W, 3) Camera Space normal，朝向相机
        mask: (H, W) 可选，前景 mask
    
    Returns:
        rgb: (H, W, 3) RGB 格式，范围 [0, 1]
    """
    # Camera Space normal → RGB: [-1, 1] → [0, 1]
    # 注意：朝向相机的 normal 的 Z 分量为负，所以取反后 Z > 0
    rgb = (-normal * 0.5 + 0.5)  # (H, W, 3)
    
    if mask is not None:
        # 中性背景（朝向相机）：RGB = [0.5, 0.5, 1.0]
        bg_color = torch.tensor([0.5, 0.5, 1.0], device=normal.device)
        rgb = rgb * mask.unsqueeze(-1).float() + bg_color * (~mask).unsqueeze(-1).float()
    
    return rgb
