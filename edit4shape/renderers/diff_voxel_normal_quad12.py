"""
可微 Voxel Normal 渲染模块 — 12-Quad 版本

重构版：继承 BaseRenderer，遵循 7 阶段渲染流水线。

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
from typing import Tuple, List, Optional, Any, Dict

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap

from edit4shape.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
)


# ============================================================================
# 常量定义
# ============================================================================

# 12 条边对应的 sub_logit corner 索引对
EDGE_CORNER_PAIRS = torch.tensor([
    [0, 1], [2, 3], [4, 5], [6, 7],  # X 轴边
    [0, 2], [1, 3], [4, 6], [5, 7],  # Y 轴边
    [0, 4], [1, 5], [2, 6], [3, 7],  # Z 轴边
], dtype=torch.long)  # (12, 2)

EDGE_TO_AXIS = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2], dtype=torch.long)  # (12,)

EPS = 1e-6

EDGE_NEIGHBOR_OFFSETS = torch.tensor([
    [[0, -1, 0], [0, 0, -1], [0, -1, -1]],
    [[0, 1, 0], [0, 0, -1], [0, 1, -1]],
    [[0, -1, 0], [0, 0, 1], [0, -1, 1]],
    [[0, 1, 0], [0, 0, 1], [0, 1, 1]],
    [[-1, 0, 0], [0, 0, -1], [-1, 0, -1]],
    [[1, 0, 0], [0, 0, -1], [1, 0, -1]],
    [[-1, 0, 0], [0, 0, 1], [-1, 0, 1]],
    [[1, 0, 0], [0, 0, 1], [1, 0, 1]],
    [[-1, 0, 0], [0, -1, 0], [-1, -1, 0]],
    [[1, 0, 0], [0, -1, 0], [1, -1, 0]],
    [[-1, 0, 0], [0, 1, 0], [-1, 1, 0]],
    [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
], dtype=torch.int)  # (12, 3, 3)


# ============================================================================
# Quad12RenderConfig - 兼容旧接口的配置
# ============================================================================

@dataclass
class Quad12RenderConfig:
    """
    12-Quad 渲染专用配置
    
    兼容旧的 RenderConfig 接口
    """
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
# 多分辨率 Crossing Weight
# ============================================================================

def _get_logit_at_level(
    sub: Any,
    coords: Tensor,
    voxel_resolution: int,
    level_resolution: int,
    default_logit: float = 0.0,
) -> Tensor:
    """查找坐标在某层的 logit"""
    K = coords.shape[0]
    device = coords.device
    M = sub.feats.shape[0]
    INVALID = 0xffffffff
    
    coords_in_bounds = (
        (coords >= 0).all(dim=-1) & 
        (coords < voxel_resolution).all(dim=-1)
    )
    
    scale = voxel_resolution // level_resolution
    coords_safe = coords.clamp(min=0, max=voxel_resolution - 1)
    parent_coords = coords_safe // scale
    
    grid_size = torch.tensor([level_resolution] * 3, device=device)
    hashmap = _init_hashmap(grid_size, 2 * M + 1, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap, sub.coords.int(), *grid_size.tolist()
    )
    
    query = torch.cat([
        torch.zeros((K, 1), dtype=torch.int, device=device),
        parent_coords.int()
    ], dim=-1)
    indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())
    
    valid = coords_in_bounds & (indices != INVALID)
    indices = indices.long().clamp(min=0, max=M - 1)
    
    child_scale = max(scale // 2, 1)
    child_coords = coords_safe // child_scale
    corner_idx = (
        (child_coords[:, 0] % 2) +
        (child_coords[:, 1] % 2) * 2 +
        (child_coords[:, 2] % 2) * 4
    ).long()
    
    sub_feats = sub.feats.float()
    logit = sub_feats[indices, corner_idx]
    logit = torch.where(valid, logit, torch.full_like(logit, default_logit))
    
    return logit


def _get_multi_level_logit_sum(
    subs: List[Any],
    coords: Tensor,
    voxel_resolution: int,
) -> Tensor:
    """获取坐标在所有分辨率的 logit 之和"""
    K = coords.shape[0]
    device = coords.device
    logit_sum = torch.zeros(K, device=device)
    
    for level, sub in enumerate(subs):
        level_resolution = voxel_resolution // (2 ** (len(subs) - level))
        level_logit = _get_logit_at_level(
            sub, coords, voxel_resolution, level_resolution, default_logit=0.0
        )
        logit_sum = logit_sum + level_logit
    
    return logit_sum


def compute_crossing_weight_soft_and(
    subs: List[Any],
    center_coords: Tensor,
    neighbor_coords: Tensor,
    voxel_resolution: int,
    temperature: float = 2.0,
    use_checkpoint: bool = True,
) -> Tensor:
    """多分辨率 crossing weight"""
    N = center_coords.shape[0]
    
    if use_checkpoint:
        center_sum = checkpoint(
            _get_multi_level_logit_sum,
            subs, center_coords, voxel_resolution,
            use_reentrant=False,
        )
    else:
        center_sum = _get_multi_level_logit_sum(subs, center_coords, voxel_resolution)
    
    neighbor_flat = neighbor_coords.reshape(-1, 3)
    
    if use_checkpoint:
        neighbor_sum_flat = checkpoint(
            _get_multi_level_logit_sum,
            subs, neighbor_flat, voxel_resolution,
            use_reentrant=False,
        )
    else:
        neighbor_sum_flat = _get_multi_level_logit_sum(subs, neighbor_flat, voxel_resolution)
    
    neighbor_sum = neighbor_sum_flat.reshape(N, 12)
    
    diff = (center_sum.unsqueeze(-1) - neighbor_sum) / temperature
    crossing = torch.tanh(diff) ** 2
    
    return crossing


# ============================================================================
# 12-Quad 法线计算
# ============================================================================

def _find_quad_neighbors(
    query_coords: Tensor,
    all_coords: Tensor,
    grid_size: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """查找每条边的 3 个邻居"""
    M = query_coords.shape[0]
    N = all_coords.shape[0]
    device = query_coords.device
    INVALID = 0xffffffff
    
    edge_offsets = EDGE_NEIGHBOR_OFFSETS.to(device)
    
    hashmap = _init_hashmap(grid_size, 2 * N + 1, device)
    coords_with_batch = torch.cat([
        torch.zeros_like(all_coords[:, :1]),
        all_coords
    ], dim=-1)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())
    
    neighbor_coords = query_coords.unsqueeze(1).unsqueeze(2) + edge_offsets
    neighbor_coords_flat = neighbor_coords.reshape(-1, 3)
    
    query = torch.cat([
        torch.zeros((M * 12 * 3, 1), dtype=torch.int, device=device),
        neighbor_coords_flat.int()
    ], dim=-1)
    
    indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())
    indices = indices.reshape(M, 12, 3)
    
    valid = (indices != INVALID)
    indices = indices.long().clamp(min=0, max=N - 1)
    
    edge_neighbor_coords = neighbor_coords[:, :, 2, :]
    
    return indices, valid, neighbor_coords, edge_neighbor_coords


def _compute_quad_normals(
    surface_pos: Tensor,
    neighbor_idx: Tensor,
    neighbor_valid: Tensor,
    neighbor_coords: Tensor,
    all_surface_pos: Tensor,
    voxel_size: float,
    origin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """计算 12 个 quad 的法线"""
    N = surface_pos.shape[0]
    device = surface_pos.device
    
    neighbor_pos = all_surface_pos[neighbor_idx]
    air_pos = (neighbor_coords.float() + 0.5) * voxel_size + origin
    
    neighbor_pos = torch.where(
        neighbor_valid.unsqueeze(-1),
        neighbor_pos,
        air_pos
    )
    
    v_center = surface_pos.unsqueeze(1)
    v_face1 = neighbor_pos[:, :, 0, :]
    v_face2 = neighbor_pos[:, :, 1, :]
    
    e1 = v_face1 - v_center
    e2 = v_face2 - v_center
    cross_result = torch.cross(e1, e2, dim=-1)
    cross_norm = cross_result.norm(dim=-1, keepdim=True)
    
    quad_normals = cross_result / cross_norm.clamp(min=EPS)
    
    cross_valid = (cross_norm.squeeze(-1) > EPS)
    face_valid = neighbor_valid[:, :, :2].all(dim=-1)
    quad_valid = cross_valid & face_valid
    
    return quad_normals, quad_valid


# ============================================================================
# 一致性翻转加权
# ============================================================================

def consistent_weighted_normal(
    normals: Tensor,
    weights: Tensor,
    eps: float = EPS,
) -> Tensor:
    """一致性翻转 + 加权求和"""
    N, K, _ = normals.shape
    device = normals.device
    
    weights = weights + eps
    max_idx = weights.argmax(dim=-1)
    ref = normals[torch.arange(N, device=device), max_idx]
    
    dots = (normals * ref.unsqueeze(1)).sum(dim=-1)
    flip = (dots < 0).unsqueeze(-1)
    aligned = torch.where(flip, -normals, normals)
    
    weighted = (weights.unsqueeze(-1) * aligned).sum(dim=1)
    weighted_norm = weighted.norm(dim=-1, keepdim=True)
    
    result = torch.where(
        weighted_norm > eps,
        weighted / weighted_norm.clamp(min=eps),
        ref
    )
    
    return result


def aggregate_to_final_normal(
    quad_normals: Tensor,
    quad_valid: Tensor,
    crossing_weights: Tensor,
    intersected_logits: Tensor,
) -> Tensor:
    """两级加权聚合"""
    N = quad_normals.shape[0]
    device = quad_normals.device
    
    edge_to_axis = EDGE_TO_AXIS.to(device)
    combined_weights = crossing_weights * quad_valid.float()
    
    axis_normals = []
    for axis in range(3):
        edge_mask = (edge_to_axis == axis)
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]
        
        normals_ax = quad_normals[:, edge_ids]
        weights_ax = combined_weights[:, edge_ids]
        
        axis_normal = consistent_weighted_normal(normals_ax, weights_ax)
        axis_normals.append(axis_normal)
    
    axis_normals = torch.stack(axis_normals, dim=1)
    axis_weights = torch.sigmoid(intersected_logits)
    
    final_normal = consistent_weighted_normal(axis_normals, axis_weights)
    
    return final_normal


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Quad12GeometryData:
    """
    12-Quad 几何数据
    """
    coords: Tensor                # (N, 3) 全部坐标
    dual_vertices: Tensor         # (N, 3) dual vertices
    intersected_logits: Tensor    # (N, 3) 交叉 logits
    surface_pos: Tensor           # (N, 3) 表面位置
    voxel_size: float
    origin: Tensor                # (3,)
    grid_size: Tensor             # (3,)
    subs: List[Any]               # 多分辨率 SparseTensor


@dataclass 
class Quad12RasterResult:
    """
    12-Quad 光栅化结果
    """
    voxel_id: Tensor      # (H, W) voxel ID
    mask: Tensor          # (H, W) bool
    visible_ids: Tensor   # (M,) 可见 voxel 索引


# ============================================================================
# Quad12NormalRenderer
# ============================================================================

class Quad12NormalRenderer(BaseRenderer):
    """
    12-Quad 法线渲染器
    
    继承 BaseRenderer，实现 7 阶段渲染流水线:
        Stage 1: prepare_inputs - 准备输入
        Stage 2: compute_camera_data - 计算相机参数
        Stage 3: process_geometry - 提取 FDG 数据
        Stage 4: rasterize_core - 硬渲染获取可见体素
        Stage 5: interpolate_attributes - 计算 12-quad 法线
        Stage 6: post_process - 采样到像素
        Stage 7: assemble_output - 组装 RenderOutput
    
    特殊点:
        - 接受 SparseTensor 作为几何输入
        - 需要额外的 subs 参数（多分辨率 SparseTensor）
    
    使用示例:
        config = RenderConfig(resolution=256, near=1.0, far=100.0)
        renderer = Quad12NormalRenderer(config, temperature=2.0)
        output = renderer.render_quad12(h, subs, extrinsics, intrinsics)
    """
    
    def __init__(
        self,
        config: RenderConfig = None,
        device: str = 'cuda',
        use_checkpoint: bool = True,
        temperature: float = 2.0,
    ):
        """
        Args:
            config: 渲染配置
            device: 计算设备
            use_checkpoint: 是否使用 gradient checkpointing
            temperature: 软与温度参数
        """
        if config is None:
            config = RenderConfig(resolution=256, near=1.0, far=100.0, ssaa=1)
        super().__init__(config, device)
        
        self.use_checkpoint = use_checkpoint
        self.temperature = temperature
        
        # 存储额外数据
        self._subs: Optional[List[Any]] = None
        self._voxel_margin: float = 0.0
    
    # ========== Stage 3: Geometry Processing ==========
    
    def _process_geometry(
        self,
        geometry: Any,  # SparseTensor
        camera_data: CameraData,
    ) -> Quad12GeometryData:
        """
        提取 FDG SparseTensor 数据
        """
        resolution = self.config.resolution
        voxel_size = 1.0 / resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)
        grid_size = torch.tensor([resolution, resolution, resolution], device=self.device)
        
        # 提取数据
        coords = geometry.coords[:, 1:].int()  # (N, 3)
        raw_vertices = geometry.feats[..., 0:3]
        dual_vertices = (1 + 2 * self._voxel_margin) * torch.sigmoid(raw_vertices) - self._voxel_margin
        intersected_logits = geometry.feats[..., 3:6]
        
        # 表面位置
        surface_pos = (coords.float() + dual_vertices) * voxel_size + origin
        
        return Quad12GeometryData(
            coords=coords,
            dual_vertices=dual_vertices,
            intersected_logits=intersected_logits,
            surface_pos=surface_pos,
            voxel_size=voxel_size,
            origin=origin,
            grid_size=grid_size,
            subs=self._subs,
        )
    
    # ========== Stage 4: Rasterization Core ==========
    
    def _rasterize_core(
        self,
        processed_geometry: Quad12GeometryData,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        硬渲染获取可见体素
        """
        import o_voxel
        
        coords = processed_geometry.coords
        voxel_size = processed_geometry.voxel_size
        origin = processed_geometry.origin
        
        # 体素中心位置
        positions = (coords.float() + 0.5) * voxel_size + origin  # (N, 3)
        attrs = torch.ones((coords.shape[0], 1), device=self.device, dtype=positions.dtype)
        
        renderer = o_voxel.rasterize.VoxelRenderer({
            "resolution": self.config.resolution,
            "near": self.config.near,
            "far": self.config.far,
            "ssaa": self.config.ssaa,
        })
        
        render_ret = renderer.render(
            positions, attrs, voxel_size,
            camera_data.extrinsics, camera_data.intrinsics
        )
        
        voxel_id = render_ret["voxel_id"]  # (H, W)
        mask = voxel_id >= 0  # (H, W)
        visible_ids = voxel_id[mask].unique()  # (M,)
        
        return RasterOutput(
            rast=Quad12RasterResult(
                voxel_id=voxel_id,
                mask=mask,
                visible_ids=visible_ids,
            ),
            depth_buffer=torch.zeros_like(voxel_id, dtype=torch.float32),
            primitive_id=voxel_id.long(),
        )
    
    # ========== Stage 5: Attribute Interpolation ==========
    
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Quad12GeometryData,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        计算 12-quad 法线
        """
        rast: Quad12RasterResult = raster_output.rast
        
        H, W = rast.voxel_id.shape
        
        if rast.visible_ids.numel() == 0:
            return {
                'normal': torch.zeros(H, W, 3, device=self.device),
                'mask': rast.mask.float(),
                'alpha': rast.mask.float(),
                'depth': torch.zeros(H, W, device=self.device),
            }
        
        # 提取可见部分数据
        visible_ids = rast.visible_ids
        coords_vis = geometry.coords[visible_ids]
        dual_vertices_vis = geometry.dual_vertices[visible_ids]
        intersected_logits_vis = geometry.intersected_logits[visible_ids]
        surface_pos_vis = geometry.surface_pos[visible_ids]
        
        # 查找邻居
        neighbor_idx, neighbor_valid, neighbor_coords, edge_neighbor_coords = \
            _find_quad_neighbors(coords_vis, geometry.coords, geometry.grid_size)
        
        # 多分辨率 crossing weight
        crossing_weights = compute_crossing_weight_soft_and(
            geometry.subs, coords_vis, edge_neighbor_coords, self.config.resolution,
            temperature=self.temperature,
            use_checkpoint=self.use_checkpoint,
        )
        
        # 12-Quad 法线
        quad_normals, quad_valid = _compute_quad_normals(
            surface_pos_vis, neighbor_idx, neighbor_valid, neighbor_coords,
            geometry.surface_pos, geometry.voxel_size, geometry.origin
        )
        
        # 两级聚合
        voxel_normals = aggregate_to_final_normal(
            quad_normals, quad_valid, crossing_weights, intersected_logits_vis
        )
        
        # 翻转到相机空间
        voxel_normals_cam = self._flip_normals_to_camera(
            voxel_normals, surface_pos_vis, camera_data.extrinsics
        )
        
        # 采样到像素
        pixel_normal = self._sample_to_pixels(
            voxel_normals_cam, visible_ids, rast.voxel_id, rast.mask, geometry.coords.shape[0]
        )
        
        return {
            'normal': pixel_normal,
            'mask': rast.mask.float(),
            'alpha': rast.mask.float(),
            'depth': torch.zeros(H, W, device=self.device),
        }
    
    # ========== 工具方法 ==========
    
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
    
    def _sample_to_pixels(
        self,
        voxel_normals: Tensor,
        visible_ids: Tensor,
        voxel_id: Tensor,
        mask: Tensor,
        num_voxels: int,
    ) -> Tensor:
        """将 voxel 法线采样到像素"""
        device = voxel_normals.device
        
        id_map = torch.zeros(num_voxels, dtype=torch.long, device=device)
        id_map[visible_ids] = torch.arange(len(visible_ids), device=device)
        
        voxel_id_mapped = id_map[voxel_id.clamp(min=0)]
        pixel_normal = voxel_normals[voxel_id_mapped]
        pixel_normal = pixel_normal * mask.unsqueeze(-1)
        
        return pixel_normal
    
    # ========== 专用渲染入口 ==========
    
    def render_quad12(
        self,
        h: Any,  # SparseTensor
        subs: List[Any],
        extrinsics: Tensor,
        intrinsics: Tensor,
        voxel_margin: float = 0.0,
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        12-Quad 法线渲染专用入口
        
        Args:
            h: FDG SparseTensor，feats (N, 7)
            subs: 多分辨率 sub_logits
            extrinsics: (4, 4) 相机外参
            intrinsics: (3, 3) 相机内参
            voxel_margin: dual_vertices 边距
            return_types: 返回类型列表
        
        Returns:
            RenderOutput: 包含 normal, mask, alpha
        """
        self._subs = subs
        self._voxel_margin = voxel_margin
        
        result = self.render(h, extrinsics, intrinsics, return_types)
        
        self._subs = None
        self._voxel_margin = 0.0
        
        return result
    
    # ========== 兼容旧接口 ==========
    
    def render_legacy(
        self,
        h: Any,
        subs: List[Any],
        config: Quad12RenderConfig,
        voxel_margin: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        兼容旧 API
        
        Returns:
            pixel_normal: (H, W, 3)
            mask: (H, W)
        """
        output = self.render_quad12(
            h, subs,
            config.extrinsic,
            config.intrinsic,
            voxel_margin,
        )
        return output.normal, output.mask


# ============================================================================
# 便捷入口函数
# ============================================================================

def render_normal_12quad(
    h: Any,
    subs: List[Any],
    config: Quad12RenderConfig,
    voxel_margin: float = 0.0,
    use_checkpoint: bool = True,
    temperature: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """
    12-Quad 法线渲染入口（兼容旧 API）
    
    Args:
        h: FDG SparseTensor，feats (N, 7)
        subs: 多分辨率 sub_logits
        config: 渲染配置
        voxel_margin: dual_vertices 边距
        use_checkpoint: 是否使用 gradient checkpointing
        temperature: 软与温度
    
    Returns:
        pixel_normal: (H, W, 3)
        mask: (H, W)
    """
    render_config = RenderConfig(
        resolution=config.resolution,
        near=config.near,
        far=config.far,
        ssaa=config.ssaa,
    )
    
    renderer = Quad12NormalRenderer(
        config=render_config,
        device=str(config.extrinsic.device),
        use_checkpoint=use_checkpoint,
        temperature=temperature,
    )
    
    output = renderer.render_quad12(
        h, subs,
        config.extrinsic,
        config.intrinsic,
        voxel_margin,
    )
    
    return output.normal, output.mask


# 保留旧的 RenderConfig 别名以兼容
RenderConfig_Quad12 = Quad12RenderConfig
