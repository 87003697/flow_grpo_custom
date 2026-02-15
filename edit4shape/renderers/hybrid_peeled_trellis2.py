# hybrid_peeled_trellis2.py

"""
混合 Normal 渲染器（Depth Peeling 版）：
  Voxel Normal (subs 可微) + Depth Peeling (intersect_logits 可微)

基于 hybrid_trellis2.py 扩展，使用 nvdiffrast DepthPeeler 进行多层渲染
并通过 per-face alpha（来自 sigmoid(intersect_logits)）进行 front-to-back
alpha 合成，使得 intersect_logits 参与梯度回传。

梯度路径:
  路径 1 (同原版): subs → occupancy_diff → voxel_normal → grid_sample_3d → pixel_normal
  路径 2 (新增):   intersect_logits → gather → sigmoid → alpha compositing → pixel_normal

调用栈:
  系统层 (decode_and_render_normal_peeled):
  ├── flexible_dual_grid_to_mesh(...)
  └── renderer.render(mesh, subs, coords, intersect_logits, extrinsics, intrinsics, ...)

  渲染器层 (render):
  ├── Phase 0: 预计算（@torch.no_grad）
  │     ├── recover_face_axis_and_voxel → face_axis_ids, face_voxel_ids
  │     └── VoxelRenderer → visible_ids → dilate → active_voxel_ids
  ├── Phase 1: _transform_vertices(...)           → vertices_clip, vertices_cam
  ├── Phase 2: _prepare_voxel_normals(...)        → voxel_normals
  ├── Phase 3: _depth_peel_render(...)            → out_dict ★ 入口编排器
  │     ├── _peel_all_chunks(...)                → all_depths/alphas/normals
  │     │     └── for chunk in split(faces, _MAX_FACES_PER_CHUNK):
  │     │           └── DepthPeeler(faces_chunk)
  │     │                 └── for layer in peel_layers:
  │     │                       ├── peeler.rasterize_next_layer()
  │     │                       └── checkpoint(_compute_one_chunk_layer)
  │     │                             ├── face_id + offset → sigmoid → alpha  (可微)
  │     │                             └── _grid_sample_normal_raw()           (可微)
  │     ├── _sort_and_composite(...)             → normal_acc, alpha_acc
  │     │     ├── argsort (per-pixel 深度排序)
  │     │     └── gather + front-to-back compositing
  │     ├── _merge_first_layer_depth(...)        → depth_first
  │     └── _assemble_output(...)                → out_dict
  └── Phase 4: _downsample(out_dict)              → final

使用方法:
    renderer = HybridPeeled26NormalRenderer({"resolution": 512, "peel_layers": 8})
    outputs = renderer.render(
        mesh, subs, coords, intersect_logits,
        extrinsics, intrinsics, voxel_resolution)
    normal = outputs.normal  # (H, W, 3)
"""

import logging
from typing import List, Any, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from flex_gemm.ops.grid_sample import grid_sample_3d
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap
import o_voxel.rasterize
from edit4shape.systems.utils.profiled_chunk import ProfiledScheduler


# =============================================================================
# 常量
# =============================================================================

# ProfiledScheduler probe 大小（仅 Normal 计算使用，渲染改用 DepthPeeler）
_NORMAL_PROBE_SIZE = 2000

# DepthPeeler 单次最大面片数。nvdiffrast 内部限制 2^24 ≈ 16.7M，
# 这里取 4M 留足安全余量。面片数超过此值时自动分 chunk 并做 per-pixel 深度归并。
_MAX_FACES_PER_CHUNK = 4_000_000


# =============================================================================
# 辅助函数（模块级）
# =============================================================================


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


def _grid_sample_normal_raw(
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
    """grid_sample_3d 渲染 raw normal（用于 depth peeling alpha 合成）

    与 _grid_sample_normal 的区别：
    - 返回相机空间 normal，范围 [-1, 1]，不做 (n+1)/2 映射
    - 背景像素 normal = 0（由 mask 清零）
    - 适合在 alpha 合成后再统一映射到 [0, 1]

    Args:
        voxel_normals: (N, 3) 每个 voxel 的法向量（相机空间）
        coords: (N, 3) voxel 坐标（整数）
        origin: (3,) voxel 网格原点
        voxel_size: voxel 尺寸
        voxel_resolution: voxel 分辨率
        vertices_batch: (1, V, 3) mesh 顶点（世界坐标）
        rast: (1, H, H, 4) 光栅化结果
        faces: (F, 3) 面索引
        rast_res: 光栅化分辨率

    Returns:
        img: (1, rast_res, rast_res, 3) raw normal [-1, 1]，背景为 0
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
    )  # (1, H*H, 3)
    del xyz_voxel

    img = pixel_normal.reshape(1, rast_res, rast_res, 3) * mask  # (1, H, H, 3)
    img = F.normalize(img, dim=-1, eps=1e-6)  # (1, H, H, 3) 归一化但不映射
    return img


@torch.no_grad()
def recover_face_axis_and_voxel(
    faces: Tensor,          # (F, 3) int — 三角形顶点索引
    coords: Tensor,         # (N, 3) int — voxel 坐标
    voxel_resolution: int,
) -> Tuple[Tensor, Tensor]:
    """从 mesh 输出反推 per-face 的 axis_id 和 source voxel_id。

    利用 Dual Contouring 的几何性质（无需修改 flexible_dual_grid_to_mesh）：
    - axis: 三角形 3 个顶点在哪个维度坐标相同 → 该维度就是 axis
    - source voxel: 3 个顶点逐维取 min = quad vertex 0 的坐标
      （数学保证：任取 4 个邻居中的 3 个，逐维 min 仍等于源 voxel 坐标）

    仅适用于 train=False（vertex index == voxel index）。

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

    # ---- Step 1: axis 判定（恒定维度） ----
    face_v_coords = coords[faces]                                   # (F, 3, 3)
    face_range = (face_v_coords.max(dim=1).values
                  - face_v_coords.min(dim=1).values)                # (F, 3)
    face_axis_ids = (face_range == 0).long().argmax(dim=1)          # (F,)

    # ---- Step 2: 源 voxel 坐标 = 逐维 min ----
    source_coords = face_v_coords.min(dim=1).values.int()           # (F, 3)

    # ---- Step 3: GPU hashmap 查找 source_coords → voxel index ----
    grid_size = torch.tensor([voxel_resolution] * 3, device=device)
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        *grid_size.tolist(),
    )
    source_key = torch.cat([
        torch.zeros(F_count, 1, dtype=torch.int, device=device),
        source_coords,
    ], dim=-1)                                                       # (F, 4)
    face_voxel_ids = _C.hashmap_lookup_3d_cuda(
        *hashmap, source_key, *grid_size.tolist()
    ).long()                                                         # (F,)

    return face_axis_ids, face_voxel_ids


def _compute_one_chunk_layer(
    rast: Tensor,                # (1, H, W, 4) 当前层光栅化结果（face_id 相对 chunk 内 1-indexed）
    faces_chunk: Tensor,         # (F_chunk, 3) chunk 内的局部 faces
    face_offset: int,            # chunk 在全局 faces 中的起始偏移
    intersect_logits: Tensor,    # (N, 3) raw logits，可微
    face_axis_ids: Tensor,       # (F_total,) long — 全局 face → axis 0/1/2
    face_voxel_ids: Tensor,      # (F_total,) long — 全局 face → 源 voxel 索引
    voxel_normals: Tensor,       # (N, 3) 相机空间法向量
    coords: Tensor,              # (N, 3) voxel 坐标
    origin: Tensor,              # (3,)
    voxel_size: float,
    voxel_resolution: int,
    vertices_batch: Tensor,      # (1, V, 3)
    rast_res: int,
    compute_normal: bool,
) -> Tuple[Tensor, Tensor]:
    """单个 chunk-layer 的可微计算（可被 checkpoint 包裹）。

    与 _peel_one_layer 的区别：
    - 不做 front-to-back 累积，只返回该层的 (alpha, normal)
    - face_id 需要 +face_offset 重映射到全局 face_axis_ids / face_voxel_ids
    - dr.interpolate 使用 chunk 内的 faces_chunk + rast，无需重映射

    梯度路径:
      路径 1: intersect_logits → gather → sigmoid → alpha
      路径 2: voxel_normals → grid_sample_3d → normal

    Args:
        rast: chunk 层的光栅化结果 (1, H, W, 4)
        faces_chunk: chunk 的局部 faces (F_chunk, 3)
        face_offset: chunk 在全局 faces 中的起始偏移量
        intersect_logits: (N, 3) 原始 logits，保留梯度
        face_axis_ids: (F_total,) 全局 face axis 映射
        face_voxel_ids: (F_total,) 全局 face voxel 映射
        voxel_normals: 相机空间法向量 (N, 3)
        coords: voxel 坐标 (N, 3)
        origin: voxel 网格原点 (3,)
        voxel_size: voxel 尺寸
        voxel_resolution: 分辨率
        vertices_batch: mesh 顶点 (1, V, 3)
        rast_res: 光栅化分辨率
        compute_normal: 是否计算 normal

    Returns:
        layer_alpha:  (1, H, W, 1) 该层 alpha（可微）
        layer_normal: (1, H, W, 3) 该层 normal（可微）
    """
    layer_mask = (rast[..., 3:4] > 0).float()                   # (1, H, W, 1)

    # ---- face_id → 全局重映射 → gather intersect_logits → sigmoid（可微） ----
    local_fid = rast[..., 3:4].long() - 1                       # (1, H, W, 1) chunk 内 0-indexed
    global_fid = (local_fid + face_offset).clamp(min=0).squeeze(-1)  # (1, H, W) 全局 face 索引
    voxel_id = face_voxel_ids[global_fid]                        # (1, H, W) — 源 voxel 索引
    axis_id = face_axis_ids[global_fid]                          # (1, H, W) — 0/1/2
    layer_logit = intersect_logits[voxel_id, axis_id]            # (1, H, W) — gather 可微
    layer_alpha = torch.sigmoid(layer_logit).unsqueeze(-1)       # (1, H, W, 1)
    layer_alpha = layer_alpha * layer_mask                       # (1, H, W, 1) 背景=0

    # ---- grid_sample_3d 采样该层 normal ----
    # dr.interpolate 使用 chunk 内的 rast + faces_chunk，索引天然匹配
    if compute_normal:
        layer_normal = _grid_sample_normal_raw(
            voxel_normals, coords, origin, voxel_size,
            voxel_resolution, vertices_batch, rast, faces_chunk, rast_res,
        )                                                        # (1, H, W, 3)
    else:
        layer_normal = torch.zeros(1, rast_res, rast_res, 3,
                                   device=rast.device)           # (1, H, W, 3)

    return layer_alpha, layer_normal


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

class HybridPeeled26NormalRenderer:
    """混合 Normal 渲染器（Depth Peeling 版）

    使用 26-neighbor occupancy 差分计算可微法向量（对 subs 可微），
    使用 nvdiffrast DepthPeeler 多层渲染 + per-face alpha 合成
    （对 intersect_logits 可微）。

    rendering_options:
        resolution: 渲染分辨率
        near/far: 近远裁剪面
        ssaa: 超采样倍率
        peel_layers: DepthPeeler 层数（默认 8，voxel mesh 通常 8~16 足够）
        grad_checkpoint: 是否启用 gradient checkpoint（训练时省显存）
    """

    def __init__(self, rendering_options: dict = {}, device: str = "cuda"):
        self.rendering_options = edict({
            "resolution": 512,
            "near": 0.1,
            "far": 100.0,
            "ssaa": 1,
            "peel_layers": 8,
            "grad_checkpoint": True,
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

        # Hard voxel renderer：用于获取可见 voxel ID（无梯度，~1ms CUDA kernel）
        self.voxel_renderer = o_voxel.rasterize.VoxelRenderer({
            "resolution": self.rendering_options["resolution"],
            "near": self.rendering_options["near"],
            "far": self.rendering_options["far"],
            "ssaa": 1,  # 只需要 voxel_id，不需要 SSAA
        })

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def render(
        self,
        mesh: Any,
        subs: List[Any],
        coords: Tensor,
        intersect_logits: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        voxel_resolution: int,
        return_types: List[str] = ["normal", "mask"],
    ) -> edict:
        """渲染可微法向量（DepthPeeler 多层 alpha 合成）

        5 步流水线:
          0. 预计算（无梯度）：face 映射 + VoxelRenderer 可见性
          1. 坐标变换
          2. 可微 voxel normal（仅 active 子集）
          3. DepthPeeler 多层渲染 + alpha 合成
          4. SSAA 下采样

        Args:
            mesh: 含 .vertices (V, 3) 和 .faces (F, 3) 的对象
            subs: 多层 sub_logits（用于 26-neighbor 差分）
            coords: (N, 3) voxel 整数坐标（vertex index == voxel index）
            intersect_logits: (N, 3) 原始 logits，保留梯度
            extrinsics: (4, 4) 相机外参
            intrinsics: (3, 3) 相机内参
            voxel_resolution: voxel 分辨率
            return_types: 需要返回的类型列表

        Returns:
            edict: 包含 normal (H, W, 3)、mask (H, W) 等
        """
        vertices, faces = mesh.vertices, mesh.faces
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return self._empty_result(return_types)

        # ============ Phase 0: 预计算（无梯度） ============
        with torch.no_grad():
            # 0a. 反推 per-face 的 axis 和 source voxel
            face_axis_ids, face_voxel_ids = recover_face_axis_and_voxel(
                faces, coords, voxel_resolution,
            )  # face_axis_ids: (F,), face_voxel_ids: (F,)

            # 0b. VoxelRenderer → 可见 voxel → dilate → active set
            active_voxel_ids = self._get_active_voxel_ids(
                coords, voxel_resolution, extrinsics, intrinsics,
            )  # (K,)

        # ============ Phase 1: 坐标变换 ============
        vertices_clip, vertices_cam, vertices_batch = self._transform_vertices(
            vertices, extrinsics, intrinsics, return_types)

        # ============ Phase 2: 可微 voxel normal（仅 active 子集） ============
        voxel_normals = self._prepare_voxel_normals(
            active_voxel_ids, vertices, faces, coords,
            subs, extrinsics, voxel_resolution, return_types)

        # ============ Phase 3: DepthPeeler 多层渲染 + alpha 合成 ============
        out_dict = self._depth_peel_render(
            vertices_clip, vertices_cam, vertices_batch,
            faces, intersect_logits, face_axis_ids, face_voxel_ids,
            voxel_normals, coords, voxel_resolution, return_types)

        # ============ Phase 4: 下采样 ============
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
    # Phase 0b: VoxelRenderer → 可见 voxel → dilate → active set
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_active_voxel_ids(
        self, coords: Tensor, voxel_resolution: int,
        extrinsics: Tensor, intrinsics: Tensor,
    ) -> Tensor:
        """Hard VoxelRenderer 获取可见 voxel → 26-邻居膨胀 → active set

        VoxelRenderer 是 CUDA kernel (~1ms)，无梯度。
        膨胀确保法线计算所需的邻居全部包含在 active set 中。

        Args:
            coords: (N, 3) int — voxel 整数坐标
            voxel_resolution: voxel 分辨率
            extrinsics: (4, 4) 相机外参
            intrinsics: (3, 3) 相机内参

        Returns:
            active_voxel_ids: (K,) long — 可见 + 膨胀后的 voxel 索引
        """
        N = coords.shape[0]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)

        # 整数坐标 → 世界坐标
        positions = (coords.float() + 0.5) * voxel_size + origin  # (N, 3)

        # VoxelRenderer：只需 voxel_id，attrs 用 dummy
        dummy_attrs = torch.zeros(N, 1, device=self.device)  # (N, 1)
        ret = self.voxel_renderer.render(
            positions, dummy_attrs, voxel_size, extrinsics, intrinsics
        )
        voxel_id_map = ret.voxel_id  # (H, W) int, -1=背景

        # 收集 unique 可见 voxel
        visible_ids = voxel_id_map[voxel_id_map >= 0].unique().long()  # (K_vis,)
        if visible_ids.numel() == 0:
            return visible_ids

        # 26-邻居膨胀：确保法线差分所需的邻居在 active set 中
        vis_coords = coords[visible_ids]  # (K_vis, 3)
        offsets, _ = _neighbor_offsets_26(self.device)  # (26, 3)
        neighbor_coords = (vis_coords.unsqueeze(1)
                           + offsets.unsqueeze(0))  # (K_vis, 26, 3)
        neighbor_coords = neighbor_coords.reshape(-1, 3)  # (K_vis*26, 3)

        # GPU hashmap 查找邻居 → 原始索引
        INVALID = 0xffffffff
        grid_size = torch.tensor([voxel_resolution] * 3, device=self.device)
        hashmap = _init_hashmap(grid_size, 2 * N, self.device)
        coords_with_batch = torch.cat([
            torch.zeros_like(coords[:, :1]), coords
        ], dim=-1)  # (N, 4)
        _C.hashmap_insert_3d_idx_as_val_cuda(
            *hashmap, coords_with_batch, *grid_size.tolist()
        )
        query = torch.cat([
            torch.zeros(neighbor_coords.shape[0], 1,
                        dtype=torch.int, device=self.device),
            neighbor_coords.int(),
        ], dim=-1)  # (K_vis*26, 4)
        neighbor_ids = _C.hashmap_lookup_3d_cuda(
            *hashmap, query, *grid_size.tolist()
        ).long()  # (K_vis*26,)

        valid = neighbor_ids != INVALID
        neighbor_ids = neighbor_ids[valid]  # (num_valid,)

        # 合并：可见 + 邻居
        active_voxel_ids = torch.cat([visible_ids, neighbor_ids]).unique()  # (K,)
        return active_voxel_ids

    # ------------------------------------------------------------------
    # Phase 2: 可微 Voxel Normal
    # ------------------------------------------------------------------

    def _prepare_voxel_normals(self, active_voxel_ids, vertices, faces, coords,
                                subs, extrinsics, voxel_resolution, return_types):
        """计算 active voxel 的可微法向量 → (N, 3) 或 None

        active_voxel_ids 由 _get_active_voxel_ids 在 Phase 0 中计算，
        是 visible voxel 的超集（含 26-邻居膨胀）。
        """
        N = vertices.shape[0]
        if "normal" not in return_types:
            return None

        voxel_normals = torch.zeros(N, 3, device=self.device)  # (N, 3)
        K = active_voxel_ids.shape[0]
        if K == 0:
            return voxel_normals

        use_ckpt = self.rendering_options["grad_checkpoint"]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)

        # 几何 v_normal（世界坐标系）
        v_normal_all = compute_vertex_normals(vertices, faces)  # (N, 3)
        active_v_normal = v_normal_all[active_voxel_ids]  # (K, 3)
        del v_normal_all

        active_coords = coords[active_voxel_ids]  # (K, 3)
        active_pos = (active_coords.float() + 0.5) * voxel_size + origin  # (K, 3)

        # 世界坐标系下翻转参考方向
        ref_normal = _flip_normals_world(
            active_v_normal, active_pos, extrinsics)  # (K, 3)
        del active_v_normal

        # ProfiledScheduler 自适应分块计算 26-neighbor normal
        results = []
        for start, size in ProfiledScheduler(
            K, self.device,
            probe_size=_NORMAL_PROBE_SIZE,
            safety_factor=1.3,
            min_chunk=500,
            max_chunk=100000,
        ):
            end = start + size
            if use_ckpt:
                chunk_result = checkpoint(
                    compute_voxel_normal,
                    active_coords[start:end], subs,
                    ref_normal[start:end], voxel_resolution,
                    use_reentrant=False,
                )
            else:
                chunk_result = compute_voxel_normal(
                    active_coords[start:end], subs,
                    ref_normal[start:end], voxel_resolution,
                )
            results.append(chunk_result)
        normal_world = torch.cat(results, dim=0)  # (K, 3)
        del ref_normal

        # 变换到相机坐标系 + 翻转
        normal_cam = _flip_normals_to_camera(
            normal_world, active_pos, extrinsics)  # (K, 3)
        del normal_world, active_pos

        voxel_normals[active_voxel_ids] = normal_cam
        del normal_cam
        return voxel_normals

    # ------------------------------------------------------------------
    # Phase 3: DepthPeeler 多层渲染 + Alpha 合成  ★ 核心变化
    # ------------------------------------------------------------------

    def _depth_peel_render(
        self, vertices_clip, vertices_cam, vertices_batch,
        faces, intersect_logits, face_axis_ids, face_voxel_ids,
        voxel_normals, coords, voxel_resolution, return_types,
    ):
        """DepthPeeler 多层渲染 + front-to-back alpha 合成（自动分 chunk）

        入口方法，依次调用四个子阶段。临时张量的生命周期限定在各子函数内，
        降低峰值显存占用。

        梯度路径:
          路径 1: intersect_logits → gather → sigmoid → alpha → composite → normal_acc
          路径 2: voxel_normals → grid_sample_3d → normal → composite → normal_acc

        Args:
            vertices_clip: (1, V, 4)
            vertices_cam: (1, V, 4) | None
            vertices_batch: (1, V, 3)
            faces: (F, 3)
            intersect_logits: (N, 3) 原始 logits，可微
            face_axis_ids: (F,) long — 每个 face 的 axis 0/1/2
            face_voxel_ids: (F,) long — 每个 face 的源 voxel 索引
            voxel_normals: (N, 3) 相机空间法向量
            coords: (N, 3) voxel 坐标
            voxel_resolution: int
            return_types: List[str]

        Returns:
            out_dict: edict，包含 normal (1, H, H, 3) 和 mask (1, H, H, 1)
        """
        rast_res = self.rendering_options["resolution"] * self.rendering_options["ssaa"]
        peel_layers = self.rendering_options["peel_layers"]
        use_ckpt = self.rendering_options["grad_checkpoint"]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)  # (3,)
        compute_normal = ("normal" in return_types and voxel_normals is not None)

        # Phase A: 逐 chunk 逐层 peel + 预计算
        all_depths, all_alphas, all_normals, fl_depths, fl_cam_depths = \
            self._peel_all_chunks(
                vertices_clip, vertices_cam, vertices_batch, faces,
                intersect_logits, face_axis_ids, face_voxel_ids,
                voxel_normals, coords, origin, voxel_size,
                voxel_resolution, rast_res, peel_layers,
                use_ckpt, compute_normal, return_types,
            )

        # Phase B+C: per-pixel 深度排序 + front-to-back composite
        normal_acc, alpha_acc = self._sort_and_composite(
            all_depths, all_alphas, all_normals, rast_res, self.device,
        )
        del all_depths, all_alphas, all_normals  # 及时释放

        # 全局首层 depth
        depth_first = self._merge_first_layer_depth(
            fl_depths, fl_cam_depths, return_types, rast_res, self.device,
        )
        del fl_depths, fl_cam_depths

        # 组装输出
        return self._assemble_output(normal_acc, alpha_acc, depth_first, return_types)

    # ---- Phase A 子函数 ----

    def _peel_all_chunks(
        self,
        vertices_clip: Tensor,    # (1, V, 4)
        vertices_cam,             # (1, V, 4) | None
        vertices_batch: Tensor,   # (1, V, 3)
        faces: Tensor,            # (F, 3)
        intersect_logits: Tensor, # (N, 3)
        face_axis_ids: Tensor,    # (F,) long
        face_voxel_ids: Tensor,   # (F,) long
        voxel_normals: Tensor,    # (N, 3)
        coords: Tensor,           # (N, 3)
        origin: Tensor,           # (3,)
        voxel_size: float,
        voxel_resolution: int,
        rast_res: int,
        peel_layers: int,
        use_ckpt: bool,
        compute_normal: bool,
        return_types: List[str],
    ):
        """Phase A: 逐 chunk 逐层 peel + 预计算 (alpha, normal, depth)

        将全局 faces 按 _MAX_FACES_PER_CHUNK 分块，每块用独立 DepthPeeler
        剥离 peel_layers 层，收集 (depth, alpha, normal)。

        Returns:
            all_depths:             List[Tensor(1, H, W)]    — 非可微
            all_alphas:             List[Tensor(1, H, W, 1)] — ★ 可微
            all_normals:            List[Tensor(1, H, W, 3)] — ★ 可微
            first_layer_depths:     List[Tensor(1, H, W)]
            first_layer_cam_depths: List[Tensor(1, H, W, 1) | None]
        """
        num_faces = faces.shape[0]
        K = (num_faces + _MAX_FACES_PER_CHUNK - 1) // _MAX_FACES_PER_CHUNK
        if K > 1:
            logging.info(
                f"[DepthPeeler] faces={num_faces} > {_MAX_FACES_PER_CHUNK}, "
                f"splitting into {K} chunks"
            )

        all_depths:  list = []  # List[Tensor(1, H, W)]    非可微，用于排序
        all_alphas:  list = []  # List[Tensor(1, H, W, 1)] ★ 可微
        all_normals: list = []  # List[Tensor(1, H, W, 3)] ★ 可微
        first_layer_depths:     list = []  # List[Tensor(1, H, W)]
        first_layer_cam_depths: list = []  # List[Tensor(1, H, W, 1) | None]

        for chunk_idx in range(K):
            off = chunk_idx * _MAX_FACES_PER_CHUNK
            size = min(_MAX_FACES_PER_CHUNK, num_faces - off)
            faces_k = faces[off : off + size]                    # (F_chunk, 3)

            with dr.DepthPeeler(self.glctx, vertices_clip, faces_k,
                                (rast_res, rast_res)) as peeler:
                for layer_idx in range(peel_layers):
                    rast, _ = peeler.rasterize_next_layer()       # (1, H, W, 4)

                    # 提前终止：该层完全空 → 后续层也空
                    if (rast[..., 3] == 0).all():
                        break

                    # depth（非可微，仅用于排序）
                    depth = rast[..., 2].detach().clone()        # (1, H, W) 独立拷贝，避免 in-place 修改 rast 存储
                    depth[rast[..., 3] == 0] = float('inf')      # 空像素 → inf

                    # 记录各 chunk 的 layer 0 信息，用于确定全局首层 depth
                    if layer_idx == 0:
                        first_layer_depths.append(depth.clone())
                        if "depth" in return_types and vertices_cam is not None:
                            cam_d = dr.interpolate(
                                vertices_cam[..., 2:3].contiguous(),
                                rast, faces_k,
                            )[0]                                 # (1, H, W, 1)
                            first_layer_cam_depths.append(cam_d)
                        else:
                            first_layer_cam_depths.append(None)

                    # 核心可微计算：checkpoint 包裹
                    fn = _compute_one_chunk_layer
                    args = (rast, faces_k, off,
                            intersect_logits, face_axis_ids, face_voxel_ids,
                            voxel_normals, coords, origin, voxel_size,
                            voxel_resolution, vertices_batch, rast_res,
                            compute_normal)
                    alpha, normal = (checkpoint(fn, *args, use_reentrant=False)
                                     if use_ckpt else fn(*args))

                    all_depths.append(depth)
                    all_alphas.append(alpha)                      # (1, H, W, 1) 可微
                    all_normals.append(normal)                    # (1, H, W, 3) 可微

        return all_depths, all_alphas, all_normals, first_layer_depths, first_layer_cam_depths

    # ---- Phase B+C 子函数 ----

    @staticmethod
    def _sort_and_composite(
        all_depths: list,   # List[Tensor(1, H, W)]
        all_alphas: list,   # List[Tensor(1, H, W, 1)]  ★ 可微
        all_normals: list,  # List[Tensor(1, H, W, 3)]  ★ 可微
        rast_res: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Phase B+C: per-pixel 深度排序 + front-to-back alpha composite

        torch.gather 对 input 可微、对 index 不可微 → 梯度正确。

        Returns:
            normal_acc: (1, H, W, 3)
            alpha_acc:  (1, H, W, 1)
        """
        normal_acc = torch.zeros(1, rast_res, rast_res, 3,
                                 device=device)                  # (1, H, W, 3)
        alpha_acc = torch.zeros(1, rast_res, rast_res, 1,
                                device=device)                   # (1, H, W, 1)
        if not all_depths:
            return normal_acc, alpha_acc

        T = len(all_depths)
        sort_idx = torch.stack(all_depths).argsort(dim=0)        # (T, 1, H, W)

        sorted_a = torch.gather(
            torch.stack(all_alphas), 0,
            sort_idx.unsqueeze(-1),                              # (T, 1, H, W, 1)
        )                                                        # (T, 1, H, W, 1)
        sorted_n = torch.gather(
            torch.stack(all_normals), 0,
            sort_idx.unsqueeze(-1).expand(-1, -1, -1, -1, 3),   # (T, 1, H, W, 3)
        )                                                        # (T, 1, H, W, 3)

        for rank in range(T):
            w = (1.0 - alpha_acc) * sorted_a[rank]               # (1, H, W, 1)
            normal_acc = normal_acc + w * sorted_n[rank]         # (1, H, W, 3)
            alpha_acc = alpha_acc + w                            # (1, H, W, 1)

        return normal_acc, alpha_acc

    # ---- 全局首层 depth 归并 ----

    @staticmethod
    def _merge_first_layer_depth(
        first_layer_depths: list,     # List[Tensor(1, H, W)]
        first_layer_cam_depths: list, # List[Tensor(1, H, W, 1) | None]
        return_types: List[str],
        rast_res: int,
        device: torch.device,
    ) -> Tensor:
        """跨 chunk 取最近首层 cam-space depth

        Returns:
            depth_first: (1, H, W, 1)
        """
        depth_first = torch.zeros(1, rast_res, rast_res, 1,
                                  device=device)                 # (1, H, W, 1)
        if not first_layer_depths:
            return depth_first

        stacked_fl = torch.stack(first_layer_depths)             # (K', 1, H, W)
        if len(first_layer_depths) == 1:
            closest_chunk = torch.zeros(
                1, rast_res, rast_res,
                dtype=torch.long, device=device,
            )                                                    # (1, H, W)
        else:
            closest_chunk = stacked_fl.argmin(dim=0)             # (1, H, W)

        if "depth" in return_types and any(d is not None for d in first_layer_cam_depths):
            cam_stack = torch.stack([
                d if d is not None
                else torch.zeros(1, rast_res, rast_res, 1, device=device)
                for d in first_layer_cam_depths
            ])                                                   # (K', 1, H, W, 1)
            idx_d = closest_chunk.unsqueeze(-1)                  # (1, H, W, 1)
            depth_first = cam_stack.gather(
                0, idx_d.unsqueeze(0),                           # (1, 1, H, W, 1)
            ).squeeze(0)                                         # (1, H, W, 1)

        return depth_first

    # ---- 输出组装 ----

    @staticmethod
    def _assemble_output(
        normal_acc: Tensor,    # (1, H, W, 3)
        alpha_acc: Tensor,     # (1, H, W, 1)
        depth_first: Tensor,   # (1, H, W, 1)
        return_types: List[str],
    ) -> edict:
        """将 composited normal / alpha / depth 组装为 out_dict"""
        out_dict = edict()

        if "normal" in return_types:
            safe_alpha = alpha_acc.clamp(min=1e-6)               # (1, H, W, 1)
            composited = normal_acc / safe_alpha                 # (1, H, W, 3)
            composited = F.normalize(composited, dim=-1, eps=1e-6)  # (1, H, W, 3)
            composited = (composited + 1) / 2                    # (1, H, W, 3) → [0, 1]
            bg = 0.5
            out_dict["normal"] = composited * alpha_acc + bg * (1 - alpha_acc)  # (1, H, W, 3)

        if "mask" in return_types:
            out_dict["mask"] = alpha_acc                         # (1, H, W, 1)

        if "depth" in return_types:
            out_dict["depth"] = depth_first                      # (1, H, W, 1)

        return out_dict

    # ------------------------------------------------------------------
    # Phase 4: SSAA 下采样
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
