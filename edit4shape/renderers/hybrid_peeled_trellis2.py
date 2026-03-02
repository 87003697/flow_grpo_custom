# hybrid_peeled_trellis2.py

"""
混合 Normal 渲染器（重心采样版）：
  Voxel Normal (subs 可微) + 单层光栅化 (intersect_logits 可微)

使用面重心（centroid）在 voxel 网格上采样 per-face normal，
单次 dr.rasterize（分 chunk z-buffer 归并）覆盖所有可见面片。
voxel normal 计算、grid_sample、翻转均在 chunk 循环内，
全部被 gradient checkpoint 包裹，最大程度节省显存。

梯度路径:
  路径 1: subs → occupancy_diff → voxel_normal → grid_sample_3d_differentiable → face_normal → pixel_normal
  路径 2: intersect_logits → gather → sigmoid → alpha → pixel_normal
  路径 3: dual_vertices → mesh_vertices → centroids → grid_sample_3d_differentiable(query_pts) → face_normal → pixel_normal
  路径 4: vertices → cross → face_normals → scatter_add → ref_normals_all → direction_weight → voxel_normal → pixel_normal

调用栈:
  系统层 (decode_and_render_normal_hybrid26):
  ├── flexible_dual_grid_to_mesh(train=True)
  └── renderer.render(mesh, subs, coords, intersect_logits, extrinsics, intrinsics, ...)

  渲染器层 (render):
  ├── Phase 0: 预计算
  │     ├── recover_face_axis_and_voxel → face_axis_ids, face_voxel_ids（@torch.no_grad）
  │     └── compute_ref_normals_from_faces → ref_normals_all（★ 对 vertices 可微）
  ├── Phase 1: _transform_vertices(...)           → vertices_clip, vertices_cam, vertices_batch
  ├── Phase 2+3: _rasterize_and_render(...)
  │     └── for chunk in split(faces, _MAX_FACES_PER_CHUNK):
  │           ├── dr.rasterize(vertices_clip, faces_chunk)  【不可微，checkpoint 外】
  │           └── checkpoint(_compute_one_chunk):
  │                 ├── _find_active_voxels_for_chunk   → centroids_world/voxel, active_voxel_ids
  │                 ├── compute_voxel_normal             → voxel_normals_world  (★ subs 可微)
  │                 ├── _sample_face_normals             → face_normals_world   (★ 双可微)
  │                 ├── _gather_pixel_normal_and_flip    → pixel_normal_cam     (per-pixel world→cam + flip)
  │                 └── _compute_chunk_alpha             → layer_alpha          (★ intersect_logits 可微)
  │     └── 跨 chunk per-pixel z-buffer 归并
  ├── Phase 4: _assemble_output                   → normal [0,1] + mask + depth
  └── Phase 5: _downsample (SSAA)                 → final

使用方法:
    renderer = Hybrid26NormalRenderer({"resolution": 512})
    outputs = renderer.render(
        mesh, subs, coords, intersect_logits,
        extrinsics, intrinsics, voxel_resolution)
    normal = outputs.normal  # (H, W, 3)
"""

from typing import List, Any, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from edit4shape.generators.trellis2.ops.grid_sample3d import grid_sample_3d_differentiable
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap
from edit4shape.renderers.mesh_peeled_trellis2 import (
    recover_face_axis_and_voxel,
    intrinsics_to_projection,
)

_MAX_FACES_PER_CHUNK = 4_000_000

# =============================================================================
# 辅助函数（保留：26-neighbor occupancy / voxel normal）
# =============================================================================


def compute_vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """计算 vertex normals（世界坐标系）

    通过累加所有相邻面的面法线（面积加权）得到 per-vertex 法线。

    Args:
        vertices: (N, 3) mesh 顶点（世界坐标）
        faces: (F, 3) 面索引

    Returns:
        v_normals: (N, 3) 每个顶点的法向量（世界坐标系，归一化）
    """
    i0 = faces[..., 0].long()  # (F,)
    i1 = faces[..., 1].long()  # (F,)
    i2 = faces[..., 2].long()  # (F,)

    v0 = vertices[i0, :]  # (F, 3)
    v1 = vertices[i1, :]  # (F, 3)
    v2 = vertices[i2, :]  # (F, 3)

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)

    v_normals = torch.zeros_like(vertices)  # (N, 3)
    v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)  # (N, 3)
    v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)  # (N, 3)
    v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)  # (N, 3)

    v_normals = F.normalize(v_normals, dim=1, eps=1e-6)  # (N, 3)
    return v_normals


def compute_ref_normals_from_faces(
    vertices: Tensor,        # (V, 3) mesh 顶点（世界坐标）★ 可微
    faces: Tensor,           # (F, 3) 面索引
    face_voxel_ids: Tensor,  # (F,) 每个面的源 voxel 索引
    num_voxels: int,         # N，voxel 总数
) -> Tensor:
    """将 face normal scatter 到源 voxel，得到 per-voxel 参考法线

    每个 voxel 最多生成 3 个面（x/y/z 轴各一个），
    通过 scatter_add 累加归一化后的面法线，再整体归一化。

    梯度路径: vertices → cross → face_normals → scatter_add → ref_normals

    Args:
        vertices: (V, 3) mesh 顶点，保留梯度
        faces: (F, 3) 面索引
        face_voxel_ids: (F,) 每个面对应的源 voxel 在 coords 中的索引
        num_voxels: voxel 总数（coords.shape[0]）

    Returns:
        ref_normals_all: (N, 3) 每个 voxel 的参考法线（世界坐标系，归一化）
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    face_normals_raw = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3)
    face_normals_unit = F.normalize(face_normals_raw, dim=-1)  # (F, 3)

    ref_normals_all = torch.zeros(num_voxels, 3, device=vertices.device)  # (N, 3)
    ref_normals_all.scatter_add_(
        0, face_voxel_ids.unsqueeze(-1).expand(-1, 3), face_normals_unit
    )  # (N, 3) 每个 voxel 累加其 ≤3 个面的法线
    ref_normals_all = F.normalize(ref_normals_all, dim=-1, eps=1e-6)  # (N, 3)
    return ref_normals_all


def _compute_neighbor_occupancy_soft(
    neighbor_coords: Tensor,       # (K, N, 3) 目标分辨率下的查询坐标
    subs: List[Any],               # [sub0, sub1, sub2, ...] 各层 sub logits
    voxel_resolution: int,         # 目标 voxel 分辨率
) -> Tensor:
    """计算查询位置的 soft occupancy（可微，支持多分辨率）

    跨层 sigmoid 累加：遍历所有层，每层独立 sigmoid 后累加。
    梯度同时流向所有命中层的 sub.feats，且各层 sigmoid 独立不会饱和。
    最终 occupancy 范围 [0, num_levels]，但在 compute_voxel_normal 中
    仅用于差分 + 归一化，绝对尺度不影响法线方向。

    Args:
        neighbor_coords: (K, N, 3) 目标分辨率下的查询坐标
        subs: 各层 sub logits
        voxel_resolution: 目标 voxel 分辨率

    Returns:
        occupancy: (K, N) 范围 [0, num_levels]，可微
    """
    device = neighbor_coords.device
    K, N = neighbor_coords.shape[0], neighbor_coords.shape[1]
    INVALID = 0xffffffff

    # 累加各层 sigmoid(logit)
    occ_sum = torch.zeros(K, N, device=device)  # (K, N)

    # 遍历所有层（不提前终止，每层都贡献）
    for level in range(len(subs) - 1, -1, -1):
        sub = subs[level]
        sub_coords = sub.coords[:, 1:]  # (M, 3)
        sub_logits = sub.feats.float()  # (M, 8)
        M = sub_coords.shape[0]

        # 当前层的分辨率
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

        exists = (indices != INVALID)  # (K, N)

        if exists.any():
            # 计算 corner_idx：用 child 层坐标
            corner_idx = (
                (neighbor_coords_child[..., 0] % 2) +
                (neighbor_coords_child[..., 1] % 2) * 2 +
                (neighbor_coords_child[..., 2] % 2) * 4
            ).long()  # (K, N)

            exists_flat = exists.reshape(-1)  # (K*N,)
            exists_idx = exists_flat.nonzero(as_tuple=True)[0]  # (num_found,)

            indices_flat = indices.long().reshape(-1)  # (K*N,)
            corner_idx_flat = corner_idx.reshape(-1)  # (K*N,)

            indices_sel = indices_flat[exists_idx].clamp(0, M - 1)  # (num_found,)
            corner_sel = corner_idx_flat[exists_idx]  # (num_found,)

            # 获取 corner logit → sigmoid
            parent_logits_sel = sub_logits[indices_sel]  # (num_found, 8)
            specific_logit_sel = parent_logits_sel.gather(
                -1, corner_sel.unsqueeze(-1)
            ).squeeze(-1)  # (num_found,)
            occ_sel = torch.sigmoid(specific_logit_sel)  # (num_found,)

            # 用 scatter_add 累加到 occ_sum（避免 in-place 修改，autograd 安全）
            contrib = torch.zeros(K * N, device=device)  # (K*N,)
            contrib.scatter_(0, exists_idx, occ_sel)  # (K*N,)
            occ_sum = occ_sum + contrib.reshape(K, N)  # (K, N)

    # 未在任何层找到的位置 occ_sum 自然为 0（空气）
    return occ_sum  # (K, N)


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
                    continue
                offsets.append([dx, dy, dz])
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                weights.append(1.0 / dist)
    
    offsets = torch.tensor(offsets, dtype=torch.int, device=device)  # (26, 3)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)  # (26,)
    return offsets, weights


def compute_voxel_normal(
    coords: Tensor,          # (K, 3)
    subs: List[Any],
    ref_normal: Tensor,      # (K, 3) 世界坐标系，翻转后的参考方向
    voxel_resolution: int,
) -> Tensor:
    """计算 26-neighbor occupancy 差分法向量（对 subs 可微）

    使用 ref_normal（几何法线）做方向引导：
    - 与参考法线同向的邻居 occupancy 差分被放大
    - 与参考法线垂直的邻居 occupancy 差分被抑制
    - 与参考法线反向的邻居 occupancy 差分被反转
    这样在高分辨率下 occupancy 接近二值时，仍能得到准确的法线方向。

    梯度路径: subs → occupancy_diff → normal
    （ref_normal 被 detach，梯度不流回 vertices）

    Args:
        coords: (K, 3) voxel 整数坐标
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

    # 构造 27 个查询坐标（中心 + 26 邻居）
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

    # 参考方向引导：放大法线方向的梯度，抑制切线方向噪声
    ref = ref_normal  # (K, 3) ★ 允许梯度流回 vertices
    direction_weight = torch.einsum('kd,nd->kn', ref, directions)  # (K, 26)

    # 融合加权 + 矩阵乘，避免分配 (K, 26, 3) 中间张量
    combined = occupancy_diff * direction_weight  # (K, 26)
    normal = combined @ weighted_dirs              # (K, 3)

    normal = F.normalize(normal, dim=-1, eps=1e-6)    # (K, 3)
    return normal


# =============================================================================
# 新子函数：_compute_one_chunk 的组成部分
# =============================================================================


@torch.no_grad()
def _find_active_voxels_for_chunk(
    centroids_voxel: Tensor,    # (F_c, 3) 面重心 voxel 坐标（detached）
    coords: Tensor,             # (N, 3) 全部 voxel 整数坐标
    voxel_resolution: int,
) -> Tuple[Tensor, Tensor]:
    """查找 centroids 的 8 trilinear 邻居在 coords 中的索引

    对每个 centroid，其三线性插值涉及 floor(q-0.5) + {0,1}^3 共 8 个邻居。
    通过 hashmap 查找这些邻居在 coords 中的行索引，去重后返回。

    Args:
        centroids_voxel: (F_c, 3) 面重心的 voxel 坐标（已 detach）
        coords: (N, 3) 全部 voxel 整数坐标
        voxel_resolution: voxel 分辨率

    Returns:
        active_voxel_ids: (K,) long — 需要计算法线的 voxel 在 coords 中的索引
        active_coords: (K, 3) int — 对应的整数坐标
    """
    device = coords.device
    F_c = centroids_voxel.shape[0]
    N = coords.shape[0]

    # 8 trilinear 邻居
    base = torch.floor(centroids_voxel - 0.5).int()  # (F_c, 3)
    offsets_8 = torch.tensor(
        [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)],
        device=device, dtype=torch.int,
    )  # (8, 3)
    neighbor_int = base[:, None, :] + offsets_8[None, :, :]  # (F_c, 8, 3)
    neighbor_int = neighbor_int.reshape(-1, 3)  # (F_c*8, 3)
    neighbor_int = neighbor_int.clamp(0, voxel_resolution - 1)  # (F_c*8, 3)

    # Hashmap 查找
    INVALID = 0xffffffff
    grid_size = torch.tensor([voxel_resolution] * 3, device=device)
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    coords_with_batch = torch.cat([
        torch.zeros_like(coords[:, :1]), coords
    ], dim=-1)  # (N, 4)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap, coords_with_batch, *grid_size.tolist()
    )

    query = torch.cat([
        torch.zeros(neighbor_int.shape[0], 1, dtype=torch.int, device=device),
        neighbor_int,
    ], dim=-1)  # (F_c*8, 4)
    indices = _C.hashmap_lookup_3d_cuda(
        *hashmap, query, *grid_size.tolist()
    ).long()  # (F_c*8,)

    valid = indices != INVALID
    active_voxel_ids = indices[valid].unique()  # (K,)
    active_coords = coords[active_voxel_ids]    # (K, 3)
    return active_voxel_ids, active_coords


def _sample_face_normals(
    voxel_normals: Tensor,     # (K, 3) active voxels 的法线（world space）
    active_coords: Tensor,     # (K, 3) active voxels 的整数坐标
    centroids_voxel: Tensor,   # (F_c, 3) 面重心 voxel 坐标 ★ 可微
    voxel_resolution: int,
) -> Tensor:
    """用 grid_sample_3d_differentiable 从 voxel normals 采样 per-face normal

    Args:
        voxel_normals: (K, 3) world space 法线（★ 对 subs 可微）
        active_coords: (K, 3) 整数坐标
        centroids_voxel: (F_c, 3) voxel 坐标（★ 对 vertices 可微）
        voxel_resolution: voxel 分辨率

    Returns:
        face_normals: (F_c, 3) world space（★ 对 voxel_normals 和 centroids_voxel 双可微）
    """
    coords_4d = torch.cat([
        torch.zeros_like(active_coords[:, :1]), active_coords
    ], dim=-1)  # (K, 4)

    voxel_shape = torch.Size([
        1, 3, voxel_resolution, voxel_resolution, voxel_resolution
    ])

    query = centroids_voxel.unsqueeze(0)  # (1, F_c, 3)
    face_normals = grid_sample_3d_differentiable(
        voxel_normals, coords_4d, voxel_shape, query, mode='trilinear'
    )  # (1, F_c, 3)

    face_normals = face_normals.squeeze(0)  # (F_c, 3)
    face_normals = F.normalize(face_normals, dim=-1, eps=1e-6)  # (F_c, 3)

    return face_normals


def _gather_pixel_normal_and_flip(
    face_normals_world: Tensor,  # (F_c, 3)
    centroids_world: Tensor,     # (F_c, 3) — 仅用于 flip 方向判断
    rast: Tensor,                # (1, H, W, 4)
    extrinsics: Tensor,          # (4, 4)
    rast_res: int,
) -> Tensor:
    """gather per-pixel normal + world→cam 变换 + per-pixel 翻转

    1. 由 rast face_id gather face_normals_world → pixel_normal_world
    2. world → cam: n_cam = n_world @ R^T
    3. flip: dot(n_cam, pos_cam) > 0 → 翻转（使法线朝向相机）

    Args:
        face_normals_world: (F_c, 3) 每个面的 world space 法线
        centroids_world: (F_c, 3) 面重心世界坐标（用于 flip 方向，detach）
        rast: (1, H, W, 4) 光栅化结果
        extrinsics: (4, 4) 外参
        rast_res: 光栅化分辨率

    Returns:
        pixel_normal_cam: (1, H, W, 3) camera space，背景=0
    """
    mask = (rast[..., 3:4] > 0).float()  # (1, H, W, 1)
    local_fid = rast[..., 3].long() - 1  # (1, H, W) chunk 内 0-indexed
    local_fid_safe = local_fid.clamp(min=0)  # (1, H, W)

    # gather: face_id → pixel normal (world space)
    pixel_normal_world = face_normals_world[local_fid_safe]  # (1, H, W, 3)
    pixel_normal_world = pixel_normal_world * mask  # (1, H, W, 3)

    # world → camera
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    pixel_normal_cam = pixel_normal_world @ R.T  # (1, H, W, 3)

    # per-pixel flip：用 detached centroid 位置判断方向
    # dot(n_cam, pos_cam) > 0 → 法线朝远离相机方向 → 需要翻转（-1）
    # dot(n_cam, pos_cam) ≤ 0 → 法线朝向相机 → 保持（+1）
    with torch.no_grad():
        pixel_pos_world = centroids_world.detach()[local_fid_safe]  # (1, H, W, 3)
        pixel_pos_cam = pixel_pos_world @ R.T + t  # (1, H, W, 3)
        flip_sign = torch.where(
            (pixel_normal_cam.detach() * pixel_pos_cam).sum(dim=-1, keepdim=True) > 0,
            -torch.ones(1, device=rast.device),   # dot > 0 → 翻转
            torch.ones(1, device=rast.device),     # dot ≤ 0 → 保持
        )  # (1, H, W, 1)

    pixel_normal_cam = pixel_normal_cam * flip_sign  # (1, H, W, 3)
    pixel_normal_cam = pixel_normal_cam * mask  # (1, H, W, 3) 背景=0
    pixel_normal_cam = F.normalize(pixel_normal_cam, dim=-1, eps=1e-6)  # (1, H, W, 3)
    return pixel_normal_cam


def _compute_chunk_alpha(
    rast: Tensor,                # (1, H, W, 4)
    face_offset: int,
    intersect_logits: Tensor,    # (N, 3) raw logits
    face_axis_ids: Tensor,       # (F_total,) long
    face_voxel_ids: Tensor,      # (F_total,) long
) -> Tensor:
    """face_id → intersect_logits gather → sigmoid → alpha

    Args:
        rast: (1, H, W, 4) 光栅化结果（face_id 1-indexed within chunk）
        face_offset: chunk 在全局 faces 中的起始偏移
        intersect_logits: (N, 3) raw logits（★ 可微）
        face_axis_ids: (F_total,) 全局 face → axis 0/1/2
        face_voxel_ids: (F_total,) 全局 face → 源 voxel 索引

    Returns:
        layer_alpha: (1, H, W, 1)（★ 对 intersect_logits 可微）
    """
    layer_mask = (rast[..., 3:4] > 0).float()  # (1, H, W, 1)

    local_fid = rast[..., 3:4].long() - 1  # (1, H, W, 1) chunk 内 0-indexed
    global_fid = (local_fid + face_offset).clamp(min=0).squeeze(-1)  # (1, H, W)
    voxel_id = face_voxel_ids[global_fid]   # (1, H, W)
    axis_id = face_axis_ids[global_fid]     # (1, H, W)
    layer_logit = intersect_logits[voxel_id, axis_id]  # (1, H, W) gather 可微
    layer_alpha = torch.sigmoid(layer_logit).unsqueeze(-1)  # (1, H, W, 1)
    layer_alpha = layer_alpha * layer_mask  # (1, H, W, 1)
    return layer_alpha


def _compute_one_chunk(
    rast: Tensor,                # (1, H, W, 4) — dr.rasterize 的输出
    faces_chunk: Tensor,         # (F_c, 3) chunk 内面索引
    face_offset: int,            # chunk 起始偏移
    vertices: Tensor,            # (V, 3) 世界坐标 ★ 可微
    coords: Tensor,              # (N, 3) voxel 整数坐标
    origin: Tensor,              # (3,)
    voxel_size: float,
    voxel_resolution: int,
    subs: List[Any],             # ★ 可微
    extrinsics: Tensor,          # (4, 4)
    intersect_logits: Tensor,    # (N, 3) ★ 可微
    face_axis_ids: Tensor,       # (F_total,) long
    face_voxel_ids: Tensor,      # (F_total,) long
    rast_res: int,
    compute_normal: bool,
    ref_normals_all: Tensor = None,  # (N, 3) 全局 ref normals（世界坐标系，★ 对 vertices 可微）
) -> Tuple[Tensor, Tensor]:
    """编排单个 chunk 的全部可微计算（被 checkpoint 包裹）

    梯度路径:
      路径 1: subs → occupancy_diff → voxel_normal → grid_sample → face_normal → pixel_normal
      路径 2: intersect_logits → sigmoid → alpha
      路径 3: vertices → centroids → grid_sample(query_pts) → face_normal → pixel_normal

    Returns:
        layer_alpha:  (1, H, W, 1)
        layer_normal: (1, H, W, 3) camera space
    """
    # ---- ① 面重心 + active voxels ----
    if compute_normal:
        # 面重心（world space, ★ 对 vertices 可微）
        centroids_world = (
            vertices[faces_chunk[:, 0]] +
            vertices[faces_chunk[:, 1]] +
            vertices[faces_chunk[:, 2]]
        ) / 3.0  # (F_c, 3)
        centroids_voxel = (centroids_world - origin) / voxel_size  # (F_c, 3)

        # 查找 active voxels（no_grad 离散操作）
        active_voxel_ids, active_coords = _find_active_voxels_for_chunk(
            centroids_voxel.detach(), coords, voxel_resolution,
        )  # active_voxel_ids: (K,), active_coords: (K, 3)

        # ---- ② 参考法线 + 26-neighbor voxel normal ----
        # 几何 vertex normal 作为方向引导（detached，不影响 subs 梯度路径）
        # flexible_dual_grid_to_mesh 绕序全局一致，无需翻转
        ref_normal = ref_normals_all[active_voxel_ids]  # (K, 3) 世界坐标系

        voxel_normals_world = compute_voxel_normal(
            active_coords, subs, ref_normal, voxel_resolution,
        )  # (K, 3) world space, ★ subs 可微

        # ---- ③ grid_sample: voxel normal → face normal ----
        face_normals_world = _sample_face_normals(
            voxel_normals_world, active_coords,
            centroids_voxel, voxel_resolution,
        )  # (F_c, 3) world space, ★ 双可微

        # ---- ④ gather + world→cam + per-pixel flip ----
        layer_normal = _gather_pixel_normal_and_flip(
            face_normals_world, centroids_world,
            rast, extrinsics, rast_res,
        )  # (1, H, W, 3) camera space
    else:
        layer_normal = torch.zeros(
            1, rast_res, rast_res, 3, device=rast.device,
        )  # (1, H, W, 3)

    # ---- ⑤ alpha ----
    layer_alpha = _compute_chunk_alpha(
        rast, face_offset,
        intersect_logits, face_axis_ids, face_voxel_ids,
    )  # (1, H, W, 1)

    return layer_alpha, layer_normal


# =============================================================================
# 渲染器类
# =============================================================================

class Hybrid26NormalRenderer:
    """混合 Normal 渲染器（重心采样版）

    使用面重心在 voxel 网格上 grid_sample per-face normal，
    26-neighbor occupancy 差分（对 subs 可微）+ per-face alpha（对 intersect_logits 可微）。
    面重心对 mesh_vertices（进而对 dual_vertices）可微。

    所有 voxel normal 计算和 grid_sample 均在 chunk 循环内，
    被 gradient checkpoint 包裹，最大程度节省显存。

    rendering_options:
        resolution: 渲染分辨率
        near/far: 近远裁剪面
        ssaa: 超采样倍率
        grad_checkpoint: 是否启用 gradient checkpoint
    """

    def __init__(self, rendering_options: dict = {}, device: str = "cuda"):
        self.rendering_options = edict({
            "resolution": 512,
            "near": 0.1,
            "far": 100.0,
            "ssaa": 1,
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
        intersect_logits: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        voxel_resolution: int,
        return_types: List[str] = ["normal", "mask"],
    ) -> edict:
        """渲染可微法向量

        流水线:
          0. 预计算：face_axis_ids, face_voxel_ids（无梯度）+ ref_normals_all（可微）
          1. 坐标变换 → vertices_clip, vertices_cam, vertices_batch
          2+3. 分 chunk 光栅化 + checkpoint(_compute_one_chunk)
          4. 输出组装
          5. SSAA 下采样

        Args:
            mesh: 含 .vertices (V, 3) 和 .faces (F, 3) 的对象
            subs: 多层 sub_logits（用于 26-neighbor 差分）
            coords: (N, 3) voxel 整数坐标
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

        # ============ Phase 0: 预计算 ============
        with torch.no_grad():
            face_axis_ids, face_voxel_ids = recover_face_axis_and_voxel(
                faces, coords, voxel_resolution,
            )  # face_axis_ids: (F,), face_voxel_ids: (F,)

        # 全局 ref normals：face normal scatter 到源 voxel（★ 对 vertices 可微）
        ref_normals_all = compute_ref_normals_from_faces(
            vertices, faces, face_voxel_ids, coords.shape[0],
        )  # (N, 3) 世界坐标系

        # ============ Phase 1: 坐标变换 ============
        vertices_clip, vertices_cam, vertices_batch = self._transform_vertices(
            vertices, extrinsics, intrinsics, return_types)

        # ============ Phase 2+3: 分 chunk 光栅化 + 可微计算 ============
        out_dict = self._rasterize_and_render(
            vertices_clip, vertices_cam, vertices,
            faces, intersect_logits, face_axis_ids, face_voxel_ids,
            coords, subs, extrinsics, voxel_resolution, return_types,
            ref_normals_all=ref_normals_all)

        # ============ Phase 5: 下采样 ============
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
    # Phase 2+3: 分 chunk 光栅化 + 可微计算
    # ------------------------------------------------------------------

    def _rasterize_and_render(
        self, vertices_clip, vertices_cam, vertices,
        faces, intersect_logits, face_axis_ids, face_voxel_ids,
        coords, subs, extrinsics, voxel_resolution, return_types,
        ref_normals_all: Tensor = None,
    ):
        """分 chunk 光栅化 + checkpoint(_compute_one_chunk)

        每个 chunk:
        1. dr.rasterize（不可微，checkpoint 外）
        2. _compute_one_chunk（可微，checkpoint 内）：
           面重心 → active voxels → ref_normal 引导 → voxel normal →
           grid_sample → gather + flip → pixel_normal_cam + alpha

        跨 chunk 用 per-pixel z-buffer 归并。

        Returns:
            out_dict: edict，包含 normal/mask/depth
        """
        rast_res = self.rendering_options["resolution"] * self.rendering_options["ssaa"]
        use_ckpt = self.rendering_options["grad_checkpoint"]
        voxel_size = 1.0 / voxel_resolution
        origin = torch.tensor([-0.5, -0.5, -0.5], device=self.device)  # (3,)
        compute_normal = ("normal" in return_types)

        num_faces = faces.shape[0]
        num_chunks = (num_faces + _MAX_FACES_PER_CHUNK - 1) // _MAX_FACES_PER_CHUNK

        chunk_depths: list = []      # List[Tensor(1, H, W)]     非可微，z-buffer 归并
        chunk_alphas: list = []      # List[Tensor(1, H, W, 1)]  ★ 可微
        chunk_normals: list = []     # List[Tensor(1, H, W, 3)]  ★ 可微
        chunk_cam_depths: list = []  # List[Tensor(1, H, W, 1)]

        for chunk_idx in range(num_chunks):
            off = chunk_idx * _MAX_FACES_PER_CHUNK
            size = min(_MAX_FACES_PER_CHUNK, num_faces - off)
            faces_k = faces[off: off + size]  # (F_chunk, 3)

            # dr.rasterize（不可微，checkpoint 外）
            rast, _ = dr.rasterize(
                self.glctx, vertices_clip, faces_k, (rast_res, rast_res)
            )  # (1, H, W, 4)

            # z-buffer depth（非可微，仅用于跨 chunk 归并）
            depth = rast[..., 2].detach().clone()  # (1, H, W)
            depth[rast[..., 3] == 0] = float('inf')
            chunk_depths.append(depth)

            # cam-space depth（用于 depth 输出）
            if "depth" in return_types and vertices_cam is not None:
                cam_d = dr.interpolate(
                    vertices_cam[..., 2:3].contiguous(), rast, faces_k
                )[0]  # (1, H, W, 1)
                chunk_cam_depths.append(cam_d)

            # 核心可微计算（checkpoint 包裹）
            fn = _compute_one_chunk
            args = (
                rast, faces_k, off,
                vertices, coords, origin, voxel_size, voxel_resolution,
                subs, extrinsics,
                intersect_logits, face_axis_ids, face_voxel_ids,
                rast_res, compute_normal,
                ref_normals_all,
            )
            alpha, normal = (
                checkpoint(fn, *args, use_reentrant=False)
                if use_ckpt else fn(*args)
            )
            chunk_alphas.append(alpha)    # (1, H, W, 1)
            chunk_normals.append(normal)  # (1, H, W, 3)

        # ---- 跨 chunk z-buffer 归并 ----
        if not chunk_depths:
            alpha_acc = torch.zeros(
                1, rast_res, rast_res, 1, device=self.device)  # (1, H, W, 1)
            normal_acc = torch.zeros(
                1, rast_res, rast_res, 3, device=self.device)  # (1, H, W, 3)
            depth_out = torch.zeros(
                1, rast_res, rast_res, 1, device=self.device)  # (1, H, W, 1)
        elif len(chunk_depths) == 1:
            alpha_acc = chunk_alphas[0]                          # (1, H, W, 1)
            normal_acc = alpha_acc * chunk_normals[0]            # (1, H, W, 3)
            depth_out = (
                chunk_cam_depths[0] if chunk_cam_depths
                else torch.zeros(1, rast_res, rast_res, 1, device=self.device)
            )  # (1, H, W, 1)
        else:
            stacked_d = torch.stack(chunk_depths)                # (K, 1, H, W)
            closest = stacked_d.argmin(dim=0)                    # (1, H, W)
            idx = closest.unsqueeze(0).unsqueeze(-1)             # (1, 1, H, W, 1)

            alpha_acc = torch.stack(chunk_alphas).gather(
                0, idx).squeeze(0)                               # (1, H, W, 1)
            normal_merged = torch.stack(chunk_normals).gather(
                0, idx.expand(-1, -1, -1, -1, 3)).squeeze(0)    # (1, H, W, 3)
            normal_acc = alpha_acc * normal_merged               # (1, H, W, 3)

            if chunk_cam_depths:
                depth_out = torch.stack(chunk_cam_depths).gather(
                    0, idx).squeeze(0)                           # (1, H, W, 1)
            else:
                depth_out = torch.zeros(
                    1, rast_res, rast_res, 1, device=self.device)  # (1, H, W, 1)

        # ---- Phase 4: 输出组装 ----
        return self._assemble_output(
            normal_acc, alpha_acc, depth_out, return_types)

    # ---- 输出组装 ----

    @staticmethod
    def _assemble_output(
        normal_acc: Tensor,    # (1, H, W, 3) alpha-weighted normal (cam space)
        alpha_acc: Tensor,     # (1, H, W, 1)
        depth_out: Tensor,     # (1, H, W, 1)
        return_types: List[str],
    ) -> edict:
        """将 normal / alpha / depth 组装为 out_dict"""
        out_dict = edict()

        if "normal" in return_types:
            safe_alpha = alpha_acc.clamp(min=1e-6)                  # (1, H, W, 1)
            composited = normal_acc / safe_alpha                    # (1, H, W, 3)
            composited = F.normalize(composited, dim=-1, eps=1e-6)  # (1, H, W, 3)
            composited = -composited * 0.5 + 0.5                     # (1, H, W, 3) → [0, 1]（与 MeshRenderer 一致）
            out_dict["normal"] = composited                         # (1, H, W, 3)

        if "mask" in return_types:
            out_dict["mask"] = alpha_acc  # (1, H, W, 1)

        if "depth" in return_types:
            out_dict["depth"] = depth_out  # (1, H, W, 1)

        return out_dict

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
