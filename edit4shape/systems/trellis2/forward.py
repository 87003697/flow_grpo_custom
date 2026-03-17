"""
Trellis2 共享前向传播 & 渲染 & 评估函数。

本模块提取自 shape.py / tex.py / tex_autograd.py / shape_autograd.py，
包含被多个训练入口共享的计算函数：

- decode_and_render_normal: Shape slat → Mesh → Normal 渲染
- decode_and_render_pbr:    Tex slat + Meshes → PBR 渲染
- trellis2_shape_forward:   Shape 完整前向（sampling → rollout → decode+render）
- trellis2_tex_forward:     Tex 完整前向（detach → rollout → decode+render）
- _detach_shape_outputs:    切断 Shape→Tex 计算图依赖（公共辅助）
- dense_sampling_no_grad:   Dense Sampling（no_grad）
- evaluate:                 统一评估函数（支持 shape-only / with_tex 模式）
"""

from __future__ import annotations

import os, sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# TRELLIS.2 参考实现路径
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh
from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh

# 运行时需要的 rollout 函数
from edit4shape.generators.trellis2.rollout import rollout_shape, rollout_tex

# State & System
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import Trellis2System

# 评估工具
from edit4shape.systems.utils import Trellis2VisualIO
from edit4shape.systems.base import EvalModeGuard

# 梯度工具
from edit4shape.systems.utils.loss import gradient_shrink

# Stage 异常
from edit4shape.systems.utils.stage_ops import StageSkipError


# ... existing imports ...
from edit4shape.generators.trellis2.ops.grid_sample3d import grid_sample_3d_differentiable
from edit4shape.generators.trellis2.ops.mesh_voxelize import mesh_to_voxel_indices_cuda
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
import cumesh



# =====================================================================
# 渲染工具函数 - Normal 渲染
# =====================================================================


@torch.no_grad()
def _build_subs_from_coords(
    finest_coords: torch.Tensor,    # (N, 3) int — finest level 所有 voxel 坐标
    original_subs: List[SparseTensor],  # 第一次 forward 产出的 subs（提供完整的 parent 坐标集）
    factor: int = 2,
) -> List[SparseTensor]:
    """
    从 finest-level 的所有 voxel 坐标，逐层折叠构建完整的 subdivision 层级，
    并与 original_subs 合并，确保所有原始 parent 坐标都被保留。

    合并策略：对每层取 original_subs 和 computed_subs 的坐标并集，
    feats 取逐元素 max（保留所有需要展开的 octant）。

    这保证了第二次 forward 时 _align_guide_sub 能找到所有 parent 对应关系。

    公式（来自 SparseSpatial2Channel）：
        subidx = sum(offset[..., i] * factor**i for i in range(DIM))
        即 x_offset * 1 + y_offset * 2 + z_offset * 4

    Args:
        finest_coords: (N, 3) int — finest level 所有 voxel 坐标
        original_subs: List[SparseTensor] — 第一次 forward 的 subs
        factor: 上采样因子（默认 2）

    Returns:
        merged_subs: List[SparseTensor] — 合并后的 subs
    """
    DIM = 3
    device = finest_coords.device
    SUBDIV_VAL = 1.0
    num_levels = len(original_subs)

    # ---- Phase 1: 从 finest coords 逐层折叠，构建 "desired" subs ----
    desired_coords_list = [None] * num_levels   # 每层的 parent coords (P, 3)
    desired_feats_list = [None] * num_levels    # 每层的 subdivision feats (P, 8)
    current_coords = finest_coords              # (N_cur, 3)

    for level in reversed(range(num_levels)):
        parent_coords = current_coords // factor                             # (N_cur, 3)
        offsets = current_coords % factor                                    # (N_cur, 3)
        subidx = sum(
            offsets[..., i] * (factor ** i) for i in range(DIM)
        )                                                                    # (N_cur,)

        parent_unique, inverse = torch.unique(
            parent_coords, dim=0, return_inverse=True)                       # (P, 3), (N_cur,)
        P = parent_unique.shape[0]

        sub_feats = torch.zeros(P, factor ** DIM, device=device)             # (P, 8)
        sub_feats[inverse, subidx.long()] = SUBDIV_VAL

        desired_coords_list[level] = parent_unique                           # (P, 3)
        desired_feats_list[level] = sub_feats                                # (P, 8)

        current_coords = parent_unique

    # ---- Phase 2: 与 original_subs 合并（坐标并集 + feats 逐元素 max）----
    merged_subs = []
    for level in range(num_levels):
        orig = original_subs[level]
        orig_coords = orig.coords[:, 1:]                                     # (N_o, 3)
        orig_feats = orig.feats                                              # (N_o, 8)
        des_coords = desired_coords_list[level]                              # (N_d, 3)
        des_feats = desired_feats_list[level]                                # (N_d, 8)

        # 拼接 + 去重，相同坐标的 feats 取 max
        all_coords = torch.cat([orig_coords, des_coords], dim=0)            # (N_o+N_d, 3)
        all_feats = torch.cat([orig_feats, des_feats], dim=0)               # (N_o+N_d, 8)

        uniq_coords, inverse = torch.unique(
            all_coords, dim=0, return_inverse=True)                          # (U, 3), (N_o+N_d,)
        U = uniq_coords.shape[0]

        # scatter_reduce max：相同坐标的 feats 取逐元素最大值
        merged_feats = torch.zeros(U, factor ** DIM, device=device,
                                   dtype=all_feats.dtype)                    # (U, 8)
        idx_expand = inverse.unsqueeze(1).expand_as(all_feats)               # (N_o+N_d, 8)
        merged_feats.scatter_reduce_(
            0, idx_expand, all_feats, reduce="amax", include_self=True)      # (U, 8)

        # 构建 SparseTensor
        batch_col = torch.zeros(U, 1, dtype=uniq_coords.dtype, device=device)
        coords_4d = torch.cat([batch_col, uniq_coords], dim=-1)             # (U, 4)
        merged_subs.append(SparseTensor(
            merged_feats, coords_4d, scale=orig._scale))

    return merged_subs


@torch.no_grad()
def _cumesh_fill_and_revoxelize(
    mesh_vertices: torch.Tensor,     # (V, 3) float — 有洞 mesh 顶点
    mesh_faces: torch.Tensor,        # (F, 3) int — 有洞 mesh 面
    resolution: int,
    max_hole_perimeter: float = 0.04,
) -> torch.Tensor:
    """
    CuMesh 补洞 + mesh_to_flexible_dual_grid 反推所有 voxel 坐标。

    流程：
    1. CuMesh fill_holes → 无洞 mesh
    2. mesh_to_flexible_dual_grid → 从无洞 mesh 提取所有 voxel 坐标

    Args:
        mesh_vertices: (V, 3) 有洞 mesh 顶点
        mesh_faces: (F, 3) 有洞 mesh 面
        resolution: voxel 分辨率
        max_hole_perimeter: CuMesh fill_holes 的最大洞周长阈值

    Returns:
        all_voxel_coords: (N', 3) int — 补洞后 mesh 对应的所有 voxel 坐标
    """
    # ---- CuMesh fill_holes ----
    _dev = mesh_vertices.device
    cu = cumesh.CuMesh()
    cu.init(mesh_vertices.to(_dev).float(), mesh_faces.to(_dev).int())
    cu.get_edges()
    cu.get_boundary_info()
    if cu.num_boundaries > 0:
        cu.get_vertex_edge_adjacency()
        cu.get_vertex_boundary_adjacency()
        cu.get_manifold_boundary_adjacency()
        cu.read_manifold_boundary_adjacency()
        cu.get_boundary_connected_components()
        cu.get_boundary_loops()
        if cu.num_boundary_loops > 0:
            cu.fill_holes(max_hole_perimeter=max_hole_perimeter)
    filled_verts, filled_faces = cu.read()

    # ---- GPU mesh voxelization: 从 filled mesh 提取所有 voxel 坐标 ----
    voxel_indices = mesh_to_voxel_indices_cuda(
        filled_verts.float(),
        filled_faces.int(),
        grid_size=resolution,
    )
    return voxel_indices  # (N', 3) int32, CUDA


def decode_and_render_normal_filled(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,
    device: torch.device,
    resolution: int,
    decode_only: bool = False,
    bg_color: tuple = (0.5, 0.5, 0.5),
    grad_shrink_scale: float = 1.0,
    max_hole_perimeter: float = 0.04,
) -> Dict[str, Any]:
    """
    解码 shape_slat，CuMesh 补洞后两次 forward 构建无洞 Mesh，渲染 Normal 图。

    采用两次 forward 方案：
    1. 第一次 forward (no_grad, pred_subdiv=True): 快速解码得到 h1 和 subs
    2. 从 h1 构建初始 mesh → CuMesh fill_holes → mesh_to_flexible_dual_grid
       得到补洞后的完整 voxel 坐标集合
    3. 从完整 voxel 坐标构建 full_subs（每层的 subdivision 模式）
    4. 第二次 forward (有梯度, pred_subdiv=False): 使用 full_subs 作为 guide_subs，
       decoder 在所有目标位置展开 voxel，得到完整的 h2
    5. 从 h2 构建无洞 mesh，用 MeshPeeledRenderer 渲染

    新 voxel 的特征由 decoder 网络的 learned weights 计算（非插值），
    intersect_logits 也是网络预测值，梯度路径完整。

    梯度路径:
    路径 1: Loss → pixel_normal → face_normal → vertices → dual_vertices → Decoder
    路径 2: Loss → sigmoid(intersect_logits) → Decoder
    """
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    voxel_margin = decoder.voxel_margin

    # ========== 第一次 forward (no_grad): 预测 subs + 构建初始 mesh ==========
    with torch.no_grad():
        h1, subs = decoder.forward_chunked(
            shape_slat, axis=3, return_subs=True, use_checkpoint=False)

    if h1.feats.shape[0] == 0:
        raise StageSkipError(
            "Shape decoder produced empty output (degenerate latent)")

    # ---- 从 h1 构建初始 mesh（有洞）----
    vertices_sp1 = h1.replace(
        (1 + 2 * voxel_margin) * torch.sigmoid(h1.feats[..., 0:3]) - voxel_margin
    )
    intersected1 = h1.replace(
        torch.ones_like(h1.feats[..., 3:6], dtype=torch.bool)
    )
    quad_lerp1 = h1.replace(F.softplus(h1.feats[..., 6:7]))

    # ---- CuMesh fill_holes + mesh_to_flexible_dual_grid → 完整 voxel 坐标 ----
    # （当前仅支持 batch_size=1）
    v1_0, i1_0, q1_0 = vertices_sp1[0], intersected1[0], quad_lerp1[0]
    init_verts, init_faces = flexible_dual_grid_to_mesh(
        v1_0.coords[:, 1:], v1_0.feats, i1_0.feats, q1_0.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=resolution,
        train=False,  # no_grad 阶段用 train=False 更快
    )
    all_voxel_coords = _cumesh_fill_and_revoxelize(
        init_verts, init_faces, resolution,
        max_hole_perimeter=max_hole_perimeter,
    )  # (N', 3) int

    # ---- 从完整 voxel 坐标 + 原始 subs 构建 merged subs ----
    full_subs = _build_subs_from_coords(
        all_voxel_coords.to(device), subs)

    del h1, vertices_sp1, intersected1, quad_lerp1  # 释放第一次 forward 的中间结果
    torch.cuda.empty_cache()

    # ========== 第二次 forward (有梯度): 使用 full_subs ==========
    # guide_subs 传入后 _execute_upsample_stage1 自动优先使用 guide_sub，
    # 无需 pred_subdiv_override context manager（避免 backward 重算时状态不一致）。
    h = decoder.forward_chunked(
        shape_slat, guide_subs=full_subs, use_checkpoint=True)

    if h.feats.shape[0] == 0:
        raise StageSkipError(
            "Shape decoder second pass produced empty output")

    # ========== 分解 h.feats → 构建 Mesh ==========
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    intersected = h.replace(
        torch.ones_like(h.feats[..., 3:6], dtype=torch.bool)
    )
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

    meshes = []
    for v, i, q in zip(vertices_sp, intersected, quad_lerp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=True,
        )
        meshes.append(Mesh(vertices, faces))

    if decode_only:
        return {"color": None, "subs": list(subs), "meshes": meshes}

    # ========== 渲染 Normal（MeshPeeledRenderer）==========
    extr_all = cameras.w2c.to(device)                                        # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)                                 # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)          # (3,)

    all_normals: List[torch.Tensor] = []

    for i, (mesh_i, h_i) in enumerate(zip(meshes, h)):
        coords_i = h_i.coords[:, 1:]                                        # (N, 3)
        intersect_logits_i = h_i.feats[..., 3:6]                            # (N, 3) ★ 网络预测
        mesh_i = mesh_i.to(device)

        view_normals: List[torch.Tensor] = []
        for v in range(num_views):
            out = renderer.render_normal(
                mesh=mesh_i,
                extrinsics=extr_all[i, v],
                intrinsics=intr_all[i, v],
                intersect_logits=intersect_logits_i,
                coords=coords_i,
                voxel_resolution=resolution,
                return_types=["normal", "mask"],
            )
            normal = out["normal"].permute(1, 2, 0)                         # (H, W, 3)
            mask = out["mask"].unsqueeze(-1).float()                         # (H, W, 1)
            normal = normal * mask + bg * (1 - mask)                         # (H, W, 3)
            view_normals.append(normal)
        all_normals.append(torch.stack(view_normals, dim=0))                 # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)                                # (B, V, H, W, 3)
    if grad_shrink_scale < 1.0:
        normals = gradient_shrink(normals, grad_shrink_scale)                # (B, V, H, W, 3)

    return {"color": normals, "subs": list(subs), "meshes": meshes}

def decode_and_render_normal(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # MeshPeeledRenderer
    device: torch.device,
    resolution: int,
    decode_only: bool = False,
    bg_color: tuple = (0.5, 0.5, 0.5),
    grad_shrink_scale: float = 1.0,  # 渲染梯度缩放（< 1.0 抑制梯度，1.0 = 不缩放）
) -> Dict[str, Any]:
    """
    解码 shape_slat 并使用 MeshPeeledRenderer 渲染 Normal 图。

    始终使用 MeshPeeledRenderer（face_normal + intersect_logits 双路可微）。
    
    梯度路径：
    路径 1: Loss → pixel_normal → face_normal → vertices → dual_vertices
    路径 2: Loss → alpha_compositing
                 → sigmoid(intersect_logits[voxel_id, axis_id]) → Decoder

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: MeshPeeledRenderer
        device: 运行设备
        resolution: Decoder 分辨率
        decode_only: 仅 decode（跳过渲染，Tex 阶段冻结 Shape 时使用）

    Returns:
        dict: {"color": (B, V, H, W, 3) | None, "subs": List[SparseTensor], "meshes": List[Mesh]}
    """
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)

    # ★ 逐层自适应 chunked forward
    h, subs = decoder.forward_chunked(
        shape_slat, axis=3, return_subs=True, use_checkpoint=True)  # h.feats: (N, 7)

    # ── 退化 latent 保护 ──
    # chunked merge 后无有效点时 h.feats.shape[0] == 0，无法构建 Mesh。
    if h.feats.shape[0] == 0:
        raise StageSkipError(
            "Shape decoder produced empty output (degenerate latent)"
        )

    voxel_margin = decoder.voxel_margin

    # ========== 分解 h.feats → 构建可微 Mesh ==========
    # 1. dual_vertices: sigmoid 变换后的顶点偏移（可微）
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    # 2. intersected: 全 True（所有 edge 都参与 mesh 构建，alpha 由 sigmoid(logits) 控制）
    intersected = h.replace(torch.ones_like(h.feats[..., 3:6], dtype=torch.bool))
    # 3. quad_lerp: softplus 变换（可微）
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

    meshes = []
    for v, i, q in zip(vertices_sp, intersected, quad_lerp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=True,       # ★ 启用可微路径（顶点梯度）
        )
        meshes.append(Mesh(vertices, faces))

    # ★ 仅需 decode（Tex 阶段冻结 Shape 时跳过 Normal 渲染）
    if decode_only:
        return {
            "color": None,
            "subs": list(subs),
            "meshes": meshes,
        }

    # ========== 渲染 Normal（MeshPeeledRenderer） ==========
    extr_all = cameras.w2c.to(device)          # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)   # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]

    # Normal 背景（可配，默认灰色）
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)  # (3,)

    all_normals: List[torch.Tensor] = []

    for i, (mesh_i, h_i) in enumerate(zip(meshes, h)):
        coords_i = h_i.coords[:, 1:]                      # (N, 3) voxel 坐标
        # ★ intersect_logits: (N, 3) 保留梯度，由 renderer 内部 gather + sigmoid
        intersect_logits_i = h_i.feats[..., 3:6]           # (N, 3)
        mesh_i = mesh_i.to(device)

        view_normals: List[torch.Tensor] = []
        for v in range(num_views):
            out = renderer.render_normal(
                mesh=mesh_i,
                extrinsics=extr_all[i, v],       # (4, 4)
                intrinsics=intr_all[i, v],       # (3, 3)
                intersect_logits=intersect_logits_i,
                coords=coords_i,
                voxel_resolution=resolution,
                return_types=["normal", "mask"],
            )
            # _downsample 输出 CHW 格式: normal (3,H,W), mask (H,W)
            normal = out["normal"].permute(1, 2, 0)         # (3, H, W) → (H, W, 3)
            mask = out["mask"].unsqueeze(-1).float()        # (H, W) → (H, W, 1)
            normal = normal * mask + bg * (1 - mask)  # (H, W, 3)
            view_normals.append(normal)

        all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)

    # ★ Gradient shrink：抑制 Normal 渲染管线传回的梯度
    if grad_shrink_scale < 1.0:
        normals = gradient_shrink(normals, grad_shrink_scale)  # (B, V, H, W, 3)

    return {
        "color": normals,       # (B, V, H, W, 3) Normal 图
        "subs": list(subs),     # List[SparseTensor]
        "meshes": meshes,       # List[Mesh]
    }


# =====================================================================
# 渲染工具函数 - Normal 渲染（Hybrid26 路径）
# =====================================================================

def decode_and_render_normal_hybrid26(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # Hybrid26NormalRenderer
    device: torch.device,
    resolution: int,
    decode_only: bool = False,
    bg_color: tuple = (0.5, 0.5, 0.5),
    grad_shrink_scale: float = 1.0,  # 渲染梯度缩放（< 1.0 抑制梯度，1.0 = 不缩放）
) -> Dict[str, Any]:
    """
    解码 shape_slat 并使用 Hybrid26NormalRenderer 渲染 Normal 图。

    使用 26-neighbor occupancy 差分计算可微法向量（对 subs 可微）。

    梯度路径：
    路径 1: Loss → pixel_normal → voxel_normal(26-neighbor) → subs → Decoder
    路径 2: Loss → alpha_compositing
                 → sigmoid(intersect_logits[voxel_id, axis_id]) → Decoder
    路径 3: Loss → pixel_normal → grid_sample_3d_differentiable(query_pts)
                 → dr.interpolate → mesh_vertices → dual_vertices → Decoder

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: Hybrid26NormalRenderer
        device: 运行设备
        resolution: Decoder 分辨率
        decode_only: 仅 decode（跳过渲染，Tex 阶段冻结 Shape 时使用）

    Returns:
        dict: {"color": (B, V, H, W, 3) | None, "subs": List[SparseTensor], "meshes": List[Mesh]}
    """
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)

    # ★ 逐层自适应 chunked forward
    h, subs = decoder.forward_chunked(
        shape_slat, axis=3, return_subs=True, use_checkpoint=True)  # h.feats: (N, 7)

    # ── 退化 latent 保护 ──
    # chunked merge 后无有效点时 h.feats.shape[0] == 0，无法构建 Mesh。
    if h.feats.shape[0] == 0:
        raise StageSkipError(
            "Shape decoder produced empty output (degenerate latent)"
        )

    voxel_margin = decoder.voxel_margin

    # ★ 归还 PyTorch reserved-but-unallocated 显存给 CUDA，
    #   供 renderer 的 grid_sample_3d 等原生 CUDA 分配使用
    torch.cuda.empty_cache()

    # ========== 分解 h.feats → 构建可微 Mesh ==========
    # 1. dual_vertices: sigmoid 变换后的顶点偏移（可微）
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    # 2. intersected: 全 True（所有 edge 都参与 mesh 构建，alpha 由 sigmoid(logits) 控制）
    intersected = h.replace(torch.ones_like(h.feats[..., 3:6], dtype=torch.bool))
    # 3. quad_lerp: softplus 变换（可微）
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

    meshes = []
    for v, i, q in zip(vertices_sp, intersected, quad_lerp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=True,       # ★ 启用可微路径：dual_vertices → mesh_vertices → grid_sample_3d_differentiable
        )
        meshes.append(Mesh(vertices, faces))

    # ★ 仅需 decode（Tex 阶段冻结 Shape 时跳过 Normal 渲染）
    if decode_only:
        return {
            "color": None,
            "subs": list(subs),
            "meshes": meshes,
        }

    # ========== 渲染 Normal（Hybrid26NormalRenderer） ==========
    extr_all = cameras.w2c.to(device)          # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)   # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]

    # Normal 背景（可配，默认灰色，与 decode_and_render_normal 一致）
    bg = torch.tensor(bg_color, device=device, dtype=torch.float32)  # (3,)

    all_normals: List[torch.Tensor] = []

    for i, (mesh_i, h_i) in enumerate(zip(meshes, h)):
        subs_i = [sub[i] for sub in subs]                  # ★ 提取 per-batch subs
        coords_i = h_i.coords[:, 1:]                      # (N, 3) voxel 坐标
        # ★ intersect_logits: (N, 3) 保留梯度，由 renderer 内部 gather + sigmoid
        intersect_logits_i = h_i.feats[..., 3:6]           # (N, 3)
        mesh_i = mesh_i.to(device)

        view_normals: List[torch.Tensor] = []
        for v in range(num_views):
            out = renderer.render(
                mesh=mesh_i,
                subs=subs_i,                     # ★ per-batch subs
                coords=coords_i,
                intersect_logits=intersect_logits_i,
                extrinsics=extr_all[i, v],       # (4, 4)
                intrinsics=intr_all[i, v],       # (3, 3)
                voxel_resolution=resolution,
                return_types=["normal", "mask"],
            )
            # 与 decode_and_render_normal 一致：renderer 输出 raw normal，由此处混合背景
            normal = out["normal"]                          # (H, W, 3)
            mask = out["mask"].unsqueeze(-1).float()        # (H, W) → (H, W, 1)
            normal = normal * mask + bg * (1 - mask)        # (H, W, 3)
            view_normals.append(normal)

        all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)

    # ★ Gradient shrink：抑制 Normal 渲染管线传回的梯度
    if grad_shrink_scale < 1.0:
        normals = gradient_shrink(normals, grad_shrink_scale)  # (B, V, H, W, 3)

    return {
        "color": normals,       # (B, V, H, W, 3) Normal 图
        "subs": list(subs),     # List[SparseTensor]
        "meshes": meshes,       # List[Mesh]
    }


# =====================================================================
# 前向传播 - Shape 阶段
# =====================================================================

def trellis2_shape_forward(
    system: Trellis2System,
    state: Trellis2State,
    global_step: int,
    is_training: bool = True,
    render_normal: bool = True,
) -> Dict[str, Any]:
    """
    Shape 阶段前向传播: Dense Sampling → Shape Rollout → Mesh Normal 渲染
    
    使用 MeshRenderer (nvdiffrast) 渲染 MeshWithVoxel，直接获取 normal（支持梯度）。
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) Normal 图
            - "subs": List[SparseTensor]
    
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.features.shape_slat: 挂载 shape latent
        - state.features.subs: 挂载解码中间结果
        - state.regularization: 挂载 reg_loss 和 reg_metric
        - state.views_generated.shape_tensor: 挂载 Normal 渲染图像
    """
    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    
    if system.accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")
    device = system.accelerator.device
    
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    
    # Dense Sampling（no_grad）
    dense_sampling_no_grad(state, system)
    
    # Shape Rollout
    # eval 时使用全局种子（对齐参考实现），train 时使用独立 Generator
    generator = None if not is_training else torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    rollout_shape(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # 根据 renderer type 选择 decode+render 函数
    renderer_type = cfg.shape.renderer.type
    if renderer_type == "hybrid26_peeled":
        decode_fn = decode_and_render_normal_hybrid26
    elif renderer_type == "mesh_peeled":
        decode_fn = decode_and_render_normal
    elif renderer_type == "mesh_filled":
        decode_fn = decode_and_render_normal_filled
    else:
        raise ValueError(f"Unknown shape renderer type: {renderer_type}")
    
    # 解码 + Normal 渲染（decode_only=True 时仅 decode，跳过渲染）
    render_out = decode_fn(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=pipeline.target_resolution,
        decode_only=(not render_normal),
        bg_color=tuple(cfg.shape.renderer.bg_color),
        grad_shrink_scale=cfg.shape.renderer.grad_shrink_scale,
    )
    
    # 挂载结果
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]  # List[Mesh]
    if render_out["color"] is not None:
        state.views_generated.shape_tensor = render_out["color"]  # (B, V, H, W, C) Normal 图
    
    return render_out


# =====================================================================
# 三阶段 Autograd — Phase 函数
# =====================================================================

def _detach_shape_outputs(state: Trellis2State) -> None:
    """
    公共辅助：detach 所有 Shape 产物，切断与 Shape 计算图的依赖。
    
    将条件嵌入、coords、shape_slat、subs、meshes 全部重建为无 grad_fn 的新张量/对象。
    被 shape_frozen_prepare_no_grad（tex_only）和 detach_shape_outputs_for_tex（shape_tex）共用。
    
    Side Effects:
        - state.views_conditioned.cond_*_embed: detach  (B, S, C)
        - state.coords: detach + clone  (N, 4)
        - state.features.shape_slat: 全新 SparseTensor
        - state.features.subs: 全新 List[SparseTensor]（若非 None）
        - state.features.meshes: 全新 List[Mesh]（若非 None）
    """
    # 1. 条件嵌入（Shape/Tex 共用）
    for attr in ('cond_512_embed', 'uncond_512_embed', 'cond_1024_embed', 'uncond_1024_embed'):
        emb = getattr(state.views_conditioned, attr, None)
        if emb is not None:
            setattr(state.views_conditioned, attr, emb.detach())  # (B, S, C)
    
    # 2. coords — 创建全新张量，避免 SparseTensor 缓存关联
    state.coords = state.coords.detach().clone()  # (N, 4)
    
    # 3. shape_slat — 全新 SparseTensor（断开 proxy chain）
    state.features.shape_slat = SparseTensor(
        coords=state.features.shape_slat.coords.detach(),
        feats=state.features.shape_slat.feats.detach(),
    )
    
    # 4. subs — 全新 List[SparseTensor]
    if state.features.subs is not None:
        state.features.subs = [
            SparseTensor(coords=s.coords.detach(), feats=s.feats.detach())
            for s in state.features.subs
        ]
    
    # 5. meshes — vertices/vertex_attrs 来自 shape decoder，需 detach
    if state.features.meshes is not None:
        state.features.meshes = [
            Mesh(
                vertices=m.vertices.detach(),  # (V, 3)
                faces=m.faces,                 # (F, 3) 整数，不需要 detach
                vertex_attrs=m.vertex_attrs.detach() if m.vertex_attrs is not None else None,
            )
            for m in state.features.meshes
        ]


def detach_shape_outputs_for_tex(state: Trellis2State) -> None:
    """Shape→Tex 转接：释放 spatial cache + detach 所有 Shape 产物。"""
    state.release_shape_spatial_cache()
    _detach_shape_outputs(state)


# =====================================================================
# 渲染工具函数 - RGB/PBR 渲染（Phase 2: Tex 训练）
# =====================================================================

def decode_and_render_pbr(
    meshes: List[Any],  # List[Mesh]，来自 Shape 阶段
    tex_slat: SparseTensor,
    subs: List[SparseTensor],
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # MeshPeeledRenderer（nvdiffrast，支持梯度）
    device: torch.device,
    resolution: int = 1024,
    use_checkpointing: bool = False,  # 使用 gradient checkpointing 减少显存
    bg_color: tuple = (1.0, 1.0, 1.0),
    grad_shrink_scale: float = 1.0,  # 渲染梯度缩放（< 1.0 抑制梯度，1.0 = 不缩放）
) -> Dict[str, Any]:
    """
    使用已解码的 Mesh 和 tex_slat 渲染 PBR 图（强制使用 chunked forward）。
    
    只调用 decode_tex（不重复调用 decode_shape），复用 Shape 阶段的 meshes。
    使用 nvdiffrast 可微渲染器进行 IBL 着色，支持梯度反向传播。
    支持 gradient checkpointing 以减少显存使用。
    
    注意：为了支持 checkpointing（要求确定性），SSAO 在 checkpointing 模式下被跳过。
    
    Args:
        meshes: List[Mesh]，来自 Shape 阶段的 decode_shape
        tex_slat: SparseTensor，tex 特征
        subs: List[SparseTensor]，shape 解码中间结果
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: MeshPeeledRenderer（已挂载 envmap）
        device: 运行设备
        resolution: 输出分辨率
        use_checkpointing: 是否使用 gradient checkpointing（默认 True）
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) PBR shaded 图
        }
    """
    
    # ---- 只解码 Tex（复用 Shape 阶段的 meshes） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    # 数值保护（safe_clamp）已在 pipeline.decode_tex 中完成
    # ★ ChunkedDecoderMixin 已注入到 tex_decoder，pipeline.decode_tex 内部会自动使用 chunked forward
    tex_result = pipeline.decode_tex(tex_slat, meshes, subs, resolution)
    mesh_with_voxels = tex_result["mesh_with_voxel"]  # List[MeshWithVoxel]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染辅助函数 ----
    # 注意：MeshPeeledRenderer 的 SSAO 使用随机采样，checkpointing 时需固定种子
    # ★ 使用 fork_rng 隔离 SSAO 的随机种子，避免 torch.manual_seed 污染全局 RNG。
    #   全局 RNG 被污染会导致 DataLoader 中 compute_views_train 的 torch.rand()
    #   每步返回相同值 → 训练视角 yaw 永远固定。
    def _render_pbr(mesh, ext, intr, seed):
        with torch.random.fork_rng(devices=[ext.device] if ext.device.type == 'cuda' else [], enabled=True):
            torch.manual_seed(seed)  # 在 fork 内部设种子，不影响全局 RNG
            out = renderer.render_pbr(mesh, ext, intr, envmap=renderer.envmap, use_envmap_bg=False)
        # ★ 使用多层合成 alpha（front-to-back compositing 的总覆盖率），
        #   而非首层 material alpha（out['alpha']）。
        #   首层 alpha 仅反映最近层的材质透明度，未包含背面半透明贡献。
        alpha = out['alpha_composite']                                    # (H, W)
        bg = torch.tensor(bg_color, device=alpha.device, dtype=torch.float32)  # (3,)
        shaded = out['shaded'] + (1 - alpha.unsqueeze(0)) * bg.view(3, 1, 1)  # (3, H, W)
        return shaded.permute(1, 2, 0)  # (H, W, 3)
    
    # ---- 使用 MeshPeeledRenderer 渲染 PBR（nvdiffrast，支持梯度） ----
    all_colors: List[torch.Tensor] = []
    
    for i, voxel in enumerate(mesh_with_voxels):
        view_colors: List[torch.Tensor] = []
        voxel = voxel.to(device)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            seed = torch.tensor(42 + i * num_views + v)  # 作为 tensor 传入 checkpoint
            
            if use_checkpointing:
                shaded = checkpoint(_render_pbr, voxel, ext_iv, intr_iv, seed, use_reentrant=False)
            else:
                shaded = _render_pbr(voxel, ext_iv, intr_iv, seed)
            
            view_colors.append(shaded)  # (H, W, 3)
        
        all_colors.append(torch.stack(view_colors, dim=0))  # (V, H, W, 3)
    
    colors = torch.stack(all_colors, dim=0)  # (B, V, H, W, 3)

    # ★ Gradient shrink：抑制 PBR 渲染管线传回的梯度
    if grad_shrink_scale < 1.0:
        colors = gradient_shrink(colors, grad_shrink_scale)  # (B, V, H, W, 3)

    return {
        "color": colors,           # (B, V, H, W, 3) PBR shaded 图
    }



# =====================================================================
# 前向传播 - Tex 阶段
# =====================================================================

def trellis2_tex_forward(
    system: Trellis2System,
    state: Trellis2State,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Tex 阶段前向传播: Tex Rollout → PBR Mesh 渲染
    
    前置条件: 
        - state.coords 已挂载（由 trellis2_shape_forward 设置）
        - state.features.shape_slat 已挂载（由 trellis2_shape_forward 设置）
        - state.features.subs 已挂载（由 trellis2_shape_forward 设置）
    
    使用 MeshPeeledRenderer (nvdiffrast) 渲染 MeshWithVoxel，进行 IBL 着色（支持梯度）。
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) PBR shaded 图
    
    Side Effects:
        - state.features.tex_slat: 挂载 tex latent
        - state.regularization: 更新 reg_loss 和 reg_metric
        - state.views_generated.pbr_tensor: 挂载 PBR 渲染图像
    """
    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    if system.accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")
    device = system.accelerator.device
    
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("tex")
    
    # 检查前置条件
    assert state.coords is not None, "state.coords 缺失，请先调用 trellis2_shape_forward"
    assert state.features.shape_slat is not None, "shape_slat 缺失，请先调用 trellis2_shape_forward"
    assert state.features.subs is not None, "subs 缺失，请先调用 trellis2_shape_forward"
    assert state.features.meshes is not None, "meshes 缺失，请先调用 trellis2_shape_forward"
    
    # ★ 彻底切断与 Shape 阶段计算图的依赖
    _detach_shape_outputs(state)
    
    # Tex Rollout
    # eval 时使用全局种子（对齐参考实现），train 时使用独立 Generator
    generator = None if not is_training else torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step + 1000)
    rollout_tex(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # RGB 渲染（使用 Tex 阶段的 renderer，复用 Shape 阶段的 meshes）
    render_out = decode_and_render_pbr(
        state.features.meshes,   # 使用 Shape 阶段解码的 meshes，避免重复 decode_shape
        state.features.tex_slat,
        state.features.subs,
        state.cameras,
        pipeline,
        system.tex.renderer,
        device,
        resolution=pipeline.target_resolution,
        bg_color=tuple(cfg.tex.renderer.bg_color),
        grad_shrink_scale=cfg.tex.renderer.grad_shrink_scale,
    )
    
    state.views_generated.pbr_tensor = render_out["color"]  # (B, V, H, W, C)
    return render_out

# =====================================================================
# Dense Sampling（no_grad）
# =====================================================================

def dense_sampling_no_grad(
    state: Trellis2State,
    system: Trellis2System,
) -> None:
    """
    Dense Sampling（no_grad）。填充 state.coords。
    
    从现有 trellis2_shape_forward 的 Dense Sampling 段提取。
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    ss_params = pipeline.get_ss_params()
    
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_512_embed,       # 始终用 512
            "neg_cond": state.views_conditioned.uncond_512_embed,  # 始终用 512
        }
        coords = pipeline.dense_sampling(
            cond_dict,
            steps=int(ss_params["steps"]),
            resolution=stage_config["ss_resolution"],
        )  # (N, 4)
    state.coords = coords


# =====================================================================
# 评估
# =====================================================================

@torch.no_grad()
def evaluate(
    system: Trellis2System,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
    with_tex: bool = False,
) -> Dict[str, Any]:
    """
    统一评估函数：执行推理并保存可视化结果。

    Args:
        system: 系统组件
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 输出目录
        with_tex: 是否包含 Tex 阶段（tex / shape_tex 模式为 True）

    Returns:
        dict: 评估日志
    """
    if eval_loader is None:
        return {}

    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    accelerator = system.accelerator
    if accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")

    pipeline = system.pipeline
    visual_io = Trellis2VisualIO(visuals_eval_dir, target_h=cfg.render_base.resolution)

    # 根据模式收集需要 eval 的模型
    models_to_eval = [
        system.shape.model,
        pipeline.pipe.models['shape_slat_decoder'],
    ]
    if with_tex:
        models_to_eval += [
            system.tex.model,
            pipeline.pipe.models['tex_slat_decoder'],
        ]
    models_to_eval = [m for m in models_to_eval if m is not None]

    # cond_resolution 取决于最终渲染阶段
    cond_resolution = (
        system.tex.config.cond_resolution if with_tex
        else system.shape.config.cond_resolution
    )

    with EvalModeGuard(*models_to_eval):
        for batch_idx, batch in enumerate(eval_loader):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=pipeline, resolution=cond_resolution)

            render_out = trellis2_shape_forward(
                system, state, global_step, is_training=False
            )

            if with_tex:
                render_out = trellis2_tex_forward(
                    system, state, global_step, is_training=False
                )

            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=False,
            )

    return {"eval_done": 1.0}