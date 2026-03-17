"""
GPU mesh voxelization (AABB-based).

Usage::

    from edit4shape.generators.trellis2.ops.mesh_voxelize import mesh_to_voxel_indices_cuda

    voxel_indices = mesh_to_voxel_indices_cuda(
        vertices,       # (V, 3) float32, CUDA
        faces,          # (F, 3) int32/int64, CUDA
        grid_size=1024,
    )
    # returns (N, 3) int32 on the same CUDA device
"""

from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load as _load_ext

# ---------- JIT compile CUDA extension ----------

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_C = _load_ext(
    name="mesh_voxelize_cuda",
    sources=[os.path.join(_curr_dir, "csrc", "mesh_voxelize.cu")],
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


@torch.no_grad()
def mesh_to_voxel_indices_cuda(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    grid_size: int,
    aabb_min: tuple = (-0.5, -0.5, -0.5),
    aabb_max: tuple = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    GPU mesh → occupied voxel indices (AABB approximation).

    Args:
        vertices: (V, 3) float, CUDA
        faces:    (F, 3) int,   CUDA
        grid_size: uniform grid resolution per axis
        aabb_min:  AABB lower corner
        aabb_max:  AABB upper corner

    Returns:
        (N, 3) int32 unique voxel coordinates, CUDA
    """
    device = vertices.device
    vertices = vertices.float().contiguous()
    faces = faces.int().contiguous()

    aabb = torch.tensor(
        [list(aabb_min), list(aabb_max)],
        dtype=torch.float32, device=device,
    )

    # 1. count candidates per face
    counts = _C.count_face_voxels(vertices, faces, grid_size, aabb)  # (F,) int32

    # 2. exclusive prefix sum → offsets
    counts_i64 = counts.long()
    cumsum = torch.cumsum(counts_i64, dim=0)     # (F,)
    total = cumsum[-1].item() if cumsum.numel() > 0 else 0
    offsets = cumsum - counts_i64                 # exclusive

    if total == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    # 3. write all candidate voxels
    candidates = _C.write_face_voxels(
        vertices, faces, offsets, total, grid_size, aabb)  # (total, 3) int32

    # 4. deduplicate
    unique_voxels = torch.unique(candidates, dim=0)

    return unique_voxels
