"""grid_sample_3d with gradient support for query coordinates.

原版 flex_gemm.ops.grid_sample.grid_sample_3d 只对 feats 有梯度，
本文件的 grid_sample_3d_differentiable 额外支持对 query_pts 的梯度。

策略：
- CUDA hashmap 查找 8 个邻居索引 + CUDA 权重（无需梯度，是离散操作）
- 根据 indices 从 coords 中查出每个邻居的实际整数坐标
- 用 PyTorch 算子重算三线性权重（对 query_pts 可微，不依赖邻居顺序假设）
- PyTorch 算子做加权求和（对 feats 和 weight 都可微）
"""
import torch
from torch import Tensor
from flex_gemm.ops.grid_sample import grid_sample_3d  # 原版（用于 feats 梯度）
from flex_gemm.ops.utils import init_hashmap
from flex_gemm.ops.grid_sample import HASHMAP_RATIO
from flex_gemm import kernels


def grid_sample_3d_differentiable(
    feats: Tensor,       # (N, C) 稀疏 voxel 特征
    coords: Tensor,      # (N, 4) 稀疏坐标 [batch, x, y, z]
    shape: torch.Size,   # [B, C, W, H, D]
    query_pts: Tensor,   # (B, L, 3) 查询坐标 ★ 需要梯度
    mode: str = 'trilinear',
    hashmap_ratio: float = None,   # hashmap 大小 = ratio * N; None 则用全局默认值
) -> Tensor:
    """grid_sample_3d，同时支持对 feats 和 query_pts 的梯度。

    策略：
    - CUDA hashmap 查找 8 个邻居索引（无需梯度，是离散操作）
    - 从 coords 中查出每个邻居的实际整数坐标（不假设邻居排列顺序）
    - PyTorch 重算三线性权重（对 query_pts 可微）
    - PyTorch 做加权求和（对 feats 和 weight 都可微）
    """
    N = coords.shape[0]
    B, L = query_pts.shape[:2]
    C, W, H, D = shape[-4:]

    # ---- Step 1: CUDA hashmap 查找邻居索引（不需要梯度） ----
    _ratio = hashmap_ratio if hashmap_ratio is not None else HASHMAP_RATIO
    with torch.no_grad():
        hashmap_keys, hashmap_vals = init_hashmap(
            shape, int(_ratio * N), coords.device)
        indices, _ = kernels.cuda \
            .hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
                hashmap_keys, hashmap_vals,
                coords.int(), query_pts.detach(), W, H, D,
            )
        # indices: (B, L, 8) uint32 — 8 个邻居在 feats 中的索引
        valid = (indices != 0xFFFFFFFF)           # (B, L, 8)
        # ★ 0xFFFFFFFF → long 后是巨大正数，必须置 0
        indices_long = indices.long()              # (B, L, 8)
        indices_long[~valid] = 0                   # 无效位置置 0，后续 weight=0 不影响结果

        # ---- Step 2: 从 coords 查出每个邻居的实际整数坐标（不假设排列顺序）----
        # coords: (N, 4) = [batch, x, y, z]
        neighbor_xyz = coords[indices_long.reshape(-1), 1:].float()  # (B*L*8, 3)
        neighbor_xyz = neighbor_xyz.reshape(B, L, 8, 3)              # (B, L, 8, 3)

    # ---- Step 3: PyTorch 重算三线性权重（对 query_pts 可微） ----
    # 权重 = prod_d(clamp(1 - |q_d - (neighbor_int_d + 0.5)|, 0))
    # 参考: flex_gemm/ops/grid_sample/grid_sample_torch.py line 116
    # 注意: CUDA kernel 对未找到的邻居可能返回 index=0（而非 INVALID），
    #       此时 neighbor_xyz 来自错误 voxel，|diff| > 1 导致 1-|diff| < 0。
    #       clamp(min=0) 确保权重非负，不影响正确邻居（|diff| ≤ 1）。
    diff = query_pts.unsqueeze(2) - (neighbor_xyz + 0.5)          # (B, L, 8, 3) ★ 可微
    weight = torch.prod(
        (1.0 - torch.abs(diff)).clamp(min=0.0), dim=-1            # (B, L, 8) ★ 可微
    )

    # 无效邻居（INVALID=0xFFFFFFFF）权重置零
    weight = weight * valid.float()               # (B, L, 8)
    weight_sum = weight.sum(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, L, 1)
    weight = weight / weight_sum                  # (B, L, 8) 归一化

    # ---- Step 4: 加权求和（对 feats 和 weight 都可微） ----
    feats_gathered = feats[indices_long.reshape(-1)]  # (B*L*8, C)
    feats_gathered = feats_gathered.reshape(B, L, 8, C)  # (B, L, 8, C)
    output = (weight.unsqueeze(-1) * feats_gathered).sum(dim=2)  # (B, L, C)

    return output
