#!/usr/bin/env python
"""
精确测试 CFG Rescale 中 std 计算的差异。

目标：
1. 验证 VarLenTensor.std(dim=[1]) 的真实行为
2. 对比我们的 varlen_std 实现
3. 定位并修复差异

关键发现：
VarLenTensor.std(dim=[1]) 的行为：
1. feats.mean(dim=1) -> (N,) 或 (N, 1)
2. segment_reduce('mean', seqlen) -> (B,) 或 (B, 1)  # 每个 batch 一个值
3. std = sqrt(mean2 - mean^2)

而我们的 varlen_std 只是简单的 dim=1 mean，没有 segment_reduce！
"""

import os
import sys
import torch
import numpy as np

# =====================================================================
# 设置参考代码路径
# =====================================================================
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

from trellis2.modules.sparse import SparseTensor


def print_separator(title: str, char: str = "=", width: int = 60):
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def test_varlen_std():
    """测试 VarLenTensor.std(dim=[1]) 的真实行为"""
    print_separator("测试 VarLenTensor.std(dim=[1]) 的行为")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据: 2 个 batch，每个 batch 有不同数量的 token
    # Batch 0: 100 tokens, Batch 1: 150 tokens
    batch_size = 2
    n_tokens_0 = 100
    n_tokens_1 = 150
    n_total = n_tokens_0 + n_tokens_1  # 250
    n_channels = 64
    
    # 创建坐标 (N, 4): [batch_idx, x, y, z]
    coords = torch.zeros(n_total, 4, dtype=torch.int32, device=device)
    coords[:n_tokens_0, 0] = 0  # batch 0
    coords[n_tokens_0:, 0] = 1  # batch 1
    # 其他坐标可以是任意值
    coords[:, 1:] = torch.randint(0, 16, (n_total, 3), device=device)
    
    # 创建特征 (N, C)
    torch.manual_seed(42)
    feats = torch.randn(n_total, n_channels, device=device)  # (250, 64)
    
    # 创建 SparseTensor
    sparse_tensor = SparseTensor(coords=coords, feats=feats)
    
    print(f"SparseTensor shape: {sparse_tensor.shape}")
    print(f"SparseTensor feats shape: {sparse_tensor.feats.shape}")
    print(f"SparseTensor seqlen: {sparse_tensor.seqlen}")
    print(f"SparseTensor ndim: {sparse_tensor.ndim}")
    
    # 计算 std
    dim = list(range(1, sparse_tensor.ndim))  # [1]
    print(f"\ndim = {dim}")
    
    ref_std = sparse_tensor.std(dim=dim, keepdim=True)
    print(f"\nref_std shape: {ref_std.shape}")
    print(f"ref_std dtype: {ref_std.dtype}")
    print(f"ref_std:\n{ref_std}")
    
    # =====================================================================
    # 测试我们的 varlen_std 实现
    # =====================================================================
    print_separator("我们的 varlen_std 实现")
    
    def our_varlen_std(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
        """当前 trellis2.py 中的实现"""
        mean = x.mean(dim=dim, keepdim=True)  # (N, 1)
        mean2 = (x ** 2).mean(dim=dim, keepdim=True)  # (N, 1)
        std = (mean2 - mean ** 2).sqrt()  # (N, 1)
        if not keepdim:
            std = std.squeeze(dim)
        return std
    
    our_std = our_varlen_std(feats, dim=1, keepdim=True)
    print(f"our_std shape: {our_std.shape}")  # 应该是 (N, 1)
    print(f"our_std[:5]:\n{our_std[:5].squeeze()}")
    print(f"our_std[n_tokens_0:n_tokens_0+5]:\n{our_std[n_tokens_0:n_tokens_0+5].squeeze()}")
    
    # =====================================================================
    # 正确的实现：需要 segment_reduce
    # =====================================================================
    print_separator("正确的 per-batch std 实现")
    
    def correct_varlen_std(
        x: torch.Tensor, 
        coords: torch.Tensor, 
        dim: int, 
        keepdim: bool = True
    ) -> torch.Tensor:
        """
        正确实现 VarLenTensor.std(dim=dim) 的行为。
        
        VarLenTensor.reduce 的逻辑：
        1. 先对 feats 做 reduce (dim)
        2. 如果 dim 不包含 0，则对结果做 segment_reduce
        
        Args:
            x: (N, C) tensor
            coords: (N, 4) tensor, coords[:, 0] 是 batch index
            dim: 要 reduce 的维度
            keepdim: 是否保持维度
        
        Returns:
            (B, 1) 或 (N, 1) tensor，取决于实现
        """
        batch_indices = coords[:, 0]  # (N,)
        batch_size = int(batch_indices.max().item()) + 1
        
        # 计算每个 batch 的 seqlen
        seqlen = torch.bincount(batch_indices.int())  # (B,)
        
        # 1. feats.mean(dim=1) -> (N,)
        mean_per_token = x.mean(dim=dim)  # (N,)
        
        # 2. segment_reduce('mean', seqlen) -> (B,)
        # 使用 torch.segment_reduce
        batch_mean = torch.segment_reduce(mean_per_token, reduce='mean', lengths=seqlen)  # (B,)
        
        # 同样计算 mean2
        mean2_per_token = (x ** 2).mean(dim=dim)  # (N,)
        batch_mean2 = torch.segment_reduce(mean2_per_token, reduce='mean', lengths=seqlen)  # (B,)
        
        # std = sqrt(mean2 - mean^2)
        batch_std = (batch_mean2 - batch_mean ** 2).sqrt()  # (B,)
        
        if keepdim:
            batch_std = batch_std.unsqueeze(1)  # (B, 1)
        
        print(f"  batch_mean shape: {batch_mean.shape}")
        print(f"  batch_mean2 shape: {batch_mean2.shape}")
        print(f"  batch_std shape: {batch_std.shape}")
        print(f"  batch_std: {batch_std.squeeze()}")
        
        return batch_std
    
    correct_std = correct_varlen_std(feats, coords, dim=1, keepdim=True)
    print(f"\ncorrect_std shape: {correct_std.shape}")  # 应该是 (B, 1)
    
    # =====================================================================
    # 对比
    # =====================================================================
    print_separator("对比结果")
    
    print(f"ref_std shape: {ref_std.shape}")
    print(f"correct_std shape: {correct_std.shape}")
    
    if ref_std.shape == correct_std.shape:
        diff = (ref_std - correct_std).abs().max().item()
        print(f"max_diff: {diff:.6e}")
        if diff < 1e-6:
            print("✅ correct_std 与参考实现完全一致！")
        else:
            print("❌ correct_std 与参考实现存在差异")
            print(f"ref_std: {ref_std.squeeze()}")
            print(f"correct_std: {correct_std.squeeze()}")
    else:
        print("形状不一致，无法直接比较")
        print(f"ref_std: {ref_std.squeeze()}")
        print(f"correct_std: {correct_std.squeeze()}")
    
    # =====================================================================
    # 验证广播机制
    # =====================================================================
    print_separator("验证广播机制")
    
    # 参考实现中 x_0_cfg * (std_pos / std_cfg) 的广播
    # std 是 (B, 1)，x_0_cfg 是 SparseTensor (feats: N, C)
    # 通过 batch_boardcast_map 将 (B, 1) 映射到 (N, 1)
    
    batch_boardcast_map = sparse_tensor.batch_boardcast_map
    print(f"batch_boardcast_map shape: {batch_boardcast_map.shape}")  # (N,)
    print(f"batch_boardcast_map[:10]: {batch_boardcast_map[:10]}")
    print(f"batch_boardcast_map[n_tokens_0-5:n_tokens_0+5]: {batch_boardcast_map[n_tokens_0-5:n_tokens_0+5]}")
    
    # 使用 batch_boardcast_map 将 (B, 1) 扩展到 (N, 1)
    expanded_std = ref_std[batch_boardcast_map]  # (N, 1)
    print(f"\nexpanded_std shape: {expanded_std.shape}")
    print(f"expanded_std[:5]: {expanded_std[:5].squeeze()}")  # 应该都是 batch 0 的 std
    print(f"expanded_std[n_tokens_0:n_tokens_0+5]: {expanded_std[n_tokens_0:n_tokens_0+5].squeeze()}")  # 应该都是 batch 1 的 std
    
    # 验证扩展后的 std 与原始 ref_std 一致
    print(f"\nref_std[0]: {ref_std[0].item():.6f}")
    print(f"ref_std[1]: {ref_std[1].item():.6f}")
    print(f"expanded_std[0]: {expanded_std[0].item():.6f} (should match ref_std[0])")
    print(f"expanded_std[n_tokens_0]: {expanded_std[n_tokens_0].item():.6f} (should match ref_std[1])")


def test_cfg_rescale_difference():
    """测试 CFG Rescale 中 std 计算的差异对最终结果的影响"""
    print_separator("CFG Rescale std 差异影响测试")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    batch_size = 2
    n_tokens_0 = 100
    n_tokens_1 = 150
    n_total = n_tokens_0 + n_tokens_1
    n_channels = 64
    
    coords = torch.zeros(n_total, 4, dtype=torch.int32, device=device)
    coords[:n_tokens_0, 0] = 0
    coords[n_tokens_0:, 0] = 1
    coords[:, 1:] = torch.randint(0, 16, (n_total, 3), device=device)
    
    torch.manual_seed(42)
    x_0_pos_feats = torch.randn(n_total, n_channels, device=device)
    x_0_cfg_feats = torch.randn(n_total, n_channels, device=device)
    
    # 创建 SparseTensor
    x_0_pos_st = SparseTensor(coords=coords, feats=x_0_pos_feats)
    x_0_cfg_st = SparseTensor(coords=coords, feats=x_0_cfg_feats)
    
    guidance_rescale = 0.5
    
    # =====================================================================
    # 参考实现
    # =====================================================================
    print("\n参考实现 (SparseTensor.std):")
    
    std_pos_ref = x_0_pos_st.std(dim=[1], keepdim=True)  # (B, 1)
    std_cfg_ref = x_0_cfg_st.std(dim=[1], keepdim=True)  # (B, 1)
    print(f"std_pos_ref shape: {std_pos_ref.shape}")
    print(f"std_cfg_ref shape: {std_cfg_ref.shape}")
    
    # x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
    # SparseTensor.__mul__ 会通过 batch_boardcast_map 将 (B, 1) 广播到 (N, 1)
    x_0_rescaled_ref = x_0_cfg_st * (std_pos_ref / std_cfg_ref)
    x_0_ref = guidance_rescale * x_0_rescaled_ref + (1 - guidance_rescale) * x_0_cfg_st
    print(f"x_0_ref feats shape: {x_0_ref.feats.shape}")
    
    # =====================================================================
    # 我们当前的实现 (错误的 per-token std)
    # =====================================================================
    print("\n我们当前的实现 (错误的 per-token std):")
    
    def our_varlen_std(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
        mean = x.mean(dim=dim, keepdim=True)
        mean2 = (x ** 2).mean(dim=dim, keepdim=True)
        std = (mean2 - mean ** 2).sqrt()
        if not keepdim:
            std = std.squeeze(dim)
        return std
    
    std_pos_our = our_varlen_std(x_0_pos_feats, dim=1, keepdim=True)  # (N, 1)
    std_cfg_our = our_varlen_std(x_0_cfg_feats, dim=1, keepdim=True)  # (N, 1)
    print(f"std_pos_our shape: {std_pos_our.shape}")
    print(f"std_cfg_our shape: {std_cfg_our.shape}")
    
    x_0_rescaled_our = x_0_cfg_feats * (std_pos_our / (std_cfg_our + 1e-8))  # (N, C)
    x_0_our = guidance_rescale * x_0_rescaled_our + (1 - guidance_rescale) * x_0_cfg_feats
    print(f"x_0_our shape: {x_0_our.shape}")
    
    # =====================================================================
    # 正确的实现 (per-batch std with segment_reduce)
    # =====================================================================
    print("\n正确的实现 (per-batch std with segment_reduce):")
    
    def correct_varlen_std(x, coords, keepdim=True):
        batch_indices = coords[:, 0]
        seqlen = torch.bincount(batch_indices.int())
        
        mean_per_token = x.mean(dim=1)
        batch_mean = torch.segment_reduce(mean_per_token, reduce='mean', lengths=seqlen)
        
        mean2_per_token = (x ** 2).mean(dim=1)
        batch_mean2 = torch.segment_reduce(mean2_per_token, reduce='mean', lengths=seqlen)
        
        batch_std = (batch_mean2 - batch_mean ** 2).sqrt()
        
        if keepdim:
            batch_std = batch_std.unsqueeze(1)
        
        return batch_std  # (B, 1)
    
    std_pos_correct = correct_varlen_std(x_0_pos_feats, coords)  # (B, 1)
    std_cfg_correct = correct_varlen_std(x_0_cfg_feats, coords)  # (B, 1)
    print(f"std_pos_correct shape: {std_pos_correct.shape}")
    print(f"std_cfg_correct shape: {std_cfg_correct.shape}")
    
    # 使用 batch_boardcast_map 将 (B, 1) 广播到 (N, 1)
    batch_boardcast_map = x_0_pos_st.batch_boardcast_map
    std_pos_expanded = std_pos_correct[batch_boardcast_map]  # (N, 1)
    std_cfg_expanded = std_cfg_correct[batch_boardcast_map]  # (N, 1)
    
    x_0_rescaled_correct = x_0_cfg_feats * (std_pos_expanded / (std_cfg_expanded + 1e-8))
    x_0_correct = guidance_rescale * x_0_rescaled_correct + (1 - guidance_rescale) * x_0_cfg_feats
    print(f"x_0_correct shape: {x_0_correct.shape}")
    
    # =====================================================================
    # 对比结果
    # =====================================================================
    print_separator("对比结果")
    
    # 参考 vs 我们当前的实现
    diff_our = (x_0_ref.feats - x_0_our).abs().max().item()
    print(f"参考 vs 我们当前: max_diff = {diff_our:.6e}")
    
    # 参考 vs 正确的实现
    diff_correct = (x_0_ref.feats - x_0_correct).abs().max().item()
    print(f"参考 vs 正确实现: max_diff = {diff_correct:.6e}")
    
    if diff_correct < 1e-5:
        print("\n✅ 正确的实现与参考完全一致！")
        print("💡 需要在 trellis2_cfg 中使用 segment_reduce 实现 per-batch std")
    else:
        print("\n⚠️ 正确的实现与参考仍有差异，需要进一步调查")


if __name__ == "__main__":
    test_varlen_std()
    test_cfg_rescale_difference()









