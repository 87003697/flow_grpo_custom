#!/usr/bin/env python3
"""
测试 CFG Rescale 中 std 的广播机制
验证 SparseTensor * (B, 1) tensor 的广播是否正确
"""

import sys
sys.path.insert(0, '/home/zhiyuan_ma/code/flow_grpo_custom/_reference_codes/TRELLIS.2')

import torch
from trellis2.modules.sparse import SparseTensor


def test_std_broadcast():
    """测试 SparseTensor.std 和广播机制"""
    
    print("=" * 60)
    print("测试 CFG Rescale 中 std 的广播机制")
    print("=" * 60)
    
    # 创建测试数据
    # 假设 B=2, 每个 batch 有不同数量的点
    batch_0_points = 100  # batch 0 有 100 个点
    batch_1_points = 150  # batch 1 有 150 个点
    N = batch_0_points + batch_1_points  # 总点数
    C = 64  # 通道数
    
    # 创建 coords: (N, 4)，第一列是 batch index
    coords_0 = torch.zeros(batch_0_points, 4, dtype=torch.int32)
    coords_0[:, 0] = 0  # batch 0
    coords_0[:, 1:] = torch.randint(0, 32, (batch_0_points, 3))
    
    coords_1 = torch.zeros(batch_1_points, 4, dtype=torch.int32)
    coords_1[:, 0] = 1  # batch 1
    coords_1[:, 1:] = torch.randint(0, 32, (batch_1_points, 3))
    
    coords = torch.cat([coords_0, coords_1], dim=0)  # (N, 4)
    
    # 创建 feats: (N, C)
    feats = torch.randn(N, C)
    
    # 创建 SparseTensor
    x = SparseTensor(feats=feats, coords=coords)
    
    print(f"\n1. SparseTensor 基本信息:")
    print(f"   - feats.shape: {x.feats.shape}")  # (N, C)
    print(f"   - coords.shape: {x.coords.shape}")  # (N, 4)
    print(f"   - x.shape: {x.shape}")  # (B, C)
    print(f"   - len(x.layout): {len(x.layout)}")  # B
    print(f"   - x.layout: {x.layout}")  # [slice(0, 100), slice(100, 250)]
    
    # 测试 std 计算
    print(f"\n2. 测试 SparseTensor.std():")
    std = x.std(dim=[1], keepdim=True)
    print(f"   - std.shape: {std.shape}")  # 应该是 (B, 1)
    print(f"   - std type: {type(std)}")  # 应该是 torch.Tensor
    print(f"   - std values: {std.flatten().tolist()}")
    
    # 测试广播机制
    print(f"\n3. 测试 SparseTensor * tensor 广播:")
    
    # 模拟 CFG rescale 中的操作
    std_pos = x.std(dim=[1], keepdim=True)  # (B, 1)
    std_cfg = x.std(dim=[1], keepdim=True) + 0.1  # (B, 1)，稍微不同
    
    scale = std_pos / std_cfg  # (B, 1)
    print(f"   - scale.shape: {scale.shape}")  # (B, 1)
    
    # 关键测试：SparseTensor * (B, 1) tensor
    x_scaled = x * scale  # 这应该正确广播
    
    print(f"   - x_scaled type: {type(x_scaled)}")  # 应该是 SparseTensor
    print(f"   - x_scaled.feats.shape: {x_scaled.feats.shape}")  # (N, C)
    print(f"   - x_scaled.shape: {x_scaled.shape}")  # (B, C)
    
    # 验证广播是否正确
    print(f"\n4. 验证广播结果:")
    
    # 手动计算期望结果
    # batch 0 的点应该乘以 scale[0]
    # batch 1 的点应该乘以 scale[1]
    expected_feats = torch.zeros_like(feats)
    expected_feats[:batch_0_points] = feats[:batch_0_points] * scale[0]  # (100, C) * scalar
    expected_feats[batch_0_points:] = feats[batch_0_points:] * scale[1]  # (150, C) * scalar
    
    actual_feats = x_scaled.feats
    
    is_close = torch.allclose(actual_feats, expected_feats, rtol=1e-5, atol=1e-5)
    max_diff = (actual_feats - expected_feats).abs().max().item()
    
    print(f"   - 广播结果正确: {is_close}")
    print(f"   - 最大差异: {max_diff}")
    
    # 测试 batch_boardcast_map
    print(f"\n5. batch_boardcast_map 详情:")
    bbm = x.batch_boardcast_map
    print(f"   - batch_boardcast_map.shape: {bbm.shape}")  # (N,)
    print(f"   - batch_boardcast_map[:5]: {bbm[:5].tolist()}")  # 前5个应该都是0
    print(f"   - batch_boardcast_map[-5:]: {bbm[-5:].tolist()}")  # 后5个应该都是1
    
    # 测试 __elemwise__ 的广播逻辑
    print(f"\n6. 验证 __elemwise__ 广播逻辑:")
    other = scale  # (B, 1)
    other_broadcasted = torch.broadcast_to(other, x.shape)  # (B, C)
    print(f"   - other.shape: {other.shape}")
    print(f"   - x.shape (broadcast target): {x.shape}")
    print(f"   - broadcasted.shape: {other_broadcasted.shape}")
    
    other_expanded = other_broadcasted[bbm]  # (N, C)
    print(f"   - expanded.shape: {other_expanded.shape}")
    
    # 验证 expanded 是否正确
    print(f"   - batch 0 scale value: {scale[0].item():.6f}")
    print(f"   - expanded[0, 0] (batch 0): {other_expanded[0, 0].item():.6f}")
    print(f"   - expanded[99, 0] (batch 0): {other_expanded[99, 0].item():.6f}")
    print(f"   - batch 1 scale value: {scale[1].item():.6f}")
    print(f"   - expanded[100, 0] (batch 1): {other_expanded[100, 0].item():.6f}")
    print(f"   - expanded[-1, 0] (batch 1): {other_expanded[-1, 0].item():.6f}")
    
    print("\n" + "=" * 60)
    if is_close:
        print("✅ CFG Rescale 广播机制验证通过！")
    else:
        print("❌ CFG Rescale 广播机制有问题！")
    print("=" * 60)
    
    return is_close


if __name__ == "__main__":
    test_std_broadcast()








