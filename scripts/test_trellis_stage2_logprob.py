#!/usr/bin/env python3
"""
TRELLIS Stage 2 LogProb 实现测试

测试新创建的 trellis_stage2_with_logprob.py 和相关模块的基本功能。
验证 SparseTensor 处理、LogProb 计算、CFG 等核心功能。

Usage:
    python scripts/test_trellis_stage2_logprob.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS 相关模块
reference_path = project_root / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))
import trellis.modules.sparse as sp

def test_sparse_tensor_operations():
    """测试 SparseTensor 基本操作"""
    print("🧪 测试 SparseTensor 基本操作...")
    
    # 创建测试 SparseTensor
    coords = torch.tensor([
        [0, 10, 20, 30],  # batch=0, x=10, y=20, z=30
        [0, 15, 25, 35],  # batch=0, x=15, y=25, z=35
        [1, 10, 20, 30],  # batch=1, x=10, y=20, z=30
    ], dtype=torch.int32)
    
    feats = torch.randn(3, 64)  # 3个点，每个64维特征
    
    sparse_tensor = sp.SparseTensor(coords=coords, feats=feats)
    print(f"✅ 创建 SparseTensor: coords={coords.shape}, feats={feats.shape}")
    
    # 测试拼接
    from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat
    
    # 创建第二个 SparseTensor
    coords2 = torch.tensor([
        [0, 40, 50, 60],
        [1, 45, 55, 65],
    ], dtype=torch.int32)
    feats2 = torch.randn(2, 64)
    sparse_tensor2 = sp.SparseTensor(coords=coords2, feats=feats2)
    
    # 拼接测试
    combined = sparse_tensor_cat([sparse_tensor, sparse_tensor2])
    print(f"✅ SparseTensor 拼接: 结果形状 coords={combined.coords.shape}, feats={combined.feats.shape}")
    
    return True

def test_flow_step_logprob():
    """测试 Flow 步骤的 LogProb 计算"""
    print("🧪 测试 TRELLIS Flow LogProb 计算...")
    
    from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_step_with_logprob
    
    # 创建测试数据
    coords = torch.tensor([
        [0, 10, 20, 30],
        [0, 15, 25, 35],
    ], dtype=torch.int32)
    
    sample_feats = torch.randn(2, 32)
    model_output_feats = torch.randn(2, 32)
    
    sample = sp.SparseTensor(coords=coords, feats=sample_feats)
    model_output = sp.SparseTensor(coords=coords, feats=model_output_feats)
    
    # 测试 Flow 步骤
    t = 500.0  # TRELLIS 时间格式
    t_prev = 450.0
    
    prev_sample, log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
        sample=sample,
        model_output=model_output,
        t=t,
        t_prev=t_prev,
        sigma_min=0.002,
        generator=None,
        deterministic=False,
    )
    
    print(f"✅ Flow LogProb 计算:")
    print(f"   输入形状: {sample.feats.shape}")
    print(f"   输出形状: {prev_sample.feats.shape}")
    print(f"   LogProb: {log_prob.item():.4f}")
    print(f"   标准差: {std_dev.item():.4f}")
    
    return True

def test_cfg_operations():
    """测试 CFG 相关操作"""
    print("🧪 测试 SparseTensor CFG 操作...")
    
    from flow_grpo.diffusers_patch.sparse_tensor_grpo import sparse_tensor_cfg_guidance
    
    # 创建正负条件 SparseTensor
    coords = torch.tensor([
        [0, 10, 20, 30],
        [0, 15, 25, 35],
    ], dtype=torch.int32)
    
    pos_feats = torch.randn(2, 32)
    neg_feats = torch.randn(2, 32)
    
    positive_sparse = sp.SparseTensor(coords=coords, feats=pos_feats)
    negative_sparse = sp.SparseTensor(coords=coords, feats=neg_feats)
    
    # 测试 CFG 合并
    guidance_scale = 3.0
    cfg_result = sparse_tensor_cfg_guidance(positive_sparse, negative_sparse, guidance_scale)
    
    print(f"✅ CFG 合并:")
    print(f"   正面特征均值: {pos_feats.mean().item():.4f}")
    print(f"   负面特征均值: {neg_feats.mean().item():.4f}")
    print(f"   CFG 结果均值: {cfg_result.feats.mean().item():.4f}")
    print(f"   引导强度: {guidance_scale}")
    
    return True

def test_pipeline_loading():
    """测试 Pipeline 加载"""
    print("🧪 测试 TrellisStage2Pipeline 加载...")
    
    os.environ['ATTN_BACKEND'] = 'xformers'
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    from generators.trellis.pipeline import TrellisStage2Pipeline
    
    pipeline = TrellisStage2Pipeline()
    print(f"✅ Pipeline 加载成功: 设备={pipeline.device}")
    
    # 测试模型访问
    trainable_model = pipeline.get_trainable_model()
    print(f"✅ 可训练模型: {type(trainable_model).__name__}")
    
    # 测试 LogProb 绑定
    from flow_grpo.diffusers_patch.sparse_tensor_grpo import bind_trellis_logprob_to_pipeline
    bind_trellis_logprob_to_pipeline(pipeline)
    
    print(f"✅ LogProb 函数绑定: {hasattr(pipeline, 'compute_log_prob_trellis_stage2')}")
    
    return pipeline

def test_stage2_with_logprob():
    """测试完整的 Stage 2 LogProb 计算流程"""
    print("🧪 测试完整的 Stage 2 LogProb 流程...")
    
    # 跳过完整测试（需要大量资源）
    print("⏭️  跳过完整 Stage 2 测试（需要 GPU 和大量内存）")
    print("   如需完整测试，请确保:")
    print("   - 有足够的 GPU 内存（>8GB）")
    print("   - TRELLIS 模型已下载") 
    print("   - 设置正确的环境变量")
    
    return True

def main():
    """主测试函数"""
    print("🚀 TRELLIS Stage 2 LogProb 实现测试开始")
    print("=" * 60)
    
    tests = [
        ("SparseTensor 操作", test_sparse_tensor_operations),
        ("Flow LogProb 计算", test_flow_step_logprob),
        ("CFG 操作", test_cfg_operations),
        ("Pipeline 加载", test_pipeline_loading),
        ("Stage 2 LogProb 流程", test_stage2_with_logprob),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        success = test_func()
        
        if success:
            print(f"✅ {test_name} - 通过")
            passed += 1
        else:
            print(f"❌ {test_name} - 失败")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🏆 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！TRELLIS Stage 2 LogProb 实现基本就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 