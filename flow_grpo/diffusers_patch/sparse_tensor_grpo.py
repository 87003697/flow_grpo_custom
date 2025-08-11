#!/usr/bin/env python3
"""
SparseTensor GRPO 适配层

实现 TRELLIS Stage 2 的 SparseTensor 格式的 GRPO 训练支持，
包括对数概率计算、CFG 处理、批量操作等核心功能。

主要功能:
- compute_log_prob_trellis_stage2: Stage 2 对数概率计算核心函数
- SparseTensor CFG 处理: 拼接/分离操作
- 批量 SparseTensor 操作: 支持训练期间的批处理

参考路径:
- Hunyuan3D LogProb: `scripts/train_hunyuan3d.py:181-232` (compute_log_prob_3d)
- TRELLIS SparseTensor: `_reference_codes/TRELLIS/trellis/modules/sparse/basic.py`
- Flow LogProb: `flow_grpo/diffusers_patch/trellis_flow_with_logprob.py`
- SD3 训练对等: `scripts/train_sd3.py:198-231` (def compute_log_prob)
- SD3 Guidance 对等: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`
- SD3 单步对等: `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:17-80`
"""
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS 相关模块  
reference_path = project_root / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))
import trellis.modules.sparse as sp

# 导入项目模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat
from .trellis_flow_with_logprob import trellis_flow_step_with_logprob


def compute_log_prob_trellis_stage2(
    pipeline: TrellisStage2Pipeline,
    sample: Dict,
    j: int,
    image_conds: Dict[str, torch.Tensor],
    config,
    **kwargs
) -> Tuple[sp.SparseTensor, torch.Tensor, torch.Tensor]:
    """
    TRELLIS Stage 2 对数概率计算，处理 SparseTensor 格式
    
    这是 GRPO 训练的核心函数，计算给定样本在当前策略下的对数概率。
    基于 `scripts/train_hunyuan3d.py:181-232` (compute_log_prob_3d) 的逻辑，
    但适配 TRELLIS 的两阶段架构和 SparseTensor 数据结构。
    
    对应 SD3 的实现:
    - `scripts/train_sd3.py:198-231` (def compute_log_prob)
    - 单步概率密度: `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:17-80`
    - CFG 合并: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`
    """
    print(f"🧮 计算样本 {j} 的 TRELLIS Stage 2 LogProb")
    
    # 提取样本信息
    coords = sample['coords']  # shape: (N, 4)
    original_slat = sample['slat']  # sp.SparseTensor
    image_idx = sample.get('image_idx', j)
    
    # 获取对应的图像条件（统一读取 patch 级官方键名）
    # 优先支持 {'cond','neg_cond'}，若不存在则兼容 {'positive','negative'}
    if 'cond' in image_conds:
        cond_patches = image_conds['cond'][image_idx:image_idx+1]       # shape: (1, P, C)
        neg_patches = image_conds.get('neg_cond', None)
        if neg_patches is not None:
            neg_patches = neg_patches[image_idx:image_idx+1]           # shape: (1, P, C)
    else:
        # 兼容早期向量风格（B,C），用于过渡
        cond_vector = image_conds['positive'][image_idx:image_idx+1]   # shape: (1, C)
        neg_vector = image_conds.get('negative', None)
        if neg_vector is not None:
            neg_vector = neg_vector[image_idx:image_idx+1]             # shape: (1, C)
        # 升级为单 patch 伪装（P=1），以复用统一入口 {'main': patches}
        cond_patches = cond_vector.unsqueeze(1)                        # shape: (1, 1, C)
        neg_patches = neg_vector.unsqueeze(1) if neg_vector is not None else None  # (1,1,C)

    # CFG 设置
    guidance_scale = getattr(config, 'guidance_scale', 3.0)
    do_classifier_free_guidance = guidance_scale > 1.0 and neg_patches is not None
    
    # ===========================================
    # Stage 2: SLAT 重新采样 + LogProb 计算
    # ===========================================
    
    # 获取训练参数
    num_inference_steps = getattr(config, 'num_inference_steps', 50)
    sigma_min = getattr(config, 'sigma_min', 0.002)
    rescale_t = getattr(config, 'rescale_t', 1.0)
    deterministic = getattr(config, 'deterministic', False)
    
    # 准备初始噪声（与原始 SLAT 相同的结构）
    noise_feats = torch.randn_like(original_slat.feats)  # shape: (N, C)
    initial_noise = sp.SparseTensor(
        coords=coords,  # 复用相同的坐标结构
        feats=noise_feats
    )
    
    # 获取 SLatFlowModel
    slat_flow_model = pipeline.get_trainable_model()
    
    # 时间步序列（TRELLIS 格式: 1000 → 0）
    t_seq = np.linspace(1.0, 0.0, num_inference_steps + 1) * 1000
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(num_inference_steps)]
    
    # 采样循环
    sample_tensor = initial_noise
    total_log_prob = torch.zeros(1, device=coords.device)
    
    for step_idx, (t, t_prev) in enumerate(t_pairs):
        # 时间步张量
        t_tensor = torch.tensor([t], device=coords.device, dtype=torch.float32)
        
        # ===========================================
        # CFG 模型预测
        # ===========================================
        
        if do_classifier_free_guidance:
            # 需要处理 SparseTensor 的 CFG：分别推理正负，再线性合并
            with torch.no_grad():
                neg_output = slat_flow_model(sample_tensor, t_tensor, {'main': neg_patches})
            with torch.no_grad():
                pos_output = slat_flow_model(sample_tensor, t_tensor, {'main': cond_patches})
            
            # CFG 合并: output = neg + guidance_scale * (pos - neg)
            cfg_output_feats = (
                neg_output.feats + guidance_scale * (pos_output.feats - neg_output.feats)
            )  # shape: (N, C)
            
            model_output = sp.SparseTensor(
                coords=sample_tensor.coords,
                feats=cfg_output_feats
            )
        else:
            # 无 CFG 的直接推理
            with torch.no_grad():
                model_output = slat_flow_model(sample_tensor, t_tensor, {'main': cond_patches})
        
        # ===========================================
        # Flow 步骤 + LogProb 计算
        # ===========================================
        
        prev_sample, step_log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
            sample=sample_tensor,
            model_output=model_output,
            t=t,
            t_prev=t_prev,
            sigma_min=sigma_min,
            generator=None,  # 使用全局随机状态
            deterministic=deterministic,
        )
        
        # 累积对数概率
        total_log_prob += step_log_prob
        
        # 更新样本
        sample_tensor = prev_sample
        
        if step_idx % 10 == 0:
            print(f"   步骤 {step_idx}/{num_inference_steps}: log_prob={step_log_prob.item():.4f}")
    
    # ===========================================
    # KL 散度计算（可选）
    # ===========================================
    
    kl_reward = getattr(config, 'kl_reward', 0.0)
    if kl_reward > 0 and not deterministic:
        # 计算与参考策略的 KL 散度
        # 这里需要实现参考策略的推理，暂时返回零
        kl_div = torch.zeros_like(total_log_prob)
        print(f"   KL 散度计算: kl={kl_div.item():.4f}")
    else:
        kl_div = torch.zeros_like(total_log_prob)
    
    print(f"✅ 样本 {j} LogProb 计算完成: total_log_prob={total_log_prob.item():.4f}")
    
    return sample_tensor, total_log_prob, kl_div


def sparse_tensor_chunk(tensor: sp.SparseTensor, chunks: int) -> List[sp.SparseTensor]:
    """
    SparseTensor 的分块操作，用于 CFG 分离正负条件
    
    基于 torch.chunk 逻辑适配 SparseTensor，保持坐标不变，分割特征维度。
    
    参考: generators/trellis/patches/sparse_tensor_utils.py 中的拼接逻辑
    
    Args:
        tensor: 要分块的 SparseTensor
        chunks: 分块数量
        
    Returns:
        List[sp.SparseTensor]: 分块后的 SparseTensor 列表
    """
    total_points = tensor.coords.shape[0]
    chunk_size = total_points // chunks
    
    chunks_list = []
    for i in range(chunks):
        start_idx = i * chunk_size
        if i == chunks - 1:  # 最后一块包含剩余的所有点
            end_idx = total_points
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_coords = tensor.coords[start_idx:end_idx]  # shape: (chunk_size, 4)
        chunk_feats = tensor.feats[start_idx:end_idx]    # shape: (chunk_size, C)
        
        # 调整坐标的批次索引
        chunk_coords = chunk_coords.clone()
        chunk_coords[:, 0] = i  # 设置新的批次索引
        
        chunk_tensor = sp.SparseTensor(
            coords=chunk_coords,
            feats=chunk_feats
        )
        chunks_list.append(chunk_tensor)
    
    return chunks_list


def sparse_tensor_cfg_guidance(
    positive_sparse: sp.SparseTensor,
    negative_sparse: sp.SparseTensor,
    guidance_scale: float
) -> sp.SparseTensor:
    """
    SparseTensor 的分类器引导（CFG）合并操作
    
    对应 SD3 中的 guidance 合并:
    - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`
      (noise_pred_uncond + w * (noise_pred_text - noise_pred_uncond))
    """
    # 验证坐标结构一致性
    assert torch.allclose(positive_sparse.coords, negative_sparse.coords), \
        "正负条件的坐标结构必须相同"
    assert positive_sparse.feats.shape == negative_sparse.feats.shape, \
        "正负条件的特征维度必须相同"
    
    # CFG 公式计算
    cfg_feats = (
        negative_sparse.feats + guidance_scale * (positive_sparse.feats - negative_sparse.feats)
    )  # shape: (N, C)
    
    # 构造输出 SparseTensor
    cfg_sparse = sp.SparseTensor(
        coords=positive_sparse.coords,  # 使用相同的坐标
        feats=cfg_feats
    )
    
    return cfg_sparse


def prepare_sparse_tensor_batch(
    sparse_list: List[sp.SparseTensor], 
    batch_size: int
) -> sp.SparseTensor:
    """
    准备 SparseTensor 批次，用于批量推理
    
    将多个 SparseTensor 拼接成一个批次，调整坐标的批次索引。
    
    Args:
        sparse_list: SparseTensor 列表
        batch_size: 期望的批次大小
        
    Returns:
        sp.SparseTensor: 批量拼接的 SparseTensor
    """
    if len(sparse_list) != batch_size:
        raise ValueError(f"SparseTensor 列表长度 {len(sparse_list)} 与批次大小 {batch_size} 不匹配")
    
    # 调整每个 SparseTensor 的批次索引
    adjusted_list = []
    for batch_idx, sparse_tensor in enumerate(sparse_list):
        adjusted_coords = sparse_tensor.coords.clone()
        adjusted_coords[:, 0] = batch_idx  # 设置批次索引
        
        adjusted_sparse = sp.SparseTensor(
            coords=adjusted_coords,
            feats=sparse_tensor.feats
        )
        adjusted_list.append(adjusted_sparse)
    
    # 使用现有的拼接函数
    return sparse_tensor_cat(adjusted_list)


def extract_sparse_tensor_from_batch(
    batch_sparse: sp.SparseTensor, 
    batch_idx: int
) -> sp.SparseTensor:
    """
    从批量 SparseTensor 中提取单个样本
    
    Args:
        batch_sparse: 批量 SparseTensor
        batch_idx: 要提取的批次索引
        
    Returns:
        sp.SparseTensor: 提取的单个 SparseTensor
    """
    # 找到属于指定批次的点
    mask = (batch_sparse.coords[:, 0] == batch_idx)  # shape: (N,)
    
    if not mask.any():
        raise ValueError(f"批次索引 {batch_idx} 在 SparseTensor 中不存在")
    
    # 提取坐标和特征
    extracted_coords = batch_sparse.coords[mask]  # shape: (N_i, 4)
    extracted_feats = batch_sparse.feats[mask]    # shape: (N_i, C)
    
    # 重置批次索引为 0
    extracted_coords = extracted_coords.clone()
    extracted_coords[:, 0] = 0
    
    return sp.SparseTensor(
        coords=extracted_coords,
        feats=extracted_feats
    )


def bind_trellis_logprob_to_pipeline(pipeline: TrellisStage2Pipeline):
    """
    将 TRELLIS LogProb 计算函数绑定到 pipeline，类似 hunyuan3d 的模式
    
    SD3 中不需要动态绑定（compute_log_prob 作为训练脚本函数使用）。
    """
    # 绑定核心 LogProb 计算函数
    if not hasattr(pipeline, 'compute_log_prob_trellis_stage2'):
        pipeline.compute_log_prob_trellis_stage2 = types.MethodType(
            compute_log_prob_trellis_stage2, pipeline
        )
        print("✅ 已绑定 compute_log_prob_trellis_stage2 到 pipeline")
    
    # 绑定 SparseTensor 工具函数
    if not hasattr(pipeline, 'sparse_tensor_cfg_guidance'):
        pipeline.sparse_tensor_cfg_guidance = types.MethodType(
            sparse_tensor_cfg_guidance, pipeline
        )
        print("✅ 已绑定 sparse_tensor_cfg_guidance 到 pipeline") 