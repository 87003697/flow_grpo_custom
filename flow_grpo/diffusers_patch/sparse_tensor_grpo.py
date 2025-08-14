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
    TRELLIS Stage 2 单步对数概率计算（单步重算）

    与 Hunyuan3D 的 `compute_log_prob_3d` 一致：只对第 j 步进行前向并计算该步的 log_prob，
    使用采样期观测到的上一时刻样本作为 `observed_prev_sample`，避免整条轨迹的图在显存中累积。
    """
    # 取单步所需的稀疏张量与时间
    latents_seq: list = sample["latents_seq"]  # 长度 steps+1
    current_sparse: sp.SparseTensor = latents_seq[j]
    observed_prev_sparse: sp.SparseTensor = latents_seq[j + 1]

    # 时间序列
    if "t_seq" in sample:
        t_seq = sample["t_seq"]
    else:
        # 回退：根据当前配置重建（须确保与采样期一致）
        num_inference_steps = int(getattr(config, 'num_inference_steps', 50))
        rescale_t = float(getattr(config, 'rescale_t', 1.0))
        import numpy as np
        t_seq = np.linspace(1.0, 0.0, num_inference_steps + 1) * 1000
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)

    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])

    # 图像条件（patch级）
    image_idx = int(sample.get("image_idx", 0))
    if 'cond' in image_conds:
        cond_patches = image_conds['cond'][image_idx:image_idx+1]
        neg_patches = image_conds.get('neg_cond', None)
        if neg_patches is not None:
            neg_patches = neg_patches[image_idx:image_idx+1]
    else:
        pos_vec = image_conds['positive'][image_idx:image_idx+1]
        neg_vec = image_conds.get('negative', None)
        cond_patches = pos_vec.unsqueeze(1)
        neg_patches = neg_vec.unsqueeze(1) if neg_vec is not None else None

    guidance_scale = float(getattr(config, 'guidance_scale', 3.0))
    do_cfg = guidance_scale > 1.0 and neg_patches is not None

    sigma_min = float(getattr(config, 'sigma_min', 0.002))
    deterministic = bool(getattr(config, 'deterministic', False))

    # 模型前向（单步）
    slat_flow_model = pipeline.get_trainable_model()
    t_tensor = torch.tensor([t], device=current_sparse.coords.device, dtype=torch.float32)

    if do_cfg:
        neg_output = slat_flow_model(current_sparse, t_tensor, neg_patches)
        pos_output = slat_flow_model(current_sparse, t_tensor, cond_patches)
        cfg_feats = neg_output.feats + guidance_scale * (pos_output.feats - neg_output.feats)
        model_output = sp.SparseTensor(coords=current_sparse.coords, feats=cfg_feats)
    else:
        model_output = slat_flow_model(current_sparse, t_tensor, cond_patches)

    # 单步 Flow + LogProb（使用观测到的 prev 作为对数似然的目标）
    prev_sample, log_prob, prev_sample_mean, std_dev = trellis_flow_step_with_logprob(
        sample=current_sparse,
        model_output=model_output,
        t=t,
        t_prev=t_prev,
        sigma_min=sigma_min,
        generator=None,
        deterministic=deterministic,
        observed_prev_sample=observed_prev_sparse,
    )

    # 计算 per-step KL（参考 SD3/Hunyuan3D）：
    # KL ≈ E[ (μ - μ_ref)^2 / (2 σ^2) ]，其中 μ=prev_sample_mean，σ=std_dev
    kl_div = torch.zeros_like(log_prob)
    if float(getattr(config, 'kl_reward', 0.0)) > 0.0 and not deterministic:
        if hasattr(slat_flow_model, 'disable_adapter'):
            # 教师前向（禁用LoRA适配器）
            with slat_flow_model.disable_adapter():
                if do_cfg:
                    neg_ref = slat_flow_model(current_sparse, t_tensor, neg_patches)
                    pos_ref = slat_flow_model(current_sparse, t_tensor, cond_patches)
                    cfg_ref_feats = neg_ref.feats + guidance_scale * (pos_ref.feats - neg_ref.feats)
                    model_output_ref = sp.SparseTensor(coords=current_sparse.coords, feats=cfg_ref_feats)
                else:
                    model_output_ref = slat_flow_model(current_sparse, t_tensor, cond_patches)
            _, _, prev_mean_ref, std_ref = trellis_flow_step_with_logprob(
                sample=current_sparse,
                model_output=model_output_ref,
                t=t,
                t_prev=t_prev,
                sigma_min=sigma_min,
                generator=None,
                deterministic=deterministic,
                observed_prev_sample=observed_prev_sparse,
            )
            diff = prev_sample_mean.feats - prev_mean_ref.feats
            denom = (std_dev + 1e-8) ** 2
            kl_scalar = (diff.pow(2).mean() / (2.0 * denom)).unsqueeze(0)
            kl_div = kl_scalar.to(log_prob.dtype)

    return prev_sample, log_prob, kl_div


def compute_log_prob_trellis_stage2_batched(
    pipeline: TrellisStage2Pipeline,
    samples: List[Dict],
    j: int,
    image_conds_list: List[Dict[str, torch.Tensor]],
    config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched 版本的单步对数概率计算。

    - 将多个样本在第 j 步的 SparseTensor 合并为 batched SparseTensor，一次前向获得 (B,) 的 log_prob。
    - KL 计算与单样本版本一致（可选，受 config.train.beta 控制）。
    
    Returns:
        Tuple[log_prob_vec, kl_vec]，形状均为 (B,)
    """
    assert len(samples) == len(image_conds_list), "samples 与 image_conds_list 数量不一致"

    # 组装 batched 当前样本与观测到的上一样本
    current_list = []  # List[sp.SparseTensor]
    prev_obs_list = []  # List[sp.SparseTensor]
    for s in samples:
        lat_seq = s["latents_seq"]  # 长度 steps+1
        current_list.append(lat_seq[j])          # SparseTensor（单样本）
        prev_obs_list.append(lat_seq[j + 1])     # SparseTensor（单样本）

    # 利用现有工具函数拼接为 batched SparseTensor
    batched_current = prepare_sparse_tensor_batch(current_list, batch_size=len(samples))  # batched SparseTensor
    batched_prev_obs = prepare_sparse_tensor_batch(prev_obs_list, batch_size=len(samples))  # batched SparseTensor

    # 时间标量（所有样本相同时间表）
    if "t_seq" in samples[0]:
        t_seq = samples[0]["t_seq"]  # (steps+1,)
    else:
        num_inference_steps = int(getattr(config, 'num_inference_steps', 50))
        rescale_t = float(getattr(config, 'rescale_t', 1.0))
        t_seq = np.linspace(1.0, 0.0, num_inference_steps + 1) * 1000
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)
    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])

    # 条件拼接（按 batch 维度）
    cond_batched = torch.cat([c["cond"] for c in image_conds_list], dim=0)          # (B, P, C)
    neg_cond_batched = None
    if any((c.get("neg_cond", None) is not None) for c in image_conds_list):
        neg_cond_batched = torch.cat([
            (c.get("neg_cond") if c.get("neg_cond") is not None else torch.zeros_like(c["cond"]))
            for c in image_conds_list
        ], dim=0)  # (B, P, C)

    # 模型前向（CFG 按 batch 维执行）
    slat_flow_model = pipeline.get_trainable_model()
    do_cfg = float(getattr(config, 'guidance_scale', 3.0)) > 1.0 and (neg_cond_batched is not None)
    t_tensor = torch.tensor([t] * len(samples), device=batched_current.coords.device, dtype=torch.float32)  # (B,)

    if do_cfg:
        neg_out = slat_flow_model(batched_current, t_tensor, neg_cond_batched)
        pos_out = slat_flow_model(batched_current, t_tensor, cond_batched)
        cfg_feats = neg_out.feats + float(getattr(config, 'guidance_scale', 3.0)) * (pos_out.feats - neg_out.feats)  # (N, C)
        model_output = sp.SparseTensor(coords=batched_current.coords, feats=cfg_feats)
    else:
        model_output = slat_flow_model(batched_current, t_tensor, cond_batched)

    # 单步 Flow+LogProb（使用观测到的上一时刻作为目标）
    _, log_prob_vec, prev_mean, std_vec = trellis_flow_step_with_logprob(
        sample=batched_current,
        model_output=model_output,
        t=t,
        t_prev=t_prev,
        sigma_min=float(getattr(config, 'sigma_min', 0.002)),
        generator=None,
        deterministic=bool(getattr(config, 'deterministic', False)),
        observed_prev_sample=batched_prev_obs,
    )  # log_prob_vec: (B,)

    # KL（可选，按 batch 计算教师输出）
    kl_vec = torch.zeros_like(log_prob_vec)
    if float(getattr(config, 'kl_reward', 0.0)) > 0.0 and not bool(getattr(config, 'deterministic', False)):
        if hasattr(slat_flow_model, 'disable_adapter'):
            with slat_flow_model.disable_adapter():
                if do_cfg:
                    neg_ref = slat_flow_model(batched_current, t_tensor, neg_cond_batched)
                    pos_ref = slat_flow_model(batched_current, t_tensor, cond_batched)
                    cfg_ref_feats = neg_ref.feats + float(getattr(config, 'guidance_scale', 3.0)) * (pos_ref.feats - neg_ref.feats)
                    model_output_ref = sp.SparseTensor(coords=batched_current.coords, feats=cfg_ref_feats)
                else:
                    model_output_ref = slat_flow_model(batched_current, t_tensor, cond_batched)
            _, _, prev_mean_ref, std_ref = trellis_flow_step_with_logprob(
                sample=batched_current,
                model_output=model_output_ref,
                t=t,
                t_prev=t_prev,
                sigma_min=float(getattr(config, 'sigma_min', 0.002)),
                generator=None,
                deterministic=bool(getattr(config, 'deterministic', False)),
                observed_prev_sample=batched_prev_obs,
            )
            diff = prev_mean.feats - prev_mean_ref.feats  # (N, C)
            denom = (std_vec + 1e-8) ** 2                  # (B,)
            # 聚合到 (B,) KL
            kl_list = []
            layout = prev_mean.layout
            for b in range(len(samples)):
                sl = layout[b]
                kl_b = (diff[sl].pow(2).mean() / (2.0 * denom[b])).unsqueeze(0)  # (1,)
                kl_list.append(kl_b)
            kl_vec = torch.cat(kl_list, dim=0)  # (B,)

    return log_prob_vec, kl_vec


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
    
    # 调整每个 SparseTensor 的批次索引：先归一化到 0，保证每个输入 shape[0]==1
    adjusted_list = []
    for batch_idx, sparse_tensor in enumerate(sparse_list):
        adjusted_coords = sparse_tensor.coords.clone()
        adjusted_coords[:, 0] = 0  # 先统一到 0，保证单样本 batch 形状为 1
        
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