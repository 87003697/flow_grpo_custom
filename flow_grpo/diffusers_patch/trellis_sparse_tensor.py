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
import os
import sys
import types
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np

# 注入 TRELLIS 官方代码路径
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_TRELLIS_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "TRELLIS")
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

# 直接从官方 trellis 导入
from trellis.modules import sparse as sp  # type: ignore

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

SparseTensor = sp.SparseTensor


# ============================================================================
# sparse_tensor_cat: 从 generators/trellis/patches/sparse_tensor_utils.py 迁移
# ============================================================================
def sparse_tensor_cat(tensors: List[sp.SparseTensor]) -> sp.SparseTensor:
    """SparseTensor的批量拼接操作，用于CFG处理
    
    参考: _reference_codes/TRELLIS/trellis/modules/sparse/basic.py:420-444 (sparse_cat)
    基于TRELLIS官方sparse_cat实现，dim=0时的逻辑
    
    Args:
        tensors (List[sp.SparseTensor]): 要拼接的稀疏张量列表
        
    Returns:
        sp.SparseTensor: 拼接后的稀疏张量
    """
    if not tensors:
        raise ValueError("输入张量列表为空")
    
    if len(tensors) == 1:
        return tensors[0]
    
    # 按照源代码逻辑进行batch维度拼接
    start = 0
    coords = []
    for input_tensor in tensors:
        coords.append(input_tensor.coords.clone().to(torch.int32))  # 形状 (N_i, 4) 确保为int32类型
        coords[-1][:, 0] += start  # 形状 (N_i, 4) 调整 batch 索引
        start += input_tensor.shape[0]  # 标量，更新下一段起始 batch 索引
    
    # 拼接坐标和特征
    combined_coords = torch.cat(coords, dim=0)  # 形状 (sum(N_i), 4)
    combined_feats = torch.cat([input_tensor.feats for input_tensor in tensors], dim=0)  # 形状 (sum(N_i), C)
    
    # 创建新的SparseTensor
    output = sp.SparseTensor(
        coords=combined_coords,  # 形状 (sum(N_i), 4)
        feats=combined_feats,    # 形状 (sum(N_i), C)
    )
    
    return output


@dataclass
class Stage2RuntimeConfig:
    """TRELLIS Stage 2 运行时配置"""
    guidance_scale: float
    deterministic: bool
    compute_kl: bool = False
    noise_level: float = 0.7
    kl_reward: float = 0.0


@dataclass
class Stage1RuntimeConfig:
    """TRELLIS Stage 1 运行时配置（与 Direct3D‑S2 对齐的最小集）"""
    steps: int = 50
    guidance_scale: float = 0.0
    deterministic: bool = False
    compute_kl: bool = False
    noise_level: float = 0.7


def set_trellis_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    steps: int,
    device: Union[str, torch.device] = 'cpu',
    rescale_t: float = 1.0,
) -> FlowMatchEulerDiscreteScheduler:
    """
    使用官方 set_timesteps 接口，同时注入 TRELLIS 官方时间序列。
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device  # 形状: 标量设备
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # 形状: (steps+1,)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)  # 形状: (steps+1,)
    sigma_seq = t_seq / 1000.0  # 形状: (steps+1,)
    scheduler.set_timesteps(
        num_inference_steps=steps,
        device=dev,
        timesteps=t_seq[:-1].tolist(),
        sigmas=sigma_seq[:-1].tolist(),
    )
    scheduler.timesteps = torch.from_numpy(t_seq).to(device=dev, dtype=torch.float32)  # 形状: (steps+1,)
    scheduler.sigmas = torch.from_numpy(sigma_seq).to(device=dev, dtype=torch.float32)  # 形状: (steps+1,)
    scheduler.num_inference_steps = steps  # 形状: 标量
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler


def create_trellis_scheduler(
    steps: int,
    device: Union[str, torch.device] = 'cpu',
    rescale_t: float = 1.0,
) -> FlowMatchEulerDiscreteScheduler:
    """
    创建并配置全新的 TRELLIS scheduler（回兼容旧用法）。
    """
    scheduler = FlowMatchEulerDiscreteScheduler()
    return set_trellis_timesteps(
        scheduler=scheduler,
        steps=steps,
        device=device,
        rescale_t=rescale_t,
    )


def trellis_flow_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    sample: sp.SparseTensor,
    model_output: sp.SparseTensor,
    timestep: float,
    prev_timestep: float,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    observed_prev_sample: Optional[sp.SparseTensor] = None,
    noise_level: float = 0.7,
) -> Tuple[sp.SparseTensor, torch.Tensor, sp.SparseTensor, torch.Tensor]:
    device = sample.feats.device  # 形状: 标量设备
    
    bidx_all = sample.coords[:, 0].to(torch.long)  # 形状: (N_total,)
    unique_batches = torch.unique(bidx_all, sorted=True)  # 形状: (BK,)
    batch_size = int(unique_batches.numel())  # 形状: 标量

    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    t_cur = torch.as_tensor(float(timestep), device=device, dtype=torch.float32)
    t_pre = torch.as_tensor(float(prev_timestep), device=device, dtype=torch.float32)
    step_index = int(scheduler.index_for_timestep(t_cur))
    prev_step_index = int(scheduler.index_for_timestep(t_pre))
    step_index = max(0, min(step_index, int(sigmas.shape[0]) - 1))
    prev_step_index = max(0, min(prev_step_index, int(sigmas.shape[0]) - 1))
    sigma = sigmas[step_index]
    sigma_prev = sigmas[prev_step_index]
    sigma_max = sigmas[1 if int(sigmas.shape[0]) > 1 else 0]

    ones_like_sigma = torch.ones_like(sigma)  # 形状: 标量
    sigma_cmp = torch.where(torch.isclose(sigma, ones_like_sigma), sigma_max, sigma)  # 形状: 标量

    std_dev_t = torch.sqrt(sigma / (1 - sigma_cmp)) * noise_level  # 形状: 标量
    dt = sigma_prev - sigma  # 形状: 标量
    step_std = std_dev_t * torch.sqrt(-dt)  # 形状: 标量

    # --- 漂移项（复用 SD3 推导公式） ---
    sample_feats = sample.feats.float()  # 形状: (N_total, C)
    model_feats = model_output.feats.float()  # 形状: (N_total, C)
    coords = sample.coords  # 形状: (N_total, 4)
    orig_dtype = sample.feats.dtype  # 形状: 标量dtype
    std_sq = std_dev_t ** 2  # 形状: 标量
    coeff_sample = 1 + (std_sq / (2 * sigma)) * dt  # 形状: 标量
    coeff_model = (1 + std_sq * (1 - sigma) / (2 * sigma)) * dt  # 形状: 标量
    prev_mean_feats_fp32 = sample_feats * coeff_sample + model_feats * coeff_model  # 形状: (N_total, C)
    prev_sample_mean = sp.SparseTensor(coords=coords, feats=prev_mean_feats_fp32.to(orig_dtype))  # 形状: 稀疏

    if deterministic:
        prev_feats_ode = sample_feats + dt * model_feats  # 形状: (N_total, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_feats_ode.to(orig_dtype))  # 形状: 稀疏
        prev_sample_mean = prev_sample  # 形状: 稀疏
        log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)  # 形状: (BK,)
        std_dev = torch.zeros(batch_size, device=device, dtype=torch.float32)  # 形状: (BK,)
        return prev_sample, log_prob, prev_sample_mean, std_dev

    if observed_prev_sample is not None:
        prev_sample = observed_prev_sample  # 形状: 稀疏
        prev_feats_fp32 = prev_sample.feats.float()  # 形状: (N_total, C)
    else:
        if generator is None:
            variance_noise = torch.randn_like(sample_feats)  # 形状: (N_total, C)
        else:
            variance_noise = torch.randn(sample_feats.shape, device=device, dtype=sample_feats.dtype, generator=generator)  # 形状: (N_total, C)
        prev_feats_fp32 = prev_mean_feats_fp32 + step_std * variance_noise  # 形状: (N_total, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_feats_fp32.to(orig_dtype))  # 形状: 稀疏

    diff = prev_feats_fp32.detach() - prev_mean_feats_fp32  # 形状: (N_total, C)
    noise_scale = step_std  # 形状: 标量
    log_prob_per_point = (
        -0.5 * (diff / noise_scale) ** 2  # 形状: (N_total, C)
        - torch.log(noise_scale)  # 形状: 标量
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))  # 形状: 标量
    )  # 形状: (N_total, C)
    log_prob_list: List[torch.Tensor] = []  # 形状: 列表(len=BK)
    for b in unique_batches.tolist():  # 形状: 标量
        mask_b = (bidx_all == int(b))  # 形状: (N_total,)
        vals = log_prob_per_point[mask_b]  # 形状: (N_b, C)
        mean_val = vals.mean() if vals.numel() > 0 else torch.zeros((), device=device, dtype=log_prob_per_point.dtype)  # 形状: 标量
        log_prob_list.append(mean_val)  # 形状: 追加标量
    log_prob = torch.stack(log_prob_list, dim=0)  # 形状: (BK,)
    std_vec = torch.full((batch_size,), float(step_std.detach()), device=device, dtype=torch.float32)  # 形状: (BK,)
    return prev_sample, log_prob, prev_sample_mean, std_vec


def trellis_flow_step_with_logprob_dense(
    scheduler: FlowMatchEulerDiscreteScheduler,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: float,
    prev_timestep: float,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    observed_prev_sample: Optional[torch.Tensor] = None,
    noise_level: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sample.device
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    t_cur = torch.as_tensor(float(timestep), device=device, dtype=torch.float32)
    t_pre = torch.as_tensor(float(prev_timestep), device=device, dtype=torch.float32)
    step_index = int(scheduler.index_for_timestep(t_cur))
    prev_step_index = int(scheduler.index_for_timestep(t_pre))
    step_index = max(0, min(step_index, int(sigmas.shape[0]) - 1))
    prev_step_index = max(0, min(prev_step_index, int(sigmas.shape[0]) - 1))
    sigma = sigmas[step_index]
    sigma_prev = sigmas[prev_step_index]
    sigma_max = sigmas[1 if int(sigmas.shape[0]) > 1 else 0]

    ones_like_sigma = torch.ones_like(sigma)
    sigma_safe = torch.clamp(sigma, min=1e-8)
    sigma_cmp = torch.where(torch.isclose(sigma_safe, ones_like_sigma), sigma_max, sigma_safe)

    std_dev_t = torch.sqrt(sigma_safe / (1 - sigma_cmp)) * noise_level
    dt = sigma_prev - sigma
    step_std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))

    sample_fp32 = sample.float()
    model_fp32 = model_output.float()
    orig_dtype = sample.dtype

    std_sq = std_dev_t ** 2
    coeff_sample = 1 + (std_sq / (2 * sigma_safe)) * dt
    coeff_model = (1 + std_sq * (1 - sigma_safe) / (2 * sigma_safe)) * dt
    prev_mean_fp32 = sample_fp32 * coeff_sample + model_fp32 * coeff_model
    prev_mean = prev_mean_fp32.to(orig_dtype)

    if deterministic:
        prev_sample = prev_mean
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=torch.float32)
        std_vec = torch.zeros(sample.shape[0], device=device, dtype=torch.float32)
        return prev_sample, log_prob, prev_mean, std_vec

    if observed_prev_sample is not None:
        prev_fp32 = observed_prev_sample.float()
    else:
        variance_noise = torch.randn_like(sample_fp32) if (generator is None) else torch.randn(
            sample_fp32.shape, device=device, dtype=sample_fp32.dtype, generator=generator
        )
        prev_fp32 = prev_mean_fp32 + step_std * variance_noise

    prev_sample = prev_fp32.to(orig_dtype)
    diff = prev_fp32.detach() - prev_mean_fp32
    noise_scale = step_std
    log_prob_per_elem = (
        -0.5 * (diff / noise_scale) ** 2
        - torch.log(noise_scale)
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    )
    log_prob = log_prob_per_elem.mean(dim=tuple(range(1, log_prob_per_elem.ndim)))
    std_vec = torch.full((sample.shape[0],), float(step_std.detach()), device=device, dtype=torch.float32)
    return prev_sample, log_prob, prev_mean, std_vec

def compute_log_prob_trellis_stage1(
    pipeline,
    samples: List[Dict],
    j: int,
    config: Stage1RuntimeConfig,
    detach_uncond: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    稠密结构流的 teacher-forcing 对数概率复算（与 Direct3D‑S2 稠密分支对齐）。
    输入 samples[k] 字段：
    - latents_seq_dense: List[Tensor(C,R,R,R)]
    - cond_patches: Tensor(1,P,C)
    - neg_patches: Optional[Tensor(1,P,C)]
    - t_seq: Tensor(T+1,) 或 ndarray
    返回：
    - prev_sample_batched: Tensor(BK,C,R,R,R)
    - log_prob_vec: Tensor(BK,)
    - kl_vec: Tensor(BK,)（占位或可选计算）
    """
    batch_size = len(samples)  # shape: ()
    if batch_size == 0:
        raise ValueError("samples 不能为空")

    target_device = pipeline.device  # shape: ()
    target_dtype = pipeline.dtype  # shape: ()

    current_stack = torch.stack(
        [s["latents_seq_dense"][j].to(device=target_device, dtype=target_dtype) for s in samples], dim=0
    )  # shape: (BK,C,R,R,R)
    prev_stack = torch.stack(
        [s["latents_seq_dense"][j + 1].to(device=target_device, dtype=target_dtype) for s in samples], dim=0
    )  # shape: (BK,C,R,R,R)

    cond_stack = torch.cat(
        [s["cond_patches"].to(device=target_device, dtype=target_dtype) for s in samples], dim=0
    )  # shape: (BK,P,C)
    neg_batched = None
    if float(config.guidance_scale) > 1.0:
        neg_sources = [s.get("neg_patches") for s in samples]
        if any(n is None for n in neg_sources):
            raise ValueError("CFG 模式下 neg_patches 不应为 None")
        neg_batched = torch.cat(
            [n.to(device=target_device, dtype=target_dtype) for n in neg_sources if n is not None], dim=0
        )  # shape: (BK,P,C)

    # 时间表（强制使用采样期保存的 t_seq，避免不一致）
    t_seq = samples[0]["t_seq"]  # shape: (T+1,)
    t = float(t_seq[j])  # shape: ()
    t_prev = float(t_seq[j + 1])  # shape: ()

    model = model_teacher = pipeline.get_flow_module("structure")
    t_tensor = torch.full((batch_size,), float(t), device=target_device, dtype=torch.float32)  # shape: (BK,)
    if float(config.guidance_scale) > 1.0 and (neg_batched is not None):
        vel_pos = model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)
        ctx = torch.no_grad() if bool(detach_uncond) else nullcontext()
        with ctx:
            vel_neg = model(current_stack, t_tensor, neg_batched)  # shape: (BK,C,R,R,R)
        model_output = vel_neg + float(config.guidance_scale) * (vel_pos - vel_neg)  # shape: (BK,C,R,R,R)
    else:
        model_output = model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)

    # 直接复用采样阶段的 Stage1 scheduler，并确保时间表一致
    scheduler = pipeline.stage1_scheduler

    prev_sample_batched, log_prob_vec, prev_mean, std_vec = trellis_flow_step_with_logprob_dense(
        scheduler=scheduler,
        sample=current_stack,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=prev_stack,
        noise_level=float(config.noise_level),
    )  # shapes: (BK,C,R,R,R), (BK,), (BK,C,R,R,R), (BK,)

    # —— KL 正则（可选）：禁用适配器的教师分布对比 ——
    kl_vec = torch.zeros_like(log_prob_vec)  # shape: (BK,)
    if bool(config.compute_kl) and (not bool(config.deterministic)):
        base_model = model_teacher  # shape: 模型
        with torch.no_grad():
            with (base_model.disable_adapter() if hasattr(base_model, "disable_adapter") else nullcontext()):
                if float(config.guidance_scale) > 1.0 and (neg_batched is not None):
                    vel_neg_ref = base_model(current_stack, t_tensor, neg_batched)  # shape: (BK,C,R,R,R)
                    vel_pos_ref = base_model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)
                    model_output_ref = vel_neg_ref + float(config.guidance_scale) * (vel_pos_ref - vel_neg_ref)  # shape: (BK,C,R,R,R)
                else:
                    model_output_ref = base_model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)

        _, _, prev_mean_ref, _ = trellis_flow_step_with_logprob_dense(
            scheduler=scheduler,
            sample=current_stack,
            model_output=model_output_ref,
            timestep=t,
            prev_timestep=t_prev,
            generator=None,
            deterministic=False,
            observed_prev_sample=prev_stack,
            noise_level=float(config.noise_level),
        )  # shapes: _, _, (BK,C,R,R,R), _

        diff = (prev_mean - prev_mean_ref)  # shape: (BK,C,R,R,R)
        diff_sq_mean = diff.pow(2).mean(dim=(1, 2, 3, 4))  # shape: (BK,)
        denom = (std_vec + 1e-8).pow(2)  # shape: (BK,)
        kl_vec = (diff_sq_mean / (2.0 * denom)).to(diff_sq_mean.dtype)  # shape: (BK,)

    return prev_sample_batched, log_prob_vec, kl_vec


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
        adjusted_coords = sparse_tensor.coords.clone()  # 形状: (N_i, 4)
        adjusted_coords[:, 0] = 0  # 形状: (N_i, 4) 先统一到 0，保证单样本 batch 形状为 1
        
        adjusted_sparse = sp.SparseTensor(
            coords=adjusted_coords,  # 形状: (N_i, 4)
            feats=sparse_tensor.feats  # 形状: (N_i, C)
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


 


def compute_log_prob_trellis_stage2(
    pipeline,  # TrellisPipelineWithLogProb 或兼容接口
    samples: List[Dict],
    j: int,
    config,
    detach_uncond: bool = False,
) -> Tuple[sp.SparseTensor, torch.Tensor, torch.Tensor]:
    """
    Batched 版本的单步对数概率计算。

    - 将多个样本在第 j 步的 SparseTensor 合并为 batched SparseTensor，一次前向获得 (B,) 的 log_prob。
    - KL 计算与单样本版本一致（可选，受 config.train.beta 控制）。
    
    Returns:
        Tuple[log_prob_vec, kl_vec]，形状均为 (B,)
    """
    batch_size = len(samples)
    if batch_size == 0:
        raise ValueError("samples 不能为空")

    target_device = pipeline.device
    target_dtype = pipeline.dtype
    current_list = [s["latents_seq"][j].to(device=target_device, dtype=target_dtype) for s in samples]
    prev_list = [s["latents_seq"][j + 1].to(device=target_device, dtype=target_dtype) for s in samples]
    batched_current = prepare_sparse_tensor_batch(current_list, batch_size=batch_size)  # batched SparseTensor
    batched_prev = prepare_sparse_tensor_batch(prev_list, batch_size=batch_size)  # batched SparseTensor
    # 可选调试：打印每批点数统计（通过环境变量开启）


    # 直接复用采样阶段已配置完成的 scheduler，避免重复 set_timesteps
    scheduler = pipeline.stage2_scheduler
    device = batched_current.coords.device

    # 条件拼接（按 batch 维度）
    cond_batched = torch.cat(
        [s["cond_patches"].to(device=target_device, dtype=target_dtype) for s in samples],
        dim=0,
    )  # 形状: (B, P, C)
    neg_cond_batched = None
    if float(config.guidance_scale) > 1.0:
        neg_sources = [s.get("neg_patches") for s in samples]
        if any(n is None for n in neg_sources):
            raise ValueError("CFG 模式下 neg_patches 不应为 None")
        neg_cond_batched = torch.cat(
            [n.to(device=target_device, dtype=target_dtype) for n in neg_sources],
            dim=0,
        )  # 形状: (B, P, C)

    # 时间序列（使用采样期记录）
    t_seq = samples[0]["t_seq"]  # (steps+1,)
    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])

    # 模型前向（CFG 按 batch 维执行）
    slat_flow_model = pipeline.get_flow_module("shape_slat")
    do_cfg = float(config.guidance_scale) > 1.0 and (neg_cond_batched is not None)
    t_tensor = torch.full((batch_size,), float(t), device=target_device, dtype=torch.float32)  # shape: (B,)

    cond_batched = cond_batched.to(device=target_device)
    if neg_cond_batched is not None:
        neg_cond_batched = neg_cond_batched.to(device=target_device)

    if do_cfg:
        pos_out = slat_flow_model(batched_current, t_tensor, cond_batched)
        ctx = torch.no_grad() if bool(detach_uncond) else nullcontext()
        with ctx:
            neg_out = slat_flow_model(batched_current, t_tensor, neg_cond_batched)
        cfg_feats = neg_out.feats + float(config.guidance_scale) * (pos_out.feats - neg_out.feats)  # (N, C)
        model_output = sp.SparseTensor(coords=batched_current.coords, feats=cfg_feats)
    else:
        model_output = slat_flow_model(batched_current, t_tensor, cond_batched)

    # 单步 Flow+LogProb（使用观测到的上一时刻作为目标）
    prev_sample_batched, log_prob_vec, prev_mean, std_vec = trellis_flow_step_with_logprob(
        scheduler=scheduler,
        sample=batched_current,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        noise_level=float(config.noise_level),
        observed_prev_sample=batched_prev,
    )  # log_prob_vec: (B,)

    # KL（可选，按 batch 计算教师输出）
    kl_vec = torch.zeros_like(log_prob_vec)
    if bool(getattr(config, "compute_kl", False)) and not bool(config.deterministic):
        teacher_model = slat_flow_model
        with torch.no_grad():
            with teacher_model.disable_adapter():
                if do_cfg:
                    neg_ref = teacher_model(batched_current, t_tensor, neg_cond_batched)  # 形状 (sum(N_b), C)
                    pos_ref = teacher_model(batched_current, t_tensor, cond_batched)      # 形状 (sum(N_b), C)
                    cfg_ref_feats = neg_ref.feats + float(config.guidance_scale) * (pos_ref.feats - neg_ref.feats)  # 形状 (sum(N_b), C)
                    model_output_ref = sp.SparseTensor(coords=batched_current.coords, feats=cfg_ref_feats)  # 形状 (sum(N_b), C)
                else:
                    model_output_ref = teacher_model(batched_current, t_tensor, cond_batched)  # 形状 (sum(N_b), C)
        _, _, prev_mean_ref, std_ref = trellis_flow_step_with_logprob(
            scheduler=scheduler,
            sample=batched_current,
            model_output=model_output_ref,
            timestep=t,
            prev_timestep=t_prev,
            generator=None,
            deterministic=bool(config.deterministic),
            observed_prev_sample=batched_prev,
            noise_level=float(config.noise_level),
        )  # prev_mean_ref.feats 形状 (sum(N_b), C), std_ref 形状 (B,)
        diff_feats = prev_mean.feats - prev_mean_ref.feats  # 形状: (sumN, C)
        kl_list: List[torch.Tensor] = []
        for b, sl in enumerate(prev_mean.layout):
            mean_sq = diff_feats[sl].pow(2).mean()  # 形状: 标量
            denom = (std_vec[b] + 1e-8) ** 2        # 形状: 标量
            kl_b = (mean_sq / (2.0 * denom)).to(mean_sq.dtype)  # 形状: 标量
            kl_list.append(kl_b.unsqueeze(0))  # 形状: (1,)
        kl_vec = torch.cat(kl_list, dim=0)  # 形状: (B,)

    return prev_sample_batched, log_prob_vec, kl_vec


def sparse_clone_with_feats(
    sparse: sp.SparseTensor,
    feats: torch.Tensor,
) -> sp.SparseTensor:
    """
    用新的特征复制 SparseTensor，保留坐标与 layout。
    """
    if feats.shape != sparse.feats.shape:
        raise ValueError(f"feats 形状不匹配: 预期 {sparse.feats.shape}, 实际 {feats.shape}")
    new_feats = feats.to(dtype=sparse.feats.dtype, device=sparse.feats.device)  # 形状: (N_total, C)
    return sp.SparseTensor(coords=sparse.coords.clone(), feats=new_feats, layout=list(getattr(sparse, "layout", [])))  # 形状: 稀疏


def sparse_batch_mse(
    pred: sp.SparseTensor,
    target: sp.SparseTensor,
) -> torch.Tensor:
    """
    按 layout 聚合的稀疏 MSE，返回 (B,)。
    """
    if len(getattr(pred, "layout", [])) != len(getattr(target, "layout", [])):
        raise ValueError("pred 与 target 的 layout 长度不一致")
    mse_list: List[torch.Tensor] = []  # 形状: (B,)
    for sl_pred, sl_tgt in zip(pred.layout, target.layout):
        diff = pred.feats[sl_pred] - target.feats[sl_tgt]  # 形状: (N_b, C)
        mse_val = diff.pow(2).mean() if diff.numel() > 0 else torch.zeros((), device=pred.feats.device, dtype=pred.feats.dtype)  # 形状: ()
        mse_list.append(mse_val.unsqueeze(0))  # 形状: (1,)
    return torch.cat(mse_list, dim=0)  # 形状: (B,)


def dense_batch_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    对稠密张量计算逐样本 MSE，返回 (BK,)。
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred 与 target 形状不一致: {pred.shape} vs {target.shape}")
    diff = pred - target  # 形状: (BK, C, R, R, R)
    mse = diff.pow(2).mean(dim=(1, 2, 3, 4))  # 形状: (BK,)
    return mse  # 形状: (BK,)


def compute_sparse_weighted_mse(
    pred: sp.SparseTensor,
    target: sp.SparseTensor,
) -> torch.Tensor:
    """
    为兼容 Direct3D 训练脚本的命名，当前等价于未加权 MSE。
    """
    return sparse_batch_mse(pred, target)  # 形状: (B,)


def compute_dense_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    为兼容 Direct3D 训练脚本的命名，当前等价于未加权 MSE。
    """
    return dense_batch_mse(pred, target)  # 形状: (BK,)


# 导出符号，便于训练脚本直接 import *
__all__ = [
    "SparseTensor",
    "Stage1RuntimeConfig",
    "Stage2RuntimeConfig",
    "set_trellis_timesteps",
    "create_trellis_scheduler",
    "trellis_flow_step_with_logprob",
    "compute_log_prob_trellis_stage2",
    "sparse_tensor_cfg_guidance",
    "sparse_tensor_cat",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
    "sparse_clone_with_feats",
    "sparse_batch_mse",
    "dense_batch_mse",
    "compute_sparse_weighted_mse",
    "compute_dense_weighted_mse",
]
