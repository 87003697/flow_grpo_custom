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
import types
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# 导入 TRELLIS 内置门面
from generators.trellis import sparse as sp

# 导入项目模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


@dataclass
class Stage2RuntimeConfig:
    """TRELLIS Stage 2 运行时配置"""
    guidance_scale: float
    deterministic: bool


@dataclass
class Stage1RuntimeConfig:
    """TRELLIS Stage 1 运行时配置（与 Direct3D‑S2 对齐的最小集）"""
    steps: int = 50
    guidance_scale: float = 0.0
    deterministic: bool = False
    compute_kl: bool = False


def create_trellis_scheduler(
    steps: int,
    device: Union[str, torch.device] = 'cpu',
    rescale_t: float = 1.0,
) -> FlowMatchEulerDiscreteScheduler:
    """
    创建与 TRELLIS 官方完全兼容的 scheduler
    
    通过直接覆盖 scheduler.sigmas，确保与 TRELLIS 的时间序列完全一致（0% 误差）。
    """
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # 形状: (steps+1,)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)
    sigmas_trellis = torch.from_numpy(t_seq / 1000.0).to(device=device, dtype=torch.float32)  # 形状: (steps+1,)
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.sigmas = sigmas_trellis
    scheduler.timesteps = sigmas_trellis * 1000
    scheduler.num_inference_steps = steps
    return scheduler


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

    # 时间序列（强制使用采样期保存的 t_seq，禁用回退重建）
    assert "t_seq" in sample, "sample 必须包含采样期保存的 t_seq"
    t_seq = sample["t_seq"]  # 形状: (steps+1,)

    # 训练与采样参数一致性断言（deterministic/sigma_min/rescale_t/num_inference_steps）
    if "sampler_params" in sample:
        sp_cfg = sample["sampler_params"]
        # 读取训练期 config（移除 fallback）
        cfg_det = bool(config.deterministic)  # 标量
        cfg_sigma_min = float(config.sigma_min)  # 标量
        cfg_rescale_t = float(config.rescale_t)  # 标量
        cfg_num_steps = int(config.num_inference_steps)  # 标量
        # 断言一致
        assert bool(sp_cfg.get('deterministic', cfg_det)) == cfg_det, "deterministic 与采样期不一致"
        assert abs(float(sp_cfg.get('sigma_min', cfg_sigma_min)) - cfg_sigma_min) < 1e-8, "sigma_min 与采样期不一致"
        assert abs(float(sp_cfg.get('rescale_t', cfg_rescale_t)) - cfg_rescale_t) < 1e-8, "rescale_t 与采样期不一致"
        assert int(sp_cfg.get('num_inference_steps', cfg_num_steps)) == cfg_num_steps, "num_inference_steps 与采样期不一致"

    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])

    # 图像条件（统一仅保留 patch 级）
    image_idx = int(sample.get("image_idx", 0))  # 标量
    cond_patches = image_conds['cond'][image_idx:image_idx+1]  # 形状 (1, P, C)
    neg_patches = image_conds.get('neg_cond', None)  # 形状 (1, P, C) 或 None
    if neg_patches is not None:
        neg_patches = neg_patches[image_idx:image_idx+1]  # 形状 (1, P, C)

    guidance_scale = float(config.guidance_scale)
    do_cfg = guidance_scale > 1.0 and neg_patches is not None

    deterministic = bool(config.deterministic)

    # 创建 TRELLIS 兼容的 scheduler（与官方完全一致）
    device = current_sparse.coords.device
    num_inference_steps = int(config.num_inference_steps)
    rescale_t = float(getattr(config, 'rescale_t', 1.0))
    scheduler = create_trellis_scheduler(steps=num_inference_steps, device=device, rescale_t=rescale_t)

    # 模型前向（单步）
    slat_flow_model = pipeline.get_trainable_model()
    t_tensor = torch.tensor([t], device=device, dtype=torch.float32)  # shape: (1,)

    # 对齐设备以避免多卡下广播失败
    cond_patches = cond_patches.to(device=current_sparse.coords.device)  # shape: (1, P, C)
    if neg_patches is not None:
        neg_patches = neg_patches.to(device=current_sparse.coords.device)  # shape: (1, P, C)

    if do_cfg:
        neg_output = slat_flow_model(current_sparse, t_tensor, neg_patches)
        pos_output = slat_flow_model(current_sparse, t_tensor, cond_patches)
        cfg_feats = neg_output.feats + guidance_scale * (pos_output.feats - neg_output.feats)
        model_output = sp.SparseTensor(coords=current_sparse.coords, feats=cfg_feats)
    else:
        model_output = slat_flow_model(current_sparse, t_tensor, cond_patches)

    # 单步 Flow + LogProb（使用观测到的 prev 作为对数似然的目标）
    prev_sample, log_prob, prev_sample_mean, std_dev = trellis_flow_step_with_logprob(
        scheduler=scheduler,
        sample=current_sparse,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=deterministic,
        observed_prev_sample=observed_prev_sparse,
    )

    # KL 计算移至训练段进行：此处固定返回零，避免重复计算
    kl_div = torch.zeros_like(log_prob)  # 形状 (1,)

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
    # 可选调试：打印每批点数统计（通过环境变量开启）
    if os.environ.get("TRELLIS_DEBUG_MEM", "0") == "1":
        B_dbg = len(samples)  # 形状: []
        counts_per_batch_dbg = torch.bincount(batched_current.coords[:, 0].to(torch.long), minlength=B_dbg)  # 形状: (B_dbg,)
        total_points_dbg = int(batched_current.coords.shape[0])  # 形状: []
        max_points_dbg = int(counts_per_batch_dbg.max().item()) if counts_per_batch_dbg.numel() > 0 else 0  # 形状: []
        channels_dbg = int(batched_current.feats.shape[1])  # 形状: []
        rank_dbg = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0  # 形状: []
        if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
            print(f"[Rank {rank_dbg}] TrainStep j={int(j)} Batched Sparse (B={B_dbg}) total_N={total_points_dbg}, max_N_per_sample={max_points_dbg}, C={channels_dbg}")

    # 时间标量（所有样本相同时间表）
    # 时间序列（强制使用采样期保存的 t_seq，禁用回退）
    assert "t_seq" in samples[0], "samples[0] 必须包含采样期保存的 t_seq"
    t_seq = samples[0]["t_seq"]  # (steps+1,)
    # 检查所有样本的步数一致
    for s in samples:
        assert "t_seq" in s and len(s["t_seq"]) == len(t_seq), "所有样本的 t_seq 必须存在且长度一致"

    # 训练与采样参数一致性断言（deterministic/sigma_min/rescale_t/num_inference_steps）
    if "sampler_params" in samples[0]:
        sp_cfg = samples[0]["sampler_params"]
        cfg_det = bool(config.deterministic)  # 标量
        cfg_sigma_min = float(config.sigma_min)  # 标量
        cfg_rescale_t = float(config.rescale_t)  # 标量
        cfg_num_steps = int(config.num_inference_steps)  # 标量
        assert bool(sp_cfg.get('deterministic', cfg_det)) == cfg_det, "deterministic 与采样期不一致"
        assert abs(float(sp_cfg.get('sigma_min', cfg_sigma_min)) - cfg_sigma_min) < 1e-8, "sigma_min 与采样期不一致"
        assert abs(float(sp_cfg.get('rescale_t', cfg_rescale_t)) - cfg_rescale_t) < 1e-8, "rescale_t 与采样期不一致"
        assert int(sp_cfg.get('num_inference_steps', cfg_num_steps)) == cfg_num_steps, "num_inference_steps 与采样期不一致"
    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])

    # 创建 TRELLIS 兼容的 scheduler（与官方完全一致）
    device = batched_current.coords.device
    num_inference_steps = int(config.num_inference_steps)
    rescale_t = float(getattr(config, 'rescale_t', 1.0))
    scheduler = create_trellis_scheduler(steps=num_inference_steps, device=device, rescale_t=rescale_t)

    # 条件拼接（按 batch 维度）
    cond_batched = torch.cat([c["cond"] for c in image_conds_list], dim=0)  # 形状: (B, P, C)
    # batched 重算不应存在缺失样本：若启用 CFG（guidance_scale>1.0），强制所有样本提供 neg_cond
    if float(config.guidance_scale) > 1.0:
        assert all((c.get("neg_cond", None) is not None) for c in image_conds_list), "CFG 模式下必须为所有样本提供 neg_cond"
        neg_cond_batched = torch.cat([c["neg_cond"] for c in image_conds_list], dim=0)  # 形状: (B, P, C)
    else:
        neg_cond_batched = None

    # 模型前向（CFG 按 batch 维执行）
    slat_flow_model = pipeline.get_trainable_model()
    base_model = slat_flow_model.module if hasattr(slat_flow_model, "module") else slat_flow_model  # ()
    do_cfg = float(config.guidance_scale) > 1.0 and (neg_cond_batched is not None)
    t_tensor = torch.tensor([t], device=batched_current.coords.device, dtype=torch.float32)  # shape: (1,)

    # 对齐设备以避免多卡下广播失败
    cond_batched = cond_batched.to(device=batched_current.coords.device)  # shape: (B, P, C)
    if neg_cond_batched is not None:
        neg_cond_batched = neg_cond_batched.to(device=batched_current.coords.device)  # shape: (B, P, C)

    if do_cfg:
        neg_out = slat_flow_model(batched_current, t_tensor, neg_cond_batched)
        neg_out = neg_out.detach() # detach the graident w.r.t. the negtive \ unconditional terms
        pos_out = slat_flow_model(batched_current, t_tensor, cond_batched)
        cfg_feats = neg_out.feats + float(config.guidance_scale) * (pos_out.feats - neg_out.feats)  # (N, C)
        model_output = sp.SparseTensor(coords=batched_current.coords, feats=cfg_feats)
    else:
        model_output = slat_flow_model(batched_current, t_tensor, cond_batched)

    # 单步 Flow+LogProb（使用观测到的上一时刻作为目标）
    _, log_prob_vec, prev_mean, std_vec = trellis_flow_step_with_logprob(
        scheduler=scheduler,
        sample=batched_current,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=batched_prev_obs,
    )  # log_prob_vec: (B,)

    # KL（可选，按 batch 计算教师输出）
    kl_vec = torch.zeros_like(log_prob_vec)
    if float(config.kl_reward) > 0.0 and not bool(config.deterministic):
        with torch.no_grad():
            with base_model.disable_adapter():
                if do_cfg:
                    neg_ref = base_model(batched_current, t_tensor, neg_cond_batched)  # 形状 (sum(N_b), C) in feats
                    pos_ref = base_model(batched_current, t_tensor, cond_batched)      # 形状 (sum(N_b), C) in feats
                    cfg_ref_feats = neg_ref.feats + float(config.guidance_scale) * (pos_ref.feats - neg_ref.feats)  # 形状 (sum(N_b), C)
                    model_output_ref = sp.SparseTensor(coords=batched_current.coords, feats=cfg_ref_feats)  # 形状 (sum(N_b), C)
                else:
                    model_output_ref = base_model(batched_current, t_tensor, cond_batched)  # 形状 (sum(N_b), C)
        _, _, prev_mean_ref, std_ref = trellis_flow_step_with_logprob(
            scheduler=scheduler,
            sample=batched_current,
            model_output=model_output_ref,
            timestep=t,
            prev_timestep=t_prev,
            generator=None,
            deterministic=bool(config.deterministic),
            observed_prev_sample=batched_prev_obs,
        )  # prev_mean_ref.feats 形状 (sum(N_b), C), std_ref 形状 (B,)
        diff = prev_mean.feats - prev_mean_ref.feats  # (N, C)
        denom = (std_vec + 1e-8) ** 2                  # (B,)
        # 聚合到 (B,) KL（按 coords[:,0] 批索引聚合）
        kl_list = []  # 形状: 列表(len=B)
        bidx_all = prev_mean.coords[:, 0].to(torch.long)  # 形状: (N_total,)
        for b in range(len(samples)):
            mask_b = (bidx_all == int(b))  # 形状: (N_total,)
            kl_b = (diff[mask_b].pow(2).mean() / (2.0 * denom[b])).unsqueeze(0)  # 形状: (1,)
            kl_list.append(kl_b)  # 形状: 追加(1,)
        kl_vec = torch.cat(kl_list, dim=0)  # (B,)

    # 将 NaN/Inf 置 0，避免在训练中传播
    log_prob_vec = torch.nan_to_num(log_prob_vec, nan=0.0, posinf=0.0, neginf=0.0)  # (B,)
    kl_vec = torch.nan_to_num(kl_vec, nan=0.0, posinf=0.0, neginf=0.0)  # (B,)

    return log_prob_vec, kl_vec


 


# === Direct3D 命名别名（对外同时提供 Direct3D 与 Trellis 两套 API 名称） ===
 


 


 


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


 
