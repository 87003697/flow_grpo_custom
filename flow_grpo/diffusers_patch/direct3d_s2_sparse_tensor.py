#!/usr/bin/env python3
"""
Direct3D-S2 稀疏张量适配层

提供与 TRELLIS 稀疏张量工具近似一致的接口，便于在 Direct3D Stage2 GRPO
训练流程中进行 reshape、replace、批量拼接、CFG 合并等操作。

核心功能：
- direct3d_flow_step_with_logprob：Direct3D 稀疏张量的一步 Flow+LogProb 计算
- compute_log_prob_direct3d_stage2(_batched)：训练阶段单样本/批量对数概率
- sparse_tensor_cfg_guidance / prepare_sparse_tensor_batch 等辅助函数
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import sys
from pathlib import Path
import torch
from contextlib import nullcontext

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

# ensure official Direct3D-S2 package (under _reference_codes) is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_DIR = _PROJECT_ROOT / "_reference_codes" / "Direct3D-S2"
if _REFERENCE_DIR.exists():
    sys.path.insert(0, str(_REFERENCE_DIR))

from direct3d_s2.modules import sparse as base_sp  # type: ignore


SparseTensor = base_sp.SparseTensor


@dataclass
class Stage2RuntimeConfig:
    guidance_scale: float
    deterministic: bool
    compute_kl: bool = False


@dataclass
class Stage1RuntimeConfig:
    steps: int = 50
    guidance_scale: float = 0.0
    deterministic: bool = False
    compute_kl: bool = False


def direct3d_flow_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    sample: SparseTensor,
    model_output: SparseTensor,
    timestep: float,
    prev_timestep: float,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    observed_prev_sample: Optional[SparseTensor] = None,
    noise_level: float = 0.7,
) -> Tuple[SparseTensor, torch.Tensor, SparseTensor, torch.Tensor]:
    device = sample.feats.device  # shape: () 设备
    batch_size = len(sample.layout)  # shape: () 批量大小BK（按layout切片数）

    # --- 调度器 sigma 信息：按 index_for_timestep 精确映射（与参考实现一致） ---
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)  # 形状: (T,)
    t_cur = torch.as_tensor(float(timestep), device=device, dtype=torch.float32)  # 形状: ()
    t_pre = torch.as_tensor(float(prev_timestep), device=device, dtype=torch.float32)  # 形状: ()
    step_index = int(scheduler.index_for_timestep(t_cur))  # 形状: 标量
    prev_step_index = int(scheduler.index_for_timestep(t_pre))  # 形状: 标量
    step_index = max(0, min(step_index, int(sigmas.shape[0]) - 1))  # 形状: 标量
    prev_step_index = max(0, min(prev_step_index, int(sigmas.shape[0]) - 1))  # 形状: 标量
    sigma = sigmas[step_index]  # 形状: ()
    sigma_prev = sigmas[prev_step_index]  # 形状: ()
    sigma_max = sigmas[1 if int(sigmas.shape[0]) > 1 else 0]  # 形状: ()

    ones_like_sigma = torch.ones_like(sigma)
    sigma_cmp = torch.where(torch.isclose(sigma, ones_like_sigma), sigma_max, sigma)

    std_dev_t = torch.sqrt(sigma / (1 - sigma_cmp)) * noise_level
    dt = sigma_prev - sigma
    step_std = std_dev_t * torch.sqrt(-dt)

    # --- 漂移项（复用 SD3 推导公式） ---
    sample_feats = sample.feats.float()
    model_feats = model_output.feats.float()
    coords = sample.coords
    orig_dtype = sample.feats.dtype

    std_sq = std_dev_t ** 2
    coeff_sample = 1 + (std_sq / (2 * sigma)) * dt
    coeff_model = (1 + std_sq * (1 - sigma) / (2 * sigma)) * dt
    prev_mean_feats_fp32 = sample_feats * coeff_sample + model_feats * coeff_model
    prev_mean = SparseTensor(coords=coords, feats=prev_mean_feats_fp32.to(orig_dtype), layout=list(sample.layout))  # shape: (N_total,C)聚合为(B, C)

    if deterministic:
        prev_feats_ode = sample_feats + dt * model_feats  # shape: (N_total, C)
        prev_sample = SparseTensor(coords=coords, feats=prev_feats_ode.to(orig_dtype), layout=list(sample.layout))  # shape: (B, C)
        prev_mean = prev_sample  # shape: (B, C)
        log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (BK,)
        std_dev = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (BK,)
        return prev_sample, log_prob, prev_mean, std_dev

    if observed_prev_sample is not None:
        prev_sample = observed_prev_sample
        prev_feats_fp32 = prev_sample.feats.float()
    else:
        if generator is None:
            variance_noise = torch.randn_like(sample_feats)
        else:
            variance_noise = torch.randn(sample_feats.shape, device=device, dtype=sample_feats.dtype, generator=generator)
        prev_feats_fp32 = prev_mean_feats_fp32 + step_std * variance_noise
        prev_sample = SparseTensor(coords=coords, feats=prev_feats_fp32.to(orig_dtype), layout=list(sample.layout))

    diff = prev_feats_fp32.detach() - prev_mean_feats_fp32  # shape: (N_total, C)
    noise_scale = step_std
    log_prob_per_point = (
        -0.5 * (diff / noise_scale) ** 2
        - torch.log(noise_scale)
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    )
    # —— 按 layout 逐候选聚合（可微） ——
    log_prob_list: List[torch.Tensor] = []  # shape: (BK,)
    for sl in prev_sample.layout:  # layout len=BK
        vals = log_prob_per_point[sl]  # shape: (N_b, C)
        mean_val = vals.mean() if vals.numel() > 0 else torch.zeros((), device=device, dtype=log_prob_per_point.dtype)  # shape: ()
        log_prob_list.append(mean_val)
    log_prob = torch.stack(log_prob_list, dim=0)  # shape: (BK,)
    std_vec = torch.full((batch_size,), float(step_std.detach()), device=device, dtype=torch.float32)  # shape: (BK,)
    return prev_sample, log_prob, prev_mean, std_vec


def direct3d_flow_step_with_logprob_dense(
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
    """Dense 版单步 SDE + logprob，与稀疏实现数学等价。

    约定：
    - sample: (BK, C, R, R, R)
    - model_output: (BK, C, R, R, R)
    - 返回：
      - prev_sample: (BK, C, R, R, R)
      - log_prob_vec: (BK,)
      - prev_sample_mean: (BK, C, R, R, R)
      - std_vec: (BK,)
    """
    device = sample.device  # shape: ()

    # --- 调度器 sigma 信息 ---
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)  # shape: (T,)
    t_cur = torch.as_tensor(float(timestep), device=device, dtype=torch.float32)  # shape: ()
    t_pre = torch.as_tensor(float(prev_timestep), device=device, dtype=torch.float32)  # shape: ()
    step_index = int(scheduler.index_for_timestep(t_cur))  # shape: ()
    prev_step_index = int(scheduler.index_for_timestep(t_pre))  # shape: ()
    step_index = max(0, min(step_index, int(sigmas.shape[0]) - 1))  # shape: ()
    prev_step_index = max(0, min(prev_step_index, int(sigmas.shape[0]) - 1))  # shape: ()
    sigma = sigmas[step_index]  # shape: ()
    sigma_prev = sigmas[prev_step_index]  # shape: ()
    sigma_max = sigmas[1 if int(sigmas.shape[0]) > 1 else 0]  # shape: ()

    ones_like_sigma = torch.ones_like(sigma)  # shape: ()
    sigma_cmp = torch.where(torch.isclose(sigma, ones_like_sigma), sigma_max, sigma)  # shape: ()

    std_dev_t = torch.sqrt(sigma / (1 - sigma_cmp)) * noise_level  # shape: ()
    dt = sigma_prev - sigma  # shape: ()
    step_std = std_dev_t * torch.sqrt(-dt)  # shape: ()

    # --- 漂移项（与稀疏实现一致） ---
    sample_fp32 = sample.float()  # shape: (BK, C, R, R, R)
    model_fp32 = model_output.float()  # shape: (BK, C, R, R, R)
    orig_dtype = sample.dtype  # shape: ()

    std_sq = std_dev_t ** 2  # shape: ()
    coeff_sample = 1 + (std_sq / (2 * sigma)) * dt  # shape: ()
    coeff_model = (1 + std_sq * (1 - sigma) / (2 * sigma)) * dt  # shape: ()
    prev_mean_fp32 = sample_fp32 * coeff_sample + model_fp32 * coeff_model  # shape: (BK, C, R, R, R)
    prev_mean = prev_mean_fp32.to(orig_dtype)  # shape: (BK, C, R, R, R)

    if deterministic:
        prev_mean_ode_fp32 = sample_fp32 + dt * model_fp32  # shape: (BK, C, R, R, R)
        prev_sample = prev_mean_ode_fp32.to(orig_dtype)  # shape: (BK, C, R, R, R)
        prev_mean = prev_sample  # shape: (BK, C, R, R, R)
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=torch.float32)  # shape: (BK,)
        std_vec = torch.zeros(sample.shape[0], device=device, dtype=torch.float32)  # shape: (BK,)
        return prev_sample, log_prob, prev_mean, std_vec

    if observed_prev_sample is not None:
        prev_fp32 = observed_prev_sample.float()  # shape: (BK, C, R, R, R)
    else:
        if generator is None:
            variance_noise = torch.randn_like(sample_fp32)  # shape: (BK, C, R, R, R)
        else:
            variance_noise = torch.randn(sample_fp32.shape, device=device, dtype=sample_fp32.dtype, generator=generator)  # shape: (BK, C, R, R, R)
        prev_fp32 = prev_mean_fp32 + step_std * variance_noise  # shape: (BK, C, R, R, R)

    prev_sample = prev_fp32.to(orig_dtype)  # shape: (BK, C, R, R, R)

    diff = prev_fp32.detach() - prev_mean_fp32  # shape: (BK, C, R, R, R)
    noise_scale = step_std  # shape: ()
    log_prob_per_elem = (
        -0.5 * (diff / noise_scale) ** 2  # shape: (BK, C, R, R, R)
        - torch.log(noise_scale)  # shape: ()
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))  # shape: ()
    )  # shape: (BK, C, R, R, R)
    log_prob = log_prob_per_elem.mean(dim=(1, 2, 3, 4))  # shape: (BK,)
    std_vec = torch.full((sample.shape[0],), float(step_std.detach()), device=device, dtype=torch.float32)  # shape: (BK,)
    return prev_sample, log_prob, prev_mean, std_vec


def compute_log_prob_direct3d_stage1(
    pipeline,
    samples: List[Dict],
    j: int,
    config: Stage1RuntimeConfig,
    detach_uncond: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """稠密分支的 teacher-forcing 对数概率复算（与 Stage2 对齐的接口）。

    输入 samples[k] 字段：
    - latents_seq_dense: List[Tensor(C,R,R,R)]
    - cond_patches: Tensor(1,P,C)
    - neg_patches: Optional[Tensor(1,P,C)]
    - t_seq: Tensor(T+1,) 或 ndarray
    返回：
    - prev_sample_batched: Tensor(BK,C,R,R,R)
    - log_prob_vec: Tensor(BK,)
    - kl_vec: Tensor(BK,)（占位零）
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

    t_seq = samples[0]["t_seq"]  # shape: (T+1,)
    t = float(t_seq[j])  # shape: ()
    t_prev = float(t_seq[j + 1])  # shape: ()

    model = pipeline.ref.dense_dit  # shape: ()
    t_tensor = torch.full((batch_size,), float(t), device=target_device, dtype=torch.float32)  # shape: (BK,)
    if float(config.guidance_scale) > 1.0 and (neg_batched is not None):
        vel_pos = model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)
        ctx = torch.no_grad() if bool(detach_uncond) else nullcontext()
        with ctx:
            vel_neg = model(current_stack, t_tensor, neg_batched)  # shape: (BK,C,R,R,R)
        model_output = vel_neg + float(config.guidance_scale) * (vel_pos - vel_neg)  # shape: (BK,C,R,R,R)
    else:
        model_output = model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)

    scheduler = pipeline.ref.dense_scheduler  # shape: ()
    prev_sample_batched, log_prob_vec, prev_mean, std_vec = direct3d_flow_step_with_logprob_dense(
        scheduler=scheduler,
        sample=current_stack,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=prev_stack,
    )  # shapes: (BK,C,R,R,R), (BK,), (BK,C,R,R,R), (BK,)

    # —— KL 正则（可选）：与禁用适配器的教师分布对比 ——
    kl_vec = torch.zeros_like(log_prob_vec)  # shape: (BK,)
    if bool(config.compute_kl) and (not bool(config.deterministic)):
        # 统一使用 pipeline 的 DDP 解包方法，避免各处散落判断
        base_model = pipeline._resolve_dense_dit_module()  # shape: 模型
        with torch.no_grad():
            with (base_model.disable_adapter() if hasattr(base_model, "disable_adapter") else torch.enable_grad()):
                if float(config.guidance_scale) > 1.0 and (neg_batched is not None):
                    vel_neg_ref = base_model(current_stack, t_tensor, neg_batched)  # shape: (BK,C,R,R,R)
                    vel_pos_ref = base_model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)
                    model_output_ref = vel_neg_ref + float(config.guidance_scale) * (vel_pos_ref - vel_neg_ref)  # shape: (BK,C,R,R,R)
                else:
                    model_output_ref = base_model(current_stack, t_tensor, cond_stack)  # shape: (BK,C,R,R,R)

        # 用同一调度步计算教师分布的均值
        _, _, prev_mean_ref, _ = direct3d_flow_step_with_logprob_dense(
            scheduler=scheduler,
            sample=current_stack,
            model_output=model_output_ref,
            timestep=t,
            prev_timestep=t_prev,
            generator=None,
            deterministic=False,
            observed_prev_sample=prev_stack,
        )  # shapes: _, _, (BK,C,R,R,R), _

        # KL = E[(μ - μ_ref)^2] / (2 σ^2) ，对 (C,R,R,R) 维求均值
        diff = (prev_mean - prev_mean_ref)  # shape: (BK,C,R,R,R)
        diff_sq_mean = diff.pow(2).mean(dim=(1, 2, 3, 4))  # shape: (BK,)
        denom = (std_vec + 1e-8).pow(2)  # shape: (BK,)
        kl_vec = (diff_sq_mean / (2.0 * denom)).to(diff_sq_mean.dtype)  # shape: (BK,)

    return prev_sample_batched, log_prob_vec, kl_vec


def sparse_tensor_cfg_guidance(
    positive_sparse: SparseTensor,
    negative_sparse: SparseTensor,
    guidance_scale: float,
) -> SparseTensor:
    cfg_feats = negative_sparse.feats + guidance_scale * (positive_sparse.feats - negative_sparse.feats)  # shape: (N_total, C)
    cfg_tensor = SparseTensor(coords=positive_sparse.coords, feats=cfg_feats, layout=list(positive_sparse.layout))  # shape: (B, C)
    return cfg_tensor


def prepare_sparse_tensor_batch(
    sparse_list: List[SparseTensor],
    batch_size: int,
) -> SparseTensor:
    if len(sparse_list) != batch_size:
        raise ValueError("SparseTensor 列表长度与 batch_size 不一致")

    coords_chunks: List[torch.Tensor] = []  # shape: (总块数,)
    feats_chunks: List[torch.Tensor] = []   # shape: (总块数,)
    layout_slices: List[slice] = []         # shape: (batch_size,)

    feature_offset = 0  # 标量，累计特征下标

    for batch_idx, sparse_tensor in enumerate(sparse_list):
        coords = sparse_tensor.coords  # shape: (N_i, 4)
        feats = sparse_tensor.feats    # shape: (N_i, C)
        layout = sparse_tensor.layout  # List[slice]

        sample_start = feature_offset  # 标量
        sample_point_count = 0  # 标量

        for sl in layout:
            coords_block = coords[sl].clone()  # shape: (M, 4)
            feats_block = feats[sl]            # shape: (M, C)

            block_size = coords_block.shape[0]  # 形状: 标量
            if block_size == 0:
                continue

            coords_block[:, 0] = int(batch_idx)  # shape: (M, 4)

            coords_chunks.append(coords_block)
            feats_chunks.append(feats_block)

            feature_offset += block_size
            sample_point_count += block_size

        if sample_point_count == 0:
            raise ValueError("拼接 SparseTensor 时出现空样本")

        sample_end = feature_offset  # 标量
        layout_slices.append(slice(sample_start, sample_end))

    if len(coords_chunks) == 0:
        raise ValueError("拼接 SparseTensor 时出现空输入")

    coords_cat = torch.cat(coords_chunks, dim=0)  # shape: (sum N_i, 4)
    feats_cat = torch.cat(feats_chunks, dim=0)    # shape: (sum N_i, C)

    combined = SparseTensor(
        feats=feats_cat,
        coords=coords_cat,
        layout=layout_slices,
    )
    return combined


def extract_sparse_tensor_from_batch(
    batch_sparse: SparseTensor,
    batch_idx: int,
) -> SparseTensor:
    mask = (batch_sparse.coords[:, 0] == batch_idx)  # shape: (N_total,)
    if not mask.any():
        raise ValueError("指定 batch_idx 不存在")
    coords = batch_sparse.coords[mask].clone()  # shape: (N_b, 4)
    coords[:, 0] = 0  # shape: (N_b, 4)
    feats = batch_sparse.feats[mask]  # shape: (N_b, C)
    return SparseTensor(coords=coords, feats=feats, layout=[slice(0, feats.shape[0])])


def compute_log_prob_direct3d_stage2(
    pipeline,
    samples: List[Dict],
    j: int,
    config: Stage2RuntimeConfig,
    detach_uncond: bool = False,
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    batch_size = len(samples)
    if batch_size == 0:
        raise ValueError("samples 不能为空")

    target_device = pipeline.device
    target_dtype = pipeline.dtype
    current_list = [s["latents_seq"][j].to(device=target_device, dtype=target_dtype) for s in samples]
    prev_list = [s["latents_seq"][j + 1].to(device=target_device, dtype=target_dtype) for s in samples]
    batched_current = prepare_sparse_tensor_batch(current_list, batch_size=batch_size)
    batched_prev = prepare_sparse_tensor_batch(prev_list, batch_size=batch_size)

    device = target_device
    cond_stack = torch.cat(
        [s["cond_patches"].to(device=target_device, dtype=target_dtype) for s in samples],
        dim=0,
    )  # shape: (batch_size, P, C)
    cond_batched = cond_stack
    neg_batched = None
    if config.guidance_scale > 1.0:
        neg_sources = [s.get("neg_patches") for s in samples]
        if any(n is None for n in neg_sources):
            raise ValueError("CFG 模式下 neg_patches 不应为 None")
        neg_stack = torch.cat(
            [n.to(device=target_device, dtype=target_dtype) for n in neg_sources if n is not None],
            dim=0,
        )  # shape: (batch_size, P, C)
        neg_batched = neg_stack

    t_seq = samples[0]["t_seq"]
    t = float(t_seq[j])
    t_prev = float(t_seq[j + 1])
    model = pipeline.get_trainable_model_stage2()

    t_tensor = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)  # 形状: (B,)
    if config.guidance_scale > 1.0 and neg_batched is not None:
        pos_out = model(batched_current, t_tensor, cond_batched)  # 形状: 稀疏(条件分支)
        ctx = torch.no_grad() if bool(detach_uncond) else nullcontext()
        with ctx:
            neg_out = model(batched_current, t_tensor, neg_batched)  # 形状: 稀疏(无条件分支)
        cfg_feats = neg_out.feats + config.guidance_scale * (pos_out.feats - neg_out.feats)  # 形状: (sumN, C)
        model_output = SparseTensor(coords=batched_current.coords, feats=cfg_feats, layout=list(batched_current.layout))  # 形状: 稀疏
    else:
        model_output = model(batched_current, t_tensor, cond_batched)

    scheduler = pipeline.ref.sparse_scheduler_512
    prev_sample_batched, log_prob_vec, prev_mean, std_vec = direct3d_flow_step_with_logprob(
        scheduler=scheduler,
        sample=batched_current,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=batched_prev,
    )
    
    # —— KL 正则（可选）：与禁用适配器的教师分布对比 ——
    kl_vec = torch.zeros_like(log_prob_vec)  # 形状: (B,)
    if config.compute_kl and (not bool(config.deterministic)):
        slat_model = pipeline.get_trainable_model_stage2()
        base_model = slat_model.module if hasattr(slat_model, "module") else slat_model
        with torch.no_grad():
            with base_model.disable_adapter():
                if config.guidance_scale > 1.0 and neg_batched is not None:
                    neg_ref = base_model(batched_current, t_tensor, neg_batched)  # feats: (sumN, C)
                    pos_ref = base_model(batched_current, t_tensor, cond_batched)  # feats: (sumN, C)
                    cfg_ref_feats = neg_ref.feats + float(config.guidance_scale) * (pos_ref.feats - neg_ref.feats)  # (sumN, C)
                    model_output_ref = SparseTensor(coords=batched_current.coords, feats=cfg_ref_feats, layout=list(batched_current.layout))
                else:
                    model_output_ref = base_model(batched_current, t_tensor, cond_batched)

        # 用同一调度步计算教师分布的均值（步级标准差与当前相同）
        _, _, prev_mean_ref, _ = direct3d_flow_step_with_logprob(
            scheduler=scheduler,
            sample=batched_current,
            model_output=model_output_ref,
            timestep=t,
            prev_timestep=t_prev,
            generator=None,
            deterministic=False,
            observed_prev_sample=batched_prev,
        )

        # KL = E[ (μ - μ_ref)^2 / (2 σ^2) ]，按 layout 聚合到 (B,)
        diff_feats = prev_mean.feats - prev_mean_ref.feats  # 形状: (sumN, C)
        kl_list: List[torch.Tensor] = []
        for b, sl in enumerate(prev_mean.layout):
            mean_sq = diff_feats[sl].pow(2).mean()  # 形状: 标量
            denom = (std_vec[b] + 1e-8) ** 2        # 形状: 标量
            kl_b = (mean_sq / (2.0 * denom)).to(mean_sq.dtype)  # 形状: 标量
            kl_list.append(kl_b.unsqueeze(0))  # 形状: (1,)
        kl_vec = torch.cat(kl_list, dim=0)  # 形状: (B,)

    return prev_sample_batched, log_prob_vec, kl_vec


__all__ = [
    "SparseTensor",
    "direct3d_flow_step_with_logprob",
    "direct3d_flow_step_with_logprob_dense",
    
    "compute_log_prob_direct3d_stage1",
    "compute_log_prob_direct3d_stage2",
    "sparse_tensor_cfg_guidance",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
]

