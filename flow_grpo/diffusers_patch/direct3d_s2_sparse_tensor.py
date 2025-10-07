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
from typing import Dict, List, Optional, Tuple
import math
import sys
from pathlib import Path
import torch

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
    batch_size = int(sample.shape[0])  # shape: () 批量大小

    # --- 调度器 sigma 信息（参考 SD3 实现） ---
    sigmas = scheduler.sigmas.to(dtype=torch.float32)
    sigmas_len = int(sigmas.shape[0])

    timesteps_attr = getattr(scheduler, "timesteps", None)
    if timesteps_attr is not None:
        schedule = timesteps_attr.to(dtype=torch.float32)
    else:
        schedule = torch.linspace(1.0, 0.0, sigmas_len, dtype=torch.float32)

    if schedule.device != sigmas.device:
        schedule = schedule.to(sigmas.device)

    t_scalar = float(timestep)
    t_tensor = torch.tensor(t_scalar, device=schedule.device, dtype=schedule.dtype)
    idx_tensor = torch.argmin((schedule - t_tensor).abs())
    step_index = int(idx_tensor.item())
    step_index = max(0, min(step_index, sigmas_len - 2))
    next_index = min(step_index + 1, sigmas_len - 1)

    sigma = sigmas[step_index].to(device)
    sigma_prev = sigmas[next_index].to(device)
    sigma_max = sigmas[1 if sigmas_len > 1 else 0].to(device)

    ones_like_sigma = torch.ones_like(sigma)
    sigma_safe = torch.clamp(sigma, min=1e-8)
    sigma_cmp = torch.where(torch.isclose(sigma, ones_like_sigma), sigma_max, torch.clamp(sigma_safe, max=1 - 1e-8))

    std_dev_t = torch.sqrt(sigma_safe / (1 - sigma_cmp)) * noise_level
    dt = sigma_prev - sigma
    step_std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))

    # --- 漂移项（复用 SD3 推导公式） ---
    sample_feats = sample.feats.float()
    model_feats = model_output.feats.float()
    coords = sample.coords
    orig_dtype = sample.feats.dtype

    std_sq = std_dev_t ** 2
    sigma_eps = torch.clamp(sigma_safe, min=1e-8)
    coeff_sample = 1 + (std_sq / (2 * sigma_eps)) * dt
    coeff_model = (1 + std_sq * (1 - sigma_eps) / (2 * sigma_eps)) * dt
    prev_mean_feats_fp32 = sample_feats * coeff_sample + model_feats * coeff_model
    prev_mean = SparseTensor(coords=coords, feats=prev_mean_feats_fp32.to(orig_dtype))  # shape: (B, C)

    if deterministic:
        prev_sample = SparseTensor(coords=coords, feats=prev_mean_feats_fp32.to(orig_dtype))
        log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)
        std_dev = torch.zeros(batch_size, device=device, dtype=torch.float32)
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
        prev_sample = SparseTensor(coords=coords, feats=prev_feats_fp32.to(orig_dtype))

    diff = prev_feats_fp32.detach() - prev_mean_feats_fp32  # shape: (N_total, C)
    noise_scale = torch.clamp(step_std, min=1e-12)
    log_prob_per_point = (
        -0.5 * (diff / noise_scale) ** 2
        - torch.log(noise_scale)
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    )
    log_prob_list: List[torch.Tensor] = []  # shape: (batch_size,)
    for sl in prev_sample.layout:
        vals = log_prob_per_point[sl]  # shape: (N_b, C)
        mean_val = vals.mean() if vals.numel() > 0 else torch.zeros((), device=device, dtype=log_prob_per_point.dtype)  # shape: ()
        log_prob_list.append(mean_val)  # shape: ()
    log_prob = torch.stack(log_prob_list, dim=0)  # shape: (B,)
    std_vec = torch.full((batch_size,), float(step_std.detach().cpu().item()), device=device, dtype=torch.float32)  # shape: (B,)
    return prev_sample, log_prob, prev_mean, std_vec


def sparse_tensor_cfg_guidance(
    positive_sparse: SparseTensor,
    negative_sparse: SparseTensor,
    guidance_scale: float,
) -> SparseTensor:
    cfg_feats = negative_sparse.feats + guidance_scale * (positive_sparse.feats - negative_sparse.feats)  # shape: (N_total, C)
    cfg_tensor = SparseTensor(coords=positive_sparse.coords, feats=cfg_feats)  # shape: (B, C)
    return cfg_tensor


def prepare_sparse_tensor_batch(
    sparse_list: List[SparseTensor],
    batch_size: int,
) -> SparseTensor:
    if len(sparse_list) != batch_size:
        raise ValueError("SparseTensor 列表长度与 batch_size 不一致")

    coords_chunks: List[torch.Tensor] = []  # shape: (总块数,)
    feats_chunks: List[torch.Tensor] = []   # shape: (总块数,)
    layout_slices: List[slice] = []         # shape: (总块数,)

    feature_offset = 0  # 标量，累计特征下标
    block_offset = 0    # 标量，累计块索引

    for sparse_tensor in sparse_list:
        coords = sparse_tensor.coords  # shape: (N_i, 4)
        feats = sparse_tensor.feats    # shape: (N_i, C)
        layout = sparse_tensor.layout  # List[slice]

        for sl in layout:
            coords_block = coords[sl].clone()  # shape: (M, 4)
            feats_block = feats[sl]            # shape: (M, C)

            block_size = coords_block.shape[0]
            if block_size == 0:
                continue

            coords_block[:, 0] = int(block_offset)

            coords_chunks.append(coords_block)
            feats_chunks.append(feats_block)

            layout_slices.append(slice(feature_offset, feature_offset + block_size))

            feature_offset += block_size
            block_offset += 1

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
    return SparseTensor(coords=coords, feats=feats)


def compute_log_prob_direct3d_stage2(
    pipeline,
    samples: List[Dict],
    j: int,
    config: Stage2RuntimeConfig,
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
    model = pipeline.get_trainable_model()

    t_tensor = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)
    if config.guidance_scale > 1.0 and neg_batched is not None:
        neg_out = model(batched_current, t_tensor, neg_batched)
        pos_out = model(batched_current, t_tensor, cond_batched)
        cfg_feats = neg_out.feats + config.guidance_scale * (pos_out.feats - neg_out.feats)
        model_output = SparseTensor(coords=batched_current.coords, feats=cfg_feats)
    else:
        model_output = model(batched_current, t_tensor, cond_batched)

    scheduler = pipeline.ref.sparse_scheduler_512
    prev_sample_batched, log_prob_vec, _, _ = direct3d_flow_step_with_logprob(
        scheduler=scheduler,
        sample=batched_current,
        model_output=model_output,
        timestep=t,
        prev_timestep=t_prev,
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=batched_prev,
    )

    kl_vec = torch.zeros_like(log_prob_vec)
    return prev_sample_batched, log_prob_vec, kl_vec


__all__ = [
    "SparseTensor",
    "direct3d_flow_step_with_logprob",
    "compute_log_prob_direct3d_stage2",
    "sparse_tensor_cfg_guidance",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
]

