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

from typing import Dict, List, Optional, Tuple
import math
import torch

from direct3d_s2.modules import sparse as base_sp  # type: ignore


SparseTensor = base_sp.SparseTensor


def direct3d_flow_step_with_logprob(
    sample: SparseTensor,
    model_output: SparseTensor,
    t: float,
    t_prev: float,
    sigma_min: float = 0.002,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    observed_prev_sample: Optional[SparseTensor] = None,
) -> Tuple[SparseTensor, torch.Tensor, SparseTensor, torch.Tensor]:
    device = sample.coords.device  # shape: () 设备
    batch_size = int(sample.shape[0])  # shape: () 批量大小
    t_tensor = torch.tensor(t, device=device, dtype=torch.float32)  # shape: () 时间标量
    t_prev_tensor = torch.tensor(t_prev, device=device, dtype=torch.float32)  # shape: () 时间标量
    t_norm = torch.clamp(t_tensor / 1000.0, 0.0, 1.0)  # shape: () 归一化时间
    t_prev_norm = torch.clamp(t_prev_tensor / 1000.0, 0.0, 1.0)  # shape: () 归一化时间
    sigma_t = torch.tensor(sigma_min, device=device, dtype=torch.float32) + (1.0 - float(sigma_min)) * t_norm  # shape: () 当前噪声
    sigma_prev = torch.tensor(sigma_min, device=device, dtype=torch.float32) + (1.0 - float(sigma_min)) * t_prev_norm  # shape: () 前一噪声
    dt_sigma = sigma_prev - sigma_t  # shape: () 噪声步长
    x_t = sample.feats  # shape: (N_total, C) 当前特征
    v_t = model_output.feats  # shape: (N_total, C) 模型输出
    coords = sample.coords  # shape: (N_total, 4) 坐标
    dt_time = torch.tensor((t - t_prev) / 1000.0, device=device, dtype=torch.float32)  # shape: () 时间步长
    prev_mean_feats = x_t - dt_time * v_t  # shape: (N_total, C) 均值特征
    prev_mean = SparseTensor(coords=coords, feats=prev_mean_feats)  # shape: (B, C)
    if deterministic:
        prev_sample = SparseTensor(coords=coords, feats=prev_mean_feats)  # shape: (B, C)
        log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (B,)
        std_dev = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (B,)
        return prev_sample, log_prob, prev_mean, std_dev
    if observed_prev_sample is not None:
        prev_sample = observed_prev_sample  # shape: (B, C)
        prev_feats = prev_sample.feats  # shape: (N_total, C)
    else:
        noise = torch.randn_like(x_t)  # shape: (N_total, C)
        if generator is not None:
            noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)  # shape: (N_total, C)
        one_minus_sigma = torch.clamp(1.0 - sigma_t, min=1e-8)  # shape: ()
        std_dev_t = torch.sqrt(sigma_t / one_minus_sigma) * 0.7  # shape: ()
        step_std = std_dev_t * torch.sqrt(torch.clamp(-dt_sigma, min=1e-12))  # shape: ()
        prev_feats = prev_mean_feats + step_std * noise  # shape: (N_total, C)
        prev_sample = SparseTensor(coords=coords, feats=prev_feats)  # shape: (B, C)
    diff = prev_feats - prev_mean_feats  # shape: (N_total, C)
    one_minus_sigma = torch.clamp(1.0 - sigma_t, min=1e-8)  # shape: ()
    std_dev_t = torch.sqrt(sigma_t / one_minus_sigma) * 0.7  # shape: ()
    step_std = std_dev_t * torch.sqrt(torch.clamp(-dt_sigma, min=1e-12))  # shape: ()
    log_prob_per_point = (
        -0.5 * (diff / (step_std + 1e-8)) ** 2
        - torch.log(step_std + 1e-8)
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    )  # shape: (N_total, C)
    log_prob_list: List[torch.Tensor] = []  # shape: (batch_size,)
    for sl in prev_sample.layout:
        vals = log_prob_per_point[sl]  # shape: (N_b, C)
        mean_val = vals.mean() if vals.numel() > 0 else torch.zeros((), device=device, dtype=log_prob_per_point.dtype)  # shape: ()
        log_prob_list.append(mean_val)  # shape: ()
    log_prob = torch.stack(log_prob_list, dim=0)  # shape: (B,)
    std_vec = torch.full((batch_size,), float(step_std), device=device, dtype=torch.float32)  # shape: (B,)
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
    adjusted = []  # shape: (batch_size,)
    for sparse_tensor in sparse_list:
        coords = sparse_tensor.coords.clone()  # shape: (N_i, 4)
        coords[:, 0] = 0  # shape: (N_i, 4)
        adjusted.append(SparseTensor(coords=coords, feats=sparse_tensor.feats))  # shape: (1, C)
    combined = base_sp.sparse_cat(adjusted, dim=0)  # shape: (B, C)
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
    sample: Dict,
    j: int,
    image_conds: Dict[str, torch.Tensor],
    config,
    **kwargs,
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    latents_seq: List[SparseTensor] = sample["latents_seq"]  # shape: [steps+1]
    current_sparse = latents_seq[j]  # shape: (B, C)
    observed_prev_sparse = latents_seq[j + 1]  # shape: (B, C)
    t_seq = sample["t_seq"]  # shape: (steps+1,)
    t = float(t_seq[j])  # shape: ()
    t_prev = float(t_seq[j + 1])  # shape: ()
    cond = image_conds["cond"][sample.get("image_idx", 0):sample.get("image_idx", 0) + 1].to(current_sparse.coords.device)  # shape: (1, P, C)
    neg_cond = image_conds.get("neg_cond")
    if neg_cond is not None:
        neg_cond = neg_cond[sample.get("image_idx", 0):sample.get("image_idx", 0) + 1].to(current_sparse.coords.device)  # shape: (1, P, C)
    guidance_scale = float(config.guidance_scale)  # shape: ()
    model = pipeline.get_trainable_model()  # shape: ()
    t_tensor = torch.tensor([t], device=current_sparse.coords.device, dtype=torch.float32)  # shape: (1,)
    if guidance_scale > 1.0 and neg_cond is not None:
        neg_out = model(current_sparse, t_tensor, neg_cond)  # shape: (B, C)
        pos_out = model(current_sparse, t_tensor, cond)  # shape: (B, C)
        cfg_feats = neg_out.feats + guidance_scale * (pos_out.feats - neg_out.feats)  # shape: (N_total, C)
        model_output = SparseTensor(coords=current_sparse.coords, feats=cfg_feats)  # shape: (B, C)
    else:
        model_output = model(current_sparse, t_tensor, cond)  # shape: (B, C)
    prev_sample, log_prob, _, _ = direct3d_flow_step_with_logprob(
        current_sparse,
        model_output,
        t,
        t_prev,
        sigma_min=float(config.sigma_min),
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=observed_prev_sparse,
    )  # shapes: (B,C), (1,), (...)
    kl_div = torch.zeros_like(log_prob)  # shape: (1,)
    return prev_sample, log_prob, kl_div


def compute_log_prob_direct3d_stage2_batched(
    pipeline,
    samples: List[Dict],
    j: int,
    image_conds_list: List[Dict[str, torch.Tensor]],
    config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    current_list = [s["latents_seq"][j] for s in samples]  # shape: (B,)
    prev_list = [s["latents_seq"][j + 1] for s in samples]  # shape: (B,)
    batched_current = prepare_sparse_tensor_batch(current_list, batch_size=len(samples))  # shape: (B, C)
    batched_prev = prepare_sparse_tensor_batch(prev_list, batch_size=len(samples))  # shape: (B, C)
    cond_batched = torch.cat([c["cond"] for c in image_conds_list], dim=0).to(batched_current.coords.device)  # shape: (B, P, C)
    neg_batched = None
    if float(config.guidance_scale) > 1.0:
        neg_batched = torch.cat([c["neg_cond"] for c in image_conds_list], dim=0).to(batched_current.coords.device)  # shape: (B, P, C)
    t_seq = samples[0]["t_seq"]  # shape: (steps+1,)
    t = float(t_seq[j])  # shape: ()
    t_prev = float(t_seq[j + 1])  # shape: ()
    model = pipeline.get_trainable_model()  # shape: ()
    t_tensor = torch.tensor([t], device=batched_current.coords.device, dtype=torch.float32)  # shape: (1,)
    if float(config.guidance_scale) > 1.0 and neg_batched is not None:
        neg_out = model(batched_current, t_tensor, neg_batched)  # shape: (B, C)
        pos_out = model(batched_current, t_tensor, cond_batched)  # shape: (B, C)
        cfg_feats = neg_out.feats + float(config.guidance_scale) * (pos_out.feats - neg_out.feats)  # shape: (N_total, C)
        model_output = SparseTensor(coords=batched_current.coords, feats=cfg_feats)  # shape: (B, C)
    else:
        model_output = model(batched_current, t_tensor, cond_batched)  # shape: (B, C)
    _, log_prob_vec, _, _ = direct3d_flow_step_with_logprob(
        batched_current,
        model_output,
        t,
        t_prev,
        sigma_min=float(config.sigma_min),
        generator=None,
        deterministic=bool(config.deterministic),
        observed_prev_sample=batched_prev,
    )  # shapes: (_, (B,), _, _)
    kl_vec = torch.zeros_like(log_prob_vec)  # shape: (B,)
    return log_prob_vec, kl_vec


__all__ = [
    "SparseTensor",
    "direct3d_flow_step_with_logprob",
    "compute_log_prob_direct3d_stage2",
    "compute_log_prob_direct3d_stage2_batched",
    "sparse_tensor_cfg_guidance",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
]

