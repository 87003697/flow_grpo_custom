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
import torch

from direct3d_s2.modules import sparse as base_sp  # type: ignore


SparseTensor = base_sp.SparseTensor


@dataclass
class Stage2RuntimeConfig:
    guidance_scale: float
    sigma_min: float
    deterministic: bool


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
    sample: Dict,
    j: int,
    image_conds: Dict[str, torch.Tensor],
    config: Stage2RuntimeConfig,
) -> Tuple[SparseTensor, torch.Tensor, torch.Tensor]:
    prev_samples_batched, log_prob_vec, kl_vec = compute_log_prob_direct3d_stage2_batched(
        pipeline=pipeline,
        samples=[sample],
        j=j,
        image_conds_list=[image_conds],
        config=config,
    )  # shapes: (batch_sparse), (1,), (1,)

    # 从批量结果中取回单个样本
    prev_sample = extract_sparse_tensor_from_batch(prev_samples_batched, batch_idx=0)  # shape: (1, C)
    log_prob = log_prob_vec[0:1]  # shape: (1,)
    kl_div = kl_vec[0:1]  # shape: (1,)
    return prev_sample, log_prob, kl_div
def compute_log_prob_direct3d_stage2_batched(
    pipeline,
    samples: List[Dict],
    j: int,
    image_conds_list: List[Dict[str, torch.Tensor]],
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
    cond_stack = torch.cat([c["cond"] for c in image_conds_list], dim=0)  # shape: (batch_size, P, C)
    cond_batched = cond_stack.to(device=target_device, dtype=target_dtype)  # shape: (batch_size, P, C)
    neg_batched = None
    if config.guidance_scale > 1.0:
        neg_sources = [c.get("neg_cond") for c in image_conds_list]
        if any(n is None for n in neg_sources):
            raise ValueError("CFG 模式下 neg_cond 不应为 None")
        neg_stack = torch.cat([n for n in neg_sources if n is not None], dim=0)  # shape: (batch_size, P, C)
        neg_batched = neg_stack.to(device=target_device, dtype=target_dtype)  # shape: (batch_size, P, C)

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

    prev_sample_batched, log_prob_vec, _, _ = direct3d_flow_step_with_logprob(
        batched_current,
        model_output,
        t,
        t_prev,
        sigma_min=float(config.sigma_min),
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
    "compute_log_prob_direct3d_stage2_batched",
    "sparse_tensor_cfg_guidance",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
]

