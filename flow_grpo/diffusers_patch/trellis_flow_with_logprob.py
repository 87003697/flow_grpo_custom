#!/usr/bin/env python3
"""
TRELLIS Flow Matching Step with Log Probability for GRPO Training

基于 flow_grpo/diffusers_patch/hunyuan3d_sde_with_logprob.py 的 SDE 理论，
但适配 TRELLIS 的 Flow Matching 框架和 SparseTensor 数据结构。

数学框架:
- TRELLIS 使用 Flow Matching 而非标准扩散 (t: 1.0 → 0.0)
- TRELLIS 时间参数化: t ∈ [0, 1000] (放大1000倍)
- SparseTensor 格式: coords (N, 4) + feats (N, C)
- Flow ODE: dx/dt = v(x_t, t) 其中 v 是速度场

SDE 扩展:
- Deterministic ODE: x_{t-dt} = x_t - dt * v(x_t, t)
- Stochastic SDE: x_{t-dt} = mean + std * noise
- LogProb: -0.5 * ((x - mean) / std)^2 - log(std) - log(√(2π))

参考路径:
- Hunyuan3D SDE: `flow_grpo/diffusers_patch/hunyuan3d_sde_with_logprob.py`
- TRELLIS 采样器: `_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py`
- Flow Matching: `_reference_codes/TRELLIS/trellis/pipelines/samplers/base.py`
- SD3 SDE/LogProb 对应: `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:17-80`
"""
import math
from typing import Optional, Tuple, Union, List, Any

import torch
import numpy as np

from generators.trellis import sparse as sp

from diffusers.utils.torch_utils import randn_tensor


def trellis_flow_step_with_logprob(
    sample: sp.SparseTensor,
    model_output: sp.SparseTensor,
    t: float,
    t_prev: float,
    sigma_min: float = 0.002,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    observed_prev_sample: Optional[sp.SparseTensor] = None,
) -> Tuple[sp.SparseTensor, torch.Tensor, sp.SparseTensor, torch.Tensor]:
    """
    TRELLIS Flow Matching 步骤 + LogProb 计算，适配 SparseTensor 格式
    
    基于 TRELLIS 的 Flow Euler 采样器，但添加了 SDE 随机性和概率密度计算。
    
    参考: 
    - _reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:75-77
    - flow_grpo/diffusers_patch/hunyuan3d_sde_with_logprob.py:25-108
    
    数学推导:
    - ODE: x_{t-Δt} = x_t - Δt * v(x_t, t)
    - SDE: x_{t-Δt} = x_t - Δt * v(x_t, t) + g(t) * sqrt(Δt) * ε, ε ~ N(0, I)
    - 本实现取 g(t) = sigma_t = sigma_min + (1 - sigma_min) * t
    
    Args:
        sample: 当前时间步的 SparseTensor 样本
        model_output: 模型预测的速度场 v(x_t, t)
        t: 当前时间步 (TRELLIS 格式: 0-1000)
        t_prev: 前一时间步 (TRELLIS 格式: 0-1000)
        sigma_min: Flow Matching 最小噪声尺度
        generator: 随机数生成器
        deterministic: 是否使用确定性（ODE）模式
        
    Returns:
        Tuple[sp.SparseTensor, torch.Tensor, sp.SparseTensor, torch.Tensor]:
            - prev_sample: 前一时间步的样本
            - log_prob: 对数概率
            - prev_sample_mean: 分布均值
            - std_dev: 标准差
    """
    # 时间步长与归一化 (使用正向步长 Δt = (t - t_prev)/1000 ≥ 0)
    device = sample.coords.device
    batch_size = int(sample.shape[0])  # 标量
    dt_abs = torch.tensor((t - t_prev) / 1000.0, device=device, dtype=torch.float32)  # 标量
    t_normalized = torch.tensor(t / 1000.0, device=device, dtype=torch.float32)       # 标量，∈ [0, 1]
    
    # 验证输入格式
    assert isinstance(sample, sp.SparseTensor), f"sample 必须是 SparseTensor，得到 {type(sample)}"
    assert isinstance(model_output, sp.SparseTensor), f"model_output 必须是 SparseTensor，得到 {type(model_output)}"
    assert sample.coords.shape[0] == model_output.coords.shape[0], "样本和模型输出的点数必须相同"
    assert torch.allclose(sample.coords, model_output.coords), "样本和模型输出的坐标必须相同"
    
    # 提取特征进行计算
    x_t = sample.feats       # (N, C)
    v_t = model_output.feats # (N, C)
    coords = sample.coords   # (N, 4)
    
    # 噪声调度：sigma_t ∈ [sigma_min, 1]
    sigma_t = torch.tensor(sigma_min, device=device, dtype=torch.float32) + (1 - float(sigma_min)) * t_normalized  # 标量
    
    # 漂移项（与 ODE 完全一致）：mean = x_t - Δt * v_t
    prev_sample_mean_feats = x_t - dt_abs * v_t  # (N, C)
    prev_sample_mean = sp.SparseTensor(coords=coords, feats=prev_sample_mean_feats)
    
    if deterministic:
        # ODE：无噪声，log_prob=0
        prev_sample_feats = prev_sample_mean_feats  # (N, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_sample_feats)
        log_prob = torch.zeros(batch_size, device=device)  # (B,)
        std_dev = torch.zeros(batch_size, device=device)   # (B,)
        return prev_sample, log_prob, prev_sample_mean, std_dev
    
    # SDE：添加扩散项（g(t) = sigma_t）
    # 噪声强度 noise_strength = g(t) * sqrt(Δt)
    noise_strength = sigma_t * torch.sqrt(torch.clamp(dt_abs, min=1e-8))  # 标量

    # 如果提供了观测到的上一步样本，则使用其特征计算对数概率（用于训练期单步重算）
    if observed_prev_sample is not None:
        # 验证坐标一致
        assert torch.allclose(observed_prev_sample.coords, coords), "observed_prev_sample 的坐标必须与当前样本一致"
        prev_sample_feats = observed_prev_sample.feats
        prev_sample = observed_prev_sample
    else:
        # 采样噪声（与传入 generator 对齐）
        if generator is None:
            variance_noise = torch.randn_like(x_t)
        else:
            variance_noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)  # (N, C)
        # 生成随机样本
        prev_sample_feats = prev_sample_mean_feats + noise_strength * variance_noise  # (N, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_sample_feats)

    # 高斯对数概率（逐 batch 聚合为标量）
    diff = prev_sample_feats.detach() - prev_sample_mean_feats  # (N, C)
    log_prob_per_point = (
        -0.5 * (diff / (noise_strength + 1e-8))**2
        - torch.log(noise_strength + 1e-8)
        - 0.5 * torch.log(2 * torch.tensor(math.pi, device=device))
    )  # (N, C)

    # 按 batch 聚合
    log_prob_list = []
    layout = prev_sample.layout  # List[slice]，长度 B
    for b in range(batch_size):
        sl = layout[b]
        log_prob_b = log_prob_per_point[sl].mean()  # 标量
        log_prob_list.append(log_prob_b)
    log_prob = torch.stack(log_prob_list, dim=0)  # (B,)
    
    std_dev = torch.full((batch_size,), float(noise_strength), device=device)  # (B,)
    return prev_sample, log_prob, prev_sample_mean, std_dev


def trellis_flow_euler_sampler_with_logprob(
    model,
    noise: sp.SparseTensor,
    cond: torch.Tensor,
    steps: int = 50,
    sigma_min: float = 0.002,
    rescale_t: float = 1.0,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    guidance_scale: float = 1.0,
    neg_cond: Optional[torch.Tensor] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[sp.SparseTensor, List[sp.SparseTensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    TRELLIS Flow Euler 采样器 + LogProb 计算的完整实现
    
    基于 _reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py，
    但添加了 GRPO 训练所需的对数概率跟踪。
    
    参考: 
    - _reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:80-119
    - flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py:124-234
    
    Args:
        model: TRELLIS SLatFlowModel
        noise: 初始噪声 SparseTensor
        cond: 正面条件
        steps: 采样步数
        sigma_min: 最小噪声尺度
        rescale_t: 时间重新缩放因子
        generator: 随机数生成器
        deterministic: 确定性模式
        guidance_scale: CFG 引导强度
        neg_cond: 负面条件（CFG）
        verbose: 显示进度条
        
    Returns:
        Tuple: (final_sample, all_latents, all_log_probs, all_kl)
    """
    sample = noise  # batched SparseTensor（批次大小由 coords[:,0] 决定）
    
    # TRELLIS 时间步序列 (1.0 → 0.0，放大1000倍)
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # (steps+1,)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)  # 重新缩放
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]  # 长度 steps
    
    # 存储结果
    all_latents_batched = [sample]  # 长度 steps+1，batched SparseTensor
    all_log_probs_batched: List[torch.Tensor] = []  # 每步一个 (B,)
    all_kl_batched: List[torch.Tensor] = []        # 每步一个 (B,)
    
    # CFG 设置
    do_classifier_free_guidance = guidance_scale > 1.0 and neg_cond is not None
    
    if verbose:
        from tqdm import tqdm
        t_pairs_iter = tqdm(t_pairs, desc="TRELLIS Flow Sampling with LogProb")
    else:
        t_pairs_iter = t_pairs

    # 纯 ODE 分支：严格对齐官方实现（无需 SDE/logprob）
    if deterministic:
        for t, t_prev in t_pairs_iter:
            batch_size = int(sample.shape[0])  # 标量
            t_tensor = torch.tensor([t] * batch_size, device=sample.coords.device, dtype=torch.float32)  # (B,)

            if do_classifier_free_guidance:
                with torch.no_grad():
                    neg_c = neg_cond
                    if neg_c is not None and neg_c.shape[0] == 1 and batch_size > 1:
                        neg_c = neg_c.repeat(batch_size, *([1] * (neg_c.dim() - 1)))  # (B, ...)
                    neg_output = model(sample, t_tensor, neg_c, **kwargs)
                with torch.no_grad():
                    pos_c = cond
                    if pos_c is not None and pos_c.shape[0] == 1 and batch_size > 1:
                        pos_c = pos_c.repeat(batch_size, *([1] * (pos_c.dim() - 1)))  # (B, ...)
                    pos_output = model(sample, t_tensor, pos_c, **kwargs)
                cfg_output_feats = (
                    neg_output.feats + guidance_scale * (pos_output.feats - neg_output.feats)
                )  # (N, C)
                model_output = sp.SparseTensor(coords=sample.coords, feats=cfg_output_feats)
            else:
                with torch.no_grad():
                    pos_c = cond
                    if pos_c is not None and pos_c.shape[0] == 1 and batch_size > 1:
                        pos_c = pos_c.repeat(batch_size, *([1] * (pos_c.dim() - 1)))  # (B, ...)
                    model_output = model(sample, t_tensor, pos_c, **kwargs)

            # Δt = (t - t_prev)/1000 ≥ 0
            dt_abs = torch.tensor((t - t_prev) / 1000.0, device=sample.coords.device, dtype=torch.float32)  # 标量
            prev_sample = sp.SparseTensor(coords=sample.coords, feats=sample.feats - dt_abs * model_output.feats)  # batched SparseTensor

            sample = prev_sample
            all_latents_batched.append(sample)
            all_log_probs_batched.append(torch.zeros(batch_size, device=sample.coords.device))  # (B,)
            all_kl_batched.append(torch.zeros(batch_size, device=sample.coords.device))        # (B,)

        # 展平为 per-sample 输出
        batch_size = int(sample.shape[0])
        latents_flat: List[sp.SparseTensor] = []
        log_probs_flat: List[torch.Tensor] = []
        kl_flat: List[torch.Tensor] = []
        for b in range(batch_size):
            for step_idx in range(len(all_latents_batched)):
                latents_flat.append(all_latents_batched[step_idx][b])
            for step_idx in range(len(all_log_probs_batched)):
                log_probs_flat.append(all_log_probs_batched[step_idx][b])
                kl_flat.append(all_kl_batched[step_idx][b])

        return sample, latents_flat, log_probs_flat, kl_flat

    # 随机/SDE 分支：沿用带 logprob 的单步函数
    for t, t_prev in t_pairs_iter:
        # 时间步张量（TRELLIS 格式）
        batch_size = int(sample.shape[0])  # 标量
        t_tensor = torch.tensor([t] * batch_size, device=sample.coords.device, dtype=torch.float32)  # (B,)
        
        # ===========================================
        # CFG 模型预测
        # ===========================================
        
        if do_classifier_free_guidance:
            # SparseTensor CFG 处理：分别推理正负条件
            # 由于 SparseTensor 的稀疏结构，我们采用分别推理的方式
            
            # 负面条件推理
            with torch.no_grad():
                neg_c = neg_cond
                if neg_c is not None and neg_c.shape[0] == 1 and batch_size > 1:
                    neg_c = neg_c.repeat(batch_size, *([1] * (neg_c.dim() - 1)))  # (B, ...)
                neg_output = model(sample, t_tensor, neg_c, **kwargs)
             
            # 正面条件推理  
            with torch.no_grad():
                pos_c = cond
                if pos_c is not None and pos_c.shape[0] == 1 and batch_size > 1:
                    pos_c = pos_c.repeat(batch_size, *([1] * (pos_c.dim() - 1)))  # (B, ...)
                pos_output = model(sample, t_tensor, pos_c, **kwargs)
             
            # CFG 合并: output = neg + guidance_scale * (pos - neg)
            cfg_output_feats = (
                neg_output.feats + guidance_scale * (pos_output.feats - neg_output.feats)
            )  # shape: (N, C)
             
            model_output = sp.SparseTensor(
                coords=sample.coords,
                feats=cfg_output_feats
            )  # batched SparseTensor
        else:
            # 无 CFG 的直接推理
            with torch.no_grad():
                pos_c = cond
                if pos_c is not None and pos_c.shape[0] == 1 and batch_size > 1:
                    pos_c = pos_c.repeat(batch_size, *([1] * (pos_c.dim() - 1)))  # (B, ...)
                    
                model_output = model(sample, t_tensor, pos_c, **kwargs)
         
        # Flow 步骤 + LogProb
        sample, log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
            sample=sample,
            model_output=model_output,
            t=t,
            t_prev=t_prev,
            sigma_min=sigma_min,
            generator=generator,
            deterministic=False,
        )
         
        # 存储结果
        all_latents_batched.append(sample)
        all_log_probs_batched.append(log_prob)                    # (B,)
        all_kl_batched.append(torch.zeros_like(log_prob))         # (B,)
     
    # 展平为 per-sample 输出
    batch_size = int(sample.shape[0])
    latents_flat: List[sp.SparseTensor] = []
    log_probs_flat: List[torch.Tensor] = []
    kl_flat: List[torch.Tensor] = []
    for b in range(batch_size):
        for step_idx in range(len(all_latents_batched)):
            latents_flat.append(all_latents_batched[step_idx][b])
        for step_idx in range(len(all_log_probs_batched)):
            log_probs_flat.append(all_log_probs_batched[step_idx][b])
            kl_flat.append(all_kl_batched[step_idx][b])

    return sample, latents_flat, log_probs_flat, kl_flat