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
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def create_trellis_scheduler(
    steps: int,
    device: Union[str, torch.device] = 'cpu',
    rescale_t: float = 1.0,
) -> FlowMatchEulerDiscreteScheduler:
    """
    创建与 TRELLIS 官方完全兼容的 scheduler
    
    通过直接覆盖 scheduler.sigmas，确保与 TRELLIS 的时间序列完全一致（0% 误差）。
    
    Args:
        steps: 采样步数
        device: 设备
        rescale_t: TRELLIS 的时间重新缩放因子（默认 1.0）
    
    Returns:
        FlowMatchEulerDiscreteScheduler: 配置好的 scheduler
    
    Example:
        >>> scheduler = create_trellis_scheduler(steps=30, device='cuda')
        >>> sigma = scheduler.sigmas[0]  # 1.0
        >>> sigma_next = scheduler.sigmas[1]  # 0.9667
    """
    # 生成 TRELLIS 的时间序列（与官方完全一致）
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # shape: (steps+1,)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)  # 重新缩放
    
    # 转换为 sigmas（归一化到 [0,1]）
    sigmas_trellis = torch.from_numpy(t_seq / 1000.0).to(device=device, dtype=torch.float32)  # shape: (steps+1,)
    
    # 创建 scheduler 并覆盖 sigmas
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.sigmas = sigmas_trellis  # shape: (steps+1,)
    scheduler.timesteps = sigmas_trellis * 1000  # TRELLIS 时间格式 (0-1000)
    scheduler.num_inference_steps = steps  # shape: ()
    
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
    """
    TRELLIS Flow Matching 步骤 + LogProb 计算，对齐 Direct3D-S2 的 SDE 实现
    
    修正说明：
    - 对齐 Direct3D-S2 的完整 SDE 漂移项公式（包含二阶修正）
    - 从 scheduler.sigmas 查表获取 sigma，而非手动插值
    - 使用统一的 sigma 域 dt 计算
    
    参考: 
    - flow_grpo/diffusers_patch/direct3d_s2_sparse_tensor.py:41-131 (标准实现)
    - flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:11-70 (原始 SD3 SDE)
    
    Args:
        scheduler: FlowMatchEulerDiscreteScheduler 调度器
        sample: 当前时间步的 SparseTensor 样本
        model_output: 模型预测的速度场 v(x_t, t)
        timestep: 当前时间步 (TRELLIS 格式: 0-1000)
        prev_timestep: 前一时间步 (TRELLIS 格式: 0-1000)
        generator: 随机数生成器
        deterministic: 是否使用确定性（ODE）模式
        observed_prev_sample: 观测到的上一步样本（训练时用于 teacher forcing）
        noise_level: 噪声级别（默认 0.7，与 SD3/Direct3D-S2 一致）
        
    Returns:
        Tuple[sp.SparseTensor, torch.Tensor, sp.SparseTensor, torch.Tensor]:
            - prev_sample: 前一时间步的样本
            - log_prob: 对数概率 (B,)
            - prev_sample_mean: 分布均值
            - std_dev: 标准差 (B,)
    """
    # 从 scheduler 获取 sigma（对齐 Direct3D-S2）
    device = sample.feats.device  # shape: () 设备
    batch_size = int(sample.shape[0])  # shape: () 批量大小

    # 调度器 sigma 信息（参考 Direct3D-S2 实现）
    sigmas = scheduler.sigmas.to(dtype=torch.float32)  # shape: (sigmas_len,)
    sigmas_len = int(sigmas.shape[0])  # shape: ()

    timesteps_attr = getattr(scheduler, "timesteps", None)
    if timesteps_attr is not None:
        schedule = timesteps_attr.to(dtype=torch.float32)  # shape: (sigmas_len,)
    else:
        schedule = torch.linspace(1.0, 0.0, sigmas_len, dtype=torch.float32)  # shape: (sigmas_len,)

    if schedule.device != sigmas.device:
        schedule = schedule.to(sigmas.device)

    # 查表获取 sigma（TRELLIS 时间参数化：t ∈ [0, 1000]，需归一化到 [0, 1]）
    t_normalized = float(timestep) / 1000.0  # shape: ()
    t_prev_normalized = float(prev_timestep) / 1000.0  # shape: ()
    
    t_tensor = torch.tensor(t_normalized, device=schedule.device, dtype=schedule.dtype)  # shape: ()
    idx_tensor = torch.argmin((schedule - t_tensor).abs())  # shape: ()
    step_index = int(idx_tensor.item())  # shape: ()
    step_index = max(0, min(step_index, sigmas_len - 2))  # shape: ()
    next_index = min(step_index + 1, sigmas_len - 1)  # shape: ()

    sigma = sigmas[step_index].to(device)  # shape: ()
    sigma_prev = sigmas[next_index].to(device)  # shape: ()
    sigma_max = sigmas[1 if sigmas_len > 1 else 0].to(device)  # shape: ()

    # dt 计算（sigma 域，≤ 0）
    dt = sigma_prev - sigma  # shape: ()
    
    # 提取特征进行计算
    sample_feats = sample.feats.float()  # shape: (N, C)
    model_feats = model_output.feats.float()  # shape: (N, C)
    coords = sample.coords  # shape: (N, 4)
    orig_dtype = sample.feats.dtype  # shape: ()

    if deterministic:
        # 纯 ODE：严格对齐 TRELLIS 官方实现
        # 参考: flow_euler.py:76  pred_x_prev = x_t - (t - t_prev) * pred_v
        # 注意：TRELLIS 用时间域 dt，我们用 sigma 域 dt（符号相反）
        # sigma 域：x_next = x + dt_sigma * v （dt_sigma < 0）
        prev_feats_ode = sample_feats + dt * model_feats  # shape: (N, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_feats_ode.to(orig_dtype))  # shape: (N, C)
        prev_sample_mean = prev_sample
        log_prob = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (B,)
        std_dev = torch.zeros(batch_size, device=device, dtype=torch.float32)  # shape: (B,)
        return prev_sample, log_prob, prev_sample_mean, std_dev

    # SDE 模式：使用 Direct3D-S2 风格的公式
    # 安全处理 sigma
    ones_like_sigma = torch.ones_like(sigma)  # shape: ()
    sigma_safe = torch.clamp(sigma, min=1e-8)  # shape: ()
    sigma_cmp = torch.where(
        torch.isclose(sigma, ones_like_sigma), 
        sigma_max, 
        torch.clamp(sigma_safe, max=1 - 1e-8)
    )  # shape: ()

    # 瞬时标准差
    std_dev_t = torch.sqrt(sigma_safe / (1 - sigma_cmp)) * noise_level  # shape: ()
    
    # 步级标准差（sigma 域）
    step_std = std_dev_t * torch.sqrt(torch.clamp(-dt, min=1e-12))  # shape: ()

    # SDE 漂移项（包含二阶修正）
    std_sq = std_dev_t ** 2  # shape: ()
    sigma_eps = torch.clamp(sigma_safe, min=1e-8)  # shape: ()
    coeff_sample = 1 + (std_sq / (2 * sigma_eps)) * dt  # shape: ()
    coeff_model = (1 + std_sq * (1 - sigma_eps) / (2 * sigma_eps)) * dt  # shape: ()
    prev_mean_feats_fp32 = sample_feats * coeff_sample + model_feats * coeff_model  # shape: (N, C)
    prev_sample_mean = sp.SparseTensor(coords=coords, feats=prev_mean_feats_fp32.to(orig_dtype))  # shape: (N, C)
    
    # SDE 采样（对齐 Direct3D-S2）
    if observed_prev_sample is not None:
        prev_sample = observed_prev_sample
        prev_feats_fp32 = prev_sample.feats.float()  # shape: (N, C)
    else:
        if generator is None:
            variance_noise = torch.randn_like(sample_feats)  # shape: (N, C)
        else:
            variance_noise = torch.randn(
                sample_feats.shape, 
                device=device, 
                dtype=sample_feats.dtype, 
                generator=generator
            )  # shape: (N, C)
        prev_feats_fp32 = prev_mean_feats_fp32 + step_std * variance_noise  # shape: (N, C)
        prev_sample = sp.SparseTensor(coords=coords, feats=prev_feats_fp32.to(orig_dtype))  # shape: (N, C)

    # 对数概率计算
    diff = prev_feats_fp32.detach() - prev_mean_feats_fp32  # shape: (N, C)
    noise_scale = torch.clamp(step_std, min=1e-12)  # shape: ()
    log_prob_per_point = (
        -0.5 * (diff / noise_scale) ** 2
        - torch.log(noise_scale)
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))
    )  # shape: (N, C)

    # 按 batch 聚合（对齐 Direct3D-S2 的 layout 处理）
    log_prob_list: List[torch.Tensor] = []  # shape: (batch_size,)
    for sl in prev_sample.layout:
        vals = log_prob_per_point[sl]  # shape: (N_b, C)
        mean_val = vals.mean() if vals.numel() > 0 else torch.zeros(
            (), device=device, dtype=log_prob_per_point.dtype
        )  # shape: ()
        log_prob_list.append(mean_val)  # shape: ()
    log_prob = torch.stack(log_prob_list, dim=0)  # shape: (B,)
    
    std_vec = torch.full(
        (batch_size,), 
        float(step_std.detach().cpu().item()), 
        device=device, 
        dtype=torch.float32
    )  # shape: (B,)
    return prev_sample, log_prob, prev_sample_mean, std_vec


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
    kl_reward: float = 0.0,
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
    device = sample.coords.device
    
    # 创建 TRELLIS 兼容的 scheduler（与官方完全一致）
    scheduler = create_trellis_scheduler(steps=steps, device=device, rescale_t=rescale_t)
    
    # TRELLIS 时间步序列（从 scheduler 中提取）
    t_seq = (scheduler.timesteps.cpu().numpy()).astype(np.float32)  # shape: (steps+1,)
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
            scheduler=scheduler,
            sample=sample,
            model_output=model_output,
            timestep=t,
            prev_timestep=t_prev,
            generator=generator,
            deterministic=False,
        )
        
        # KL 奖励（可选，参考 SD3/Hunyuan3D）：与禁用适配器的教师分布进行比较
        kl_vec = torch.zeros_like(log_prob)  # (B,)
        if (kl_reward > 0.0) and (not deterministic):
            base_model = model.module if hasattr(model, "module") else model  # ()
            # 教师前向：在禁用适配器的上下文中计算参考输出
            with torch.no_grad():
                with base_model.disable_adapter():
                    if do_classifier_free_guidance:
                        # 负面条件参考输出  # feats 形状: (sum(N_b), C)
                        neg_ref = base_model(sample, t_tensor, neg_cond, **kwargs)
                        # 正面条件参考输出  # feats 形状: (sum(N_b), C)
                        pos_ref = base_model(sample, t_tensor, cond, **kwargs)
                        # CFG 合并  # 形状: (N, C)
                        cfg_ref_feats = (
                            neg_ref.feats + guidance_scale * (pos_ref.feats - neg_ref.feats)
                        )  # (N, C)
                        model_output_ref = sp.SparseTensor(coords=sample.coords, feats=cfg_ref_feats)  # (N, C)
                    else:
                        # 无 CFG 的参考输出  # 形状: (N, C)
                        model_output_ref = base_model(sample, t_tensor, cond, **kwargs)  # SparseTensor
            
            # 计算参考分布的均值与方差（与当前相同步长）
            _, _, prev_mean_ref, std_ref = trellis_flow_step_with_logprob(
                scheduler=scheduler,
                sample=sample,  # batched SparseTensor
                model_output=model_output_ref,  # SparseTensor
                timestep=t,  # 标量
                prev_timestep=t_prev,  # 标量
                generator=generator,  # 生成器或 None
                deterministic=False,  # 随机分支
            )
            # KL = E[ (μ - μ_ref)^2 / (2 σ^2) ]，按 batch 聚合
            diff_feats = sample_mean.feats - prev_mean_ref.feats  # (N, C)
            layout = sample_mean.layout  # List[slice]，长度 B
            kl_list = []
            for b in range(batch_size):
                sl = layout[b]  # 切片
                mean_sq = diff_feats[sl].pow(2).mean()  # 标量 ()
                denom = (std_dev[b] + 1e-8) ** 2  # 标量 ()
                kl_b = (mean_sq / (2.0 * denom)).to(mean_sq.dtype)  # 标量 ()
                kl_list.append(kl_b.unsqueeze(0))  # (1,)
            kl_vec = torch.cat(kl_list, dim=0)  # (B,)

        # 存储结果
        all_latents_batched.append(sample)
        all_log_probs_batched.append(log_prob)  # (B,)
        all_kl_batched.append(kl_vec)  # (B,)
     
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