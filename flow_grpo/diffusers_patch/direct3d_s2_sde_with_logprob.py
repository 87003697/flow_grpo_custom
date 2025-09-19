import math
from typing import Optional, Tuple

import torch


def sde_step_with_logprob(
    prev_mean: torch.Tensor,
    t_cur: torch.Tensor,
    t_prev: torch.Tensor,
    rescale_t: float,
    sigma_min: float,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct3D‑S2 单步 SDE + log_prob（贴近 SD3 的简洁风格）。

    x = μ + s·ε，ε ~ N(0,I)；log p = -0.5·((x-μ)/s)^2 - log(s) - 0.5·log(2π)
    sigma(t) = sigma_min + (1 - sigma_min)·clamp(t/rescale_t, 0, 1)
    std_dev_t = sqrt(sigma/(1 - sigma))·0.7；s = std_dev_t·sqrt(max(-(sigma_prev - sigma_t), 1e-12))
    返回：
    - prev_sample: 采样结果（同 prev_mean 形状）
    - log_prob(1,): 基于 eps 写法的对数概率（均值聚合后标量）
    - noise_strength(()): 步级噪声尺度 = std_dev_t*sqrt(-dt_sigma)
    - sq_sum(1,): (x - μ)^2 的元素和
    - n_dims(1,): 元素总数（float32 标量张量）
    """

    device = prev_mean.device  # 设备标量
    target_dtype = prev_mean.dtype  # 数据类型标量

    # ---- 时间 → sigma 域映射与步级标准差（fp32） ----
    t_cur_f = t_cur.to(device=device, dtype=torch.float32)  # 标量 ()
    t_prev_f = t_prev.to(device=device, dtype=torch.float32)  # 标量 ()

    # 归一化时间到 [0,1]
    t_norm = torch.clamp(t_cur_f / float(rescale_t), 0.0, 1.0)  # 标量 ()
    t_prev_norm = torch.clamp(t_prev_f / float(rescale_t), 0.0, 1.0)  # 标量 ()

    # 定义 sigma(t) 的线性调度，并构造 sigma 域步长
    sigma_t = float(sigma_min) + (1.0 - float(sigma_min)) * t_norm  # 标量 ()
    sigma_prev = float(sigma_min) + (1.0 - float(sigma_min)) * t_prev_norm  # 标量 ()
    dt_sigma = sigma_prev - sigma_t  # 标量 (≤0)

    # Hunyuan/SD3 风格瞬时尺度：std_dev_t = sqrt(sigma/(1-sigma)) * 0.7
    one_minus_sigma = torch.clamp(1.0 - sigma_t, min=1e-8)  # 标量 ()
    std_dev_t = torch.sqrt(sigma_t / one_minus_sigma) * 0.7  # 标量 ()

    # 步级标准差：s = std_dev_t * sqrt(-dt_sigma)
    step_std = std_dev_t * torch.sqrt(torch.clamp(-dt_sigma, min=1e-12))  # 标量 ()

    # ---- 分布均值（与 TRELLIS 对齐语义）----
    prev_sample_mean = prev_mean  # 同 prev_mean 形状

    # ---- ODE/确定性或退化情形：返回均值与零 log_prob ----
    if bool(deterministic) or (float(step_std.item()) <= 1e-12):
        noise_strength = step_std.to(torch.float32).view(())  # 标量 ()
        return (
            prev_sample_mean,
            torch.zeros((1,), device=device, dtype=torch.float32),
            noise_strength,
            torch.zeros((1,), device=device, dtype=torch.float32),
            torch.tensor(float(prev_mean.numel()), device=device, dtype=torch.float32).view(1),
        )

    # ---- 采样或复用给定的 prev_sample ----
    if prev_sample is None:
        # 某些 torch 版本的 randn_like 不支持 generator 参数；改用 randn 明确传入形状
        noise = torch.randn(prev_mean.shape, device=device, dtype=torch.float32)  # 同 prev_mean 形状
        if generator is not None:
            noise = torch.randn(prev_mean.shape, device=device, dtype=torch.float32, generator=generator)  # 同 prev_mean 形状
        prev_sample = (prev_sample_mean.to(torch.float32) + step_std * noise).to(target_dtype)  # 同 prev_mean 形状

    # ---- 对数概率（eps-form，SUM over dims）→ (1,)
    # eps_hat = (x - mu) / s；log p(eps_hat) = -0.5 * (||eps_hat||^2 + D * log(2π))
    diff = (prev_sample.to(torch.float32) - prev_sample_mean.to(torch.float32))  # 同 prev_mean 形状
    s_safe = torch.clamp(step_std.to(torch.float32), min=1e-12)  # 标量 ()
    log_2pi = torch.log(torch.tensor(2.0 * math.pi, device=device, dtype=torch.float32))  # 标量 ()
    D = torch.tensor(float(diff.numel()), device=device, dtype=torch.float32)  # 标量 ()
    eps_hat_sq_sum = ((diff / s_safe) ** 2).sum()  # 标量 ()
    log_prob = (-0.5 * (eps_hat_sq_sum + D * log_2pi)).view(1)  # (1,)
    sq_sum = (diff.to(torch.float32) ** 2).sum().view(1)  # (1,)
    noise_strength = step_std.to(torch.float32).view(())  # 标量 ()
    n_dims = torch.tensor(float(diff.numel()), device=device, dtype=torch.float32).view(1)  # (1,)

    out_std = step_std.to(torch.float32).view(1)  # (1,)
    return prev_sample, log_prob.to(torch.float32), noise_strength, sq_sum, n_dims



