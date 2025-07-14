"""
Hunyuan3D SDE Step with Log Probability for GRPO Training

Based on complex SDE theory similar to SD3, but adapted for Hunyuan3D's reversed timesteps.

Mathematical Framework:
- Hunyuan3D uses reversed timesteps (1000→0, sigmas: 1.0→0.0)
- Complex SDE formulation with noise scaling and drift correction
- dt = sigma_next - sigma is negative (sigma_next < sigma)
- SDE theory provides proper log probability computation for GRPO

SDE Formulation:
- std_dev_t = f(sigma) based on theoretical noise schedule
- prev_sample_mean = complex drift term with SDE corrections
- noise_term = std_dev_t * sqrt(|dt|) * noise
- deterministic mode degenerates to simple ODE: sample + dt * model_output
"""
import math
from typing import Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def hunyuan3d_sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Predict the sample from the previous timestep using SDE theory adapted for Hunyuan3D.
    
    参考SD3的实现模式：
    - deterministic=True: 使用简单稳定的ODE积分
    - deterministic=False: 使用SDE积分（带随机噪声）
    
    Args:
        scheduler: The FlowMatchEulerDiscreteScheduler instance
        model_output: The direct output from learned flow model (velocity)
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        prev_sample: The previous sample for KL computation (if provided)
        generator: A random number generator
        deterministic: Whether to use deterministic (ODE) or stochastic (SDE) mode

    Returns:
        Tuple containing:
        - prev_sample: The predicted sample for the previous timestep
        - log_prob: Log probability of the step
        - prev_sample_mean: Mean of the predicted sample distribution
        - std_dev: Standard deviation of the noise
    """
    # Error checking: cannot pass both generator and prev_sample
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )
    
    # Initialize step_index if needed
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)
    
    # Get current step index
    step_index = scheduler.step_index
    
    # Ensure we don't go out of bounds
    if step_index >= len(scheduler.sigmas) - 1:
        step_index = len(scheduler.sigmas) - 2
    
    # Get sigmas - Hunyuan3D uses reversed timesteps (decreasing sigmas)
    device = sample.device
    dtype = sample.dtype
    
    sigma = scheduler.sigmas[step_index].to(device=device, dtype=dtype)
    sigma_next = scheduler.sigmas[step_index + 1].to(device=device, dtype=dtype)
    
    # 🔧 关键修复：如果sigma转换后变成0，使用float32精度
    if sigma == 0.0 or sigma_next == 0.0:
        sigma = scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
        sigma_next = scheduler.sigmas[step_index + 1].to(device=device, dtype=torch.float32)
    
    # 🔧 终极修复：确保sigma永远不为0，避免除零错误
    sigma = torch.clamp(sigma, min=1e-8)
    sigma_next = torch.clamp(sigma_next, min=1e-8)
    
    # Compute dt = sigma_next - sigma (negative since sigma_next < sigma)
    dt = sigma_next - sigma
    
    # ==================== SDE Theory (参考SD3实现) ====================
    
    # 1. Compute theoretical noise scaling (参考SD3的简洁实现)
    # 🔧 关键修复：使用推理时的实际sigma_max，而不是训练时的scheduler.sigma_max
    sigma_max = scheduler.sigmas.max().item()  # 推理时的实际最大值 (1.0)
    
    # 🔧 简化：直接使用SD3的处理方式，但用正确的sigma_max
    condition = sigma >= sigma_max
    denominator_value = torch.where(condition, sigma_max - 1e-8, sigma / sigma_max)
    final_denominator = 1 - denominator_value
    
    # 🔧 确保分母不为零或负数
    final_denominator = torch.clamp(final_denominator, min=1e-8)
    
    std_dev_t = torch.sqrt(sigma / final_denominator) * 0.7
    
    # 2. SDE mean computation (参考SD3)
    # prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt) + model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    sample_coeff = 1 + std_dev_t**2 / (2 * sigma) * dt
    progress_ratio = 1 - sigma / sigma_max
    model_coeff = 1 + std_dev_t**2 * progress_ratio / (2 * sigma) * dt
    
    prev_sample_mean = sample * sample_coeff + model_output * model_coeff
    
    # 🔧 添加：简化的NaN检查
    if torch.isnan(prev_sample_mean).any():
        print(f"    🔧 DEBUG: prev_sample_mean有NaN - sigma={sigma:.6f}, std_dev_t={std_dev_t:.6f}")
    
    # 3. Noise scaling
    noise_std = std_dev_t * torch.sqrt(torch.abs(dt))
    
    # ==================== Sample Generation ====================
    
    # 🔧 修复：统一使用简单的ODE作为均值
    # 理论基础：SDE = ODE + noise，所以均值应该是ODE结果
    sample_float = sample.to(torch.float32)
    model_output_float = model_output.to(torch.float32)
    ode_result = sample_float + dt * model_output_float
    ode_result = ode_result.to(dtype)
    
    # Deterministic mode: use simple ODE
    if deterministic:
        prev_sample = ode_result
        # 🔧 修复：deterministic模式下，prev_sample_mean应该也是ODE结果
        prev_sample_mean = ode_result
    else:
        # Stochastic SDE mode: ODE + noise
        if prev_sample is None:
            # Generate noise
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            # Apply SDE: ODE + noise
            prev_sample = ode_result + noise_std * variance_noise
            # 🔧 修复：SDE的均值应该是ODE结果
            prev_sample_mean = ode_result
        else:
            # Use provided prev_sample (for KL computation)
            # 🔧 修复：这种情况下prev_sample_mean也应该是ODE结果
            prev_sample_mean = ode_result
    
    # ==================== Log Probability Computation ====================
    
    if deterministic:
        # For ODE: log probability is zero (deterministic process)
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=dtype)
        # 🔧 修复：deterministic模式下，noise_std应该是0
        noise_std = torch.zeros_like(std_dev_t)
    else:
        # For SDE: Gaussian log probability
        if prev_sample is not None:
            # Use provided prev_sample for KL computation
            diff = prev_sample.detach() - prev_sample_mean
        else:
            diff = prev_sample.detach() - prev_sample_mean
        
        # Gaussian log probability density
        log_prob = (
            -0.5 * (diff ** 2) / (noise_std ** 2)
            - torch.log(noise_std)
            - 0.5 * math.log(2 * math.pi)
        )
        
        # Mean along all dimensions except batch
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # Update step index for next iteration (only if not out of bounds)
    if scheduler._step_index < len(scheduler.sigmas) - 2:
        scheduler._step_index += 1
    
    return prev_sample, log_prob, prev_sample_mean, noise_std


def hunyuan3d_scheduler_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    return_dict: bool = True,
) -> Union[object, Tuple]:
    """
    Compatibility wrapper function that maintains the original scheduler interface
    while adding complex SDE computation with log probability.
    
    This function implements sophisticated SDE theory comparable to SD3, but adapted
    for Hunyuan3D's reversed timestep flow matching framework.
    
    Args:
        scheduler: The FlowMatchEulerDiscreteScheduler instance
        model_output: The direct output from learned flow model
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        generator: A random number generator for stochastic sampling
        deterministic: Whether to use deterministic (ODE) or stochastic (SDE) sampling
        return_dict: Whether to return a dict or tuple
        
    Returns:
        If return_dict is True, returns a dict-like object with prev_sample and additional fields
        Otherwise returns a tuple (prev_sample, log_prob, prev_sample_mean, std_dev)
    """
    prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        generator=generator,
        deterministic=deterministic,
    )
    
    if not return_dict:
        return prev_sample, log_prob, prev_sample_mean, std_dev
    
    # Return a simple object with the required attributes
    class SchedulerOutput:
        def __init__(self, prev_sample, log_prob, prev_sample_mean, std_dev):
            self.prev_sample = prev_sample
            self.log_prob = log_prob
            self.prev_sample_mean = prev_sample_mean
            self.std_dev = std_dev
    
    return SchedulerOutput(prev_sample, log_prob, prev_sample_mean, std_dev) 