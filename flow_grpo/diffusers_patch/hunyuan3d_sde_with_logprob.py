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
    Predict the sample from the previous timestep using complex SDE theory adapted for Hunyuan3D.
    
    This implementation follows SD3's SDE framework but adapts it for Hunyuan3D's:
    - Reversed timesteps (sigma: 1.0→0.0)
    - Negative dt values  
    - Flow matching framework
    
    Mathematical Details:
    - std_dev_t: theoretical noise scaling based on sigma evolution
    - SDE mean: complex drift correction accounting for reversed flow
    - Deterministic mode: degenerates to simple ODE (sample + dt * model_output)
    - Log probability: proper Gaussian density accounting for SDE dynamics
    
    Args:
        scheduler: The FlowMatchEulerDiscreteScheduler instance
        model_output: The direct output from learned flow model (velocity)
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        prev_sample: The previous sample for KL computation (if provided)
        generator: A random number generator
        deterministic: Whether to use deterministic (ODE) mode

    Returns:
        Tuple containing:
        - prev_sample: The predicted sample for the previous timestep
        - log_prob: Log probability of the step
        - prev_sample_mean: Mean of the SDE predicted sample distribution
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
    
    # For numerical stability, add small epsilon to avoid division by zero
    sigma_eps = torch.clamp(sigma, min=1e-8)
    sigma_max = scheduler.sigmas[0].item()  # Max sigma (beginning of reversed process)
    
    # Compute dt = sigma_next - sigma (negative since sigma_next < sigma)
    dt = sigma_next - sigma
    
    # ==================== Complex SDE Theory ====================
    
    # 1. Compute theoretical noise scaling (adapted from SD3 for reversed timesteps)
    # For reversed flow: std_dev_t represents the noise level at current sigma
    # Formula adapted for decreasing sigmas: sqrt(sigma / (1 - sigma/sigma_max)) * scaling
    std_dev_t = torch.sqrt(sigma_eps / (1 - torch.where(sigma >= sigma_max, 
                                                        sigma_max - 1e-8, 
                                                        sigma / sigma_max))) * 0.7
    
    # 2. Complex SDE drift correction (adapted for reversed timesteps)
    # Original SD3: sample*(1+std_dev_t**2/(2*sigma)*dt) + model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    # Adapted for Hunyuan3D reversed flow:
    # - Account for negative dt
    # - Adjust for decreasing sigma schedule
    
    # Drift coefficient for sample term
    sample_coeff = 1 + std_dev_t**2 / (2 * sigma_eps) * dt
    
    # Drift coefficient for model output term  
    # For reversed flow: (1 - sigma/sigma_max) represents progress toward clean data
    progress_ratio = 1 - sigma / sigma_max
    model_coeff = 1 + std_dev_t**2 * progress_ratio / (2 * sigma_eps) * dt
    
    # SDE mean with complex drift corrections
    prev_sample_mean = sample * sample_coeff + model_output * model_coeff
    
    # 3. Noise term scaling
    # For reversed flow with negative dt: sqrt(|dt|)
    noise_std = std_dev_t * torch.sqrt(torch.abs(dt))
    
    # ==================== Sample Generation ====================
    
    # Deterministic mode: degenerate to simple ODE (like SD3 does)
    if deterministic:
        # Simple Euler integration (matches original Hunyuan3D ODE)
        prev_sample = sample + dt * model_output
    else:
        # Stochastic SDE mode
        if prev_sample is None:
            # Generate noise
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            # Apply SDE: mean + noise
            prev_sample = prev_sample_mean + noise_std * variance_noise
        else:
            # Use provided prev_sample (for KL computation)
            pass
    
    # ==================== Log Probability Computation ====================
    
    if deterministic:
        # For ODE: log probability is determined by Jacobian (approximately zero for simple Euler)
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=dtype)
    else:
        # For SDE: Gaussian log probability
        if prev_sample is not None:
            # Use provided prev_sample for KL computation
            diff = prev_sample.detach() - prev_sample_mean
        else:
            diff = prev_sample.detach() - prev_sample_mean
        
        # Gaussian log probability density
        # log p(x) = -0.5 * (x - μ)² / σ² - log(σ) - 0.5 * log(2π)
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