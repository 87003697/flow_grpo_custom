# Adapted from flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
# Modified for Hunyuan3D's FlowMatchEulerDiscreteScheduler with reversed timesteps

import math
from typing import Optional, Tuple, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from generators.hunyuan3d.hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput


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
    Predict the sample from the previous timestep by reversing the SDE with log probability calculation.
    
    This function adapts the Hunyuan3D FlowMatchEulerDiscreteScheduler to support log probability
    computation for GRPO training, while maintaining compatibility with the original deterministic behavior.
    
    Args:
        scheduler: The Hunyuan3D FlowMatchEulerDiscreteScheduler instance
        model_output (`torch.FloatTensor`): The direct output from learned diffusion model
        timestep (`Union[float, torch.FloatTensor]`): The current discrete timestep in the diffusion chain
        sample (`torch.FloatTensor`): A current instance of a sample created by the diffusion process
        prev_sample (`torch.FloatTensor`, optional): Pre-computed previous sample for deterministic mode
        generator (`torch.Generator`, optional): A random number generator for stochastic sampling
        deterministic (`bool`): Whether to use deterministic (ODE) or stochastic (SDE) sampling
        
    Returns:
        Tuple containing:
        - prev_sample: The computed previous sample
        - log_prob: Log probability of the transition (batch_size,)
        - prev_sample_mean: Mean of the transition distribution
        - std_dev: Standard deviation of the transition distribution
    """
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )
    
    # Handle timestep input - ensure it's a tensor
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    
    # Initialize step index if needed
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep[0])
    
    # Upcast to avoid precision issues
    sample = sample.to(torch.float32)
    
    # Get current and next sigmas
    # Note: Hunyuan3D uses reversed timesteps, so we need to handle this carefully
    batch_size = sample.shape[0]
    step_indices = []
    
    for i, t in enumerate(timestep):
        try:
            # Convert timestep to CPU for index lookup since scheduler.timesteps is on CPU
            t_cpu = t.cpu() if isinstance(t, torch.Tensor) else t
            step_idx = scheduler.index_for_timestep(t_cpu)
            step_indices.append(step_idx)
        except:
            # Fallback to current step_index if timestep lookup fails
            step_indices.append(scheduler.step_index)
    
    # Ensure we have valid indices - keep on CPU for indexing
    step_indices = step_indices if isinstance(step_indices, list) else step_indices.tolist()
    next_step_indices = [min(idx + 1, len(scheduler.sigmas) - 1) for idx in step_indices]
    
    # Get sigmas - scheduler.sigmas are on CPU, so we get them and move to device
    sigma = scheduler.sigmas[step_indices].to(sample.device, sample.dtype)
    sigma_next = scheduler.sigmas[next_step_indices].to(sample.device, sample.dtype)
    
    # Reshape sigmas for broadcasting with sample dimensions
    while sigma.dim() < sample.dim():
        sigma = sigma.unsqueeze(-1)
    while sigma_next.dim() < sample.dim():
        sigma_next = sigma_next.unsqueeze(-1)
    
    # Compute dt (time step difference)
    dt = sigma_next - sigma
    
    # For deterministic mode, use the original Hunyuan3D step
    if deterministic:
        prev_sample = sample + dt * model_output
        # Return zero log probability for deterministic case
        log_prob = torch.zeros(batch_size, device=sample.device, dtype=sample.dtype)
        prev_sample_mean = prev_sample
        std_dev = torch.zeros_like(sigma.squeeze())
        return prev_sample, log_prob, prev_sample_mean, std_dev
    
    # For stochastic mode, implement SDE with log probability
    # Adapt SD3's approach to Hunyuan3D's simpler sigma schedule
    
    # Compute noise scale based on the flow matching formulation
    # Using a simplified version adapted for Hunyuan3D's linear schedule
    sigma_max = scheduler.sigmas[0].item()  # Maximum sigma value
    
    # Prevent division by zero and numerical issues
    sigma_clamped = torch.clamp(sigma, min=1e-8, max=1.0 - 1e-8)
    
    # Compute standard deviation for the SDE
    # Adapted from SD3 but simplified for Hunyuan3D's linear schedule
    std_dev_t = torch.sqrt(sigma_clamped / (1 - sigma_clamped)) * 0.7
    
    # Compute the mean of the transition distribution
    # This follows the flow matching SDE formulation
    drift_coeff = 1 + std_dev_t**2 / (2 * sigma_clamped) * dt
    diffusion_coeff = 1 + std_dev_t**2 * (1 - sigma_clamped) / (2 * sigma_clamped)
    
    prev_sample_mean = sample * drift_coeff + model_output * diffusion_coeff * dt
    
    # Compute noise variance - ensure it's positive
    dt_abs = torch.abs(dt)  # Take absolute value to handle negative dt
    noise_variance = std_dev_t**2 * dt_abs
    # Ensure minimum variance to avoid numerical issues
    noise_variance = torch.clamp(noise_variance, min=1e-8)
    noise_std = torch.sqrt(noise_variance)
    
    # Sample noise if not provided
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + noise_std * variance_noise
    
    # Compute log probability of the transition
    # This follows the Gaussian log probability formula: log(p(x)) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2π)
    noise_diff = prev_sample.detach() - prev_sample_mean
    
    # Compute log probability components
    quadratic_term = -(noise_diff**2) / (2 * noise_variance)
    log_norm_term = -torch.log(noise_std)
    log_2pi_term = -0.5 * torch.log(2 * torch.as_tensor(math.pi, device=sample.device))
    
    # Combine all terms
    log_prob = quadratic_term + log_norm_term + log_2pi_term
    
    # Sum over all dimensions except batch dimension to get total log probability
    log_prob = log_prob.sum(dim=tuple(range(1, log_prob.ndim)))
    
    # Cast back to original dtype
    prev_sample = prev_sample.to(model_output.dtype)
    
    return prev_sample, log_prob, prev_sample_mean, noise_std.squeeze()


def hunyuan3d_scheduler_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    return_dict: bool = True,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Wrapper function that maintains compatibility with the original scheduler interface
    while adding log probability computation.
    
    Args:
        scheduler: The Hunyuan3D FlowMatchEulerDiscreteScheduler instance
        model_output: The direct output from learned diffusion model
        timestep: The current discrete timestep in the diffusion chain
        sample: A current instance of a sample created by the diffusion process
        generator: A random number generator for stochastic sampling
        deterministic: Whether to use deterministic (ODE) or stochastic (SDE) sampling
        return_dict: Whether to return a dict or tuple
        
    Returns:
        If return_dict is True, returns FlowMatchEulerDiscreteSchedulerOutput with additional fields
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
    
    # Return in the original scheduler output format
    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample) 