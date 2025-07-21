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
from generators.hunyuan3d.hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

def hunyuan3d_sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Predict the sample from the previous timestep using SDE theory adapted for Hunyuan3D.
    
    参考SD3的实现模式：
    - deterministic=True: 使用简单稳定的ODE积分
    - deterministic=False: 使用SDE积分（带随机噪声）
    
    Args:
        scheduler: The FlowMatchEulerDiscreteScheduler instance in Hunyuan3D
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
    # this sigmas is reversed as in diffusers, and it ends with XXX, XXX, 1., 1.
    sigmas_inverted = 1 - self.sigmas
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = sigmas_inverted[step_index].view(-1, 1, 1)
    sigma_prev = sigmas_inverted[prev_step_index].view(-1, 1, 1)
    sigma_max = sigmas_inverted[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*0.7
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)-model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    # prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    # No noise is added during evaluation
    if determistic:
        prev_sample = sample - dt * model_output

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)


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