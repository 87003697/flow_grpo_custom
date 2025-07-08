"""
Hunyuan3D Pipeline with Log Probability for GRPO Training

Adapted from flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py
Modified for Hunyuan3D's 3D generation pipeline and FlowMatchEulerDiscreteScheduler

Key modifications:
- Uses the same function signature pattern as SD3 (self as first parameter)
- Proper device and dtype handling using self._execution_device
- Correct type conversions during SDE steps
- Compatible with GRPO training framework
"""
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from PIL import Image

from .hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob


@torch.no_grad()
def hunyuan3d_pipeline_with_logprob(
    self,
    image: Union[str, Image.Image, torch.Tensor],
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    generator: Optional[torch.Generator] = None,
    output_type: str = "trimesh",
    box_v: float = 1.01,
    octree_resolution: int = 384,
    mc_level: float = 0.0,
    mc_algo: str = None,
    num_chunks: int = 8000,
    deterministic: bool = False,
    kl_reward: float = 0.0,
):
    """
    Generate 3D mesh using Hunyuan3D pipeline with log probability computation for GRPO training.
    
    This function follows the same pattern as SD3's pipeline_with_logprob:
    - First parameter is self (pipeline object)
    - Uses self._execution_device for device access
    - Proper dtype conversions during SDE steps
    - Returns (output, all_latents, all_log_probs, all_kl)
    
    Args:
        self: Hunyuan3DDiTFlowMatchingPipeline instance
        image: Input image (path, PIL Image, or tensor)
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        generator: Random generator for reproducibility
        output_type: Output format ("trimesh" or "latent")
        box_v: Bounding box volume for mesh extraction
        octree_resolution: Resolution for octree-based mesh extraction
        mc_level: Marching cubes level
        mc_algo: Marching cubes algorithm
        num_chunks: Number of chunks for mesh processing
        deterministic: Whether to use deterministic (ODE) mode
        kl_reward: KL reward coefficient
        
    Returns:
        tuple: (meshes, all_latents, all_log_probs, all_kl)
    """
    # Get device from pipeline (following SD3 pattern)
    device = self._execution_device if hasattr(self, '_execution_device') else next(self.model.parameters()).device
    
    # Set guidance scale
    self._guidance_scale = guidance_scale
    
    # Prepare image condition
    if isinstance(image, str):
        image_pil = Image.open(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGBA")
    else:
        image_pil = image
    
    # Encode image condition using pipeline's method
    try:
        # Use the pipeline's encode_cond method
        cond = self.encode_cond(
            image=image_pil,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
    except Exception as e:
        print(f"⚠️ Warning: encode_cond failed ({e}), using fallback")
        # Fallback: basic image processing
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        if isinstance(image_pil, Image.Image):
            cond = transform(image_pil).unsqueeze(0).to(device=device)
        else:
            cond = image_pil.to(device=device)
    
    # Ensure condition is on the right device
    if isinstance(cond, torch.Tensor):
        cond = cond.to(device=device)
    elif isinstance(cond, dict):
        cond = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                for k, v in cond.items()}
    
    batch_size = 1  # Single image for now
    
    # Prepare timesteps using scheduler
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    
    # Prepare initial latents
    latents = self.prepare_latents(batch_size, self.model.dtype, device, generator)
    
    # Storage for returns
    all_latents = [latents.clone()]
    all_log_probs = []
    all_kl = []
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Store original latents for KL computation
        latents_ori = latents.clone()
        
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        
        # Broadcast timestep to batch dimension
        timestep = t.expand(latent_model_input.shape[0])
        
        # Predict noise using the model
        noise_pred = self.model(latent_model_input, timestep, cond)
        
        # Apply classifier-free guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Store original dtype
        latents_dtype = latents.dtype
        
        # SDE step with log probability (following SD3 pattern)
        latents, log_prob, prev_latents_mean, std_dev_t = hunyuan3d_sde_step_with_logprob(
            self.scheduler,
            noise_pred.float(),  # Convert to float for computation
            t.unsqueeze(0),
            latents.float(),     # Convert to float for computation
            generator=generator,
            deterministic=deterministic,
        )
        
        # Store previous latents for KL computation
        prev_latents = latents.clone()
        
        # Convert back to original dtype if needed
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        
        # Store results
        all_latents.append(latents.clone())
        all_log_probs.append(log_prob)
        
        # Compute KL divergence if needed (following SD3 pattern)
        if kl_reward > 0 and not deterministic:
            # Expand latents for CFG
            latent_model_input_ref = torch.cat([latents_ori] * 2) if self.do_classifier_free_guidance else latents_ori
            
            # Disable adapter for reference computation (if available)
            with getattr(self.model, 'disable_adapter', lambda: torch.no_grad())():
                noise_pred_ref = self.model(latent_model_input_ref, timestep, cond)
            
            # Apply CFG to reference
            if self.do_classifier_free_guidance:
                noise_pred_ref_uncond, noise_pred_ref_cond = noise_pred_ref.chunk(2)
                noise_pred_ref = noise_pred_ref_uncond + self.guidance_scale * (noise_pred_ref_cond - noise_pred_ref_uncond)
            
            # Compute reference step
            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = hunyuan3d_sde_step_with_logprob(
                self.scheduler,
                noise_pred_ref.float(),
                t.unsqueeze(0),
                latents_ori.float(),
                prev_sample=prev_latents.float(),
                deterministic=deterministic,
            )
            
            # Compute KL divergence
            assert std_dev_t.shape == ref_std_dev_t.shape
            kl = (prev_latents_mean - ref_prev_latents_mean)**2 / (2 * std_dev_t**2)
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
        else:
            # No KL reward computation needed
            all_kl.append(torch.zeros(batch_size, device=device))
    
    # Generate final output
    if output_type == "latent":
        meshes = latents
    else:
        # Convert latents to mesh using VAE
        latents = latents.to(dtype=self.vae.dtype)
        latents = 1. / self.vae.scale_factor * latents
        latents_decoded = self.vae(latents)
        
        # Extract mesh using marching cubes
        meshes = self.vae.latents2mesh(
            latents_decoded,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True,
        )
    
    # Return in the same format as SD3
    return meshes, all_latents, all_log_probs, all_kl
