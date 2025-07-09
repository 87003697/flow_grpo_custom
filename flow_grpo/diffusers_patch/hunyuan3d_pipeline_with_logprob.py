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
import time

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
    mc_level: float = -1/512,
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
    
    # Compute whether to use classifier-free guidance
    do_classifier_free_guidance = guidance_scale >= 0 and not (
        hasattr(self.model, 'guidance_embed') and
        self.model.guidance_embed is True
    )
    
    # Prepare image condition
    if isinstance(image, list):
        # Handle list of images
        image_pil = image
    elif isinstance(image, str):
        image_pil = Image.open(image).convert("RGBA")
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGBA")
    else:
        image_pil = image
    
    # Encode image condition using pipeline's method
    # 🔧 修复：正确使用 prepare_image 方法
    cond_inputs = self.prepare_image(image_pil)
    image_tensor = cond_inputs.pop('image')
    
    # 🔧 关键修复：batch_size 应该基于实际的图像数量
    batch_size = image_tensor.shape[0]
    
    # Use the pipeline's encode_cond method
    cond = self.encode_cond(
        image=image_tensor,
        additional_cond_inputs=cond_inputs,
        do_classifier_free_guidance=do_classifier_free_guidance,
        dual_guidance=False,
    )
    
    # 🔧 修复：不需要手动处理设备，encode_cond 已经处理了
    # Ensure condition is on the right device
    # if isinstance(cond, torch.Tensor):
    #     cond = cond.to(device=device)
    # elif isinstance(cond, dict):
    #     cond = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
    #             for k, v in cond.items()}
    
    # batch_size = 1  # Single image for now # This line is now handled by the new_code
    
    # Prepare timesteps using scheduler - 🔧 修复：使用 FlowMatching 的 timestep 处理方式
    import numpy as np
    from generators.hunyuan3d.hy3dshape.pipelines import retrieve_timesteps
    
    # NOTE: this is slightly different from common usage, we start from 0.
    sigmas = np.linspace(0, 1, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )
    
    # Prepare initial latents
    latents = self.prepare_latents(batch_size, self.dtype, device, generator)
    
    # 🔧 修复：使用 guidance 参数而不是 guidance_cond
    guidance = None
    if hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True:
        guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=self.dtype)
    
    # Storage for returns
    all_latents = [latents.clone()]
    all_log_probs = []
    all_kl = []
    
    # Denoising loop
    print(f"🔄 开始扩散采样 ({num_inference_steps}步)")
    total_step_time = 0
    
    for i, t in enumerate(timesteps):
        step_start = time.time()
        step_memory_start = torch.cuda.memory_allocated() / 1024**3
        
        # Store original latents for KL computation
        latents_ori = latents.clone()
        
        # 🔧 添加调试信息 - 检查输入latents
        print(f"  🔍 步骤 {i+1} 输入检查:")
        print(f"    latents shape: {latents.shape}")
        print(f"    latents min: {latents.min().item():.6f}")
        print(f"    latents max: {latents.max().item():.6f}")
        print(f"    latents mean: {latents.mean().item():.6f}")
        print(f"    latents has nan: {torch.isnan(latents).any().item()}")
        print(f"    latents has inf: {torch.isinf(latents).any().item()}")
        
        # Expand the latents if we are doing classifier-free guidance
        if do_classifier_free_guidance:
            latents_model_input = torch.cat([latents] * 2)
        else:
            latents_model_input = latents
        
        # Call the model - 🔧 修复：使用正确的Hunyuan3D模型API
        # NOTE: we assume model get timesteps ranged from 0 to 1
        timestep = t.expand(latents_model_input.shape[0]).to(latents.dtype)
        timestep = timestep / self.scheduler.config.num_train_timesteps
        noise_pred = self.model(latents_model_input, timestep, cond, guidance=guidance)
        
        # 🔧 添加调试信息 - 检查模型输出
        print(f"    noise_pred shape: {noise_pred.shape}")
        print(f"    noise_pred min: {noise_pred.min().item():.6f}")
        print(f"    noise_pred max: {noise_pred.max().item():.6f}")
        print(f"    noise_pred mean: {noise_pred.mean().item():.6f}")
        print(f"    noise_pred has nan: {torch.isnan(noise_pred).any().item()}")
        print(f"    noise_pred has inf: {torch.isinf(noise_pred).any().item()}")
        
        # Apply classifier-free guidance
        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 🔧 添加调试信息 - 检查CFG后的输出
            print(f"    noise_pred_after_cfg min: {noise_pred.min().item():.6f}")
            print(f"    noise_pred_after_cfg max: {noise_pred.max().item():.6f}")
            print(f"    noise_pred_after_cfg mean: {noise_pred.mean().item():.6f}")
            print(f"    noise_pred_after_cfg has nan: {torch.isnan(noise_pred).any().item()}")
            print(f"    noise_pred_after_cfg has inf: {torch.isinf(noise_pred).any().item()}")
        
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
        
        # 🔧 添加调试信息 - 检查SDE步骤后的输出
        print(f"    latents_after_sde min: {latents.min().item():.6f}")
        print(f"    latents_after_sde max: {latents.max().item():.6f}")
        print(f"    latents_after_sde mean: {latents.mean().item():.6f}")
        print(f"    latents_after_sde has nan: {torch.isnan(latents).any().item()}")
        print(f"    latents_after_sde has inf: {torch.isinf(latents).any().item()}")
        
        # Store previous latents for KL computation
        prev_latents = latents.clone()
        
        # Convert back to original dtype if needed
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        
        # Store results
        all_latents.append(latents.clone())
        all_log_probs.append(log_prob)
        
        # Print step timing
        step_end = time.time()
        step_memory_end = torch.cuda.memory_allocated() / 1024**3
        step_duration = step_end - step_start
        step_memory_delta = step_memory_end - step_memory_start
        total_step_time += step_duration
        
        print(f"  步骤 {i+1:2d}/{num_inference_steps}: "
              f"{step_duration:.2f}s, "
              f"显存: {step_memory_end:.2f}GB ({step_memory_delta:+.2f}GB)")
        
        # Compute KL divergence if needed (following SD3 pattern)
        if kl_reward > 0 and not deterministic:
            # Expand latents for CFG
            latent_model_input_ref = torch.cat([latents_ori] * 2) if do_classifier_free_guidance else latents_ori
            
            # Disable adapter for reference computation (if available)
            with getattr(self.model, 'disable_adapter', lambda: torch.no_grad())():
                # 🔧 修复：使用正确的Hunyuan3D模型API
                timestep_ref = t.expand(latent_model_input_ref.shape[0]).to(latents.dtype)
                timestep_ref = timestep_ref / self.scheduler.config.num_train_timesteps
                noise_pred_ref = self.model(latent_model_input_ref, timestep_ref, cond, guidance=guidance)
            
            # Apply CFG to reference
            if do_classifier_free_guidance:
                noise_pred_ref_uncond, noise_pred_ref_cond = noise_pred_ref.chunk(2)
                noise_pred_ref = noise_pred_ref_uncond + guidance_scale * (noise_pred_ref_cond - noise_pred_ref_uncond)
            
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
    
    print(f"✅ 扩散采样完成，总耗时: {total_step_time:.2f}秒")
    
    # Handle different output types
    if output_type == "latent":
        meshes = latents
    else:
        # Convert latents to mesh using VAE
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        latents = 1. / self.vae.scale_factor * latents
        latents_decoded = self.vae(latents)
        
        # Extract mesh using marching cubes
        mesh_output = self.vae.latents2mesh(
            latents_decoded,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True,
        )
        
        # 🔧 关键修复：统一转换为 kiui.Mesh 格式
        from generators.hunyuan3d.hy3dshape.pipelines import export_to_kiui
        meshes = export_to_kiui(mesh_output)
    
    # Return in the same format as SD3
    return meshes, all_latents, all_log_probs, all_kl
