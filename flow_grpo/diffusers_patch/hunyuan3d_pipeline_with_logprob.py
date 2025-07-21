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
import subprocess
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm

from .hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob
from generators.hunyuan3d.hy3dshape.pipelines import retrieve_timesteps


@contextmanager
def gpu_timer(name):
    """简单的GPU计时器"""
    start_time = time.time()
    print(f"🕐 开始: {name}")
    try:
        yield
    finally:
        end_time = time.time()
        print(f"✅ 完成: {name} - 耗时: {end_time - start_time:.2f}秒")


@torch.no_grad()
def hunyuan3d_pipeline_with_logprob(
    self,
    positive_image_cond: Optional[torch.Tensor] = None,  # 🔧 SD3式：直接传入正面条件
    negative_image_cond: Optional[torch.Tensor] = None,  # 🔧 SD3式：直接传入负面条件
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    generator: Optional[torch.Generator] = None,
    output_type: str = "trimesh",
    box_v: float = 1.01,
    octree_resolution: int = 384,
    mc_level: float = 0.0,  # 🔧 正确：与参考代码一致
    mc_algo: str = None,
    num_chunks: int = 8000,
    kl_reward: float = 0.0,
    determistic: bool = False,
):
    """
    Generate 3D mesh using Hunyuan3D pipeline with log probability computation for GRPO training.
    
    This function follows the same pattern as SD3's pipeline_with_logprob:
    - First parameter is self (pipeline object)
    - Uses self._execution_device for device access
    - Proper dtype conversions during SDE steps
    - Returns (output, all_latents, all_log_probs, all_kl) or (output, all_latents, all_log_probs, all_kl, image_cond)
    
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
        tuple: (meshes, all_latents, all_log_probs, all_kl) or (meshes, all_latents, all_log_probs, all_kl, image_cond)
    """
    # Get device from pipeline (following SD3 pattern)
    device = self._execution_device if hasattr(self, '_execution_device') else next(self.model.parameters()).device
    
    # Set guidance scale
    self._guidance_scale = guidance_scale

    # Prepare image condition
    assert positive_image_cond is not None, "positive_image_cond is required"
    if guidance_scale > 1.0:
        assert negative_image_cond is not None, "negative_image_cond is required"
        cond_for_generation = {}
        for key in positive_image_cond.keys():
            pos_cond_tensor = positive_image_cond[key]
            neg_cond_tensor = negative_image_cond[key]
            cond_for_generation[key] = torch.cat([pos_cond_tensor, neg_cond_tensor], dim=0)
        do_classifier_free_guidance = True
    else:
        cond_for_generation = positive_image_cond
        do_classifier_free_guidance = False

    # Prepare latent latents
    batch_size = positive_image_cond['main'].shape[0]
    latents = self.prepare_latents(batch_size, self.dtype, device, generator)


    # Prepare timesteps
    sigmas = np.linspace(0, 1, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )
    timesteps = timesteps[:-1]
    self._num_timesteps = len(timesteps)

    # Storage for returns
    all_latents = [latents.clone()]
    all_log_probs = []
    all_kl = []
    
    # 🔧 选择扩散方法
    if True:
        # 使用参考代码的标准方法
        for i, t in enumerate(tqdm(timesteps, desc="Hunyuan3D Denoising with Log Probability")):
            latents_ori = latents.clone()
            
            # Expand the latents if we are doing classifier-free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            # NOTE: we assume model get timesteps ranged from 0 to 1
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.config.num_train_timesteps
            
            # 模型预测
            noise_pred = self.model(latent_model_input, timestep, cond_for_generation, guidance=None)
            
            # Apply classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents, log_prob, prev_latents_mean, std_dev_t = hunyuan3d_sde_step_with_logprob(
                self.scheduler, 
                model_output=noise_pred.float(), 
                timestep=t.unsqueeze(0), 
                sample=latents.float(),
                generator=generator,
                determistic=determistic,
            )
            prev_latents = latents.clone()
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)

    print(f"✅ 扩散采样完成")

    # Handle different output types
    if output_type == "latent":
        meshes = latents
    else:
        # Convert latents to mesh using VAE
        vae_was_on_cpu = next(self.vae.parameters()).device.type == 'cpu'
        if vae_was_on_cpu:
            print("🔧 临时将VAE移动到GPU进行Volume Decoding...")
            self.vae.to(self.device)
        
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        latents = 1. / self.vae.scale_factor * latents
        
        # 🔧 关键修复：添加VAE解码步骤
        latents = self.vae(latents)

        # 🔧 生成网格
        with gpu_timer("Volume Decoding"):
            mesh_output = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=True,
            )
        
        # 🚀 内存优化：VAE使用完毕，移回CPU释放显存
        if vae_was_on_cpu:
            print("🔧 VAE使用完毕，移回CPU释放显存...")
            self.vae.to('cpu')
            # 清理GPU缓存
            torch.cuda.empty_cache()
        
        # 🔧 关键修复：统一转换为 kiui.Mesh 格式
        from generators.hunyuan3d.hy3dshape.pipelines import export_to_kiui
        meshes = export_to_kiui(mesh_output)

    # 🔍 打印调试信息
    print(f"🔍 Pipeline Debug:")
    print(f"  len(all_latents): {len(all_latents)}")
    print(f"  len(all_log_probs): {len(all_log_probs)}")
    print(f"  len(all_kl): {len(all_kl)}")
    if all_latents:
        print(f"  latents[0].shape: {all_latents[0].shape}")
    if all_log_probs:
        print(f"  log_probs[0].shape: {all_log_probs[0].shape}")

    # Return in the same format as SD3
    return meshes, all_latents, all_log_probs, all_kl
