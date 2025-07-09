"""
Hunyuan3D GRPO Trainer

3D adaptation of GRPO training for Hunyuan3D model.
Handles 3D mesh generation, reward computation, and GRPO training steps.
"""
import time
import subprocess
from contextlib import contextmanager
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from concurrent import futures

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
from .diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob
from .diffusers_patch.hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob

@contextmanager
def gpu_timer(name):
    """综合监控：耗时 + GPU显存 + GPU利用率"""
    
    # 开始前状态
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    start_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    
    print(f"🕐 开始: {name}")
    print(f"  📊 初始显存: {start_memory:.2f}GB (已分配) / {start_reserved:.2f}GB (已保留)")
    
    # 获取GPU利用率
    def get_gpu_utilization():
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    
    start_util = get_gpu_utilization()
    print(f"  ⚡ 初始GPU利用率: {start_util}%")
    
    try:
        yield
    finally:
        # 结束后状态
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**3
        end_reserved = torch.cuda.memory_reserved() / 1024**3
        end_util = get_gpu_utilization()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        reserved_delta = end_reserved - start_reserved
        
        print(f"✅ 完成: {name}")
        print(f"  ⏱️  耗时: {duration:.2f}秒")
        print(f"  📊 结束显存: {end_memory:.2f}GB (已分配) / {end_reserved:.2f}GB (已保留)")
        print(f"  📈 显存变化: {memory_delta:+.2f}GB (已分配) / {reserved_delta:+.2f}GB (已保留)")
        print(f"  ⚡ 结束GPU利用率: {end_util}%")
        print(f"  🔥 平均GPU利用率: {(start_util + end_util) / 2:.1f}%")
        print()


class Hunyuan3DGRPOTrainer:
    """
    GRPO Trainer adapted for Hunyuan3D 3D generation.
    """
    
    def __init__(
        self,
        pipeline,  # 现在直接是 Hunyuan3DDiTFlowMatchingPipeline
        reward_config: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the 3D GRPO trainer.
        
        Args:
            pipeline: Hunyuan3DDiTFlowMatchingPipeline for 3D generation
            reward_config: 3D reward configuration dict, e.g., {"geometric_quality": 0.3, "uni3d": 0.7}
            device: Device to run training on
        """
        self.pipeline = pipeline
        self.device = device
        
        # Set default reward config if not provided
        if reward_config is None:
            reward_config = {
                "geometric_quality": 0.3,
                "uni3d": 0.7
            }
        
        # Create reward function using new rewards_mesh.py
        self.reward_fn = multi_mesh_score(device, reward_config)
        
        # Move pipeline to device (no need to reassign, just move components)
        self.pipeline.to(device)
    
    def sample_meshes_with_rewards(
        self,
        images: List[str],
        prompts: List[str],
        batch_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        deterministic: bool = False,
        kl_reward: float = 0.0,
        executor: Optional[futures.ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """Sample 3D meshes and compute rewards."""
        
        with gpu_timer("🎯 3D网格生成"):
            # Get the actual pipeline to use
            actual_pipeline = self.pipeline  # 现在直接使用 pipeline
            
            # Process in batches
            all_meshes = []
            all_latents = []
            all_log_probs = []
            all_kl = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]
                
                # Generate meshes with log probabilities
                meshes, latents, log_probs, kl = hunyuan3d_pipeline_with_logprob(
                    actual_pipeline,
                    image=batch_images[0] if len(batch_images) == 1 else batch_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    deterministic=deterministic,
                    kl_reward=kl_reward,
                    mc_level=-1/512,  # 🔧 显式传递mc_level参数
                )
                
                all_meshes.extend(meshes if isinstance(meshes, list) else [meshes])
                all_latents.extend(latents)
                all_log_probs.extend(log_probs)
                all_kl.extend(kl)
        
        with gpu_timer("🏆 奖励函数计算"):
            # Compute rewards asynchronously if executor provided
            if executor:
                reward_future = executor.submit(self.reward_fn, all_meshes, images, prompts)
                rewards = reward_future
            else:
                rewards = self.reward_fn(all_meshes, images, prompts)
        
        with gpu_timer("📦 结果打包"):
            # Convert to tensors
            latents_tensor = torch.stack(all_latents) if all_latents else torch.empty(0)
            log_probs_tensor = torch.stack(all_log_probs) if all_log_probs else torch.empty(0)
            kl_tensor = torch.stack(all_kl) if all_kl else torch.empty(0)
            
            return {
                "meshes": all_meshes,
                "images": images,
                "prompts": prompts,
                "latents": latents_tensor,
                "log_probs": log_probs_tensor,
                "kl": kl_tensor,
                "rewards": rewards,
                "timesteps": torch.randint(0, 1000, (len(images),)),  # Placeholder
            }
    
    def _compute_rewards_sync(
        self, 
        meshes: List, 
        images: List[str], 
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards synchronously."""
        # Use the new reward function from rewards_mesh.py
        reward_details, _ = self.reward_fn(meshes, prompts, {})
        
        # Convert to tensors and move to device
        rewards = {}
        for key, scores in reward_details.items():
            if isinstance(scores, (list, tuple)):
                rewards[key] = torch.tensor(scores, device=self.device, dtype=torch.float32)
            else:
                rewards[key] = torch.tensor([scores], device=self.device, dtype=torch.float32)
        
        return rewards
    
    def _compute_rewards_async(
        self, 
        meshes: List, 
        images: List[str], 
        prompts: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compute rewards asynchronously (for compatibility with original trainer)."""
        rewards = self._compute_rewards_sync(meshes, images, prompts)
        metadata = {
            "num_meshes": len(meshes),
            "avg_score": rewards["avg"].mean().item() if "avg" in rewards else 0.0,
        }
        return rewards, metadata
    
    def compute_log_prob_3d(
        self,
        pipeline,
        sample: Dict[str, torch.Tensor],
        step_index: int,
        config: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probability for a specific timestep in 3D generation.
        
        Args:
            pipeline: Hunyuan3D pipeline
            sample: Sample data containing latents, timesteps, etc.
            step_index: Index of the timestep to compute
            config: Training configuration
            
        Returns:
            Tuple of (prev_sample, log_prob, prev_sample_mean, std_dev)
        """
        # Get the latents and timestep for this step
        latents = sample["latents"][:, step_index]  # Current latents
        next_latents = sample["next_latents"][:, step_index]  # Target next latents
        timestep = sample["timesteps"][:, step_index]  # Current timestep
        
        # Get image conditions (stored in sample or need to recompute)
        if "image_cond" in sample:
            cond = sample["image_cond"]
        else:
            # Recompute conditions from images
            # This would need the original images, which should be stored in sample
            raise NotImplementedError("Image condition recomputation not implemented")
        
        # Prepare model input
        latent_model_input = latents
        if hasattr(config.train, 'cfg') and config.train.cfg:
            # Add negative conditioning for classifier-free guidance
            latent_model_input = torch.cat([latent_model_input, latent_model_input])
            cond = torch.cat([sample.get("neg_cond", cond), cond])
        
        # Convert timestep to normalized format
        timestep_normalized = timestep.float() / pipeline.scheduler.config.num_train_timesteps
        timestep_tensor = timestep_normalized.to(latents.dtype)
        
        # Predict noise using the model
        with torch.cuda.amp.autocast():
            noise_pred = pipeline.model(latent_model_input, timestep_tensor, cond)
        
        # Apply classifier-free guidance if enabled
        if hasattr(config.train, 'cfg') and config.train.cfg:
            noise_pred_neg, noise_pred_pos = noise_pred.chunk(2)
            guidance_scale = getattr(config.sample, 'guidance_scale', 5.0)
            noise_pred = noise_pred_neg + guidance_scale * (noise_pred_pos - noise_pred_neg)
        
        # 🔧 使用config中的deterministic设置
        deterministic = getattr(config, 'deterministic', False)
        
        # Compute log probability using our SDE implementation
        prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
            scheduler=pipeline.scheduler,
            model_output=noise_pred,
            timestep=timestep[0],  # Assume batch has same timestep
            sample=latents,
            prev_sample=next_latents,  # Use target as reference
            deterministic=deterministic,
        )
        
        return prev_sample, log_prob, prev_sample_mean, std_dev
    
    def train_step(
        self,
        samples: Dict[str, torch.Tensor],
        pipeline,
        optimizer: torch.optim.Optimizer,
        config: Any,
        accelerator: Any,
    ) -> Dict[str, float]:
        """
        Perform a single GRPO training step for 3D generation.
        
        Args:
            samples: Batch of samples with latents, rewards, etc.
            pipeline: Hunyuan3D pipeline
            optimizer: Optimizer for training
            config: Training configuration
            accelerator: HuggingFace accelerator
            
        Returns:
            Dictionary of training metrics
        """
        info = defaultdict(list)
        num_timesteps = samples["timesteps"].shape[1]
        
        # Train on each timestep
        for j in range(num_timesteps):
            with accelerator.accumulate(pipeline.model):
                with accelerator.autocast():
                    # Compute log probability for current timestep
                    prev_sample, log_prob, prev_sample_mean, std_dev = self.compute_log_prob_3d(
                        pipeline, samples, j, config
                    )
                    
                    # Compute reference log probability if beta > 0 (KL regularization)
                    if getattr(config.train, 'beta', 0) > 0:
                        with torch.no_grad():
                            # Disable adapter for reference computation
                            with pipeline.model.disable_adapter():
                                _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = self.compute_log_prob_3d(
                                    pipeline, samples, j, config
                                )
                
                # GRPO loss computation
                advantages = torch.clamp(
                    samples["advantages"][:, j],
                    -getattr(config.train, 'adv_clip_max', 5.0),
                    getattr(config.train, 'adv_clip_max', 5.0),
                )
                
                # Compute policy ratio
                ratio = torch.exp(log_prob - samples["log_probs"][:, j])
                
                # PPO clipped loss
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - getattr(config.train, 'clip_range', 0.2),
                    1.0 + getattr(config.train, 'clip_range', 0.2),
                )
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                
                # KL divergence loss
                if getattr(config.train, 'beta', 0) > 0:
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim))) / (2 * std_dev ** 2)
                    kl_loss = torch.mean(kl_loss)
                    loss = policy_loss + config.train.beta * kl_loss
                else:
                    kl_loss = torch.tensor(0.0)
                    loss = policy_loss
                
                # Collect metrics
                info["approx_kl"].append(
                    0.5 * torch.mean((log_prob - samples["log_probs"][:, j]) ** 2)
                )
                info["clipfrac"].append(
                    torch.mean(
                        (torch.abs(ratio - 1.0) > getattr(config.train, 'clip_range', 0.2)).float()
                    )
                )
                info["policy_loss"].append(policy_loss)
                if getattr(config.train, 'beta', 0) > 0:
                    info["kl_loss"].append(kl_loss)
                info["loss"].append(loss)
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        pipeline.parameters(), 
                        getattr(config.train, 'max_grad_norm', 1.0)
                    )
                optimizer.step()
                optimizer.zero_grad()
        
        # Convert metrics to scalars
        return {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}


def create_3d_reward_function(
    reward_config: Optional[Dict[str, float]] = None,
    device: str = "cuda"
):
    """
    Create a reward function for 3D mesh evaluation.
    
    Args:
        reward_config: 3D reward configuration dict, e.g., {"geometric_quality": 0.3, "uni3d": 0.7}
        device: Device for computation
        
    Returns:
        Reward function compatible with GRPO training
    """
    if reward_config is None:
        reward_config = {
            "geometric_quality": 0.3,
            "uni3d": 0.7
        }
    
    # Create reward function using new rewards_mesh.py
    reward_fn = multi_mesh_score(device, reward_config)
    
    def reward_fn_wrapper(meshes, images, prompts, only_strict=True):
        """
        Compute rewards for generated meshes.
        
        Args:
            meshes: List of generated mesh objects
            images: List of input image paths
            prompts: List of text prompts
            only_strict: Whether to use strict evaluation mode
            
        Returns:
            Tuple of (rewards_dict, metadata_dict)
        """
        # Use the new reward function
        reward_details, _ = reward_fn(meshes, prompts, {})
        
        # Convert to numpy arrays for compatibility
        rewards = {}
        for key, scores in reward_details.items():
            if isinstance(scores, (list, tuple)):
                rewards[key] = np.array(scores, dtype=np.float32)
            else:
                rewards[key] = np.array([scores], dtype=np.float32)
        
        # Create metadata
        metadata = {
            "num_meshes": len(meshes),
        }
        
        # Add mean and std for each score type
        for key, scores in rewards.items():
            if len(scores) > 0:
                metadata[f"{key}_mean"] = scores.mean()
                metadata[f"{key}_std"] = scores.std()
        
        return rewards, metadata
    
    return reward_fn_wrapper
