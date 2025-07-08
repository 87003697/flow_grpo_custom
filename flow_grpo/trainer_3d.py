"""
Hunyuan3D GRPO Trainer

3D adaptation of GRPO training for Hunyuan3D model.
Handles 3D mesh generation, reward computation, and GRPO training steps.
"""
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent import futures

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.mesh_basic_scorer import MeshBasicScorer
from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
from .diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob
from .diffusers_patch.hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob


class Hunyuan3DGRPOTrainer:
    """
    GRPO Trainer adapted for Hunyuan3D 3D generation.
    """
    
    def __init__(
        self,
        pipeline: Hunyuan3DPipeline,
        basic_scorer: Optional[MeshBasicScorer] = None,
        uni3d_scorer: Optional[Uni3DScorer] = None,
        device: str = "cuda",
    ):
        """
        Initialize the 3D GRPO trainer.
        
        Args:
            pipeline: Hunyuan3D pipeline for 3D generation
            basic_scorer: Basic geometric mesh quality scorer  
            uni3d_scorer: Uni3D semantic alignment scorer
            device: Device to run training on
        """
        self.pipeline = pipeline
        self.basic_scorer = basic_scorer or MeshBasicScorer()
        self.uni3d_scorer = uni3d_scorer or Uni3DScorer()
        self.device = device
        
        # Move components to device
        if hasattr(self.pipeline, 'pipeline') and hasattr(self.pipeline.pipeline, 'to'):
            self.pipeline.pipeline = self.pipeline.pipeline.to(device)
        elif hasattr(self.pipeline, 'to'):
            self.pipeline = self.pipeline.to(device)
    
    def sample_meshes_with_rewards(
        self,
        images: List[str],
        prompts: List[str], 
        batch_size: int = 4,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        kl_reward: float = 0.0,
        executor: Optional[futures.ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """
        Generate 3D meshes and compute rewards.
        
        Args:
            images: List of image paths for 3D generation
            prompts: List of text prompts corresponding to images
            batch_size: Batch size for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            generator: Random generator for reproducibility
            deterministic: Whether to use deterministic mode
            kl_reward: KL reward coefficient
            executor: Thread executor for async reward computation
            
        Returns:
            Dictionary containing meshes, latents, log_probs, rewards, etc.
        """
        all_meshes = []
        all_latents = []
        all_log_probs = []
        all_kl = []
        all_images = []
        all_prompts = []
        
        # Get the actual pipeline to use
        actual_pipeline = self.pipeline.pipeline if hasattr(self.pipeline, 'pipeline') else self.pipeline
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            batch_meshes = []
            batch_latents = []
            batch_log_probs = []
            batch_kl = []
            
            # Generate meshes for each image in batch
            for image_path, prompt in zip(batch_images, batch_prompts):
                # Generate single mesh with log probabilities
                meshes, latents, log_probs, kl = hunyuan3d_pipeline_with_logprob(
                    actual_pipeline,  # Pass pipeline as first parameter (self)
                    image=image_path,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="trimesh",
                    deterministic=deterministic,
                    kl_reward=kl_reward,
                )
                
                batch_meshes.extend(meshes if isinstance(meshes, list) else [meshes])
                batch_latents.append(torch.stack(latents, dim=1))  # (1, num_steps+1, ...)
                batch_log_probs.append(torch.stack(log_probs, dim=1))  # (1, num_steps)
                batch_kl.append(torch.stack(kl, dim=1))  # (1, num_steps)
            
            all_meshes.extend(batch_meshes)
            all_latents.extend(batch_latents)
            all_log_probs.extend(batch_log_probs)
            all_kl.extend(batch_kl)
            all_images.extend(batch_images)
            all_prompts.extend(batch_prompts)
        
        # Concatenate tensors
        latents = torch.cat(all_latents, dim=0)  # (batch_size, num_steps+1, ...)
        log_probs = torch.cat(all_log_probs, dim=0)  # (batch_size, num_steps)
        kl = torch.cat(all_kl, dim=0)  # (batch_size, num_steps)
        
        # Generate timesteps
        actual_pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = actual_pipeline.scheduler.timesteps.repeat(
            len(all_meshes), 1
        )  # (batch_size, num_steps)
        
        # Compute rewards asynchronously if executor provided
        if executor:
            reward_future = executor.submit(
                self._compute_rewards_async, 
                all_meshes, 
                all_images,
                all_prompts
            )
        else:
            rewards = self._compute_rewards_sync(all_meshes, all_images, all_prompts)
            reward_future = None
        
        return {
            "meshes": all_meshes,
            "images": all_images,
            "prompts": all_prompts,
            "timesteps": timesteps,
            "latents": latents[:, :-1],  # Remove last latent (final result)
            "next_latents": latents[:, 1:],  # Remove first latent (initial noise)
            "log_probs": log_probs,
            "kl": kl,
            "rewards": reward_future if reward_future else rewards,
        }
    
    def _compute_rewards_sync(
        self, 
        meshes: List, 
        images: List[str], 
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards synchronously."""
        geometric_scores = []
        semantic_scores = []
        
        for mesh, image_path, prompt in zip(meshes, images, prompts):
            # Compute geometric quality score
            geo_score = self.basic_scorer.score_mesh(mesh)
            geometric_scores.append(geo_score)
            
            # Compute semantic alignment score
            semantic_score = self.uni3d_scorer.score_mesh_with_image(mesh, image_path)
            semantic_scores.append(semantic_score)
        
        # Convert to tensors
        geometric_scores = torch.tensor(geometric_scores, device=self.device, dtype=torch.float32)
        semantic_scores = torch.tensor(semantic_scores, device=self.device, dtype=torch.float32)
        
        # Combine scores (weighted average)
        avg_scores = 0.3 * geometric_scores + 0.7 * semantic_scores
        
        return {
            "geometric": geometric_scores,
            "semantic": semantic_scores,
            "avg": avg_scores,
        }
    
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
            "avg_geometric": rewards["geometric"].mean().item(),
            "avg_semantic": rewards["semantic"].mean().item(),
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
        
        # Compute log probability using our SDE implementation
        prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
            scheduler=pipeline.scheduler,
            model_output=noise_pred,
            timestep=timestep[0],  # Assume batch has same timestep
            sample=latents,
            prev_sample=next_latents,  # Use target as reference
            deterministic=False,
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
    basic_scorer: Optional[MeshBasicScorer] = None,
    uni3d_scorer: Optional[Uni3DScorer] = None,
    device: str = "cuda"
):
    """
    Create a reward function for 3D mesh evaluation.
    
    Args:
        basic_scorer: Basic geometric mesh scorer
        uni3d_scorer: Uni3D semantic scorer  
        device: Device for computation
        
    Returns:
        Reward function compatible with GRPO training
    """
    if basic_scorer is None:
        basic_scorer = MeshBasicScorer()
    if uni3d_scorer is None:
        uni3d_scorer = Uni3DScorer()
    
    def reward_fn(meshes, images, prompts, only_strict=True):
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
        geometric_scores = []
        semantic_scores = []
        
        for mesh, image_path, prompt in zip(meshes, images, prompts):
            # Geometric quality
            geo_score = basic_scorer.score(mesh)
            geometric_scores.append(geo_score)
            
            # Semantic alignment  
            semantic_score = uni3d_scorer.score_mesh_with_image(mesh, image_path)
            semantic_scores.append(semantic_score)
        
        # Convert to numpy arrays
        geometric_scores = np.array(geometric_scores, dtype=np.float32)
        semantic_scores = np.array(semantic_scores, dtype=np.float32)
        
        # Combined score
        avg_scores = 0.3 * geometric_scores + 0.7 * semantic_scores
        
        rewards = {
            "geometric": geometric_scores,
            "semantic": semantic_scores,
            "avg": avg_scores,
        }
        
        metadata = {
            "num_meshes": len(meshes),
            "geometric_mean": geometric_scores.mean(),
            "semantic_mean": semantic_scores.mean(),
            "avg_mean": avg_scores.mean(),
            "geometric_std": geometric_scores.std(),
            "semantic_std": semantic_scores.std(),
            "avg_std": avg_scores.std(),
        }
        
        return rewards, metadata
    
    return reward_fn
