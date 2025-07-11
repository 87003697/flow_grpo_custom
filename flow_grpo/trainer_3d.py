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
        pipeline: Hunyuan3DPipeline,  # 明确：只接受 Hunyuan3DPipeline
        reward_config: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the 3D GRPO trainer.
        
        Args:
            pipeline: Hunyuan3DPipeline 包装类
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
        
        # Move core pipeline to device (明确的访问路径)
        self.pipeline.core_pipeline.to(device)
    
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
            # 明确：总是使用 core_pipeline
            actual_pipeline = self.pipeline.core_pipeline
            
            # Process in batches
            all_meshes = []
            all_latents = []
            all_log_probs = []
            all_kl = []
            # 🔧 新增：保存图像条件用于训练
            all_image_conds = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]
                
                # Generate meshes with log probabilities
                meshes, latents, log_probs, kl, image_cond = hunyuan3d_pipeline_with_logprob(
                    actual_pipeline,
                    image=batch_images[0] if len(batch_images) == 1 else batch_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    deterministic=True,  # 🔧 明确设置为True，确保与原生方法一致
                    kl_reward=kl_reward,
                    return_image_cond=True,  # 🔧 新增：请求返回图像条件
                )
                
                all_meshes.extend(meshes if isinstance(meshes, list) else [meshes])
                all_latents.extend(latents)
                all_log_probs.extend(log_probs)
                # 🔧 修复：使用append而不是extend，保持KL的二维结构
                all_kl.append(kl)  # 保持(batch_size,)的结构，而不是拍平
                # 🔧 新增：保存图像条件
                all_image_conds.append(image_cond)
        
        with gpu_timer("🏆 奖励函数计算"):
            # Compute rewards asynchronously if executor provided
            if executor:
                reward_future = executor.submit(self.reward_fn, all_meshes, images, prompts)
                rewards = reward_future
            else:
                rewards = self.reward_fn(all_meshes, images, prompts)
        
        with gpu_timer("📦 结果打包"):
            # 🔍 Hunyuan3D Trainer Debug: 处理pipeline返回数据
            # ⚠️ 重要：SD3和Hunyuan3D的latent shape不同，但数据处理模式相同
            # SD3: all_latents是list of tensors，每个shape为(batch_size, 16, 32, 32)
            # Hunyuan3D: all_latents是list of tensors，每个shape为(batch_size, 1024, 64)
            # 相同点：都是lists → stack → 分割为current/next states
            print(f"🔍 Hunyuan3D Trainer Debug - 原始数据:")
            print(f"  len(all_latents): {len(all_latents)} (SD3也是: num_steps+1)")
            print(f"  len(all_log_probs): {len(all_log_probs)} (SD3也是: num_steps)")
            print(f"  len(all_kl): {len(all_kl)} (SD3也是: num_steps)")
            print(f"  len(all_image_conds): {len(all_image_conds)} (新增：图像条件)")
            if all_latents:
                print(f"  all_latents[0].shape: {all_latents[0].shape} (Hunyuan3D: (batch, 1024, 64))")
                print(f"  对比SD3: all_latents[0].shape = (batch, 16, 32, 32)")
            if all_log_probs:
                print(f"  all_log_probs[0].shape: {all_log_probs[0].shape} (与SD3相同: (batch,))")
            if all_kl:
                # 🔧 修复：安全检查 all_kl[0] 的类型
                if isinstance(all_kl[0], torch.Tensor):
                    print(f"  all_kl[0].shape: {all_kl[0].shape} (与SD3相同: (batch,))")
                else:
                    print(f"  all_kl[0] 类型: {type(all_kl[0])}, 长度: {len(all_kl[0]) if hasattr(all_kl[0], '__len__') else 'N/A'}")
            if all_image_conds:
                print(f"  all_image_conds[0] 类型: {type(all_image_conds[0])} (新增：图像条件)")
                if isinstance(all_image_conds[0], dict):
                    print(f"    字典包含keys: {list(all_image_conds[0].keys())}")
                    for key, value in all_image_conds[0].items():
                        if isinstance(value, torch.Tensor):
                            print(f"      {key}.shape: {value.shape}")
                        else:
                            print(f"      {key}: {type(value)}")
                elif isinstance(all_image_conds[0], torch.Tensor):
                    print(f"  all_image_conds[0].shape: {all_image_conds[0].shape}")
                else:
                    print(f"  all_image_conds[0]: {type(all_image_conds[0])}")
            
            # Convert to tensors
            # 🔧 修复：按SD3方式stack - (batch, steps+1, ...)
            latents_tensor = torch.stack(all_latents, dim=1) if all_latents else torch.empty(0)
            print(f"  🔧 修复后 latents_tensor.shape: {latents_tensor.shape if latents_tensor.numel() > 0 else 'empty'}")
            print(f"    期望格式: (batch, steps+1, 1024, 64)")
            # 🔧 修复：按SD3方式stack - (batch, steps)
            log_probs_tensor = torch.stack(all_log_probs, dim=1) if all_log_probs else torch.empty(0)
            print(f"  🔧 修复后 log_probs_tensor.shape: {log_probs_tensor.shape if log_probs_tensor.numel() > 0 else 'empty'}")
            print(f"    期望格式: (batch, steps)")
            
            # 🔧 新增：处理图像条件
            if all_image_conds:
                # 图像条件在所有步骤中都相同，只需要第一个
                image_cond_tensor = all_image_conds[0]  # 使用第一个batch的图像条件
                print(f"  🔧 新增 image_cond_tensor 类型: {type(image_cond_tensor)}")
                if isinstance(image_cond_tensor, dict):
                    print(f"    字典包含keys: {list(image_cond_tensor.keys())}")
                    for key, value in image_cond_tensor.items():
                        if isinstance(value, torch.Tensor):
                            print(f"      {key}.shape: {value.shape}")
                elif isinstance(image_cond_tensor, torch.Tensor):
                    print(f"    tensor.shape: {image_cond_tensor.shape}")
                print(f"    用于训练阶段的条件计算")
            else:
                image_cond_tensor = None
            
            # 🔍 Hunyuan3D Trainer Debug - 转换后的tensor形状:
            # ⚠️ 当前问题：我们的stack方式与SD3不同！
            # SD3方式: torch.stack(data, dim=1) → (batch_size, num_steps+1, ...)
            # 当前方式: torch.stack(data, dim=0) → (num_steps+1, batch_size, ...)
            print(f"🔍 Hunyuan3D Trainer Debug - 转换后:")
            if latents_tensor.numel() > 0:
                print(f"  latents_tensor.shape: {latents_tensor.shape}")
                print(f"  当前: (steps+1, batch, 1024, 64)")
                print(f"  SD3应为: (batch, steps+1, 16, 32, 32)")
            if log_probs_tensor.numel() > 0:
                print(f"  log_probs_tensor.shape: {log_probs_tensor.shape}")
                print(f"  当前: (steps, batch)")
                print(f"  SD3应为: (batch, steps)")
            
            # 🔧 修复：确保all_kl中都是tensor并按SD3方式stack
            if all_kl:
                # 🔍 SD3 KL处理参考: 
                # SD3: all_kl是list of tensors，每个shape为(batch_size,)
                # SD3方式: torch.stack(all_kl, dim=1) → (batch_size, num_steps)
                # Hunyuan3D: 相同的数据结构，但需要正确的stack方式
                print(f"🔍 KL tensor处理 - 对比SD3:")
                print(f"  all_kl长度: {len(all_kl)} (SD3也是: num_steps)")
                
                # 将all_kl中的每个元素转换为tensor（如果还不是的话）
                all_kl_tensors = []
                for i, kl in enumerate(all_kl):
                    if isinstance(kl, torch.Tensor):
                        all_kl_tensors.append(kl)
                        if i == 0:
                            print(f"  all_kl[0].shape: {kl.shape} (SD3 ref: (1,))")
                    elif isinstance(kl, (list, tuple)):
                        # 🔧 修复：对于list/tuple，先转换为tensor再stack
                        if len(kl) > 0 and isinstance(kl[0], torch.Tensor):
                            # 如果是tensor列表，先stack成2D tensor
                            kl_tensor = torch.stack(kl)  # (num_steps, batch_size)
                            kl_tensor = kl_tensor.transpose(0, 1)  # (batch_size, num_steps)
                        else:
                            # 如果是数值列表，直接转换
                            kl_tensor = torch.tensor(kl)
                        all_kl_tensors.append(kl_tensor)
                    else:
                        all_kl_tensors.append(torch.tensor(kl))
                
                # 🔧 修复：现在all_kl_tensors中的每个元素都应该是(batch_size, num_steps)
                # 我们需要在batch维度上拼接
                kl_tensor = torch.cat(all_kl_tensors, dim=0)  # (total_batch_size, num_steps)
                print(f"  最终kl_tensor.shape: {kl_tensor.shape} (SD3应为: (batch_size, num_steps))")
            else:
                kl_tensor = torch.empty(0)
            
            
            # 🔍 最终验证 - 所有tensor形状
            print(f"🔍 最终验证 - 所有tensor形状:")
            
            # 🔧 修复：生成正确形状的timesteps - (batch_size, num_steps)
            num_steps = latents_tensor.shape[1] - 1 if latents_tensor.numel() > 0 else 20  # steps = latents_steps - 1
            timesteps_tensor = torch.randint(0, 1000, (len(images), num_steps), device=self.device)
            print(f"  🔧 修复后 timesteps.shape: {timesteps_tensor.shape}")
            print(f"    期望格式: (batch, steps)")
            print(f"    设备: {timesteps_tensor.device}")
            
            temp_result = {
                "meshes": all_meshes,
                "images": images,
                "prompts": prompts,
                "latents": latents_tensor,
                "log_probs": log_probs_tensor,
                "kl": kl_tensor,
                "rewards": rewards,
                "timesteps": timesteps_tensor,
                "image_cond": image_cond_tensor,  # 🔧 新增：图像条件
            }
            for key, value in temp_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}.shape: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                else:
                    print(f"  {key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
            print(f"  ==========================================")
            
            return {
                "meshes": all_meshes,
                "images": images,
                "prompts": prompts,
                "latents": latents_tensor,
                "log_probs": log_probs_tensor,
                "kl": kl_tensor,
                "rewards": rewards,
                "timesteps": timesteps_tensor,
                "image_cond": image_cond_tensor,  # 🔧 新增：图像条件
            }
    
    def _compute_rewards_sync(
        self, 
        meshes: List, 
        images: List[str], 
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards synchronously and return as tensors on the correct device."""
        # Use the reward function to compute scores
        reward_details, _ = self.reward_fn(meshes, prompts, {})
        
        # 🔧 优化：直接在目标设备上创建tensor，避免设备转换
        rewards = {}
        for key, scores in reward_details.items():
            if isinstance(scores, (list, tuple)):
                # 🔧 优化：直接在目标设备上创建，避免CPU->CUDA转换
                rewards[key] = torch.tensor(scores, device=self.device, dtype=torch.float32)
                print(f"🔧 优化：{key} 奖励直接在 {self.device} 上创建，形状 {rewards[key].shape}")
            else:
                rewards[key] = torch.tensor([scores], device=self.device, dtype=torch.float32)
                print(f"🔧 优化：{key} 奖励(标量)直接在 {self.device} 上创建")
        
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
        
        # 🔧 修复：使用保存的图像条件
        if "image_cond" in sample and sample["image_cond"] is not None:
            cond = sample["image_cond"]
            print(f"🔧 使用保存的图像条件: {cond.shape}")
        else:
            # 🔧 修复：实现图像条件重计算逻辑
            print(f"🔧 重新计算图像条件...")
            # 从原始图像重新计算条件
            if "images" in sample:
                # 使用pipeline的条件编码器重新计算
                images = sample["images"]
                # 假设pipeline有conditioner属性
                if hasattr(pipeline, 'conditioner'):
                    # 重新加载和编码图像
                    from PIL import Image
                    import torch
                    
                    # 加载图像
                    if isinstance(images[0], str):
                        # 如果是路径，加载图像
                        pil_images = [Image.open(img_path).convert('RGB') for img_path in images]
                        # 转换为tensor（这里需要根据实际的预处理逻辑调整）
                        # 暂时使用简化的处理
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])
                        image_tensors = torch.stack([transform(img) for img in pil_images])
                        image_tensors = image_tensors.to(latents.device)
                        
                        # 使用条件编码器
                        with torch.no_grad():
                            cond = pipeline.conditioner(image_tensors)
                        print(f"🔧 重新计算的图像条件: {cond.shape}")
                    else:
                        raise ValueError("Unsupported image format for condition recomputation")
                else:
                    raise ValueError("Pipeline does not have conditioner for image condition recomputation")
            else:
                raise ValueError("No images available for condition recomputation")
        
        # Prepare model input
        latent_model_input = latents
        if hasattr(config.train, 'cfg') and config.train.cfg:
            # Add negative conditioning for classifier-free guidance
            latent_model_input = torch.cat([latent_model_input, latent_model_input])
            # 🔧 修复：为CFG准备负条件
            if hasattr(sample, 'neg_cond') and sample['neg_cond'] is not None:
                neg_cond = sample['neg_cond']
            else:
                # 使用零条件作为负条件
                neg_cond = torch.zeros_like(cond)
            cond = torch.cat([neg_cond, cond])
        
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
