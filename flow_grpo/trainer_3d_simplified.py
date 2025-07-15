"""
简化版Hunyuan3D GRPO Trainer - 仿照SD3的简洁方式

主要简化：
1. 移除复杂的设备检查和内存监控
2. 简化条件处理
3. 使用accelerator统一管理设备
4. 简化错误处理
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from concurrent import futures

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from .diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob
from .diffusers_patch.hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob


class Hunyuan3DGRPOTrainer:
    """简化版GRPO Trainer for Hunyuan3D"""
    
    def __init__(
        self,
        pipeline: Hunyuan3DPipeline,
        reward_config: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        初始化简化版训练器
        
        Args:
            pipeline: Hunyuan3DPipeline实例
            reward_config: 奖励函数配置
            device: 设备
        """
        self.pipeline = pipeline
        self.device = device
        
        # 默认奖励配置
        if reward_config is None:
            reward_config = {"geometric_quality": 0.3, "uni3d": 0.7}
        
        # 🚀 简化：直接创建奖励函数
        self.reward_fn = multi_mesh_score(device, reward_config)
        
        # 🚀 简化：直接移动到设备
        self.pipeline.core_pipeline.to(device)
    
    def sample_meshes_with_rewards(
        self,
        images: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        deterministic: bool = False,
        kl_reward: float = 0.0,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        mc_algo: str = None,
        box_v: float = 1.01,
        num_chunks: int = 8000,
        executor: Optional[futures.ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """
        简化版采样函数
        
        Args:
            images: 输入图像路径列表
            num_inference_steps: 推理步数
            guidance_scale: 引导scale
            deterministic: 是否确定性
            kl_reward: KL奖励系数
            其他参数: mesh生成参数
            
        Returns:
            包含latents, log_probs, kl, rewards等的字典
        """
        # 🚀 简化：直接处理图像
        from PIL import Image
        pil_images = [Image.open(img_path).convert('RGBA') for img_path in images]
        
        # 🚀 简化：直接编码条件
        core_pipeline = self.pipeline.core_pipeline
        cond_inputs = core_pipeline.prepare_image(pil_images)
        image_tensor = cond_inputs.pop('image')
        
        # 🚀 编码正面图像条件（不使用CFG）
        positive_image_cond = core_pipeline.encode_cond(
            image=image_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=False,  # 分离编码
            dual_guidance=False,  # 🔧 修复：添加缺失的dual_guidance参数
        )
        
        # 🚀 简化：直接使用pipeline生成
        meshes, all_latents, all_log_probs, all_kl = hunyuan3d_pipeline_with_logprob(
            core_pipeline,
            image=pil_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            deterministic=deterministic,
            kl_reward=kl_reward,
            positive_image_cond=positive_image_cond,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            mc_algo=mc_algo,
            box_v=box_v,
            num_chunks=num_chunks,
        )
        
        # 🚀 简化：直接计算奖励
        rewards, reward_metadata = self.reward_fn(meshes, images, {}, images=images)
        
        # 🚀 简化：从reward字典中提取平均分数
        if isinstance(rewards, dict):
            # multi_mesh_score返回一个字典，使用'avg'键的值
            avg_rewards = rewards.get('avg', [0.0] * len(meshes))
        else:
            # 如果是列表或数字，直接使用
            avg_rewards = rewards if isinstance(rewards, list) else [rewards]
        
        # 确保奖励是数字列表
        if not isinstance(avg_rewards, list):
            avg_rewards = [avg_rewards]
        
        rewards_tensor = {
            "avg": torch.tensor(avg_rewards, device=self.device, dtype=torch.float32)
        }
        
        return {
            "meshes": meshes,
            "images": images,
            "prompts": [f"3D model from {img}" for img in images],
            "latents": all_latents,
            "log_probs": all_log_probs,
            "kl": all_kl,
            "rewards": rewards_tensor,
            "timesteps": torch.arange(num_inference_steps, device=self.device).unsqueeze(0).repeat(len(images), 1),
            "positive_image_cond": positive_image_cond,
            "metadata": reward_metadata,
        }
    
    def compute_log_prob_3d(
        self,
        pipeline,
        sample: Dict[str, torch.Tensor],
        step_index: int,
        config: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        简化版log概率计算
        
        Args:
            pipeline: 管道
            sample: 样本数据
            step_index: 时间步索引
            config: 配置
            
        Returns:
            (prev_sample, log_prob, prev_sample_mean, std_dev)
        """
        # 🚀 简化：直接获取数据
        latents = sample["latents"][:, step_index]
        next_latents = sample["next_latents"][:, step_index]
        timestep = sample["timesteps"][:, step_index]
        
        # 🚀 简化：直接使用条件
        if "positive_image_cond" in sample:
            cond = sample["positive_image_cond"]
            if isinstance(cond, dict) and 'main' in cond:
                cond = cond['main']
        else:
            raise ValueError("No image conditions found in sample")
        
        # 🚀 简化：直接预测噪声
        with torch.amp.autocast('cuda'):
            contexts = {'main': cond}
            noise_pred = pipeline.model(latents, timestep.float(), contexts)
        
        # 🚀 简化：直接计算log概率
        deterministic = getattr(config, 'deterministic', False)
        
        prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
            scheduler=pipeline.scheduler,
            model_output=noise_pred,
            timestep=timestep[0],
            sample=latents,
            prev_sample=next_latents,
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
        简化版训练步骤
        
        Args:
            samples: 样本数据
            pipeline: 管道
            optimizer: 优化器
            config: 配置
            accelerator: 加速器
            
        Returns:
            训练指标字典
        """
        info = defaultdict(list)
        num_timesteps = samples["timesteps"].shape[1]
        
        # 🚀 简化：直接训练每个时间步
        for j in range(num_timesteps):
            with accelerator.accumulate(pipeline.model):
                with accelerator.autocast():
                    # 计算log概率
                    prev_sample, log_prob, prev_sample_mean, std_dev = self.compute_log_prob_3d(
                        pipeline, samples, j, config
                    )
                    
                    # 参考log概率（如果需要KL正则化）
                    if getattr(config.train, 'beta', 0) > 0:
                        with torch.no_grad():
                            with pipeline.model.disable_adapter():
                                _, log_prob_ref, _, _ = self.compute_log_prob_3d(
                                    pipeline, samples, j, config
                                )
                    else:
                        log_prob_ref = log_prob
                
                # 🚀 简化：直接计算GRPO损失
                advantages = torch.clamp(
                    samples["advantages"],
                    -getattr(config.train, 'adv_clip_max', 5.0),
                    getattr(config.train, 'adv_clip_max', 5.0),
                )
                
                # 如果advantages是2D的，取对应时间步
                if advantages.dim() == 2:
                    advantages = advantages[:, j]
                
                # 计算比率
                ratio = torch.exp(log_prob - samples["log_probs"][:, j])
                
                # PPO损失
                loss1 = -advantages * ratio
                loss2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - getattr(config.train, 'clip_range', 0.2),
                    1.0 + getattr(config.train, 'clip_range', 0.2),
                )
                loss = torch.max(loss1, loss2).mean()
                
                # KL损失
                if getattr(config.train, 'beta', 0) > 0:
                    kl_loss = getattr(config.train, 'beta', 0) * (log_prob - log_prob_ref)
                    loss = loss + kl_loss.mean()
                
                # 🚀 简化：直接反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if getattr(config.train, 'max_grad_norm', None) is not None:
                    accelerator.clip_grad_norm_(
                        pipeline.model.parameters(),
                        config.train.max_grad_norm
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录指标
                info["loss"].append(loss.item())
                info["ratio"].append(ratio.mean().item())
                info["advantages"].append(advantages.mean().item())
        
        # 🚀 简化：直接返回平均指标
        return {k: np.mean(v) for k, v in info.items()} 