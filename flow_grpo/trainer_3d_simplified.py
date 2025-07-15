"""
简化版Hunyuan3D GRPO Trainer - 统一数据格式版本

核心原则：
1. 统一入口：在采样阶段确保数据格式统一
2. 简单假设：训练阶段直接假设数据格式正确
3. 保持原生接口：始终使用contexts={'main': cond}
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
    """简化版GRPO Trainer - 统一数据格式处理"""
    
    def __init__(
        self,
        pipeline: Hunyuan3DPipeline,
        reward_config: Dict[str, float] = {"geometric_quality": 0.3, "uni3d": 0.7},
        device: str = "cuda",
    ):
        """初始化训练器"""
        self.pipeline = pipeline
        self.device = device
        self.reward_fn = multi_mesh_score(device, reward_config)
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
        num_chunks: int = 50000,
        num_meshes_per_image: int = 1,  # 🔧 添加：每个图像的mesh候选数量
        executor: Optional[futures.ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """采样阶段：确保数据格式统一"""
        from PIL import Image
        
        # 🔧 多候选生成：为每个图像生成多个候选mesh
        all_pil_images = []
        for img_path in images:
            # 为当前图像生成 num_meshes_per_image 个候选
            candidate_images = [img_path] * num_meshes_per_image
            pil_candidates = [Image.open(path).convert('RGBA') for path in candidate_images]
            all_pil_images.extend(pil_candidates)
        
        pil_images = all_pil_images
        
        core_pipeline = self.pipeline.core_pipeline
        
        # 编码图像条件
        cond_inputs = core_pipeline.prepare_image(pil_images)
        image_tensor = cond_inputs.pop('image')
        
        positive_image_cond = core_pipeline.encode_cond(
            image=image_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=False,
            dual_guidance=False,
        )
        
        # 🔧 关键：在这里统一格式，后续不再处理
        if not isinstance(positive_image_cond, dict):
            positive_image_cond = {'main': positive_image_cond}
        
        # 调用pipeline
        meshes, all_latents, all_log_probs, all_kl, returned_pos_cond = hunyuan3d_pipeline_with_logprob(
            core_pipeline,
            image=pil_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            deterministic=deterministic,
            kl_reward=kl_reward,
            return_image_cond=True,
            positive_image_cond=positive_image_cond,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            mc_algo=mc_algo,
            box_v=box_v,
            num_chunks=num_chunks,
        )
        
        # 计算奖励
        reward_details, metadata = self.reward_fn(meshes, None, {}, images)
        rewards_tensor = {
            "avg": torch.tensor(reward_details['avg'], device=self.device, dtype=torch.float32)
        }
        
        # 处理latents数据
        latents_tensor = torch.stack(all_latents, dim=1)
        current_latents = latents_tensor[:, :-1]  # 前n-1个时间步
        next_latents = latents_tensor[:, 1:]      # 后n-1个时间步
        
        # 处理log_probs
        log_probs_tensor = torch.stack(all_log_probs, dim=1)
        
        # 🔧 修复：处理KL tensor，确保维
        kl_tensor = torch.stack(all_kl, dim=1)
        
        # 处理timesteps
        timesteps_tensor = self._get_timesteps(len(images), num_inference_steps - 1)
        
        # 🔧 简化：直接使用
        returned_pos_cond = returned_pos_cond['main']
        
        return {
            "meshes": meshes,
            "images": images,
            "latents": current_latents,
            "next_latents": next_latents,
            "log_probs": log_probs_tensor,
            "kl": kl_tensor,  # 🔧 修复：处理KL tensor为2维张量
            "rewards": rewards_tensor,
            "timesteps": timesteps_tensor,
            "positive_image_cond": returned_pos_cond,  # 🔧 使用pipeline返回的统一格式
        }
    
    def _get_timesteps(self, batch_size: int, num_steps: int) -> torch.Tensor:
        """生成标准化的时间步张量"""
        scheduler_timesteps = self.pipeline.core_pipeline.scheduler.timesteps
        if len(scheduler_timesteps) < num_steps:
            self.pipeline.core_pipeline.scheduler.set_timesteps(num_steps + 1, device=self.device)
            scheduler_timesteps = self.pipeline.core_pipeline.scheduler.timesteps
        
        used_timesteps = scheduler_timesteps[:num_steps]
        return used_timesteps.unsqueeze(0).repeat(batch_size, 1)
    
    def compute_log_prob_3d(
        self,
        pipeline,
        sample: Dict[str, torch.Tensor],
        step_index: int,
        config: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练阶段：假设数据格式已经统一"""
        # 获取数据
        latents = sample["latents"][:, step_index]
        next_latents = sample["next_latents"][:, step_index]
        timestep = sample["timesteps"][:, step_index]
        
        # 🔧 简化：直接使用统一格式的tensor
        cond = sample["positive_image_cond"]
        
        # 🔧 简单处理：确保batch_size匹配
        if cond.shape[0] != latents.shape[0]:
            cond = cond.repeat_interleaved(latents.shape[0] // cond.shape[0], dim=0)
        
        # 🔧 简单处理：时间步标准化
        timestep_normalized = timestep.float() / pipeline.scheduler.config.num_train_timesteps
        
        # 🔧 简单处理：构建contexts
        contexts = {'main': cond}
        
        # 模型预测
        with torch.amp.autocast('cuda'):
            noise_pred = pipeline.model(latents, timestep_normalized, contexts)
        
        # 计算log概率
        prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
            scheduler=pipeline.scheduler,
            model_output=noise_pred,
            timestep=timestep[0],
            sample=latents,
            prev_sample=next_latents,
            deterministic=getattr(config, 'deterministic', False),
        )
        
        return prev_sample, log_prob, prev_sample_mean, std_dev
    
    def train_step(
        self,
        samples: Dict[str, torch.Tensor],
        pipeline,
        optimizer: torch.optim.Optimizer,
        config: Any,
        accelerator: Any,
        autocast=None,  # 🔧 添加autocast参数
    ) -> Dict[str, float]:
        """简化版训练步骤"""
        info = defaultdict(list)
        num_timesteps = samples["timesteps"].shape[1]
        
        # 训练每个时间步
        for j in range(num_timesteps):
            with accelerator.accumulate(pipeline.model):
                with (autocast() if autocast else accelerator.autocast()):
                    # 计算log概率
                    prev_sample, log_prob, prev_sample_mean, std_dev = self.compute_log_prob_3d(
                        pipeline, samples, j, config
                    )
                    
                    # 参考log概率
                    with torch.no_grad():
                        # 🔧 按照SD3模式：安全访问DDP包装后的模型
                        model_for_adapter = pipeline.model.module if hasattr(pipeline.model, 'module') else pipeline.model
                        with model_for_adapter.disable_adapter():
                            _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = self.compute_log_prob_3d(
                                    pipeline, samples, j, config
                                )
                    
                    # 计算GRPO损失
                    advantages = torch.clamp(
                        samples["advantages"][:, j],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    
                    # 计算比率
                    ratio = torch.exp(log_prob - samples["log_probs"][:, j])
                    
                    # PPO损失
                    loss1 = -advantages * ratio
                    loss2 = -advantages * torch.clamp(
                        ratio,
                        1.0 - config.train.clip_range,
                        1.0 + config.train.clip_range,
                    )
                    policy_loss = torch.max(loss1, loss2).mean()
                    
                    # 🔧 修复：正确的KL损失计算，使用prev_sample_mean而不是log_prob
                    if getattr(config.train, 'beta', 0) > 0:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim))) / (2 * std_dev ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + config.train.beta * kl_loss
                    else:
                        kl_loss = torch.tensor(0.0, device=policy_loss.device)
                        loss = policy_loss
                    
                # 反向传播
                accelerator.backward(loss)
                
                # 记录信息
                info["loss"].append(loss.item())
                info["kl_loss"].append(kl_loss.mean().item())
                info["advantages"].append(advantages.mean().item())
                info["ratio"].append(ratio.mean().item())
                
                # 梯度裁剪 - 🔧 完全禁用以解决FP16问题
                if accelerator.sync_gradients:
                    # 🔧 完全跳过梯度裁剪以避免FP16 unscaling问题
                    print(f"⚠️  梯度裁剪已禁用以解决FP16问题")
                
                optimizer.step()
                optimizer.zero_grad()
        
        # 返回平均统计
        return {key: np.mean(values) for key, values in info.items()} 