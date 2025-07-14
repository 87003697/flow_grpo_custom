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
        sample_batch_size: int = 1,      # 🔧 新增：采样batch size
        train_batch_size: int = 2,       # 🔧 新增：训练batch size
    ):
        """
        Initialize the 3D GRPO trainer with SD3-style batch handling.
        
        Args:
            pipeline: Hunyuan3DPipeline 包装类
            reward_config: 3D reward configuration dict, e.g., {"geometric_quality": 0.3, "uni3d": 0.7}
            device: Device to run training on
            sample_batch_size: Batch size for sampling phase
            train_batch_size: Batch size for training phase
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
        
        # 🔧 新增：仿照SD3预先准备负面图像条件
        print("🔧 预先准备负面图像条件...")
        self._prepare_negative_conditions(sample_batch_size, train_batch_size)
    
    def _prepare_negative_conditions(self, sample_batch_size: int, train_batch_size: int):
        """
        预先准备不同batch size的负面图像条件，仿照SD3策略
        
        Args:
            sample_batch_size: 采样阶段的batch size
            train_batch_size: 训练阶段的batch size
        """
        # 获取pipeline的核心组件
        core_pipeline = self.pipeline.core_pipeline
        
        # 🔧 SD3式：生成单个负面条件，然后复制到不同batch size
        with torch.no_grad():
            # 使用conditioner的unconditional_embedding方法
            if hasattr(core_pipeline, 'conditioner'):
                # 生成单个样本的负面条件
                neg_cond_single = core_pipeline.conditioner.unconditional_embedding(
                    batch_size=1,
                    device=self.device
                )
                print(f"🔧 生成的单个负面条件类型: {type(neg_cond_single)}")
                
                if isinstance(neg_cond_single, dict):
                    print(f"🔧 负面条件字典keys: {list(neg_cond_single.keys())}")
                    for key, value in neg_cond_single.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}.shape: {value.shape}")
                
                # 🔧 SD3式：预先准备不同batch size的负面条件
                # 仿照SD3: sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
                self.sample_neg_image_cond = self._expand_condition(neg_cond_single, sample_batch_size)
                self.train_neg_image_cond = self._expand_condition(neg_cond_single, train_batch_size)
                
                print(f"🔧 SD3式：采样阶段负面条件准备完成 (batch_size={sample_batch_size})")
                print(f"🔧 SD3式：训练阶段负面条件准备完成 (batch_size={train_batch_size})")
            else:
                raise ValueError("Pipeline does not have conditioner for negative condition preparation")
    
    def _expand_condition(self, cond_single: Union[torch.Tensor, Dict], target_batch_size: int):
        """
        将单个条件扩展到指定的batch size
        
        Args:
            cond_single: 单个样本的条件 (batch_size=1)
            target_batch_size: 目标batch size
            
        Returns:
            扩展后的条件 (batch_size=target_batch_size)
        """
        if isinstance(cond_single, torch.Tensor):
            return cond_single.repeat(target_batch_size, 1, 1)
        elif isinstance(cond_single, dict):
            expanded_cond = {}
            for key, value in cond_single.items():
                if isinstance(value, torch.Tensor):
                    expanded_cond[key] = value.repeat(target_batch_size, 1, 1)
                else:
                    expanded_cond[key] = value
            return expanded_cond
        else:
            raise ValueError(f"Unsupported condition type: {type(cond_single)}")
    
    def _get_negative_condition_for_batch(self, batch_size: int, mode: str = "sample"):
        """
        获取指定batch size的负面条件，仿照SD3的动态裁剪方式
        
        Args:
            batch_size: 需要的batch size
            mode: "sample" 或 "train"
            
        Returns:
            对应batch size的负面条件
        """
        if mode == "sample":
            base_neg_cond = self.sample_neg_image_cond
        else:
            base_neg_cond = self.train_neg_image_cond
        
        # 🔧 SD3式：动态裁剪到所需batch size
        # 仿照SD3: train_neg_prompt_embeds[:len(sample["prompt_embeds"])]
        if isinstance(base_neg_cond, dict):
            result = {}
            for key, value in base_neg_cond.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value[:batch_size]
                else:
                    result[key] = value
            return result
        else:
            return base_neg_cond[:batch_size]
    
    def sample_meshes_with_rewards(
        self,
        images: List[str],
        input_batch_size: int = 2,           # 🔧 新增：输入图像数量
        num_meshes_per_image: int = 2,       # 🔧 新增：每个图像的候选数量
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        deterministic: bool = False,
        kl_reward: float = 0.0,
        # 🔧 新增：mesh 配置参数
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        mc_algo: str = None,
        box_v: float = 1.01,
        num_chunks: int = 8000,
        executor: Optional[futures.ThreadPoolExecutor] = None,
    ) -> Dict[str, Any]:
        """
        Sample meshes with rewards using multi-candidate generation.
        
        Args:
            images: List of image paths
            input_batch_size: Number of different images to process
            num_meshes_per_image: Number of mesh candidates per image
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            deterministic: Whether to use deterministic mode
            kl_reward: KL reward coefficient
            octree_resolution: Resolution for octree-based mesh extraction
            mc_level: Marching cubes level (iso-value)
            mc_algo: Marching cubes algorithm ('mc', 'dmc', or None)
            box_v: Bounding box volume for mesh extraction
            num_chunks: Number of chunks for mesh processing
            executor: Thread executor for async reward computation
            
        Returns:
            Dictionary with generated meshes, latents, log_probs, etc.
        """
        
        # 🔧 多候选生成逻辑
        all_meshes = []
        all_latents = []
        all_log_probs = []
        all_kl = []
        all_positive_image_conds = []  # 🔧 修复：分离存储正面条件
        expanded_images = []
        
        total_meshes = input_batch_size * num_meshes_per_image
        
        with gpu_timer(f"🎯 多候选生成 - {input_batch_size}图像×{num_meshes_per_image}候选={total_meshes}mesh"):
            # 为每个图像生成多个候选
            for i in range(input_batch_size):
                if i >= len(images):
                    break  # 防止越界
                    
                image_path = images[i]
                
                # 🔧 为当前图像生成多个候选
                with gpu_timer(f"图像{i+1}/{input_batch_size} - {num_meshes_per_image}个候选"):
                    # 重复当前图像，生成多个候选
                    candidate_images = [image_path] * num_meshes_per_image
                    
                    # 获取实际pipeline
                    actual_pipeline = self.pipeline.core_pipeline if hasattr(self.pipeline, 'core_pipeline') else self.pipeline
                    
                    # 🔧 SD3式：先编码正面图像条件
                    from PIL import Image
                    if isinstance(candidate_images[0], str):
                        pil_images = [Image.open(img_path).convert('RGBA') for img_path in candidate_images]
                    else:
                        pil_images = candidate_images
                    
                    # 使用pipeline的方法编码图像条件
                    cond_inputs = actual_pipeline.prepare_image(pil_images)
                    image_tensor = cond_inputs.pop('image')
                    
                    # 编码正面图像条件（不使用CFG）
                    positive_image_cond = actual_pipeline.encode_cond(
                        image=image_tensor,
                        additional_cond_inputs=cond_inputs,
                        do_classifier_free_guidance=False,  # 🔧 SD3式：分离编码
                        dual_guidance=False,
                    )
                    
                    # 🔧 SD3式：获取对应的负面条件
                    current_batch_size = len(candidate_images)
                    negative_image_cond = self._get_negative_condition_for_batch(current_batch_size, mode="sample")
                    
                    print(f"🔧 SD3式条件编码完成:")
                    print(f"  正面条件: {positive_image_cond.shape if isinstance(positive_image_cond, torch.Tensor) else 'dict'}")
                    print(f"  负面条件: {negative_image_cond.shape if isinstance(negative_image_cond, torch.Tensor) else 'dict'}")
                    
                    # 🔧 SD3式：传递分离的条件到pipeline
                    meshes, latents, log_probs, kl, returned_pos_cond = hunyuan3d_pipeline_with_logprob(
                        actual_pipeline,
                        image=candidate_images,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        deterministic=deterministic,
                        kl_reward=kl_reward,
                        # 🔧 新增：传递 mesh 配置参数
                        octree_resolution=octree_resolution,
                        mc_level=mc_level,
                        mc_algo=mc_algo,
                        box_v=box_v,
                        num_chunks=num_chunks,
                        return_image_cond=True,
                        positive_image_cond=positive_image_cond,  # 🔧 SD3式：直接传入
                        negative_image_cond=negative_image_cond,  # 🔧 SD3式：直接传入
                    )
                    
                    all_meshes.extend(meshes if isinstance(meshes, list) else [meshes])
                    all_latents.extend(latents)
                    all_log_probs.extend(log_probs)
                    all_kl.append(kl)
                    all_positive_image_conds.append(returned_pos_cond)  # 🔧 修复：存储返回的正面条件
                    
                    # 扩展图像列表（用于reward计算）
                    expanded_images.extend(candidate_images)
        
        # 🔧 新增：验证数据一致性
        expected_total = min(input_batch_size, len(images)) * num_meshes_per_image
        assert len(all_meshes) == expected_total, f"Expected {expected_total} meshes, got {len(all_meshes)}"
        assert len(expanded_images) == expected_total, f"Expected {expected_total} images, got {len(expanded_images)}"
        
        # 计算rewards
        with gpu_timer("🏆 奖励函数计算"):
            if executor:
                reward_future = executor.submit(
                    self.reward_fn, 
                    all_meshes, 
                    None,  # prompts - 不需要
                    {},    # metadata
                    expanded_images  # 🔧 传递图像
                )
                rewards = reward_future
            else:
                rewards = self.reward_fn(all_meshes, None, {}, expanded_images)
        
        # 数据打包
        with gpu_timer("📦 结果打包"):
            # 按SD3方式处理tensor
            latents_tensor = torch.stack(all_latents, dim=1) if all_latents else torch.empty(0)
            log_probs_tensor = torch.stack(all_log_probs, dim=1) if all_log_probs else torch.empty(0)
            
            # 处理KL tensor
            if all_kl:
                all_kl_tensors = []
                for kl in all_kl:
                    if isinstance(kl, torch.Tensor):
                        all_kl_tensors.append(kl)
                    elif isinstance(kl, (list, tuple)):
                        if len(kl) > 0 and isinstance(kl[0], torch.Tensor):
                            kl_tensor = torch.stack(kl).transpose(0, 1)  # (batch_size, num_steps)
                        else:
                            kl_tensor = torch.tensor(kl)
                        all_kl_tensors.append(kl_tensor)
                    else:
                        all_kl_tensors.append(torch.tensor(kl))
                
                kl_tensor = torch.cat(all_kl_tensors, dim=0)
            else:
                kl_tensor = torch.empty(0)
            
            # 🔧 完全仿照SD3的时间步处理方式
            # SD3方式: timesteps = pipeline.scheduler.timesteps.repeat(batch_size, 1)
            # 不再使用复杂的retrieve_timesteps，直接使用scheduler的timesteps
            
            # 生成timesteps - 🔧 完全仿照SD3的实现
            num_steps = latents_tensor.shape[1] - 1 if latents_tensor.numel() > 0 else 20
            
            # 🔧 关键修复：完全按照SD3的timesteps处理方式
            # SD3: timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)
            actual_pipeline = self.pipeline.core_pipeline if hasattr(self.pipeline, 'core_pipeline') else self.pipeline
            
            try:
                # 🔧 SD3式：直接使用scheduler的timesteps，而不是重新生成
                # 确保scheduler的timesteps已经正确设置
                if hasattr(actual_pipeline.scheduler, 'timesteps') and len(actual_pipeline.scheduler.timesteps) > 0:
                    # 使用scheduler现有的timesteps
                    scheduler_timesteps = actual_pipeline.scheduler.timesteps
                    
                    # 如果scheduler的timesteps数量不够，重新设置
                    if len(scheduler_timesteps) < num_steps:
                        actual_pipeline.scheduler.set_timesteps(num_steps, device=self.device)
                        scheduler_timesteps = actual_pipeline.scheduler.timesteps
                    
                    # 取前num_steps个时间步
                    used_timesteps = scheduler_timesteps[:num_steps]
                    
                    # 🔧 SD3式：重复timesteps到每个样本
                    # SD3: timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)
                    timesteps_tensor = used_timesteps.unsqueeze(0).repeat(expected_total, 1)
                    
                    print(f"🔧 SD3式时间步生成成功:")
                    print(f"  使用scheduler.timesteps: {scheduler_timesteps[:5]}... (前5个)")
                    print(f"  生成的时间步形状: {timesteps_tensor.shape}")
                    print(f"  时间步范围: [{timesteps_tensor.min():.1f}, {timesteps_tensor.max():.1f}]")
                    
                else:
                    # 如果scheduler没有timesteps，先设置
                    actual_pipeline.scheduler.set_timesteps(num_steps, device=self.device)
                    scheduler_timesteps = actual_pipeline.scheduler.timesteps
                    timesteps_tensor = scheduler_timesteps.unsqueeze(0).repeat(expected_total, 1)
                    
                    print(f"🔧 重新设置scheduler timesteps:")
                    print(f"  timesteps形状: {timesteps_tensor.shape}")
                    print(f"  时间步范围: [{timesteps_tensor.min():.1f}, {timesteps_tensor.max():.1f}]")
                
            except Exception as e:
                print(f"🚨 SD3式时间步生成失败，回退到安全模式: {e}")
                # 🔧 安全回退：生成标准的递减时间步序列
                max_timesteps = getattr(actual_pipeline.scheduler.config, 'num_train_timesteps', 1000)
                timesteps_list = torch.linspace(max_timesteps-1, 0, num_steps, device=self.device, dtype=torch.long)
                timesteps_tensor = timesteps_list.unsqueeze(0).repeat(expected_total, 1)
                
                print(f"🔧 安全回退时间步:")
                print(f"  生成递减序列: [{timesteps_tensor[0, 0]:.0f} -> {timesteps_tensor[0, -1]:.0f}]")
                print(f"  时间步形状: {timesteps_tensor.shape}")
            
            # 🔧 修复：处理正面图像条件，仿照SD3的方式
            positive_image_cond_tensor = all_positive_image_conds[0] if all_positive_image_conds else None
            
            return {
                "meshes": all_meshes,
                "images": expanded_images,
                "latents": latents_tensor,
                "log_probs": log_probs_tensor,
                "kl": kl_tensor,
                "rewards": rewards,
                "timesteps": timesteps_tensor,
                "positive_image_cond": positive_image_cond_tensor,  # 🔧 修复：使用正确的字段名
                # 🔧 新增：元数据
                "metadata": {
                    "input_batch_size": input_batch_size,
                    "num_meshes_per_image": num_meshes_per_image,
                    "total_meshes": expected_total,
                }
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
        Uses SD3-style dynamic condition assembly for flexible batch handling.
        
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
        
        current_batch_size = latents.shape[0]
        print(f"🔧 SD3式条件处理：当前batch_size={current_batch_size}")
        
        # 🔧 SD3式动态条件组合 - 简化版本，避免复杂的映射逻辑
        if "positive_image_cond" in sample and sample["positive_image_cond"] is not None:
            # 获取正面条件
            pos_cond = sample["positive_image_cond"]
            if isinstance(pos_cond, dict) and 'main' in pos_cond:
                pos_cond = pos_cond['main']
            
            print(f"🔧 正面条件原始形状: {pos_cond.shape}")
            
            # 🔧 关键修复：简化条件处理，避免复杂的metadata映射
            # 如果条件的batch size不匹配，直接重复或切片
            if pos_cond.shape[0] != current_batch_size:
                if pos_cond.shape[0] == 1:
                    # 如果只有1个条件，重复到current_batch_size
                    pos_cond = pos_cond.repeat(current_batch_size, 1, 1)
                    print(f"🔧 重复条件到batch size: {pos_cond.shape}")
                elif pos_cond.shape[0] > current_batch_size:
                    # 如果条件太多，切片到current_batch_size
                    pos_cond = pos_cond[:current_batch_size]
                    print(f"🔧 切片条件到batch size: {pos_cond.shape}")
                else:
                    # 如果条件太少，重复最后一个条件
                    last_cond = pos_cond[-1:].repeat(current_batch_size - pos_cond.shape[0], 1, 1)
                    pos_cond = torch.cat([pos_cond, last_cond], dim=0)
                    print(f"🔧 扩展条件到batch size: {pos_cond.shape}")
            
            print(f"🔧 最终正面条件: {pos_cond.shape}")
            
            # 🔧 完全仿照SD3的CFG处理逻辑
            if hasattr(config.train, 'cfg') and config.train.cfg:
                # 🔧 SD3式：动态组合负面和正面条件
                neg_cond = self._get_negative_condition_for_batch(current_batch_size, mode="train")
                if isinstance(neg_cond, dict) and 'main' in neg_cond:
                    neg_cond = neg_cond['main']
                
                print(f"🔧 负面条件（动态裁剪）: {neg_cond.shape}")
                
                # 🔧 SD3式：组合CFG格式 [negative_batch, positive_batch]
                cond = torch.cat([neg_cond, pos_cond], dim=0)
                print(f"🔧 SD3式组合后CFG条件: {cond.shape}")
            else:
                # 🔧 SD3式：禁用CFG时只使用正面条件
                cond = pos_cond
                print(f"🔧 非CFG模式，使用正面条件: {cond.shape}")
                
        else:
            # 🔧 向后兼容：处理旧格式的image_cond
            if "image_cond" in sample and sample["image_cond"] is not None:
                cond = sample["image_cond"]
                if isinstance(cond, dict) and 'main' in cond:
                    cond = cond['main']
                print(f"🔧 向后兼容：使用image_cond {cond.shape}")
            else:
                raise ValueError("No image conditions found in sample")
        
        # 🔧 准备模型输入 - 仿照SD3的latent_model_input处理
        latent_model_input = latents
        if hasattr(config.train, 'cfg') and config.train.cfg:
            # 🔧 CFG模式：复制latents（仿照SD3）
            latent_model_input = torch.cat([latent_model_input, latent_model_input])
            print(f"🔧 CFG模式：latent_model_input.shape = {latent_model_input.shape}")
            print(f"🔧 CFG模式：cond.shape = {cond.shape}")
        
        # 🔧 关键修复：确保训练阶段与采样阶段的时间步处理完全一致
        # 采样阶段：timestep = timestep / self.scheduler.config.num_train_timesteps
        # 训练阶段：也必须做相同的标准化
        
        # Convert timestep to normalized format
        # 🔧 重要修复：保持与采样阶段一致的标准化
        # 采样阶段：timestep / self.scheduler.config.num_train_timesteps
        timestep_float = timestep.float()
        timestep_normalized = timestep_float / pipeline.scheduler.config.num_train_timesteps
        
        # 🔧 确保数据类型与latents一致（仿照采样阶段）
        timestep_tensor = timestep_normalized.to(device=latents.device, dtype=latents.dtype)
        
        # 🔧 CFG模式下也需要复制timestep（保持与SD3一致）
        if hasattr(config.train, 'cfg') and config.train.cfg:
            timestep_tensor = torch.cat([timestep_tensor, timestep_tensor])
        
        # 🔧 移除clamp操作：SD3不对timesteps进行clamp
        # timestep_tensor = torch.clamp(timestep_tensor, 0.0, 1.0)  # 删除这行
        
        # 🔧 验证所有张量的形状和设备
        print(f"🔍 模型输入验证:")
        print(f"  latent_model_input.shape: {latent_model_input.shape}")
        print(f"  timestep_tensor.shape: {timestep_tensor.shape}")
        print(f"  timestep_tensor.dtype: {timestep_tensor.dtype}")
        print(f"  timestep_tensor.device: {timestep_tensor.device}")
        print(f"  timestep_tensor范围: [{timestep_tensor.min():.6f}, {timestep_tensor.max():.6f}]")
        print(f"  cond.shape: {cond.shape}")
        print(f"  所有张量在同一设备: {latent_model_input.device == timestep_tensor.device == cond.device}")
        
        # 🔧 添加更安全的NaN检查
        try:
            has_nan_latent = torch.isnan(latent_model_input).any().item()
            has_nan_timestep = torch.isnan(timestep_tensor).any().item()
            has_nan_cond = torch.isnan(cond).any().item()
            print(f"  NaN检查: latent={has_nan_latent}, timestep={has_nan_timestep}, cond={has_nan_cond}")
        except Exception as e:
            print(f"  NaN检查失败: {e}")
        
        # Predict noise using the model
        # 🔧 修复：使用推荐的torch.amp.autocast('cuda')替代过时的torch.cuda.amp.autocast
        with torch.amp.autocast('cuda'):
            # 🔧 修复：模型期望的是contexts参数，不是cond
            contexts = {'main': cond}
            
            try:
                noise_pred = pipeline.model(latent_model_input, timestep_tensor, contexts)
                print(f"🎉 模型调用成功！noise_pred.shape: {noise_pred.shape}")
            except Exception as e:
                print(f"❌ 模型调用失败: {e}")
                print(f"🔍 错误类型: {type(e)}")
                
                # 🔧 添加更详细的调试信息
                print(f"🔍 详细调试信息:")
                print(f"  pipeline.model: {type(pipeline.model)}")
                print(f"  latent_model_input: shape={latent_model_input.shape}, dtype={latent_model_input.dtype}, device={latent_model_input.device}")
                print(f"  timestep_tensor: shape={timestep_tensor.shape}, dtype={timestep_tensor.dtype}, device={timestep_tensor.device}")
                print(f"  contexts['main']: shape={contexts['main'].shape}, dtype={contexts['main'].dtype}, device={contexts['main'].device}")
                
                # 重新抛出异常以便查看完整堆栈
                raise
        
        # 🔧 Apply classifier-free guidance if enabled（仿照SD3）
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
                            # 🔧 修复：直接使用 disable_adapter，PeftModel 必须支持此方法
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
                        pipeline.model.parameters(),  # 🔧 修复：使用 model.parameters() 而不是 pipeline.parameters()
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
    Create a reward function for 3D mesh evaluation with image support.
    
    Args:
        reward_config: 3D reward configuration dict, e.g., {"geometric_quality": 0.3, "uni3d": 0.7}
        device: Device for computation
        
    Returns:
        Reward function compatible with GRPO training and image input
    """
    if reward_config is None:
        reward_config = {
            "geometric_quality": 0.3,
            "uni3d": 0.7  # 🔧 现在可以使用图像了！
        }
    
    # Create reward function using new rewards_mesh.py
    reward_fn = multi_mesh_score(device, reward_config)
    
    def reward_fn_wrapper(meshes, prompts, metadata, images=None):  # 🔧 新增 images 参数
        """
        Compute rewards for generated meshes with image support.
        
        Args:
            meshes: List of generated mesh objects
            prompts: List of text prompts (can be None)
            metadata: Metadata dictionary
            images: List of input image paths (for image-based scoring)
            
        Returns:
            Tuple of (rewards_dict, metadata_dict)
        """
        # 🔧 传递 images 参数
        reward_details, _ = reward_fn(meshes, prompts, metadata, images)
        
        # Convert to numpy arrays for compatibility
        rewards = {}
        for key, scores in reward_details.items():
            if isinstance(scores, (list, tuple)):
                rewards[key] = np.array(scores, dtype=np.float32)
            else:
                rewards[key] = np.array([scores], dtype=np.float32)
        
        # Create metadata
        reward_metadata = {
            "num_meshes": len(meshes),
        }
        
        # Add mean and std for each score type
        for key, scores in rewards.items():
            if len(scores) > 0:
                reward_metadata[f"{key}_mean"] = scores.mean()
                reward_metadata[f"{key}_std"] = scores.std()
        
        return rewards, reward_metadata
    
    return reward_fn_wrapper
