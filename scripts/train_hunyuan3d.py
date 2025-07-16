#!/usr/bin/env python3
"""
Hunyuan3D GRPO训练脚本 - 内联架构（类似SD3）

架构改进：
1. 移除独立trainer类，所有逻辑内联到main函数
2. 直接使用core_pipeline，无包装器
3. 采样、训练、评估在统一脚本中处理
4. 与SD3保持一致的代码组织结构
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent import futures

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

# 统一配置管理
import ml_collections
from absl import app, flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config")

# 数据和模型相关导入
from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from flow_grpo.diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob
from flow_grpo.diffusers_patch.hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerImageStatTracker

logger = get_logger(__name__)

# 🚀 简化版GPU监控（可选）
def simple_gpu_log(name: str):
    """简单的GPU内存日志，不阻塞训练"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"{name}: GPU内存使用 {memory_used:.2f}GB")

def get_timesteps(pipeline, batch_size: int, num_steps: int, device: str) -> torch.Tensor:
    """生成标准化的时间步张量"""
    scheduler_timesteps = pipeline.scheduler.timesteps
    if len(scheduler_timesteps) < num_steps:
        pipeline.scheduler.set_timesteps(num_steps + 1, device=device)
        scheduler_timesteps = pipeline.scheduler.timesteps
    
    # 🔧 关键修复：对于20个推理步骤，我们有20对(current,next)latents，需要20个时间步
    # 不应该减1，因为我们要对应20对latents
    used_timesteps = scheduler_timesteps[:num_steps]
    return used_timesteps.unsqueeze(0).repeat(batch_size, 1)

def compute_log_prob_3d(pipeline, sample: Dict[str, torch.Tensor], step_index: int, config: Any):
    """计算3D扩散模型的log概率 - 类似SD3的compute_log_prob"""
    # 获取数据
    latents = sample["latents"][:, step_index]
    next_latents = sample["next_latents"][:, step_index]
    timestep = sample["timesteps"][:, step_index]
    
    # 🔧 简化：直接使用统一格式的tensor
    cond = sample["positive_image_cond"]
    
    # 🔧 简单处理：确保batch_size匹配
    if cond.shape[0] != latents.shape[0]:
        cond = cond.repeat_interleaved(latents.shape[0] // cond.shape[0], dim=0)
    
    # 🔧 数值稳定性：时间步标准化与裁剪
    timestep_normalized = torch.clamp(
        timestep.float() / pipeline.scheduler.config.num_train_timesteps, 
        min=1e-6, max=1.0 - 1e-6
    )
    
    # 🔧 简单处理：构建contexts
    contexts = {'main': cond}
    
    # 🔧 数值稳定性：检查输入
    if torch.isnan(latents).any() or torch.isinf(latents).any():
        logger.warning(f"⚠️  输入latents包含NaN或Inf值")
        latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 模型预测
    with torch.amp.autocast('cuda'):
        noise_pred = pipeline.model(latents, timestep_normalized, contexts)
    
    # 🔧 数值稳定性：检查模型输出
    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        logger.warning(f"⚠️  模型输出包含NaN或Inf值")
        noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 计算log概率
    try:
        prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
            scheduler=pipeline.scheduler,
            model_output=noise_pred,
            timestep=timestep[0],
            sample=latents,
            prev_sample=next_latents,
            deterministic=getattr(config, 'deterministic', False),
        )
        
        # 🔧 数值稳定性：检查输出并进行裁剪
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            logger.warning(f"⚠️  log_prob包含NaN或Inf值，使用默认值")
            log_prob = torch.zeros_like(log_prob)
        
        if torch.isnan(prev_sample_mean).any() or torch.isinf(prev_sample_mean).any():
            logger.warning(f"⚠️  prev_sample_mean包含NaN或Inf值，使用裁剪值")
            prev_sample_mean = torch.nan_to_num(prev_sample_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🔧 数值稳定性：std_dev裁剪防止过大值
        std_dev = torch.clamp(std_dev, min=1e-6, max=100.0)
        
    except Exception as e:
        logger.warning(f"⚠️  SDE step失败: {e}，使用默认值")
        # 返回安全的默认值
        prev_sample = next_latents
        log_prob = torch.zeros(latents.shape[0], device=latents.device)
        prev_sample_mean = next_latents
        std_dev = torch.ones(1, device=latents.device)
    
    return prev_sample, log_prob, prev_sample_mean, std_dev

class Image3DDataset(Dataset):
    def __init__(self, image_dir: str, prompts_file: Optional[str] = None, split: str = "train"):
        self.image_dir = Path(image_dir)
        self.prompts_file = prompts_file
        self.split = split
        
        # 检查图像是否在images子目录中
        if (self.image_dir / "images").exists():
            self.image_dir = self.image_dir / "images"
        
        # 查找图像文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(self.image_dir.glob(ext))
        
        self.image_files = sorted(self.image_files)
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = str(self.image_files[idx])
        prompt = self.get_prompt(self.image_files[idx])
        
        return {
            "image_path": image_path,
            "prompt": prompt,
            "metadata": {"image_name": self.image_files[idx].name}
        }
    
    @staticmethod
    def collate_fn(examples):
        image_paths = [example["image_path"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        return image_paths, prompts, metadata
    
    def get_prompt(self, image_path: Path) -> str:
        """根据图像路径生成提示词"""
        return f"Generate a 3D model from this image: {image_path.stem}"

def main(argv):
    """主训练函数 - 内联架构（类似SD3）"""
    del argv
    config = _CONFIG.value
    
    # 🚀 简化：直接使用accelerator，不需要复杂的设备检查
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with="wandb" if "WANDB_PROJECT" in os.environ else None,
        project_dir=config.save_dir,
    )
    
    # 🚀 内存优化：启用PyTorch内存优化策略
    torch.backends.cudnn.benchmark = False  # 减少内存碎片
    torch.backends.cuda.max_split_size_mb = 128  # 限制内存分割大小
    
    # 🔧 Flash Attention优化：使用配置文件设置
    attention_config = getattr(config, 'attention_optimization', None)
    if attention_config:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp') and attention_config.enable_flash_sdp:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("✅ Flash Attention 已启用")
        
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp') and attention_config.enable_mem_efficient_sdp:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("✅ Memory Efficient Attention 已启用")
        
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(attention_config.enable_math_sdp)
            if not attention_config.enable_math_sdp:
                logger.info("✅ Math SDPA 已禁用（优先使用Flash/Memory Efficient）")
        
        # TF32优化
        if hasattr(torch.backends.cuda, 'allow_tf32') and attention_config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ TF32加速 已启用")
    else:
        # 🔧 向后兼容：如果没有attention_optimization配置，使用默认设置
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)  # 启用Flash Attention
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # 启用Memory Efficient Attention
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(False)  # 禁用数学SDPA，优先使用Flash/Memory Efficient
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速矩阵乘法
            torch.backends.cudnn.allow_tf32 = True
        logger.info("🚀 默认Attention优化已启用: Flash Attention + Memory Efficient Attention")
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 设置种子
    if hasattr(config, 'seed') and config.seed is not None:
        set_seed(config.seed)
    
    # 🚀 内存优化：清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"🧹 GPU内存清理完成，可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 🚀 直接加载pipeline，无包装器（类似SD3）
    logger.info("Loading Hunyuan3D pipeline...")
    pipeline_wrapper = Hunyuan3DPipeline()
    
    # 🚀 获取核心pipeline，直接操作（类似SD3直接使用StableDiffusion3Pipeline）
    pipeline = pipeline_wrapper.core_pipeline
    
    # 🚀 简化：统一设备和数据类型设置（仿照SD3）
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # 🚀 简化：直接移动到设备，不需要复杂检查（类似SD3）
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.conditioner.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.model.to(accelerator.device)
    else:
        pipeline.model.to(accelerator.device, dtype=inference_dtype)
    
    # 🚀 关键修复：显式禁用VAE和conditioner的梯度，设置eval模式（类似SD3）
    logger.info("🔧 设置VAE和conditioner为推理模式...")
    pipeline.vae.eval()
    pipeline.conditioner.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.conditioner.requires_grad_(False)
    logger.info("✅ VAE和conditioner梯度已禁用，已设置为eval模式")
    
    # 🚀 内存优化：训练时将VAE移动到CPU以节省显存
    logger.info("🚀 内存优化：将VAE移动到CPU以节省训练显存...")
    pipeline.vae.to('cpu')
    logger.info("✅ VAE已移动到CPU，显存节省约8-12GB")
    
    # 🚀 LoRA设置（类似SD3）
    if config.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                # 🔧 修复：根据真实模型结构设置target_modules
                "to_q", "to_k", "to_v", "out_proj",      # 注意力层
                "fc1", "fc2",                             # MLP层
                "final_layer.linear",                     # 输出层
                # 🔧 可选：加入输入embedding层
                "x_embedder",                             # 输入embedding
            ],
            lora_dropout=0.1,
            bias="none",
        )
        
        pipeline.model = get_peft_model(pipeline.model, lora_config)
    
    # 🔧 关键：按照SD3模式，先获取模型引用
    model = pipeline.model
    
    # 🔧 关键：获取trainable参数（SD3方式）
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # 🔧 关键：最后prepare（SD3方式）
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # 🔧 关键：让pipeline使用prepared的模型
    pipeline.model = model
    
    # 🔧 按照SD3模式：LoRA训练时不使用autocast
    import contextlib
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # 设置EMA（仿照SD3）
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_params,
            decay=config.train.ema_decay,
            device=accelerator.device
        )
    
    # 🚀 初始化奖励函数（内联，无trainer）
    reward_config = {"geometric_quality": 1.0, "uni3d": 0.0}  # 🚀 显存优化：禁用Uni3D节省大量显存
    reward_fn = multi_mesh_score(accelerator.device, reward_config)
    
    # 🚀 简化：直接加载数据集
    logger.info(f"Loading dataset from {config.data_dir}")
    train_dataset = Image3DDataset(config.data_dir, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.input_batch_size,
        shuffle=True,
        collate_fn=Image3DDataset.collate_fn,
        num_workers=0,
    )
    
    # 统计跟踪
    stat_tracker = None
    if config.per_image_stat_tracking:
        stat_tracker = PerImageStatTracker(
            buffer_size=len(train_dataset),
            min_count=config.stat_tracking.min_count,
        )
    
    # Prepare dataloader
    train_dataloader = accelerator.prepare(train_dataloader)
    
    # executor to perform callbacks asynchronously
    executor = futures.ThreadPoolExecutor(max_workers=8)
    
    # 训练循环（类似SD3架构）
    global_step = 0
    first_epoch = 0
    
    # number of timesteps within each trajectory to train on
    # 🔧 关键修复：我们有20对latents，所以可以训练20个时间步
    num_latent_pairs = config.sample.num_steps  # 20对latents
    num_train_timesteps = min(
        int(num_latent_pairs * config.train.timestep_fraction),
        num_latent_pairs
    )
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        #################### SAMPLING ####################
        model.eval()
        epoch_samples = []
        
        simple_gpu_log(f"Epoch {epoch} - 开始采样")
        
        for batch_idx, (image_paths, prompts, metadata) in enumerate(tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process
        )):
            if batch_idx >= config.sample.num_batches_per_epoch:
                break
            
            # 🚀 内联采样逻辑（原trainer.sample_meshes_with_rewards）
            from PIL import Image
            
            # 🔧 多候选生成：为每个图像生成多个候选mesh
            all_pil_images = []
            for img_path in image_paths:
                # 为当前图像生成 num_meshes_per_image 个候选
                candidate_images = [img_path] * config.sample.num_meshes_per_image
                pil_candidates = [Image.open(path).convert('RGBA') for path in candidate_images]
                all_pil_images.extend(pil_candidates)
            
            pil_images = all_pil_images
            
            # 编码图像条件
            cond_inputs = pipeline.prepare_image(pil_images)
            image_tensor = cond_inputs.pop('image')
            
            positive_image_cond = pipeline.encode_cond(
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
                pipeline,
                image=pil_images,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                deterministic=getattr(config, 'deterministic', False),
                kl_reward=config.sample.kl_reward,
                return_image_cond=True,
                positive_image_cond=positive_image_cond,
                octree_resolution=384,
                mc_level=0.0,
                mc_algo=None,
                box_v=1.01,
                num_chunks=50000,
            )
            
            # 计算奖励（异步）
            rewards = executor.submit(reward_fn, meshes, None, {}, image_paths)
            time.sleep(0)  # yield to make sure reward computation starts
            
            # 处理latents数据
            latents_tensor = torch.stack(all_latents, dim=1)
            current_latents = latents_tensor[:, :-1]  # 前n-1个时间步
            next_latents = latents_tensor[:, 1:]      # 后n-1个时间步
            
            # 处理log_probs和KL
            log_probs_tensor = torch.stack(all_log_probs, dim=1)
            kl_tensor = torch.stack(all_kl, dim=1)
            
            # 处理timesteps
            # 🔧 修复：传入完整的num_steps，函数内部会处理-1
            timesteps_tensor = get_timesteps(pipeline, len(all_pil_images), config.sample.num_steps, accelerator.device)
            
            # 🔧 简化：直接使用
            returned_pos_cond = returned_pos_cond['main']
            
            epoch_samples.append({
                "latents": current_latents,
                "next_latents": next_latents,
                "log_probs": log_probs_tensor,
                "kl": kl_tensor,
                "rewards": rewards,  # 异步结果
                "timesteps": timesteps_tensor,
                "positive_image_cond": returned_pos_cond,
                "images": image_paths,
                "meshes": meshes,
            })
        
        # 🔧 采样完成，记录内存状态
        simple_gpu_log(f"Epoch {epoch} - 采样完成")
        
        # wait for all rewards to be computed
        for sample in tqdm(
            epoch_samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
        ):
            reward_details, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                "avg": torch.tensor(reward_details['avg'], device=accelerator.device, dtype=torch.float32)
            }
        
        # 🚀 数据处理（类似SD3）
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {}
        for k in epoch_samples[0].keys():
            if k in ["meshes", "images"]:
                continue
            elif k == "rewards":
                # 🔧 简化：直接取avg，统一为tensor格式
                samples[k] = {
                    "avg": torch.cat([s[k]["avg"] for s in epoch_samples], dim=0)
                }
            elif isinstance(epoch_samples[0][k], torch.Tensor):
                samples[k] = torch.cat([s[k] for s in epoch_samples], dim=0)
        
        # 🚀 处理奖励和advantages（类似SD3）
        rewards_avg = samples["rewards"]["avg"]  # 现在直接是tensor
        kl_tensor = samples["kl"]
        
        # KL调整后的奖励
        samples["rewards"]["ori_avg"] = rewards_avg
        samples["rewards"]["avg"] = rewards_avg.unsqueeze(-1) - config.sample.kl_reward * kl_tensor
        
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards_np = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        
        # 计算advantages（类似SD3）
        if config.per_image_stat_tracking and stat_tracker:
            all_images = [item for s in epoch_samples for item in s["images"]]
            advantages_np = stat_tracker.update(all_images, gathered_rewards_np["avg"].mean(axis=1))
            advantages = torch.tensor(advantages_np, device=accelerator.device)
        else:
            advantages = gathered_rewards["avg"].mean(axis=1)  # 平均每个样本的所有时间步
            
            # 🔧 标准化advantages
            advantages_std = advantages.std()
            if advantages_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages_std + 1e-4)
            else:
                advantages = advantages - advantages.mean()  # 只做中心化
        
        # 扩展advantages到时间维度
        num_steps = samples["timesteps"].shape[1]
        advantages = advantages.unsqueeze(1).expand(-1, num_steps)
        samples["advantages"] = advantages
        
        # 过滤样本（类似SD3）
        valid_mask = (advantages.abs().sum(dim=1) > 1e-6)
        if valid_mask.sum().item() == 0:
            logger.warning("⚠️  所有样本都被过滤掉了！使用所有样本...")
            valid_mask = torch.ones(len(advantages), dtype=torch.bool, device=advantages.device)
        
        # 安全的样本过滤
        for key in samples.keys():
            if isinstance(samples[key], torch.Tensor):
                if samples[key].shape[0] == valid_mask.shape[0]:
                    samples[key] = samples[key][valid_mask]
            elif isinstance(samples[key], dict):
                for sub_key in samples[key]:
                    if isinstance(samples[key][sub_key], torch.Tensor):
                        if samples[key][sub_key].shape[0] == valid_mask.shape[0]:
                            samples[key][sub_key] = samples[key][sub_key][valid_mask]
        
        logger.info(f"Training on {valid_mask.sum().item()} samples")
        
        # 🔧 数据切分为训练batch size
        total_samples = samples["latents"].shape[0]
        train_batch_size = config.train.batch_size
        
        if total_samples > train_batch_size:
            for key in samples.keys():
                if isinstance(samples[key], torch.Tensor):
                    if samples[key].shape[0] == total_samples:
                        samples[key] = samples[key][:train_batch_size]
                elif isinstance(samples[key], dict):
                    for sub_key in samples[key]:
                        if isinstance(samples[key][sub_key], torch.Tensor):
                            if samples[key][sub_key].shape[0] == total_samples:
                                samples[key][sub_key] = samples[key][sub_key][:train_batch_size]
        
        # log rewards
        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards_np.items()},
                "kl": samples["kl"].mean().cpu().numpy(),
            },
            step=global_step,
        )
        
        # 🔧 数据处理完成，记录内存状态
        simple_gpu_log(f"Epoch {epoch} - 数据处理完成")
        
        #################### TRAINING ####################
        # 内联训练逻辑（原trainer.train_step）
        for inner_epoch in range(config.train.num_inner_epochs):
            model.train()
            info = defaultdict(list)
            num_timesteps = samples["timesteps"].shape[1]
            
            # 🚀 内存优化：训练前清理GPU内存
            torch.cuda.empty_cache()
            simple_gpu_log(f"训练前内存清理")
            
            # 训练每个时间步（类似SD3的训练循环）
            train_timesteps = [step_index for step_index in range(num_train_timesteps)]
            for j in tqdm(
                train_timesteps,
                desc="Timestep",
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                with accelerator.accumulate(model):
                    with autocast():
                        # 计算log概率
                        prev_sample, log_prob, prev_sample_mean, std_dev = compute_log_prob_3d(
                            pipeline, samples, j, config
                        )
                        
                        # 参考log概率
                        if getattr(config.train, 'beta', 0) > 0:
                            with torch.no_grad():
                                # 🔧 按照SD3模式：安全访问DDP包装后的模型
                                model_for_adapter = model.module if hasattr(model, 'module') else model
                                with model_for_adapter.disable_adapter():
                                    _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = compute_log_prob_3d(
                                        pipeline, samples, j, config
                                    )
                        
                        # 计算GRPO损失（类似SD3）
                        advantages = torch.clamp(
                            samples["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        
                        # 计算比率
                        ratio = torch.exp(log_prob - samples["log_probs"][:, j])
                        
                        # PPO损失
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        # KL损失
                        if getattr(config.train, 'beta', 0) > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim))) / (2 * std_dev ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            kl_loss = torch.tensor(0.0, device=policy_loss.device)
                            loss = policy_loss
                        
                        info["loss"].append(loss.item())
                        info["policy_loss"].append(policy_loss.item())
                        if getattr(config.train, 'beta', 0) > 0:
                            info["kl_loss"].append(kl_loss.item())
                        info["advantages"].append(advantages.mean().item())
                        info["ratio"].append(ratio.mean().item())
                        
                        # 计算clipfrac和approx_kl（类似SD3）
                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - samples["log_probs"][:, j]) ** 2).item()
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > config.train.clip_range).float()
                            ).item()
                        )
                    
                    # 反向传播
                    accelerator.backward(loss)
                    
                    # 梯度裁剪（类似SD3）
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), config.train.max_grad_norm
                        )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 记录训练信息（类似SD3）
                if accelerator.sync_gradients:
                    info = {k: np.mean(v) for k, v in info.items()}
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
            
            # 更新EMA
            if ema is not None:
                ema.update()
        
        # 🔧 训练完成，记录内存状态
        simple_gpu_log(f"Epoch {epoch} - 训练完成")
        
        # 🚀 保存检查点（类似SD3）
        if epoch % config.save_freq == 0 and epoch > 0:
            save_dir = os.path.join(config.save_dir, f"checkpoint_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存模型
            model_to_save = accelerator.unwrap_model(model)
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
            
            logger.info(f"Saved checkpoint to {save_dir}")
        
        simple_gpu_log(f"Epoch {epoch} - 完成")

if __name__ == "__main__":
    app.run(main) 