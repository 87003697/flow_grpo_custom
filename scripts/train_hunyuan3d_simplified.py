#!/usr/bin/env python3
"""
简化版Hunyuan3D训练脚本 - 仿照SD3的简洁内存管理

主要简化：
1. 移除复杂的设备检查
2. 简化GPU内存监控
3. 简化批量处理
4. 使用accelerator统一管理设备
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

# 🔧 添加torch.profiler用于GPU内存分析
from torch.profiler import profile, record_function, ProfilerActivity

# 统一配置管理
import ml_collections
from absl import app, flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config")

# 数据和模型相关导入
from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from flow_grpo.trainer_3d_simplified import Hunyuan3DGRPOTrainer  # 🔧 使用简化版trainer
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerImageStatTracker

logger = get_logger(__name__)

# 🚀 简化版GPU监控（可选）
def simple_gpu_log(name: str):
    """简单的GPU内存日志，不阻塞训练"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"{name}: GPU内存使用 {memory_used:.2f}GB")

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
    """主训练函数 - 简化版"""
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
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)  # 启用Flash Attention
    
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
    
    # 🚀 简化：直接加载模型，统一设备管理
    logger.info("Loading Hunyuan3D pipeline...")
    pipeline_wrapper = Hunyuan3DPipeline()
    
    # 🚀 简化：使用accelerator统一管理设备，仿照SD3
    core_pipeline = pipeline_wrapper.core_pipeline
    
    # 🚀 简化：统一设备和数据类型设置（仿照SD3）
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # 🚀 简化：直接移动到设备，不需要复杂检查
    core_pipeline.vae.to(accelerator.device, dtype=torch.float32)
    core_pipeline.conditioner.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        core_pipeline.model.to(accelerator.device)
    else:
        core_pipeline.model.to(accelerator.device, dtype=inference_dtype)
    
    # 🚀 关键修复：显式禁用VAE和conditioner的梯度，设置eval模式
    logger.info("🔧 设置VAE和conditioner为推理模式...")
    core_pipeline.vae.eval()
    core_pipeline.conditioner.eval()
    
    # 显式禁用梯度以节省显存
    for param in core_pipeline.vae.parameters():
        param.requires_grad = False
    for param in core_pipeline.conditioner.parameters():
        param.requires_grad = False
    
    logger.info("✅ VAE和conditioner梯度已禁用，已设置为eval模式")
    
    # 🚀 内存优化：训练时将VAE移动到CPU以节省显存
    logger.info("🚀 内存优化：将VAE移动到CPU以节省训练显存...")
    core_pipeline.vae.to('cpu')
    logger.info("✅ VAE已移动到CPU，显存节省约8-12GB")
    
    # 🚀 简化：按照SD3模式设置LoRA和prepare
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
        
        core_pipeline.model = get_peft_model(core_pipeline.model, lora_config)
    
    # 🔧 关键：按照SD3模式，先获取模型引用
    model = core_pipeline.model
    
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
    
    # 🔧 关键：让core_pipeline使用prepared的模型
    core_pipeline.model = model
    
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
    
    # 🚀 简化：创建trainer，不需要复杂的batch size配置
    reward_config = {"geometric_quality": 1.0, "uni3d": 0.0}  # 🚀 显存优化：禁用Uni3D节省大量显存
    trainer = Hunyuan3DGRPOTrainer(
        pipeline=pipeline_wrapper,
        reward_config=reward_config,
        device=accelerator.device,
    )
    
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
    
    # 训练循环
    global_step = 0
    first_epoch = 0
    
    # 🔧 启用torch.profiler进行详细的GPU内存分析
    prof_dir = "profiler_logs"
    os.makedirs(prof_dir, exist_ok=True)
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        # 🔧 开始profiling这个epoch
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,  # 🔧 关键：启用内存profiling
            with_stack=True,
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
        ) as prof:
            
            # 🚀 简化：直接进行采样，不需要复杂的GPU监控
            model.eval()
            epoch_samples = []
            
            with record_function("🔍 SAMPLING_PHASE"):
                simple_gpu_log(f"Epoch {epoch} - 开始采样")
                
                for batch_idx, (image_paths, prompts, metadata) in enumerate(tqdm(
                    train_dataloader, 
                    desc=f"Epoch {epoch}: sampling",
                    disable=not accelerator.is_local_main_process
                )):
                    if batch_idx >= config.sample.num_batches_per_epoch:
                        break
                    
                    with record_function(f"SAMPLE_BATCH_{batch_idx}"):
                        # 🚀 简化：直接采样，不需要复杂的候选处理
                        results = trainer.sample_meshes_with_rewards(
                            images=image_paths,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            deterministic=getattr(config, 'deterministic', False),
                            num_meshes_per_image=config.sample.num_meshes_per_image,  # 🔧 添加：多候选参数
                            kl_reward=config.sample.kl_reward,
                        )
                        
                        epoch_samples.append(results)
            
            # 🔧 采样完成，记录内存状态
            simple_gpu_log(f"Epoch {epoch} - 采样完成")
            
            with record_function("🔍 DATA_PROCESSING_PHASE"):
                # 🚀 简化：直接合并样本，统一数据格式
                all_samples = {}
                for k in epoch_samples[0].keys():
                    if k in ["meshes", "images", "prompts", "metadata"]:
                        continue
                    elif k == "rewards":
                        # 🔧 简化：直接取avg，统一为tensor格式
                        all_samples[k] = torch.cat([s[k]["avg"] for s in epoch_samples], dim=0)
                    elif isinstance(epoch_samples[0][k], torch.Tensor):
                        all_samples[k] = torch.cat([s[k] for s in epoch_samples], dim=0)
                    else:
                        # 🔧 简化：对于非tensor数据，先转换为tensor再合并
                        if k == "kl":
                            # kl现在应该是tensor，直接合并
                            all_samples[k] = torch.cat([s[k] for s in epoch_samples], dim=0)
                        else:
                            # 其他情况，尝试转换为tensor
                            all_samples[k] = torch.cat([s[k] for s in epoch_samples], dim=0)
                
                # 🚀 简化：直接处理奖励，统一格式
                rewards_avg = all_samples["rewards"]  # 现在直接是tensor
                kl_tensor = all_samples["kl"]
                
                # 🔧 简化：直接计算KL调整后的奖励
                all_samples["rewards"] = rewards_avg.unsqueeze(-1) - config.sample.kl_reward * kl_tensor
                
                # 🚀 简化：让accelerator处理分布式gather
                gathered_rewards = accelerator.gather(all_samples["rewards"])
                gathered_rewards_np = gathered_rewards.cpu().numpy()
                
                # 🔧 调试：检查rewards的分布
                logger.info(f"🔍 调试 - gathered_rewards统计:")
                logger.info(f"  shape: {gathered_rewards.shape}")
                logger.info(f"  mean: {gathered_rewards.mean().item():.6f}")
                logger.info(f"  std: {gathered_rewards.std().item():.6f}")
                logger.info(f"  min: {gathered_rewards.min().item():.6f}")
                logger.info(f"  max: {gathered_rewards.max().item():.6f}")
                
                # 🚀 简化：直接计算advantages，不需要复杂的分布式处理
                if config.per_image_stat_tracking and stat_tracker:
                    all_images = [item for s in epoch_samples for item in s["images"]]
                    advantages_np = stat_tracker.update(all_images, gathered_rewards_np.mean(axis=1))
                    advantages = torch.tensor(advantages_np, device=accelerator.device)
                else:
                    advantages = gathered_rewards.mean(axis=1)  # 平均每个样本的所有时间步
                    
                    # 🔧 调试：检查标准化前的advantages
                    logger.info(f"🔍 调试 - 标准化前advantages:")
                    logger.info(f"  shape: {advantages.shape}")
                    logger.info(f"  mean: {advantages.mean().item():.6f}")
                    logger.info(f"  std: {advantages.std().item():.6f}")
                    logger.info(f"  min: {advantages.min().item():.6f}")
                    logger.info(f"  max: {advantages.max().item():.6f}")
                    
                    # 🔧 修复：只有在标准差足够大时才标准化
                    advantages_std = advantages.std()
                    if advantages_std > 1e-8:
                        advantages = (advantages - advantages.mean()) / (advantages_std + 1e-4)
                        logger.info(f"✅ 标准化完成，std = {advantages_std.item():.6f}")
                    else:
                        logger.warning(f"⚠️  标准差过小({advantages_std.item():.6f})，跳过标准化")
                        advantages = advantages - advantages.mean()  # 只做中心化
                
                # 🔧 调试：检查标准化后的advantages
                logger.info(f"🔍 调试 - 标准化后advantages:")
                logger.info(f"  mean: {advantages.mean().item():.6f}")
                logger.info(f"  std: {advantages.std().item():.6f}")
                logger.info(f"  min: {advantages.min().item():.6f}")
                logger.info(f"  max: {advantages.max().item():.6f}")
                
                # 🔧 简化：直接扩展advantages到时间维度
                num_steps = all_samples["timesteps"].shape[1]
                advantages = advantages.unsqueeze(1).expand(-1, num_steps)
                all_samples["advantages"] = advantages
                
                # 🔧 调试：检查扩展后的advantages
                logger.info(f"🔍 调试 - 扩展后advantages:")
                logger.info(f"  shape: {advantages.shape}")
                logger.info(f"  abs().sum(dim=1): {advantages.abs().sum(dim=1)}")
                
                # 🚀 简化：直接过滤样本，不需要复杂的mask处理
                valid_mask = (advantages.abs().sum(dim=1) > 1e-6)
                logger.info(f"🔍 调试 - valid_mask: {valid_mask.sum().item()}/{len(valid_mask)} 个有效样本")
                
                # 🔧 如果没有有效样本，降低阈值或跳过过滤
                if valid_mask.sum().item() == 0:
                    logger.warning("⚠️  所有样本都被过滤掉了！尝试降低过滤阈值...")
                    valid_mask = (advantages.abs().sum(dim=1) > 1e-8)
                    logger.info(f"🔍 降低阈值后: {valid_mask.sum().item()}/{len(valid_mask)} 个有效样本")
                    
                    if valid_mask.sum().item() == 0:
                        logger.warning("⚠️  仍然没有有效样本！跳过过滤，使用所有样本...")
                        valid_mask = torch.ones(len(advantages), dtype=torch.bool, device=advantages.device)
                
                # 🔧 修复：安全的样本过滤，处理形状不匹配
                for key in all_samples.keys():
                    if isinstance(all_samples[key], torch.Tensor):
                        # 检查tensor维度是否与valid_mask匹配
                        if all_samples[key].shape[0] == valid_mask.shape[0]:
                            all_samples[key] = all_samples[key][valid_mask]
                        else:
                            print(f"⚠️  跳过过滤 {key}: shape {all_samples[key].shape} vs mask {valid_mask.shape}")
                    else:
                        print(f"⚠️  跳过非tensor类型 {key}: {type(all_samples[key])}")
                
                logger.info(f"Training on {valid_mask.sum().item()} samples")
                
                # 🔧 修复：确保训练时按照config.train.batch_size切分数据
                if "latents" in all_samples:
                    latents = all_samples["latents"]
                    if isinstance(latents, list):
                        # 如果是列表,先转换为tensor
                        latents = torch.stack(latents, dim=1)  # [B, T, ...]
                    all_samples["latents"] = latents[:, :-1]
                    all_samples["next_latents"] = latents[:, 1:]
                
                # 🔧 关键修复：将数据切分为符合train.batch_size的小批次
                total_samples = all_samples["latents"].shape[0]
                train_batch_size = config.train.batch_size
                
                if total_samples > train_batch_size:
                    # 只取前train_batch_size个样本进行训练
                    for key in all_samples.keys():
                        if isinstance(all_samples[key], torch.Tensor):
                            # 检查tensor维度是否与total_samples匹配
                            if all_samples[key].shape[0] == total_samples:
                                all_samples[key] = all_samples[key][:train_batch_size]
                                logger.info(f"🔧 切分 {key}: {total_samples} → {train_batch_size}")
                            elif key == "positive_image_cond":
                                # 特殊处理positive_image_cond：它可能有不同的batch size但仍需要切分
                                if all_samples[key].shape[0] >= train_batch_size:
                                    all_samples[key] = all_samples[key][:train_batch_size]
                                    logger.info(f"🔧 切分 {key}: {all_samples[key].shape[0]} → {train_batch_size}")
                                else:
                                    # 如果positive_image_cond的batch size小于train_batch_size，重复它
                                    repeat_factor = train_batch_size // all_samples[key].shape[0]
                                    remainder = train_batch_size % all_samples[key].shape[0]
                                    repeated_cond = all_samples[key].repeat(repeat_factor, 1, 1, 1)
                                    if remainder > 0:
                                        repeated_cond = torch.cat([repeated_cond, all_samples[key][:remainder]], dim=0)
                                    all_samples[key] = repeated_cond
                                    logger.info(f"🔧 扩展 {key}: {all_samples[key].shape[0]} → {train_batch_size}")
                            else:
                                logger.info(f"⚠️  跳过切分 {key}: shape {all_samples[key].shape} vs total_samples {total_samples}")
                        else:
                            logger.info(f"⚠️  跳过非tensor类型 {key}: {type(all_samples[key])}")
                    
                    logger.info(f"🔧 数据切分：从{total_samples}个样本切分为{train_batch_size}个样本用于训练")
            
            # 🔧 数据处理完成，记录内存状态
            simple_gpu_log(f"Epoch {epoch} - 数据处理完成")
            
            # 🚀 简化：直接训练（仿照SD3）
            with record_function("🔍 TRAINING_PHASE"):
                for inner_epoch in range(config.train.num_inner_epochs):
                    model.train()
                    
                    # 🚀 内存优化：训练前清理GPU内存
                    torch.cuda.empty_cache()
                    simple_gpu_log(f"训练前内存清理")
                    
                    with record_function("⚠️  CRITICAL_TRAIN_STEP"):
                        # 🚀 简化：直接使用trainer训练（这里会OOM）
                        loss_info = trainer.train_step(
                            samples=all_samples,
                            pipeline=core_pipeline,
                            optimizer=optimizer,
                            config=config,
                            accelerator=accelerator,
                            autocast=autocast,  # 🔧 传入autocast函数
                        )
                    
                    # 更新EMA
                    if ema is not None:
                        ema.update()
            
            # 🔧 训练完成，记录内存状态
            simple_gpu_log(f"Epoch {epoch} - 训练完成")
            
            prof.step()  # 🔧 profiler step
        
        # 🚀 简化：直接记录日志
        accelerator.log({
            "epoch": epoch,
            "reward_avg": gathered_rewards_np.mean(),
            "kl": all_samples["kl"].mean().cpu().numpy(),
            "advantages": advantages.mean().cpu().numpy(),
        }, step=global_step)
        
        global_step += 1
        
        # 🚀 简化：直接保存检查点
        if epoch % config.save_freq == 0:
            save_dir = os.path.join(config.save_dir, f"checkpoint_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存模型
            model_to_save = accelerator.unwrap_model(model)
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
            
            logger.info(f"Saved checkpoint to {save_dir}")
        
        simple_gpu_log(f"Epoch {epoch} - 完成")
    
    # 🔧 保存profiler报告
    logger.info(f"🔍 Profiler日志保存在: {prof_dir}")
    logger.info("📊 查看方法: tensorboard --logdir profiler_logs")

if __name__ == "__main__":
    app.run(main) 