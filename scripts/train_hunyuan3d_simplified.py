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
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 设置种子
    if hasattr(config, 'seed') and config.seed is not None:
        set_seed(config.seed)
    
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
    
    # 🚀 简化：直接设置LoRA（仿照SD3）
    if config.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "img_attn.qkv", "img_attn.proj",
                "txt_attn.qkv", "txt_attn.proj",
                "img_mlp.0", "img_mlp.2",
                "txt_mlp.0", "txt_mlp.2",
                "latent_in", "cond_in",
                "final_layer.linear"
            ],
            lora_dropout=0.1,
            bias="none",
        )
        
        core_pipeline.model = get_peft_model(core_pipeline.model, lora_config)
        trainable_params = [p for p in core_pipeline.model.parameters() if p.requires_grad]
    else:
        trainable_params = core_pipeline.model.parameters()
    
    # 🚀 简化：直接使用accelerator.prepare（仿照SD3）
    model = accelerator.prepare(core_pipeline.model)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # 设置EMA（仿照SD3）
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_params,
            decay=config.train.ema_decay,
            device=accelerator.device
        )
    
    # 🚀 简化：创建trainer，不需要复杂的batch size配置
    reward_config = {"geometric_quality": 0.3, "uni3d": 0.7}
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
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        # 🚀 简化：直接进行采样，不需要复杂的GPU监控
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
            
            # 🚀 简化：直接采样，不需要复杂的候选处理
            results = trainer.sample_meshes_with_rewards(
                images=image_paths,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                deterministic=getattr(config, 'deterministic', False),
                kl_reward=config.sample.kl_reward,
            )
            
            epoch_samples.append(results)
        
        # 🚀 简化：直接合并样本，不需要复杂的设备检查
        all_samples = {}
        for k in epoch_samples[0].keys():
            if k in ["meshes", "images", "prompts", "metadata"]:
                continue
            elif k == "rewards":
                # 特殊处理rewards字典
                all_samples[k] = {}
                for reward_key in epoch_samples[0][k].keys():
                    all_samples[k][reward_key] = torch.cat([s[k][reward_key] for s in epoch_samples], dim=0)
            elif isinstance(epoch_samples[0][k], torch.Tensor):
                all_samples[k] = torch.cat([s[k] for s in epoch_samples], dim=0)
            else:
                all_samples[k] = [item for s in epoch_samples for item in s[k]]
        
        # 🚀 简化：直接移动到设备，不需要复杂检查（仿照SD3）
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor) and value.device != accelerator.device:
                all_samples[key] = value.to(accelerator.device)
            elif isinstance(value, list) and key == "kl":
                # 🔧 修复：将kl列表转换为tensor
                all_samples[key] = torch.cat(value, dim=0) if isinstance(value[0], torch.Tensor) else torch.tensor(value, device=accelerator.device)
        
        # 🚀 简化：直接处理奖励
        rewards_avg = all_samples["rewards"]["avg"]
        kl_tensor = all_samples["kl"]
        
        # 🔧 确保kl_tensor是tensor
        if not isinstance(kl_tensor, torch.Tensor):
            kl_tensor = torch.tensor(kl_tensor, device=accelerator.device, dtype=torch.float32)
        
        all_samples["rewards"]["ori_avg"] = rewards_avg.clone()
        all_samples["rewards"]["avg"] = (
            rewards_avg.unsqueeze(-1) - 
            config.sample.kl_reward * kl_tensor
        )
        
        # 🚀 简化：直接gather奖励（仿照SD3）
        gathered_rewards = {
            key: accelerator.gather(value) 
            for key, value in all_samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() 
            for key, value in gathered_rewards.items()
        }
        
        # 🚀 简化：直接计算advantages（仿照SD3）
        if config.per_image_stat_tracking and stat_tracker:
            all_images = [item for s in epoch_samples for item in s["images"]]
            advantages_np = stat_tracker.update(all_images, gathered_rewards['avg'])
            advantages = torch.tensor(advantages_np, device=accelerator.device)
        else:
            advantages = gathered_rewards['avg']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            advantages = torch.tensor(advantages, device=accelerator.device)
        
        # 🚀 简化：直接按进程分割（仿照SD3）
        advantages = advantages.reshape(accelerator.num_processes, -1)[accelerator.process_index]
        all_samples["advantages"] = advantages
        
        # 🚀 简化：直接过滤样本（仿照SD3）
        mask = (all_samples["advantages"].abs() > 1e-6)
        
        # 🔧 修复：正确应用mask，特别处理嵌套字典
        filtered_samples = {}
        for k, v in all_samples.items():
            if k == "rewards":
                # 特殊处理rewards字典
                filtered_samples[k] = {}
                for reward_key, reward_tensor in v.items():
                    # 确保reward_tensor和mask形状匹配
                    if reward_tensor.shape == mask.shape:
                        filtered_samples[k][reward_key] = reward_tensor[mask]
                    else:
                        # 如果形状不匹配,保持原样
                        filtered_samples[k][reward_key] = reward_tensor
            elif isinstance(v, torch.Tensor) and len(v) == len(mask):
                filtered_samples[k] = v[mask]
            else:
                # 对于不需要过滤或长度不匹配的数据，保持原样
                filtered_samples[k] = v
        
        all_samples = filtered_samples
        
        logger.info(f"Training on {mask.sum().item()} samples")
        
        # 🚀 简化：SD3式数据重组
        if "latents" in all_samples:
            latents = all_samples["latents"]
            if isinstance(latents, list):
                # 如果是列表,先转换为tensor
                latents = torch.stack(latents, dim=1)  # [B, T, ...]
            all_samples["latents"] = latents[:, :-1]
            all_samples["next_latents"] = latents[:, 1:]
        
        # 🚀 简化：直接训练（仿照SD3）
        for inner_epoch in range(config.train.num_inner_epochs):
            model.train()
            
            # 🚀 简化：直接使用trainer训练
            loss_info = trainer.train_step(
                samples=all_samples,
                pipeline=core_pipeline,
                optimizer=optimizer,
                config=config,
                accelerator=accelerator,
            )
            
            # 更新EMA
            if ema is not None:
                ema.update()
        
        # 🚀 简化：直接记录日志（仿照SD3）
        accelerator.log({
            "epoch": epoch,
            **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
            "kl": all_samples["kl"].mean().cpu().numpy(),
        }, step=global_step)
        
        global_step += 1
        
        # 🚀 简化：直接保存检查点
        if epoch % config.save_freq == 0:
            save_dir = os.path.join(config.save_dir, f"checkpoint_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存模型
            model_to_save = accelerator.unwrap_model(model)
            if hasattr(model_to_save, "save_pretrained"):
                model_to_save.save_pretrained(save_dir)
            else:
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
            
            # 保存配置
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                import json
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info(f"Saved checkpoint to {save_dir}")
        
        simple_gpu_log(f"Epoch {epoch} - 完成")

if __name__ == "__main__":
    app.run(main) 