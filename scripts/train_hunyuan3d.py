#!/usr/bin/env python3
"""
Hunyuan3D GRPO Training Script

3D reinforcement learning training for Hunyuan3D using GRPO.
Adapted from scripts/train_sd3.py for 3D mesh generation.
"""
import os
import sys
import argparse
import tempfile
import random
import time
import logging
from pathlib import Path
from collections import defaultdict
from concurrent import futures
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

import subprocess
from contextlib import contextmanager
from torch.cuda.amp import autocast

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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# HuggingFace imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Project imports
from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from flow_grpo.trainer_3d import Hunyuan3DGRPOTrainer
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker
from reward_models.rewards_mesh import multi_mesh_score

logger = logging.getLogger(__name__)


class Image3DDataset(Dataset):
    """Dataset for image-to-3D generation tasks."""
    
    def __init__(self, image_dir: str, prompts_file: Optional[str] = None, split: str = "train", image_files: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing input images
            prompts_file: Optional file containing text prompts (one per line)
            split: Dataset split ("train" or "test")
            image_files: Optional list of image file names to use directly
        """
        self.image_dir = Path(image_dir)
        self.split = split
        
        # Use provided image_files if available, otherwise find all
        if image_files:
            self.image_paths = [self.image_dir / f for f in image_files]
        else:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            self.image_paths = [
                p for p in self.image_dir.rglob("*") 
                if p.suffix.lower() in image_extensions
            ]
        
        # Load prompts
        if prompts_file and os.path.exists(prompts_file):
            with open(prompts_file, 'r') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            self.prompts = []
        
        logger.info(f"Loaded {len(self.image_paths)} images and {len(self.prompts)} prompts for {split}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image_path = self.image_paths[idx]
        prompt = self.get_prompt(image_path)
        
        # Create metadata
        metadata = {
            "image_path": str(image_path),
            "prompt": prompt,
            "split": self.split
        }
        
        return str(image_path), prompt, metadata
    
    @staticmethod
    def collate_fn(examples):
        """Collate function for DataLoader."""
        image_paths = [example[0] for example in examples]
        prompts = [example[1] for example in examples]
        metadata = [example[2] for example in examples]
        
        return image_paths, prompts, metadata

    def get_prompt(self, image_path: Path) -> str:
        """Generate a prompt for the given image."""
        idx = self.image_paths.index(image_path)
        if idx < len(self.prompts):
            return self.prompts[idx]
        else:
            # Generate prompt from filename
            filename = image_path.stem
            # Simple filename-to-prompt conversion
            if filename.isdigit():
                # For numbered files like 1.png, 2.png
                return f"a 3D model, high quality"
            else:
                # For descriptive filenames like walking_cat.png
                # Convert underscores to spaces and clean up
                prompt = filename.replace('_', ' ').replace('-', ' ')
                return f"a 3D model of {prompt}, high quality"


def create_config():
    """Create default configuration."""
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    
    # Basic settings
    config.data_dir = "data/3d_training"
    config.save_dir = "checkpoints/hunyuan3d_grpo"
    config.resume_from = None
    config.num_epochs = 100
    config.mixed_precision = "fp16"
    config.seed = 42
    config.use_lora = False
    config.eval_freq = 10
    config.save_freq = 10
    config.per_prompt_stat_tracking = True
    config.deterministic = False  # 🔧 默认使用SDE模式
    
    # Sample configuration
    config.sample = SimpleNamespace()
    config.sample.batch_size = 1  # Batch size for sampling
    config.sample.num_batches_per_epoch = 2  # Number of batches to sample per epoch (reduced for faster testing)
    config.sample.num_steps = 20  # Number of denoising steps
    config.sample.guidance_scale = 5.0
    config.sample.kl_reward = 0.1
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 4
    config.sample.global_std = 0.5
    
    # Training config
    config.train = SimpleNamespace()
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 2
    config.train.num_inner_epochs = 1
    config.train.learning_rate = 1e-5
    config.train.beta = 0.01  # KL coefficient
    config.train.clip_range = 0.2
    config.train.adv_clip_max = 5.0
    config.train.max_grad_norm = 1.0
    config.train.cfg = True
    config.train.ema = True
    config.train.ema_decay = 0.999
    
    return config


def unwrap_model(model, accelerator):
    """Unwrap model from accelerator."""
    model = accelerator.unwrap_model(model)
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(save_dir, pipeline, global_step, accelerator, ema, config):
    """Save training checkpoint."""
    save_path = os.path.join(save_dir, f"checkpoint_{global_step}")
    
    # Save accelerator state
    accelerator.save_state(save_path)
    
    # Save additional info
    save_info = {
        "global_step": global_step,
        "config": config,
    }
    
    if ema is not None:
        save_info["ema"] = ema.state_dict()
    
    torch.save(save_info, os.path.join(save_path, "training_info.pt"))
    logger.info(f"Saved checkpoint to {save_path}")


def evaluate_3d(
    trainer: Hunyuan3DGRPOTrainer,
    test_dataloader: DataLoader,
    config,
    accelerator: Accelerator,
    global_step: int,
    executor: futures.ThreadPoolExecutor,
):
    """Evaluate 3D generation quality."""
    logger.info("Starting 3D evaluation...")
    
    trainer.pipeline.model.eval()  # 使用trainer.pipeline.model而不是trainer.pipeline.pipeline
    
    eval_results = []
    eval_meshes = []
    eval_rewards = []
    
    with torch.no_grad():
        for batch_idx, (image_paths, prompts, metadata) in enumerate(test_dataloader):
            if batch_idx >= 3:  # Limit evaluation batches
                break
            
            # Generate meshes
            results = trainer.sample_meshes_with_rewards(
                images=image_paths,
                prompts=prompts,
                batch_size=len(image_paths),
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                deterministic=True,  # Use deterministic for evaluation
                kl_reward=0.0,  # No KL reward during evaluation
                executor=executor,
            )
            
            # Wait for rewards
            if hasattr(results["rewards"], 'result'):
                rewards, reward_metadata = results["rewards"].result()
            else:
                rewards = results["rewards"]
                reward_metadata = {}
            
            eval_meshes.extend(results["meshes"])
            eval_rewards.append(rewards)
            
            # Store results
            for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
                eval_results.append({
                    "image_path": image_path,
                    "prompt": prompt,
                    "geometric_score": rewards["geometric"][i],
                    "semantic_score": rewards["semantic"][i], 
                    "avg_score": rewards["avg"][i],
                })
    
    # Aggregate results
    if eval_rewards:
        all_geometric = np.concatenate([r["geometric"] for r in eval_rewards])
        all_semantic = np.concatenate([r["semantic"] for r in eval_rewards])
        all_avg = np.concatenate([r["avg"] for r in eval_rewards])
        
        eval_metrics = {
            "eval_geometric_mean": all_geometric.mean(),
            "eval_semantic_mean": all_semantic.mean(),
            "eval_avg_mean": all_avg.mean(),
            "eval_geometric_std": all_geometric.std(),
            "eval_semantic_std": all_semantic.std(),
            "eval_avg_std": all_avg.std(),
            "eval_num_samples": len(all_avg),
        }
        
        # Log to accelerator
        accelerator.log(eval_metrics, step=global_step)
        
        logger.info(f"Evaluation results: avg_score={all_avg.mean():.3f}, "
                   f"geometric={all_geometric.mean():.3f}, semantic={all_semantic.mean():.3f}")
        
        # Save some example meshes
        if accelerator.is_main_process and eval_meshes:
            eval_dir = os.path.join(config.save_dir, f"eval_{global_step}")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save a few example meshes
            num_to_save = min(5, len(eval_meshes))
            for i in range(num_to_save):
                mesh_path = os.path.join(eval_dir, f"eval_mesh_{i}.glb")
                eval_meshes[i].export(mesh_path)
            
            logger.info(f"Saved {num_to_save} evaluation meshes to {eval_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Hunyuan3D with GRPO")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data/3d_training", help="Training data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints/hunyuan3d_grpo", help="Save directory")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic (ODE) mode instead of stochastic (SDE) mode for both rollout and training")
    # 移除FlashVDM选项，完全使用标准Volume Decoding
    
    args = parser.parse_args()
    
    with gpu_timer("🚀 完整训练初始化"):
        # Create configuration
        config = create_config()
        
        # Override with command line arguments
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.save_dir:
            config.save_dir = args.save_dir
        if args.resume_from:
            config.resume_from = args.resume_from
        if args.num_epochs:
            config.num_epochs = args.num_epochs
        if args.batch_size:
            config.sample.train_batch_size = args.batch_size
        if args.learning_rate:
            config.train.learning_rate = args.learning_rate
        if args.mixed_precision:
            config.mixed_precision = args.mixed_precision
        
        # 🔧 添加deterministic配置
        config.deterministic = args.deterministic
        if args.deterministic:
            logger.info("🎯 使用确定性模式 (ODE) 进行rollout和训练")
        else:
            logger.info("🎲 使用随机模式 (SDE) 进行rollout和训练")
        
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            log_with="wandb" if "WANDB_PROJECT" in os.environ else None,
            project_dir=config.save_dir,
        )
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state)
        
        # Set seed
        if args.seed is not None:
            set_seed(args.seed)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize pipeline and models
        logger.info("Loading Hunyuan3D pipeline...")
        with gpu_timer("Hunyuan3D模型加载"):
            # 🎯 使用包装器（统一接口）
            pipeline_wrapper = Hunyuan3DPipeline()
            
            # 🔧 始终使用标准Volume Decoding（确保稳定性）
            logger.info("🔧 使用标准 Volume Decoding（推荐用于稳定性）")
            logger.info("✅ 标准 Volume Decoding 已启用")
            
            # 移动核心pipeline到指定设备
            pipeline_wrapper.core_pipeline.to(accelerator.device)
        
        # Initialize reward models
        logger.info("Setting up reward configuration...")
        with gpu_timer("奖励函数初始化"):
            reward_config = {
                "geometric_quality": 0.3,
                "uni3d": 0.7
            }
            
            # Create trainer - 明确：只传递包装类
            trainer = Hunyuan3DGRPOTrainer(
                pipeline=pipeline_wrapper,  # 传递包装类，不是内部pipeline
                reward_config=reward_config,
                device=accelerator.device,
            )
    
    # Create reward function
    # reward_fn = create_3d_reward_function(reward_config, accelerator.device)  # 删除重复创建
    
    # Setup datasets
    logger.info(f"Loading datasets from {config.data_dir}")
    
    # Check if we have train/test structure or just images/ structure
    train_dir = os.path.join(config.data_dir, "train")
    test_dir = os.path.join(config.data_dir, "test")
    images_dir = os.path.join(config.data_dir, "images")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Standard train/test structure
        train_dataset = Image3DDataset(
            image_dir=train_dir,
            prompts_file=os.path.join(config.data_dir, "train_prompts.txt"),
            split="train"
        )
        test_dataset = Image3DDataset(
            image_dir=test_dir, 
            prompts_file=os.path.join(config.data_dir, "test_prompts.txt"),
            split="test"
        )
    elif os.path.exists(images_dir):
        # Single images/ folder - split it for train/test
        logger.info("Using images/ folder, splitting for train/test")
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # For reproducible splits
        
        # Split 80/20 for train/test
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        logger.info(f"Split {len(image_files)} images: {len(train_files)} train, {len(test_files)} test")
        
        # Create datasets with image file lists
        train_dataset = Image3DDataset(
            image_dir=images_dir,
            prompts_file=None,  # Will generate prompts from filenames
            split="train",
            image_files=train_files
        )
        test_dataset = Image3DDataset(
            image_dir=images_dir,
            prompts_file=None,
            split="test", 
            image_files=test_files
        )
    else:
        raise FileNotFoundError(f"Neither train/test directories nor images/ directory found in {config.data_dir}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        shuffle=True,
        collate_fn=Image3DDataset.collate_fn,
        num_workers=2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        shuffle=False,
        collate_fn=Image3DDataset.collate_fn,
        num_workers=2,
    )
    
    # Setup model for training - 明确访问路径：通过core_pipeline
    core_pipeline = trainer.pipeline.core_pipeline  # 获取核心pipeline
    model = core_pipeline.model          # 核心扩散模型
    vae = core_pipeline.vae              # VAE编码器
    conditioner = core_pipeline.conditioner  # 条件编码器
    
    if config.use_lora:
        # Add LoRA adapters
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_v", "to_k", "to_out.0"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()
    
    # Move models to device - 注意：VAE和conditioner已经在正确的设备上了
    model = accelerator.prepare(model)
    # vae.to(accelerator.device, dtype=torch.float32)  # 删除这行，VAE已经在正确设备上
    # conditioner.to(accelerator.device)  # 删除这行，conditioner已经在正确设备上
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Setup EMA
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_params,
            decay=config.train.ema_decay,
            device=accelerator.device,
        )
    
    # Setup stat tracking
    stat_tracker = None
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    
    # Prepare for training
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )
    
    # Thread executor for async operations
    executor = futures.ThreadPoolExecutor(max_workers=4)
    
    # Training info
    samples_per_epoch = len(train_dataloader) * config.sample.train_batch_size
    total_train_batch_size = (
        config.train.batch_size * 
        accelerator.num_processes * 
        config.train.gradient_accumulation_steps
    )
    
    logger.info("***** Running 3D GRPO Training *****")
    logger.info(f"  Num training samples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")
    
    # Initialize training
    first_epoch = 0
    global_step = 0
    
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    
    # Training loop
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        #################### SAMPLING ####################
        model.eval()  # 只需要设置核心扩散模型为eval模式
        
        epoch_samples = []
        with gpu_timer(f"📊 Epoch {epoch} - 完整采样阶段"):
            for batch_idx, (image_paths, prompts, metadata) in enumerate(tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process
            )):
                if batch_idx >= config.sample.num_batches_per_epoch:
                    break
                
                # Sample meshes with rewards
                with gpu_timer(f"样本 {batch_idx+1}/{config.sample.num_batches_per_epoch} - 采样+评分"):
                    results = trainer.sample_meshes_with_rewards(
                        images=image_paths,
                        prompts=prompts,
                        batch_size=len(image_paths),
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        deterministic=config.deterministic,
                        kl_reward=config.sample.kl_reward,
                        executor=executor,
                    )
                
                epoch_samples.append(results)
        
        # Wait for all rewards
        for sample in tqdm(
            epoch_samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
        ):
            if hasattr(sample["rewards"], 'result'):
                rewards, reward_metadata = sample["rewards"].result()
                sample["rewards"] = {
                    key: torch.as_tensor(value, device=accelerator.device).float()
                    for key, value in rewards.items()
                }
        
        # Collate samples
        all_samples = {
            k: torch.cat([s[k] for s in epoch_samples], dim=0)
            if not isinstance(epoch_samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in epoch_samples], dim=0)
                for sub_key in epoch_samples[0][k]
            }
            for k in epoch_samples[0].keys()
            if k not in ["meshes", "images", "prompts"]  # Skip non-tensor data
        }
        
        # 🔍 Hunyuan3D Train Debug: 采样后的数据形状
        # ⚠️ 重要对比：
        # SD3: latents (batch_size, num_steps+1, 16, 32, 32)
        # Hunyuan3D: latents (batch_size, num_steps+1, 1024, 64)
        # 相同点：log_probs (batch_size, num_steps), kl (batch_size, num_steps), rewards (batch_size,)
        print(f"🔍 Hunyuan3D Train Debug - 采样后数据:")
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor):
                if key == "latents":
                    print(f"  {key}.shape: {value.shape} (Hunyuan3D vs SD3)")
                    print(f"    Hunyuan3D: (batch, steps+1, 1024, 64)")
                    print(f"    SD3:       (batch, steps+1, 16, 32, 32)")
                else:
                    print(f"  {key}.shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}.shape: {sub_value.shape}")
        
        # Adjust rewards with KL penalty
        all_samples["rewards"]["ori_avg"] = all_samples["rewards"]["avg"].clone()
        
        # 🔧 修复：按照SD3的方式处理KL tensor
        rewards_avg = all_samples["rewards"]["avg"]  # shape: (batch_size,)
        kl_tensor = all_samples["kl"]  # shape: (batch_size, num_steps) - 已经通过torch.cat合并
        
        # 🔧 调试：打印tensor形状
        print(f"🔍 Tensor shapes debug:")
        print(f"  rewards_avg.shape: {rewards_avg.shape}")
        print(f"  kl_tensor.shape: {kl_tensor.shape}")
        
        # 🔧 修复：确保维度匹配
        # rewards_avg: (batch_size,) -> (batch_size, 1)
        # kl_tensor: (batch_size, num_steps)
        # 结果: (batch_size, num_steps)
        all_samples["rewards"]["avg"] = (
            rewards_avg.unsqueeze(-1) -  # (batch_size, 1)
            config.sample.kl_reward * kl_tensor  # (batch_size, num_steps)
        )  # 结果: (batch_size, num_steps)
        
        # Gather rewards across processes
        gathered_rewards = {
            key: accelerator.gather(value) 
            for key, value in all_samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() 
            for key, value in gathered_rewards.items()
        }
        
        # Log metrics
        accelerator.log({
            "epoch": epoch,
            **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
            "kl": all_samples["kl"].mean().cpu().numpy(),
        }, step=global_step)
        
        # Compute advantages
        if config.per_prompt_stat_tracking and stat_tracker:
            # Per-prompt stat tracking - 只有当我们处理所有样本时才启用
            all_prompts = []
            for sample in epoch_samples:
                all_prompts.extend(sample["prompts"])
            
            # 🔧 修复：只有当处理的样本数等于训练集大小时才使用per-prompt跟踪
            if len(all_prompts) == len(train_dataset):
                advantages = stat_tracker.update(all_prompts, gathered_rewards['avg'])
                advantages = torch.as_tensor(advantages, device=accelerator.device)
            else:
                logger.warning(f"Processed {len(all_prompts)} samples but have {len(train_dataset)} in dataset. Using global advantages.")
                # 使用全局advantages
                advantages = gathered_rewards['avg']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
                advantages = torch.as_tensor(advantages, device=accelerator.device)
        else:
            # Global advantages
            advantages = gathered_rewards['avg']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            advantages = torch.as_tensor(advantages, device=accelerator.device)
        
        # �� 修复：正确处理advantages的维度
        # 关键问题：advantages现在是(batch_size, num_steps)，但我们需要在batch维度上进行筛选
        # 解决方案：计算每个样本的平均advantage，用于筛选整个样本
        print(f"🔍 Advantages处理 - 修复前:")
        print(f"  advantages.shape: {advantages.shape}")
        print(f"  期望: (batch_size, num_steps) 或 (batch_size,)")
        
        if advantages.dim() == 2:
            # 如果advantages是2D的 (batch_size, num_steps)，计算每个样本的平均advantage
            sample_advantages = advantages.mean(dim=1)  # (batch_size,)
            print(f"  计算样本平均advantages: {sample_advantages.shape}")
        else:
            # 如果advantages是1D的 (batch_size,)，直接使用
            sample_advantages = advantages
            print(f"  直接使用advantages: {sample_advantages.shape}")
        
        # 按进程分割 - 现在在batch维度上分割
        batch_size = sample_advantages.shape[0]
        samples_per_process = batch_size // accelerator.num_processes
        
        # 取当前进程的部分
        start_idx = accelerator.process_index * samples_per_process
        end_idx = start_idx + samples_per_process
        if end_idx > batch_size or accelerator.process_index == accelerator.num_processes - 1:
            end_idx = batch_size  # 最后一个进程处理剩余的样本
        
        print(f"🔍 进程分割:")
        print(f"  进程 {accelerator.process_index}/{accelerator.num_processes}")
        print(f"  处理样本 {start_idx}:{end_idx} (共{batch_size}个)")
        
        # 为所有tensor分配advantages，保持原始形状
        if advantages.dim() == 2:
            # 如果原始advantages是2D的，保持2D形状
            all_samples["advantages"] = advantages[start_idx:end_idx].to(accelerator.device)
        else:
            # 如果原始advantages是1D的，保持1D形状
            all_samples["advantages"] = sample_advantages[start_idx:end_idx].to(accelerator.device)
        
        # 同时更新所有其他tensor到相同的样本范围
        for key, value in all_samples.items():
            if key != "advantages" and isinstance(value, torch.Tensor):
                all_samples[key] = value[start_idx:end_idx]
            elif key != "advantages" and isinstance(value, dict):
                all_samples[key] = {
                    sub_key: sub_value[start_idx:end_idx] 
                    for sub_key, sub_value in value.items()
                }
        
        # Filter out zero-advantage samples - 现在在正确的维度上进行筛选
        if all_samples["advantages"].dim() == 2:
            # 如果advantages是2D的，使用平均值来筛选
            mask = (all_samples["advantages"].mean(dim=1).abs() > 1e-6)
        else:
            # 如果advantages是1D的，直接筛选
            mask = (all_samples["advantages"].abs() > 1e-6)
        
        # 🔧 修复：确保mask在正确的设备上
        mask = mask.to(accelerator.device)
        
        print(f"🔍 样本筛选:")
        print(f"  mask.shape: {mask.shape}")
        print(f"  mask.device: {mask.device}")
        print(f"  筛选前样本数: {all_samples['advantages'].shape[0]}")
        print(f"  筛选后样本数: {mask.sum().item()}")
        
        # 应用mask到所有tensor
        filtered_samples = {}
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor):
                # 🔧 修复：确保tensor在正确的设备上
                if value.device != accelerator.device:
                    print(f"  ⚠️  {key} 设备不匹配: {value.device} -> {accelerator.device}")
                    value = value.to(accelerator.device)
                filtered_samples[key] = value[mask]
            elif isinstance(value, dict):
                filtered_samples[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        # 🔧 修复：确保tensor在正确的设备上
                        if sub_value.device != accelerator.device:
                            print(f"  ⚠️  {key}.{sub_key} 设备不匹配: {sub_value.device} -> {accelerator.device}")
                            sub_value = sub_value.to(accelerator.device)
                        filtered_samples[key][sub_key] = sub_value[mask]
                    else:
                        filtered_samples[key][sub_key] = sub_value
            else:
                filtered_samples[key] = value
        
        all_samples = filtered_samples
        
        logger.info(f"Training on {mask.sum().item()} samples with non-zero advantages")
        
        # 🔍 修复后的tensor形状验证
        print(f"🔍 修复后的tensor形状验证:")
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}.shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"    {sub_key}.shape: {sub_value.shape}")
        print(f"  所有tensor的第一维应该相同！")
        
        # 在 all_samples 处理后，添加SD3式的数据重组
        if "latents" in all_samples:
            # 🔍 SD3式数据重组: 将latents分割为current和next状态
            # ⚠️ 重要：虽然latent shape不同，但分割方式相同
            # SD3: latents (batch, steps+1, 16, 32, 32) → current/next (batch, steps, 16, 32, 32)
            # Hunyuan3D: latents (batch, steps+1, 1024, 64) → current/next (batch, steps, 1024, 64)
            # 通用方式: latents[:, :-1] for current, latents[:, 1:] for next
            latents = all_samples["latents"]
            print(f"🔍 SD3式数据重组前: latents.shape = {latents.shape}")
            print(f"  Hunyuan3D: (batch, steps+1, 1024, 64)")
            print(f"  SD3对比:   (batch, steps+1, 16, 32, 32)")

            all_samples["latents"] = latents[:, :-1]  # 当前状态
            all_samples["next_latents"] = latents[:, 1:]  # 下一个状态

            print(f"🔍 SD3式数据重组后:")
            print(f"  latents.shape: {all_samples['latents'].shape} (current states)")
            print(f"  next_latents.shape: {all_samples['next_latents'].shape} (next states)")
            print(f"  两者都应为: (batch_size, num_steps, ...)")
        
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            model.train()  # 只需要设置核心扩散模型为训练模式
            
            # Shuffle samples
            batch_size = all_samples["timesteps"].shape[0]
            perm = torch.randperm(batch_size, device=accelerator.device)
            shuffled_samples = {}
            for k, v in all_samples.items():
                if isinstance(v, torch.Tensor):
                    shuffled_samples[k] = v[perm]
                elif isinstance(v, dict):
                    shuffled_samples[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, torch.Tensor):
                            shuffled_samples[k][sub_k] = sub_v[perm]
                        else:
                            shuffled_samples[k][sub_k] = sub_v
                else:
                    shuffled_samples[k] = v
            
            # Train step
            train_metrics = trainer.train_step(
                samples=shuffled_samples,
                pipeline=trainer.pipeline.core_pipeline,
                optimizer=optimizer,
                config=config,
                accelerator=accelerator,
            )
            
            # Log training metrics
            accelerator.log({
                **train_metrics,
                "epoch": epoch,
                "inner_epoch": inner_epoch,
            }, step=global_step)
            
            global_step += 1
            
            # Update EMA
            if ema:
                ema.step(trainable_params, global_step)
        
        # Evaluation
        if epoch % config.eval_freq == 0 and epoch > 0:
            evaluate_3d(
                trainer, test_dataloader, config, accelerator, 
                global_step, executor
            )
        
        # Save checkpoint
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_checkpoint(config.save_dir, trainer.pipeline, global_step, accelerator, ema, config)
        
        # Clear stat tracker
        if stat_tracker:
            stat_tracker.clear()
    
    logger.info("Training completed!")
    
    # Final save
    if accelerator.is_main_process:
        save_checkpoint(config.save_dir, trainer.pipeline, global_step, accelerator, ema, config)
    
    # Cleanup
    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
