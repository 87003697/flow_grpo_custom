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
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return int(result.stdout.strip().split('\n')[0])
        except:
            return -1
    
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
            # 🎯 混合方案：使用包装器初始化（获得补丁和路径设置）
            wrapper = Hunyuan3DPipeline()
            
            # 🔧 提取内部管道（避免嵌套调用）
            pipeline = wrapper.pipeline  # 这是经过补丁的 Hunyuan3DDiTFlowMatchingPipeline
            
            # 🚀 启用 FlashVDM 加速优化
            logger.info("启用 FlashVDM 加速优化...")
            pipeline.enable_flashvdm(
                enabled=True,
                adaptive_kv_selection=True,
                topk_mode='mean',
                mc_algo='mc',  # 使用标准 marching cubes（无需额外依赖）
                replace_vae=True  # 使用 turbo VAE
            )
            logger.info("✅ FlashVDM 优化已启用")
            
            pipeline.to(accelerator.device)
        
        # Initialize reward models
        logger.info("Setting up reward configuration...")
        with gpu_timer("奖励函数初始化"):
            reward_config = {
                "geometric_quality": 0.3,
                "uni3d": 0.7
            }
            
            # Create trainer
            trainer = Hunyuan3DGRPOTrainer(
                pipeline=pipeline,
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
    
    # Setup model for training
    model = trainer.pipeline.model  # 核心扩散模型
    vae = trainer.pipeline.vae      # VAE编码器
    conditioner = trainer.pipeline.conditioner  # 条件编码器
    
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
                        deterministic=False,
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
        
        # Adjust rewards with KL penalty
        all_samples["rewards"]["ori_avg"] = all_samples["rewards"]["avg"].clone()
        all_samples["rewards"]["avg"] = (
            all_samples["rewards"]["avg"].unsqueeze(-1) - 
            config.sample.kl_reward * all_samples["kl"]
        )
        
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
            # Per-prompt stat tracking
            all_prompts = []
            for sample in epoch_samples:
                all_prompts.extend(sample["prompts"])
            
            advantages = stat_tracker.update(all_prompts, gathered_rewards['avg'])
            advantages = torch.as_tensor(advantages, device=accelerator.device)
        else:
            # Global advantages
            advantages = gathered_rewards['avg']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            advantages = torch.as_tensor(advantages, device=accelerator.device)
        
        # Reshape advantages to match samples
        all_samples["advantages"] = advantages.reshape(
            accelerator.num_processes, -1, 1
        )[accelerator.process_index].to(accelerator.device)
        
        # Filter out zero-advantage samples
        mask = (all_samples["advantages"].abs().sum(dim=1) != 0)
        all_samples = {k: v[mask] for k, v in all_samples.items()}
        
        logger.info(f"Training on {mask.sum().item()} samples with non-zero advantages")
        
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            model.train()  # 只需要设置核心扩散模型为训练模式
            
            # Shuffle samples
            batch_size = all_samples["timesteps"].shape[0]
            perm = torch.randperm(batch_size, device=accelerator.device)
            shuffled_samples = {k: v[perm] for k, v in all_samples.items()}
            
            # Train step
            train_metrics = trainer.train_step(
                samples=shuffled_samples,
                pipeline=trainer.pipeline.pipeline,
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
