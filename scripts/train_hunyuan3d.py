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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# HuggingFace imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Project imports
from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.mesh_basic_scorer import MeshBasicScorer
from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
from flow_grpo.trainer_3d import Hunyuan3DGRPOTrainer, create_3d_reward_function
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ema import EMAModuleWrapper


logger = get_logger(__name__)


class Image3DDataset(Dataset):
    """Dataset for image-to-3D generation tasks."""
    
    def __init__(self, image_dir: str, prompts_file: Optional[str] = None, split: str = "train"):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing input images
            prompts_file: Optional file containing text prompts (one per line)
            split: Dataset split ("train" or "test")
        """
        self.image_dir = Path(image_dir)
        self.split = split
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_paths = [
            p for p in self.image_dir.rglob("*") 
            if p.suffix.lower() in image_extensions
        ]
        
        # Load prompts if provided
        if prompts_file and os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            # Generate default prompts
            self.prompts = [f"A 3D object from image {i}" for i in range(len(self.image_paths))]
        
        # Ensure prompt count matches image count
        if len(self.prompts) != len(self.image_paths):
            # Repeat or truncate prompts to match images
            if len(self.prompts) < len(self.image_paths):
                self.prompts = (self.prompts * ((len(self.image_paths) // len(self.prompts)) + 1))[:len(self.image_paths)]
            else:
                self.prompts = self.prompts[:len(self.image_paths)]
        
        logger.info(f"Loaded {len(self.image_paths)} images and {len(self.prompts)} prompts for {split}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        prompt = self.prompts[idx]
        
        return {
            "image_path": image_path,
            "prompt": prompt,
            "metadata": {
                "image_name": self.image_paths[idx].name,
                "index": idx,
            }
        }
    
    @staticmethod
    def collate_fn(examples):
        """Collate function for DataLoader."""
        image_paths = [ex["image_path"] for ex in examples]
        prompts = [ex["prompt"] for ex in examples]
        metadata = [ex["metadata"] for ex in examples]
        
        return image_paths, prompts, metadata


def create_config():
    """Create default configuration for 3D training."""
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    
    # General training config
    config.num_epochs = 100
    config.save_freq = 10
    config.eval_freq = 5
    config.device = "cuda"
    config.mixed_precision = "fp16"
    config.gradient_checkpointing = False
    config.use_lora = True
    config.per_prompt_stat_tracking = True
    
    # Sample config
    config.sample = SimpleNamespace()
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 4
    config.sample.num_batches_per_epoch = 8
    config.sample.num_steps = 20  # Number of denoising steps
    config.sample.guidance_scale = 5.0
    config.sample.global_std = True
    config.sample.kl_reward = 0.1
    
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
    
    # Paths
    config.data_dir = "data/3d_training"
    config.save_dir = "checkpoints/hunyuan3d_grpo"
    config.resume_from = None
    
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
    reward_fn,
    executor: futures.ThreadPoolExecutor,
):
    """Evaluate 3D generation quality."""
    logger.info("Starting 3D evaluation...")
    
    trainer.pipeline.pipeline.eval()
    
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
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Initialize pipeline and models
    logger.info("Loading Hunyuan3D pipeline...")
    pipeline = Hunyuan3DPipeline()
    
    # Initialize reward models
    logger.info("Loading reward models...")
    basic_scorer = MeshBasicScorer()
    uni3d_scorer = Uni3DScorer()
    
    # Create trainer
    trainer = Hunyuan3DGRPOTrainer(
        pipeline=pipeline,
        basic_scorer=basic_scorer,
        uni3d_scorer=uni3d_scorer,
        device=accelerator.device,
    )
    
    # Create reward function
    reward_fn = create_3d_reward_function(basic_scorer, uni3d_scorer, accelerator.device)
    
    # Setup datasets
    logger.info(f"Loading datasets from {config.data_dir}")
    train_dataset = Image3DDataset(
        image_dir=os.path.join(config.data_dir, "train"),
        prompts_file=os.path.join(config.data_dir, "train_prompts.txt"),
        split="train"
    )
    test_dataset = Image3DDataset(
        image_dir=os.path.join(config.data_dir, "test"), 
        prompts_file=os.path.join(config.data_dir, "test_prompts.txt"),
        split="test"
    )
    
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
    model = trainer.pipeline.pipeline.model
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
        trainer.pipeline.pipeline.eval()
        
        epoch_samples = []
        for batch_idx, (image_paths, prompts, metadata) in enumerate(tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process
        )):
            if batch_idx >= config.sample.num_batches_per_epoch:
                break
            
            # Sample meshes with rewards
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
            trainer.pipeline.pipeline.train()
            
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
                global_step, reward_fn, executor
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
