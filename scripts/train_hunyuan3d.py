#!/usr/bin/env python3
"""
Hunyuan3D GRPO Training Script

Based on SD3 architecture with Hunyuan3D-specific modifications.
"""

import sys
import os
import time
import logging
from PIL import Image
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent import futures
import contextlib
import datetime
import tempfile
import random
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

import ml_collections
from absl import app, flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config")

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from flow_grpo.diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob
from flow_grpo.diffusers_patch.hunyuan3d_sde_with_logprob import hunyuan3d_sde_step_with_logprob
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerImageStatTracker

logger = get_logger(__name__)

class Image3DDataset(Dataset):
    def __init__(self, image_dir: str, prompts_file: Optional[str] = None, split: str = "train"):
        self.image_dir = Path(image_dir)
        self.prompts_file = prompts_file
        self.split = split
        
        if (self.image_dir / "images").exists():
            self.image_dir = self.image_dir / "images"
        
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
        return f"Generate a 3D model from this image: {image_path.stem}"

def compute_log_prob_3d(pipeline, sample: Dict[str, torch.Tensor], step_index: int, config: Any):
    """Compute log probability for 3D diffusion model - similar to SD3's compute_log_prob"""
    latents = sample["latents"][:, step_index]
    next_latents = sample["next_latents"][:, step_index]
    timestep = sample["timesteps"][:, step_index]
    
    positive_image_cond = sample["positive_image_cond"]
    
    timestep_normalized = torch.clamp(
        timestep.float() / pipeline.scheduler.config.num_train_timesteps, 
        min=1e-6, max=1.0 - 1e-6
    )
    
    contexts = {}
    for key, cond in positive_image_cond.items():
        if cond.shape[0] != latents.shape[0]:
            cond = cond.repeat_interleaved(latents.shape[0] // cond.shape[0], dim=0)
        contexts[key] = cond
    
    if torch.isnan(latents).any() or torch.isinf(latents).any():
        latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
    
    with torch.amp.autocast('cuda'):
        noise_pred = pipeline.model(latents, timestep_normalized, contexts)
    
    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if not hasattr(pipeline.scheduler, 'hunyuan3d_sde_step_with_logprob'):
        import types
        pipeline.scheduler.hunyuan3d_sde_step_with_logprob = types.MethodType(
            hunyuan3d_sde_step_with_logprob, pipeline.scheduler
        )
    
    prev_sample, log_prob, prev_sample_mean, std_dev = pipeline.scheduler.hunyuan3d_sde_step_with_logprob(
        model_output=noise_pred,
        timestep=timestep[0],
        sample=latents,
        prev_sample=next_latents,
    )
    
    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
        log_prob = torch.zeros_like(log_prob)
    
    if torch.isnan(prev_sample_mean).any() or torch.isinf(prev_sample_mean).any():
        prev_sample_mean = torch.nan_to_num(prev_sample_mean, nan=0.0, posinf=1.0, neginf=-1.0)
    
    std_dev = torch.clamp(std_dev, min=1e-6, max=100.0)

    return prev_sample, log_prob, prev_sample_mean, std_dev

def save_meshes_for_wandb(meshes, image_paths, rewards, epoch, tmpdir, device="cuda"):
    """保存mesh并生成预览图 - 无fallback版本"""
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_for_training
    
    mesh_files = []
    preview_files = []
    
    for idx, (mesh, img_path, reward) in enumerate(zip(meshes, image_paths, rewards)):
        # 保存mesh文件
        mesh_path = os.path.join(tmpdir, f"mesh_{idx}.obj")
        mesh.write(mesh_path)
        
        # 生成预览图
        preview_path = os.path.join(tmpdir, f"preview_{idx}.png")
        render_mesh_for_training(mesh_path, preview_path, device=device)
        
        mesh_files.append(mesh_path)
        preview_files.append(preview_path)
    
    return mesh_files, preview_files

def save_ckpt_hunyuan3d(model, ema, optimizer, epoch, global_step, save_dir, accelerator):
    """Save checkpoint in SD3 style"""
    checkpoint_dir = os.path.join(save_dir, f"checkpoints", f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)
    model_state = unwrapped_model.state_dict()
    
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    torch.save(model_state, model_path)
    
    if ema is not None:
        ema_state = ema.state_dict()
        ema_path = os.path.join(checkpoint_dir, "pytorch_model_ema.bin")
        torch.save(ema_state, ema_path)
    
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
    torch.save(optimizer.state_dict(), optimizer_path)
    
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch_version": torch.__version__,
    }
    metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Checkpoint saved to: {checkpoint_dir}")

def calculate_zero_std_ratio_images(image_names, gathered_rewards):
    """
    计算每个唯一图像对应奖励值的标准差为零的比例
    
    参数:
        image_names: 图像名称列表
        gathered_rewards: 包含奖励值的字典，须包含'ori_avg'键
        
    返回:
        zero_std_ratio: 标准差为零的比例
    """
    # 将图像名称列表转换为NumPy数组
    image_array = np.array(image_names)
    
    # 获取唯一图像名称及其分组信息
    unique_images, inverse_indices, counts = np.unique(
        image_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # 分组获取每个图像对应的奖励值
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # 计算每个分组的标准差
    image_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # 计算零标准差的比例
    zero_std_count = np.count_nonzero(image_std_devs == 0)
    zero_std_ratio = zero_std_count / len(image_std_devs)
    
    return zero_std_ratio

def main(argv):
    """Main training function - inline architecture similar to SD3"""
    del argv
    config = _CONFIG.value
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo-3d",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    
    logger.info(f"\n{config}")
    
    set_seed(config.seed, device_specific=True)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Load Hunyuan3D pipeline
    logger.info("Loading Hunyuan3D pipeline...")
    pipeline_wrapper = Hunyuan3DPipeline()
    pipeline = pipeline_wrapper.core_pipeline
    
    # Enable FlashVDM if configured
    flashvdm_config = getattr(config, 'flashvdm', None)
    if flashvdm_config and flashvdm_config.enabled:
        pipeline.enable_flashvdm(
            enabled=flashvdm_config.enabled,
            adaptive_kv_selection=flashvdm_config.adaptive_kv_selection,
            topk_mode=flashvdm_config.topk_mode,
            mc_algo=flashvdm_config.mc_algo,
            replace_vae=flashvdm_config.replace_vae
        )
    
    # Freeze parameters
    pipeline.vae.requires_grad_(False)
    pipeline.conditioner.requires_grad_(False)
    pipeline.model.requires_grad_(not config.use_lora)
    
    # Set precision
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # Move to devices
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.conditioner.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.model.to(accelerator.device)
    else:
        pipeline.model.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.vae.eval()
    pipeline.conditioner.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.conditioner.requires_grad_(False)
    
    # LoRA setup
    if config.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "to_q", "to_k", "to_v", "out_proj",
                "fc1", "fc2",
                "final_layer.linear",
            ],
            lora_dropout=0.1,
            bias="none",
        )
        
        pipeline.model = get_peft_model(pipeline.model, lora_config)
    
    model = pipeline.model
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    model, optimizer = accelerator.prepare(model, optimizer)
    pipeline.model = model
    
    # EMA
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_params,
            decay=config.train.ema_decay,
            device=accelerator.device
        )
    
    # Enable TF32
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Reward function
    reward_config = {"geometric_quality": 1.0, "uni3d": 0.0}
    reward_fn = multi_mesh_score(accelerator.device, reward_config)
    
    # Dataset
    logger.info(f"Loading dataset from {config.data_dir}")
    train_dataset = Image3DDataset(config.data_dir, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.input_batch_size,
        shuffle=True,
        collate_fn=Image3DDataset.collate_fn,
        num_workers=0,
    )
    
    # Stat tracker
    stat_tracker = None
    if config.per_image_stat_tracking:
        stat_tracker = PerImageStatTracker(
            global_std=getattr(config.sample, 'global_std', False)
        )
    
    train_dataloader = accelerator.prepare(train_dataloader)
    
    executor = futures.ThreadPoolExecutor(max_workers=8)
    
    # Training loop
    global_step = 0
    first_epoch = 0
    
    train_iter = iter(train_dataloader)
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    for epoch in range(first_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        #################### SAMPLING ####################
        model.eval()
        samples = []
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            try:
                image_paths, prompts, metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                image_paths, prompts, metadata = next(train_iter)
            
            pil_images = [Image.open(path).convert('RGBA') for path in image_paths]
            
            # Encode image conditions
            cond_inputs = pipeline.prepare_image(pil_images)
            image_cond = pipeline.encode_cond(
                image=cond_inputs.pop('image'),
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=True,
                dual_guidance=False,
            )
            
            positive_image_cond = {}
            negative_image_cond = {}
            for key in image_cond.keys():
                batch_size = image_cond[key].shape[0]
                positive_image_cond[key] = image_cond[key][:batch_size//2].repeat_interleave(config.sample.num_meshes_per_image, dim=0)
                negative_image_cond[key] = image_cond[key][batch_size//2:].repeat_interleave(config.sample.num_meshes_per_image, dim=0)

            # Generate meshes
            with torch.no_grad():
                meshes, all_latents, all_log_probs, all_kl = hunyuan3d_pipeline_with_logprob(
                    pipeline,
                    positive_image_cond=positive_image_cond,
                    negative_image_cond=negative_image_cond,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    kl_reward=config.sample.kl_reward,
                    octree_resolution=384,
                    mc_level=0.0,
                    mc_algo=None,
                    box_v=1.01,
                    num_chunks=50000,
                )
            
            latents = torch.stack(all_latents, dim=1)
            log_probs = torch.stack(all_log_probs, dim=1)
            kls = torch.stack(all_kl, dim=1)
            kl = kls.detach()
            
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.train.batch_size, 1
            )
            # Fix: timesteps should match current_latents/next_latents (20 steps), not latents (21 steps) 
            timesteps = timesteps[:, :-1]  # Remove last timestep to match SD3 behavior
            
            # Compute rewards asynchronously
            rewards = executor.submit(reward_fn, meshes, None, {}, image_paths)
            time.sleep(0)
            
            current_latents = latents[:, :-1]
            next_latents = latents[:, 1:]



            samples.append({
                "image_paths": image_paths,
                "positive_image_cond": positive_image_cond,
                "timesteps": timesteps,
                "latents": current_latents,
                "next_latents": next_latents,
                "log_probs": log_probs,
                "kl": kl,
                "rewards": rewards,
            })
            
        # Wait for rewards
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
        
        # Collate samples
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict) and k != "image_paths"
            else (
                {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                    for sub_key in samples[0][k]
                }
                if isinstance(samples[0][k], dict)
                else [path for s in samples for path in s[k]]
            )
            for k in samples[0].keys()
        }

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) - config.sample.kl_reward*samples["kl"]
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # 保存mesh (每10个epoch)
        if epoch % 10 == 0 and accelerator.is_main_process:
            # 创建本地保存目录 (仿照SD3的logdir模式)
            mesh_save_dir = os.path.join(config.logdir, config.run_name, "generated_meshes", f"epoch_{epoch}")
            os.makedirs(mesh_save_dir, exist_ok=True)
            
            # 选择前2个mesh（对应第一张图片的2个生成结果）
            num_samples = min(2, len(meshes))
            
            sampled_meshes = meshes[:num_samples]
            sampled_paths = samples["image_paths"][:num_samples]
            sampled_rewards = gathered_rewards['avg'][:num_samples]
            
            # 本地保存和渲染
            mesh_files, preview_files = save_meshes_for_wandb(
                sampled_meshes, sampled_paths, sampled_rewards, epoch, mesh_save_dir, accelerator.device
            )
            
            # 上传到wandb
            accelerator.log({
                "generated_meshes": [wandb.Object3D(f) for f in mesh_files],
                "mesh_previews": [
                    wandb.Image(preview_files[i], caption=f"{os.path.basename(sampled_paths[i])}")
                    for i in range(len(preview_files))
                ],
            }, step=global_step)
        # log rewards
        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                "kl": samples["kl"].mean().cpu().numpy(),
                "kl_abs": samples["kl"].abs().mean().cpu().numpy()
            },
            step=global_step,
        )

        # per-image mean/std tracking
        if config.per_image_stat_tracking:
            # gather the image paths across processes (类似SD3中gather prompts)
            # 扩展image_names以匹配gathered_rewards的长度
            total_gathered_samples = len(gathered_rewards["avg"])
            image_names = []
            for i in range(total_gathered_samples):
                path_idx = i % len(samples["image_paths"])
                image_names.append(os.path.basename(samples["image_paths"][path_idx]))
            
            advantages = stat_tracker.update(image_names, gathered_rewards['avg'].mean(axis=1))
            if accelerator.is_local_main_process:
                print("len(image_names)", len(image_names))
                print("len unique image_names", len(set(image_names)))

            group_size, trained_image_num = stat_tracker.get_stats()

            zero_std_ratio = calculate_zero_std_ratio_images(image_names, gathered_rewards)

            accelerator.log(
                {
                    "group_size": group_size,
                    "trained_image_num": trained_image_num,
                    "zero_std_ratio": zero_std_ratio,
                },
                step=global_step,
            )
            # stat_tracker.clear()  # 保持历史累积，不清除统计
        else:
            advantages = (gathered_rewards['avg'].mean(axis=1) - gathered_rewards['avg'].mean(axis=1).mean()) / (gathered_rewards['avg'].mean(axis=1).std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        num_steps = samples["timesteps"].shape[1]
        advantages = advantages.unsqueeze(1).expand(-1, num_steps)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())

        del samples["rewards"]
        del samples["image_paths"]

        # Get the mask for samples where all advantages are zero across the time dimension (SD3 style)
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        accelerator.log(
            {
                "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
            },
            step=global_step,
        )
        # Filter out samples where the entire time dimension of advantages is zero
        # (SD3 logic with Hunyuan3D data structure adaptation)
        filtered_samples = {}
        for k, v in samples.items():
            if k == "positive_image_cond":
                # Handle nested dict specially
                filtered_samples[k] = {sub_k: sub_v[mask] for sub_k, sub_v in v.items()}
            elif isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0]:
                # Apply mask to tensors with matching batch dimension
                filtered_samples[k] = v[mask]
            else:
                # Keep unchanged for dimension mismatches (Hunyuan3D specific)
                filtered_samples[k] = v
        samples = filtered_samples

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps  # Now timesteps matches latents/log_probs (20 steps)
        
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            # Handle positive_image_cond separately due to nested dict structure
            shuffled_samples = {}
            for k, v in samples.items():
                if k == "positive_image_cond":
                    shuffled_samples[k] = {sub_k: sub_v[perm] for sub_k, sub_v in v.items()}
                else:
                    shuffled_samples[k] = v[perm]
            samples = shuffled_samples

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    # torch.randperm(num_timesteps, device=accelerator.device)
                    torch.arange(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {}
            for k, v in samples.items():
                if k == "positive_image_cond":
                    samples_batched[k] = {
                        sub_k: sub_v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *sub_v.shape[1:])
                        for sub_k, sub_v in v.items()
                    }
                else:
                    samples_batched[k] = v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration (SD3 style adaptation)
            batch_size = samples_batched["timesteps"].shape[0]
            samples_list = []
            for i in range(batch_size):
                sample_dict = {}
                for k, v in samples_batched.items():
                    if k == "positive_image_cond":
                        sample_dict[k] = {sub_k: sub_v[i] for sub_k, sub_v in v.items()}
                    else:
                        sample_dict[k] = v[i]
                samples_list.append(sample_dict)
            samples_batched = samples_list
            
            # train
            model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):

                
                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(model):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev = compute_log_prob_3d(
                                pipeline, sample, j, config
                            )
                            
                            if getattr(config.train, 'beta', 0) > 0:
                                with torch.no_grad():
                                    model_for_adapter = model.module if hasattr(model, 'module') else model
                                    with model_for_adapter.disable_adapter():
                                        _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = compute_log_prob_3d(
                                            pipeline, sample, j, config
                                        )

                        # grpo logic
                        
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if getattr(config.train, 'beta', 0) > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim))) / (2 * std_dev ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if getattr(config.train, 'beta', 0) > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(model.parameters(), global_step)
            # make sure we did an optimization step at the end of the inner epoch
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % getattr(config, 'save_freq', 10) == 0:
            save_ckpt_hunyuan3d(
                model, 
                ema,
                optimizer, 
                epoch, 
                global_step, 
                config.save_dir,
                accelerator
            )

if __name__ == "__main__":
    app.run(main) 