#!/usr/bin/env python3
"""
Hunyuan3D GRPO Training Script

Based on SD3 architecture with Hunyuan3D-specific modifications.
"""

import os
import sys
import time
import logging
from PIL import Image
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
# from concurrent import futures  # 🔧 已移除：不再使用线程池
import contextlib
import datetime
import tempfile
import random
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
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
from reward_models.rewards_mesh import MeshScorer
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
        from PIL import Image
        image_path = str(self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # 直接返回PIL对象
        prompt = self.get_prompt(self.image_files[idx])
        
        return {
            "image": image,
            "image_path": image_path,
            "prompt": prompt,
            "metadata": {"image_name": self.image_files[idx].name}
        }
    
    @staticmethod
    def collate_fn(examples):
        images = [example["image"] for example in examples]  # PIL对象列表
        image_paths = [example["image_path"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        return images, image_paths, prompts, metadata
    
    def get_prompt(self, image_path: Path) -> str:
        return f"Generate a 3D model from this image: {image_path.stem}"

class DistributedImageRepeatSampler(Sampler):
    """
    Hunyuan3D专用的分布式重复采样器
    确保每张图像在所有GPU上生成多个mesh，实现真正的group比较
    类似SD3的DistributedKRepeatSampler但适配图像输入
    """
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每张图像重复的次数(num_meshes_per_image)
        self.num_replicas = num_replicas  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同图像数
        self.total_samples = self.num_replicas * self.batch_size
        
        # 🔧 修复：处理total_samples < k的情况（单GPU小batch场景）
        if self.total_samples < self.k:
            logger.warning(f"total_samples({self.total_samples}) < k({self.k}), 调整为简单重复模式")
            self.m = self.total_samples  # 使用所有可用样本
            self.simple_repeat_mode = True
        else:
            assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
            self.m = self.total_samples // self.k  # 不同图像数
            self.simple_repeat_mode = False
        
        self.epoch = 0

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            if self.simple_repeat_mode:
                # 🔧 简单重复模式：当total_samples < k时
                # 随机选择图像并重复填满batch
                available_indices = torch.randperm(len(self.dataset), generator=g).tolist()
                
                # 创建足够的样本来填满所有GPU的batch
                repeated_indices = []
                for i in range(self.total_samples):
                    repeated_indices.append(available_indices[i % len(available_indices)])
                
                # 将样本分配到各个卡
                per_card_samples = []
                for i in range(self.num_replicas):
                    start = i * self.batch_size
                    end = start + self.batch_size
                    per_card_samples.append(repeated_indices[start:end])
                
                yield per_card_samples[self.rank]
            else:
                # 🔧 标准重复模式：当total_samples >= k时
                # 随机选择m个不同的图像
                indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
                
                # 每张图像重复k次，生成总样本数n*b
                repeated_indices = [idx for idx in indices for _ in range(self.k)]
                
                # 打乱顺序确保均匀分配
                shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
                shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
                
                # 将样本分割到各个卡
                per_card_samples = []
                for i in range(self.num_replicas):
                    start = i * self.batch_size
                    end = start + self.batch_size
                    per_card_samples.append(shuffled_samples[start:end])
                
                # 返回当前卡的样本索引
                yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态

def compute_log_prob_3d(pipeline, sample: Dict[str, Any], j: int, image_conds: Dict[str, torch.Tensor], config: Any):
    """
    计算3D扩散模型的对数概率 - 结构与SD3的compute_log_prob完全对齐
    """
    # 调整1：在函数内部从sample和j中提取数据
    latents = sample["latents"][:, j]
    next_latents = sample["next_latents"][:, j]
    timestep = sample["timesteps"][:, j]
    
    # 步骤1: 模型前向传播，根据CFG配置准备输入并预测噪声
    if config.train.cfg:
        # CFG路径: 准备拼接后的输入
        model_latents = torch.cat([latents] * 2)
        model_timestep = torch.cat([timestep] * 2)
        
        # Hunyuan特有预处理
        timestep_normalized = model_timestep.float() / 1000.0
        # 🔧 CFG修复：image_conds已经是[negative, positive]格式，直接使用
        contexts = image_conds
        if torch.isnan(model_latents).any(): model_latents = torch.nan_to_num(model_latents)
            
        # 模型预测
        with torch.amp.autocast('cuda'):
            noise_pred_combined = pipeline.model(model_latents, timestep_normalized, contexts)
            
        # 应用CFG
        noise_pred_neg, noise_pred_pos = noise_pred_combined.chunk(2)
        noise_pred = noise_pred_neg + config.sample.guidance_scale * (noise_pred_pos - noise_pred_neg)
        
    else:
        # 非CFG路径: 使用原始输入
        model_latents = latents
        model_timestep = timestep
        
        # Hunyuan特有预处理
        timestep_normalized = model_timestep.float() / 1000.0
        contexts = image_conds
        if torch.isnan(model_latents).any(): model_latents = torch.nan_to_num(model_latents)
    
        # 模型预测
        with torch.amp.autocast('cuda'):
            noise_pred = pipeline.model(model_latents, timestep_normalized, contexts)

    # 步骤2: SDE步骤计算log_prob (与SD3的流程一致)
    prev_sample, log_prob, prev_sample_mean, std_dev = pipeline.scheduler.hunyuan3d_sde_step_with_logprob(
        model_output=noise_pred,
        timestep=timestep[0],
        sample=latents,
        prev_sample=next_latents,
    )
    
    return prev_sample, log_prob, prev_sample_mean, std_dev

def save_meshes_for_wandb(meshes, image_paths, rewards, epoch, tmpdir, device="cuda"):
    """保存mesh并生成预览图 - 文件名基于原始图像名称"""
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_for_training
    import os
    
    mesh_files = []
    preview_files = []
    
    for idx, (mesh, img_path, reward) in enumerate(zip(meshes, image_paths, rewards)):
        # 🔧 新增：从图像路径提取文件名（不含扩展名）
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        # 清理文件名中的特殊字符，避免文件系统问题
        image_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # 🔧 修改：使用图像名称命名mesh文件
        mesh_path = os.path.join(tmpdir, f"{image_name}_mesh_{idx}.obj")
        mesh.write(mesh_path)

        # 🔧 修改：使用图像名称命名预览图
        preview_path = os.path.join(tmpdir, f"{image_name}_preview_{idx}.png")
        render_mesh_for_training(mesh_path, preview_path, device=device)
        print(f"💾 渲染已保存: {preview_path}")
        
        mesh_files.append(mesh_path)
        preview_files.append(preview_path)
    
    return mesh_files, preview_files

def save_ckpt_hunyuan3d(model, ema, optimizer, epoch, global_step, save_dir, accelerator):
    import os
    """Save checkpoint in SD3 style - LoRA compatible"""
    
    # 🔧 修复：使用全局的_CONFIG变量，而不是从模块导入
    config = _CONFIG.value

    checkpoint_dir = os.path.join(config.logdir, config.run_name, f"checkpoints", f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if config.use_lora:
        # LoRA模式：只保存适配器权重
        lora_save_dir = os.path.join(checkpoint_dir, "lora")
        os.makedirs(lora_save_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'save_pretrained'):
            unwrapped_model.save_pretrained(lora_save_dir)
            logger.info(f"✅ LoRA适配器已保存到: {lora_save_dir}")
        else:
            logger.warning("⚠️ 模型没有save_pretrained方法，fallback到state_dict")
            model_state = unwrapped_model.state_dict()
            model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(model_state, model_path)
    else:
        # 全模型训练：保存完整权重
        unwrapped_model = accelerator.unwrap_model(model)
        model_state = unwrapped_model.state_dict()
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        torch.save(model_state, model_path)
    
    # 保存优化器状态
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # 保存EMA状态 (如果存在)
    if ema is not None:
        ema_state = ema.state_dict()
        ema_path = os.path.join(checkpoint_dir, "pytorch_model_ema.bin")
        torch.save(ema_state, ema_path)
    
    # 保存训练元数据
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch_version": torch.__version__,
        "use_lora": config.use_lora,
    }
    metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Checkpoint已保存到: {checkpoint_dir}")

def eval_hunyuan3d(pipeline, test_dataloader, config, accelerator, global_step, mesh_scorer, ema=None, model=None):
    """
    Hunyuan3D评估函数 - 仿照SD3的eval函数结构
    """
    if config.train.ema and ema is not None:
        # 使用EMA权重进行评估
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        ema.copy_ema_to(trainable_params, store_temp=True)
    
    model.eval()
    all_rewards = defaultdict(list)
    
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        images, image_paths, prompts, metadata = test_batch
        
        # 编码图像条件（与训练时相同的流程）
        cond_inputs = pipeline.prepare_image(images)
        image_cond = pipeline.encode_cond(
            image=cond_inputs.pop('image'),
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=True,
            dual_guidance=False,
        )

        # 准备条件张量（仿照SD3评估模式：每个输入只生成1个输出）
        positive_image_cond = {}
        negative_image_cond = {}
        for key in image_cond.keys():
            batch_size = image_cond[key].shape[0]
            # 🔧 修复：评估时每张图像只生成1个mesh（仿照SD3：每个prompt只生成1张图像）
            positive_image_cond[key] = image_cond[key][:batch_size//2]  # 不使用repeat_interleave
            negative_image_cond[key] = image_cond[key][batch_size//2:]  # 不使用repeat_interleave

        # 生成mesh（使用与训练时完全相同的参数和返回值处理）
        with torch.no_grad():
            model.eval()  # 恢复eval模式
            meshes, all_latents, all_log_probs, all_kl = hunyuan3d_pipeline_with_logprob(
                pipeline,
                positive_image_cond=positive_image_cond,
                negative_image_cond=negative_image_cond,
                num_inference_steps=config.sample.eval_num_steps,  # 评估用的步数
                guidance_scale=config.sample.guidance_scale,
                kl_reward=0.0,  # 评估时不使用KL奖励
                octree_resolution=384,
                mc_level=0.0,
                mc_algo=None,
                box_v=1.01,
                num_chunks=50000,
                determistic=False,
                # 🔧 移除determistic参数，与训练时保持一致
            )
        
        # 计算奖励
        rewards, reward_metadata = mesh_scorer.score(meshes, images, metadata, config.reward_fn.to_dict())
        
        # 收集奖励
        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    # 聚合所有奖励
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    
    # 保存评估样本（类似SD3）
    if accelerator.is_main_process:
        # 🔧 修复：使用统一的目录结构，而不是临时目录
        eval_save_dir = os.path.join(config.logdir, config.run_name, "eval_meshes", f"step_{global_step}")
        os.makedirs(eval_save_dir, exist_ok=True)
        
        num_samples = min(15, len(meshes))
        sample_indices = range(num_samples)
        
        # 保存mesh预览到统一目录
        mesh_files, preview_files = save_meshes_for_wandb(
            meshes[:num_samples], 
            image_paths[:num_samples], 
            all_rewards['avg'][:num_samples], 
            "eval", 
            eval_save_dir,  # 🔧 使用统一目录而不是临时目录
            "cuda"
        )
        
        # 记录到wandb
        accelerator.log({
            "eval_mesh_previews": [
                wandb.Image(
                    preview_files[i],
                    caption=f"{os.path.basename(image_paths[i])} | avg: {all_rewards['avg'][i]:.2f}",
                )
                for i in range(len(preview_files))
            ],
            **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
        }, step=global_step)
        
        logger.info(f"✅ 已保存 {len(mesh_files)} 个评估mesh到 {eval_save_dir}")
    
    # 恢复训练权重
    if config.train.ema and ema is not None:
        ema.copy_temp_to(trainable_params)

def calculate_zero_std_ratio_images(image_names, gathered_rewards):
    """
    Calculates the ratio of image groups with zero standard deviation in their rewards.
    This function is now simplified to be more robust.
    """
    if 'avg' not in gathered_rewards or gathered_rewards['avg'].size == 0:
        return 0.0

    rewards_flat = gathered_rewards['avg'].flatten()
    image_names_flat = np.array(image_names)

    if rewards_flat.shape[0] != image_names_flat.shape[0]:
        # Fallback if shapes mismatch, though this shouldn't happen with prior fixes.
        return 0.0

    unique_names = np.unique(image_names_flat)
    if len(unique_names) == 0:
        return 0.0

    std_devs = [np.std(rewards_flat[image_names_flat == name]) for name in unique_names]
    zero_std_count = np.count_nonzero(np.array(std_devs) == 0)
    
    return zero_std_count / len(unique_names)


def main(argv):
    import os
    """Main training function - inline architecture similar to SD3"""
    del argv
    config = _CONFIG.value
    
    # 🔧 多GPU修复：使用环境变量同步时间戳
    if "HUNYUAN3D_TIMESTAMP" not in os.environ:
        # 第一个启动的进程设置时间戳
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        os.environ["HUNYUAN3D_TIMESTAMP"] = unique_id
    else:
        # 后续进程读取已设置的时间戳
        unique_id = os.environ["HUNYUAN3D_TIMESTAMP"]
    
    # 设置run_name
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    # 直接创建accelerator（无需临时accelerator）
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
    
    # 3. 移除“猴子补丁”：在pipeline初始化后，只执行一次SDE函数的动态绑定
    if not hasattr(pipeline.scheduler, 'hunyuan3d_sde_step_with_logprob'):
        import types
        pipeline.scheduler.hunyuan3d_sde_step_with_logprob = types.MethodType(
            hunyuan3d_sde_step_with_logprob, pipeline.scheduler
        )

    # Enable FlashVDM if configured
    if config.flashvdm and config.flashvdm.enabled:
        pipeline.enable_flashvdm(
            enabled=config.flashvdm.enabled,
            adaptive_kv_selection=config.flashvdm.adaptive_kv_selection,
            topk_mode=config.flashvdm.topk_mode,
            mc_algo=config.flashvdm.mc_algo,
            replace_vae=config.flashvdm.replace_vae
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
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
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
        # 🔧 修复：如果配置中没有ema_decay，使用默认值0.999
        ema_decay = getattr(config.train, 'ema_decay', 0.999)
        ema = EMAModuleWrapper(
            trainable_params,
            decay=ema_decay,
            device=accelerator.device
        )
    
    # Enable TF32
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Reward function - 🔧 NEW: 使用MeshScorer类，一次性加载模型
    reward_config = config.reward_fn.to_dict()
    
    # 创建MeshScorer实例 - 只在初始化时加载一次模型
    mesh_scorer = MeshScorer(device=accelerator.device)
    
    # 创建适配器函数，使用scorer实例
    def reward_fn(meshes, images, metadata):
        """奖励函数适配器，使用MeshScorer实例"""
        return mesh_scorer.score(meshes, images, metadata, reward_config)
    
    # Dataset
    logger.info(f"Loading dataset from {config.data_dir}")
    train_dataset = Image3DDataset(config.data_dir, split="train")
    test_dataset = Image3DDataset(config.data_dir, split="test")  # 🔧 新增：测试数据集
    
    # 🔧 修复Group处理：使用分布式重复采样器（类似SD3）
    # Create DistributedImageRepeatSampler for proper group comparison
    train_sampler = DistributedImageRepeatSampler(
        train_dataset,
        config.sample.input_batch_size,
        config.sample.num_meshes_per_image,  # k: 每张图像重复次数
        accelerator.num_processes,
        accelerator.process_index,
        config.seed,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # 使用batch_sampler而不是sampler
        collate_fn=Image3DDataset.collate_fn,
        num_workers=0,
    )
    
    # 🔧 新增：创建测试数据加载器（仿照SD3模式）
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # 需要在配置中添加此参数
        collate_fn=Image3DDataset.collate_fn,
        shuffle=False,
        num_workers=4,
    )
    
    # Stat tracker
    stat_tracker = None
    if config.per_image_stat_tracking:
        stat_tracker = PerImageStatTracker(
            global_std=config.sample.global_std  # 🔧 修复：只传递global_std参数，与PerPromptStatTracker一致
        )
    
    train_dataloader = accelerator.prepare(train_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)  # 🔧 新增：准备测试数据加载器

    
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
            # 🔧 新增：定期评估（仿照SD3模式）
            if i==0 and epoch % config.eval_freq == 0 and epoch>0:
                eval_hunyuan3d(
                    pipeline, 
                    test_dataloader, 
                    config, 
                    accelerator, 
                    epoch,
                    mesh_scorer, 
                    ema, 
                    model
                )
            
            # 🔧 修复Group处理：设置epoch以同步所有GPU的随机状态
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            
            try:
                images, image_paths, prompts, metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                images, image_paths, prompts, metadata = next(train_iter)
            
            # 🔧 调试信息：打印当前batch的图像信息
            if accelerator.is_local_main_process:
                logger.info(f"Batch {i}: processing {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")
            
            # Encode image conditions
            cond_inputs = pipeline.prepare_image(images)
            image_cond = pipeline.encode_cond(
                image=cond_inputs.pop('image'),
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=True,
                dual_guidance=False,
            )
            
            # 🔧 修复Group处理：现在每张图像会在多个GPU上重复处理
            # 每个GPU对同一图像生成不同的mesh样本，实现真正的group比较
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
            if torch.isnan(kl).any():
                import pdb; pdb.set_trace()
            
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.train.batch_size, 1
            )
            # Fix: timesteps should match current_latents/next_latents (20 steps), not latents (21 steps) 
            timesteps = timesteps[:, :-1]  # Remove last timestep to match SD3 behavior


            # 🔧 调试：添加奖励计算前的日志
            if accelerator.is_local_main_process:
                logger.info(f"🎯 开始计算奖励: {len(meshes)} meshes, device={accelerator.device}")

            # 计算奖励（使用扩展后的路径）
            rewards, reward_metadata = reward_fn(meshes, images, {})
            if accelerator.is_local_main_process:
                logger.info(f"✅ 奖励计算完成: {rewards}")

                
            rewards_tensor = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            
            
            current_latents = latents[:, :-1]
            next_latents = latents[:, 1:]

            samples.append({
                "image_paths": image_paths,
                "positive_image_cond": positive_image_cond,
                "negative_image_cond": negative_image_cond,
                "timesteps": timesteps,
                "latents": current_latents,
                "next_latents": next_latents,
                "log_probs": log_probs,
                "kl": kl,
                "rewards": rewards_tensor,
            })


            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # 🔧 线程池移除：无需等待rewards，已经同步计算完成
        
        # Collate samples (Re-written for clarity and correctness)
        collated_samples = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                collated_samples[k].append(v)

        final_samples = {}
        for k, v_list in collated_samples.items():
            if k in ["positive_image_cond", "negative_image_cond", "rewards"]:
                # It's a list of dictionaries, need to merge them
                merged_dict = defaultdict(list)
                for d in v_list:
                    for sub_k, sub_v in d.items():
                        merged_dict[sub_k].append(sub_v)
                final_samples[k] = {sub_k: torch.cat(sub_v_list, dim=0) for sub_k, sub_v_list in merged_dict.items()}
            elif k == "image_paths":
                # Flatten the list of lists
                final_samples[k] = [path for sublist in v_list for path in sublist]
            else:
                # Regular tensors
                final_samples[k] = torch.cat(v_list, dim=0)
        samples = final_samples


        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) - config.sample.kl_reward*samples["kl"]
        # gather rewards across processes
        # 🔄 SD3 Debug: 分布式Gather - 收集所有GPU的奖励数据
        # samples["rewards"]["avg"].shape = (local_batch_size, 1) 每个GPU的本地奖励
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        # gathered_rewards["avg"].shape = (total_batch_size, 1) 所有GPU的奖励汇总
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # 保存mesh - 🔧 修改：使用save_freq而不是硬编码的10
        save_visualizations = getattr(config, 'save_visualizations', False)  # 默认禁用
        mesh_save_freq = getattr(config, 'mesh_save_freq', config.save_freq)  # 默认与checkpoint保存频率一致
        if epoch % mesh_save_freq == 0 and accelerator.is_main_process and save_visualizations:
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
                sampled_meshes, sampled_paths, sampled_rewards, epoch, mesh_save_dir, "cuda"
            )
            
            # 🔧 只上传预览图到wandb，不上传3D对象
            accelerator.log({
                "mesh_previews": [
                    wandb.Image(preview_files[i], caption=f"{os.path.basename(sampled_paths[i])}")
                    for i in range(len(preview_files))
                ],
            }, step=epoch)
            
            logger.info(f"✅ 已保存 {len(mesh_files)} 个mesh可视化到 {mesh_save_dir}")
        # log rewards
        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                "kl": samples["kl"].mean().cpu().numpy(),
                "kl_abs": samples["kl"].abs().mean().cpu().numpy()
            },
            step=epoch,
        )

        # per-image mean/std tracking
        if config.per_image_stat_tracking:
            gathered_rewards_for_tracker = gathered_rewards['avg'].flatten()
            image_names_for_tracker = []
            for i in range(len(gathered_rewards_for_tracker)):
                path_idx = i % len(samples["image_paths"])
                image_names_for_tracker.append(os.path.basename(samples["image_paths"][path_idx]))

            # 🔧 FIX: 将字符串图像名转换为数值ID，防止数值爆炸
            unique_names = list(set(image_names_for_tracker))
            name_to_id = {name: idx for idx, name in enumerate(unique_names)}
            image_ids_for_tracker = np.array([name_to_id[name] for name in image_names_for_tracker])
            
            advantages = stat_tracker.update(image_ids_for_tracker, gathered_rewards_for_tracker)
            if accelerator.is_local_main_process:
                print("len(image_names)", len(image_names_for_tracker))
                print("len unique image_names", len(set(image_names_for_tracker)))

            group_size, trained_image_num = stat_tracker.get_stats()
            zero_std_ratio = calculate_zero_std_ratio_images(image_names_for_tracker, gathered_rewards)

            accelerator.log(
                {
                    "group_size": group_size,
                    "trained_image_num": trained_image_num,
                    "zero_std_ratio": zero_std_ratio,
                },
                step=epoch,
            )
        else:
            logger.warning("使用全局标准化可能导致不同图像间的不合理比较，建议启用per_image_stat_tracking")
            advantages = (gathered_rewards['avg'].mean(axis=1) - gathered_rewards['avg'].mean(axis=1).mean()) / (gathered_rewards['avg'].mean(axis=1).std() + 1e-4)

        # ungather advantages
        advantages = torch.as_tensor(advantages)
        num_steps = samples["timesteps"].shape[1] # num_steps = config.sample.num_steps (如20)
        advantages = advantages.unsqueeze(1).expand(-1, num_steps) # advantages.shape = (total_batch_size, num_steps)
        # 🔄 SD3 Debug: 分布式Ungather - 每个GPU只保留自己对应的数据切片
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            # reshape: (total_batch_size, num_steps) -> (num_processes, local_batch_size, num_steps)
            # [process_index]: 选择当前GPU对应的切片 -> (local_batch_size, num_steps)
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            # 🔄 SD3 Debug: 打印本地GPU的优势函数和KL散度统计信息
            print("advantages: ", samples["advantages"].abs().mean()) # samples["advantages"].shape = (local_batch_size, num_steps)
            print("kl: ", samples["kl"].mean()) # samples["kl"].shape = (local_batch_size, num_steps)

        # 🔄 SD3 Debug: 内存优化 - 删除不再需要的大数据结构
        del samples["rewards"] # 已完成优势函数计算，奖励数据不再需要
        del samples["image_paths"] # 图像路径只用于统计跟踪，现在可以删除

        # 🔄 SD3 Debug: 数据过滤 - 筛选有效的训练样本
        # Get the mask for samples where all advantages are zero across the time dimension (SD3 style)
        mask = (samples["advantages"].abs().sum(dim=1) != 0) # mask.shape = (local_batch_size,)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = max(mask.sum(), 1) # 有效样本数量, 至少保留一个样本
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        accelerator.log(
            {
                "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                "valid_samples_ratio": mask.sum().item() / mask.shape[0],
            },
            step=epoch,
        )
        # 🔄 SD3 Debug: 应用mask过滤 - 移除advantages全为零的无效样本
        # Filter out samples where the entire time dimension of advantages is zero
        # (SD3 logic with Hunyuan3D data structure adaptation)
        filtered_samples = {}
        for k, v in samples.items():
            # 🔧 修复：对image conditions应用正确的mask过滤
            if k in ["positive_image_cond", "negative_image_cond"]:
                # image_cond的第一个维度是batch_size * num_meshes_per_image
                # 需要扩展mask来匹配这个维度
                filtered_image_cond = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor) and sub_v.shape[0] == mask.shape[0]:
                        filtered_image_cond[sub_k] = sub_v[mask]
                    else:
                        filtered_image_cond[sub_k] = sub_v
                filtered_samples[k] = filtered_image_cond
                continue  # 🔧 关键修复：跳过后续的通用处理逻辑

            if isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0]:
                # Apply mask to tensors with matching batch dimension
                # v.shape = (local_batch_size, ...) -> 过滤后 -> (filtered_batch_size, ...)
                filtered_samples[k] = v[mask]
            else:
                # Keep unchanged for dimension mismatches (Hunyuan3D specific)
                filtered_samples[k] = v
        samples = filtered_samples

        # 🔄 SD3 Debug: 验证过滤后的数据维度
        total_batch_size, num_timesteps = samples["timesteps"].shape
        # total_batch_size = filtered_batch_size, num_timesteps = config.sample.num_steps
        assert num_timesteps == config.sample.num_steps  # Now timesteps matches latents/log_probs (20 steps)
        
        #################### TRAINING ####################
        
        for inner_epoch in range(config.train.num_inner_epochs):
            
            # ╔═══════════════════════════════════════════════════════════════════════════════╗
            # ║                    🔄 数据重组阶段1 - 沿batch维度随机打乱                    ║
            # ╠═══════════════════════════════════════════════════════════════════════════════╣
            # ║ 目的：打破数据的原有顺序，增加训练随机性，避免模型学习到数据排列的偏见        ║
            # ║ 原理：对batch中的所有样本进行随机重排，但保持每个样本内部的时序关系不变      ║
            # ║ 实现：生成随机排列索引，所有tensor按相同顺序重排，保持样本间的对应关系       ║
            # ╚═══════════════════════════════════════════════════════════════════════════════╝
            
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device) # perm.shape = (total_batch_size,)
            # 🔍 perm示例: 如果total_batch_size=4，可能生成 [2, 0, 3, 1]
            # 表示: 新位置0取原位置2的样本，新位置1取原位置0的样本，以此类推
            
            # Handle dictionary and tensor shuffles
            for k, v in samples.items():
                if k in ["positive_image_cond", "negative_image_cond", "rewards"]:
                    samples[k] = {sub_k: sub_v[perm] for sub_k, sub_v in v.items()}
                else:
                    samples[k] = v[perm]


            # ╔═══════════════════════════════════════════════════════════════════════════════╗
            # ║                 🔄 数据重组阶段2 - 沿时间维度独立打乱每个样本                 ║
            # ╠═══════════════════════════════════════════════════════════════════════════════╣
            # ║ 目的：对每个样本的时间步进行独立重排，增加时序训练的多样性和鲁棒性           ║
            # ║ 原理：GRPO可以在任意时间步组合上训练，不需要严格按扩散过程的固定顺序       ║
            # ║ 实现：为每个样本生成独立的时间步排列，但当前为了调试稳定性使用固定顺序      ║
            # ║ 效果：破坏时间步间的相关性，让模型学习更泛化的策略                         ║
            # ╚═══════════════════════════════════════════════════════════════════════════════╝
            
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    # 🔧 可选随机化: torch.randperm(num_timesteps, device=accelerator.device)
                    # 🔧 当前固定顺序: 为了调试和复现性，暂时使用时间步的自然顺序
                    torch.arange(num_timesteps, device=accelerator.device) # 当前使用顺序，不随机
                    for _ in range(total_batch_size)
                ]
            ) # perms.shape = (total_batch_size, num_timesteps)
            
            # 🔍 perms数据结构示例: 如果total_batch_size=4, num_timesteps=20
            # perms = tensor([[0,1,2,...,19],    # 样本0的时间步排列: 按顺序
            #                 [0,1,2,...,19],    # 样本1的时间步排列: 按顺序  
            #                 [0,1,2,...,19],    # 样本2的时间步排列: 按顺序
            #                 [0,1,2,...,19]])   # 样本3的时间步排列: 按顺序
            # 🔄 如果启用随机化，每行将是[0-19]的不同随机排列，实现独立的时序打乱
            
            # 对所有包含时间维度的tensor进行重排
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                # 🔍 高级索引详解:
                # torch.arange(total_batch_size)[:, None] 创建列向量: [[0], [1], [2], [3]]
                # perms 是矩阵: [[perm0], [perm1], [perm2], [perm3]]
                # 组合索引 [batch_indices, time_indices] 实现: 
                #   - 对样本0，取 samples[key][0, perm0]
                #   - 对样本1，取 samples[key][1, perm1]  
                #   - 对样本2，取 samples[key][2, perm2]
                #   - 对样本3，取 samples[key][3, perm3]
                # 结果: 每个样本的时间维度按其专属排列重新排序
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],  # (total_batch_size, 1)
                    perms,  # (total_batch_size, num_timesteps)
                ]
                # 🔍 变换说明: samples[key].shape保持 (total_batch_size, num_timesteps, ...)
                # 但每个样本内部的时间步顺序可能完全改变（当前保持原序）

            # ╔═══════════════════════════════════════════════════════════════════════════════╗
            # ║                    🔄 数据重组阶段3 - Rebatch为训练子批次                     ║
            # ╠═══════════════════════════════════════════════════════════════════════════════╣
            # ║ 目的：将大batch重组为多个小batch，便于梯度累积和显存管理                   ║
            # ║ 原理：GRPO需要在多个子批次上分别计算梯度，最后累积更新参数                 ║
            # ║ 数学：total_batch_size -> (num_batches_per_epoch, batch_size_per_batch)      ║
            # ║ 好处：可以用小显存训练大batch_size，提高训练稳定性                         ║
            # ╚═══════════════════════════════════════════════════════════════════════════════╝
            
            # --- START: REWRITTEN DATA RESTRUCTURING ---

            # Step 1: Split all tensors and dicts into chunks for each sub-batch
            chunk_size = total_batch_size // config.sample.num_batches_per_epoch
            batched_tensors = {}
            for k, v in samples.items():
                if k == 'image_paths':
                    batched_tensors[k] = [v[i:i + chunk_size] for i in range(0, len(v), chunk_size)]
                elif isinstance(v, dict):
                    # Handle nested dictionaries
                    batched_tensors[k] = [{sub_k: sub_v.chunk(config.sample.num_batches_per_epoch, dim=0)[i] for sub_k, sub_v in v.items()} for i in range(config.sample.num_batches_per_epoch)]
                else:
                    # Handle regular tensors
                    batched_tensors[k] = list(v.chunk(config.sample.num_batches_per_epoch, dim=0))

            # Step 2: Convert to list of dicts for easier iteration
            num_sub_batches = config.sample.num_batches_per_epoch
            samples_batched = [{} for _ in range(num_sub_batches)]
            for k, v_chunks in batched_tensors.items():
                for i in range(num_sub_batches):
                    samples_batched[i][k] = v_chunks[i]

            # --- END: REWRITTEN DATA RESTRUCTURING ---


            # train
            model.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):

                # 调整4：将image_conds的准备工作移到外层循环，与SD3对齐
                if config.train.cfg:
                    # Concatenate conditions for CFG
                    image_conds = {
                        k: torch.cat([sample["negative_image_cond"][k], sample["positive_image_cond"][k]])
                        for k in sample["positive_image_cond"]
                    }
                else:
                    image_conds = sample["positive_image_cond"]

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
                            # 调整2：移除预先切片，直接传递sample和j
                            prev_sample, log_prob, prev_sample_mean, std_dev = compute_log_prob_3d(
                                pipeline, sample, j, image_conds, config
                            )
                            # log_prob.shape = torch.Size([1, 16, 32, 32])
                            # prev_sample.shape = torch.Size([1, 4096, 64])
                            # std_dev.shape = torch.Size([1, 1, 1])
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    model_for_adapter = model.module if hasattr(model, 'module') else model
                                    with model_for_adapter.disable_adapter():
                                        _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = compute_log_prob_3d(
                                            pipeline, sample, j, image_conds, config
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
                        if config.train.beta > 0:
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
                        if config.train.beta > 0:
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
        
        # 🔧 NEW: 增强epoch间的内存清理，防止段错误
        if epoch >= 0:  # 每个epoch后都清理
            if accelerator.is_local_main_process:
                print(f"🧹 Epoch {epoch} 结束: 执行内存清理防止段错误...")
            
            # 1. 强制CUDA同步所有流
            torch.cuda.synchronize()

            # 4. 深度显存清理
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 清理进程间共享内存
            
            # 5. Python垃圾回收
            import gc
            gc.collect()
            
            # 6. 检查显存状态
            if torch.cuda.is_available() and accelerator.is_local_main_process:
                for i in range(torch.cuda.device_count()):
                    if i in [6, 7]:  # 你使用的GPU
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"📊 GPU{i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        # Save checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.save_freq == 0:
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