#!/usr/bin/env python3
"""
Train Hunyuan3D with GRPO - 使用统一的配置管理系统

使用与SD3相同的配置管理方式：
- absl.flags 用于命令行参数
- ml_collections 用于复杂配置结构
- 与 train_sd3.py 完全一致的接口
"""

# 🔧 删除RMSNorm补丁：PyTorch 2.6.0+ 原生支持RMSNorm，无需补丁
# 🔧 应用RMSNorm兼容性补丁（仿照官方代码）
import sys
import os
# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# from pytorch_rmsnorm_patch import apply_rmsnorm_patch
# apply_rmsnorm_patch()

import math
import random
import time
import logging
from functools import partial
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
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

# 🔧 统一配置管理 - 与SD3保持一致
import ml_collections
from absl import app
from absl import flags
from ml_collections import config_flags

# 🔧 导入统一的配置文件
_CONFIG = config_flags.DEFINE_config_file("config")

# 🔧 与SD3保持一致的进度条配置
tqdm = partial(tqdm, dynamic_ncols=True)

# 数据和模型相关导入
# from datasets.image_datasets import ImageDataset  # 🔧 暂时移除
from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from reward_models.rewards_mesh import multi_mesh_score
from reward_models.uni3d_scorer.uni3d_scorer import Uni3DScorer
from flow_grpo.trainer_3d import Hunyuan3DGRPOTrainer
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerImageStatTracker  # 🔧 修改：使用 PerImageStatTracker

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)

# 🔧 添加GPU计时和监控功能
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
            return 0
    
    start_util = get_gpu_utilization()
    print(f"  ⚡ 初始GPU利用率: {start_util}%")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        end_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        end_util = get_gpu_utilization()
        
        # 计算平均GPU利用率
        avg_util = (start_util + end_util) / 2
        
        print(f"✅ 完成: {name}")
        print(f"  ⏱️  耗时: {end_time - start_time:.2f}秒")
        print(f"  📊 结束显存: {end_memory:.2f}GB (已分配) / {end_reserved:.2f}GB (已保留)")
        print(f"  📈 显存变化: {end_memory - start_memory:+.2f}GB (已分配) / {end_reserved - start_reserved:+.2f}GB (已保留)")
        print(f"  ⚡ 结束GPU利用率: {end_util}%")
        print(f"  🔥 平均GPU利用率: {avg_util:.1f}%")
        print("")

# HuggingFace imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

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


# 🔧 移除内置配置函数：改用外部配置文件（与SD3保持一致）
# def create_config():
#     """Create default configuration."""
#     from types import SimpleNamespace
#     
#     config = SimpleNamespace()
#     
#     # Basic settings
#     config.data_dir = "data/3d_training"
#     config.save_dir = "checkpoints/hunyuan3d_grpo"
#     config.resume_from = None
#     config.num_epochs = 100
#     config.mixed_precision = "fp16"
#     config.seed = 42
#     config.use_lora = False
#     config.eval_freq = 10
#     config.save_freq = 10
#     config.per_prompt_stat_tracking = True
#     config.deterministic = False  # 🔧 默认使用SDE模式
#     
#     # Sample configuration
#     config.sample = SimpleNamespace()
#     config.sample.input_batch_size = 2           # 🔧 新增：每次处理多少张不同图像
#     config.sample.num_meshes_per_image = 2       # 🔧 新增：每张图像生成多少个mesh候选
#     config.sample.num_batches_per_epoch = 2      # 每个epoch采样多少次
#     config.sample.num_steps = 20                 # 扩散步数
#     config.sample.guidance_scale = 5.0
#     config.sample.kl_reward = 0.1
#     config.sample.test_batch_size = 4
#     config.sample.global_std = 0.5
#     
#     # Training config
#     config.train = SimpleNamespace()
#     config.train.batch_size = 2                  # 🔧 修改：减少到2避免CUDA错误
#     config.train.gradient_accumulation_steps = 2
#     config.train.num_inner_epochs = 1
#     config.train.learning_rate = 1e-5
#     config.train.beta = 0.01  # KL coefficient
#     config.train.clip_range = 0.2
#     config.train.adv_clip_max = 5.0
#     config.train.max_grad_norm = 1.0
#     config.train.cfg = False  # 🔧 修复：禁用训练时的CFG，因为采样时已经生成了CFG格式的条件
#     config.train.ema = True
#     config.train.ema_decay = 0.999
#     
#     return config


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
                input_batch_size=len(image_paths),  # 🔧 适配评估模式
                num_meshes_per_image=1,  # 🔧 评估时每个图像只生成一个mesh
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                deterministic=True,  # Use deterministic for evaluation
                kl_reward=0.0,  # No KL reward during evaluation
                # 🔧 新增：传递 mesh 配置参数
                octree_resolution=config.mesh.octree_resolution,
                mc_level=config.mesh.mc_level,
                mc_algo=config.mesh.mc_algo,
                box_v=config.mesh.box_v,
                num_chunks=config.mesh.num_chunks,
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
            for i, image_path in enumerate(image_paths):
                eval_results.append({
                    "image_path": image_path,
                    "geometric_score": rewards["geometric"][i] if "geometric" in rewards else 0.0,
                    "semantic_score": rewards["uni3d"][i] if "uni3d" in rewards else 0.0,
                    "avg_score": rewards["avg"][i],
                })
    
    # Aggregate results
    if eval_rewards:
        all_geometric = np.concatenate([r.get("geometric", [0.0] * len(r["avg"])) for r in eval_rewards])
        all_semantic = np.concatenate([r.get("uni3d", [0.0] * len(r["avg"])) for r in eval_rewards])
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


def train_step_with_sub_batching(trainer, all_samples, config, optimizer, accelerator):
    """训练步骤，支持子批次处理"""
    
    total_batch_size = all_samples["timesteps"].shape[0]
    train_batch_size = config.train.batch_size
    
    # 🔧 验证约束
    assert total_batch_size % train_batch_size == 0, \
        f"total_batch_size ({total_batch_size}) must be divisible by train_batch_size ({train_batch_size})"
    
    num_sub_batches = total_batch_size // train_batch_size
    
    train_metrics = {
        "policy_loss": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "train_batch_size": train_batch_size,
        "num_sub_batches": num_sub_batches,
    }
    
    print(f"🔧 子批次训练：{total_batch_size} 样本分为 {num_sub_batches} 个子批次，每批 {train_batch_size} 样本")
    
    # 分批训练
    for sub_batch_idx in range(num_sub_batches):
        start_idx = sub_batch_idx * train_batch_size
        end_idx = start_idx + train_batch_size
        
        print(f"  子批次 {sub_batch_idx+1}/{num_sub_batches}: 样本 {start_idx}:{end_idx}")
        
        # 切片子批次
        sub_batch_samples = {}
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor):
                sub_batch_samples[key] = value[start_idx:end_idx]
            elif isinstance(value, dict):
                sub_batch_samples[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        sub_batch_samples[key][sub_key] = sub_value[start_idx:end_idx]
                    else:
                        sub_batch_samples[key][sub_key] = sub_value
            else:
                sub_batch_samples[key] = value
        
        # 训练子批次
        sub_metrics = trainer.train_step(
            samples=sub_batch_samples,
            pipeline=trainer.pipeline.core_pipeline,
            optimizer=optimizer,
            config=config,
            accelerator=accelerator,
        )
        
        # 累积指标
        for key, value in sub_metrics.items():
            if key in train_metrics:
                train_metrics[key] += value / num_sub_batches
    
    return train_metrics


def main(argv):
    """Main training function."""
    # 🔧 统一配置系统：使用与SD3相同的配置标志
    # 删除未使用的argv参数警告
    del argv
    
    config = _CONFIG.value
    
    with gpu_timer("🚀 完整训练初始化"):
        # 🔧 添加deterministic配置
        if hasattr(config, 'deterministic') and config.deterministic:
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
        if hasattr(config, 'seed') and config.seed is not None:
            from accelerate.utils import set_seed
            set_seed(config.seed)
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 🔧 验证约束条件
        total_meshes = config.sample.input_batch_size * config.sample.num_meshes_per_image
        assert config.train.batch_size <= total_meshes, \
            f"train.batch_size ({config.train.batch_size}) must be <= total_meshes ({total_meshes})"
        assert total_meshes % config.train.batch_size == 0, \
            f"total_meshes ({total_meshes}) must be divisible by train.batch_size ({config.train.batch_size})"
        
        print(f"🔧 Batch size配置:")
        print(f"  input_batch_size: {config.sample.input_batch_size}")
        print(f"  num_meshes_per_image: {config.sample.num_meshes_per_image}")
        print(f"  total_meshes: {total_meshes}")
        print(f"  train.batch_size: {config.train.batch_size}")
        
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
            
            # Create trainer - 明确：只传递包装类，启用SD3式batch处理
            trainer = Hunyuan3DGRPOTrainer(
                pipeline=pipeline_wrapper,  # 传递包装类，不是内部pipeline
                reward_config=reward_config,
                device=accelerator.device,
                sample_batch_size=config.sample.input_batch_size,  # 🔧 修复：使用 input_batch_size
                train_batch_size=config.train.batch_size,         # 🔧 新增：训练阶段batch size
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
        batch_size=config.sample.input_batch_size,  # 🔧 修复：使用 input_batch_size
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
        # 🔧 Hunyuan3DDiT 的正确 LoRA 配置（基于真实架构分析）
        lora_config = LoraConfig(
            r=32,  # 增加rank以获得更好效果
            lora_alpha=64,  # 增加alpha scaling
            target_modules=[
                # DoubleStreamBlock - 图像流注意力层
                "img_attn.qkv", "img_attn.proj",
                # DoubleStreamBlock - 图像条件流注意力层（虽然叫txt，但处理的是图像条件）
                "txt_attn.qkv", "txt_attn.proj", 
                # DoubleStreamBlock - MLP 层
                "img_mlp.0", "img_mlp.2",
                "txt_mlp.0", "txt_mlp.2",
                # SingleStreamBlock - 融合层
                "linear1", "linear2",
                # 关键输入/输出层
                "latent_in", "cond_in",  # 输入层
                "final_layer.linear"     # 最终输出层
            ],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        # 🔧 关键修复：将 LoRA 模型设置回 pipeline，确保 trainer 可以访问 disable_adapter()
        core_pipeline.model = model
        
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
    if config.per_image_stat_tracking:  # 🔧 修改：使用 per_image_stat_tracking
        stat_tracker = PerImageStatTracker(config.sample.global_std)
    
    # Prepare for training
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )
    
    # Thread executor for async operations
    executor = futures.ThreadPoolExecutor(max_workers=4)
    
    # Training info
    samples_per_epoch = len(train_dataloader) * config.sample.input_batch_size
    total_train_batch_size = (
        config.train.batch_size * 
        accelerator.num_processes * 
        config.train.gradient_accumulation_steps
    )
    
    logger.info("***** Running 3D GRPO Training *****")
    logger.info(f"  Num training samples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.input_batch_size}")
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
                        input_batch_size=config.sample.input_batch_size,        # 🔧 新增
                        num_meshes_per_image=config.sample.num_meshes_per_image, # 🔧 新增
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        deterministic=config.deterministic,
                        kl_reward=config.sample.kl_reward,
                        # 🔧 新增：传递 mesh 配置参数
                        octree_resolution=config.mesh.octree_resolution,
                        mc_level=config.mesh.mc_level,
                        mc_algo=config.mesh.mc_algo,
                        box_v=config.mesh.box_v,
                        num_chunks=config.mesh.num_chunks,
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
                # 🔧 修复：正确处理不同类型的reward数据，确保维度一致
                sample["rewards"] = {}
                for key, value in rewards.items():
                    if isinstance(value, (list, tuple)):
                        # 列表或元组，直接转换
                        sample["rewards"][key] = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                    elif isinstance(value, np.ndarray):
                        # 🔧 关键修复：numpy数组，直接转换（不要嵌套）
                        sample["rewards"][key] = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                    elif isinstance(value, torch.Tensor):
                        # 已经是张量，确保在正确设备上
                        sample["rewards"][key] = value.to(device=accelerator.device, dtype=torch.float32)
                    elif isinstance(value, (int, float)):
                        # 标量，转换为单元素张量
                        sample["rewards"][key] = torch.tensor([value], device=accelerator.device, dtype=torch.float32)
                    else:
                        # 其他类型，尝试转换
                        sample["rewards"][key] = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
                    
                    # 🔧 调试：打印每个reward的形状
                    print(f"🔍 reward {key}: shape={sample['rewards'][key].shape}, dtype={sample['rewards'][key].dtype}, device={sample['rewards'][key].device}")
                
                print(f"🔧 修复：rewards处理完成，设备 {accelerator.device}")
        
        # 🔧 调试：在collate之前检查每个样本的数据类型
        print(f"🔍 样本数据调试 - 检查每个字段的类型:")
        for i, sample in enumerate(epoch_samples):
            print(f"  样本 {i}:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: Tensor, shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, dict):
                    print(f"    {key}: dict with keys {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"      {sub_key}: Tensor, shape={sub_value.shape}, dtype={sub_value.dtype}")
                        else:
                            print(f"      {sub_key}: {type(sub_value)} = {sub_value}")
                elif isinstance(value, (list, tuple)):
                    print(f"    {key}: {type(value)} with {len(value)} items")
                    if len(value) > 0:
                        print(f"      first item type: {type(value[0])}")
                else:
                    print(f"    {key}: {type(value)} = {value}")
        
        # Collate samples
        all_samples = {
            k: torch.cat([s[k] for s in epoch_samples], dim=0)
            if not isinstance(epoch_samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in epoch_samples], dim=0)
                for sub_key in epoch_samples[0][k]
                if isinstance(epoch_samples[0][k][sub_key], torch.Tensor)  # 🔧 修复：只连接张量
            }
            for k in epoch_samples[0].keys()
            if k not in ["meshes", "images", "prompts", "positive_image_cond", "metadata"]  # 🔧 修复：跳过positive_image_cond和metadata
        }
        
        # 🔧 修复：单独处理positive_image_cond，因为它是字典且在所有样本中相同
        if "positive_image_cond" in epoch_samples[0]:
            all_samples["positive_image_cond"] = epoch_samples[0]["positive_image_cond"]  # 使用第一个样本的positive_image_cond
        
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
        # 🔧 优化：保持rewards在CUDA上，只在需要日志时转CPU
        gathered_rewards_for_log = {
            key: value.cpu().numpy() 
            for key, value in gathered_rewards.items()
        }
        
        # Log metrics (使用CPU版本)
        accelerator.log({
            "epoch": epoch,
            **{f"reward_{key}": value.mean() for key, value in gathered_rewards_for_log.items()},
            "kl": all_samples["kl"].mean().cpu().numpy(),
        }, step=global_step)
        
        # 🔧 优化：直接在CUDA上计算advantages，避免不必要的设备转换
        if config.per_image_stat_tracking and stat_tracker:
            # �� 修复：Hunyuan3D使用图像路径进行统计跟踪，而不是文本提示
            all_images = []
            for sample in epoch_samples:
                all_images.extend(sample["images"])  # 🔧 图像路径列表
            
            # 🔧 修复：只有当处理的样本数等于训练集大小时才使用per-image跟踪
            if len(all_images) == len(train_dataset):
                # stat_tracker需要CPU数据，但我们立即转回CUDA
                advantages_np = stat_tracker.update(all_images, gathered_rewards['avg'].cpu().numpy())
                # 🔧 优化：直接在目标设备上创建tensor，避免中间转换
                advantages = torch.tensor(advantages_np, device=accelerator.device, dtype=torch.float32)
                print(f"🔧 优化：使用per-image advantages，直接在CUDA上创建")
            else:
                logger.warning(f"Processed {len(all_images)} samples but have {len(train_dataset)} in dataset. Using global advantages.")
                # 🔧 优化：直接在CUDA上计算global advantages，无需CPU转换
                advantages = gathered_rewards['avg']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
                print(f"🔧 优化：使用global advantages，保持在CUDA上计算")
        else:
            # 🔧 优化：直接在CUDA上计算global advantages
            advantages = gathered_rewards['avg']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            print(f"🔧 优化：使用global advantages，保持在CUDA上计算")
        
        print(f"🔧 设备优化：advantages在设备 {advantages.device} 上，形状 {advantages.shape}")
        
        #  修复：正确处理advantages的维度
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
            # 🔧 优化：advantages已经在正确设备上，无需.to()操作
            all_samples["advantages"] = advantages[start_idx:end_idx]
            print(f"🔧 优化：2D advantages切片，无设备转换")
        else:
            # 如果原始advantages是1D的，保持1D形状
            # 🔧 优化：sample_advantages已经在正确设备上
            all_samples["advantages"] = sample_advantages[start_idx:end_idx]
            print(f"🔧 优化：1D advantages切片，无设备转换")
        
        # 🔧 优化：一次性检查所有tensor的设备，减少重复检查
        print(f"🔧 设备检查：开始统一设备检查...")
        
        # 🔧 优化：强制设备一致性检查，确保所有tensor都在正确设备上
        print(f"🔧 设备检查：验证所有tensor设备一致性...")
        
        # 🔧 修复：处理cuda和cuda:0的设备表示差异
        def devices_match(tensor_device, target_device):
            """检查两个设备是否匹配，处理cuda和cuda:0的差异"""
            tensor_str = str(tensor_device)
            target_str = str(target_device)
            
            # 如果完全相同，直接返回True
            if tensor_str == target_str:
                return True
            
            # 处理cuda和cuda:0的等价性
            if (tensor_str == "cuda:0" and target_str == "cuda") or (tensor_str == "cuda" and target_str == "cuda:0"):
                return True
            
            return False
        
        # 同时更新所有其他tensor到相同的样本范围
        for key, value in all_samples.items():
            if key != "advantages" and isinstance(value, torch.Tensor):
                all_samples[key] = value[start_idx:end_idx]
                # 🔧 强制检查：确保tensor在正确设备上
                assert devices_match(value.device, accelerator.device), f"❌ {key} 在错误设备上: {value.device}, 期望: {accelerator.device}"
            elif key != "advantages" and isinstance(value, dict):
                all_samples[key] = {
                    sub_key: sub_value[start_idx:end_idx] 
                    for sub_key, sub_value in value.items()
                }
                # 🔧 强制检查：确保嵌套tensor在正确设备上
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        assert devices_match(sub_value.device, accelerator.device), f"❌ {key}.{sub_key} 在错误设备上: {sub_value.device}, 期望: {accelerator.device}"
        
        print(f"✅ 所有tensor设备一致性验证通过: {accelerator.device}")
        
        # Filter out zero-advantage samples - 现在在正确的维度上进行筛选
        if all_samples["advantages"].dim() == 2:
            # 如果advantages是2D的，使用平均值来筛选
            mask = (all_samples["advantages"].mean(dim=1).abs() > 1e-6)
        else:
            # 如果advantages是1D的，直接筛选
            mask = (all_samples["advantages"].abs() > 1e-6)
        
        # 🔧 优化：mask已经在正确设备上，无需转换
        print(f"🔧 优化：mask在设备 {mask.device} 上，形状 {mask.shape}")
        
        print(f"🔍 样本筛选:")
        print(f"  mask.shape: {mask.shape}")
        print(f"  mask.device: {mask.device}")
        print(f"  筛选前样本数: {all_samples['advantages'].shape[0]}")
        print(f"  筛选后样本数: {mask.sum().item()}")
        
        # 🔧 优化：简化设备检查，只在真正需要时转换
        filtered_samples = {}
        for key, value in all_samples.items():
            if isinstance(value, torch.Tensor):
                # 🔧 优化：所有tensor应该已经在正确设备上，直接应用mask
                filtered_samples[key] = value[mask]
            elif isinstance(value, dict):
                filtered_samples[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        # 🔧 修复：positive_image_cond是按图像数量而不是mesh数量，不应用mask
                        if key == "positive_image_cond":
                            filtered_samples[key][sub_key] = sub_value  # 不应用mask
                        else:
                            # 🔧 优化：所有嵌套tensor也应该在正确设备上
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
        # 🔧 GPU内存优化：在训练前清理显存
        torch.cuda.empty_cache()
        print(f"🔧 GPU内存清理：训练前释放缓存")
        
        for inner_epoch in range(config.train.num_inner_epochs):
            model.train()  # 只需要设置核心扩散模型为训练模式
            
            # 🔧 关键修复：处理batch size不一致问题
            # 获取主要数据的batch size
            batch_size = all_samples["timesteps"].shape[0]
            
            # 🔧 修复：确保positive_image_cond的batch size与其他数据一致
            if "positive_image_cond" in all_samples and isinstance(all_samples["positive_image_cond"], dict):
                pos_cond = all_samples["positive_image_cond"]
                if "main" in pos_cond and pos_cond["main"].shape[0] != batch_size:
                    print(f"🔧 修复batch size不一致: positive_image_cond.main从{pos_cond['main'].shape[0]}扩展到{batch_size}")
                    # 重复条件以匹配batch size
                    current_size = pos_cond["main"].shape[0]
                    repeat_factor = batch_size // current_size
                    remainder = batch_size % current_size
                    
                    repeated_cond = pos_cond["main"].repeat(repeat_factor, 1, 1)
                    if remainder > 0:
                        repeated_cond = torch.cat([repeated_cond, pos_cond["main"][:remainder]], dim=0)
                    
                    all_samples["positive_image_cond"]["main"] = repeated_cond
                    print(f"🔧 修复完成: positive_image_cond.main.shape = {all_samples['positive_image_cond']['main'].shape}")
            
            # 🔧 修复：确保所有rewards的形状一致
            if "rewards" in all_samples and isinstance(all_samples["rewards"], dict):
                for reward_key, reward_value in all_samples["rewards"].items():
                    if isinstance(reward_value, torch.Tensor):
                        if reward_value.ndim == 2 and reward_value.shape[0] == batch_size:
                            # 如果是二维且第一维正确，取平均值转为一维
                            if reward_key == "avg":
                                all_samples["rewards"][reward_key] = reward_value.mean(dim=1)
                                print(f"🔧 修复rewards形状: {reward_key} 从 {reward_value.shape} 转为 {all_samples['rewards'][reward_key].shape}")
                        elif reward_value.shape[0] != batch_size:
                            print(f"🚨 警告: rewards[{reward_key}].shape[0]={reward_value.shape[0]} != batch_size={batch_size}")
            
            # 🔧 验证所有tensor的batch size一致性
            print(f"🔍 Shuffle前批次大小验证:")
            for k, v in all_samples.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}.shape[0]: {v.shape[0]}")
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, torch.Tensor):
                            print(f"  {k}[{sub_k}].shape[0]: {sub_v.shape[0]}")
            
            # Shuffle samples
            perm = torch.randperm(batch_size, device=accelerator.device)
            print(f"🔧 生成shuffle perm: {perm} (max_index={perm.max()}, batch_size={batch_size})")
            
            shuffled_samples = {}
            for k, v in all_samples.items():
                if isinstance(v, torch.Tensor):
                    shuffled_samples[k] = v[perm]
                elif isinstance(v, dict):
                    shuffled_samples[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, torch.Tensor):
                            # 🔧 添加安全检查
                            if sub_v.shape[0] != batch_size:
                                print(f"🚨 错误：{k}[{sub_k}].shape[0]={sub_v.shape[0]} != batch_size={batch_size}")
                                raise ValueError(f"Tensor {k}[{sub_k}] batch size mismatch")
                            shuffled_samples[k][sub_k] = sub_v[perm]
                        else:
                            shuffled_samples[k][sub_k] = sub_v
                else:
                    shuffled_samples[k] = v
            
            # 🔧 使用子批次训练或直接训练
            total_batch_size = shuffled_samples["timesteps"].shape[0]
            if total_batch_size > config.train.batch_size:
                # 使用子批次训练
                train_metrics = train_step_with_sub_batching(
                    trainer=trainer,
                    all_samples=shuffled_samples,
                    config=config,
                    optimizer=optimizer,
                    accelerator=accelerator,
                )
            else:
                # 直接训练
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
    # 🔧 统一主函数调用：使用与SD3相同的absl.app.run
    app.run(main)
