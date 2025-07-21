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

def get_timesteps(pipeline, batch_size: int, num_steps: int, device: str) -> torch.Tensor:
    """生成标准化的时间步张量"""
    if hasattr(pipeline.scheduler, 'timesteps'):
        # 扩散调度器有timesteps属性
        timesteps = pipeline.scheduler.timesteps[:num_steps]
    else:
        # 手动生成时间步
        timesteps = torch.linspace(
            pipeline.scheduler.config.num_train_timesteps - 1, 
            0, 
            num_steps, 
            dtype=torch.long
        )
    
    # 扩展到batch维度
    timesteps = timesteps.unsqueeze(0).repeat(batch_size, 1)
    return timesteps.to(device)

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
    prev_sample, log_prob, prev_sample_mean, std_dev = hunyuan3d_sde_step_with_logprob(
        scheduler=pipeline.scheduler,
        model_output=noise_pred,
        timestep=timestep[0],
        sample=latents,
        prev_sample=next_latents,
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

    return prev_sample, log_prob, prev_sample_mean, std_dev


def save_ckpt_hunyuan3d(model, ema, optimizer, epoch, global_step, save_dir, accelerator):
    """
    SD3风格的检查点保存函数
    
    Args:
        model: 训练模型
        ema: EMA包装器
        optimizer: 优化器
        epoch: 当前epoch
        global_step: 全局步数
        save_dir: 保存目录
        accelerator: Accelerator对象
    """
    checkpoint_dir = os.path.join(save_dir, f"checkpoints", f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 🔧 SD3对齐：保存模型状态
    unwrapped_model = accelerator.unwrap_model(model)
    model_state = unwrapped_model.state_dict()
    
    # 🔧 SD3对齐：保存主模型
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    torch.save(model_state, model_path)
    
    # 🔧 SD3对齐：保存EMA（如果存在）
    if ema is not None:
        ema_state = ema.state_dict()
        ema_path = os.path.join(checkpoint_dir, "pytorch_model_ema.bin")
        torch.save(ema_state, ema_path)
    
    # 🔧 SD3对齐：保存优化器状态
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # 🔧 SD3对齐：保存训练元信息
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "pytorch_version": torch.__version__,
    }
    metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ SD3风格检查点已保存到: {checkpoint_dir}")


def main(argv):
    """主训练函数 - 内联架构（类似SD3）+ SD3内存管理策略"""
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
    
    # ✨ 新增：SD3风格的TF32优化管理
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("✅ SD3风格TF32优化已启用")
    
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
    
    # ✨ 新增：SD3风格的精度管理 - 更智能的inference_dtype选择
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
        logger.info("✅ 使用FP16推理精度")
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
        logger.info("✅ 使用BF16推理精度")
    else:
        logger.info("✅ 使用FP32推理精度")
    
    # ✨ 新增：SD3风格的模型参数冻结策略 - 明确管理哪些参数需要梯度
    logger.info("🔧 SD3风格参数冻结策略...")
    pipeline.vae.requires_grad_(False)
    pipeline.conditioner.requires_grad_(False)
    pipeline.model.requires_grad_(not config.use_lora)
    logger.info("✅ 模型参数梯度设置完成")
    
    # ✨ 新增：SD3风格的分层设备移动 - 不同组件使用不同精度优化内存使用
    logger.info("🔧 SD3风格分层设备移动...")
    
    # VAE保持FP32（SD3策略）- 用于高精度解码
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    
    # Conditioner使用推理精度（SD3策略）- 节省内存
    pipeline.conditioner.to(accelerator.device, dtype=inference_dtype)
    
    # Model的精度策略：LoRA时不强制精度转换（SD3策略）
    if config.use_lora:
        pipeline.model.to(accelerator.device)  # LoRA时让系统自动管理精度
        logger.info("✅ LoRA模式：模型精度由系统自动管理")
    else:
        pipeline.model.to(accelerator.device, dtype=inference_dtype)
        logger.info(f"✅ 全参数训练：模型使用{inference_dtype}精度")
    
    # 🚀 关键修复：显式禁用VAE和conditioner的梯度，设置eval模式（类似SD3）
    logger.info("🔧 设置VAE和conditioner为推理模式...")
    pipeline.vae.eval()
    pipeline.conditioner.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.conditioner.requires_grad_(False)
    logger.info("✅ VAE和conditioner梯度已禁用，已设置为eval模式")
    
    # ✨ 新增：SD3风格的内存优化策略选择
    memory_optimization_level = getattr(config, 'memory_optimization_level', 'aggressive')
    
    if memory_optimization_level == 'aggressive':
        # 🚀 内存优化：训练时将VAE移动到CPU以节省显存（Hunyuan3D特有）
        logger.info("🚀 激进内存优化：将VAE移动到CPU以节省训练显存...")
        pipeline.vae.to('cpu')
        logger.info("✅ VAE已移动到CPU，显存节省约8-12GB")
    elif memory_optimization_level == 'moderate':
        # SD3风格：VAE保留在GPU但使用FP16
        if inference_dtype != torch.float32:
            pipeline.vae.to(accelerator.device, dtype=inference_dtype)
            logger.info(f"✅ 中等内存优化：VAE使用{inference_dtype}精度")
    else:
        # conservative: 保持VAE在GPU FP32（SD3默认）
        logger.info("✅ 保守内存策略：VAE保持GPU FP32精度")
    
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
            ],
            lora_dropout=0.1,
            bias="none",
        )
        
        pipeline.model = get_peft_model(pipeline.model, lora_config)
    
    # 🔧 关键：按照SD3模式，先获取模型引用
    model = pipeline.model
    
    # 🔧 关键：获取trainable参数（SD3方式）
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # ✨ 新增：SD3风格的优化器初始化
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            logger.info("✅ 使用8bit Adam优化器")
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        logger.info("✅ 使用标准AdamW优化器")
    
    # 设置优化器（SD3风格的参数设置）
    optimizer = optimizer_cls(
        trainable_params,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    # 🔧 关键：最后prepare（SD3方式）
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # 🔧 关键：让pipeline使用prepared的模型
    pipeline.model = model
    
    # ✨ 新增：SD3风格的autocast策略 - 根据LoRA使用情况智能选择
    import contextlib
    if config.use_lora:
        autocast = contextlib.nullcontext  # LoRA训练时不使用autocast节省内存
        logger.info("✅ LoRA模式：禁用autocast以节省内存")
    else:
        autocast = accelerator.autocast  # 全参数训练时使用autocast提升性能
        logger.info("✅ 全参数模式：启用autocast提升性能")
    
    # 设置EMA（仿照SD3）
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_params,
            decay=config.train.ema_decay,
            device=accelerator.device
        )
        logger.info("✅ EMA已启用")
    
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
            global_std=getattr(config.sample, 'global_std', False)
        )
    
    # Prepare dataloader
    train_dataloader = accelerator.prepare(train_dataloader)
    
    # executor to perform callbacks asynchronously
    executor = futures.ThreadPoolExecutor(max_workers=8)
    
    # 训练循环（类似SD3架构）
    global_step = 0
    first_epoch = 0
    
    # 🔧 SD3对齐：创建数据迭代器
    train_iter = iter(train_dataloader)
    
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
        samples = []  # 🔧 SD3对齐：直接使用samples列表，不用epoch_samples
        
        simple_gpu_log(f"Epoch {epoch} - 开始采样")
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),  # 🔧 SD3对齐：直接遍历batch数量
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # 🔧 SD3对齐：从train_iter获取数据，处理StopIteration
            try:
                image_paths, prompts, metadata = next(train_iter)
            except StopIteration:
                # 如果数据不够，重新开始迭代器
                train_iter = iter(train_dataloader)
                image_paths, prompts, metadata = next(train_iter)
            
            # 🚀 内联采样逻辑（原trainer.sample_meshes_with_rewards）
            from PIL import Image
            
            # 🔧 多候选生成：为每个图像生成多个候选mesh
            # all_pil_images = []
            # for img_path in image_paths:
            #     # 为当前图像生成 num_meshes_per_image 个候选
            #     candidate_images = [img_path] * config.sample.num_meshes_per_image
            #     pil_candidates = [Image.open(path).convert('RGBA') for path in candidate_images]
            #     all_pil_images.extend(pil_candidates)
            pil_images = [Image.open(path).convert('RGBA') for path in image_paths]
            
            # 编码图像条件
            cond_inputs = pipeline.prepare_image(pil_images)
            image_cond = pipeline.encode_cond(
                image=cond_inputs.pop('image'),
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=True,
                dual_guidance=False,
            )
            # image_cond.keys() = ['main']
            # image_cond['main'].shape = torch.Size([2, 1370, 1024])
            positive_image_cond = {}
            negative_image_cond = {}
            for key in image_cond.keys():
                batch_size = image_cond[key].shape[0]
                positive_image_cond[key] = image_cond[key][:batch_size//2].repeat_interleave(config.sample.num_meshes_per_image, dim=0)
                negative_image_cond[key] = image_cond[key][batch_size//2:].repeat_interleave(config.sample.num_meshes_per_image, dim=0)

            # 调用pipeline
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
            
            # 🔧 SD3对齐：处理latents数据
            latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
            log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)
            kl = torch.stack(all_kl, dim=1)  # (batch_size, num_steps)
            
            # 🔧 SD3对齐：timesteps处理
            timesteps = pipeline.scheduler.timesteps.repeat(
                len(pil_images), 1
            )  # (batch_size, num_steps)
            
            # 计算奖励（异步）
            rewards = executor.submit(reward_fn, meshes, None, {}, image_paths)
            time.sleep(0)  # yield to make sure reward computation starts
            
            # 🔧 SD3对齐：处理latents切片
            current_latents = latents[:, :-1]  # 前n-1个时间步
            next_latents = latents[:, 1:]      # 后n-1个时间步
            
            # 🔧 SD3对齐：简化positive_image_cond处理
            if isinstance(returned_pos_cond, dict):
                positive_image_cond_tensor = returned_pos_cond['main']
            else:
                positive_image_cond_tensor = returned_pos_cond
            
            samples.append({
                "latents": current_latents,
                "next_latents": next_latents,
                "log_probs": log_probs,
                "kl": kl,
                "rewards": rewards,  # 异步结果
                "timesteps": timesteps,
                "positive_image_cond": positive_image_cond_tensor,
            })
            
        # 🔧 采样完成，记录内存状态
        simple_gpu_log(f"Epoch {epoch} - 采样完成")
        
        # # 🔧 SD3对齐：早期epoch跳过检查（重新启用以避免问题）
        # if epoch < 2:
        #     continue
        # NOTE: 没什么用，注释掉了
            
        # 🔧 检查samples是否为空，避免IndexError
        if not samples:
            logger.warning(f"⚠️  Epoch {epoch}: No samples collected, skipping training")
            continue
            
        # 🔧 SD3对齐：等待所有奖励计算完成
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
        ):
            reward_details, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.tensor(reward_details['avg'], device=accelerator.device, dtype=torch.float32)
        
        # 🔧 SD3对齐：collate samples into dict（完全按照SD3方式）
        samples = {k: torch.cat([s[k] for s in samples], dim=0) for k in samples[0].keys()}
        
        # 🚀 处理奖励和advantages（类似SD3）
        rewards_avg = samples["rewards"]  # 现在直接是tensor
        kl_tensor = samples["kl"]
        
        # 🔧 SD3对齐：KL调整后的奖励，保持SD3的结构
        samples["rewards"] = {"avg": rewards_avg}  # 重新包装为dict结构
        samples["rewards"]["ori_avg"] = rewards_avg
        samples["rewards"]["avg"] = rewards_avg.unsqueeze(-1) - config.sample.kl_reward * kl_tensor
        
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards_np = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        
        # 计算advantages（类似SD3）
        if config.per_image_stat_tracking and stat_tracker:
            # 🔧 修复：使用简化的图像路径处理
            # 注意：这里我们不再有images信息，所以简化处理
            advantages_np = stat_tracker.update(
                list(range(len(gathered_rewards_np["avg"]))),  # 使用索引代替图像路径
                gathered_rewards_np["avg"].mean(axis=1)
            )
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
        
        # 🔧 SD3对齐：数据清理和样本过滤（参考SD3实现）
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())
        
        # 🔧 SD3对齐：删除训练不需要的键
        del samples["rewards"]
        if "images" in samples:
            del samples["images"]

        # 🔧 修复：直接使用前面过滤后的samples
        num_batches = getattr(config.sample, 'num_batches_per_epoch', 1)
        
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps
        
        #################### TRAINING ####################
        # 内联训练逻辑 - 完全对齐SD3架构
        for inner_epoch in range(config.train.num_inner_epochs):
            # 🔧 SD3对齐：批次维度随机化
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            
            # 🔧 SD3对齐：时间维度随机化（每个样本独立）
            if getattr(config.train, 'shuffle_timesteps', False):  # 注意默认为False
                if total_batch_size > 0:  # 添加边界检查
                    perms = torch.stack([
                        torch.randperm(num_timesteps, device=accelerator.device)
                        for _ in range(total_batch_size)
                    ])
                else:
                    perms = torch.empty(0, num_timesteps, device=accelerator.device, dtype=torch.long)
            else:
                # SD3默认：使用顺序时间步
                if total_batch_size > 0:  # 添加边界检查
                    perms = torch.stack([
                        torch.arange(num_timesteps, device=accelerator.device)
                        for _ in range(total_batch_size)
                    ])
                else:
                    perms = torch.empty(0, num_timesteps, device=accelerator.device, dtype=torch.long)
            
            # 对时间相关的键进行重排
            for key in ["timesteps", "latents", "next_latents", "log_probs", "advantages"]:
                if key in samples:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None],
                        perms,
                    ]
            
            # 🔧 SD3对齐：重新批处理
            samples_batched = {
                k: v.reshape(-1, total_batch_size // num_batches, *v.shape[1:])
                for k, v in samples.items()
            }
            
            # 转换为list of dicts格式（SD3风格）
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            
            # 🔧 SD3对齐：双重循环训练结构
            model.train()
            info = defaultdict(list)
            
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # 训练每个时间步（SD3风格）
                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    # 🔧 SD3对齐：梯度累积包装器
                    with accelerator.accumulate(model):
                        with autocast():
                            # 计算log概率
                            prev_sample, log_prob, prev_sample_mean, std_dev = compute_log_prob_3d(
                                pipeline, sample, j, config
                            )
                            
                            # 参考log概率（KL正则化）
                            if getattr(config.train, 'beta', 0) > 0:
                                with torch.no_grad():
                                    # 🔧 SD3风格：安全访问DDP包装后的模型
                                    model_for_adapter = model.module if hasattr(model, 'module') else model
                                    with model_for_adapter.disable_adapter():
                                        _, log_prob_ref, prev_sample_mean_ref, std_dev_ref = compute_log_prob_3d(
                                            pipeline, sample, j, config
                                        )
                        
                        # 🔧 SD3对齐：GRPO损失计算
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
                        
                        # KL损失（SD3风格）
                        if getattr(config.train, 'beta', 0) > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim))) / (2 * std_dev ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss
                        
                        # 🔧 SD3对齐：记录统计信息
                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > config.train.clip_range).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if getattr(config.train, 'beta', 0) > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)
                        
                        # 🔧 SD3对齐：反向传播和优化
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # 🔧 SD3对齐：记录训练信息和更新全局步数
                    if accelerator.sync_gradients:
                        # 记录训练统计信息（SD3风格）
                        step_info = {k: torch.tensor(v).mean().item() for k, v in info.items()}
                        step_info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(step_info, step=global_step)
                        global_step += 1
                        
                        # 清空统计信息
                        info = defaultdict(list)
                        
                        # 🔧 SD3对齐：EMA更新
                    if ema is not None:
                            ema.step(model.parameters())
            
            # 记录epoch统计信息
            logger.info(f"Epoch {epoch}.{inner_epoch} completed")
        
        # 🔧 训练完成，记录内存状态
        simple_gpu_log(f"Epoch {epoch} - 训练完成")
        
        # 🔧 SD3对齐：周期性保存检查点
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
            logger.info(f"Checkpoint saved at epoch {epoch}")
        
        simple_gpu_log(f"Epoch {epoch} - 完成")

if __name__ == "__main__":
    app.run(main) 