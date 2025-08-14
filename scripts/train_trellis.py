#!/usr/bin/env python3
"""
TRELLIS Stage 2 GRPO Training Script

- 两阶段：Stage 1 冻结在线生成稀疏坐标；Stage 2 对 SLatFlowModel 做 GRPO
- 复用 Hunyuan3D/SD3 的 GRPO 训练框架与指标定义
- 稀疏张量：coords(N,4) + feats(N,C)，接入 Flow Matching + SDE + LogProb
- 遵循 TRELLIS_DEV.md 约束：仅训练 Stage 2，无 try/except，无 fallback
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import hashlib

import ml_collections
from absl import app
from ml_collections import config_flags
from PIL import Image

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS/GRPO 相关模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from flow_grpo.diffusers_patch.trellis_stage2_with_logprob import trellis_stage2_with_logprob
from flow_grpo.diffusers_patch.sparse_tensor_grpo import compute_log_prob_trellis_stage2, compute_log_prob_trellis_stage2_batched
from flow_grpo.stat_tracking import PerImageStatTracker
from flow_grpo.ema import EMAModuleWrapper
from reward_models.rewards_mesh import MeshScorer
from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
from generators.trellis.utils import convert_trellis_to_trimesh

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config")
from peft import LoraConfig, get_peft_model, PeftModel
from flow_grpo.peft_sparse.sparse_lora import register_sparse_linear_with_peft


def compute_advantages_per_image(
    image_names: List[str],
    rewards_np_local: np.ndarray,
    accelerator: Accelerator,
    stat_tracker: Optional[PerImageStatTracker],
    epoch: int,
) -> np.ndarray:
    """按图像分组计算优势，并记录与 Hunyuan3D 一致的统计。

    返回当前进程对应的本地优势向量，顺序与 `image_names`/`rewards_np_local` 对齐。
    """
    device = accelerator.device

    # 将字符串图片名映射为跨进程稳定的整型 ID（md5 前 8 字节）
    def name_to_stable_id(name: str) -> int:
        h = hashlib.md5(name.encode("utf-8")).digest()
        # 限制到 63-bit 有符号范围，避免 np/torch 转换溢出
        return int.from_bytes(h[:8], byteorder="big", signed=False) & 0x7fffffffffffffff  # 标量

    # 直接构造 torch.long 避免 numpy 溢出
    image_ids_list = [name_to_stable_id(n) for n in image_names]  # 长度 N
    image_ids_local_tensor = torch.tensor(image_ids_list, device=device, dtype=torch.long)  # 形状 (N,)
    rewards_local_tensor = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # 形状 (N,)

    rewards_global_tensor = accelerator.gather(rewards_local_tensor)  # 形状 (G*N,)
    image_ids_global_tensor = accelerator.gather(image_ids_local_tensor)  # 形状 (G*N,)

    # 计算全局优势（按图像分组或全局标准化），保持在 torch 上
    if stat_tracker is None:
        eps = 1e-8  # 标量
        mean = rewards_global_tensor.mean()  # ()
        std = rewards_global_tensor.std()  # ()
        advantages_global_tensor = (rewards_global_tensor - mean) / (std + eps)  # 形状 (G*N,)
    else:
        advantages_global_tensor = stat_tracker.update_torch(image_ids_global_tensor, rewards_global_tensor)  # 形状 (G*N,)

    # 记录 per-image 统计（仅主进程）
    if accelerator.is_main_process and stat_tracker is not None:
        group_size, trained_image_num = stat_tracker.get_stats()  # 标量, 标量
        unique_ids = torch.unique(image_ids_global_tensor)  # 形状 (U,)
        if unique_ids.numel() > 0:
            stds = []
            for uid in unique_ids.tolist():
                mask = (image_ids_global_tensor == uid)
                stds.append(rewards_global_tensor[mask].std().item())
            zero_std_ratio = float((torch.tensor(stds) == 0).sum().item() / len(stds))
        else:
            zero_std_ratio = 0.0  # 标量
        accelerator.log(
            {
                "group_size": float(group_size),
                "trained_image_num": int(trained_image_num),
                "zero_std_ratio": zero_std_ratio,
            },
            step=epoch,
        )

    # 切回当前进程本地片段
    local_n = rewards_np_local.shape[0]  # 标量
    world = accelerator.num_processes  # 标量
    rank = accelerator.process_index  # 标量
    assert advantages_global_tensor.numel() % world == 0, "Global advantages size not divisible by world size"
    per_rank = advantages_global_tensor.numel() // world  # 标量
    assert per_rank == local_n, "Local sample count mismatch across processes"
    advantages_local_tensor = advantages_global_tensor.reshape(world, per_rank)[rank]  # 形状 (N,)

    return advantages_local_tensor.detach().cpu().numpy().astype(np.float64)


def save_meshes_for_preview(
    meshes,
    repeated_image_paths,
    rewards,
    epoch: int,
    save_dir: str,
    device_str: str = "cuda",
):
    """保存 mesh 为 OBJ 并渲染预览 PNG（对齐 Hunyuan3D 的可视化体验）。

    - meshes: List[kiui.Mesh]
    - repeated_image_paths: 与 meshes 一一对应的图像路径列表（每图像重复 K 次）
    - rewards: 与 meshes 对齐的奖励向量（可选，仅用于命名/扩展，当前未写入文件名）
    - epoch: 当前 epoch 编号
    - save_dir: 输出目录
    - device_str: 渲染设备字符串（"cuda" 或 "cpu"）
    """
    import os
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_for_training

    os.makedirs(save_dir, exist_ok=True)

    mesh_files = []
    preview_files = []

    for idx, (mesh, img_path) in enumerate(zip(meshes, repeated_image_paths)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        safe_base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).rstrip()

        mesh_path = os.path.join(save_dir, f"{safe_base}_mesh_{idx}.obj")
        preview_path = os.path.join(save_dir, f"{safe_base}_preview_{idx}.png")

        mesh.write(mesh_path)
        render_mesh_for_training(mesh_path, preview_path, device=device_str)

        mesh_files.append(mesh_path)
        preview_files.append(preview_path)

    return mesh_files, preview_files


class Image3DDataset(Dataset):
    """最小图像数据集（与 Hunyuan3D 保持一致接口）"""
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        if (self.image_dir / "images").exists():
            self.image_dir = self.image_dir / "images"
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(sorted(self.image_dir.glob(ext)))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = str(self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        return {
            "image": image,
            "image_path": image_path,
            "metadata": {"image_name": self.image_files[idx].name}
        }

    @staticmethod
    def collate_fn(examples):
        images = [ex["image"] for ex in examples]
        image_paths = [ex["image_path"] for ex in examples]
        metadata = [ex["metadata"] for ex in examples]
        return images, image_paths, metadata


class DistributedImageRepeatSampler(Sampler):
    """分布式重复采样器（与 SD3/Hunyuan3D 对齐的 K-repeat 采样器）。

    - dataset: Image3DDataset
    - batch_size: 每卡输入图像数（形如 B）
    - k: 每图生成候选数（形如 K）
    - num_replicas: 总卡数（形如 G）
    - rank: 当前卡编号 [0, G)
    - seed: 随机种子
    """
    def __init__(self, dataset: Dataset, batch_size: int, k: int, num_replicas: int, rank: int, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)

        self.total_samples = self.num_replicas * self.batch_size  # 标量
        if self.total_samples < self.k:
            self.simple_repeat_mode = True
            self.m = self.total_samples  # 标量
        else:
            assert self.total_samples % self.k == 0, f"k can not div n*b, k{self.k}-num_replicas{self.num_replicas}-batch_size{self.batch_size}"
            self.simple_repeat_mode = False
            self.m = self.total_samples // self.k  # 标量

        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if self.simple_repeat_mode:
                # 可用图像索引（随机顺序）
                available = torch.randperm(len(self.dataset), generator=g).tolist()  # 形状 (N,)
                # 填满所有卡的 batch
                repeated = [available[i % len(available)] for i in range(self.total_samples)]  # 形状 (G*B,)
                # 切分到各卡
                per_card = [repeated[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_replicas)]
                yield per_card[self.rank]
            else:
                indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()  # 形状 (m,)
                repeated = [idx for idx in indices for _ in range(self.k)]  # 形状 (G*B,)
                shuffled_idx = torch.randperm(len(repeated), generator=g).tolist()  # 形状 (G*B,)
                shuffled = [repeated[i] for i in shuffled_idx]  # 形状 (G*B,)
                per_card = [shuffled[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_replicas)]
                yield per_card[self.rank]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    dataset = Image3DDataset(config.data_dir)
    # 分布式 K-repeat 采样器（与 SD3/Hunyuan3D 对齐）
    batch_size = int(config.sample.input_batch_size)
    k = int(config.sample.num_meshes_per_image)
    sampler = DistributedImageRepeatSampler(
        dataset,
        batch_size=batch_size,
        k=k,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=int(config.seed),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=Image3DDataset.collate_fn,
    )
    return loader


def build_optimizer(params, config: ml_collections.ConfigDict):
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.adam_weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            params,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.adam_weight_decay,
        )
    return optimizer


def compute_advantages(rewards: np.ndarray, stat_tracker: PerImageStatTracker, image_names: List[str], use_global_std: bool) -> np.ndarray:
    """将字符串图像名映射为整数ID，并计算标准化优势。

    与 PerImageStatTracker 接口对齐：update(image_ids, rewards)
    """
    if stat_tracker is None:
        # 纯全局标准化：A = (r - mean) / (std + eps)
        rewards_arr = np.array(rewards, dtype=np.float64)
        mean = float(rewards_arr.mean())
        std = float(rewards_arr.std())
        eps = 1e-8
        advantages = (rewards_arr - mean) / (std + eps)
        return advantages.astype(np.float64)

    # 根据配置同步归一化策略
    stat_tracker.global_std = bool(use_global_std)

    # 将图像名映射为稳定的整数ID
    unique_names = list(set(image_names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    image_ids = np.array([name_to_id[name] for name in image_names], dtype=np.int64)

    # 计算优势（PerImageStatTracker 内部完成标准化）
    advantages = stat_tracker.update(image_ids, np.array(rewards, dtype=np.float64))
    return advantages


def eval_trellis(
    pipeline: TrellisStage2Pipeline,
    test_dataloader: DataLoader,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    mesh_scorer: MeshScorer,
):
    """Trellis 评估流程（对齐 SD3/Hunyuan3D 的评估风格）"""
    model = pipeline.get_trainable_model()
    model.eval()

    all_rewards: Dict[str, List[np.ndarray]] = defaultdict(list)

    for eval_batch in tqdm(
        test_dataloader,
        desc="Eval:",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        images, image_paths, metadata = eval_batch

        # 编码图像条件（与训练一致）
        cond_dict = pipeline.prepare_image_conditions(images)  # {'cond': (B, P, C), 'neg_cond': (B,P,C)

        # 生成mesh（评估每图只生成1个候选）
        meshes, _, _, _ = trellis_stage2_with_logprob(
            pipeline=pipeline,
            num_inference_steps=int(getattr(config.sample, 'eval_num_steps', config.sample.num_steps)),
            guidance_scale=float(config.sample.guidance_scale),
            kl_reward=0.0,
            deterministic=bool(config.deterministic),
            sparse_structure_sampler_params=dict(config.sparse_structure_sampler_params),
            slat_sampler_params=dict(config.slat_sampler_params),
            stage1_cond_dict=cond_dict,
            num_candidates=1,
            output_type="kiui",
        )

        meshes = [m.to(accelerator.device) for m in meshes]

        # 计算奖励并聚合到主进程
        rewards_dict, _ = mesh_scorer.score(meshes, images, metadata, dict(config.reward_fn))
        for key, value in rewards_dict.items():
            gathered = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(gathered)

    # 聚合并记录
    all_rewards_np = {key: (np.concatenate(v) if len(v) > 0 else np.array([])) for key, v in all_rewards.items()}
    if accelerator.is_main_process:
        metrics = {f"eval_reward_{k}": (float(np.mean(val)) if val.size > 0 else 0.0) for k, val in all_rewards_np.items()}
        accelerator.log(metrics, step=epoch + 1)


def repeat_image_conds(cond: torch.Tensor, k: int) -> torch.Tensor:
    # cond: (B, C) -> (B*k, C) 用于生成多个 candidates
    B, C = cond.shape  # (B, C)
    cond_expanded = cond.unsqueeze(1).expand(B, k, C).reshape(B * k, C)  # (B*k, C)
    return cond_expanded


def main(_):
    config: ml_collections.ConfigDict = _CONFIG.value

    # 训练时间步数量（与 SD3/Hunyuan3D 对齐，用于放大梯度累积步数）
    num_train_timesteps = int(float(config.sample.num_steps) * float(getattr(config.train, 'timestep_fraction', 1.0)))  # 标量

    # 基础加速器（将梯度累积步数乘以时间步数，对齐 SD3/Hunyuan3D）
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=config.logdir),
        log_with=["wandb"],
        gradient_accumulation_steps=int(config.train.gradient_accumulation_steps) * max(1, num_train_timesteps),  # 标量
    )
    set_seed(config.seed)

    if accelerator.is_main_process:
        import wandb
        run_name = config.run_name if len(config.run_name) > 0 else f"trellis_stage2_{int(time.time())}"
        wandb.init(project="flow-grpo-trellis", name=run_name, config=dict(config))

    # 构建 Pipeline 并移动到设备
    pipeline = TrellisStage2Pipeline(model_path=config.pretrained.model)
    device = accelerator.device
    pipeline.to(device)
    if device.type == 'cuda':
        # 强制将 TRELLIS 内部模块切到 GPU，避免 CPU 运行过慢
        pipeline.cuda()

    # 仅训练 Stage 2 (SLatFlowModel) + LoRA 对齐 Hunyuan3D
    slat_model: nn.Module = pipeline.get_trainable_model()
    if bool(getattr(config, 'use_lora', False)):
        # 注册SparseLinear的LoRA支持
        register_sparse_linear_with_peft()
        # 依据 TRELLIS SLatFlowModel 实际命名：
        # - 注意力层: SparseMultiHeadAttention 内部为 `to_qkv`(self), `to_q`/`to_kv`(cross), `to_out`
        # - 前馈层: SparseFeedForwardNet 的顺序容器 `mlp` 下标 0/2 为线性层
        target_modules = [
            "to_qkv",   # self-attn linear (Tensor input via _linear)
            "to_q",     # cross-attn q
            "to_kv",    # cross-attn kv
            "to_out",   # attn out proj
        ]
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        lora_path = getattr(config.train, 'lora_path', None)
        if isinstance(lora_path, str) and len(lora_path) > 0:
            slat_model = PeftModel.from_pretrained(slat_model, lora_path)
            slat_model.set_adapter("default")
        else:
            slat_model = get_peft_model(slat_model, lora_cfg)

    # 仅优化可训练参数（LoRA时仅适配器参数）
    trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, config)
    slat_model, optimizer = accelerator.prepare(slat_model, optimizer)
    # 回写到 pipeline，确保采样/重算均使用LoRA包装后的模型
    if hasattr(pipeline, 'core_pipeline') and hasattr(pipeline.core_pipeline, 'models'):
        if 'slat_flow_model' in pipeline.core_pipeline.models:
            pipeline.core_pipeline.models['slat_flow_model'] = slat_model

    # 恢复断点（最小实现）
    if isinstance(config.resume_from, str) and len(config.resume_from) > 0:
        ckpt_path = Path(config.resume_from) / "pytorch_model.bin"
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location="cpu")
            slat_model.load_state_dict(state.get("model", state))
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            accelerator.print(f"🔁 Resumed from {str(ckpt_path)}")

    # EMA（可选）
    ema = None
    if bool(getattr(config.train, 'ema', False)):
        ema_decay = float(getattr(config.train, 'ema_decay', 0.999))
        ema = EMAModuleWrapper(trainable_params, decay=ema_decay, device=accelerator.device)

    # 数据与奖励
    train_loader = dataloader_from_config(config, accelerator)
    mesh_scorer = MeshScorer(device=device)

    # 按配置启用/禁用按图像统计
    stat_tracker = PerImageStatTracker(global_std=config.sample.global_std) if bool(getattr(config, 'per_image_stat_tracking', True)) else None

    gradient_accumulation_steps = int(config.train.gradient_accumulation_steps)

    # 缓存最近一批用于可视化的对象
    last_meshes = None
    last_image_paths = None
    last_rewards = None

    for epoch in range(config.num_epochs):
        # 采样阶段：对每张图像生成 K 个候选并打分
        all_samples = []
        accelerator.print(f"[Epoch {epoch}] Sampling...")
        max_train_batches = int(getattr(config.sample, 'num_batches_per_epoch', 0))
        processed_batches = 0
        for batch_idx, (batch_images, batch_paths, batch_meta) in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process)):
            # 设置 epoch*inner_idx 以同步所有卡的采样（与 SD3/Hunyuan3D 对齐）
            if hasattr(train_loader.batch_sampler, 'set_epoch'):
                train_loader.batch_sampler.set_epoch(epoch * max(1, int(getattr(config.sample, 'num_batches_per_epoch', 1))) + batch_idx)
            # 提取图像条件（DINOv2 / 官方接口）
            cond_dict = pipeline.prepare_image_conditions(batch_images)  # {'cond': (B, P, C), 'neg_cond': (B,P,C)}
            # 简化：将 patch 维度聚合成一个向量，得到 (B, C)
            pos = cond_dict['cond'].mean(dim=1)
            neg = cond_dict['neg_cond'].mean(dim=1)

            # 重复条件，生成每图 K 个候选
            k = int(config.sample.num_meshes_per_image)
            pos_rep = repeat_image_conds(pos, k)
            neg_rep = repeat_image_conds(neg, k) if config.sample.guidance_scale > 1.0 else None

            image_conds = {"positive": pos_rep}
            if neg_rep is not None:
                image_conds["negative"] = neg_rep

            meshes, all_latents, all_log_probs, all_kl = trellis_stage2_with_logprob(
                pipeline=pipeline,
                num_inference_steps=int(config.sample.num_steps),
                guidance_scale=float(config.sample.guidance_scale),
                kl_reward=float(config.sample.kl_reward),
                deterministic=bool(config.deterministic),
                sparse_structure_sampler_params=dict(config.sparse_structure_sampler_params),
                slat_sampler_params=dict(config.slat_sampler_params),
                stage1_cond_dict=cond_dict,
                num_candidates=int(config.sample.num_meshes_per_image),
                output_type="kiui",
            )

            # 将 mesh 迁移到与 scorer 相同设备，避免 CPU/GPU 混用
            meshes = [m.to(device) for m in meshes]

            # 打分
            rewards_dict, _ = mesh_scorer.score(meshes, batch_images * k, batch_meta * k, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # np.ndarray

            # 缓存用于周期末保存/可视化
            last_meshes = meshes
            last_image_paths = batch_paths
            last_rewards = rewards

            # 记录样本条目（逐样本切片 log_prob/latent），并构造 step 索引用于后续时间维随机打乱
            steps = int(config.sample.num_steps)
            for s in range(len(meshes)):
                # per-sample latent/logprob 切片
                latent_start = s * (steps + 1)
                latent_end = latent_start + (steps + 1)
                logprob_start = s * steps
                logprob_end = logprob_start + steps

                final_latent = all_latents[latent_end - 1]
                # 保存整条时序（单步重算所需）
                latents_seq = all_latents[latent_start:latent_end]  # 长度 steps+1
                sample_log_probs = all_log_probs[logprob_start:logprob_end]  # 长度 steps
                old_log_probs = torch.stack(sample_log_probs)  # 形状 [steps]

                # 对齐采样器的时间序列（与 trellis_flow_euler_sampler_with_logprob 一致）
                t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000
                t_seq = float(config.slat_sampler_params.rescale_t) * t_seq / (1 + (float(config.slat_sampler_params.rescale_t) - 1) * t_seq / 1000)

                # 保存对应的条件（用于训练期重算）
                pos_cond_s = pos_rep[s:s+1]
                neg_cond_s = neg_rep[s:s+1] if neg_rep is not None else None

                # 同步保存 patch 级 cond/neg_cond（官方接口）
                cond_patches_s = cond_dict['cond'][s//k:s//k+1]
                neg_patches_s = cond_dict['neg_cond'][s//k:s//k+1]

                all_samples.append({
                    "coords": final_latent.coords,     # (N,4)
                    "slat": final_latent,             # SparseTensor
                    "image_idx": 0,                   # 重算时索引（与image_conds对齐，传入单样本条件）
                    "latents_seq": latents_seq,      # [steps+1] SparseTensor 序列
                    "old_log_probs": old_log_probs,  # [steps] 采样期每步 log_prob
                    "t_seq": t_seq,                  # [steps+1] 时间序列（numpy数组）
                    "reward": float(rewards[s]),     # 标量
                    "image_name": batch_meta[s // k]["image_name"],
                    "pos_cond": pos_cond_s,
                    "neg_cond": neg_cond_s,
                    "cond_patches": cond_patches_s,
                    "neg_patches": neg_patches_s,
                    "time_indices": np.arange(steps, dtype=int),  # (steps,)
                })

            # 可视化与落盘（仅主进程；按频率；只对首个batch执行，避免过多IO）
            save_visualizations = bool(getattr(config, 'save_visualizations', False))
            mesh_save_freq = int(getattr(config, 'mesh_save_freq', int(config.save_freq)))
            if save_visualizations and accelerator.is_main_process and ((epoch + 1) % mesh_save_freq == 0) and batch_idx == 0:
                run_name_dir = config.run_name if isinstance(config.run_name, str) and len(config.run_name) > 0 else "trellis_stage2"
                mesh_save_dir = os.path.join(config.logdir, run_name_dir, "generated_meshes", f"epoch_{epoch+1}")

                # 展开 batch 的 image_paths（每图重复 K 次），与 meshes 对齐
                repeated_paths = []
                for p in batch_paths:
                    repeated_paths.extend([p] * k)

                # 仅保存前若干样本，避免IO过大
                num_samples_to_save = min(2, len(meshes))
                sampled_meshes = meshes[:num_samples_to_save]
                sampled_paths = repeated_paths[:num_samples_to_save]
                sampled_rewards = rewards[:num_samples_to_save]

                device_str = "cuda" if device.type == 'cuda' else "cpu"
                mesh_files, preview_files = save_meshes_for_preview(
                    sampled_meshes,
                    sampled_paths,
                    sampled_rewards,
                    epoch + 1,
                    mesh_save_dir,
                    device_str,
                )
                accelerator.print(f"✅ 已保存 {len(mesh_files)} 个mesh与预览到 {mesh_save_dir}")

            processed_batches += 1
            if max_train_batches > 0 and processed_batches >= max_train_batches:
                break

        # 统计与优势（与 Hunyuan3D 一致：分布式聚合后按图像标准化）
        image_names = [s["image_name"] for s in all_samples]  # (N,)
        rewards_np_local = np.array([s["reward"] for s in all_samples], dtype=np.float64)  # (N,)
        advantages_local_np = compute_advantages_per_image(
            image_names=image_names,
            rewards_np_local=rewards_np_local,
            accelerator=accelerator,
            stat_tracker=stat_tracker,
            epoch=epoch,
        )  # (N,)

        # ===== 先拼后切：构造“字典的张量” =====
        # 聚合条件（patch 级）
        pos_cond_batched = torch.cat([s["cond_patches"] for s in all_samples], dim=0)  # (N, P, C)
        neg_cond_batched = torch.cat([s["neg_patches"] for s in all_samples], dim=0)  # (N, P, C)

        # 旧 log_prob 与优势（时间维复制）
        old_log_probs = torch.stack([s["old_log_probs"] for s in all_samples], dim=0).to(accelerator.device)  # (N, steps)
        steps = int(config.sample.num_steps)  # 标量
        adv_vec = torch.from_numpy(advantages_local_np).to(accelerator.device, dtype=torch.float32)  # (N,)
        advantages = adv_vec.unsqueeze(1).expand(-1, steps)  # (N, steps)

        # 稀疏时序（保持为列表，后续按子批按步拼 batched SparseTensor）
        latents_seq_list = [s["latents_seq"] for s in all_samples]  # 长度 N, 每项长度 steps+1

        # 可选：时间步矩阵（当前不用于前向，便于形状对齐与重组接口一致）
        t_seq = all_samples[0]["t_seq"] if len(all_samples) > 0 else np.linspace(1.0, 0.0, steps + 1) * 1000
        timesteps = torch.as_tensor(t_seq[:-1], device=accelerator.device, dtype=torch.float32).unsqueeze(0).expand(len(all_samples), -1)  # (N, steps)

        # 过滤：移除优势全零的样本，并确保能被 num_batches_per_epoch 整除
        num_batches_epoch = int(getattr(config.sample, 'num_batches_per_epoch', 1))
        mask = (advantages.abs().sum(dim=1) != 0)  # (N,)
        true_count = int(mask.sum().item())
        if true_count % max(1, num_batches_epoch) != 0:
            false_indices = torch.where(~mask)[0]
            num_to_flip = (max(1, num_batches_epoch) - (true_count % max(1, num_batches_epoch))) % max(1, num_batches_epoch)
            if false_indices.numel() >= num_to_flip and num_to_flip > 0:
                flip_idx = torch.randperm(false_indices.numel(), device=accelerator.device)[:num_to_flip]
                mask[false_indices[flip_idx]] = True

        # 应用过滤到所有键
        idx_keep = torch.where(mask)[0].tolist()
        pos_cond_batched = pos_cond_batched[idx_keep]
        neg_cond_batched = neg_cond_batched[idx_keep]
        old_log_probs = old_log_probs[idx_keep]
        advantages = advantages[idx_keep]
        timesteps = timesteps[idx_keep]
        latents_seq_list = [latents_seq_list[i] for i in idx_keep]

        # 有效比例日志
        valid_samples_ratio = float(len(idx_keep) / max(1, len(all_samples))) if len(all_samples) > 0 else 0.0

        # Step-1: 沿 batch 维随机打乱（与 Hunyuan3D 一致）
        total_batch_size = old_log_probs.shape[0]
        g = torch.Generator(device=accelerator.device)
        g.manual_seed(int(config.seed) + int(epoch))
        perm = torch.randperm(total_batch_size, generator=g, device=accelerator.device)
        pos_cond_batched = {"cond": pos_cond_batched[perm]}
        neg_cond_batched = {"neg_cond": neg_cond_batched[perm]}
        old_log_probs = old_log_probs[perm]
        advantages = advantages[perm]
        timesteps = timesteps[perm]
        latents_seq_list = [latents_seq_list[i] for i in perm.cpu().tolist()]

        # Step-2: 沿时间维独立打乱（当前保持恒等序列）
        num_timesteps = old_log_probs.shape[1]
        perms = torch.stack([torch.arange(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])  # (B, T)
        for key_name, tensor_ref in [("timesteps", timesteps), ("old_log_probs", old_log_probs), ("advantages", advantages)]:
            tensor_src = tensor_ref
            tensor_ref = tensor_src[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            if key_name == "timesteps":
                timesteps = tensor_ref
            elif key_name == "old_log_probs":
                old_log_probs = tensor_ref
            elif key_name == "advantages":
                advantages = tensor_ref

        # Step-3: 等分 chunk 为子批（“张量视角”的字典）
        assert total_batch_size % max(1, num_batches_epoch) == 0, "内部约束失败：样本数需可整除 num_batches_per_epoch"
        chunk_size = total_batch_size // max(1, num_batches_epoch)

        samples_batched = []
        for i in range(num_batches_epoch):
            start = i * chunk_size
            end = start + chunk_size
            sub_dict = {
                "positive_image_cond": {k: v[start:end] for k, v in pos_cond_batched.items()},
                "negative_image_cond": {k: v[start:end] for k, v in neg_cond_batched.items()},
                "old_log_probs": old_log_probs[start:end],
                "advantages": advantages[start:end],
                "timesteps": timesteps[start:end],
                # 稀疏时序使用列表切片
                "latents_seq": latents_seq_list[start:end],
            }
            samples_batched.append(sub_dict)

        accelerator.log({
            "actual_batch_size": chunk_size,
            "num_sub_batches": len(samples_batched),
            "valid_samples_ratio": float(valid_samples_ratio),
        }, step=epoch)

        # ===== 训练阶段：外层 inner-epoch × 内层子批循环 × 时间步循环（对齐 Hunyuan3D） =====
        accelerator.print(f"[Epoch {epoch}] Training...")
        slat_model.train()

        global_step = 0
        steps_to_train = max(1, int(float(getattr(config.train, 'timestep_fraction', 1.0)) * steps))  # 标量
        import numpy as _np_idx
        train_step_indices = _np_idx.linspace(0, steps - 1, steps_to_train, dtype=int)  # 形状 (steps_to_train,)
        autocast_ctx = accelerator.autocast

        num_inner_epochs = int(getattr(config.train, 'num_inner_epochs', 1))  # 标量
        for inner_epoch in range(num_inner_epochs):
            # 每个 inner-epoch 打乱子批顺序（列表级），增强训练随机性
            if len(samples_batched) > 1:
                g_local = torch.Generator(device=accelerator.device)
                g_local.manual_seed(int(config.seed) + int(epoch) * 9973 + int(inner_epoch) * 101)  # 标量
                perm_list = torch.randperm(len(samples_batched), generator=g_local, device=accelerator.device).cpu().tolist()  # 形状 (num_sub_batches,)
                samples_batched_shuffled = [samples_batched[i] for i in perm_list]
            else:
                samples_batched_shuffled = samples_batched

            for batch_idx_sub, sample in enumerate(tqdm(samples_batched_shuffled, disable=not accelerator.is_main_process)):
                # 将 batched 条件拆成 per-sample 列表，与稀疏时序一一对应
                B_sub = sample["old_log_probs"].shape[0]  # 标量
                image_conds_list = []  # 长度 B_sub
                for bi in range(B_sub):
                    image_conds_list.append({
                        "cond": sample["positive_image_cond"]["cond"][bi:bi+1],  # 形状 (1, P, C)
                        "neg_cond": sample["negative_image_cond"]["neg_cond"][bi:bi+1],  # 形状 (1, P, C)
                    })

                # 构造与 compute_log_prob_trellis_stage2_batched 接口一致的样本列表（仅包含稀疏时序）
                batch_samples_list = [
                    {"latents_seq": sample["latents_seq"][bi], "t_seq": t_seq} for bi in range(B_sub)
                ]  # 长度 B_sub

                for j in train_step_indices:
                    j = int(j)  # 标量
                    with accelerator.accumulate(slat_model):
                        with autocast_ctx():
                            # SD3 风格：直接小批量前向，通过配置控制显存而非微分批
                            log_prob_vec, kl_vec = compute_log_prob_trellis_stage2_batched(
                                pipeline=pipeline,
                                samples=batch_samples_list,
                                j=j,
                                image_conds_list=image_conds_list,
                                config=ml_collections.FrozenConfigDict({
                                    "guidance_scale": float(config.sample.guidance_scale),
                                    "num_inference_steps": int(config.sample.num_steps),
                                    "sigma_min": float(config.slat_sampler_params.sigma_min),
                                    "rescale_t": float(config.slat_sampler_params.rescale_t),
                                    "deterministic": bool(config.deterministic),
                                    "kl_reward": float(config.sample.kl_reward),
                                }),
                            )  # log_prob_vec: (B_sub,), kl_vec: (B_sub,)

                        adv_vec = torch.clamp(sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max)  # 形状 (B_sub,)
                        old_lp_vec = sample["old_log_probs"][:, j]  # 形状 (B_sub,)
                        ratio_vec = torch.exp(log_prob_vec - old_lp_vec)  # 形状 (B_sub,)
                        unclipped = -adv_vec * ratio_vec  # 形状 (B_sub,)
                        clipped = -adv_vec * torch.clamp(ratio_vec, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)  # 形状 (B_sub,)
                        loss_vec = torch.maximum(unclipped, clipped)  # 形状 (B_sub,)
                        if float(getattr(config.train, 'beta', 0.0)) > 0.0:
                            loss_vec = loss_vec + float(config.train.beta) * kl_vec  # 形状 (B_sub,)
                        loss = loss_vec.mean()  # 标量 ()

                        accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(slat_model.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1  # 标量
                    if bool(getattr(config.train, 'ema', False)) and ema is not None:
                        ema.step([p for p in slat_model.parameters() if p.requires_grad], global_step)

        # 评估节奏对齐：每个 epoch 的首个采样步已触发可视化；此处按“每 epoch 首批”统一触发评估
        if accelerator.is_main_process and int(getattr(config, 'eval_freq', 1)) > 0 and ((epoch + 1) % int(config.eval_freq) == 0):
            eval_bs = int(getattr(config.sample, 'test_batch_size', 1))
            eval_loader = DataLoader(
                Image3DDataset(config.data_dir),
                batch_size=eval_bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
                collate_fn=Image3DDataset.collate_fn,
            )
            # 使用 EMA 权重评估（如启用）
            if bool(getattr(config.train, 'ema', False)) and ema is not None:
                trainable = [p for p in slat_model.parameters() if p.requires_grad]
                ema.copy_ema_to(trainable, store_temp=True)
                eval_trellis(pipeline, eval_loader, config, accelerator, epoch, mesh_scorer)
                ema.copy_temp_to(trainable)
            else:
                eval_trellis(pipeline, eval_loader, config, accelerator, epoch, mesh_scorer)

        # 保存节奏对齐：每 epoch 末保存（频率由 save_freq 控制）
        if accelerator.is_main_process and (epoch + 1) % int(config.save_freq) == 0:
            save_dir = Path(config.save_dir) / f"checkpoint_{epoch+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            if bool(getattr(config, 'use_lora', False)):
                lora_dir = save_dir / "lora"
                lora_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(slat_model)
                unwrapped.save_pretrained(str(lora_dir))
                # 仍保存优化器与配置，便于继续训练
                torch.save(optimizer.state_dict(), str(save_dir / "optimizer.bin"))
                torch.save(dict(config), str(save_dir / "config.bin"))
                accelerator.print(f"💾 Saved LoRA adapter: {str(lora_dir)}")
            else:
                # 保存最小必要状态（整模）
                to_save = {
                    "model": accelerator.get_state_dict(slat_model),
                    "optimizer": optimizer.state_dict(),
                    "config": dict(config),
                }
                # 附加 EMA 权重（如启用）
                if bool(getattr(config.train, 'ema', False)) and ema is not None:
                    to_save["ema_state"] = ema.state_dict()
                torch.save(to_save, str(save_dir / "pytorch_model.bin"))
                accelerator.print(f"💾 Saved: {str(save_dir)}")


            # 保存可视化（与保存频率一致）
            if last_meshes is not None and last_image_paths is not None and last_rewards is not None:
                viz_dir = Path(config.logdir) / config.run_name / "generated_meshes" / f"epoch_{epoch+1}"
                viz_dir.mkdir(parents=True, exist_ok=True)

                num_samples = min(2, len(last_meshes))
                mesh_files, preview_files = save_meshes_for_preview(
                    last_meshes[:num_samples],
                    last_image_paths[:num_samples],
                    last_rewards[:num_samples],
                    epoch + 1,
                    str(viz_dir),
                    device_str=device.type,
                )

                # 上传预览到 W&B（仅主进程）
                import wandb
                accelerator.log(
                    {
                        "mesh_previews": [
                            wandb.Image(preview_files[i], caption=os.path.basename(last_image_paths[i]))
                            for i in range(len(preview_files))
                        ]
                    },
                    step=epoch + 1,
                )


if __name__ == "__main__":
    app.run(main) 