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
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

import ml_collections
from absl import app
from ml_collections import config_flags

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS/GRPO 相关模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from flow_grpo.diffusers_patch.trellis_stage2_with_logprob import trellis_stage2_with_logprob
from flow_grpo.diffusers_patch.sparse_tensor_grpo import compute_log_prob_trellis_stage2
from flow_grpo.stat_tracking import PerImageStatTracker
from reward_models.rewards_mesh import MeshScorer
from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
from generators.trellis.utils import convert_trellis_to_trimesh

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config")


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
        from PIL import Image
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


def dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    dataset = Image3DDataset(config.data_dir)
    # 简化：单机单进程 DataLoader；多机场景建议定制分布式重复采样器
    loader = DataLoader(
        dataset,
        batch_size=config.sample.input_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
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
    advantages = stat_tracker.update(np.array(image_names), np.array(rewards), type='grpo')
    return advantages


def repeat_image_conds(cond: torch.Tensor, k: int) -> torch.Tensor:
    # cond: (B, C) -> (B*k, C) 用于生成多个 candidates
    B, C = cond.shape
    cond_expanded = cond.unsqueeze(1).expand(B, k, C).reshape(B * k, C)
    return cond_expanded


def main(_):
    config: ml_collections.ConfigDict = _CONFIG.value

    # 基础加速器
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=config.logdir),
        log_with=["wandb"],
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

    # 仅训练 Stage 2 (SLatFlowModel)
    slat_model: nn.Module = pipeline.get_trainable_model()
    optimizer = build_optimizer(slat_model.parameters(), config)
    slat_model, optimizer = accelerator.prepare(slat_model, optimizer)

    # 恢复断点（最小实现）
    if isinstance(config.resume_from, str) and len(config.resume_from) > 0:
        ckpt_path = Path(config.resume_from) / "pytorch_model.bin"
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location="cpu")
            slat_model.load_state_dict(state.get("model", state))
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            accelerator.print(f"🔁 Resumed from {str(ckpt_path)}")

    # 数据与奖励
    train_loader = dataloader_from_config(config, accelerator)
    mesh_scorer = MeshScorer(device=device)

    stat_tracker = PerImageStatTracker(global_std=config.sample.global_std)

    gradient_accumulation_steps = int(config.train.gradient_accumulation_steps)

    for epoch in range(config.num_epochs):
        # 采样阶段：对每张图像生成 K 个候选并打分
        all_samples = []
        accelerator.print(f"[Epoch {epoch}] Sampling...")
        for batch_images, batch_paths, batch_meta in tqdm(train_loader, disable=not accelerator.is_main_process):
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
                output_type="kiui",
            )

            # 打分
            rewards_dict, _ = mesh_scorer.score(meshes, batch_images * k, batch_meta * k, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # np.ndarray

            # 记录样本条目（逐样本切片 log_prob/latent）
            steps = int(config.sample.num_steps)
            for s in range(len(meshes)):
                # per-sample latent/logprob 切片
                latent_start = s * (steps + 1)
                latent_end = latent_start + (steps + 1)
                logprob_start = s * steps
                logprob_end = logprob_start + steps

                final_latent = all_latents[latent_end - 1]
                sample_log_probs = all_log_probs[logprob_start:logprob_end]
                old_log_prob = torch.stack(sample_log_probs).sum()

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
                    "old_log_prob": old_log_prob,    # 采样期总 log_prob
                    "reward": float(rewards[s]),     # 标量
                    "image_name": batch_meta[s // k]["image_name"],
                    "pos_cond": pos_cond_s,
                    "neg_cond": neg_cond_s,
                    "cond_patches": cond_patches_s,
                    "neg_patches": neg_patches_s,
                })

        # 统计与优势
        image_names = [s["image_name"] for s in all_samples]
        rewards_np = np.array([s["reward"] for s in all_samples], dtype=np.float32)
        advantages_np = compute_advantages(rewards_np, stat_tracker, image_names, use_global_std=config.sample.global_std)

        # 训练阶段：按 GRPO 重算 log_prob，计算 ratio/clip 损失
        accelerator.print(f"[Epoch {epoch}] Training...")
        slat_model.train()

        # 将 advantages 写回样本
        for idx, s in enumerate(all_samples):
            s["advantages"] = torch.tensor(advantages_np[idx], device=device, dtype=torch.float32)

        # 简化：单批遍历+梯度累积
        global_step = 0
        for i, sample in enumerate(tqdm(all_samples, disable=not accelerator.is_main_process)):
            with accelerator.accumulate(slat_model):
                # 准备图像条件（按样本索引）
                # 注意：训练期重算按 step j 遍历；此处将 LogProb 总和作为示例，方便与 sd3/hunyuan3d 对齐
                # 更精细可逐步 j 重算并对应保存的 step-wise log_probs
                image_conds_train = {
                    "cond": sample["cond_patches"],
                    "neg_cond": sample["neg_patches"],
                }

                # 重算 log_prob（聚合）
                final_slat, log_prob, kl_div = compute_log_prob_trellis_stage2(
                    pipeline=pipeline,
                    sample=sample,
                    j=0,
                    image_conds=image_conds_train,
                    config=ml_collections.FrozenConfigDict({
                        "guidance_scale": float(config.sample.guidance_scale),
                        "num_inference_steps": int(config.sample.num_steps),
                        "sigma_min": float(config.slat_sampler_params.sigma_min),
                        "rescale_t": float(config.slat_sampler_params.rescale_t),
                        "deterministic": bool(config.deterministic),
                        "kl_reward": float(config.sample.kl_reward),
                    }),
                )

                # 计算 GRPO policy loss（对齐 sd3/hunyuan3d 的公式）
                advantages = torch.clamp(sample["advantages"], -config.train.adv_clip_max, config.train.adv_clip_max)
                old_log_prob = sample["old_log_prob"].to(log_prob.device)
                ratio = torch.exp(log_prob - old_log_prob)
                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                policy_loss = torch.mean(torch.maximum(unclipped, clipped))

                # KL loss（可选）
                kl_coeff = float(config.train.beta)
                if kl_coeff > 0:
                    policy_loss = policy_loss + kl_coeff * kl_div

                accelerator.backward(policy_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(slat_model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

        if accelerator.is_main_process and (epoch + 1) % int(config.save_freq) == 0:
            save_dir = Path(config.save_dir) / f"checkpoint_{epoch+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # 保存最小必要状态
            to_save = {
                "model": accelerator.get_state_dict(slat_model),
                "optimizer": optimizer.state_dict(),
                "config": dict(config),
            }
            torch.save(to_save, str(save_dir / "pytorch_model.bin"))
            accelerator.print(f"💾 Saved: {str(save_dir)}")


if __name__ == "__main__":
    app.run(main) 