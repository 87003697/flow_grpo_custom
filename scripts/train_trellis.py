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
from PIL import Image

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
    """将字符串图像名映射为整数ID，并计算标准化优势。

    与 PerImageStatTracker 接口对齐：update(image_ids, rewards)
    """
    # 根据配置同步归一化策略
    stat_tracker.global_std = bool(use_global_std)

    # 将图像名映射为稳定的整数ID
    unique_names = list(set(image_names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    image_ids = np.array([name_to_id[name] for name in image_names], dtype=np.int64)

    # 计算优势（PerImageStatTracker 内部完成标准化）
    advantages = stat_tracker.update(image_ids, np.array(rewards, dtype=np.float64))
    return advantages


def save_meshes_for_wandb(meshes, image_paths, rewards, epoch, tmpdir, device="cuda"):
    """保存mesh并生成预览图 - 文件名基于原始图像名称

    说明:
    - meshes: List[KiuiMesh]（来自 `trellis_stage2_with_logprob(..., output_type="kiui")`）
    - image_paths: List[str]
    - rewards: np.ndarray 或 List[float]
    - epoch: int，用于记录/命名
    - tmpdir: 保存目录
    """
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_for_training
    import os

    mesh_files = []
    preview_files = []

    for idx, (mesh, img_path, reward) in enumerate(zip(meshes, image_paths, rewards)):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        image_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        mesh_path = os.path.join(tmpdir, f"{image_name}_mesh_{idx}.obj")
        mesh.write(mesh_path)

        preview_path = os.path.join(tmpdir, f"{image_name}_preview_{idx}.png")
        render_mesh_for_training(mesh_path, preview_path, device=device)
        print(f"💾 渲染已保存: {preview_path}")

        mesh_files.append(mesh_path)
        preview_files.append(preview_path)

    return mesh_files, preview_files


def repeat_image_conds(cond: torch.Tensor, k: int) -> torch.Tensor:
    # cond: (B, C) -> (B*k, C) 用于生成多个 candidates
    B, C = cond.shape  # (B, C)
    cond_expanded = cond.unsqueeze(1).expand(B, k, C).reshape(B * k, C)  # (B*k, C)
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
    if device.type == 'cuda':
        # 强制将 TRELLIS 内部模块切到 GPU，避免 CPU 运行过慢
        pipeline.cuda()

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

    # 缓存最近一批用于可视化的对象
    last_meshes = None
    last_image_paths = None
    last_rewards = None

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

            # 将 mesh 迁移到与 scorer 相同设备，避免 CPU/GPU 混用
            meshes = [m.to(device) for m in meshes]

            # 打分
            rewards_dict, _ = mesh_scorer.score(meshes, batch_images * k, batch_meta * k, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # np.ndarray

            # 缓存用于周期末保存/可视化
            last_meshes = meshes
            last_image_paths = batch_paths
            last_rewards = rewards

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

            # 保存可视化（与保存频率一致）
            if last_meshes is not None and last_image_paths is not None and last_rewards is not None:
                viz_dir = Path(config.logdir) / config.run_name / "generated_meshes" / f"epoch_{epoch+1}"
                viz_dir.mkdir(parents=True, exist_ok=True)

                num_samples = min(2, len(last_meshes))
                mesh_files, preview_files = save_meshes_for_wandb(
                    last_meshes[:num_samples],
                    last_image_paths[:num_samples],
                    last_rewards[:num_samples],
                    epoch + 1,
                    str(viz_dir),
                    device=device.type,
                )

                # 上传预览到 W&B（仅主进程）
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