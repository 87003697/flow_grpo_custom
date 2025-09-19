#!/usr/bin/env python3
"""Direct3D‑S2 GRPO Training (Stage2 sparse512 with log_prob)

最小骨架：
 - 复用 Direct3D‑S2 pipeline wrapper (`Direct3DS2PipelineWithLogProb`)
 - 借鉴 `scripts/train_trellis.py` / `train_hunyuan3d.py` 的结构
 - 仅实现：构建模型 + 采样接口 + LoRA 注入 + 简化主循环 (占位)；
   后续可逐步补充完整 GRPO（优势/ratio/logprob 对齐）。

约束 (参考 DEV.md)：
 - 仅训练 sparse_dit_512 (LoRA)
 - 无 try/except；行内张量注释
"""
import os, sys, math, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import ml_collections
from absl import app
from ml_collections import config_flags

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import Direct3DS2PipelineWithLogProb
from reward_models.rewards_mesh import MeshScorer
from flow_grpo.stat_tracking import PerImageStatTracker
from peft import LoraConfig, get_peft_model
from flow_grpo.peft_sparse.sparse_lora import register_sparse_linear_with_peft

logger = get_logger(__name__)
_CONFIG = config_flags.DEFINE_config_file("config")


def build_pipeline_and_models(cfg: ml_collections.ConfigDict, accelerator: Accelerator):
    """构建 Direct3D‑S2 pipeline 与奖励模型。
    返回：pipeline, mesh_scorer
    参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:68-172`（from_pretrained）与 `:54-66`（to）
    """
    pipeline = Direct3DS2PipelineWithLogProb.from_pretrained(
        cfg.pretrained.pipeline_path,
        subfolder=cfg.pretrained.subfolder,
        dtype=torch.float16 if cfg.mixed_precision in ["fp16", "bf16"] else torch.float32,
        minimal_512_only=bool(cfg.pretrained.minimal_512_only),
    )
    pipeline.to(str(accelerator.device))

    # 注册稀疏层以支持 LoRA
    if cfg.use_lora:
        register_sparse_linear_with_peft()
        # 找到 sparse_dit_512
        target = pipeline.ref.sparse_dit_512
        # Direct3D MultiHeadAttention layer names: to_qkv (self), to_q (cross), to_kv (cross), to_out
        lora_cfg = LoraConfig(
            r=int(cfg.lora.lora_rank),
            lora_alpha=int(cfg.lora.lora_rank),
            target_modules=["to_qkv", "to_q", "to_kv", "to_out"],
            lora_dropout=0.0,
            bias="none",
        )
        target = get_peft_model(target, lora_cfg)
        # 记录命中层
        if accelerator.is_main_process:
            matched = []
            for name, module in target.named_modules():
                if any(name.endswith(x) for x in lora_cfg.target_modules):
                    # 检查是否被 LoRA 包装（存在 lora_A 属性）
                    if hasattr(module, 'lora_A'):
                        matched.append(name)
            logger.info(f"[LoRA] Matched projection layers: {len(matched)} -> {matched[:20]}" + (" (truncated)" if len(matched) > 20 else ""))
        # 覆盖回 pipeline 引用
        pipeline.ref.sparse_dit_512 = target
        trainable = [n for n, p in target.named_parameters() if p.requires_grad]
        if accelerator.is_main_process:
            logger.info(f"[LoRA] Trainable params: {len(trainable)} layers | total_params={sum(p.numel() for p in target.parameters() if p.requires_grad)/1e6:.2f}M")
    else:
        for p in pipeline.ref.parameters():
            p.requires_grad = False

    # MeshScorer 接口: MeshScorer(score_fns_cfg, device, verbose=False, camera_normal_cfg=None)
    score_fns_cfg = {
        "uni3d": float(cfg.reward_fn.uni3d),
        "camera_normal": float(cfg.reward_fn.camera_normal),
    }
    mesh_scorer = MeshScorer(score_fns_cfg, device=str(accelerator.device), verbose=False, camera_normal_cfg=dict(cfg.camera_normal))

    return pipeline, mesh_scorer


def grpo_sampling_step(
    accelerator: Accelerator,
    pipeline: Direct3DS2PipelineWithLogProb,
    images: List[Any],
    cfg: ml_collections.ConfigDict,
    generator: Optional[torch.Generator],
):
    """执行一次候选采样：
    - 输入 images (list[PIL]) 长度 = input_batch_size
    - 调用 pipeline.sample_candidates_with_logprob
    返回原始 mesh 列表与 logprob、latents 结构（简化，后续主循环再整理）。
    参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:253-291, 320-341`（sparse512 采样与解码核心片段）
    """
    image = images[0]  # 目前 batch=1 假设
    meshes, all_latents, all_log_probs, all_kl = pipeline.sample_candidates_with_logprob(
        image=image,
        num_candidates=int(cfg.sample.num_candidates),
        dense_params={"num_inference_steps": int(cfg.sample.num_inference_steps_dense)},
        sparse_params_512={"num_inference_steps": int(cfg.sample.num_inference_steps_sparse512)},
        guidance_scale=float(cfg.sample.guidance_scale),
        use_sde=bool(cfg.sample.use_sde),
        sigma_min=float(cfg.sample.sigma_min),
        rescale_t=float(cfg.sample.rescale_t),
        generator=generator,
    )
    return meshes, all_latents, all_log_probs, all_kl


def create_dataloader(cfg: ml_collections.ConfigDict):
    """最小数据加载：遍历指定目录 images 子目录，仅返回 PIL 图像（无 prompt）。"""
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from pathlib import Path

    class _ImageSet(Dataset):
        def __init__(self, root: str):
            """参考：无官方对应（自定义数据集），与 Direct3D 参考无关"""
            self.root = Path(root)
            if (self.root / "images").exists():
                self.root = self.root / "images"
            exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
            files = []
            for e in exts:
                files.extend(self.root.glob(e))
            self.files = sorted(files)
            if not self.files:
                raise ValueError(f"No images found in {self.root}")
        def __len__(self):
            """参考：无官方对应（数据集长度）"""
            return len(self.files)
        def __getitem__(self, idx):
            """参考：无官方对应（读取图像并转 RGB）"""
            fp = self.files[idx]
            img = Image.open(fp).convert("RGB")
            return {"image": img, "path": str(fp)}

    ds = _ImageSet(cfg.data_dir)

    def _collate(batch):
        """参考：无官方对应（自定义 collate）"""
        images = [b["image"] for b in batch]
        paths = [b["path"] for b in batch]
        return images, paths

    return DataLoader(ds, batch_size=int(cfg.sample.input_batch_size), shuffle=True, collate_fn=_collate)


def main(_):
    """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:23-66, 68-172`（仅用于理解 pipeline 构建与 device 设置；本函数为训练脚本自定义）"""
    cfg = _CONFIG.value
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        project_config=ProjectConfiguration(project_dir=cfg.logdir, logging_dir=os.path.join(cfg.logdir, cfg.run_name)),
        log_with=["wandb"] if accelerator_state_is_main() else None,
    )
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="direct3d_s2_grpo", name=cfg.run_name, config=dict(cfg))

    set_seed(int(cfg.seed))

    pipeline, mesh_scorer = build_pipeline_and_models(cfg, accelerator)
    dataloader = create_dataloader(cfg)
    use_stat = bool(cfg.per_image_stat_tracking)
    stat_tracker = PerImageStatTracker(global_std=bool(cfg.sample.global_std)) if use_stat else None

    # 仅 LoRA 参数进入优化器（sparse_dit_512 已被 LoRA 包装）
    params = [p for p in pipeline.ref.sparse_dit_512.parameters() if p.requires_grad]
    optim_groups = [
        {"params": params, "lr": float(cfg.train.learning_rate), "weight_decay": float(cfg.train.adam_weight_decay)}
    ]
    optimizer = optim.AdamW(optim_groups, betas=(cfg.train.adam_beta1, cfg.train.adam_beta2), eps=cfg.train.adam_epsilon)

    # 仅将可训练的 LoRA 模块交给 accelerator，pipeline 作为包装无需放入 DDP
    train_module = pipeline.ref.sparse_dit_512  # LoRA 注入后的模块 (nn.Module)
    train_module, optimizer = accelerator.prepare(train_module, optimizer)
    # 回写（加速器包装后可能替换参数引用）
    pipeline.ref.sparse_dit_512 = train_module

    # 基础训练循环（最小 GRPO：奖励 -> 基线优势 -> sum logprob -> Policy Loss）
    for epoch in range(int(cfg.num_epochs)):
        for step, (images, paths) in enumerate(dataloader):
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(int(cfg.seed) + epoch * 1000 + step)

            meshes, all_latents, all_log_probs, all_kl = grpo_sampling_step(accelerator, pipeline, images, cfg, generator)
            # 奖励计算：调用 MeshScorer.score，当前每张图像只生成一组候选
            # 构造 metadata：最小包含 image_path 以便 camera_normal scorer 使用
            meta_list = [{"image_path": paths[0]} for _ in range(len(meshes))]
            score_fns_cfg = {"uni3d": float(cfg.reward_fn.uni3d), "camera_normal": float(cfg.reward_fn.camera_normal)}
            details, meta_out = mesh_scorer.score(meshes, [images[0]] * len(meshes), meta_list, score_fns_cfg)
            rewards_tensor = torch.as_tensor(details["avg"], device=accelerator.device, dtype=torch.float32)  # (K,)
            rewards = [float(x) for x in rewards_tensor.tolist()]

            # 整理 log_prob：all_log_probs 为按时间展开 (K * steps) 个 (1,) 张量。
            steps_sparse = int(cfg.sample.num_inference_steps_sparse512)
            K = int(cfg.sample.num_candidates)
            assert len(all_log_probs) == K * steps_sparse, "log_prob 数量与 (K*steps) 不匹配"
            # 聚合每个候选的总 log_prob (sum over time)
            cand_logprobs = []  # 长度 K
            for k in range(K):
                seg = all_log_probs[k * steps_sparse : (k + 1) * steps_sparse]
                cand_logprobs.append(torch.stack(seg).sum())  # 标量
            cand_logprobs = torch.stack(cand_logprobs)  # (K,)

            # 基线优势： r - mean(r)  (最简单版本；后续可引入 winrate / rank / global std)
            baseline = rewards_tensor.mean()
            advantages = rewards_tensor - baseline  # (K,)
            # 可选标准化（提升数值稳定）；仅在方差>0时
            var = advantages.var(unbiased=False)
            if var > 0:
                advantages = advantages / (var.sqrt() + 1e-6)

            # Policy Loss = - E[adv * logprob]; 这里将 logprob 视为整条轨迹总和
            loss = -(advantages.detach() * cand_logprobs).mean()

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(train_module.parameters(), float(cfg.train.max_grad_norm))
            optimizer.step()

            # 统计记录（示例）
            if stat_tracker is not None:
                # 这里 stat_tracker.update 需要 image_ids 与 rewards，本最小版本使用单一 image_id=0
                image_ids = torch.zeros(len(rewards_tensor), dtype=torch.long)
                _ = stat_tracker.update_torch(image_ids, rewards_tensor)  # 返回 advantages (未进一步使用)

            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} Step {step} | rewards={rewards} loss={float(loss.item()):.4f} adv_norm={float(advantages.norm().item()):.4f} logprob_steps={len(all_log_probs)}")

            break  # 每 epoch 仅跑一个 batch（最小验证）；移除以进行完整训练

        if accelerator.is_main_process and stat_tracker is not None:
            avg_group_size, num_images = stat_tracker.get_stats()
            logger.info(f"[Epoch {epoch}] StatTracker avg_group_size={avg_group_size:.2f} images_tracked={num_images}")

    if accelerator.is_main_process:
        logger.info("Training skeleton finished (no parameter updates performed).")


def accelerator_state_is_main():
    # 简单帮助函数（加速器尚未构建时不调用 distributed API）
    # 参考：无官方对应（训练脚本辅助）
    return True


if __name__ == "__main__":
    app.run(main)
