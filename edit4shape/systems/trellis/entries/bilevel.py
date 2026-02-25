"""
Trellis 双层蒸馏版（VSD - Variational Score Distillation）训练入口。

基于 standard 训练入口，仅重写 main()：
- _compute_loss_and_backward: 支持嵌套 loss_dict（lora_stats）
- checkpoint: 增加 guidance LoRA 状态保存/加载

共享组件（build_system, trellis_forward, evaluate 等）全部从 system.py / forward.py 导入。
"""

# =====================================================================
# 标准库 + 第三方库
# =====================================================================
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import ml_collections
from absl import app
from accelerate import Accelerator

# =====================================================================
# TRELLIS 参考实现路径
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.systems.base import (
    TrainModeGuard,
    CheckpointIO,
    build_run_paths,
)
from edit4shape.systems.utils import MetricLogger
from edit4shape.systems.utils.visual import TrellisVisualIO
from edit4shape.generators.trellis.state import TrellisState
from edit4shape.guidance import create_bilevel_guidance

from edit4shape.systems.trellis.system import (
    TrellisSystem,
    _CONFIG,
    build_system,
    build_dataloaders,
)
from edit4shape.systems.trellis.forward import (
    trellis_forward,
    evaluate,
)


# =====================================================================
# BilevelCheckpointIO — 支持 guidance LoRA 状态的检查点管理器
# =====================================================================

@dataclass
class BilevelCheckpointIO(CheckpointIO):
    """
    在 CheckpointIO 基础上，额外保存/加载 guidance LoRA 权重。

    保存时：在 checkpoint 子目录内写入 guidance_lora.pt，
           同时在 ckpt_dir 根目录维护 guidance_lora_latest.pt。
    加载时：优先从 checkpoint 目录内加载，fallback 到 latest。
    """

    guidance: Any = None  # BilevelDistillationGuidance

    def save(self, system, state, cfg, epoch, global_step):
        """保存 3D 模型 + LoRA 检查点。"""
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"

        # 1) 主模型（accelerator.save_state + meta.json）
        super().save(system, state, cfg, epoch, global_step)

        # 2) LoRA 权重（仅主进程）
        if self.accelerator.is_main_process and self.guidance is not None:
            lora_sd = self.guidance.get_lora_state_dict()
            torch.save(lora_sd, target / "guidance_lora.pt")
            torch.save(lora_sd, self.ckpt_dir / "guidance_lora_latest.pt")
            logging.info(
                f"[BilevelCheckpointIO] LoRA saved: {target / 'guidance_lora.pt'} "
                f"(tensors={len(lora_sd)})"
            )

    def load(self, path, mode="train"):
        """加载 3D 模型 + LoRA 检查点。"""
        start_epoch = super().load(path, mode)

        if path and self.guidance is not None:
            root = Path(path)
            lora_path = root / "guidance_lora.pt"
            if not lora_path.exists():
                lora_path = root.parent / "guidance_lora_latest.pt"
            if lora_path.exists():
                self.guidance.load_lora_state_dict(
                    torch.load(lora_path, map_location="cpu")
                )
                logging.info(f"[BilevelCheckpointIO] LoRA restored: {lora_path}")

        return start_epoch


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """双层蒸馏（VSD）训练主入口。"""
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1-3: 环境、Accelerator、目录
    # =====================================================
    TrellisSystem.setup_env_and_seed(cfg)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.use_wandb else None,
    )

    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis-bilevel-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    visual_io = TrellisVisualIO(
        visuals_train_dir,
        target_h=cfg.renderer.resolution,
        vis_freq=int(cfg.freq.save.visual),
        accelerator=accelerator,
    )

    # =====================================================
    # Step 4-6: 数据、系统、检查点
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    system = build_system(cfg, accelerator, guidance_factory=create_bilevel_guidance)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    ckpt_root = run_root / "checkpoints"
    ckpt_io = BilevelCheckpointIO(accelerator, ckpt_root, guidance=system.guidance)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环
    # =====================================================
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models

    def _compute_loss_and_backward(state: TrellisState) -> Dict[str, Any]:
        """计算 loss 并反向传播（支持嵌套 loss_dict）。"""
        guidance_loss = state.guidance.loss.to(accelerator.device) * cfg.train.loss.guidance
        total = guidance_loss
        if state.regularization.reg_loss is not None:
            total = total + cfg.train.loss.reg * state.regularization.reg_loss

        accelerator.backward(total)

        # 构建日志（展平嵌套 dict）
        logs = {}
        for k, v in (state.guidance.loss_dict or {}).items():
            if v is None:
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    logs[f"loss/{k}/{sub_k}"] = float(sub_v)
            elif isinstance(v, torch.Tensor):
                logs[f"loss/{k}"] = v.item()
            else:
                logs[f"loss/{k}"] = float(v)

        logs["loss/total"] = total.item()
        if state.regularization.reg_loss is not None:
            logs["loss/reg"] = state.regularization.reg_loss.item()
        return logs

    state = None  # 防止空 batch 时 UnboundLocalError

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1

            with accelerator.accumulate(pipe_models['slat_flow_model']):
                with TrainModeGuard(
                    pipe_models['slat_flow_model'],
                    pipe_models['slat_decoder_mesh'],
                    pipe_models['slat_decoder_gs'],
                ):
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=pipeline)

                    # ---- 前向传播 ----
                    render_out = trellis_forward(
                        system, state, cfg, accelerator.device, global_step, is_training=True
                    )

                    # ---- Guidance ----
                    guidance_result = system.guidance.compute_guidance(
                        render_out["color"],
                        state.views_conditioned.image_pils,
                        guidance_cfg=cfg.train.guidance,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(guidance_result)

                    # ---- Loss & Backward ----
                    train_log = _compute_loss_and_backward(state)

                # ---- 优化器步进 ----
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe_models["slat_flow_model"].parameters(), 10.0)
                    system.optimizer.step()
                    system.optimizer.zero_grad()

            # 仅主进程按频率保存可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(
                    state=state,
                    epoch=epoch,
                    step=global_step,
                    pipe=system.guidance.pipe if system.guidance else None,
                    n_progress_samples=cfg.freq.save.progress_samples,
                )

            train_logger.log_step(train_log, len(batch['image_pils']), global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点（自动包含 LoRA）----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0) and state is not None:
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)
