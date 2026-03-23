"""
Trellis Contrastive FlowEdit Autograd 训练入口 — Latent 空间对比学习。

训练流程：
  Phase 0:   Dense Sampling
  Phase 1:   Pretrained Rollout (teacher_context, no_grad) → clean z₀
  Phase 2:   Add noise → zₜ (随机时间步)
  Phase 3:   Student velocity prediction + VelocityTracker proxy
  Phase 3.5: (可选) Teacher velocity reg → reg_grad
  Phase 4a:  Decode/render Teacher z₀ → src images (PIL)
  Phase 4b:  FlowEdit edit src → tgt images (PIL)
  Phase 4c:  DINOv2 encode src / tgt → c_src, c_tgt
  Phase 5a:  Teacher denoise zₜ with c_tgt → positive (detached)
  Phase 5b:  Teacher denoise zₜ with c_src → negative (detached)
  Phase 5c:  Contrastive loss → backward → v_proxy.grad
  Phase 5d:  Relay → θ.grad

★ 与 flowedit_autograd.py 的关键区别：
  - 不在 2D 图像空间做 guidance backward
  - 对比 loss 在归一化 latent 空间（SLAT features）
  - 不需要 3-sub-step decode 显存优化（loss 不经过 decode）
  - FlowEdit 仅用于生成正样本条件，不提供梯度
  - 额外两次 Teacher velocity 预测（c_tgt / c_src）

梯度传播路径：
  contrastive_loss → z0_hat_norm → v_proxy → (relay) → v_student → θ.grad
"""

# =====================================================================
# 标准库 + 第三方库
# =====================================================================
import gc
import logging
import os
import sys
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
from edit4shape.systems.utils.profiler import PhaseProfiler
from edit4shape.generators.trellis.state import TrellisContrastiveState
from edit4shape.guidance import create_guidance

from edit4shape.systems.trellis.system import (
    TrellisSystem,
    _CONFIG,
    build_flowedit_system,
    build_dataloaders,
)
from edit4shape.systems.trellis.forward import evaluate
from edit4shape.systems.trellis.autograd_template import trellis_contrastive_step
from edit4shape.systems.trellis.stage_ops import TrellisContrastiveOps

# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    Contrastive FlowEdit Autograd 训练主入口。

    流程：
    1. 环境设置 + Accelerator
    2. 构建 DataLoader + TrellisSystem
    3. 加载检查点
    4. 训练循环（trellis_contrastive_step）
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    TrellisSystem.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.use_wandb else None,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis-contrastive",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = TrellisVisualIO(
        visuals_train_dir,
        target_h=cfg.renderer.resolution,
        vis_freq=vis_freq,
        accelerator=accelerator,
    )

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_flowedit_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
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
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（★ Contrastive FlowEdit Autograd）
    # =====================================================
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")

    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models

    profiler = PhaseProfiler(
        enabled=True,
        verbose=accelerator.is_main_process,
    )

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1

            with accelerator.accumulate(pipe_models['slat_flow_model']):
                with TrainModeGuard(pipe_models['slat_flow_model']):
                    state = TrellisContrastiveState()
                    state.attach_batch(batch, pipeline=pipeline)

                    # ★ Contrastive FlowEdit 训练步：
                    # P0 → P1(pretrained rollout) → P2(加噪)
                    # → P3(student velocity) → P3.5(reg)
                    # → P4a/b/c(render→edit→encode)
                    # → P5a/b/c/d(teacher denoise→contrastive loss→relay)
                    train_log = trellis_contrastive_step(
                        ops=TrellisContrastiveOps(),
                        state=state,
                        system=system,
                        global_step=global_step,
                        profiler=profiler,
                    )

                # 标准 optimizer step
                accelerator.clip_grad_norm_(
                    pipe_models['slat_flow_model'].parameters(), 10.0
                )
                system.optimizer.step()
                system.optimizer.zero_grad()

            # 可视化保存
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                _pipe = system.guidance.pipe if system.guidance else None
                _n_prog = cfg.freq.save.progress_samples
                visual_io.save_contrastive_train(
                    state, epoch, global_step,
                    pipe=_pipe, n_progress_samples=_n_prog,
                )

            # profiler 收集 + 日志合并
            time_log = profiler.collect(global_step, print_freq=10)
            train_log.update(time_log)

            train_logger.log_step(train_log, len(batch['image_pils']), global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)
