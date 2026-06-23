"""
Trellis FlowEdit Autograd 训练入口 — Rollout + Finetuned 单步去噪。

训练流程：
  Phase 0: ops.pre_rollout (dense_sampling)
  Phase 1: ops.rollout (pretrained 或 student, no_grad) → clean z₀
  Phase 2: ops.add_noise → zₜ (随机时间步)
  Phase 3: ops.finetune_denoise → ẑ₀ (有梯度)
  Phase 4a: decode/render (no_grad) → detached comp_rgb
  Phase 4b: guidance backward → rgb_grad (FlowEdit, 不变)
  Phase 4c: decode/render (有梯度) + backward(rgb_grad) → θ.grad

★ 与 hybrid_autograd.py 的关键区别：
  - 不需要 VJP、proxy chain
  - 不需要 no_sync / 手动 all-reduce
  - 标准 autograd 训练，使用 accelerator.accumulate() 即可
  - 3D Generator 端只做单步去噪（显存固定且低）
  - 2D FlowEdit Guidance 完全不变

Pretrained 模型通过 strategy.sparse_teacher_context() 获取：
  - LoRA 模式：disable_adapter()，零额外显存
  - Full 模式：替换为 self._sparse_teacher（额外显存）
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
from edit4shape.generators.trellis.state import TrellisState
from edit4shape.guidance import create_guidance

from edit4shape.systems.trellis.system import (
    TrellisSystem,
    _CONFIG,
    build_flowedit_system,
    build_dataloaders,
)
from edit4shape.systems.trellis.forward import evaluate
from edit4shape.systems.trellis.autograd_template import trellis_flowedit_step
from edit4shape.systems.trellis.stage_ops import TrellisFlowEditOps

# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    FlowEdit Autograd 训练主入口。

    流程：
    1. 环境设置 + Accelerator
    2. 构建 DataLoader + TrellisSystem
    3. 加载检查点
    4. 训练循环（trellis_flowedit_step）
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
        mixed_precision="no",
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.use_wandb else None,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis-flowedit",
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
    if cfg.checkpoint:
        system.guidance.load_checkpoint(cfg.checkpoint, loss_cfg=cfg.train.guidance.loss)

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
    # Step 8: 训练循环（★ FlowEdit Autograd）
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

            # ★ 标准 accelerator.accumulate() 即可：
            #   - 不需要 no_sync hack（没有 VJP 循环）
            #   - DDP 自动 all-reduce 在 accumulate 边界触发
            with accelerator.accumulate(pipe_models['slat_flow_model']):
                with TrainModeGuard(pipe_models['slat_flow_model']):  # 只有 flow_model 需要 train mode
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=pipeline)

                    # ★ FlowEdit 训练步：
                    # P0 → P1(pretrained rollout) → P2(加噪)
                    # → P3(finetuned 单步去噪)
                    # → P4a/P4b/P4c(decode + guidance + backward)
                    train_log = trellis_flowedit_step(
                        ops=TrellisFlowEditOps(),
                        state=state,
                        system=system,
                        global_step=global_step,
                        profiler=profiler,
                    )

                # ★ 标准 optimizer step（accelerator 自动处理梯度累积边界）
                accelerator.clip_grad_norm_(
                    pipe_models['slat_flow_model'].parameters(), 10.0
                )
                system.optimizer_sparse.step()
                system.optimizer_sparse.zero_grad()

            # 可视化保存
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                _pipe = system.guidance.pipe if system.guidance else None
                _n_prog = cfg.freq.save.progress_samples
                visual_io.save_color_train(
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
            if accelerator.is_main_process:
                system.guidance.save_checkpoint(ckpt_root / f"checkpoint_{epoch}_{global_step}")


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)
