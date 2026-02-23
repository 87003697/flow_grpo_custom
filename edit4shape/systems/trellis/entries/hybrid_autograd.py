"""
Trellis 双路渲染 Autograd 训练入口 — Mesh Normal + GS Color 同时 guidance。

训练流程（3-sub-step Phase 2，每路渲染独立循环）：
  Phase 0: ops.pre_rollout (dense_sampling)
  Phase 1: ops.rollout (no_grad + proxy chain)
  循环每路渲染 {
    Phase 2a: ops.decode_render_dict(renderer_key=key) no_grad → detached comp_rgb
    Phase 2b: guidance-only backward → rgb_grad（各路独立 cfg/weight）
    Phase 2c: ops.decode_render_dict(renderer_key=key) with grad + backward(rgb_grad)
  }
  Phase 3: ops.vjp_loop (合并 reg_grads + VJP)

多路梯度通过 proxy.grad += 自动累加。

渲染策略：TrellisHybridOps
  - get_render_passes() → [("mesh", guidance_normal_cfg, w1), ("gs", guidance_color_cfg, w2)]
  - decode_render_dict(renderer_key="mesh") → Mesh Normal
  - decode_render_dict(renderer_key="gs")   → GS Color

配置要求（cfg.train 下）：
  guidance_normal:      Mesh Normal guidance 配置
  guidance_color:       GS Color guidance 配置
  loss.guidance_normal: Mesh Normal guidance 权重
  loss.guidance_color:  GS Color guidance 权重
"""

# =====================================================================
# 标准库 + 第三方库
# =====================================================================
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
    build_hybrid_system,
    build_dataloaders,
)
from edit4shape.systems.trellis.forward import evaluate
from edit4shape.systems.trellis.phases import trellis_hybrid_three_phase_step
from edit4shape.systems.trellis.stage_ops import TrellisHybridOps


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    双路渲染 Autograd 训练主入口。

    流程：
    1. 环境设置 + Accelerator
    2. 构建 DataLoader + TrellisSystem（含 mesh + gs 双渲染器）
    3. 加载检查点
    4. 训练循环（hybrid_three_phase_step）
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
            project_name="trellis-distillation",
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
    # Step 5: 构建系统组件（★ 双路渲染：mesh + gs）
    # =====================================================
    system = build_hybrid_system(cfg, accelerator, guidance_factory=create_guidance)
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
            visuals_eval_dir=visuals_eval_dir
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（★ 双路渲染三阶段 Autograd）
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
                with TrainModeGuard(
                    pipe_models['slat_flow_model'],
                    pipe_models['slat_decoder_mesh'],
                    pipe_models['slat_decoder_gs'],
                ):
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=pipeline)

                    # ★ 双路渲染三阶段训练步：
                    # P0 → P1
                    # → 循环 { P2a(no_grad) → P2b(guid_bw) → P2c(grad) } × 2路
                    # → P3(VJP)
                    train_log = trellis_hybrid_three_phase_step(
                        ops=TrellisHybridOps(),
                        state=state,
                        system=system,
                        global_step=global_step,
                        profiler=profiler,
                        clean_for_vjp=lambda s: s.prepare_for_vjp(),
                    )

                # ---- 优化器步进 ----
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe_models["slat_flow_model"].parameters(), 10.0)
                    system.optimizer.step()
                    system.optimizer.zero_grad()

            # 仅主进程按频率保存可视化（★ Hybrid: 分别保存 Normal + Color）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                _pipe = system.guidance.pipe if system.guidance else None
                _n_prog = cfg.freq.save.progress_samples
                visual_io.save_normal_train(
                    state, epoch, global_step,
                    pipe=_pipe, n_progress_samples=_n_prog,
                )
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
                visuals_eval_dir=visuals_eval_dir
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
