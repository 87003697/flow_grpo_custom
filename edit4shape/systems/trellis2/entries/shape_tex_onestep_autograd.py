"""
Trellis2 Shape+Tex 双阶段训练系统 — Onestep 单步去噪 + FlowEdit 2D Guidance 版本。

同时训练 Shape 和 Tex 两个 Flow Model，每个阶段分别使用 Onestep 策略：
  - Pretrained Rollout + 单步去噪 + 3-sub-step Decode
  - P5 relay backward 在 model.no_sync() 下执行，手动 all-reduce
  - P4c OOM 安全降级（跳过 P5），不会导致分布式死锁
  - 不需要 VJP、proxy chain

整体编排（每个训练步）：
  ┌─────────── Shape Onestep ──────────────┐
  │ Dense Sampling (no_grad)               │
  │ P1: Pretrained Rollout → clean z₀     │
  │ P2: add_noise → zₜ                    │
  │ P3: predict_cfg_velocity → v_student   │
  │ P3.5: teacher velocity → reg_grad      │
  │ P4a/4b/4c: 3-sub-step decode+guidance │
  │ P5: relay → shape θ.grad              │
  │ optimizer_shape.step()                 │
  └────────────────────────────────────────┘
         ↓ detach shape 产物
  ┌─────────── Tex Onestep ────────────────┐
  │ (pre_rollout = no-op, shape 产物复用)  │
  │ P1: Pretrained Rollout → clean z₀     │
  │ P2: add_noise → zₜ                    │
  │ P3: predict_cfg_velocity → v_student   │
  │ P3.5: teacher velocity → reg_grad      │
  │ P4a/4b/4c: 3-sub-step decode+guidance │
  │ P5: relay → tex θ.grad                │
  │ optimizer_tex.step()                   │
  └────────────────────────────────────────┘

依赖：
  - TRELLIS.2 参考实现
  - Accelerate 分布式训练库
  - nvdiffrast / nvdiffrec_render
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 标准库 & 第三方库
# =====================================================================
import logging
from typing import Any, Dict

import torch
from accelerate import Accelerator
from absl import app
from ml_collections import config_flags

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import (
    detach_shape_outputs_for_tex,
    evaluate as _evaluate,
)
from edit4shape.systems.trellis2.stage_ops import ShapeOps, TexOpsFromShape
from edit4shape.systems.trellis2.autograd_template import onestep_step, sync_grads_and_step
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, PhaseProfiler
from edit4shape.guidance import create_guidance
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

# =====================================================================
# 统一接口：build_system / evaluate
# =====================================================================

def build_system(cfg, accelerator, guidance_factory):
    """构建 Shape+Tex 双阶段训练系统。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="shape_tex")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Shape+Tex 双阶段评估。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=True)


# =====================================================================
# Onestep 编排函数
# =====================================================================

def onestep_shape_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Shape Onestep 训练步（双阶段模式）。

    委托给 onestep_step 模板，注入 ShapeOps。

    Returns:
        合并的日志字典（key 前缀 "shape/"，不含 profiler 计时）
    """
    return onestep_step(
        ops=ShapeOps(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        prefix="shape/",
    )


def onestep_tex_step_from_shape(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Tex Onestep 训练步（从已有 Shape 产物出发）。

    委托给 onestep_step 模板，注入 TexOpsFromShape。
    TexOpsFromShape 的 pre_rollout 为 no-op（Shape 产物由上游提供）。

    前置条件：
        state.coords / features.shape_slat / subs / meshes 已就绪（detached）

    Returns:
        合并的日志字典（key 前缀 "tex/"，不含 profiler 计时）
    """
    return onestep_step(
        ops=TexOpsFromShape(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        prefix="tex/",
    )


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    同时训练 Shape 和 Tex 两个 Flow Model，使用 Onestep 策略。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.shape_tex_onestep_autograd \
            --config=configs/trellis2_shape_tex_onestep.py
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
    # =====================================================
    use_wandb = cfg.use_wandb
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape+tex-onestep",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(
        visuals_train_dir, target_h=cfg.render_base.resolution,
        vis_freq=vis_freq, accelerator=accelerator,
    )

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_optimizers(accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, epoch=start_epoch, global_step=global_step,
            eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（双阶段 Onestep — no_sync + 手动 all-reduce）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)
        shape_accum_count = 0
        tex_accum_count = 0

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])

            state = Trellis2State()
            state.attach_batch(
                batch, pipeline=system.pipeline,
                resolution=system.tex.config.cond_resolution,
            )

            # ============================================
            # Shape Onestep Forward + Backward + Update
            # ============================================
            # ★ P5 relay backward 在 model.no_sync() 下执行（onestep_step 内部）
            with TrainModeGuard(system.shape.model):
                shape_log = onestep_shape_step(
                    state, system, global_step, profiler,
                )

            shape_accum_count += 1
            if shape_accum_count >= accum_steps:
                sync_grads_and_step(system.shape.model, system.shape.optimizer, shape_grad_clipper, n_accumulated=shape_accum_count)
                shape_accum_count = 0

            # ★ 保存 Shape 可视化（在 Tex guidance 覆盖 views_edited 之前）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)

            # ============================================
            # Detach 转接：Shape → Tex
            # ============================================
            detach_shape_outputs_for_tex(state)

            # ============================================
            # Tex Onestep Forward + Backward + Update
            # ============================================
            with TrainModeGuard(system.tex.model):
                tex_log = onestep_tex_step_from_shape(
                    state, system, global_step, profiler,
                )

            tex_accum_count += 1
            if tex_accum_count >= accum_steps:
                sync_grads_and_step(system.tex.model, system.tex.optimizer, tex_grad_clipper, n_accumulated=tex_accum_count)
                tex_accum_count = 0

            # ★ 保存 Tex 可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

            # ============================================
            # Logging
            # ============================================
            profiler_log = profiler.collect(
                global_step, print_freq=int(cfg.freq.profiler),
            )
            shape_log.update(profiler_log)

            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)

            # 释放当前 step 残留引用
            del state, shape_log, tex_log
            torch.cuda.empty_cache()

        # ---- 尾部 flush：处理不完整累积窗口 ----
        if shape_accum_count > 0:
            sync_grads_and_step(system.shape.model, system.shape.optimizer, shape_grad_clipper, n_accumulated=shape_accum_count)
        if tex_accum_count > 0:
            sync_grads_and_step(system.tex.model, system.tex.optimizer, tex_grad_clipper, n_accumulated=tex_accum_count)

        # ---- 周期性评估（epoch 级别） ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, epoch=epoch, global_step=global_step,
                eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
