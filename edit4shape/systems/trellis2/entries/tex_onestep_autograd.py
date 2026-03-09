"""
Trellis2 Tex 训练系统 — Onestep 单步去噪 + FlowEdit 2D Guidance 版本。

采用 Pretrained Rollout + 单步去噪 + 3-sub-step Decode 策略：
  - 不需要 VJP、proxy chain
  - P5 relay backward 在 model.no_sync() 下执行，手动 all-reduce
  - P4c OOM 安全降级（跳过 P5），不会导致分布式死锁
  - 3D Generator 端只做单步去噪（显存固定且低）
  - 2D FlowEdit Guidance 完全复用

训练流程：
  P0: Shape 冻结前置（no_grad shape forward + detach）
  P1: Pretrained Rollout (teacher, no_grad) → clean tex z₀
  P2: 加噪 z₀ → zₜ
  P3: predict_cfg_velocity (student, with grad) → v_student
      → setup VelocityTracker proxy → v_proxy
      → ẑ₀ = zₜ - t·v_proxy → denormalize → update tex_slat
  P3.5: (可选) teacher velocity → reg backward → reg_grad
  P4a: decode_render_pbr (no_grad) → detached comp_rgb
  P4b: guidance-only backward → rgb_grad
  P4c: decode_render_pbr (with grad) + backward(rgb_grad) → v_proxy.grad
  P5: relay → θ_tex.grad

与 tex_autograd.py（VJP 三阶段）的区别：
  - 不使用 RolloutTracker / proxy chain / VJP 循环
  - 使用 VelocityTracker 做 velocity 空间的梯度追踪
  - 使用 Pretrained Rollout 获取 clean z₀ 而非 proxy rollout
  - 显存峰值 ≈ max(guidance, decode_render) + 单步 velocity 预测

依赖：
  - TRELLIS.2 参考实现
  - Accelerate 分布式训练库
  - nvdiffrast / nvdiffrec_render (PBR 渲染)
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 标准库 & 第三方库
# =====================================================================
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
from edit4shape.systems.trellis2.forward import evaluate as _evaluate
from edit4shape.systems.trellis2.stage_ops import TexOps
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
    """构建 Tex-only 训练系统（Shape 冻结）。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="tex")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Tex 评估（含 Shape Forward + Tex Forward）。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=True)


# =====================================================================
# Onestep 编排函数
# =====================================================================

def onestep_tex_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler = None,
) -> Dict[str, Any]:
    """
    Tex-only Onestep 训练步（同步 Guidance 版本）。

    委托给 onestep_step 模板，注入 TexOps。

    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler

    Returns:
        合并的日志字典（含 profiler 计时）
    """
    merged = onestep_step(
        ops=TexOps(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
    )
    merged.update(profiler.collect(global_step, print_freq=int(system.cfg.freq.profiler)))
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    只训练 Tex Flow Model，使用 Onestep 策略（Pretrained Rollout + 单步去噪）。
    Shape 阶段使用冻结的模型生成几何。

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.tex_onestep_autograd \
            --config=configs/trellis2_tex_onestep.py
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator（含 wandb 日志）
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

    # 初始化 wandb trackers
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-tex-onestep",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.render_base.resolution, vis_freq=vis_freq, accelerator=accelerator)

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
    # Step 8: 训练循环（Onestep — no_sync + 手动 all-reduce）
    # =====================================================
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)
        accum_count = 0

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])

            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.tex.config.cond_resolution)

            # ★ P5 relay backward 在 model.no_sync() 下执行（onestep_step 内部），
            #   不触发 DDP all-reduce，防止 P4c OOM 时死锁。
            with TrainModeGuard(system.tex.model):
                tex_log = onestep_tex_step(state, system, global_step, profiler=profiler)

            accum_count += 1
            if accum_count >= accum_steps:
                sync_grads_and_step(system.tex.model, system.tex.optimizer, grad_clipper, n_accumulated=accum_count)
                accum_count = 0

            # Logging & Visualization
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)

            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 残留引用
            del state, tex_log
            torch.cuda.empty_cache()

        # ---- 尾部 flush：处理不完整累积窗口 ----
        if accum_count > 0:
            sync_grads_and_step(system.tex.model, system.tex.optimizer, grad_clipper, n_accumulated=accum_count)
            accum_count = 0

        # ---- 周期性评估（epoch 级别）----
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
