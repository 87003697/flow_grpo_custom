"""
Trellis2 Shape+Tex 双阶段训练系统 — 三阶段 Autograd 版本。

同时训练 Shape 和 Tex 两个 Flow Model，每个阶段分别使用三阶段 Autograd 策略，
将 rollout / decoder+renderer 的计算图隔离，任意时刻只有一个阶段的计算图驻留显存。

整体编排（每个训练步）：
  ┌─────────── Shape Training ───────────┐
  │ Dense Sampling (no_grad)             │
  │ P1: rollout + RolloutTracker         │
  │ P2a: decode + render Normal          │
  │ P2: guidance + backward → proxy.grad │
  │ P3: VJP → shape θ.grad              │
  │ optimizer_shape.step()               │
  └──────────────────────────────────────┘
         ↓ detach shape 产物
  ┌─────────── Tex Training ─────────────┐
  │ P1: tex rollout + RolloutTracker     │
  │ P2a: decode_tex + render PBR         │
  │ P2: guidance + backward → proxy.grad │
  │ P3: VJP → tex θ.grad                │
  │ optimizer_tex.step()                 │
  └──────────────────────────────────────┘

三阶段流程（与 shape_autograd / tex_autograd 对齐）：
  Phase 1: rollout no_grad → slat（proxy chain，不含模型图）+ reg_loss
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad → 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)

复用关系：
- Shape Phase 函数：从 trellis2_shape_autograd 导入
- Tex Phase 函数：从 trellis2_tex_autograd 导入
- 新增：shape_phase2 变体（不释放 decode cache）+ detach 转接 + 编排 + 训练循环

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
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
from pathlib import Path

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
from edit4shape.systems.trellis2.autograd_template import three_phase_step
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
# Shape 三阶段编排
# =====================================================================

def three_phase_shape_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Shape 三阶段训练步（双阶段模式 — 保留 subs/meshes 给 Tex 阶段）。
    
    委托给通用模板 three_phase_step，注入 ShapeOps 和双阶段的清理策略
    （keep_decode_cache=True，保留 subs/meshes 供后续 Tex 使用）。
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler
    
    Returns:
        合并的日志字典（key 前缀 "shape/"，不含 profiler 计时）
    """
    return three_phase_step(
        ops=ShapeOps(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        clean_for_vjp=lambda s: s.prepare_for_shape_vjp(keep_decode_cache=True),
        prefix="shape/",
    )


# =====================================================================
# Tex 三阶段编排（从已有 Shape 产物出发，跳过 shape forward）
# =====================================================================

def three_phase_tex_step_from_shape(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler,
) -> Dict[str, Any]:
    """
    Tex 三阶段训练步（从已有 Shape 产物出发）。
    
    委托给通用模板 three_phase_step，注入 TexOpsFromShape 和 tex 清理策略。
    TexOpsFromShape 的 pre_rollout 为 no-op（Shape 产物由上游提供），
    decode_render 增加 meshes 可用性检查。
    
    前置条件:
        - state.coords / features.shape_slat / subs / meshes 已就绪（detached）
    
    Args:
        state: 已 detach 的状态（含 Shape 产物）
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler
    
    Returns:
        合并的日志字典（key 前缀 "tex/"，不含 profiler 计时）
    """
    return three_phase_step(
        ops=TexOpsFromShape(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        clean_for_vjp=lambda s: s.prepare_for_tex_vjp(),
        prefix="tex/",
    )


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd 版本）。
    
    同时训练 Shape 和 Tex 两个 Flow Model。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理
    训练策略使用三阶段 Autograd（显存 O(1)，不随步数增长）。
    
    流程: Dense Sampling → Shape 三阶段 → Detach → Tex 三阶段
    
    配置文件示例：
        python -m edit4shape.systems.trellis2.shape_tex_autograd \\
            --config=config/trellis2_shape_tex_distillation.py
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
            project_name="trellis2-shape+tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(
        visuals_train_dir, target_h=cfg.renderer.resolution,
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
    # Step 8: 训练循环（双阶段三阶段 Autograd）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    # ★ 自适应梯度裁剪（TRELLIS.2 默认参数：max_norm=1.0, clip_percentile=95）
    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(
                batch, pipeline=system.pipeline,
                resolution=system.tex.config.cond_resolution,
            )
            
            # ============================================
            # Shape 三阶段 Forward + Backward + Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_log = three_phase_shape_step(
                        state, system, global_step, profiler,
                    )
                
                if accelerator.sync_gradients:
                    shape_grad_clipper(system.shape.model.parameters())
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # ★ 保存 Shape 可视化（在 Tex guidance 覆盖 views_edited 之前）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)
            
            # ============================================
            # Detach 转接：Shape → Tex
            # ============================================
            detach_shape_outputs_for_tex(state)
            
            # ============================================
            # Tex 三阶段 Forward + Backward + Update
            # ============================================
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_log = three_phase_tex_step_from_shape(
                        state, system, global_step, profiler,
                    )
                
                if accelerator.sync_gradients:
                    tex_grad_clipper(system.tex.model.parameters())
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # ★ 保存 Tex 可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)
            
            # ============================================
            # Logging
            # ============================================
            # 收集 profiler 计时（在最后一个 tick 之后）
            profiler_log = profiler.collect(
                global_step, print_freq=int(cfg.freq.profiler),
            )
            shape_log.update(profiler_log)
            
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 释放当前 step 残留引用
            del state, shape_log, tex_log
            torch.cuda.empty_cache()
        
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
