"""
Trellis2 Tex 训练系统 — 三阶段 Autograd 版本。

基于 trellis2_tex.py 的共享组件（Trellis2System, Trellis2State, build_system,
decode_and_render_pbr, evaluate），
本模块仅实现 **Autograd 三阶段训练策略**：

核心流程：
  Phase 0: Shape 冻结前置（no_grad，获取几何条件）
  Phase 1: Tex Rollout（no_grad + RolloutTracker 记录 proxy 轨迹）
           + 计算 reg_loss → autograd.grad → reg_grads（纯数据，存 tracker）
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）→ 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)，完全不感知 reg

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ Phase 2 一次性 backward 算出                       ↑ Phase 3 逐步 VJP

与 trellis2_tex.py 的区别：
- trellis2_tex.py: 标准 forward → guidance → backward（端到端计算图）
- 本模块: 三阶段 Autograd（显存 O(1)，不随步数增长）

独有组件：
1. three_phase_tex_step: 三阶段编排（委托给 three_phase_step + TexOps）
2. main: 训练主循环（使用三阶段策略）

Phase 函数（shape_frozen_prepare_no_grad, tex_phase1_rollout, tex_phase2a_decode_render 等）
已抽取到 phases.py / stage_ops.py，由通用模板 three_phase_step 统一调度。
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# 标准库 & 第三方库
# =====================================================================
from pathlib import Path
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
from edit4shape.systems.utils.autograd_template import three_phase_step
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
# 三阶段 Autograd — 编排函数
# =====================================================================

def three_phase_tex_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler = None,
) -> Dict[str, Any]:
    """
    Tex-only 三阶段训练步（同步 Guidance 版本）。
    
    委托给通用模板 three_phase_step，注入 TexOps 和 tex-only 的清理策略。
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler，用于测量各阶段耗时（enabled=False 时为空操作）
    
    Returns:
        合并的日志字典（含 profiler 计时）
    """
    merged = three_phase_step(
        ops=TexOps(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        clean_for_vjp=lambda s: s.prepare_for_tex_vjp(),
    )
    merged.update(profiler.collect(global_step, print_freq=int(system.cfg.freq.profiler)))
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd 版本）。
    
    只训练 Tex Flow Model，使用 PBR 渲染监督纹理。
    Shape 阶段使用冻结的模型生成几何。
    训练策略使用三阶段 Autograd（显存 O(1)）。
    
    流程: Dense Sampling → Shape Rollout (frozen) → Tex Rollout → PBR 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2.tex_autograd --config=configs/trellis2_tex.py
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
            project_name="trellis2-tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
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
    # ★ Fix #2: 对齐 shape_autograd 签名 load(path, mode)
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
    # Step 8: 训练循环（三阶段 Autograd — cond-level proxy）
    # =====================================================
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    # ★ 自适应梯度裁剪（TRELLIS.2 默认参数：max_norm=1.0, clip_percentile=95）
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    # ★ Fix #5: 添加 PhaseProfiler（与 shape_autograd 对齐）
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_log = three_phase_tex_step(state, system, global_step, profiler=profiler)
                
                # Optimizer Step
                if accelerator.sync_gradients:
                    grad_clipper(system.tex.model.parameters())
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # Logging & Visualization
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 PBR 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 残留引用
            del state, tex_log
            torch.cuda.empty_cache()

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
        # ★ Fix #2: 对齐 shape_autograd 签名 save(epoch, step)
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
