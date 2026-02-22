"""
Trellis2 Shape 训练系统 — 三阶段 Autograd 版本。

采用三阶段反向传播策略，将 rollout / decoder+renderer 的计算图隔离，
任意时刻只有一个阶段的计算图驻留显存，大幅降低显存峰值。

三阶段流程：
  Phase 1: rollout no_grad → shape_slat（proxy chain，不含模型图）
           + 计算 reg_loss（连着 cond_proxy 的图）→ state.regularization.reg_loss
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）→ 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)，完全不感知 reg

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ Phase 2 一次性 backward 算出                       ↑ Phase 3 逐步 VJP

特性：
- 无 slat_proxy 中间层：loss.backward() 一路反传穿过 renderer → decoder → slat → scheduler → CFG → cond_proxy
- 三阶段显存隔离：峰值仅为 decode/render 图 + proxy chain
- 支持 mesh 与 mesh_peeled 两种 Normal 渲染模式
- 评估路径仍使用单阶段 forward（trellis2_shape_forward）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
"""

# =====================================================================
# 标准库 & 第三方库
# =====================================================================
import os, sys
from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import Accelerator
from absl import app
from ml_collections import config_flags

# =====================================================================
# TRELLIS.2 参考实现路径设置
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.guidance import create_guidance
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, PhaseProfiler
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system,
)
from edit4shape.systems.trellis2.forward import evaluate as _evaluate


# =====================================================================
# 统一接口：build_system / evaluate
# =====================================================================

def build_system(cfg, accelerator, guidance_factory):
    """构建 Shape-only 训练系统。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="shape")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Shape-only 评估。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=False)


# =====================================================================
# 三阶段 Autograd — 编排函数
# =====================================================================

def three_phase_shape_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler = None,
) -> Dict[str, Any]:
    """
    Shape-only 三阶段训练步（同步 Guidance 版本）。
    
    委托给通用模板 three_phase_step，注入 ShapeOps 和 shape-only 的清理策略。
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler，用于测量各阶段耗时（enabled=False 时为空操作）
    
    Returns:
        合并的日志字典（含 profiler 计时）
    """
    from edit4shape.systems.trellis2.stage_ops import ShapeOps
    from edit4shape.systems.utils.autograd_template import three_phase_step
    
    merged = three_phase_step(
        ops=ShapeOps(),
        state=state,
        system=system,
        global_step=global_step,
        profiler=profiler,
        clean_for_vjp=lambda s: s.prepare_for_shape_vjp(),
    )
    merged.update(profiler.collect(global_step, print_freq=int(system.cfg.freq.profiler)))
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    训练 Shape Flow Model，使用 Normal 渲染监督几何。
    
    流程: Dense Sampling → Shape Rollout → Normal 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2.shape_autograd --config=configs/trellis2_shape.py
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
            project_name="trellis2-shape-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    from edit4shape.systems.trellis2.system import build_dataloaders
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
    # Step 8: 训练循环（三阶段 Autograd — velocity-level proxy）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.shape.config.cond_resolution)
            
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_log = three_phase_shape_step(state, system, global_step, profiler=profiler)
                
                # Optimizer Step
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # Logging & Visualization
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 残留引用
            del state, shape_log
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
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
