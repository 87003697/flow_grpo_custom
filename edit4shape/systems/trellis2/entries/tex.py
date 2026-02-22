"""
Trellis2 Tex 训练系统（专注于 Tex 阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 纹理生成系统训练，支持从单张图像生成 3D 模型的纹理。
核心流程：
- 图像条件 -> Dense Sampling -> Shape Rollout (frozen) -> Tex Rollout -> PBR 渲染 -> Guidance Loss

特性：
- 专注 Tex 阶段训练：使用 PBR 渲染监督纹理
- Shape 阶段使用冻结的模型生成几何
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_tex: 执行 Tex 阶段的去噪采样
4. trellis2_tex_forward: Tex 阶段前向传播（使用 MeshPeeledRenderer 渲染 PBR）
5. evaluate: 评估循环，生成 mesh 并保存可视化结果
6. main: 训练主循环

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- MeshPeeledRenderer 渲染 PBR + IBL 着色（支持梯度）

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
import sys
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
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import (
    trellis2_shape_forward,
    trellis2_tex_forward,
    evaluate as _evaluate,
)


def build_system(cfg, accelerator, guidance_factory):
    """构建 Tex-only 训练系统（Shape 冻结）。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="tex")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Tex 评估（含 Shape Forward + Tex Forward）。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=True)


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    只训练 Tex Flow Model，使用 PBR 渲染监督纹理。
    Shape 阶段使用冻结的模型生成几何。
    
    流程: Dense Sampling → Shape Rollout (frozen) → Tex Rollout → PBR 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2.tex --config=configs/trellis2_tex.py
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
    use_wandb = cfg.use_wandb #getattr(cfg, 'use_wandb', False)
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
        run_name = cfg.run_name #getattr(cfg, 'run_name', 'trellis2-tex-distillation')
        accelerator.init_trackers(
            project_name="trellis2-tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": run_name}},
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
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    
    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system,
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
    # Step 8: 训练循环（只训练 Tex 阶段）
    # =====================================================
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
        # ---- 计算总 loss ----
        # guidance.loss 在 Guidance 设备上，需要移到训练设备
        guidance_loss = state.guidance.loss.to(accelerator.device) * cfg.tex.train.loss.guidance  # ()
        total = guidance_loss  # ()
        if state.regularization.reg_loss is not None:
            total = total + cfg.tex.train.loss.reg * state.regularization.reg_loss  # ()
        
        # ---- 反向传播 ----
        accelerator.backward(total)
        
        # ---- 构建日志（直接复用 loss_dict）----
        logs = {f"loss/{k}": v.item() for k, v in (state.guidance.loss_dict or {}).items() if v is not None}
        logs["loss/total"] = total.item()
        if state.regularization.reg_loss is not None:
            logs["loss/reg"] = state.regularization.reg_loss.item()
        if state.regularization.reg_metric is not None:
            logs["loss/reg_metric"] = state.regularization.reg_metric
        return logs
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            # ============================================
            # Shape Forward（冻结，只用于生成几何）
            # ============================================
            with torch.no_grad():
                _ = trellis2_shape_forward(
                    system, state, global_step,
                    is_training=False  # Shape 阶段不训练
                )
            
            # ============================================
            # Tex Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_render_out = trellis2_tex_forward(
                        system, state, global_step,
                        is_training=True
                    )
                    tex_rgb = tex_render_out["color"]  # (B, V, H, W, 3) - RGB 图
                    
                    # Tex Guidance（使用 RGB 监督纹理）
                    tex_guidance_result = system.guidance.compute_guidance(
                        tex_rgb,
                        state.views_conditioned.image_pils,
                        guidance_cfg=cfg.tex.guidance,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(tex_guidance_result)
                    
                    # Tex Loss & Backward
                    tex_log = _compute_loss_and_backward(state)
                
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # ============================================
            # Logging
            # ============================================
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 PBR 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 的计算图和碎片缓存，防止 OOM
            del state, tex_render_out, tex_guidance_result, tex_log
            torch.cuda.empty_cache()
        
        # ============================================
        # Epoch 结束后：周期性评估和检查点保存
        # ============================================
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)
        
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)