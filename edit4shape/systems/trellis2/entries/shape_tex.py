"""
Trellis2 Shape+Tex 双阶段训练系统。

本模块实现了基于 TRELLIS.2 架构的 3D 生成系统训练，同时训练 Shape 和 Tex 两个阶段。
核心流程：
- Stage 1 (Shape): 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss
- Stage 2 (Tex): Tex Rollout -> MeshWithVoxel -> PBR Voxel 渲染 -> Guidance Loss

特性：
- 双阶段同时训练：Shape 阶段用 Normal 渲染监督几何，Tex 阶段用 PBR Voxel 渲染监督纹理
- 每个 batch 分两步计算 Guidance Loss
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. Trellis2System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape / rollout_tex: 执行 Shape/Tex 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（渲染 Mesh Normal）
5. trellis2_tex_forward: Tex 阶段前向传播（使用 MeshPeeledRenderer 渲染 PBR）
6. evaluate: 评估循环，生成 mesh 并保存可视化结果
7. main: 训练主循环（依次执行 Shape Guidance 和 Tex Guidance）

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- Shape 阶段: MeshRenderer 直接渲染 normal（支持梯度）
- Tex 阶段: MeshPeeledRenderer 渲染 PBR + IBL 着色（支持梯度）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

import os
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
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper
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
    """构建 Shape+Tex 双阶段训练系统。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="shape_tex")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Shape+Tex 双阶段评估。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=True)


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    同时训练 Shape 和 Tex 两个 Flow Model。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理
    
    流程: Dense Sampling → Shape Rollout → Tex Rollout → RGB 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2.shape_tex --config=config/trellis2_shape_tex_distillation.py
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
            project_name="trellis2-shape+tex-distillation",
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
    # Step 8: 训练循环
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    # ★ 自适应梯度裁剪（TRELLIS.2 默认参数：max_norm=1.0, clip_percentile=95）
    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    
    def _compute_loss_and_backward(state: Trellis2State, stage_train_cfg) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。

        Args:
            state: 当前训练状态
            stage_train_cfg: 当前阶段的 train 配置（cfg.shape.train 或 cfg.tex.train）
        """
        # ---- 计算总 loss ----
        # guidance.loss 在 Guidance 设备上，需要移到训练设备
        guidance_loss = state.guidance.loss.to(accelerator.device) * stage_train_cfg.loss.guidance  # ()
        total = guidance_loss  # ()
        if state.regularization.reg_loss is not None:
            total = total + stage_train_cfg.loss.reg * state.regularization.reg_loss  # ()
        
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
            
            # # ★ DEBUG: 开启 detect_anomaly 以获取详细的 backward 错误信息
            # with torch.autograd.set_detect_anomaly(True):
            
            # ============================================
            # Stage 1: Shape Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_render_out = trellis2_shape_forward(
                            system, state, global_step,
                            is_training=True
                        )
                    shape_normal = shape_render_out["color"]  # (B, V, H, W, 3) - Normal 图
                    
                    # Shape Guidance（使用 Normal 监督几何）
                    shape_guidance_result = system.guidance.compute_guidance(
                        shape_normal,
                        state.views_conditioned.image_pils,
                        guidance_cfg=cfg.shape.guidance,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(shape_guidance_result)
                    
                    # Shape Loss & Backward
                    shape_log = _compute_loss_and_backward(state, cfg.shape.train)
                
                if accelerator.sync_gradients:
                    shape_grad_clipper(system.shape.model.parameters())
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # ★ 保存 Shape 可视化（在 Tex guidance 覆盖 views_edited 之前）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)
            
            # ============================================
            # Stage 2: Tex Forward → Backward → Update
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
                    tex_log = _compute_loss_and_backward(state, cfg.tex.train)
                
                if accelerator.sync_gradients:
                    tex_grad_clipper(system.tex.model.parameters())
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # ============================================
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # ★ 保存 Tex 可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)
            
            # 释放当前 step 残留引用
            del state, shape_render_out, shape_guidance_result, shape_log
            del tex_render_out, tex_guidance_result, tex_log
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
