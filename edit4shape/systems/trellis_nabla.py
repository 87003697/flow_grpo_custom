"""
Trellis Nabla 版 - 基于 Score Matching 的训练系统。

本模块实现 Nabla-R2D3 风格的 Score Matching 训练，与 trellis.py 的主要区别：
- 使用 SDE 采样（rollout_sparse_sde）而非 ODE 采样
- 使用 Score Matching Loss 而非直接的 Guidance Loss
- 支持 RolloutTracker 记录采样轨迹

共享组件（从 trellis.py 导入）：
- build_dataloaders: 数据加载器构建
- decode_and_render_mesh/gs: 解码和渲染函数
- trellis_forward: 标准前向传播（用于评估）
- evaluate: 评估循环

独有组件：
- build_system: 系统构建（与 trellis.py 相同，但保留用于未来扩展）
- main: Nabla 训练主循环（使用 SDE rollout + Score Matching）
"""

# =====================================================================
# 标准库导入
# =====================================================================
from pathlib import Path
from typing import Any, Dict

# =====================================================================
# 第三方库导入
# =====================================================================
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator

# =====================================================================
# 项目内部导入
# =====================================================================

# 使用 absl 的 config_flags 管理配置文件
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")

# =====================================================================
# TRELLIS 参考实现路径设置
# =====================================================================
import os
import sys
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    TrainModeGuard,
    EvalModeGuard,
    System,
    CheckpointIO,
    build_run_paths,
    compute_guidance_device,
)
from edit4shape.systems.utils import MetricLogger, VisualIO

# =====================================================================
# 从 trellis.py 导入共享组件
# =====================================================================
from edit4shape.systems.trellis import (
    # 数据加载
    build_dataloaders,
    # 渲染函数
    decode_and_render_mesh,
    decode_and_render_gs,
    # 前向传播和评估
    trellis_forward,
    evaluate,
)

# =====================================================================
# 从 state 模块导入 TrellisState
# =====================================================================
from edit4shape.generators.trellis.state import TrellisState

# =====================================================================
# 从 rollout 模块导入采样函数
# =====================================================================
from edit4shape.generators.trellis.rollout import (
    rollout_sparse,           # ODE 采样（用于评估）
    rollout_sparse_sde,       # SDE 采样（用于 Nabla 训练）
    compute_score_matching_loss,  # Score Matching Loss
)

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance

# SparseTensor 类型（用于 reward_gradients）
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Nabla Loss 计算
# =====================================================================

def _compute_nabla_loss_and_backward(
    state: TrellisState,
    system: System,
    cfg,  # ml_collections.ConfigDict
    device: torch.device,
    reward_gradients: SparseTensor,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    """
    Nabla 版本的 loss 计算和反向传播。
    
    使用 Score Matching Loss 而非直接的 Guidance Loss。
    
    Args:
        state: 状态对象（含 tracker.rollout）
        system: 系统组件
        cfg: 配置对象，需包含:
            - cfg.train.loss.score_matching: score matching loss 权重
            - cfg.nabla.*: nabla 相关配置
        device: 运行设备
        reward_gradients: reward 梯度 SparseTensor
        accelerator: 分布式加速器
    
    Returns:
        logs: 日志字典
    """
    # ---- 计算 Score Matching Loss ----
    score_loss = compute_score_matching_loss(
        state=state,
        system=system,
        cfg=cfg,
        device=device,
        reward_gradients=reward_gradients,
    )
    
    # ---- 加权 ----
    score_weight = cfg.train.loss.score_matching
    total = score_loss * score_weight
    
    # ---- 反向传播 ----
    accelerator.backward(total)
    
    # ---- 构建日志 ----
    logs = {
        "loss/score_matching": score_loss.item(),
        "loss/total": total.item(),
    }
    
    return logs


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict, 
    accelerator: Accelerator,
    guidance_factory: callable,
) -> System:
    """
    构建完整的 Trellis Nabla 系统。
    
    与 trellis.py 的 build_system 结构相同，但保留为独立函数以便未来扩展 Nabla 特定配置。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        System: 包含所有组件的系统实例
    """
    # ---- 1. 构建 Pipeline (核心生成管道) ----
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    pipeline = build_pipeline_from_reference(cfg, accelerator)

    # ---- 2. 构建 Renderer (3D 渲染器) ----
    # 根据配置选择渲染方式：
    # - "mesh": 基于 nvdiffrast 的可微分网格光栅化
    # - "gs": 基于 3D Gaussian Splatting 的渲染
    renderer_type = cfg.renderer.type  # "mesh" 或 "gs"
    
    if renderer_type == "gs":
        from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
        rendering_options = {
            "resolution": cfg.renderer.resolution,  # 渲染分辨率 (像素)
            "near": cfg.renderer.near,  # 近裁剪面距离
            "far": cfg.renderer.far,    # 远裁剪面距离
            "ssaa": cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            "bg_color": cfg.renderer.bg_color,  # 背景色模式
        }
        renderer = GaussianRenderer(rendering_options)
    else:
        # ---- Mesh 光栅化渲染器 (nvdiffrast) ----
        # 优势：支持精确的几何渲染，法线/深度图质量高
        # 适用场景：训练、精细渲染
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
        renderer_cfg = TrellisRendererConfig(
            resolution=cfg.renderer.resolution,  # 渲染分辨率 (像素)
            ssaa=cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            near=cfg.renderer.near,  # 近裁剪面距离
            far=cfg.renderer.far,    # 远裁剪面距离
        )
        renderer = TrellisMeshRasterizer(cfg=renderer_cfg, device=str(accelerator.device))

    # ---- 3. 构建 Guidance、Strategy 和 Optimizer ----
    # 仅在训练模式下创建
    guidance = None
    optimizer = None
    strategy = None

    if not cfg.eval_only:
        # 3a. 使用工厂函数创建 Guidance
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        # 3b. 创建训练策略
        from edit4shape.systems.utils.strategy import LoRAStrategy, FrozenStrategy
        from edit4shape.generators.trellis.training_adpter import (
            register_sparse_linear_with_peft,
            inject_lora_to_slat,
            build_optimizer_for_slat,
            TrellisFullFinetuneStrategy,
        )
        
        train_mode = cfg.train.get("mode", "full")  # 默认全参微调
        train_device = accelerator.device
        teacher_device = compute_guidance_device(accelerator.device)
        
        # 根据训练模式创建策略
        if train_mode == "full":
            strategy = TrellisFullFinetuneStrategy(
                pipeline, train_device, teacher_device, cfg.pretrained.model
            )
        elif train_mode == "lora":
            register_sparse_linear_with_peft()
            inject_lora_to_slat(pipeline, cfg.lora)
            strategy = LoRAStrategy(pipeline, train_device, teacher_device)
        else:
            strategy = FrozenStrategy(pipeline, train_device, teacher_device)
        
        strategy.setup()
        
        # 3c. 启用 slat_flow_model 的 Gradient Checkpointing（节省显存）
        slat_model = pipeline.pipe.models['slat_flow_model']
        for block in slat_model.blocks:
            block.use_checkpoint = True
        
        # 3d. 为学生模型创建优化器
        optimizer = build_optimizer_for_slat(strategy.student, cfg.train.optimizer)

    return System(
        pipeline=pipeline, 
        renderer=renderer, 
        guidance=guidance, 
        optimizer=optimizer, 
        strategy=strategy,
    )


# =====================================================================
# Nabla 前向传播 - 使用 SDE 采样
# =====================================================================

def trellis_forward_sde(
    system: System,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
) -> Dict[str, Any]:
    """
    Trellis Nabla 前向传播：Dense Sampling → SDE Rollout → Decode → Render
    
    与 trellis_forward 的区别：
    - 使用 rollout_sparse_sde 进行 SDE 采样
    - 自动记录采样轨迹到 state.tracker.rollout
    
    Args:
        system: 系统组件
        state: TrellisState 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
    
    Returns:
        render_out: 渲染输出字典
    """
    pipeline = system.pipeline
    
    # ---- 1. Dense Sampling（结构生成）----
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()
    with torch.no_grad():
        cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)
    state.coords = coords
    
    # ---- 2. SDE Rollout（记录轨迹）----
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    
    rollout_sparse_sde(
        state=state,
        cfg=cfg,
        system=system,
        device=device,
        generator=generator,
    )
    latents = state.features.slat  # SparseTensor (挂载于 rollout_sparse)
    
    # ---- 3. 解码 & 渲染 ----
    renderer_type = cfg.renderer.type
    
    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, system.renderer, device
        )  # dict with "color": (B,V,H,W,C), "gaussians": list
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, system.renderer, device
        )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
    
    state.views_generated.image_tensor = render_out["color"]  # (B,V,H,W,C) 挂载生成图用于可视化
    
    return render_out


# =====================================================================
# 主函数入口 - Nabla 训练
# =====================================================================

def main(argv) -> None:
    """
    Nabla 训练主入口。
    
    与 trellis.py 的 main 主要区别：
    - 使用 trellis_forward_sde 进行 SDE 采样
    - 使用 compute_score_matching_loss 计算 Score Matching Loss
    - 训练目标是匹配 reward-adjusted transition score
    """
    del argv  # absl.app.run 会传入 argv；本函数不使用
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator（含 wandb 日志）
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
    
    # 初始化 wandb trackers
    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis-nabla",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(
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
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
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
    # Step 7: 评估模式（仅评估不训练）
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
    # Step 8: Nabla 训练循环
    # =====================================================
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    
    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            
            # 使用 accumulate 上下文管理器处理梯度累积
            with accelerator.accumulate(pipe_models['slat_flow_model']):
                # ---- 在 TrainModeGuard 下执行训练 ----
                with TrainModeGuard(
                    pipe_models['slat_flow_model'],
                    pipe_models['slat_decoder_mesh'],
                    pipe_models['slat_decoder_gs'],
                ):
                    # ---- 创建状态并挂载数据 ----
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=pipeline)  # 挂载所有数据
                    
                    # ---- SDE 前向传播（记录轨迹）----
                    render_out = trellis_forward_sde(
                        system, state, cfg, accelerator.device, global_step
                    )
                    comp_rgb = render_out["color"]  # (B,V,H,W,C)
                    
                    # ---- 计算 Guidance（获取 reward gradients）----
                    guidance_result = system.guidance.compute_guidance(
                        comp_rgb, 
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(guidance_result)
                    
                    # ---- 计算 reward gradients ----
                    # TODO: 实现 compute_reward_gradients，从 guidance_result 中提取 reward 梯度
                    # 目前使用零梯度作为占位符
                    reward_gradients = state.features.slat.replace(
                        torch.zeros_like(state.features.slat.feats)
                    )  # SparseTensor, 占位符
                    
                    # ---- 计算 Score Matching Loss 并反向传播 ----
                    train_log = _compute_nabla_loss_and_backward(
                        state=state,
                        system=system,
                        cfg=cfg,
                        device=accelerator.device,
                        reward_gradients=reward_gradients,
                        accelerator=accelerator,
                    )
                
                # ---- 优化器步进 ----
                if accelerator.sync_gradients:
                    system.optimizer.step()
                    system.optimizer.zero_grad()
            
            # 仅主进程按频率保存可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(
                    state=state,
                    epoch=epoch,
                    step=global_step,
                    pipe=system.guidance.pipe if system.guidance else None,
                    n_progress_samples=cfg.freq.save.progress_samples,
                )
            
            # 自动累积并在 sync_gradients 时发射平均日志
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
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
