from __future__ import annotations

# =====================================================================
# Imports
# =====================================================================
from typing import Optional, TYPE_CHECKING

import ml_collections
import torch
from accelerate import Accelerator
from tqdm import tqdm

from edit4shape.generators.trellis2.state import DebugTracker, Trellis2State
from edit4shape.generators.trellis2.rollout.base import (
    _compute_regularization,
    _predict_velocity,
    trellis2_cfg_sparse,
)

if TYPE_CHECKING:
    from edit4shape.systems.trellis2 import Trellis2System

# =====================================================================
# Rollout - Tex 阶段
# =====================================================================

def rollout_tex(
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    system: Trellis2System,
    device: torch.device,
    resolution: int = 1024,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> DebugTracker:
    """
    Tex 阶段去噪采样。
    
    Args:
        state: Trellis2State，包含条件编码、坐标、shape_slat 等
        cfg: 配置对象
        system: 系统组件
        device: 运行设备
        resolution: 模型分辨率（512 或 1024）
        generator: 随机数生成器
        is_training: 是否为训练模式
    
    Returns:
        DebugTracker: 包含每步中间变量的跟踪器
    
    Side Effects:
        - state.features.tex_slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 更新 reg_loss 和 reg_metric
    """
    tracker = DebugTracker()
    pipeline = system.pipeline
    stage = "tex"
    
    # ---- 1. 获取采样参数 ----
    sampler_params = pipeline.get_sampler_params(stage)
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["guidance_strength"])
    cfg_rescale = float(sampler_params["guidance_rescale"])
    cfg_min, cfg_max = pipeline.get_cfg_interval(stage)
    sigma_min = pipeline.pipe.tex_slat_sampler.sigma_min  # 从 sampler 获取 sigma_min
    
    # ---- 2. 初始化 ----
    # Tex Rollout 使用 flow_resolution 对应的条件编码（对齐参考实现）
    cond_emb, uncond_emb = state.extract_embeddings(resolution=resolution)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)
    
    assert state.coords is not None, "state.coords 缺失"
    assert state.features.shape_slat is not None, "shape_slat 缺失，需先执行 rollout_shape"
    
    # ★ generator 处理：
    # - None: 使用全局种子（与参考实现一致，适用于 eval）
    # - 显式传入: 使用传入的 Generator（适用于可控的训练）
    # 注意：不自动创建 Generator，保持与参考实现的行为一致
    
    # 使用已 normalized 且已 detach 的 shape_slat 作为 tex 的条件
    # shape_slat_norm 在 rollout_shape 挂载时已 detach 并清空缓存
    shape_cond = state.features.shape_slat_norm
    
    # ★ 使用 SparseTensor 贯穿整个流程，对齐参考实现
    x_t = pipeline.init_latents(
        coords=state.coords,
        stage=stage,
        resolution=resolution,
        generator=generator,  # 可以是 None，使用全局种子
    )  # SparseTensor
    
    # ---- 3. Scheduler 配置 ----
    scheduler = pipeline.scheduler(stage)
    scheduler.set_timesteps(steps, device=device)
    
    # ---- 4. 正则化配置 ----
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_eps = cfg.reg.eps #getattr(cfg.reg, 'eps', 1e-2)  # 兼容旧配置
    reg_enabled = reg_type != "none" and is_training
    
    # Tex 阶段独立计算正则化（不累加 shape 阶段的）
    reg_loss_sum = 0.0
    
    # ---- 5. 去噪循环 ----
    # 使用基于索引的 API 确保时间步精度与参考实现完全一致
    step_indices = scheduler.get_timesteps_for_loop()  # [0, 1, ..., steps-1]
    steps_iter = tqdm(step_indices, desc="Tex Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for step_idx in steps_iter:
        # 使用精确的 numpy float64 时间步值（对齐参考实现）
        t_val = scheduler.get_precise_t(step_idx)  # float64 精度
        t_norm = t_val  # 直接使用，scheduler.timesteps 已经是 0-1 范围
        use_cfg = cfg_min <= t_norm <= cfg_max
        
        # ---- cond 预测（使用 SparseTensor 流程） ----
        # 注：Flow Model 已启用 block-level checkpointing，无需在此处包裹 checkpoint
        if is_training:
            cond_pred = _predict_velocity(
                pipeline, x_t, t_val, cond_emb,
                stage, resolution, shape_cond
            )  # SparseTensor
        else:
            with torch.no_grad():
                cond_pred = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, shape_cond
                )  # SparseTensor
        
        # ---- uncond 预测 + CFG 混合（在 SparseTensor 上进行） ----
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, x_t, t_val, uncond_emb,
                    stage, resolution, shape_cond
                )  # SparseTensor
            velocity = trellis2_cfg_sparse(
                cond_pred, uncond_pred, cfg_strength,
                guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                sigma_min=sigma_min
            )  # SparseTensor
        else:
            velocity = cond_pred  # SparseTensor
        
        # ---- 正则化（DMD / KL）----
        if reg_enabled:
            with system.strategy.teacher_context(stage, resolution), torch.no_grad():
                teacher_cond = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, shape_cond
                )  # SparseTensor
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_velocity(
                        pipeline, x_t, t_val, uncond_emb,
                        stage, resolution, shape_cond
                    )  # SparseTensor
                    teacher_vel = trellis2_cfg_sparse(
                        teacher_cond, teacher_uncond, cfg_strength,
                        guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                        sigma_min=sigma_min
                    )  # SparseTensor
                else:
                    teacher_vel = teacher_cond  # SparseTensor
            
            # 正则化在 feats 上计算
            # 使用正确的 x0 公式（对齐参考实现 FlowEulerSampler._pred_to_xstart）：
            # x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
            coeff = sigma_min + (1 - sigma_min) * t_val  # scalar
            x0_stu = (1 - sigma_min) * x_t.feats - coeff * velocity.feats  # (N, C)
            x0_tea = (1 - sigma_min) * x_t.feats - coeff * teacher_vel.feats  # (N, C)
            
            reg_loss = _compute_regularization(
                x0_stu, x0_tea, t_norm,
                reg_type=reg_type, weight_mode=weight_mode, eps=reg_eps
            )
            reg_loss_sum = reg_loss_sum + reg_loss
        
        # ---- Scheduler 步进（使用 SparseTensor 流程） ----
        # scheduler.step_by_index 直接接收 SparseTensor，返回 SparseTensor
        x_t = scheduler.step_by_index(velocity, step_idx, x_t).prev_sample  # SparseTensor
        
        # ---- 记录调试信息 ----
        tracker.log(
            t=t_val,
            latents=x_t.feats,  # (N, C)
            velocity=velocity.feats,  # (N, C)
            cond_pred=cond_pred.feats,  # (N, C)
            uncond_pred=uncond_pred.feats if use_cfg and uncond_emb is not None else None,  # (N, C)
        )
    
    # ---- 6. 反归一化 ----
    # x_t 已经是 SparseTensor，直接使用
    tex_slat_normalized = x_t  # SparseTensor
    tex_slat = pipeline.denormalize(tex_slat_normalized, stage)  # SparseTensor
    
    # ---- 7. 挂载到 state（同时保存 normalized 和 denormalized 版本）----
    state.features.tex_slat = tex_slat  # denormalized，保留梯度用于 decode
    
    # tex_slat_norm 备用，直接 detach 切断依赖
    norm_detached = tex_slat_normalized.detach()
    norm_detached.clear_spatial_cache()
    state.features.tex_slat_norm = norm_detached
    
    num_steps = max(1, len(step_indices))
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None
    
    return tracker

