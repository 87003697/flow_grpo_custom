"""
ODE Rollout - 标准 Euler 采样

用于推理和 ReFL/DRaFT 风格的训练（可选 x0/v 正则化）。
支持三阶段 Autograd 模式（通过 tracker 参数）。
"""

from typing import Optional, Any
import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator
import ml_collections

from trellis.modules.sparse import SparseTensor

from .base import (
    predict_sparse_velocity_with_cfg, _predict_sparse_cond_velocity, mix_cfg_sparse,
    prepare_embeddings,
    predict_dense_velocity_with_cfg,
)
from .autograd_tracker import RolloutTracker


# =====================================================================
# 类型别名（避免循环导入）
# =====================================================================
TrellisState = Any
System = Any


# =====================================================================
# Rollout - 核心采样循环（训练/评估共用）
# =====================================================================

def rollout_sparse(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
    tracker: Optional[RolloutTracker] = None,
) -> Optional[RolloutTracker]:
    """
    稀疏特征去噪采样（SLAT Stage 2）。
    
    核心流程: x_T (噪声) → 迭代去噪 → x_0 (特征) → 反归一化
    
    三阶段 Autograd 模式（tracker is not None）:
        - 模型推理在 torch.no_grad() 下执行（is_training 应设为 False）
        - cond_pred 上插入 proxy 节点（requires_grad=True）
        - proxy → CFG → velocity → scheduler 构建轻量 proxy chain
        - reg_loss 通过 proxy chain 连接到 autograd 图
        - reg 梯度通过 autograd.grad 提前计算，存入 tracker.reg_grads
    
    Args:
        state: TrellisState 状态对象，包含条件编码、坐标等
        cfg: 配置对象，cfg.rollout.reg.type ("none"|"x0"|"v")
        system: 系统组件（pipeline、renderer 等）
        device: 运行设备
        generator: 随机数生成器（用于可复现性）
        is_training: 是否为训练模式（tracker 模式下应为 False）
        tracker: 三阶段 Autograd 的 proxy 记录器。传入时启用 proxy chain 模式。
    
    Returns:
        tracker: 如果传入了 tracker，返回填充后的 tracker；否则返回 None。
    
    Side Effects:
        - state.stage2.z0: 挂载反归一化后的 SparseTensor
        - state.stage2.reg_loss: 挂载 reg_loss
    """
    pipeline = system.pipeline
    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = pipeline.sparse.get_runtime_params()
    
    # ---- 1. 初始化 ----
    cond_emb, uncond_emb = prepare_embeddings(state, device)
    
    assert state.coords is not None, "state.coords 缺失"
    assert generator is not None, "generator 必须由调用方提供"

    # ★ 使用 SparseTensor 贯穿整个流程（对齐 trellis2 实现）
    x_t = pipeline.sparse.init_latents(
        coords=state.coords,
        in_channels=pipeline.sparse.resolve_flow_module().in_channels,
        generator=generator
    )  # SparseTensor
    
    # ---- 2. Scheduler 配置 ----
    scheduler = pipeline.sparse.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    
    # ---- 3. 正则化配置 ----
    reg_type = cfg.rollout.reg.type
    reg_enabled = reg_type != "none" and (is_training or tracker is not None)
    
    reg_loss_sum = 0.0
    
    # ---- 4. 去噪循环 ----
    steps = list(scheduler.timesteps)[:-1]
    steps_iter = tqdm(steps, desc="Rollout", leave=False,
                      disable=not (is_training or tracker is not None) or not Accelerator().is_main_process)
    
    B = cond_emb.shape[0]  # ()
    
    for t in steps_iter:
        t_val = float(t.item())
        t_norm = t_val / 1000.0
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
        use_cfg = cfg_min <= t_val <= cfg_max
        
        if tracker is not None:
            # ============================================================
            # ★ 三阶段 Autograd 模式：分步推理 + proxy 插入
            # ============================================================
            
            # ---- 条件速度预测（no_grad，仅推理）----
            with torch.no_grad():
                cond_pred = _predict_sparse_cond_velocity(
                    pipeline, x_t, t_batch, cond_emb
                )  # SparseTensor
            
            # ---- Proxy 插入（cond_pred 层级，CFG 之前）----
            tracker.timesteps.append(t_val)
            tracker.input_trajectory.append(x_t.feats.detach().clone())  # (N, C)
            cond_proxy = cond_pred.feats.detach().clone().requires_grad_(True)  # (N, C)
            tracker.output_trajectory.append(cond_proxy)
            cond_pred = cond_pred.replace(cond_proxy)  # SparseTensor with proxy feats
            
            # ---- CFG 混合（proxy 之后，velocity 依赖 cond_proxy）----
            if use_cfg and uncond_emb is not None:
                with torch.no_grad():
                    uncond_pred = _predict_sparse_cond_velocity(
                        pipeline, x_t, t_batch, uncond_emb
                    )  # SparseTensor
                velocity = mix_cfg_sparse(
                    cond_pred, uncond_pred, slat_guidance, uncond_mode="detach"
                )  # SparseTensor，feats 依赖 cond_proxy
            else:
                velocity = cond_pred  # SparseTensor，feats == cond_proxy
            
            # ---- 正则化（velocity 通过 proxy chain 有梯度）----
            if reg_enabled:
                with system.strategy.sparse_teacher_context(), torch.no_grad():
                    teacher_vel = predict_sparse_velocity_with_cfg(
                        pipeline, x_t, t_val, cond_emb, uncond_emb,
                        slat_guidance, cfg_min, cfg_max, device,
                    )  # SparseTensor
                
                if reg_type == "x0":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)，依赖 proxy
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x0_regularization(x0_stu, x0_tea, t_norm)
                elif reg_type == "x1":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)，依赖 proxy
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x1_regularization(x0_stu, x0_tea)
                elif reg_type == "v":
                    reg_loss = _compute_v_regularization(velocity.feats, teacher_vel.feats)
                else:
                    raise ValueError(f"Unknown reg_type: {reg_type}")
                
                reg_loss_sum = reg_loss_sum + reg_loss
        
        elif is_training:
            # ============================================================
            # 原始训练模式：端到端计算图
            # ============================================================
            velocity = predict_sparse_velocity_with_cfg(
                pipeline, x_t, t_val, cond_emb, uncond_emb,
                slat_guidance, cfg_min, cfg_max, device,
            )  # SparseTensor
            
            if reg_enabled:
                with system.strategy.sparse_teacher_context(), torch.no_grad():
                    teacher_vel = predict_sparse_velocity_with_cfg(
                        pipeline, x_t, t_val, cond_emb, uncond_emb,
                        slat_guidance, cfg_min, cfg_max, device,
                    )  # SparseTensor
                
                if reg_type == "x0":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x0_regularization(x0_stu, x0_tea, t_norm)
                elif reg_type == "x1":
                    x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)
                    x0_tea = x_t.feats - t_norm * teacher_vel.feats  # (N, C)
                    reg_loss = _compute_x1_regularization(x0_stu, x0_tea)
                elif reg_type == "v":
                    reg_loss = _compute_v_regularization(velocity.feats, teacher_vel.feats)
                else:
                    raise ValueError(f"Unknown reg_type: {reg_type}")
                
                reg_loss_sum = reg_loss_sum + reg_loss
        
        else:
            # ============================================================
            # 推理模式：no_grad
            # ============================================================
            with torch.no_grad():
                velocity = predict_sparse_velocity_with_cfg(
                    pipeline, x_t, t_val, cond_emb, uncond_emb,
                    slat_guidance, cfg_min, cfg_max, device,
                )  # SparseTensor
        
        # ---- Scheduler 步进（使用 SparseTensor）----
        x_t = scheduler.step(velocity, t, x_t).prev_sample  # SparseTensor
    
    # ---- 5. 反归一化 ----
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats = x_t.feats * std + mean  # (N, C)
    
    # ---- 6. 挂载到 state ----
    state.stage2.z0 = x_t.replace(denorm_feats)  # SparseTensor with denormalized feats
    
    # ---- 7. 正则化处理 ----
    num_steps = max(1, len(steps))
    if reg_enabled:
        reg_loss_avg = reg_loss_sum / num_steps
        
        if tracker is not None and len(tracker.output_trajectory) > 0:
            # ★ 三阶段模式：提前用 autograd.grad 算好 reg 梯度，存入 tracker
            reg_grads = torch.autograd.grad(
                reg_loss_avg,
                tracker.output_trajectory,
                retain_graph=True,  # slat 共享 proxy chain，Phase 2 还需要
            )
            tracker.reg_grads = [g.detach().clone() for g in reg_grads]  # T × (N, C)
            tracker.reg_loss_val = reg_loss_avg.item()
            # 保留 reg_loss（含图）供 Phase 2 合并 backward
            state.stage2.reg_loss = reg_loss_avg
        else:
            state.stage2.reg_loss = reg_loss_avg
    else:
        state.stage2.reg_loss = None
    
    return tracker


# =====================================================================
# Rollout Dense — Stage 1 (sparse_structure_flow_model)
# =====================================================================

def rollout_dense(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    Dense 特征去噪采样（Stage 1 — sparse_structure_flow_model）。

    核心流程: x_T (噪声) → 迭代去噪 → x_0 (z_s)
    Stage 1 无 normalization，raw latent 直接作为 z₀。

    仅支持推理模式（no_grad），用于 contrastive 训练中获取 teacher z_s。

    Args:
        state: 状态对象，包含条件编码
        cfg: 配置对象
        system: 系统组件（pipeline 等）
        device: 运行设备
        generator: 随机数生成器

    Side Effects:
        - state.stage1.z0: 挂载 Dense Tensor (B, C, R, R, R)
    """
    pipeline = system.pipeline
    ss_steps, ss_guidance, ss_rescale_t, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

    # ---- 1. 初始化 ----
    cond_emb, uncond_emb = prepare_embeddings(state, device)

    assert generator is not None, "generator 必须由调用方提供"
    x_t = pipeline.dense.init_latents(batch_size=1, generator=generator)  # (1, C, R, R, R)

    # ---- 2. 时间步序列 ----
    _, t_pairs = pipeline.dense.scheduler(ss_steps, ss_rescale_t)

    # ---- 3. 去噪循环 ----
    with torch.no_grad():
        for t, t_prev in t_pairs:
            t_val = float(t)
            velocity = predict_dense_velocity_with_cfg(
                pipeline, x_t, t_val, cond_emb, uncond_emb,
                ss_guidance, ss_cfg_min, ss_cfg_max, device,
            )  # (B, C, R, R, R)

            # Euler step: x_{t-1} = x_t - (t - t_prev) * v
            delta = t_val - float(t_prev)
            x_t = x_t - delta * velocity

    # ---- 4. 挂载到 state（无 normalization）----
    state.stage1.z0 = x_t  # (B, C, R, R, R)


# =====================================================================
# 正则化函数（x0 / v）
# =====================================================================

def _compute_x0_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    t_norm: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    x0 正则化 Loss：MSE(x0_stu, x0_tea) / t²
    
    除以 t² 使其与 v 正则化在数值上等价。
    梯度可流向历史步（通过 x_t 中累积的计算图）。
    
    Args:
        x0_student: 学生模型预测的 x0 (N, C)
        x0_teacher: 教师模型预测的 x0 (N, C)
        t_norm: 归一化时间步 (0~1)
        eps: 防止除零
        
    Returns:
        loss: 标量
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    mse = (diff ** 2).mean()  # scalar
    return mse / (t_norm ** 2 + eps)  # scalar


def _compute_x1_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    x1 正则化 Loss：MSE(x0_stu, x0_tea)，不除以 t²。
    
    与 x0 正则化的区别：去掉 1/t² 缩放，
    使得小 t（接近 x0）时的正则化权重不会被放大。
    
    Args:
        x0_student: 学生模型预测的 x0 (N, C)
        x0_teacher: 教师模型预测的 x0 (N, C)
        
    Returns:
        loss: 标量
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar


def _compute_v_regularization(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    v 正则化 Loss：MSE(v_stu, v_tea)
    
    直接对速度场做 MSE，梯度仅流向当前步的模型调用。
    
    Args:
        v_student: 学生模型预测的速度 (N, C)
        v_teacher: 教师模型预测的速度 (N, C)
        
    Returns:
        loss: 标量
    """
    diff = v_student - v_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar
