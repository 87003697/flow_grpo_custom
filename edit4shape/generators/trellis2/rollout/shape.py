from __future__ import annotations

# =====================================================================
# Imports
# =====================================================================
from typing import Optional, TYPE_CHECKING

from edit4shape.generators.trellis2.rollout.tracker import RolloutTracker

import ml_collections
import torch
from accelerate import Accelerator
from torch.utils.checkpoint import checkpoint as ckpt
from tqdm import tqdm

from edit4shape.generators.trellis2.state import Trellis2State
import torch.nn.functional as F
from edit4shape.generators.trellis2.rollout.base import (
    _predict_velocity,
    trellis2_cfg_sparse,
    _compute_v_regularization,
    _compute_x0_regularization,
)

if TYPE_CHECKING:
    from edit4shape.systems.trellis2 import Trellis2System

# =====================================================================
# Rollout - Shape 阶段
# =====================================================================

def rollout_shape(
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    system: Trellis2System,
    device: torch.device,
    resolution: int = 1024,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
    tracker: Optional[RolloutTracker] = None,
) -> Optional[RolloutTracker]:
    """
    Shape 阶段去噪采样。
    
    Args:
        state: Trellis2State，包含条件编码、坐标等
        cfg: 配置对象
        system: 系统组件
        device: 运行设备
        resolution: 模型分辨率（512 或 1024）
        generator: 随机数生成器
        is_training: 是否为训练模式
        tracker: 可选的 RolloutTracker。传入时记录每步 input/output proxy，
                 用于三阶段 Autograd 架构。slat 将包含 proxy chain（有 autograd 图）。
                 ⚠️ 使用 tracker 时，调用方不可包裹 torch.no_grad()，
                 否则 proxy chain 的 autograd 图无法构建。
    
    Returns:
        传入的 tracker（已填充轨迹数据），或 None（未传入 tracker 时）。
    
    Side Effects:
        - state.features.shape_slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 挂载 reg_loss 和 reg_metric
    """
    pipeline = system.pipeline
    stage = "shape"
    
    # ---- 1. 获取采样参数 ----
    sampler_params = pipeline.get_sampler_params(stage)
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["guidance_strength"])
    cfg_rescale = float(sampler_params["guidance_rescale"])
    cfg_min, cfg_max = pipeline.get_cfg_interval(stage)
    sigma_min = pipeline.pipe.shape_slat_sampler.sigma_min  # 从 sampler 获取 sigma_min
    
    # ---- 2. 初始化 ----
    # Shape Rollout 使用 flow_resolution 对应的条件编码（对齐参考实现）
    cond_emb, uncond_emb = state.extract_embeddings(resolution=resolution)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)
    
    assert state.coords is not None, "state.coords 缺失"
    
    # ★ generator 处理：
    # - None: 使用全局种子（与参考实现一致，适用于 eval）
    # - 显式传入: 使用传入的 Generator（适用于可控的训练）
    # 注意：不自动创建 Generator，保持与参考实现的行为一致
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
    
    # ---- 4. 正则化配置（对齐 trellis.py ode.py 结构）----
    # reg_type: "none" | "x0" | "v"
    #   - "v": 对 CFG velocity 做 MSE（梯度仅当前步）
    #   - "x0": 对 x0 预测做 MSE / t²（梯度可流向历史步）
    reg_type = cfg.reg.type
    reg_enabled = reg_type in ("v", "x0") and (is_training or tracker is not None)
    
    reg_loss_sum = 0.0
    
    # ---- 5. 去噪循环 ----
    # 使用基于索引的 API 确保时间步精度与参考实现完全一致
    step_indices = scheduler.get_timesteps_for_loop()  # [0, 1, ..., steps-1]
    num_steps = max(1, len(step_indices))
    steps_iter = tqdm(step_indices, desc="Shape Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for step_idx in steps_iter:
        # 使用精确的 numpy float64 时间步值（对齐参考实现）
        t_val = scheduler.get_precise_t(step_idx)  # float64 精度
        t_norm = t_val  # 直接使用，scheduler.timesteps 已经是 0-1 范围
        use_cfg = cfg_min <= t_norm <= cfg_max
        
        # ---- cond 预测（使用 SparseTensor 流程） ----
        if is_training:
            # step-level gradient checkpoint：释放 flow model 中间激活，backward 时重算
            cond_pred_feats = ckpt(
                lambda *a: _predict_velocity(*a).feats,
                pipeline, x_t, t_val, cond_emb, stage, resolution, None,
                use_reentrant=False,
            )  # (N, C)
            cond_pred = x_t.replace(cond_pred_feats)  # SparseTensor
        else:
            with torch.no_grad():
                cond_pred = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, None
                )  # SparseTensor
        
        # ---- Tracker: proxy 建在 cond_pred 上（CFG 之前）----
        # ★ proxy 建在 cond_pred 而非 CFG 后的 velocity 上：
        #   Phase 2 backward 沿 scheduler → CFG → cond_proxy chain 反传
        #   → cond_proxy.grad 自动包含 CFG 缩放因子
        #   → Phase 3 只需重算 cond_pred，无需 uncond / CFG 混合
        if tracker is not None:
            tracker.timesteps.append(t_val)  # float64 精度
            tracker.input_trajectory.append(x_t.feats.detach().clone())  # (N, C)
            cond_proxy = cond_pred.feats.detach().clone().requires_grad_(True)  # (N, C)
            tracker.output_trajectory.append(cond_proxy)
            cond_pred = cond_pred.replace(cond_proxy)  # 用 proxy 替换 cond_pred
        
        # ---- uncond 预测 + CFG 混合（在 SparseTensor 上进行） ----
        # 当 tracker 存在时，cond_pred 已被 proxy 替换，
        # CFG 混合结果 velocity 依赖 cond_proxy → 构建 proxy chain
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, x_t, t_val, uncond_emb,
                    stage, resolution, None
                )  # SparseTensor
            velocity = trellis2_cfg_sparse(
                cond_pred, uncond_pred, cfg_strength,
                guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                sigma_min=sigma_min
            )  # SparseTensor
        else:
            velocity = cond_pred  # SparseTensor
        
        # ---- 正则化（对齐 trellis.py ode.py 结构）----
        # teacher 使用 CFG velocity，reg 统一对 velocity 计算，tracker/非 tracker 一致
        if reg_enabled:
            with system.strategy.teacher_context(stage, resolution), torch.no_grad():
                teacher_cond = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, None
                )  # SparseTensor
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_velocity(
                        pipeline, x_t, t_val, uncond_emb,
                        stage, resolution, None
                    )  # SparseTensor
                    teacher_vel = trellis2_cfg_sparse(
                        teacher_cond, teacher_uncond, cfg_strength,
                        guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                        sigma_min=sigma_min
                    )  # SparseTensor
                else:
                    teacher_vel = teacher_cond
            
            if reg_type == "x0":
                # x0 正则化：x_t 不 detach，梯度可流向历史步
                x0_stu = x_t.feats - t_norm * velocity.feats  # (N, C)
                x0_tea = x_t.feats.detach() - t_norm * teacher_vel.feats  # (N, C)
                reg_loss = _compute_x0_regularization(x0_stu, x0_tea, t_norm)
            elif reg_type == "v":
                reg_loss = _compute_v_regularization(velocity.feats, teacher_vel.feats)
            else:
                raise ValueError(f"Unknown reg_type: {reg_type}. Use 'x0', 'v', or 'none'.")
            
            reg_loss_sum = reg_loss_sum + reg_loss
        
        # ---- Scheduler 步进（使用 SparseTensor 流程） ----
        # scheduler.step_by_index 直接接收 SparseTensor，返回 SparseTensor
        x_t = scheduler.step_by_index(velocity, step_idx, x_t).prev_sample  # SparseTensor
        
    # ---- 6. 反归一化 ----
    # x_t 已经是 SparseTensor，直接使用
    shape_slat_normalized = x_t  # SparseTensor
    shape_slat = pipeline.denormalize(shape_slat_normalized, stage)  # SparseTensor
    
    # ---- 7. 挂载到 state（同时保存 normalized 和 denormalized 版本）----
    state.features.shape_slat = shape_slat  # denormalized，保留梯度用于 decode
    
    # shape_slat_norm 备用，直接 detach 切断依赖
    norm_detached = shape_slat_normalized.detach()
    norm_detached.clear_spatial_cache()
    state.features.shape_slat_norm = norm_detached
    
    # reg_loss_sum 在 tracker 路径下保留计算图（连着 cond_proxy），Phase 2 会合并 backward
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None
    
    return tracker
