"""
SDE Rollout - Nabla-R2D3 风格的 Score Function Matching 训练

核心流程:
1. 使用 SDE 采样生成轨迹，记录每步的状态和概率
2. 重新计算学生模型的 transition score
3. 与冻结的参考 transition score 匹配
"""

from typing import Optional, Any
import torch
from tqdm import tqdm
from accelerate import Accelerator
import ml_collections

from trellis.modules.sparse import SparseTensor

from .base import predict_velocity_with_cfg
from edit4shape.generators.trellis.state.tracker import RolloutTracker, StepRecord


# =====================================================================
# 类型别名
# =====================================================================
TrellisState = Any  # 避免循环导入
System = Any


# =====================================================================
# SDE Rollout 核心函数
# =====================================================================

def rollout_sparse_sde(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = True,
    track_trajectory: bool = False,
) -> RolloutTracker:
    """
    SDE 采样 + 轨迹追踪（用于 Nabla-R2D3 训练）
    
    核心流程: x_T (噪声) → SDE 迭代去噪 → x_0 (特征)
    同时记录每步的状态、概率信息供后续 Score Matching 使用
    
    Args:
        state: TrellisState 状态对象，包含条件编码、坐标等
        cfg: 配置对象，需包含:
            - cfg.rollout.noise_level: SDE 噪声水平 (默认 0.7)
            - cfg.rollout.sde_type: SDE 类型 ('sde' 或 'cps')
            - cfg.seed: 随机种子
        system: 系统组件（pipeline、renderer 等）
        device: 运行设备
        generator: 随机数生成器（用于可复现性）
        is_training: 是否为训练模式
        track_trajectory: 是否将 tracker 挂载到 state（用于 Score Matching 训练）
    
    Returns:
        tracker: RolloutTracker，包含完整的采样轨迹
    
    Side Effects:
        - state.features.slat: 挂载反归一化后的 SparseTensor
        - state.tracker.rollout: 仅当 track_trajectory=True 时挂载
    """
    pipeline = system.pipeline
    _, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 1. 初始化 ----
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None
    
    assert state.coords is not None, "state.coords 缺失"
    generator = generator or torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    # 初始化 x_T
    x_t = pipeline.init_latents(
        coords=state.coords,
        in_channels=pipeline._resolve_slat_flow_module().in_channels,
        generator=generator
    )  # SparseTensor
    
    # ---- 2. Scheduler 配置 ----
    scheduler = pipeline.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    cfg_min, cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]
    
    # ---- 3. SDE 配置 ----
    sde_noise_level = cfg.rollout.noise_level
    sde_type = cfg.rollout.sde_type
    
    # ---- 4. 初始化 Tracker ----
    tracker = RolloutTracker(device=device)
    tracker.set_initial_latent(x_t)
    
    # ---- 5. SDE 去噪循环 ----
    steps = list(scheduler.timesteps)[:-1]  # 去掉最后一步（t=0）
    steps_iter = tqdm(steps, desc="SDE Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for t in steps_iter:
        t_val = float(t.item())
        
        # ---- 速度场预测（checkpointing 由模型内部 block 处理）----
        if is_training:
            velocity = predict_velocity_with_cfg(
                pipeline, x_t, t_val, cond_emb, uncond_emb,
                slat_guidance, cfg_min, cfg_max, device,
            )  # SparseTensor
        else:
            with torch.no_grad():
                velocity = predict_velocity_with_cfg(
                    pipeline, x_t, t_val, cond_emb, uncond_emb,
                    slat_guidance, cfg_min, cfg_max, device,
                )  # SparseTensor
        
        # ---- SDE 步进 ----
        x_prev, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sde_step(
            noise_pred=velocity,
            t=t_val,
            latents=x_t,
            noise_level=sde_noise_level,
            prev_sample=None,
            generator=generator,
            sde_type=sde_type,
            return_sqrt_dt=True,
        )  # x_prev: SparseTensor, log_prob: (B,)
        
        # ---- 记录步骤 ----
        tracker.record_step(
            t=t_val,
            x_t=x_t,
            x_prev=x_prev,
            velocity=velocity,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
            log_prob=log_prob,
            sqrt_dt=sqrt_dt,
        )
        
        # 更新状态
        x_t = x_prev
    
    # ---- 6. 反归一化 ----
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats = x_t.feats * std + mean  # (N, C)
    
    # ---- 7. 挂载到 state ----
    state.features.slat = x_t.replace(denorm_feats)
    if track_trajectory:
        state.attach_rollout_tracker(tracker)  # 仅当 track_trajectory=True 时挂载
    
    return tracker


# =====================================================================
# Score Matching Loss 计算
# =====================================================================

def compute_score_matching_loss(
    state: TrellisState,
    system: System,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    reward_gradients: SparseTensor,
) -> torch.Tensor:
    """
    计算 Nabla-R2D3 风格的 Score Matching Loss
    
    Loss = E_t [ ||∇_θ log p_θ(x_{t-1}|x_t) - (∇_ref log p(x_{t-1}|x_t) + λ·∇R)||² ]
    
    其中:
    - ∇_θ log p_θ: 学生模型的 transition score（需要计算梯度）
    - ∇_ref log p: 参考模型的 transition score（冻结）
    - ∇R: reward 梯度（必须提供）
    
    Args:
        state: 状态对象（含 tracker.rollout）
        system: 系统组件
        cfg: 配置，需包含:
            - cfg.nabla.num_steps: 选择计算的步数
            - cfg.nabla.selection_mode: 时间步选择模式
            - cfg.nabla.reward_weight: reward 梯度权重 λ
        device: 运行设备
        reward_gradients: 外部计算的 reward 梯度（必须提供）
        
    Returns:
        loss: 标量 Tensor
    """
    tracker = state.tracker.rollout
    if tracker is None:
        raise ValueError("state.tracker.rollout 为 None，请先调用 rollout_sparse_sde")
    pipeline = system.pipeline
    _, _, _, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()
    cfg_min, cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]
    
    # 获取条件编码
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None
    
    # 配置
    num_steps = cfg.nabla.num_steps
    selection_mode = cfg.nabla.selection_mode
    reward_weight = cfg.nabla.reward_weight
    
    # 选择时间步
    selected_records = tracker.select_timesteps(
        mode=selection_mode,
        num_steps=num_steps,
    )
    
    if not selected_records:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = torch.tensor(0.0, device=device)
    
    for record in selected_records:
        # ---- 用当前学生模型重新计算速度场（需要梯度，checkpointing 由模型内部 block 处理）----
        new_velocity = predict_velocity_with_cfg(
            pipeline, record.x_t, record.t,
            cond_emb, uncond_emb, slat_guidance,
            cfg_min, cfg_max, device,
        )  # SparseTensor，有梯度
        
        # ---- 计算 Student Transition Score ----
        stu_score = tracker.compute_transition_score_student(record, new_velocity)  # (N, C)
        
        # ---- 获取 Reference Transition Score ----
        ref_score = tracker.compute_transition_score_reference(record)  # (N, C)
        
        # ---- 目标 Score = 参考 Score + λ·Reward 梯度 ----
        target_score = ref_score + reward_weight * reward_gradients.feats  # (N, C)
        
        # ---- 计算 MSE Loss ----
        step_loss = ((stu_score - target_score.detach()) ** 2).mean()
        total_loss = total_loss + step_loss
    
    # 平均
    loss = total_loss / len(selected_records)
    
    return loss
