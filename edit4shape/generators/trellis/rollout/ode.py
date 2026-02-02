"""
ODE Rollout - 标准 Euler 采样

用于推理和 ReFL/DRaFT 风格的训练（可选 DMD/KL 正则化）。
"""

from typing import Optional, Any
import torch
from tqdm import tqdm
from accelerate import Accelerator
import ml_collections

from trellis.modules.sparse import SparseTensor

from .base import predict_velocity_with_cfg


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
) -> None:
    """
    稀疏特征去噪采样（SLAT Stage 2）。
    
    核心流程: x_T (噪声) → 迭代去噪 → x_0 (特征) → 反归一化
    
    Args:
        state: TrellisState 状态对象，包含条件编码、坐标等
        cfg: 配置对象，cfg.reg.type ("none"|"vsd"|"kl"), cfg.reg.weight_mode
        system: 系统组件（pipeline、renderer 等）
        device: 运行设备
        generator: 随机数生成器（用于可复现性）
        is_training: 是否为训练模式
    
    Side Effects:
        - state.features.slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 挂载 reg_loss
    """
    pipeline = system.pipeline
    _, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 1. 初始化 ----
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B,S,C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B,S,C)
    
    assert state.coords is not None, "state.coords 缺失"
    generator = generator or torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    # ★ 使用 SparseTensor 贯穿整个流程（对齐 trellis2 实现）
    # 这样在跨设备转移时，SparseTensor 会正确处理内部状态
    x_t = pipeline.init_latents(
        coords=state.coords,
        in_channels=pipeline.pipe.models['slat_flow_model'].in_channels,
        generator=generator
    )  # SparseTensor
    
    # ---- 2. Scheduler 配置 ----
    scheduler = pipeline.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    cfg_min, cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]
    
    # ---- 3. 正则化配置 ----
    # reg_type: "none" | "dmd" | "kl"
    #   - "dmd": DMD 风格，grad 在 no_grad 中计算，通过伪 loss 注入（符合 Self-Forcing 原理）
    #   - "kl": KL 风格，直接可导的 MSE loss（原始实现）
    # weight_mode: "uniform" | "t" | "ada"
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_eps = cfg.reg.eps
    # 正则化需要: 开启 reg、训练模式、strategy 有教师
    reg_enabled = reg_type != "none" and is_training
    
    reg_loss_sum = 0.0
    
    # ---- 4. 去噪循环 ----
    steps = list(scheduler.timesteps)[:-1]
    steps_iter = tqdm(steps, desc="Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for t in steps_iter:
        t_val = float(t.item())
        t_norm = t_val / 1000.0
        
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
        
        # ---- 正则化（DMD 风格）----
        if reg_enabled:
            # 使用 strategy.teacher_context() 统一处理教师模型获取
            # - LoRA 模式: 禁用 adapters
            # - Full 模式: 使用冻结教师副本（auto_device 装饰器自动处理设备转移）
            with system.strategy.teacher_context(), torch.no_grad():
                teacher_vel = predict_velocity_with_cfg(
                    pipeline, x_t, t_val, cond_emb, uncond_emb,
                    slat_guidance, cfg_min, cfg_max, device,
                )  # SparseTensor
            
            x0_stu = x_t.feats - t_norm * velocity.feats       # (N,C)
            x0_tea = x_t.feats - t_norm * teacher_vel.feats    # (N,C)
            
            # 根据 reg_type 选择正则化函数（DMD 和 KL 都使用真 Loss 模式）
            if reg_type == "dmd":
                reg_loss = _compute_dmd_regularization(
                    x0_student=x0_stu,
                    x0_teacher=x0_tea,
                    t_norm=t_norm,
                    weight_mode=weight_mode,
                    eps=reg_eps,
                )
            elif reg_type == "kl":
                reg_loss = _compute_kl_regularization(
                    x0_student=x0_stu,
                    x0_teacher=x0_tea,
                    t_norm=t_norm,
                    weight_mode=weight_mode,
                    eps=reg_eps,
                )
            else:
                raise ValueError(f"Unknown reg_type: {reg_type}. Use 'dmd', 'kl', or 'none'.")
            
            reg_loss_sum = reg_loss_sum + reg_loss
        
        # ---- Scheduler 步进（使用 SparseTensor）----
        x_t = scheduler.step(velocity, t, x_t).prev_sample  # SparseTensor
    
    # ---- 5. 反归一化 ----
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1,C)
    mean = torch.tensor(norm['mean'])[None].to(device) # (1,C)
    denorm_feats = x_t.feats * std + mean  # (N,C)
    
    # ---- 6. 挂载到 state ----
    state.features.slat = x_t.replace(denorm_feats)  # SparseTensor with denormalized feats
    
    num_steps = max(1, len(steps))
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None


# =====================================================================
# 正则化函数（DMD/KL）
# =====================================================================

def _compute_dmd_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    t_norm: float,
    weight_mode: str = "uniform",
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    DMD 风格正则化 Loss
    
    Args:
        x0_student: 学生模型预测的 x0 (N, C)
        x0_teacher: 教师模型预测的 x0 (N, C)
        t_norm: 归一化时间步
        weight_mode: 权重模式 ("uniform", "t", "ada")
        eps: 防止除零
        
    Returns:
        loss: 标量
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    
    if weight_mode == "uniform":
        weight = 1.0
    elif weight_mode == "t":
        weight = t_norm
    elif weight_mode == "ada":
        weight = 1.0 / (t_norm + eps)
    else:
        weight = 1.0
    
    loss = weight * (diff ** 2).mean()
    return loss


def _compute_kl_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    t_norm: float,
    weight_mode: str = "uniform",
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    KL 风格正则化 Loss（实际使用 MSE 近似）
    
    Args:
        x0_student: 学生模型预测的 x0 (N, C)
        x0_teacher: 教师模型预测的 x0 (N, C)
        t_norm: 归一化时间步
        weight_mode: 权重模式
        eps: 防止除零
        
    Returns:
        loss: 标量
    """
    diff = x0_student - x0_teacher.detach()  # (N, C)
    
    if weight_mode == "uniform":
        weight = 1.0
    elif weight_mode == "t":
        weight = t_norm
    elif weight_mode == "ada":
        weight = 1.0 / (t_norm + eps)
    else:
        weight = 1.0
    
    loss = weight * (diff ** 2).mean()
    return loss

