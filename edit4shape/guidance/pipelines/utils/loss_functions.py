"""
Loss 计算函数与时间步采样工具。

设计原则：
1. 单步 Loss：计算单个 (src, target) 对的 loss
2. 多步聚合：将多个 loss 聚合为一个
3. 时间步采样：支持多时间步采样（MTS）

两个正交维度：
- 归一化方式：ada=False（原始 MSE）/ ada=True（归一化梯度）
- 聚合方式：final / mean / weighted / inv_weighted
"""

from typing import List, Literal, Optional
import torch
import torch.nn.functional as F


# =============================================================================
# 类型定义
# =============================================================================

ReduceMode = Literal["final", "mean", "weighted", "inv_weighted"]


# =============================================================================
# 工具函数
# =============================================================================

def reduce_losses(losses: List[torch.Tensor], mode: ReduceMode = "final") -> torch.Tensor:
    """
    聚合多步 loss。
    
    Args:
        losses: List of scalar losses
        mode:
            - "final": 只取最后一个
            - "mean": 均匀平均
            - "weighted": 1/k 加权（前期大）
            - "inv_weighted": k/K 加权（后期大）
    
    Returns:
        scalar loss
    """
    if len(losses) == 0:
        raise ValueError("losses list is empty")
    
    if mode == "final":
        return losses[-1]  # scalar
    
    losses_t = torch.stack(losses)  # [K]
    K = len(losses_t)  # 步数
    
    if mode == "mean":
        return losses_t.mean()  # scalar
    
    elif mode == "weighted":
        # 1/k 加权：[1, 1/2, 1/3, ..., 1/K]
        w = 1.0 / torch.arange(1, K + 1, device=losses_t.device, dtype=losses_t.dtype)  # [K]
    
    elif mode == "inv_weighted":
        # k/K 加权：[1/K, 2/K, ..., 1]
        w = torch.arange(1, K + 1, device=losses_t.device, dtype=losses_t.dtype) / K  # [K]
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: final, mean, weighted, inv_weighted")
    
    w = w / w.sum()  # [K] 归一化
    return (losses_t * w).sum()  # scalar


def normalize_grad(grad: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    梯度归一化：grad / (|grad|.mean + eps)
    
    Args:
        grad: [B, seq, C] 原始梯度
        eps: 数值稳定 epsilon
    
    Returns:
        [B, seq, C] 归一化后的梯度
    """
    normalizer = torch.abs(grad).mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
    return grad / normalizer  # [B, seq, C]


# =============================================================================
# 单步 Loss（基础构建块）
# =============================================================================

def mse_loss_step(
    src: torch.Tensor,
    target: torch.Tensor,
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    单步 MSE Loss。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        target: [B, seq, C] 目标 latent
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if ada:
        grad = (src - target).detach().float()  # [B, seq, C]
        grad_norm = normalize_grad(grad, eps)  # [B, seq, C]
        return (src.float() * grad_norm).mean()  # scalar
    else:
        return F.mse_loss(
            src.float(),  # [B, seq, C]
            target.detach().float()  # [B, seq, C]
        )  # scalar


def csd_loss_step(
    src: torch.Tensor,
    x0_high: torch.Tensor,
    x0_low: torch.Tensor,
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    单步 CSD Loss：MSE(src, high) - MSE(src, low)
    
    让 src 向 x0_high 靠拢，远离 x0_low。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        x0_high: [B, seq, C] 高 CFG 预测（吸引）
        x0_low: [B, seq, C] 低 CFG 预测（排斥）
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if ada:
        grad = (x0_low - x0_high).detach().float()  # [B, seq, C]
        grad_norm = normalize_grad(grad, eps)  # [B, seq, C]
        return (src.float() * grad_norm).mean()  # scalar
    else:
        loss_pos = F.mse_loss(
            src.float(),  # [B, seq, C]
            x0_high.detach().float()  # [B, seq, C]
        )  # scalar
        loss_neg = F.mse_loss(
            src.float(),  # [B, seq, C]
            x0_low.detach().float()  # [B, seq, C]
        )  # scalar
        return loss_pos - loss_neg  # scalar


# =============================================================================
# 多步 Loss（组合接口）
# =============================================================================

def mse_loss(
    src: torch.Tensor,
    targets: List[torch.Tensor],
    reduce: ReduceMode = "final",
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    多步 MSE Loss。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        targets: List[[B, seq, C]] 目标 latent 列表
        reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if len(targets) == 0:
        return torch.tensor(0.0, device=src.device, requires_grad=True)  # scalar
    
    losses = [
        mse_loss_step(src, t, ada=ada, eps=eps)  # scalar
        for t in targets
    ]
    return reduce_losses(losses, reduce)  # scalar


def csd_loss(
    src: torch.Tensor,
    x0_highs: List[torch.Tensor],
    x0_lows: List[torch.Tensor],
    reduce: ReduceMode = "final",
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    多步 CSD Loss。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        x0_highs: List[[B, seq, C]] 高 CFG 预测列表
        x0_lows: List[[B, seq, C]] 低 CFG 预测列表
        reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if len(x0_highs) == 0:
        return torch.tensor(0.0, device=src.device, requires_grad=True)  # scalar
    
    losses = [
        csd_loss_step(src, h, l, ada=ada, eps=eps)  # scalar
        for h, l in zip(x0_highs, x0_lows)
    ]
    return reduce_losses(losses, reduce)  # scalar


# =============================================================================
# 时间步采样（MTS）
# =============================================================================

def sample_timesteps_uniform(
    min_step: int,
    max_step: int,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    ascending: bool = True,
) -> List[torch.Tensor]:
    """
    均匀分区采样多个时间步（Multi-timestep Sampling）。
    
    将 [min_step, max_step] 均分为 num_steps 个区间，每个区间随机采样一个 t。
    
    例如 num_steps=4, min=20, max=500:
        区间 0: [20, 140)   → 采样 t_0
        区间 1: [140, 260)  → 采样 t_1
        区间 2: [260, 380)  → 采样 t_2
        区间 3: [380, 500]  → 采样 t_3
    
    Args:
        min_step: 最小时间步（如 20）
        max_step: 最大时间步（如 500）
        num_steps: 采样数量（m）
        batch_size: 批次大小
        device: 设备
        generator: 随机数生成器
        ascending: 是否从小到大排列（默认 True）
            - True: [t_small, ..., t_large]（用于 noise inversion）
            - False: [t_large, ..., t_small]（用于正常去噪）
    
    Returns:
        List[Tensor(B,)]: 时间步列表
    """
    timesteps = []
    step_range = max_step - min_step
    for i in range(num_steps):
        t_lo = min_step + step_range * i // num_steps  # 区间下界
        t_hi = min_step + step_range * (i + 1) // num_steps  # 区间上界
        t = torch.randint(t_lo, t_hi + 1, (batch_size,), device=device, generator=generator)  # (B,)
        timesteps.append(t)
    
    if not ascending:
        timesteps = timesteps[::-1]  # 反转为从大到小
    
    return timesteps
