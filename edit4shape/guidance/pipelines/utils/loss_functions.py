"""
Loss 计算函数与时间步采样工具。

设计原则：
1. 单步 Loss：计算单个 (src, target) 对的 loss
2. 多步聚合：将多个 loss 聚合为一个
3. LossMixin：为 Tracker 提供统一的 loss 计算能力
4. 时间步采样：支持多时间步采样（MTS）

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


def normalize_grad(
    grad: torch.Tensor,
    src: torch.Tensor,
    pos: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Ada 归一化：grad / (|src - pos|.mean + eps)
    
    使用 src 到 pos（吸引目标）的 L1 距离均值作为归一化因子。
    
    Args:
        grad: [B, seq, C] 梯度方向（通常是 neg - pos 或 src - target）
        src: [B, seq, C] 源 latent（要优化的）
        pos: [B, seq, C] 正样本（吸引目标）
        eps: 数值稳定 epsilon
    
    Returns:
        [B, seq, C] 归一化后的梯度
    """
    normalizer = torch.abs(src - pos).detach().mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
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
        target: [B, seq, C] 目标 latent（也是 pos）
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if ada:
        grad = (src - target).detach().float()  # [B, seq, C]
        # MSE 场景：target 就是 pos
        grad_norm = normalize_grad(grad, src.float(), target.detach().float(), eps)  # [B, seq, C]
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
        x0_high: [B, seq, C] 高 CFG 预测（pos，吸引）
        x0_low: [B, seq, C] 低 CFG 预测（neg，排斥）
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if ada:
        grad = (x0_low - x0_high).detach().float()  # [B, seq, C] 方向：neg - pos
        # CSD 场景：x0_high 是 pos
        grad_norm = normalize_grad(grad, src.float(), x0_high.detach().float(), eps)  # [B, seq, C]
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


def delta_loss_step(
    src: torch.Tensor,
    z_next: torch.Tensor,
    z_curr: torch.Tensor,
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    单步 Delta Loss：MSE(src, z_next) - MSE(src, z_curr)
    
    让 src 向后一步的 z_edit 靠拢，远离当前步的 z_edit。
    用于沿编辑轨迹的对比蒸馏。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        z_next: [B, seq, C] 后一步的 z_edit（pos，吸引）
        z_curr: [B, seq, C] 当前步的 z_edit（neg，排斥）
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if ada:
        grad = (z_curr - z_next).detach().float()  # [B, seq, C] 方向：neg - pos
        # Delta 场景：z_next 是 pos
        grad_norm = normalize_grad(grad, src.float(), z_next.detach().float(), eps)  # [B, seq, C]
        return (src.float() * grad_norm).mean()  # scalar
    else:
        loss_pos = F.mse_loss(src.float(), z_next.detach().float())  # scalar
        loss_neg = F.mse_loss(src.float(), z_curr.detach().float())  # scalar
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


def delta_loss(
    src: torch.Tensor,
    z_edits: List[torch.Tensor],
    reduce: ReduceMode = "mean",
    ada: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Delta Loss：用相邻步的 z_edit 构造对比。
    
    Loss = sum_k [ MSE(src, z_edit[k+1]) - MSE(src, z_edit[k]) ]
    
    让 src 沿着编辑轨迹的方向移动：向后续步骤靠拢，远离之前的步骤。
    
    注意：z_edits[0] 是第一步编辑后的结果（不是 src），因为 tracker.record()
    在 z_edit 更新之后才调用。
    
    Args:
        src: [B, seq, C] 源 latent（有梯度）
        z_edits: List[[B, seq, C]] 每一步的 z_edit（即 tracker.x0_preds）
        reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
        ada: 是否使用自适应归一化
        eps: ada 模式的 epsilon
    
    Returns:
        scalar loss
    """
    if len(z_edits) < 2:
        return torch.tensor(0.0, device=src.device, requires_grad=True)  # scalar
    
    losses = []
    for k in range(len(z_edits) - 1):
        z_next = z_edits[k + 1]  # 后一步（吸引）
        z_curr = z_edits[k]      # 当前步（排斥）
        loss_k = delta_loss_step(src, z_next, z_curr, ada=ada, eps=eps)  # scalar
        losses.append(loss_k)
    
    return reduce_losses(losses, reduce)  # scalar


# =============================================================================
# LossMixin - Tracker 用的 loss 计算 Mixin
# =============================================================================

class LossMixin:
    """
    Loss 计算 Mixin。
    
    为 Tracker 提供统一的 loss 计算能力：
    - MSE Loss: MSE(src, x0_preds)
    - CSD Loss: MSE(src, x0_high) - MSE(src, x0_low)
    - Delta Loss: MSE(src, z_edit[k+1]) - MSE(src, z_edit[k])
    - 混合 Loss: mse_weight * MSE + csd_weight * CSD + delta_weight * Delta
    
    子类需要提供：
    - x0_preds: List[Tensor] 预测 x0 列表（MSE 目标，FlowEdit 中为 z_edit）
    - x0_highs: List[Tensor] 高 CFG 预测列表（CSD 吸引）
    - x0_lows: List[Tensor] 低 CFG 预测列表（CSD 排斥）
    """
    
    # 子类需要定义这些字段
    x0_preds: List[torch.Tensor]  # MSE 目标（FlowEdit 中为 z_edit）
    x0_highs: List[torch.Tensor]  # CSD 吸引
    x0_lows: List[torch.Tensor]   # CSD 排斥
    
    def compute_mse_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算 MSE Loss: MSE(src, x0_preds)
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        return mse_loss(src, self.x0_preds, reduce=reduce, ada=ada, eps=eps)
    
    def compute_csd_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算 CSD Loss：MSE(src, x0_high) - MSE(src, x0_low)
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        return csd_loss(src, self.x0_highs, self.x0_lows, reduce=reduce, ada=ada, eps=eps)
    
    def compute_delta_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算 Delta Loss：MSE(src, z_edit[k+1]) - MSE(src, z_edit[k])
        
        使用 x0_preds 作为 z_edits（在 FlowEdit 中它们是相同的）。
        让 src 沿着编辑轨迹的方向移动。
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        return delta_loss(src, self.x0_preds, reduce=reduce, ada=ada, eps=eps)
    
    def loss(
        self,
        src: torch.Tensor,
        mse_weight: float = 0.0,
        csd_weight: float = 1.0,
        delta_weight: float = 0.0,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算混合 Loss。
        
        Loss = mse_weight * MSE(src, x0_preds) 
             + csd_weight * CSD(src, x0_highs, x0_lows)
             + delta_weight * Delta(src, x0_preds)
        
        Args:
            src: [B, seq, C] 有梯度的源 latent
            mse_weight: MSE loss 权重
            csd_weight: CSD loss 权重
            delta_weight: Delta loss 权重（沿编辑轨迹的对比）
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)
        
        if mse_weight > 0 and self.x0_preds:
            loss_mse = self.compute_mse_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + mse_weight * loss_mse
        
        if csd_weight > 0 and self.x0_highs and self.x0_lows:
            loss_csd = self.compute_csd_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + csd_weight * loss_csd
        
        if delta_weight > 0 and len(self.x0_preds) >= 2:
            loss_delta = self.compute_delta_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + delta_weight * loss_delta
        
        return total_loss