"""Loss 管理和梯度计算"""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union


# =====================================================================
# Gradient Shrink（forward 不变，backward 梯度乘以 scale）
# =====================================================================

class _GradientShrink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def gradient_shrink(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward 保持不变，backward 时梯度乘以 scale（0~1 抑制梯度）。"""
    if scale >= 1.0:
        return x
    return _GradientShrink.apply(x, scale)


# =====================================================================
# 统一加权 Loss（真 Loss 模式，可直接 backward）
# =====================================================================

def apply_gradient_loss(
    stu: torch.Tensor,
    tea: torch.Tensor,
    clean: torch.Tensor,
    weight_mode: str = "uniform",
    t_norm: Optional[float] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    eps: float = 1e-2,
) -> torch.Tensor:
    """
    统一的加权 loss（真 Loss 模式，可直接 backward）。
    
    支持三种权重模式:
    - uniform: MSE(stu, tea)
    - t: 时间步加权 MSE
    - ada: 自适应归一化（DMD eq.8 风格）
    
    Args:
        stu: 学生模型预测（需要梯度回传）
        tea: 教师模型预测（无梯度）
        clean: 真实 clean 数据（用于 ada 模式的 normalizer）
        weight_mode: "uniform" | "t" | "ada"
        t_norm: 归一化时间步（t 模式必需）
        dim: ada 模式的归一化维度（None 表示全局）
        eps: ada 模式的 epsilon
    
    Returns:
        loss: 真 loss，可直接 backward
    """
    tea = tea.detach()  # 确保教师无梯度
    
    if weight_mode == "uniform":
        # 标准 MSE
        return F.mse_loss(stu.float(), tea.float())
    
    elif weight_mode == "t":
        # 时间步加权 MSE
        if t_norm is None:
            raise ValueError("weight_mode='t' 需要提供 t_norm")
        diff = (stu - tea).float()  # [...]
        weighted_diff = diff * t_norm
        return (weighted_diff ** 2).mean()
    
    elif weight_mode == "ada":
        # 自适应归一化：先计算目标梯度，再归一化
        # 目标梯度 = stu - tea
        grad_raw = (stu - tea).detach().float()  # 无梯度，用于计算 normalizer
        
        # normalizer = |clean - tea|.mean(dim) + eps（或 |stu - tea|.mean()）
        with torch.no_grad():
            if dim is None:
                normalizer = grad_raw.abs().mean() + eps  # scalar
            else:
                normalizer = grad_raw.abs().mean(dim=dim, keepdim=True) + eps
            grad_normalized = grad_raw / normalizer
        
        # 构造 loss，使 ∂loss/∂stu = grad_normalized
        return (stu.float() * grad_normalized).mean()
    
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}. Use 'uniform', 't', or 'ada'.")


# =====================================================================
# LossDict - 统一 Loss 管理
# =====================================================================

class LossDict:
    """
    统一 loss 管理：累加、加权、日志生成。
    
    用法：
        losses = LossDict(device="cuda:0")  # 指定目标设备
        losses.add("ssim", loss_ssim, weight=cfg.ssim_weight)
        losses.add("lpips", loss_lpips, weight=cfg.lpips_weight)
        
        total = losses.total()           # 自动求和（所有 loss 已移到同一设备）
        logs = losses.to_logs()          # {"loss/ssim": ..., "loss/lpips": ..., "loss/total": ...}
    """
    
    def __init__(self, device: torch.device = None):
        self._items: Dict[str, torch.Tensor] = {}  # 加权后的 loss
        self._raw: Dict[str, torch.Tensor] = {}    # 原始 loss（用于日志）
        self._device = device  # 目标设备，用于统一 tensor 位置
    
    def add(
        self,
        name: str,
        loss: Optional[torch.Tensor],
        weight: float = 1.0,
    ) -> "LossDict":
        """
        添加 loss 项。
        
        Args:
            name: loss 名称（如 "ssim", "lpips"）
            loss: loss tensor 或 None
            weight: 权重（默认 1.0，表示权重已在外部应用）
        
        Returns:
            self（支持链式调用）
        """
        if loss is None or weight <= 0:
            return self
        
        # 移动到目标设备（如果指定）
        if self._device is not None and loss.device != self._device:
            loss = loss.to(self._device)
        
        weighted = loss * weight if weight != 1.0 else loss
        self._items[name] = weighted
        self._raw[name] = loss
        return self
    
    def total(self) -> torch.Tensor:
        """计算加权 loss 总和"""
        if not self._items:
            device = self._device if self._device else "cpu"
            return torch.tensor(0.0, device=device)
        
        return sum(self._items.values())
    
    def to_logs(self, prefix: str = "loss/") -> Dict[str, torch.Tensor]:
        """
        生成日志字典。
        
        Args:
            prefix: key 前缀（默认 "loss/"）
        
        Returns:
            dict: {"loss/ssim": tensor, "loss/lpips": tensor, "loss/total": tensor}
        """
        logs = {}
        for name, val in self._raw.items():
            logs[f"{prefix}{name}"] = val.detach()
        
        if self._items:
            logs[f"{prefix}total"] = self.total().detach()
        
        return logs
    
    def __bool__(self) -> bool:
        """是否有任何 loss"""
        return bool(self._items)
