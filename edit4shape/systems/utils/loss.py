"""Loss 管理和梯度注入"""
import torch
from typing import Any, Dict, Optional, Tuple, Union

from edit4shape.guidance.base import SpecifyGradient


# =====================================================================
# 统一梯度注入 Loss（DMD eq.8 风格）
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
    统一的加权梯度注入 loss（使用 SpecifyGradient）。
    
    支持三种权重模式:
    - uniform: 不加权
    - t: 时间步加权 (grad *= t_norm)
    - ada: DMD eq.8 自适应归一化 (grad /= |clean - tea|.mean() + eps)
    
    Args:
        stu: 学生模型预测（需要梯度回传）
        tea: 教师模型预测（已 detach）
        clean: 真实 clean 数据（用于 ada 模式的 normalizer）
        weight_mode: "uniform" | "t" | "ada"
        t_norm: 归一化时间步（t 模式必需）
        dim: ada 模式的归一化维度（None 表示全局）
        eps: ada 模式的 epsilon
    
    Returns:
        loss: 用于反向传播的 loss（通过 SpecifyGradient 注入梯度）
    """
    with torch.no_grad():
        grad = stu.detach() - tea  # 学生 - 教师
        grad = torch.nan_to_num(grad)  # 处理 NaN
        
        if weight_mode == "uniform":
            weighted_grad = grad
        elif weight_mode == "t":
            if t_norm is None:
                raise ValueError("weight_mode='t' 需要提供 t_norm")
            weighted_grad = grad * t_norm
        elif weight_mode == "ada":
            # DMD eq.8: normalizer = |clean - tea|.mean(dim) + eps
            diff = clean - tea
            if dim is None:
                normalizer = diff.abs().mean() + eps  # scalar
            else:
                normalizer = diff.abs().mean(dim=dim, keepdim=True) + eps  # 保持维度
            weighted_grad = grad / normalizer
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}. Use 'uniform', 't', or 'ada'.")
        
        # 计算 loss_value 用于日志显示
        loss_value = 0.5 * (grad ** 2).mean()
    
    return SpecifyGradient.apply(stu, weighted_grad, loss_value)


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




