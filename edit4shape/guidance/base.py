"""
Guidance 模块。

提供 FlowEdit 图像编辑 Guidance，用于 3D 生成训练。

核心功能：
- 将渲染图像通过 FlowEdit 编辑后与原图比较
- 计算 SSIM/LPIPS/Latent MSE/DINO loss
- 自动求导，无需手动计算梯度

设备分配：
- Guidance 模型默认运行在 训练设备 + 1 的 GPU 上
- 例如：训练在 cuda:0，则 FlowEdit 在 cuda:1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any, List
import torch
from torch.autograd import Function

if TYPE_CHECKING:
    from edit4shape.guidance.backends.local import LocalGuidance
    from edit4shape.guidance.flowedit.state_tracker import FlowEditStateTracker


# =====================================================================
# SpecifyGradient - 梯度注入工具
# =====================================================================

class SpecifyGradient(Function):
    """
    自定义 autograd Function，用于将预计算的梯度注入到反向传播中。
    
    用于 VSD 正则化：将 Student-Teacher 差异作为梯度注入，
    使得 loss.backward() 能将梯度穿透 rollout 链回传到 LoRA 参数。
    
    Usage:
        grad = x0_student - x0_teacher  # 预计算的梯度
        loss = SpecifyGradient.apply(latents, grad)  # 返回伪 loss
        loss.backward()  # 梯度会注入到 latents
    
    Reference: threestudio
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, gt_grad: torch.Tensor) -> torch.Tensor:
        """
        前向传播：保存梯度，返回标量 1。
        
        Args:
            input_tensor: 需要注入梯度的张量
            gt_grad: 预计算的梯度（与 input_tensor 形状相同）
        
        Returns:
            标量 tensor（用于 backward 触发）
        """
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)
    
    @staticmethod
    def backward(ctx, grad_scale: torch.Tensor):
        """
        反向传播：返回预计算的梯度。
        
        Args:
            grad_scale: 来自后续层的梯度（通常为 1）
        
        Returns:
            (gt_grad * grad_scale, None): 注入的梯度
        """
        gt_grad, = ctx.saved_tensors
        return gt_grad * grad_scale, None


@dataclass
class GuidanceResult:
    """
    Guidance 计算结果。
    
    Attributes:
        edited_imgs: 编辑后的图像 (B,V,C,H,W)
        loss_ssim: SSIM loss（可直接 backward）
        loss_lpips: LPIPS loss（可直接 backward）
        loss_latent_mse: Latent MSE loss（可直接 backward）
        loss_dino: DINOv3 特征空间 loss（可直接 backward）
        trackers: FlowEdit 中间状态跟踪器列表（用于多步监督）
    """
    edited_imgs: torch.Tensor                                           # (B,V,C,H,W)
    loss_ssim: Optional[torch.Tensor] = None                            # 标量 loss
    loss_lpips: Optional[torch.Tensor] = None                           # 标量 loss
    loss_latent_mse: Optional[torch.Tensor] = None                      # 标量 loss
    loss_dino: Optional[torch.Tensor] = None                            # 标量 loss
    trackers: Optional[List["FlowEditStateTracker"]] = None             # 中间状态跟踪器


def create_guidance(cfg: Any, train_device: torch.device) -> "LocalGuidance":
    """
    创建 Guidance 实例。
    
    Guidance 模型（FlowEdit）会自动分配到 train_device + 1 的 GPU 上，
    实现模型并行，避免显存竞争。
    
    Args:
        cfg: 配置对象，需包含 guidance 子配置和 train.loss 权重配置
        train_device: 训练使用的设备（如 cuda:0）
    
    Returns:
        LocalGuidance: Guidance 实例
    
    Example:
        >>> guidance = create_guidance(cfg, accelerator.device)
        >>> result = guidance.compute_guidance(comp_rgb, condition_images)
        >>> loss = result.loss_ssim + result.loss_lpips
        >>> loss.backward()
    """
    from edit4shape.guidance.backends.local import LocalGuidance
    return LocalGuidance(cfg, train_device)  # 传入完整 cfg

