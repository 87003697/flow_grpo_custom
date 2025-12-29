"""
Guidance 模块。

提供 FlowEdit 图像编辑 Guidance，用于 3D 生成训练。

核心功能：
- 将渲染图像通过 FlowEdit 编辑后与原图比较
- 计算 SSIM/LPIPS/Latent MSE loss
- 自动求导，无需手动计算梯度

设备分配：
- Guidance 模型默认运行在 训练设备 + 1 的 GPU 上
- 例如：训练在 cuda:0，则 FlowEdit 在 cuda:1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any
import torch

if TYPE_CHECKING:
    from edit4shape.guidance.backends.local import LocalGuidance


@dataclass
class GuidanceResult:
    """
    Guidance 计算结果。
    
    Attributes:
        edited_imgs: 编辑后的图像 (B,V,C,H,W)
        loss_ssim: SSIM loss（可直接 backward，用 .item() 获取 float）
        loss_lpips: LPIPS loss（可直接 backward，用 .item() 获取 float）
        loss_latent_mse: Latent MSE loss（可直接 backward，用 .item() 获取 float）
    """
    edited_imgs: torch.Tensor                        # (B,V,C,H,W)
    loss_ssim: Optional[torch.Tensor] = None         # 标量 loss
    loss_lpips: Optional[torch.Tensor] = None        # 标量 loss
    loss_latent_mse: Optional[torch.Tensor] = None   # 标量 loss


def create_guidance(cfg: Any, train_device: torch.device) -> "LocalGuidance":
    """
    创建 Guidance 实例。
    
    Guidance 模型（FlowEdit）会自动分配到 train_device + 1 的 GPU 上，
    实现模型并行，避免显存竞争。
    
    Args:
        cfg: 配置对象，需包含 guidance 子配置
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
    return LocalGuidance(cfg.guidance, train_device)

