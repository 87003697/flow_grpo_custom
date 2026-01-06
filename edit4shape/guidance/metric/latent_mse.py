"""Latent MSE loss。"""
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from .base import BaseMetric


class LatentMSEMetric(BaseMetric):
    """
    VAE Latent 空间 MSE Metric。
    
    需要外部提供 encode 函数将图像编码到 latent 空间。
    """
    
    name = "latent_mse"
    
    def __init__(
        self,
        weight: float,
        device: torch.device,
        encode_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        **kwargs,
    ):
        """
        初始化 Latent MSE Metric。
        
        Args:
            weight: loss 权重
            device: 设备
            encode_fn: 编码函数，输入 (B,C,H,W) 图像，输出 latent
        """
        super().__init__(weight, device)
        self.encode_fn = encode_fn
        print(f"[LatentMSEMetric] Initialized (weight={weight})")
    
    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # latent 格式，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        计算 Latent MSE loss。
        
        Args:
            rendered: 渲染图（有梯度）
            target: 目标 latent（无梯度，由 FlowEdit 返回）
        """
        if self.encode_fn is None:
            raise ValueError("LatentMSEMetric requires encode_fn to be set.")
        
        # 编码渲染图到 latent
        rendered_latent = self.encode_fn(rendered)  # latent 格式
        
        # 统一为 float32 计算 MSE loss
        loss = F.mse_loss(
            rendered_latent.float(),
            target.detach().float()
        )  # scalar
        return loss






