"""像素空间 MSE loss。"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseMetric


class MSEMetric(BaseMetric):
    """
    像素空间 MSE Metric。

    直接计算渲染图与编辑图在像素空间的均方误差。
    梯度路径：loss → MSE → comp_rgb → 3D model
    """

    name = "mse"

    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)
        logging.info(f"[MSEMetric] Initialized (weight={weight})")

    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # (B,C,H,W) [0,1]，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """计算像素空间 MSE loss。"""
        return F.mse_loss(rendered, target.detach())  # scalar
