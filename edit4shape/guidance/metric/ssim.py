"""SSIM loss。"""
from typing import Optional

import torch
from pytorch_msssim import ssim

from .base import BaseMetric


class SSIMMetric(BaseMetric):
    """
    SSIM 相似度 Metric。
    
    SSIM 越高越相似，loss = 1 - SSIM。
    """
    
    name = "ssim"
    
    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)
        print(f"[SSIMMetric] Initialized (weight={weight})")
    
    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # (B,C,H,W) [0,1]，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """计算 SSIM loss。"""
        ssim_val = ssim(rendered, target, data_range=1.0, size_average=True)  # scalar
        return 1 - ssim_val





