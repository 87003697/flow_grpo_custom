"""LPIPS loss。"""
from typing import Optional

import torch
import lpips as lpips_lib

from .base import BaseMetric


class LPIPSMetric(BaseMetric):
    """
    LPIPS 感知相似度 Metric。
    
    使用 VGG 网络计算特征距离，LPIPS 越低越相似。
    """
    
    name = "lpips"
    
    def __init__(self, weight: float, device: torch.device, **kwargs):
        super().__init__(weight, device)
        
        print(f"[LPIPSMetric] Loading VGG model...")
        self.fn = lpips_lib.LPIPS(net='vgg').to(device).eval()
        for p in self.fn.parameters():
            p.requires_grad = False
        print(f"[LPIPSMetric] Initialized (weight={weight})")
    
    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # (B,C,H,W) [0,1]，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """计算 LPIPS loss。"""
        # LPIPS 需要 [-1, 1] 范围
        r = rendered * 2 - 1  # (B,C,H,W), [0,1] → [-1,1]
        t = target * 2 - 1    # (B,C,H,W), [0,1] → [-1,1]
        return self.fn(r, t).mean()  # scalar
    
    def cleanup(self) -> None:
        if hasattr(self, 'fn'):
            del self.fn
            print("[LPIPSMetric] Cleaned up.")








