"""Metric 基类定义。"""
from abc import ABC, abstractmethod
from typing import Optional, Any

import torch


class BaseMetric(ABC):
    """
    相似度 Metric 基类。
    
    所有 metric 必须实现 compute 方法，返回标量 loss。
    """
    
    name: str = "base"  # 子类必须覆盖
    
    def __init__(self, weight: float, device: torch.device, **kwargs):
        """
        初始化 Metric。
        
        Args:
            weight: loss 权重（仅用于记录，实际加权在外部进行）
            device: 计算设备
            **kwargs: 子类特定参数
        """
        self.weight = weight
        self.device = device
    
    @abstractmethod
    def compute(
        self,
        rendered: torch.Tensor,
        target: Any,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        计算 loss（不乘权重）。
        
        Args:
            rendered: 渲染图 (B,C,H,W) [0,1]，有梯度
            target: 目标（可以是图像张量或其他格式）
            **kwargs: 额外参数
        
        Returns:
            标量 loss，或 None（如果不计算）
        """
        pass
    
    def cleanup(self) -> None:
        """释放资源（可选实现）。"""
        pass





