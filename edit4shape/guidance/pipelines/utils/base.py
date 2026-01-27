"""
基础类定义。

包含:
- BaseStateTracker: 状态追踪器抽象基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import torch
from PIL import Image


@dataclass
class BaseStateTracker(ABC):
    """
    状态追踪器抽象基类。
    
    所有 Guidance 方法的 Tracker（FlowEdit、CSD、SDS）都继承此类。
    
    共有属性：
        - height, width: 图像尺寸（用于 decode）
    
    子类必须实现：
        - target: 目标 latent（用于 loss 计算）
        - loss(): 主要 loss 计算方法
    """
    
    height: int = 0
    width: int = 0
    
    # =========================================================================
    # 抽象属性和方法
    # =========================================================================
    
    @property
    @abstractmethod
    def target(self) -> torch.Tensor:
        """
        目标 latent [B, seq, C]。
        
        用于 loss 计算，子类根据算法返回不同的目标：
        - FlowEdit: 最终编辑后的 latent
        - CSD: x0_high（高 CFG 预测）
        - SDS: x0（模型预测）
        """
        pass
    
    @abstractmethod
    def loss(
        self, 
        src: torch.Tensor, 
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        计算 loss（真 loss，可直接 backward）。
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: 数值稳定 epsilon
        
        Returns:
            标量 loss
        """
        pass
    
    # =========================================================================
    # 共有方法
    # =========================================================================
    
    def _decode_latent(self, pipe: Any, latent: torch.Tensor) -> Optional[Image.Image]:
        """
        Decode 单个 latent 为图像。
        
        Args:
            pipe: Pipeline（需要 _decode_latent_to_image 方法）
            latent: [1, seq, C]
        
        Returns:
            PIL Image
        """
        with torch.no_grad():
            images = pipe._decode_latent_to_image(
                latent,
                self.height,
                self.width,
                "pil"
            )
            return images[0] if images else None
    
    def _concat_images_horizontal(self, *images: Image.Image) -> Optional[Image.Image]:
        """
        水平拼接多张图像。
        
        Args:
            *images: PIL Image 列表
        
        Returns:
            拼接后的 PIL Image
        """
        images = [img for img in images if img is not None]
        if not images:
            return None
        
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return combined
