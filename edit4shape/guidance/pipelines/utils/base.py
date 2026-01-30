"""
基础类定义。

包含:
- BaseStateTracker: 状态追踪器抽象基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional
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


# =============================================================================
# 时间步采样（MTS）
# =============================================================================

def sample_timesteps_uniform(
    min_step: int,
    max_step: int,
    num_steps: int,
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    ascending: bool = True,
) -> List[torch.Tensor]:
    """
    均匀分区采样多个时间步（Multi-timestep Sampling）。
    
    将 [min_step, max_step] 均分为 num_steps 个区间，每个区间随机采样一个 t。
    
    例如 num_steps=4, min=20, max=500:
        区间 0: [20, 140)   → 采样 t_0
        区间 1: [140, 260)  → 采样 t_1
        区间 2: [260, 380)  → 采样 t_2
        区间 3: [380, 500]  → 采样 t_3
    
    Args:
        min_step: 最小时间步（如 20）
        max_step: 最大时间步（如 500）
        num_steps: 采样数量（m）
        batch_size: 批次大小
        device: 设备
        generator: 随机数生成器
        ascending: 是否从小到大排列（默认 True）
            - True: [t_small, ..., t_large]（用于 noise inversion）
            - False: [t_large, ..., t_small]（用于正常去噪）
    
    Returns:
        List[Tensor(B,)]: 时间步列表
    """
    timesteps = []
    step_range = max_step - min_step
    for i in range(num_steps):
        t_lo = min_step + step_range * i // num_steps  # 区间下界
        t_hi = min_step + step_range * (i + 1) // num_steps  # 区间上界
        t = torch.randint(t_lo, t_hi + 1, (batch_size,), device=device, generator=generator)  # (B,)
        timesteps.append(t)
    
    if not ascending:
        timesteps = timesteps[::-1]  # 反转为从大到小
    
    return timesteps
