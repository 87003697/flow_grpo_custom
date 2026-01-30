"""
可视化 Mixin。

为多步 Tracker 提供中间步可视化能力。
"""

from abc import ABC, abstractmethod
from typing import List, Any
import torch
from PIL import Image


class StepVisualizationMixin(ABC):
    """
    步骤可视化 Mixin。
    
    为多步 Tracker 提供中间步可视化能力。
    子类需要实现 `step_latents` 属性，返回要可视化的 latent 列表。
    
    要求子类定义以下字段：
        - images: List[Image.Image] 用于存储 decode 后的图像
        - height, width: int 图像尺寸
    """
    
    # 子类需要定义这些字段
    images: List[Image.Image]
    height: int
    width: int
    
    @property
    @abstractmethod
    def step_latents(self) -> List[torch.Tensor]:
        """
        返回要可视化的 latent 列表。
        
        子类根据自身数据结构返回：
        - StateTracker: self.z_edits
        - ContrastStateTracker: self.z_edits
        
        Returns:
            List of [B, seq, C] tensors
        """
        pass
    
    @property
    def has_images(self) -> bool:
        """是否已 decode 图像"""
        return len(self.images) > 0
    
    def decode_uniform_samples(self, pipe: Any, n_samples: int = 4) -> None:
        """
        均匀采样 n_samples 个中间步进行并行 decode。
        
        Args:
            pipe: Pipeline（需要 _decode_latent_to_image 方法）
            n_samples: 采样步数，必须是完全平方数（4, 9, 16, ...）
        
        结果存储到 self.images。
        """
        K = len(self.step_latents)
        if K == 0:
            return
        
        # 计算均匀采样的索引
        if n_samples >= K:
            indices = list(range(K))
            n_samples = K
        else:
            indices = [int(i * (K - 1) / (n_samples - 1)) for i in range(n_samples)]
        
        # 收集要 decode 的 latents
        sampled_latents = [self.step_latents[i] for i in indices]
        
        # 固定为单张解码，避免 OOM
        max_batch = 1
        
        # 分批 decode
        decoded_images = []
        with torch.no_grad():
            for start in range(0, n_samples, max_batch):
                chunk_latents = torch.cat(
                    sampled_latents[start : start + max_batch], 
                    dim=0
                )  # [batch, seq_len, C]
                chunk_images = pipe._decode_latent_to_image(
                    chunk_latents, 
                    self.height, 
                    self.width, 
                    "pil"
                )
                decoded_images.extend(chunk_images)
        
        self.images = decoded_images
    
    def get_progress_grid(self, pipe: Any, n_samples: int = 4) -> Image.Image:
        """
        将 n_samples 张中间步图像合成一张 √n × √n 网格图。
        
        Args:
            pipe: Pipeline
            n_samples: 采样步数，必须是完全平方数（4, 9, 16, ...）
        
        Returns:
            合成的网格 PIL Image
        """
        # 检查是否为完全平方数
        grid_size = int(n_samples ** 0.5)
        assert grid_size * grid_size == n_samples, f"n_samples must be a perfect square, got {n_samples}"
        
        # Decode 中间步（如果还没 decode 或数量不对）
        if not self.has_images or len(self.images) != n_samples:
            self.decode_uniform_samples(pipe, n_samples)
        
        # 合成网格
        img_w, img_h = self.images[0].size
        grid_img = Image.new("RGB", (img_w * grid_size, img_h * grid_size), (255, 255, 255))
        
        for i, img in enumerate(self.images):
            row = i // grid_size
            col = i % grid_size
            grid_img.paste(img, (col * img_w, row * img_h))
        
        return grid_img
