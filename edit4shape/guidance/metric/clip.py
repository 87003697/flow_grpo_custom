"""CLIP 图像特征相似度 loss。"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel

from .base import BaseMetric


class CLIPMetric(BaseMetric):
    """
    CLIP 图像特征空间相似度 Metric。
    
    使用 CLIP Vision Encoder 提取图像级特征，计算余弦相似度，loss = 1 - similarity。
    保持可导：rendered 保持梯度，target 无梯度。
    
    参考：reward_models/rewards.py 中 CLIPModel 的使用方式。
    """
    
    name = "clip"
    
    def __init__(
        self,
        weight: float,
        device: torch.device,
        model_path: str = "pretrained_weights/clip/clip-vit-large-patch14",
        image_size: int = 224,
        **kwargs,
    ):
        super().__init__(weight, device)
        
        logging.info(f"[CLIPMetric] Loading CLIP: {model_path}")
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # CLIP 归一化参数（OpenAI CLIP 标准值）
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)  # (1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)   # (1,3,1,1)
        self.size = image_size
        logging.info(f"[CLIPMetric] Initialized (weight={weight}, size={image_size})")
    
    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # (B,C,H,W) [0,1]，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """计算 CLIP 图像特征相似度 loss。"""
        # Resize（可导）
        r = F.interpolate(rendered, size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,3,size,size)
        t = F.interpolate(target.detach(), size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,3,size,size)
        
        # CLIP 归一化（可导）
        r = (r - self.mean) / self.std  # (B,3,size,size)
        t = (t - self.mean) / self.std  # (B,3,size,size)
        
        # 提取图像级特征（rendered 保持梯度，target 无梯度）
        feats_r = self.model.get_image_features(pixel_values=r)  # (B, D)
        with torch.no_grad():
            feats_t = self.model.get_image_features(pixel_values=t)  # (B, D)
        
        # 余弦相似度
        feats_r = F.normalize(feats_r, dim=-1)  # (B, D)
        feats_t = F.normalize(feats_t.detach(), dim=-1)  # (B, D)
        sim = (feats_r * feats_t).sum(dim=-1).mean()  # scalar
        
        return 1 - sim  # scalar
    
    def cleanup(self) -> None:
        if hasattr(self, 'model'):
            del self.model
            logging.info("[CLIPMetric] Cleaned up.")
