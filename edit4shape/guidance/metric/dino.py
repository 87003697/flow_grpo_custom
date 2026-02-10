"""DINOv3 特征相似度 loss。"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

from .base import BaseMetric


class DINOMetric(BaseMetric):
    """
    DINOv3 特征空间相似度 Metric。
    
    使用 DINOv3 提取特征，计算余弦相似度，loss = 1 - similarity。
    保持可导：rendered 保持梯度，target 无梯度。
    """
    
    name = "dino"
    
    def __init__(
        self,
        weight: float,
        device: torch.device,
        model_path: str = "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size: int = 518,
        **kwargs,
    ):
        super().__init__(weight, device)
        
        logging.info(f"[DINOMetric] Loading DINOv3: {model_path}")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # ImageNet 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)  # (1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)   # (1,3,1,1)
        self.size = image_size
        logging.info(f"[DINOMetric] Initialized (weight={weight}, size={image_size})")
    
    def compute(
        self,
        rendered: torch.Tensor,  # (B,C,H,W) [0,1]，有梯度
        target: torch.Tensor,    # (B,C,H,W) [0,1]，无梯度
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """计算 DINOv3 特征相似度 loss。"""
        # Resize（可导）
        r = F.interpolate(rendered, size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,C,size,size)
        t = F.interpolate(target.detach(), size=(self.size, self.size), mode='bilinear', align_corners=False)  # (B,C,size,size)
        
        # ImageNet 归一化（可导）
        r = (r - self.mean) / self.std  # (B,C,size,size)
        t = (t - self.mean) / self.std  # (B,C,size,size)
        
        # 提取特征（rendered 保持梯度，target 无梯度）
        feats_r = self.model(r).last_hidden_state  # (B, N, D)
        with torch.no_grad():
            feats_t = self.model(t).last_hidden_state  # (B, N, D)
        
        # 余弦相似度
        feats_r = F.normalize(feats_r, dim=-1)  # (B, N, D)
        feats_t = F.normalize(feats_t.detach(), dim=-1)  # (B, N, D)
        sim = (feats_r * feats_t).sum(dim=-1).mean()  # scalar
        
        return 1 - sim  # loss

    @torch.no_grad()
    def compute_from_pil(
        self,
        rendered_pils: list[Image.Image],
        target_pils: list[Image.Image],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """PIL 输入路径，使用 AutoImageProcessor 预处理。"""
        r_inputs = self.processor(images=rendered_pils, return_tensors="pt")
        t_inputs = self.processor(images=target_pils, return_tensors="pt")

        r = r_inputs["pixel_values"].to(self.device)  # (B,3,H,W)
        t = t_inputs["pixel_values"].to(self.device)  # (B,3,H,W)

        feats_r = self.model(r).last_hidden_state[:, 0]  # (B, D)
        feats_t = self.model(t).last_hidden_state[:, 0]  # (B, D)

        feats_r = F.normalize(feats_r, dim=-1)  # (B, D)
        feats_t = F.normalize(feats_t, dim=-1)  # (B, D)
        sim = (feats_r * feats_t).sum(dim=-1).mean()  # scalar

        return 1 - sim  # loss
    
    def cleanup(self) -> None:
        if hasattr(self, 'model'):
            del self.model
            logging.info("[DINOMetric] Cleaned up.")

