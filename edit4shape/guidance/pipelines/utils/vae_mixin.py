"""
可微分 VAE 编码 Mixin。

为 Pipeline 提供可微分 VAE encode 能力。
"""

import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents


class DifferentiableVAEMixin:
    """
    为 Pipeline 提供可微分 VAE encode 能力的 Mixin。
    
    要求 Pipeline 具有以下属性:
        - self.vae: VAE 模型
        - self.latent_channels: latent 通道数
    """
    
    def _encode_vae_image_differentiable(self, image: torch.Tensor) -> torch.Tensor:
        """
        可微分版本的 VAE encode（不带 @torch.no_grad）。
        
        与 _encode_vae_image 相同的逻辑，但保留梯度用于反向传播。
        
        Args:
            image: [B, C, 1, H, W] 图像，[-1, 1] 范围，bfloat16
        
        Returns:
            normalized latent [B, C_lat, 1, H_lat, W_lat]
        """
        # VAE encode，保留梯度
        image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
        
        # 标准化
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        return (image_latents - latents_mean) / latents_std  # [B, C_lat, 1, H_lat, W_lat]
