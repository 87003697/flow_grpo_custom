"""
Qwen-Image Pipeline 工具模块。

包含:
- FlowEditStateTracker: FlowEdit 中间状态跟踪器
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin
"""

from dataclasses import dataclass, field
from typing import List, Any
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents


# =============================================================================
# FlowEditStateTracker
# =============================================================================

@dataclass
class FlowEditStateTracker:
    """
    FlowEdit 中间状态跟踪器。
    
    记录每步的 packed latent 和 t 值，支持多步监督和可视化。
    
    Latent 格式说明:
        - packed:   [B, seq_len, C]  其中 seq_len = H_lat * W_lat, C = latent_channels * 4
        - unpacked: [B, C, T, H_lat, W_lat]  标准 VAE latent 格式
    
    Attributes:
        latents: 每步的 packed latent，每个元素 shape: [B, seq_len, C]
        t_values: 每步的时间步 t ∈ (0, 1]
        height: 图像高度（用于 unpack 时计算 H_lat = height // vae_scale_factor // 2）
        width: 图像宽度（用于 unpack 时计算 W_lat = width // vae_scale_factor // 2）
        images: decode 后的中间步图像（按需填充）
    """
    
    latents: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq_len, C] packed latents
    t_values: List[float] = field(default_factory=list)         # 每步的 t
    height: int = 0
    width: int = 0
    images: List[Image.Image] = field(default_factory=list)     # decode 后的中间步图像
    
    def record(self, z_edit: torch.Tensor, t: float) -> None:
        """
        记录一个中间状态。
        
        Args:
            z_edit: 当前 packed latent [B, seq, C]
            t: 当前时间步
        """
        self.latents.append(z_edit.detach().clone())
        self.t_values.append(t)
    
    @property
    def final(self) -> torch.Tensor:
        """最终 latent [B, seq, C]"""
        return self.latents[-1]
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 latent [K, B, seq, C]"""
        return torch.stack(self.latents, dim=0)
    
    def __len__(self) -> int:
        return len(self.latents)
    
    # =========================================================================
    # Loss 计算
    # =========================================================================
    
    def loss_final(self, rendered: torch.Tensor) -> torch.Tensor:
        """
        只用最终 latent 计算 loss。
        
        Args:
            rendered: 渲染图的 packed latent [B, seq, C]
        """
        return F.mse_loss(rendered.float(), self.final.float())
    
    def loss_mean(self, rendered: torch.Tensor) -> torch.Tensor:
        """
        所有中间步均匀加权的 loss。
        
        Args:
            rendered: 渲染图的 packed latent [B, seq, C]
        """
        losses = []
        for lat in self.latents:
            # lat: [B, seq, C], rendered: [B, seq, C]
            diff = rendered.float() - lat.float()  # [B, seq, C]
            mse_per_sample = (diff ** 2).mean(dim=(1, 2))  # [B]
            losses.append(mse_per_sample.mean())  # scalar
        return torch.stack(losses).mean()  # scalar
    
    def loss_weighted(self, rendered: torch.Tensor) -> torch.Tensor:
        """
        用编辑次数的倒数加权的 loss。
        
        第 k 次编辑（k=1,2,...,K）权重 = 1/k
        编辑越多，权重越低。
        
        Args:
            rendered: 渲染图的 packed latent [B, seq, C]
        """
        losses = []
        for lat in self.latents:
            # lat: [B, seq, C], rendered: [B, seq, C]
            diff = rendered.float() - lat.float()  # [B, seq, C]
            mse_per_sample = (diff ** 2).mean(dim=(1, 2))  # [B]
            losses.append(mse_per_sample.mean())  # scalar
        losses = torch.stack(losses)  # [K]
        K = len(losses)
        
        # 权重 = 1/k, k = 1, 2, ..., K
        w = 1.0 / torch.arange(1, K + 1, device=losses.device, dtype=losses.dtype)  # [K]
        w = w / w.sum()  # 归一化
        
<<<<<<<< HEAD:edit4shape/guidance/pipelines/qwen_image_edit/state_tracker.py
        return (losses * w).sum()

========
        return (losses * w).sum()  # scalar
    
>>>>>>>> origin/trellis_distill:edit4shape/guidance/pipelines/qwen_image_edit/utils.py
    def loss_ada(self, rendered: torch.Tensor) -> torch.Tensor:
        """
        自适应归一化的 loss（线性缩放，放在 MSE 外面）。
        
        用每步 target latent 的绝对值均值作为归一化因子（per-sample）。
        当 target 幅度大时 loss 被缩小，幅度小时 loss 被放大。
        
        Args:
            rendered: 渲染图的 packed latent [B, seq, C]
        """
        losses = []
        for lat in self.latents:
            # lat: [B, seq, C], rendered: [B, seq, C]
            diff = rendered.float() - lat.float()  # [B, seq, C]
            mse_per_sample = (diff ** 2).mean(dim=(1, 2))  # [B]
            normalizer = lat.abs().mean(dim=(1, 2)) + 1e-2  # [B]
            losses.append((mse_per_sample / normalizer.detach()).mean())  # scalar
        return torch.stack(losses).mean()  # scalar
<<<<<<<< HEAD:edit4shape/guidance/pipelines/qwen_image_edit/state_tracker.py

========
    
>>>>>>>> origin/trellis_distill:edit4shape/guidance/pipelines/qwen_image_edit/utils.py
    def loss_ada_position(self, rendered: torch.Tensor) -> torch.Tensor:
        """
        Position-wise 自适应归一化的 loss。
        
        每个 position (token) 有自己的 normalizer，粒度比 ada 更细。
        注意：epsilon 设为 1e-2 以避免小幅度 position 导致梯度爆炸。
        
        Args:
            rendered: 渲染图的 packed latent [B, seq, C]
        """
        losses = []
        for lat in self.latents:
            # lat: [B, seq, C], rendered: [B, seq, C]
            diff = rendered.float() - lat.float()  # [B, seq, C]
            mse_per_position = (diff ** 2).mean(dim=2)  # [B, seq]
            normalizer = lat.abs().mean(dim=2) + 1e-2  # [B, seq]
            losses.append((mse_per_position / normalizer.detach()).mean())  # scalar
        return torch.stack(losses).mean()  # scalar
    
    # =========================================================================
    # Decode 与可视化
    # =========================================================================
    
    @property
    def has_images(self) -> bool:
        """是否已 decode 图像"""
        return len(self.images) > 0
    
    def decode_uniform_samples(self, pipe: Any, n_samples: int = 4) -> None:
        """
        均匀采样 n_samples 个中间步进行并行 decode。
        
        Args:
            pipe: FlowEdit pipeline（需要 _decode_latent_to_image 方法）
            n_samples: 采样步数，必须是完全平方数（4, 9, 16, ...）
        
        结果存储到 self.images。
        """
        K = len(self.latents)
        if K == 0:
            return
        
        # 计算均匀采样的索引
        if n_samples >= K:
            indices = list(range(K))
            n_samples = K
        else:
            indices = [int(i * (K - 1) / (n_samples - 1)) for i in range(n_samples)]
        
        # 收集要 decode 的 latents
        # 每个 latent: [B, seq_len, C] packed，这里 B=1
        sampled_latents = [self.latents[i] for i in indices]
        
        # 固定为单张解码，避免 OOM
        max_batch = 1
        
        # 分批 decode（单张）
        decoded_images = []
        with torch.no_grad():
            for start in range(0, n_samples, max_batch):
                chunk_latents = torch.cat(  # [1, seq_len, C] packed
                    sampled_latents[start : start + max_batch], 
                    dim=0
                )
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
            pipe: FlowEdit pipeline
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
        
        # 合成网格：grid_size × grid_size
        img_w, img_h = self.images[0].size
        grid_img = Image.new("RGB", (img_w * grid_size, img_h * grid_size), (255, 255, 255))
        
        for i, img in enumerate(self.images):
            row = i // grid_size
            col = i % grid_size
            grid_img.paste(img, (col * img_w, row * img_h))
        
        return grid_img


# =============================================================================
# DifferentiableVAEMixin
# =============================================================================

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
