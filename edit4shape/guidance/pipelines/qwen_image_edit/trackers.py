"""
Qwen-Image Pipeline 状态追踪器。

命名规则：
- z_edit(s): 编辑后的 latent
- x0: 模型预测的 x0（干净图）
- x0_high(s) / x0_low(s): 高/低 CFG 预测
- t / ts: 时间步
- noise: 噪声

包含:
- SDSStateTracker: SDS 状态追踪器
- CSDStateTracker: CSD 状态追踪器
- FlowEditStateTracker: FlowEdit 状态追踪器
- ContrastStateTracker: Multi-Step CSD 状态追踪器
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
import torch
from PIL import Image

from ..utils import (
    BaseStateTracker,
    NoiseMixin,
    NoiseMode,
    StepVisualizationMixin,
    mse_loss_step,
    csd_loss_step,
    mse_loss,
    csd_loss,
)


# =============================================================================
# SDSStateTracker
# =============================================================================

@dataclass
class SDSStateTracker(BaseStateTracker):
    """
    SDS 状态追踪器。
    
    SDS Loss = MSE(src, x0)
    让 src 向模型预测的 x0 靠拢。
    
    Attributes:
        x0: [B, seq, C] 模型预测的 x0（吸引目标）
        t: [B] 采样的时间步
        t_norm: [B, 1, 1] 归一化时间步（用于加权）
        noise: [B, seq, C] 使用的噪声
        image: decode 后的预测图像（按需填充）
    """
    
    x0: torch.Tensor = None           # [B, seq, C] 模型预测的 x0
    t: torch.Tensor = None            # [B] 采样的时间步
    t_norm: torch.Tensor = None       # [B, 1, 1] 归一化时间步
    noise: torch.Tensor = None        # [B, seq, C] 使用的噪声
    image: Image.Image = None         # decode 后的预测图像
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = x0"""
        return self.x0
    
    def loss(
        self, 
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        SDS Loss：MSE(src, x0)
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        return mse_loss_step(
            src,      # [B, seq, C]
            self.x0,  # [B, seq, C]
            ada=ada,
            eps=eps,
        )  # scalar
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode x0 为图像"""
        self.image = self._decode_latent(pipe, self.x0)
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Image.Image:
        """生成对比图：src | x0"""
        src_img = self._decode_latent(pipe, src)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, self.image)


# =============================================================================
# CSDStateTracker
# =============================================================================

@dataclass
class CSDStateTracker(BaseStateTracker):
    """
    CSD 状态追踪器。
    
    CSD Loss = MSE(src, x0_high) - MSE(src, x0_low)
    让 src 向 x0_high 靠拢，同时远离 x0_low。
    
    Attributes:
        x0_high: [B, seq, C] 高 CFG 预测（吸引目标）
        x0_low: [B, seq, C] 低 CFG 预测（排斥目标）
        t: [B] 采样的时间步
        t_norm: [B, 1, 1] 归一化时间步
        noise: [B, seq, C] 使用的噪声
        image: decode 后的预测图像（按需填充）
    """
    
    x0_high: torch.Tensor = None      # [B, seq, C] 高 CFG 预测（吸引目标）
    x0_low: torch.Tensor = None       # [B, seq, C] 低 CFG 预测（排斥目标）
    t: torch.Tensor = None            # [B] 采样的时间步
    t_norm: torch.Tensor = None       # [B, 1, 1] 归一化时间步
    noise: torch.Tensor = None        # [B, seq, C] 使用的噪声
    image: Image.Image = None         # decode 后的预测图像
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = x0_high"""
        return self.x0_high
    
    def loss(
        self, 
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        CSD Loss：MSE(src, x0_high) - MSE(src, x0_low)
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        return csd_loss_step(
            src,           # [B, seq, C]
            self.x0_high,  # [B, seq, C]
            self.x0_low,   # [B, seq, C]
            ada=ada,
            eps=eps,
        )  # scalar
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode x0_high 为图像"""
        self.image = self._decode_latent(pipe, self.x0_high)
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Image.Image:
        """生成对比图：src | x0_high"""
        src_img = self._decode_latent(pipe, src)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, self.image)


# =============================================================================
# FlowEditStateTracker
# =============================================================================

@dataclass
class FlowEditStateTracker(BaseStateTracker, StepVisualizationMixin, NoiseMixin):
    """
    FlowEdit 多步编辑状态追踪器。
    
    记录每步的编辑 latent 和时间步，支持多步监督和可视化。
    
    Latent 格式说明:
        - packed:   [B, seq_len, C]  其中 seq_len = H_lat * W_lat, C = latent_channels * 4
        - unpacked: [B, C, T, H_lat, W_lat]  标准 VAE latent 格式
    
    Attributes:
        z_edits: 每步的编辑 latent，每个 [B, seq, C]
        ts: 每步的时间步 t ∈ (0, 1]
        images: decode 后的中间步图像（按需填充）
        _noise: 当前噪声（NoiseMixin 使用）
        _noise_mode: 噪声模式（NoiseMixin 使用）
    """
    
    z_edits: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq, C]
    ts: List[float] = field(default_factory=list)
    images: List[Image.Image] = field(default_factory=list)
    _noise: Optional[torch.Tensor] = None
    _noise_mode: NoiseMode = "fixed"
    
    def record(self, z_edit: torch.Tensor, t: float) -> None:
        """
        记录一个中间状态。
        
        Args:
            z_edit: [B, seq, C] 当前编辑 latent
            t: 当前时间步
        """
        self.z_edits.append(z_edit.detach().clone())  # [B, seq, C]
        self.ts.append(t)
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最终编辑后的 latent [B, seq, C]"""
        return self.z_edits[-1] if self.z_edits else None
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 latent [K, B, seq, C]"""
        return torch.stack(self.z_edits, dim=0)
    
    def __len__(self) -> int:
        return len(self.z_edits)
    
    @property
    def step_latents(self) -> List[torch.Tensor]:
        """实现 StepVisualizationMixin 要求的属性"""
        return self.z_edits
    
    # =========================================================================
    # Loss 计算
    # =========================================================================
    
    def loss(
        self, 
        src: torch.Tensor,
        reduce: str = "final",
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        FlowEdit Loss。
        
        Args:
            src: [B, seq, C] 有梯度
            reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        return mse_loss(
            src,           # [B, seq, C]
            self.z_edits,  # List of [B, seq, C]
            reduce=reduce,
            ada=ada,
            eps=eps,
        )  # scalar


# =============================================================================
# ContrastStateTracker (Multi-Step CSD)
# =============================================================================

@dataclass
class ContrastStateTracker(BaseStateTracker, StepVisualizationMixin, NoiseMixin):
    """
    Multi-Step CSD (Contrast) 状态追踪器。
    
    在每个去噪步记录高低 CFG 的 x0 预测，用于 Multi-Step CSD loss 计算。
    
    Attributes:
        x0_highs: 每步的高 CFG x0 预测，每个 [B, seq, C]
        x0_lows: 每步的低 CFG x0 预测，每个 [B, seq, C]
        ts: 每步的时间步
        z_edits: 每步的编辑 latent（用于可视化），每个 [B, seq, C]
        _noise: 当前噪声（NoiseMixin 使用）
        _noise_mode: 噪声模式（NoiseMixin 使用）
    """
    
    x0_highs: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq, C]
    x0_lows: List[torch.Tensor] = field(default_factory=list)   # List of [B, seq, C]
    ts: List[float] = field(default_factory=list)
    z_edits: List[torch.Tensor] = field(default_factory=list)   # List of [B, seq, C]
    images: List[Image.Image] = field(default_factory=list)
    _noise: Optional[torch.Tensor] = None
    _noise_mode: NoiseMode = "fixed"
    
    def record(
        self, 
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
        t: float,
        z_edit: Optional[torch.Tensor] = None,
    ) -> None:
        """
        记录一个中间状态。
        
        Args:
            x0_high: [B, seq, C] 高 CFG 预测
            x0_low: [B, seq, C] 低 CFG 预测
            t: 当前时间步
            z_edit: [B, seq, C] 编辑 latent（可选）
        """
        self.x0_highs.append(x0_high.detach().clone())  # [B, seq, C]
        self.x0_lows.append(x0_low.detach().clone())    # [B, seq, C]
        self.ts.append(t)
        if z_edit is not None:
            self.z_edits.append(z_edit.detach().clone())  # [B, seq, C]
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最后一步的 x0_high [B, seq, C]"""
        return self.x0_highs[-1] if self.x0_highs else None
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def __len__(self) -> int:
        return len(self.x0_highs)
    
    @property
    def step_latents(self) -> List[torch.Tensor]:
        """实现 StepVisualizationMixin 要求的属性"""
        return self.z_edits
    
    # =========================================================================
    # Loss 计算
    # =========================================================================
    
    def loss(
        self, 
        src: torch.Tensor,
        reduce: str = "inv_weighted",
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        计算 Multi-Step CSD Loss。
        
        Args:
            src: [B, seq, C] 有梯度
            reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        return csd_loss(
            src,            # [B, seq, C]
            self.x0_highs,  # List of [B, seq, C]
            self.x0_lows,   # List of [B, seq, C]
            reduce=reduce,
            ada=ada,
            eps=eps,
        )  # scalar
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode 最终的 x0_high 为图像"""
        if self.x0_highs:
            self.image = self._decode_latent(pipe, self.x0_highs[-1])
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Optional[Image.Image]:
        """生成对比图：src | x0_high_final"""
        src_img = self._decode_latent(pipe, src)
        if not hasattr(self, 'image') or self.image is None:
            if self.x0_highs:
                self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, getattr(self, 'image', None))
