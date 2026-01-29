"""
Qwen-Image Pipeline 状态追踪器。

命名规则：
- z_edit(s): 编辑后的 latent
- x0: 模型预测的 x0（干净图）
- x0_high(s) / x0_low(s): 高/低 CFG 预测
- t / ts: 时间步
- noise: 噪声

包含:
- DistillationStateTracker: 单步蒸馏状态追踪器（统一 SDS + CSD）
- FlowEditStateTracker: FlowEdit 多步状态追踪器（支持 MSE + CSD 混合 loss）
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
# DistillationStateTracker（统一 SDS + CSD）
# =============================================================================

@dataclass
class DistillationStateTracker(BaseStateTracker):
    """
    单步蒸馏状态追踪器（统一 SDS + CSD）。
    
    通过 sds_weight 和 csd_weight 控制 loss 类型：
    - sds_weight=1, csd_weight=0 → 纯 SDS: MSE(src, x0_high)
    - sds_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_high) - MSE(src, x0_low)
    - sds_weight=1, csd_weight=1 → 混合模式
    
    Attributes:
        x0_high: [B, seq, C] 高 CFG 预测（吸引目标）
        x0_low: [B, seq, C] 低 CFG 预测（排斥目标，仅 CSD 模式使用）
        t: [B] 采样的时间步
        t_norm: [B, 1, 1] 归一化时间步（用于加权）
        noise: [B, seq, C] 使用的噪声
        image: decode 后的预测图像（按需填充）
    """
    
    x0_high: torch.Tensor = None      # [B, seq, C] 高 CFG 预测（吸引目标）
    x0_low: torch.Tensor = None       # [B, seq, C] 低 CFG 预测（排斥目标，可选）
    t: torch.Tensor = None            # [B] 采样的时间步
    t_norm: torch.Tensor = None       # [B, 1, 1] 归一化时间步
    noise: torch.Tensor = None        # [B, seq, C] 使用的噪声
    height: int = None                # 图像高度
    width: int = None                 # 图像宽度
    image: Image.Image = None         # decode 后的预测图像
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = x0_high"""
        return self.x0_high
    
    def loss(
        self, 
        src: torch.Tensor,
        sds_weight: float = 0.0,
        csd_weight: float = 1.0,
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        统一 Loss 计算。
        
        Loss = sds_weight * MSE(src, x0_high) + csd_weight * (MSE(src, x0_high) - MSE(src, x0_low))
        
        Args:
            src: [B, seq, C] 有梯度
            sds_weight: SDS loss 权重（MSE 吸引）
            csd_weight: CSD loss 权重（差分吸引-排斥）
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)
        
        # SDS Loss: MSE(src, x0_high)
        if sds_weight > 0:
            loss_sds = mse_loss_step(
                src,           # [B, seq, C]
                self.x0_high,  # [B, seq, C]
                ada=ada,
                eps=eps,
            )  # scalar
            total_loss = total_loss + sds_weight * loss_sds
        
        # CSD Loss: MSE(src, x0_high) - MSE(src, x0_low)
        if csd_weight > 0:
            loss_csd = csd_loss_step(
                src,           # [B, seq, C]
                self.x0_high,  # [B, seq, C]
                self.x0_low,   # [B, seq, C]
                ada=ada,
                eps=eps,
            )  # scalar
            total_loss = total_loss + csd_weight * loss_csd
        
        return total_loss
    
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
# FlowEditStateTracker (统一版本)
# =============================================================================

@dataclass
class FlowEditStateTracker(BaseStateTracker, StepVisualizationMixin, NoiseMixin):
    """
    FlowEdit 统一状态追踪器。
    
    同时记录：
    - z_edits: 编辑后的 latent（用于 MSE loss）
    - x0_highs: 高 CFG x0 预测（用于 CSD loss）
    - x0_lows: 低 CFG x0 预测（用于 CSD loss）
    
    支持通过 csd_weight 和 mse_weight 灵活组合 loss：
    - csd_weight=1, mse_weight=0 → 纯 CSD
    - csd_weight=0, mse_weight=1 → 纯 MSE
    - csd_weight=1, mse_weight=0.5 → 混合模式
    
    Latent 格式说明:
        - packed:   [B, seq_len, C]  其中 seq_len = H_lat * W_lat, C = latent_channels * 4
        - unpacked: [B, C, T, H_lat, W_lat]  标准 VAE latent 格式
    
    Attributes:
        z_edits: 每步的编辑 latent，每个 [B, seq, C]
        x0_highs: 每步的高 CFG x0 预测，每个 [B, seq, C]
        x0_lows: 每步的低 CFG x0 预测，每个 [B, seq, C]
        ts: 每步的时间步 t ∈ (0, 1]
        images: decode 后的中间步图像（按需填充）
        _noise: 当前噪声（NoiseMixin 使用）
        _noise_mode: 噪声模式（NoiseMixin 使用）
    """
    
    # 编辑 latent（MSE loss 目标）
    z_edits: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq, C]
    
    # x0 预测（CSD loss 目标）
    x0_highs: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq, C]
    x0_lows: List[torch.Tensor] = field(default_factory=list)   # List of [B, seq, C]
    
    # 时间步
    ts: List[float] = field(default_factory=list)
    
    # 可视化
    images: List[Image.Image] = field(default_factory=list)
    
    # 噪声管理
    _noise: Optional[torch.Tensor] = None
    _noise_mode: NoiseMode = "fixed"
    
    def record(
        self, 
        z_edit: torch.Tensor,
        t: float,
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录一个中间状态。
        
        Args:
            z_edit: [B, seq, C] 编辑后的 latent
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG x0 预测
            x0_low: [B, seq, C] 低 CFG x0 预测
        """
        self.z_edits.append(z_edit.detach().clone())      # [B, seq, C]
        self.x0_highs.append(x0_high.detach().clone())    # [B, seq, C]
        self.x0_lows.append(x0_low.detach().clone())      # [B, seq, C]
        self.ts.append(t)
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最终编辑后的 latent [B, seq, C]"""
        return self.z_edits[-1]
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 z_edit latent [K, B, seq, C]"""
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
        csd_weight: float = 1.0,
        mse_weight: float = 0.0,
        reduce: str = "mean",
        ada: bool = False,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        统一的 Loss 计算。
        
        Loss = csd_weight * CSD_Loss + mse_weight * MSE_Loss
        
        其中：
        - CSD_Loss = MSE(src, x0_high) - MSE(src, x0_low)
        - MSE_Loss = MSE(src, z_edits)
        
        通过权重控制模式：
        - csd_weight=1, mse_weight=0 → 纯 CSD
        - csd_weight=0, mse_weight=1 → 纯 MSE
        - csd_weight=1, mse_weight=0.5 → 混合模式
        
        Args:
            src: [B, seq, C] 有梯度
            csd_weight: CSD loss 权重
            mse_weight: MSE loss 权重
            reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
        
        Returns:
            scalar loss
        """
        # CSD Loss: MSE(src, x0_high) - MSE(src, x0_low)
        loss_csd = csd_loss(
            src,            # [B, seq, C]
            self.x0_highs,  # List of [B, seq, C]
            self.x0_lows,   # List of [B, seq, C]
            reduce=reduce,
            ada=ada,
            eps=eps,
        )  # scalar
        
        # MSE Loss: MSE(src, z_edits)
        loss_mse = mse_loss(
            src,           # [B, seq, C]
            self.z_edits,  # List of [B, seq, C]
            reduce=reduce,
            ada=ada,
            eps=eps,
        )  # scalar
        
        return csd_weight * loss_csd + mse_weight * loss_mse
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode 最终的 z_edit 为图像"""
        self.image = self._decode_latent(pipe, self.z_edits[-1])
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Image.Image:
        """生成对比图：src | z_edit_final"""
        src_img = self._decode_latent(pipe, src)
        if not hasattr(self, 'image') or self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, self.image)
