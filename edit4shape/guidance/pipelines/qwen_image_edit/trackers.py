"""
Qwen-Image Pipeline 状态追踪器。

命名规则：
- z_edits: 编辑后的 latent（FlowEdit 专用）
- x0_highs / x0_lows: 高/低 CFG 预测
- ts: 时间步列表
- noise: 噪声

包含:
- MultiStepTrackerMixin: 多步追踪器公共逻辑
- DistillationStateTracker: 蒸馏状态追踪器（支持 MTS）
- FlowEditStateTracker: FlowEdit 状态追踪器（支持 MSE + CSD 混合 loss）
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
import torch
from PIL import Image

from ..utils import (
    BaseStateTracker,
    NoiseMixin,
    NoiseMode,
    NoiseInversionMixin,
    StepVisualizationMixin,
    ReduceMode,
    reduce_losses,
    mse_loss_step,
    csd_loss_step,
    mse_loss,
    csd_loss,
)


# =============================================================================
# MultiStepTrackerMixin - 公共逻辑
# =============================================================================

class MultiStepTrackerMixin:
    """
    多步追踪器公共逻辑 Mixin。
    
    提供：
    - x0_highs / x0_lows 相关属性
    - num_steps 属性
    - CSD loss 计算
    """
    
    x0_highs: List[torch.Tensor]  # List[[B, seq, C]] 高 CFG 预测
    x0_lows: List[torch.Tensor]   # List[[B, seq, C]] 低 CFG 预测
    ts: List                      # 时间步列表
    
    @property
    def num_steps(self) -> int:
        """返回记录的步数"""
        return len(self.x0_highs) if self.x0_highs else 0
    
    def _compute_csd_loss(
        self,
        src: torch.Tensor,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算 CSD Loss：MSE(src, x0_high) - MSE(src, x0_low)
        
        Args:
            src: [B, seq, C] 有梯度
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        return csd_loss(
            src,
            self.x0_highs,
            self.x0_lows,
            reduce=reduce,
            ada=ada,
            eps=eps,
        )
    
    def _compute_mse_loss(
        self,
        src: torch.Tensor,
        targets: List[torch.Tensor],
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        计算 MSE Loss。
        
        Args:
            src: [B, seq, C] 有梯度
            targets: List[[B, seq, C]] 目标 latent 列表
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式
        
        Returns:
            scalar loss
        """
        return mse_loss(
            src,
            targets,
            reduce=reduce,
            ada=ada,
            eps=eps,
        )


# =============================================================================
# DistillationStateTracker（支持 MTS + Noise Inversion）
# =============================================================================

@dataclass
class DistillationStateTracker(BaseStateTracker, MultiStepTrackerMixin, NoiseInversionMixin):
    """
    蒸馏状态追踪器（支持多时间步 MTS + DNAEdit 风格 Noise Inversion）。
    
    通过 mse_weight 和 csd_weight 控制 loss 类型：
    - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_high)
    - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_high) - MSE(src, x0_low)
    - mse_weight=1, csd_weight=1 → 混合模式
    
    统一接口（自动根据 use_inversion 切换模式）：
        tracker.init_noise(x_src, noise, use_inversion=True)
        for t in timesteps:
            latents_noisy = tracker.get_noisy_latent(x_src, t)
            v_pred = model(latents_noisy, t)
            tracker.update(latents_noisy, v_pred, t)
    
    Attributes:
        x0_highs: List[[B, seq, C]] 高 CFG 预测（MSE/CSD 目标）
        x0_lows: List[[B, seq, C]] 低 CFG 预测（CSD 排斥目标）
        x0_cfgs: List[[B, seq, C]] CFG 后预测（MSE 目标）
    """
    
    x0_highs: List[torch.Tensor] = None  # List[[B, seq, C]] 纯 cond 预测（CSD 吸引目标）
    x0_lows: List[torch.Tensor] = None   # List[[B, seq, C]] 纯 uncond 预测（CSD 排斥目标）
    x0_cfgs: List[torch.Tensor] = None   # List[[B, seq, C]] CFG 后预测（MSE 目标）
    height: int = None                   # 图像高度
    width: int = None                    # 图像宽度
    image: Image.Image = None            # decode 后的预测图像
    
    # 噪声管理字段
    _noise: Optional[torch.Tensor] = None           # 噪声
    _use_inversion: bool = False                    # 是否启用 noise inversion
    _noise_mode: str = "aligned_cond"               # inversion 使用的速度模式
    
    # NoiseInversionMixin 需要的字段
    _noise_inv: Optional[torch.Tensor] = None       # 噪声（inversion 模式）
    _z_prev: Optional[torch.Tensor] = None          # 上一步的 latent 位置
    _t_prev: float = 0.0                            # 上一时间步
    
    # =========================================================================
    # 统一噪声管理接口（自动根据 use_inversion 切换模式）
    # =========================================================================
    
    def init_noise(
        self,
        x_src: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noise_mode: str = "fixed",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化噪声（统一接口）。
        
        Args:
            x_src: [B, seq, C] 源 latent
            noise: [B, seq, C] 噪声（可选，不提供则随机生成）
            noise_mode: 噪声模式
                - "random": 每次随机噪声
                - "fixed": 固定噪声
                - "aligned_cond": Noise Inversion（基于 v_cond）
                - "aligned_uncond": Noise Inversion（基于 v_uncond）
                - "aligned_cfg": Noise Inversion（基于 v_cfg）
            seed: 随机种子
        
        Returns:
            [B, seq, C] 噪声
        """
        self._noise_mode = noise_mode
        self._use_inversion = noise_mode.startswith("aligned")
        
        # 生成或使用提供的噪声
        if noise is not None:
            self._noise = noise.clone()
        elif seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn_like(x_src, generator=generator)
        else:
            self._noise = torch.randn_like(x_src)
        
        # Inversion 模式额外初始化
        if self._use_inversion:
            self._noise_inv = self._noise.clone()
            self._z_prev = x_src.clone()
            self._t_prev = 0.0
        
        return self._noise
    
    def get_noisy_latent(self, x_src: torch.Tensor, t: float) -> torch.Tensor:
        """
        获取 noisy latent（统一接口）。
        
        Args:
            x_src: [B, seq, C] 源 latent（干净图像）
            t: 当前时间步 [0, 1]
        
        Returns:
            [B, seq, C] noisy latent
        """
        if self._use_inversion:
            # Noise Inversion 模式：插值
            return self.get_noise_inv(t)
        elif self._noise_mode == "random":
            # Random 模式：每次随机噪声
            noise = torch.randn_like(x_src)  # [B, seq, C]
            return (1.0 - t) * x_src + t * noise
        else:
            # Fixed 模式：使用初始化时的固定噪声
            return (1.0 - t) * x_src + t * self._noise
    
    def update(
        self,
        z_t: torch.Tensor,
        v_cond: torch.Tensor,
        v_uncond: torch.Tensor,
        v_cfg: torch.Tensor,
        t: float,
    ) -> None:
        """
        计算并记录 x0，更新 Noise Inversion 状态。
        
        内部计算 x0（Flow Matching 公式: x0 = z_t - t * v）：
        - x0_high = z_t - t * v_cond   （纯 cond，CSD 吸引目标）
        - x0_low = z_t - t * v_uncond  （纯 uncond，CSD 排斥目标）
        - x0_cfg = z_t - t * v_cfg     （CFG 后，MSE 目标）
        
        Args:
            z_t: [B, seq, C] 当前 noisy latent
            v_cond: [B, seq, C] 纯 cond 速度
            v_uncond: [B, seq, C] 纯 uncond 速度
            v_cfg: [B, seq, C] CFG 后速度
            t: 当前时间步 [0, 1]
        """
        # 计算 x0（Flow Matching 公式: x0 = z_t - t * v）
        x0_high = z_t - t * v_cond    # [B, seq, C] 纯 cond（CSD 吸引目标）
        x0_low = z_t - t * v_uncond   # [B, seq, C] 纯 uncond（CSD 排斥目标）
        x0_cfg = z_t - t * v_cfg      # [B, seq, C] CFG 后（MSE 目标）
        
        # 记录 x0_high
        if self.x0_highs is None:
            self.x0_highs = []
        self.x0_highs.append(x0_high)
        
        # 记录 x0_low
        if self.x0_lows is None:
            self.x0_lows = []
        self.x0_lows.append(x0_low)
        
        # 记录 x0_cfg
        if self.x0_cfgs is None:
            self.x0_cfgs = []
        self.x0_cfgs.append(x0_cfg)
        
        # 更新 Noise Inversion（根据 noise_mode 选择速度）
        if self._use_inversion:
            v_inv = {
                "aligned_cond": v_cond,
                "aligned_uncond": v_uncond,
                "aligned_cfg": v_cfg,
            }.get(self._noise_mode, v_cond)
            self.update_noise_inv(z_t, v_inv, t)
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """最终噪声（inversion 模式返回更新后的噪声）"""
        if self._use_inversion:
            return self._noise_inv
        return self._noise
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最后一个 x0_cfg（CFG 后的最佳预测）"""
        return self.x0_cfgs[-1] if self.x0_cfgs else None
    
    def loss(
        self, 
        src: torch.Tensor,
        mse_weight: float = 0.0,
        csd_weight: float = 1.0,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        统一 Loss 计算（支持多时间步聚合）。
        
        Loss = mse_weight * MSE(src, x0_cfgs) + csd_weight * CSD(src, x0_highs, x0_lows)
        
        - MSE: 蒸馏到 CFG 后的最佳预测 x0_cfg
        - CSD: 对比学习，吸引纯 cond (x0_high)，排斥纯 uncond (x0_low)
        
        Args:
            src: [B, seq, C] 有梯度
            mse_weight: MSE loss 权重（蒸馏到 x0_cfg）
            csd_weight: CSD loss 权重（吸引 x0_high，排斥 x0_low）
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
        
        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)
        
        # MSE Loss: MSE(src, x0_cfgs) — 蒸馏到 CFG 后的预测
        if mse_weight > 0 and self.x0_cfgs:
            loss_mse = self._compute_mse_loss(src, self.x0_cfgs, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + mse_weight * loss_mse
        
        # CSD Loss: MSE(src, x0_high) - MSE(src, x0_low) — 对比纯 cond vs 纯 uncond
        if csd_weight > 0 and self.x0_lows:
            loss_csd = self._compute_csd_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + csd_weight * loss_csd
        
        return total_loss
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode x0_high 为图像（使用最后一个时间步）"""
        self.image = self._decode_latent(pipe, self.target)
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Image.Image:
        """生成对比图：src | x0_high"""
        src_img = self._decode_latent(pipe, src)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, self.image)


# =============================================================================
# FlowEditStateTracker（支持 MSE + CSD 混合 loss）
# =============================================================================

@dataclass
class FlowEditStateTracker(BaseStateTracker, MultiStepTrackerMixin, StepVisualizationMixin, NoiseMixin):
    """
    FlowEdit 状态追踪器。
    
    同时记录：
    - z_edits: 编辑后的 latent（MSE loss 目标）
    - x0_highs: 高 CFG x0 预测（CSD loss 目标）
    - x0_lows: 低 CFG x0 预测（CSD 排斥目标）
    
    通过 mse_weight 和 csd_weight 灵活组合 loss：
    - mse_weight=0, csd_weight=1 → 纯 CSD
    - mse_weight=1, csd_weight=0 → 纯 MSE
    - mse_weight=0.5, csd_weight=1 → 混合模式
    
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
    
    # 尺寸（与 DistillationStateTracker 统一）
    height: int = None
    width: int = None
    
    # 可视化
    image: Image.Image = None  # 单张预测图像
    images: List[Image.Image] = field(default_factory=list)  # 中间步图像
    
    # 噪声管理（NoiseMixin）
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
        self.z_edits.append(z_edit.detach().clone())
        self.x0_highs.append(x0_high.detach().clone())
        self.x0_lows.append(x0_low.detach().clone())
        self.ts.append(t)
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最终编辑后的 latent"""
        return self.z_edits[-1] if self.z_edits else None
    
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
        mse_weight: float = 0.0,
        csd_weight: float = 1.0,
        ada: bool = False,
        eps: float = 1e-4,
        reduce: ReduceMode = "mean",
    ) -> torch.Tensor:
        """
        统一 Loss 计算。
        
        Loss = mse_weight * MSE(src, z_edits) + csd_weight * CSD(src, x0_highs, x0_lows)
        
        Args:
            src: [B, seq, C] 有梯度
            mse_weight: MSE loss 权重（吸引到 z_edit）
            csd_weight: CSD loss 权重（吸引高 CFG，排斥低 CFG）
            ada: 是否使用自适应归一化
            eps: ada 模式的 epsilon
            reduce: 聚合方式 "final" | "mean" | "weighted" | "inv_weighted"
        
        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=src.device, dtype=src.dtype)
        
        # MSE Loss: MSE(src, z_edits)
        if mse_weight > 0:
            loss_mse = self._compute_mse_loss(src, self.z_edits, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + mse_weight * loss_mse
        
        # CSD Loss: MSE(src, x0_high) - MSE(src, x0_low)
        if csd_weight > 0:
            loss_csd = self._compute_csd_loss(src, ada=ada, eps=eps, reduce=reduce)
            total_loss = total_loss + csd_weight * loss_csd
        
        return total_loss
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode 最终的 z_edit 为图像"""
        self.image = self._decode_latent(pipe, self.target)
    
    def get_comparison_image(self, pipe: Any, src: torch.Tensor) -> Image.Image:
        """生成对比图：src | z_edit_final"""
        src_img = self._decode_latent(pipe, src)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(src_img, self.image)
