"""
Qwen-Image Pipeline 状态追踪器。

命名规则：
- x0_preds: 预测的 x0（MSE loss 目标）
- x0_highs / x0_lows: 高/低 CFG 预测（CSD loss 目标）
- ts: 时间步列表
- noise: 噪声

包含:
- StateTracker: 统一状态追踪器（random / fixed / aligned 模式）
- InversionStateTracker: Inversion 状态追踪器（inversion_* 模式）
- TrajectoryStateTracker: 轨迹对齐状态追踪器（traj_* 模式）
- create_distillation_tracker: 工厂函数，根据 noise_mode 创建 Tracker（distillation 用）
- create_flowedit_tracker: 工厂函数，根据 noise_mode 创建 Tracker（flowedit 用）
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional, Union
import torch
from PIL import Image

from ..utils import (
    BaseStateTracker,
    BaseNoiseMixin,
    NaiveInversionMixin,
    TrajectoryNoiseMixin,
    NoiseMode,
    VisualizationMixin,
    LossMixin,
)


# =============================================================================
# StateTracker - 统一状态追踪器
# =============================================================================

@dataclass
class StateTracker(BaseStateTracker, LossMixin, VisualizationMixin, BaseNoiseMixin):
    """
    统一状态追踪器（random / fixed / aligned 模式）。
    
    存储：
    - x0_preds: 预测的 x0（MSE loss 目标）
    - x0_highs: 高 CFG 预测（CSD 吸引目标）
    - x0_lows: 低 CFG 预测（CSD 排斥目标）
    - ts: 时间步列表
    
    通过 mse_weight 和 csd_weight 控制 loss 类型：
    - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_pred)
    - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_high) - MSE(src, x0_low)
    - mse_weight=1, csd_weight=1 → 混合模式
    
    使用方式：
        tracker = StateTracker(height=H, width=W)
        tracker.init(x_src, mode="aligned", seed=0)
        for t in timesteps:
            noise = tracker.get_noise(x_src)
            z_t = (1 - t) * x_src + t * noise
            v_cond, v_uncond, v_cfg = model(z_t, t)
            x0_pred = z_t - t * v_cfg   # 或其他方式计算
            x0_high = z_t - t * v_cond
            x0_low = z_t - t * v_uncond
            tracker.record(x0_pred, t, x0_high, x0_low)
            tracker.update(v_cond, v_uncond, v_cfg, t)  # 更新噪声
    """
    
    # 预测结果
    x0_preds: List[torch.Tensor] = field(default_factory=list)  # MSE 目标
    x0_highs: List[torch.Tensor] = field(default_factory=list)  # CSD 吸引
    x0_lows: List[torch.Tensor] = field(default_factory=list)   # CSD 排斥
    ts: List[float] = field(default_factory=list)               # 时间步
    
    # 尺寸
    height: int = None
    width: int = None
    
    # 可视化
    image: Image.Image = None
    images: List[Image.Image] = field(default_factory=list)
    
    # 噪声管理（BaseNoiseMixin）
    _noise: Optional[torch.Tensor] = None
    _mode: NoiseMode = "fixed"
    
    # =========================================================================
    # 记录方法
    # =========================================================================
    
    def record(
        self, 
        x0_pred: torch.Tensor,
        t: float,
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录一步状态。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG 预测（CSD 吸引）
            x0_low: [B, seq, C] 低 CFG 预测（CSD 排斥）
        """
        self.x0_preds.append(x0_pred.detach().clone())  # [B, seq, C]
        self.x0_highs.append(x0_high.detach().clone())  # [B, seq, C]
        self.x0_lows.append(x0_low.detach().clone())  # [B, seq, C]
        self.ts.append(t)
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent [B, seq, C]，返回最后一个 x0_pred"""
        if not self.x0_preds:
            return None
        return self.x0_preds[-1]  # [B, seq, C]
    
    @property
    def num_steps(self) -> int:
        """返回记录的步数"""
        return len(self.x0_preds)
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 x0_pred [K, B, seq, C]"""
        return torch.stack(self.x0_preds, dim=0)  # [K, B, seq, C]
    
    def __len__(self) -> int:
        return len(self.x0_preds)


# =============================================================================
# InversionStateTracker - Naive Inversion 模式
# =============================================================================

@dataclass
class InversionStateTracker(BaseStateTracker, LossMixin, VisualizationMixin, NaiveInversionMixin):
    """
    Inversion 状态追踪器（inversion_cond / inversion_uncond / inversion_cfg 模式）。
    
    使用 Euler 积分更新噪声：ε_new = ε + (Δt / t) * (v_pred - v_ideal)
    
    存储：
    - x0_preds: 预测的 x0（MSE loss 目标）
    - x0_highs: 高 CFG 预测（CSD 吸引目标）
    - x0_lows: 低 CFG 预测（CSD 排斥目标）
    - ts: 时间步列表
    
    使用方式：
        tracker = InversionStateTracker(height=H, width=W)
        tracker.init(x_src, mode="inversion_cfg", seed=0)
        for t in timesteps:
            noise = tracker.get_noise(x_src)
            z_t = (1 - t) * x_src + t * noise
            v_cond, v_uncond, v_cfg = model(z_t, t)
            x0_pred = z_t - t * v_cfg
            x0_high = z_t - t * v_cond
            x0_low = z_t - t * v_uncond
            tracker.record(x0_pred, t, x0_high, x0_low)
            tracker.update(v_cond, v_uncond, v_cfg, t)  # Naive Inversion 更新
    """
    
    # 预测结果
    x0_preds: List[torch.Tensor] = field(default_factory=list)  # MSE 目标
    x0_highs: List[torch.Tensor] = field(default_factory=list)  # CSD 吸引
    x0_lows: List[torch.Tensor] = field(default_factory=list)   # CSD 排斥
    ts: List[float] = field(default_factory=list)               # 时间步
    
    # 尺寸
    height: int = None
    width: int = None
    
    # 可视化
    image: Image.Image = None
    images: List[Image.Image] = field(default_factory=list)
    
    # NaiveInversionMixin 需要的字段
    _noise: Optional[torch.Tensor] = None
    _mode: NoiseMode = "inversion_cfg"
    _x_src: Optional[torch.Tensor] = None
    _t_prev: float = 0.0
    
    # =========================================================================
    # 记录方法
    # =========================================================================
    
    def record(
        self, 
        x0_pred: torch.Tensor,
        t: float,
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录一步状态。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG 预测（CSD 吸引）
            x0_low: [B, seq, C] 低 CFG 预测（CSD 排斥）
        """
        self.x0_preds.append(x0_pred.detach().clone())  # [B, seq, C]
        self.x0_highs.append(x0_high.detach().clone())  # [B, seq, C]
        self.x0_lows.append(x0_low.detach().clone())  # [B, seq, C]
        self.ts.append(t)
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent [B, seq, C]，返回最后一个 x0_pred"""
        if not self.x0_preds:
            return None
        return self.x0_preds[-1]  # [B, seq, C]
    
    @property
    def num_steps(self) -> int:
        """返回记录的步数"""
        return len(self.x0_preds)
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 x0_pred [K, B, seq, C]"""
        return torch.stack(self.x0_preds, dim=0)  # [K, B, seq, C]
    
    def __len__(self) -> int:
        return len(self.x0_preds)


# =============================================================================
# TrajectoryStateTracker - 轨迹对齐模式
# =============================================================================

@dataclass
class TrajectoryStateTracker(BaseStateTracker, LossMixin, VisualizationMixin, TrajectoryNoiseMixin):
    """
    轨迹对齐状态追踪器（traj_cond / traj_uncond / traj_cfg 模式）。
    
    使用 DNAEdit 风格的轨迹对齐更新噪声：
        v_theoretical = (z_curr - z_prev) / dt
        noise -= (v_theoretical - v_model) * (1 - t)
    
    存储：
    - x0_preds: 预测的 x0（MSE loss 目标）
    - x0_highs: 高 CFG 预测（CSD 吸引目标）
    - x0_lows: 低 CFG 预测（CSD 排斥目标）
    - ts: 时间步列表
    
    使用方式：
        tracker = TrajectoryStateTracker(height=H, width=W)
        tracker.init(x_src, mode="traj_uncond", seed=0)
        for t in timesteps:
            noise = tracker.get_noise(x_src)
            latents_src = (1 - t) * x_src + t * noise
            latents_tgt = z_edit + latents_src - x_src
            v_cond, v_uncond, v_cfg = model(latents_tgt, t)
            x0_pred = z_edit
            x0_high = latents_tgt - t * v_cond
            x0_low = latents_tgt - t * v_uncond
            tracker.record(x0_pred, t, x0_high, x0_low)
            tracker.update(v_cond, v_uncond, v_cfg, t, z_curr=latents_tgt)
    """
    
    # 预测结果
    x0_preds: List[torch.Tensor] = field(default_factory=list)  # MSE 目标
    x0_highs: List[torch.Tensor] = field(default_factory=list)  # CSD 吸引
    x0_lows: List[torch.Tensor] = field(default_factory=list)   # CSD 排斥
    ts: List[float] = field(default_factory=list)               # 时间步
    
    # 尺寸
    height: int = None
    width: int = None
    
    # 可视化
    image: Image.Image = None
    images: List[Image.Image] = field(default_factory=list)
    
    # TrajectoryNoiseMixin 需要的字段
    _noise: Optional[torch.Tensor] = None
    _z_prev: Optional[torch.Tensor] = None
    _t_prev: float = 1.0
    _mode: NoiseMode = "traj_uncond"
    
    # =========================================================================
    # 记录方法
    # =========================================================================
    
    def record(
        self, 
        x0_pred: torch.Tensor,
        t: float,
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录一步状态。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG 预测（CSD 吸引）
            x0_low: [B, seq, C] 低 CFG 预测（CSD 排斥）
        """
        self.x0_preds.append(x0_pred.detach().clone())  # [B, seq, C]
        self.x0_highs.append(x0_high.detach().clone())  # [B, seq, C]
        self.x0_lows.append(x0_low.detach().clone())  # [B, seq, C]
        self.ts.append(t)
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent [B, seq, C]，返回最后一个 x0_pred"""
        if not self.x0_preds:
            return None
        return self.x0_preds[-1]  # [B, seq, C]
    
    @property
    def num_steps(self) -> int:
        """返回记录的步数"""
        return len(self.x0_preds)
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 x0_pred [K, B, seq, C]"""
        return torch.stack(self.x0_preds, dim=0)  # [K, B, seq, C]
    
    def __len__(self) -> int:
        return len(self.x0_preds)


# =============================================================================
# 工厂函数
# =============================================================================

# 类型别名
DistillationTracker = Union[StateTracker, InversionStateTracker]
FlowEditTracker = Union[StateTracker, TrajectoryStateTracker]


def create_distillation_tracker(
    noise_mode: str,
    height: int = None,
    width: int = None,
) -> DistillationTracker:
    """
    工厂函数：根据 noise_mode 创建对应的 Tracker（distillation 用）。
    
    Args:
        noise_mode: 噪声模式
            - "random" / "fixed" / "aligned" → StateTracker
            - "inversion_cond" / "inversion_uncond" / "inversion_cfg" → InversionStateTracker
        height: 图像高度
        width: 图像宽度
    
    Returns:
        StateTracker 或 InversionStateTracker
    """
    if noise_mode.startswith("inversion"):
        return InversionStateTracker(height=height, width=width)
    else:
        return StateTracker(height=height, width=width)


def create_flowedit_tracker(
    noise_mode: str,
    height: int = None,
    width: int = None,
) -> FlowEditTracker:
    """
    工厂函数：根据 noise_mode 创建对应的 Tracker（flowedit 用）。
    
    Args:
        noise_mode: 噪声模式
            - "random" / "fixed" / "aligned" → StateTracker
            - "traj_cond" / "traj_uncond" / "traj_cfg" → TrajectoryStateTracker
        height: 图像高度
        width: 图像宽度
    
    Returns:
        StateTracker 或 TrajectoryStateTracker
    """
    if noise_mode.startswith("traj"):
        return TrajectoryStateTracker(height=height, width=width)
    else:
        return StateTracker(height=height, width=width)
