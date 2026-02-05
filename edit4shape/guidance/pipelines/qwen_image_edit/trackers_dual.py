"""
双分支状态追踪器。

DualBranchTracker：src 和 tgt 两个子 tracker 分别记录状态，
根据 update_mode 决定用哪个分支的速度更新共享噪声（aligned 模式）。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import torch
from PIL import Image

from .trackers import StateTracker
from ..utils import LossMixin, VisualizationMixin


# 噪声更新模式
# - "src": 用 src 分支的速度更新噪声 (aligned)
# - "tgt": 用 tgt 分支的速度更新噪声 (aligned)
# - "avg": 用两个分支速度的平均值更新噪声 (aligned)
# - "fixed": 不更新噪声（固定噪声）
# - "random": 每步重新采样噪声
NoiseUpdateMode = Literal["src", "tgt", "avg", "fixed", "random"]


@dataclass
class DualBranchTracker(LossMixin, VisualizationMixin):
    """
    双分支 Tracker（aligned 模式，共享噪声）。
    
    src 和 tgt 两个子 tracker 分别记录各自分支的 x0 预测，
    根据 update_mode 决定如何更新共享噪声。
    
    update_mode 选项:
    - "src": 用 src 分支的速度更新噪声 (aligned)
    - "tgt": 用 tgt 分支的速度更新噪声 (aligned, 默认)
    - "avg": 用两个分支速度的平均值更新噪声 (aligned)
    - "fixed": 不更新噪声（固定噪声）
    - "random": 每步重新采样噪声
    
    使用方式：
        tracker = DualBranchTracker(update_mode="tgt", height=H, width=W)
        tracker.init(x_src)
        
        for t in timesteps:
            noise = tracker.get_noise()
            # ... 计算两个分支的速度 ...
            
            tracker.update_src(v_cond_src, v_uncond_src, v_cfg_src, t)
            tracker.update_tgt(v_cond_tgt, v_uncond_tgt, v_cfg_tgt, t)
            tracker.step()  # 根据 update_mode 更新 noise
            
            tracker.record_src(x_src, t, x0_high_src, x0_low_src)
            tracker.record_tgt(z_edit, t, x0_high_tgt, x0_low_tgt)
        
        # 访问聚合数据
        all_x0_preds = tracker.x0_preds  # src + tgt
        loss = tracker.loss(src_latent, csd_weight=1.0)
    """
    
    update_mode: NoiseUpdateMode = "tgt"
    height: int = None
    width: int = None
    
    # 两个子 tracker
    src: StateTracker = field(default=None)
    tgt: StateTracker = field(default=None)
    
    # 共享噪声
    _noise: torch.Tensor = None
    
    # 速度缓存
    _v_src: Dict[str, torch.Tensor] = None
    _v_tgt: Dict[str, torch.Tensor] = None
    
    # 可视化
    images: List[Image.Image] = field(default_factory=list)
    
    def __post_init__(self):
        self.src = StateTracker(height=self.height, width=self.width)
        self.tgt = StateTracker(height=self.height, width=self.width)
    
    def init(self, x_src: torch.Tensor, seed: int = None) -> torch.Tensor:
        """
        初始化共享噪声。
        
        Args:
            x_src: [B, seq, C] 源 latent
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        if seed is not None:
            gen = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn(
                x_src.shape, generator=gen, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
        else:
            self._noise = torch.randn_like(x_src)  # [B, seq, C]
        return self._noise
    
    def get_noise(self) -> torch.Tensor:
        """获取当前共享噪声 [B, seq, C]。"""
        return self._noise
    
    def update_src(
        self, 
        v_cond: torch.Tensor, 
        v_uncond: torch.Tensor, 
        v_cfg: torch.Tensor, 
        t: float,
    ) -> None:
        """
        缓存 src 分支的速度。
        
        Args:
            v_cond: [B, seq, C] src 条件速度
            v_uncond: [B, seq, C] src 无条件速度
            v_cfg: [B, seq, C] src CFG 速度
            t: 当前时间步 [0, 1]
        """
        self._v_src = {"v_cond": v_cond, "v_uncond": v_uncond, "t": t}
    
    def update_tgt(
        self, 
        v_cond: torch.Tensor, 
        v_uncond: torch.Tensor, 
        v_cfg: torch.Tensor, 
        t: float,
    ) -> None:
        """
        缓存 tgt 分支的速度。
        
        Args:
            v_cond: [B, seq, C] tgt 条件速度
            v_uncond: [B, seq, C] tgt 无条件速度
            v_cfg: [B, seq, C] tgt CFG 速度
            t: 当前时间步 [0, 1]
        """
        self._v_tgt = {"v_cond": v_cond, "v_uncond": v_uncond, "t": t}
    
    def step(self) -> None:
        """
        根据 update_mode 更新共享噪声。
        
        必须在 update_src 和 update_tgt 都调用后调用。
        aligned 更新公式：ε -= (v_cond - v_uncond) * (1 - t)
        
        update_mode:
        - "src": 用 src 分支的速度 (aligned)
        - "tgt": 用 tgt 分支的速度 (aligned)
        - "avg": 用两个分支速度的平均值 (aligned)
        - "fixed": 不更新噪声
        - "random": 每步重新采样噪声
        """
        # fixed 模式：不更新噪声
        if self.update_mode == "fixed":
            self._v_src = None
            self._v_tgt = None
            return
        
        # random 模式：每步重新采样噪声
        if self.update_mode == "random":
            self._noise = torch.randn_like(self._noise)  # [B, seq, C] 随机噪声
            self._v_src = None
            self._v_tgt = None
            return
        
        # aligned 模式：选择用哪个分支的速度
        if self.update_mode == "src":
            v_cond = self._v_src["v_cond"]      # [B, seq, C]
            v_uncond = self._v_src["v_uncond"]  # [B, seq, C]
        elif self.update_mode == "tgt":
            v_cond = self._v_tgt["v_cond"]      # [B, seq, C]
            v_uncond = self._v_tgt["v_uncond"]  # [B, seq, C]
        elif self.update_mode == "avg":
            v_cond = 0.5 * (self._v_src["v_cond"] + self._v_tgt["v_cond"])      # [B, seq, C]
            v_uncond = 0.5 * (self._v_src["v_uncond"] + self._v_tgt["v_uncond"])  # [B, seq, C]
        
        t = self._v_tgt["t"]
        
        # aligned 更新：ε -= (v_cond - v_uncond) * (1 - t)
        v_delta = v_cond - v_uncond  # [B, seq, C]
        
        self._noise = self._noise.to(torch.float32)  # [B, seq, C]
        self._noise -= v_delta.to(torch.float32) * (1.0 - t)  # [B, seq, C]
        self._noise = self._noise.to(v_delta.dtype)  # [B, seq, C]
        
        # 清空缓存
        self._v_src = None
        self._v_tgt = None
    
    def record_src(
        self, 
        x0_pred: torch.Tensor, 
        t: float, 
        x0_high: torch.Tensor, 
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录 src 分支的 x0 预测。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG 预测（CSD 吸引）
            x0_low: [B, seq, C] 低 CFG 预测（CSD 排斥）
        """
        self.src.record(x0_pred, t, x0_high, x0_low)
    
    def record_tgt(
        self, 
        x0_pred: torch.Tensor, 
        t: float, 
        x0_high: torch.Tensor, 
        x0_low: torch.Tensor,
    ) -> None:
        """
        记录 tgt 分支的 x0 预测。
        
        Args:
            x0_pred: [B, seq, C] 预测的 x0（MSE 目标）
            t: 当前时间步
            x0_high: [B, seq, C] 高 CFG 预测（CSD 吸引）
            x0_low: [B, seq, C] 低 CFG 预测（CSD 排斥）
        """
        self.tgt.record(x0_pred, t, x0_high, x0_low)
    
    # =========================================================================
    # 聚合属性
    # =========================================================================
    
    @property
    def x0_preds(self) -> List[torch.Tensor]:
        """两个分支的 x0_preds 聚合 [src + tgt]"""
        return self.src.x0_preds + self.tgt.x0_preds
    
    @property
    def x0_highs(self) -> List[torch.Tensor]:
        """两个分支的 x0_highs 聚合 [src + tgt]"""
        return self.src.x0_highs + self.tgt.x0_highs
    
    @property
    def x0_lows(self) -> List[torch.Tensor]:
        """两个分支的 x0_lows 聚合 [src + tgt]"""
        return self.src.x0_lows + self.tgt.x0_lows
    
    @property
    def ts(self) -> List[float]:
        """两个分支的 ts 聚合 [src + tgt]"""
        return self.src.ts + self.tgt.ts
    
    @property
    def noise(self) -> torch.Tensor:
        """当前共享噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def target(self) -> torch.Tensor:
        """最终目标 = tgt 的最后一个 x0_pred"""
        return self.tgt.target
    
    @property
    def num_steps(self) -> int:
        """总步数 = src + tgt"""
        return len(self.src.x0_preds) + len(self.tgt.x0_preds)
    
    def __len__(self) -> int:
        return self.num_steps
    
    # =========================================================================
    # Loss 计算（覆盖 LossMixin.loss，分别计算两个子 tracker 再相加）
    # =========================================================================
    
    def loss(self, src: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算两个分支 loss 之和。
        
        注意：这里覆盖 LossMixin.loss()，因为需要分别对两个子 tracker 计算 loss 再相加，
        而不是用聚合的 x0_preds 计算（数学上不等价）。
        
        Args:
            src: [B, seq, C] 有梯度的源 latent
            **kwargs: 传递给子 tracker 的 loss() 方法
        
        Returns:
            两个分支 loss 之和
        """
        loss_src = self.src.loss(src, **kwargs) if self.src.x0_preds else torch.tensor(0.0, device=src.device)  # []
        loss_tgt = self.tgt.loss(src, **kwargs) if self.tgt.x0_preds else torch.tensor(0.0, device=src.device)  # []
        return loss_src + loss_tgt  # []
