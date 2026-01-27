"""
噪声管理 Mixin（累积补偿模式）。

为 Tracker 提供噪声管理能力。
支持多种噪声模式：random, fixed, aligned_cond, aligned_uncond, aligned_cfg
"""

from typing import Optional, Literal
import torch


NoiseMode = Literal["random", "fixed", "aligned_cond", "aligned_uncond", "aligned_cfg"]


class NoiseMixin:
    """
    噪声管理 Mixin（累积补偿模式）。
    
    为 FlowEditStateTracker 和 ContrastStateTracker 提供统一的噪声管理能力。
    
    使用 DNAEdit 风格的累积更新策略：
    1. 根据 noise_mode 选择目标速度 v_tgt
    2. 计算速度偏差 v_delta = v_tgt - v_src
    3. 累积更新 noise -= v_delta * (1 - t)
    
    支持的噪声模式：
    - random: 每步随机采样
    - fixed: 固定噪声（初始化后不变）
    - aligned_cond: 累积 v_cond - v_src
    - aligned_uncond: 累积 v_uncond - v_src
    - aligned_cfg: 累积 v_cfg - v_src
    
    使用方法：
        @dataclass
        class MyTracker(BaseStateTracker, NoiseMixin):
            _noise: Optional[torch.Tensor] = None
            _noise_mode: NoiseMode = "fixed"
    """
    
    # 子类需要定义这些字段
    _noise: Optional[torch.Tensor]
    _noise_mode: NoiseMode
    
    def init_noise(
        self, 
        x_src: torch.Tensor, 
        mode: NoiseMode = "fixed",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化噪声。
        
        Args:
            x_src: [B, seq, C] 参考 tensor
            mode: 噪声模式
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._noise_mode = mode
        
        if seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn_like(x_src, generator=generator)  # [B, seq, C]
        else:
            self._noise = torch.randn_like(x_src)  # [B, seq, C]
        
        return self._noise
    
    def get_noise(self, x_src: torch.Tensor) -> torch.Tensor:
        """
        获取当前噪声。
        
        Args:
            x_src: [B, seq, C] 参考 tensor
        
        Returns:
            [B, seq, C] 噪声
        """
        if self._noise is None:
            raise RuntimeError("请先调用 init_noise() 初始化噪声")
        
        if self._noise_mode == "random":
            return torch.randn_like(x_src)  # [B, seq, C]
        return self._noise  # [B, seq, C]
    
    def update_noise(
        self,
        v_src: torch.Tensor,     # 源速度
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 时间步 [0, 1]
    ) -> None:
        """
        累积更新噪声。
        
        公式：noise -= (v_tgt - v_src) * (1 - t)
        
        数学推导：
        1. RF 插值：z_t = (1-t)·x0 + t·ε
        2. 速度场：v = ε - x0
        3. x0 变化导致：Δε = -(1-t)·Δv
        
        Args:
            v_src: [B, seq, C] 源速度（如 v_uncond）
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度
            v_cfg: [B, seq, C] CFG 速度
            t: 当前时间步 [0, 1]（已归一化）
        """
        if not self._noise_mode.startswith("aligned"):
            return
        
        # 根据模式选择目标速度
        v_tgt = {
            "aligned_cfg": v_cfg,
            "aligned_cond": v_cond,
            "aligned_uncond": v_uncond,
        }.get(self._noise_mode, v_cfg)
        
        # 计算速度偏差并累积更新
        v_delta = v_tgt - v_src  # [B, seq, C]
        self._noise = self._noise.to(torch.float32)
        self._noise -= v_delta.to(torch.float32) * (1.0 - t)  # 核心公式
        self._noise = self._noise.to(v_delta.dtype)
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """当前噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def noise_mode(self) -> NoiseMode:
        """当前噪声模式"""
        return self._noise_mode
    
    @property
    def is_aligned(self) -> bool:
        """是否为对齐模式"""
        return self._noise_mode.startswith("aligned")
