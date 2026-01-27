"""
噪声管理 Mixin。

为 Tracker 提供噪声管理能力。
支持多种噪声模式：random, fixed, aligned_cond, aligned_uncond, aligned_cfg
"""

from typing import Optional, Literal
import torch


NoiseMode = Literal["random", "fixed", "aligned_cond", "aligned_uncond", "aligned_cfg"]


class NoiseMixin:
    """
    噪声管理 Mixin。
    
    为 FlowEditStateTracker 和 ContrastStateTracker 提供统一的噪声管理能力。
    
    支持的噪声模式：
    - random: 每步随机采样
    - fixed: 固定噪声（初始化后不变）
    - aligned_cond: 从条件预测对齐
    - aligned_uncond: 从无条件预测对齐
    - aligned_cfg: 从 CFG 组合预测对齐
    
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
        z_tgt: torch.Tensor,
        v_cond: torch.Tensor,
        v_uncond: Optional[torch.Tensor],
        v_cfg: torch.Tensor,
        t: float,
    ) -> None:
        """
        更新噪声（仅 aligned 模式生效）。
        
        公式：noise = z_tgt + (1 - t) * v
        
        Args:
            z_tgt: [B, seq, C] target 中间状态
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度（可选）
            v_cfg: [B, seq, C] CFG 组合速度
            t: 当前时间步
        """
        if not self._noise_mode.startswith("aligned"):
            return
        
        # 选择速度
        if self._noise_mode in ("aligned", "aligned_cfg"):
            v = v_cfg
        elif self._noise_mode == "aligned_cond":
            v = v_cond
        elif self._noise_mode == "aligned_uncond":
            v = v_uncond if v_uncond is not None else v_cond
        else:
            return
        
        # 反推噪声: noise = z_tgt + (1 - t) * v
        self._noise = z_tgt + (1 - t) * v  # [B, seq, C]
    
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
