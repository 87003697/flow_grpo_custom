"""
噪声管理 Mixin。

包含:
- NoiseMixin: 累积补偿模式（aligned_* 模式）
- NoiseInversionMixin: DNAEdit 风格的 Noise Inversion
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
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 时间步 [0, 1]
    ) -> None:
        """
        累积更新噪声。
        
        公式：noise -= (v_tgt - v_uncond) * (1 - t)
        
        数学推导：
        1. RF 插值：z_t = (1-t)·x0 + t·ε
        2. 速度场：v = ε - x0
        3. x0 变化导致：Δε = -(1-t)·Δv
        
        Args:
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
        
        # 计算速度偏差并累积更新（以 v_uncond 为基准）
        v_delta = v_tgt - v_uncond  # [B, seq, C]
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


class NoiseInversionMixin:
    """
    DNAEdit 风格的 Noise Inversion Mixin（精简版）。
    
    命名与 NoiseMixin 对齐，使用 _inv 后缀区分。
    内部维护 _t_prev，简化外部调用（只需传当前 t）。
    
    核心逻辑：
    1. 维护 _z_curr（当前 latent 位置，从 x_src 开始）
    2. 维护 _t_prev（上一时间步，从 0 开始）
    3. 用插值公式预测当前时间步的 noisy latent
    4. 计算 delta_v = 理论速度 - 模型预测速度
    5. 同时更新 _z_curr、_noise_inv 和 _t_prev
    
    使用方法：
        tracker.init_noise_inv(x_src, noise)
        for t in timesteps:  # 从小到大
            latents_noisy = tracker.get_noise_inv(t)
            v_pred = model(latents_noisy, t)
            tracker.update_noise_inv(latents_noisy, v_pred, t)
        inverted_noise = tracker.noise_inv
    """
    
    # 子类需要定义这些字段
    _noise_inv: Optional[torch.Tensor]  # 噪声（会被 inversion 更新）
    _z_prev: Optional[torch.Tensor]     # 上一步的 latent 位置
    _t_prev: float                       # 上一时间步
    
    def init_noise_inv(
        self, 
        x_src: torch.Tensor, 
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化 noise inversion。
        
        Args:
            x_src: [B, seq, C] 源 latent（干净图像）
            noise: [B, seq, C] 初始噪声（可选）
            seed: 随机种子（可选）
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._z_prev = x_src.clone()
        self._t_prev = 0.0  # 从 t=0 开始
        
        if noise is not None:
            self._noise_inv = noise.clone()
        elif seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise_inv = torch.randn_like(x_src, generator=generator)  # [B, seq, C]
        else:
            self._noise_inv = torch.randn_like(x_src)  # [B, seq, C]
        
        return self._noise_inv
    
    def get_noise_inv(self, t: float) -> torch.Tensor:
        """
        获取当前时间步的 noisy latent。
        
        公式: latents_noisy = (t - t_prev) / (1 - t_prev) * (noise - z_curr) + z_curr
        
        Args:
            t: 当前时间步 [0, 1]
        
        Returns:
            [B, seq, C] noisy latent（插值结果）
        """
        if self._z_prev is None or self._noise_inv is None:
            raise RuntimeError("请先调用 init_noise_inv() 初始化")
        
        ratio = (t - self._t_prev) / (1.0 - self._t_prev + 1e-8)  # 防止除零
        z_t = ratio * (self._noise_inv - self._z_prev) + self._z_prev  # [B, seq, C]
        return z_t
    
    def update_noise_inv(
        self,
        z_t: torch.Tensor,    # 当前步的 noisy latent
        v_pred: torch.Tensor, # 模型预测的速度
        t: float,             # 当前时间步 [0, 1]
    ) -> None:
        """
        执行一步 noise inversion：更新 _z_prev、_noise_inv 和 _t_prev。
        
        公式:
            dt = t - t_prev
            delta_v = (z_t - z_prev) / dt - v_pred
            z_prev = z_t - delta_v * dt
            noise -= delta_v * (1 - t_prev)
            t_prev = t
        
        Args:
            z_t: [B, seq, C] 当前 noisy latent
            v_pred: [B, seq, C] 模型预测的速度
            t: 当前时间步 [0, 1]
        """
        dt = t - self._t_prev
        if abs(dt) < 1e-8:
            return
        
        # 计算理论速度 vs 模型预测的差异
        v_theoretical = (z_t - self._z_prev) / dt  # [B, seq, C]
        delta_v = v_theoretical - v_pred  # [B, seq, C]
        dx = delta_v * dt  # [B, seq, C]
        
        # 更新 _z_prev
        self._z_prev = self._z_prev.to(torch.float32)
        self._z_prev = z_t - dx
        self._z_prev = self._z_prev.to(v_pred.dtype)
        
        # 更新 _noise_inv
        self._noise_inv = self._noise_inv.to(torch.float32)
        self._noise_inv -= delta_v.to(torch.float32) * (1.0 - self._t_prev)  # 核心公式
        self._noise_inv = self._noise_inv.to(v_pred.dtype)
        
        # 更新 _t_prev
        self._t_prev = t
    
    @property
    def noise_inv(self) -> Optional[torch.Tensor]:
        """Inversion 后的噪声 [B, seq, C]"""
        return self._noise_inv
    
    @property
    def z_prev(self) -> Optional[torch.Tensor]:
        """上一步的 latent 位置 [B, seq, C]"""
        return self._z_prev
    
    @property
    def t_prev(self) -> float:
        """上一时间步"""
        return self._t_prev
