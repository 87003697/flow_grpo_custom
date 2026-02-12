"""
噪声管理 Mixin。

包含:
- BaseNoiseMixin: 噪声管理基类（random / fixed / aligned 模式）
- NaiveInversionMixin: Naive Inversion（inversion_cond / inversion_uncond / inversion_cfg 模式）
- TrajectoryNoiseMixin: 轨迹对齐（traj_cond / traj_uncond / traj_cfg 模式）
- NoiseInversionMixin: DNAEdit 风格的 Noise Inversion（保留但不推荐使用）
"""

from typing import Optional, Literal
import torch


# =============================================================================
# 噪声模式类型
# =============================================================================

NoiseMode = Literal[
    # BaseNoiseMixin 支持的模式
    "random",   # 每步随机采样
    "fixed",    # 固定噪声（初始化后不变）
    "aligned",  # DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    "delta",    # 双分支差分补偿 ε -= v_delta * dt，需要传入 v_delta
    # NaiveInversionMixin 支持的模式
    "inversion_cond",   # Naive Inversion（用 v_cond）
    "inversion_uncond", # Naive Inversion（用 v_uncond）
    "inversion_cfg",    # Naive Inversion（用 v_cfg）
    # TrajectoryNoiseMixin 支持的模式
    "traj_cond",   # 轨迹对齐（用 v_cond）
    "traj_uncond", # 轨迹对齐（用 v_uncond）
    "traj_cfg",    # 轨迹对齐（用 v_cfg）
]


# =============================================================================
# BaseNoiseMixin - 噪声管理基类
# =============================================================================

class BaseNoiseMixin:
    """
    噪声管理基类。
    
    支持的噪声模式：
    - random: 每步随机采样
    - fixed: 固定噪声（初始化后不变）
    - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    - delta: 双分支差分补偿 ε -= v_delta * dt，update() 时需传入 v_delta
    
    使用方法：
        @dataclass
        class MyTracker(BaseStateTracker, BaseNoiseMixin):
            _noise: Optional[torch.Tensor] = None
            _mode: NoiseMode = "fixed"
    """
    
    # 子类需要定义这些字段
    _noise: Optional[torch.Tensor]
    _mode: NoiseMode
    
    def init(
        self, 
        x_src: torch.Tensor, 
        mode: NoiseMode = "fixed",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化噪声。
        
        Args:
            x_src: [B, seq, C] 源 latent
            mode: 噪声模式
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._mode = mode
        
        if seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn(
                x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
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
            raise RuntimeError("请先调用 init() 初始化噪声")
        
        if self._mode == "random":
            return torch.randn_like(x_src)  # [B, seq, C]
        return self._noise  # [B, seq, C]
    
    def update(
        self,
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 时间步 [0, 1]
        dt: float,               # 时间步差（必传）
        v_delta: Optional[torch.Tensor] = None,  # delta 模式必传：v_cfg_tgt - v_cfg_src
        **kwargs,                # 忽略其他参数（兼容 TrajectoryNoiseMixin）
    ) -> None:
        """
        更新噪声。
        
        aligned 模式：ε -= (v_cond - v_uncond) * (1 - t)
        delta 模式：ε -= v_delta * dt，保证 z_t_tgt = (1-t)*z_edit + t*ε₀
        
        Args:
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度
            v_cfg: [B, seq, C] CFG 速度
            t: 当前时间步 [0, 1]
            dt: 时间步差（t_prev - t_curr 或 t_curr - t_prev，取决于调用方向）
            v_delta: [B, seq, C] 差分速度（delta 模式必传）
            **kwargs: 忽略其他参数
        """
        if self._mode == "aligned":
            # DNAEdit 风格累积补偿
            v_delta = v_cond - v_uncond  # [B, seq, C]
            self._noise = self._noise.to(torch.float32)
            self._noise -= v_delta.to(torch.float32) * (1.0 - t)
            self._noise = self._noise.to(v_delta.dtype)
        
        elif self._mode == "delta":
            # 双分支差分补偿：ε -= v_delta * dt
            if v_delta is None:
                raise ValueError("delta 模式必须传入 v_delta（v_cfg_tgt - v_cfg_src）")
            self._noise = self._noise.to(torch.float32)  # [B, seq, C]
            self._noise -= v_delta.to(torch.float32) * dt  # [B, seq, C]
            self._noise = self._noise.to(v_delta.dtype)  # [B, seq, C]
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """当前噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def mode(self) -> NoiseMode:
        """当前噪声模式"""
        return self._mode


# =============================================================================
# TrajectoryNoiseMixin - 轨迹对齐（DNAEdit 风格，单向 denoising）
# =============================================================================

class TrajectoryNoiseMixin:
    """
    轨迹对齐 Mixin（DNAEdit 风格，单向 denoising）。
    
    核心公式（简化版）：
        目标轨迹：latents_tgt = z_edit + t*(noise - x_src)
        理论速度：v_theoretical = noise - x_src
        delta_v = v_theoretical - v_model
        noise -= delta_v * t
    
    支持三种 v_model：
    - traj_cond: 用 v_cond
    - traj_uncond: 用 v_uncond
    - traj_cfg: 用 v_cfg
    
    使用方法：
        @dataclass
        class MyTracker(BaseStateTracker, TrajectoryNoiseMixin):
            _noise: Optional[torch.Tensor] = None
            _x_src: Optional[torch.Tensor] = None
            _mode: NoiseMode = "traj_uncond"
    """
    
    # 子类需要定义这些字段
    _noise: Optional[torch.Tensor]
    _x_src: Optional[torch.Tensor]
    _mode: NoiseMode
    
    def init(
        self, 
        x_src: torch.Tensor, 
        mode: NoiseMode = "traj_uncond",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化轨迹对齐。
        
        Args:
            x_src: [B, seq, C] 源 latent
            mode: traj_cond / traj_uncond / traj_cfg
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._mode = mode
        self._x_src = x_src  # [B, seq, C] 保存用于计算理论速度
        
        if seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn(
                x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
        else:
            self._noise = torch.randn_like(x_src)  # [B, seq, C]
        
        return self._noise
    
    def get_noise(self, x_src: torch.Tensor = None) -> torch.Tensor:
        """
        获取当前噪声。
        
        Args:
            x_src: [B, seq, C] 参考 tensor（未使用）
        
        Returns:
            [B, seq, C] 噪声
        """
        if self._noise is None:
            raise RuntimeError("请先调用 init() 初始化")
        return self._noise  # [B, seq, C]
    
    def update(
        self,
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 当前时间步 [0, 1]
        dt: float,               # 时间步差（必传，本模式不使用，接口一致）
        **kwargs,                # 忽略其他参数（如 z_curr）
    ) -> None:
        """
        更新噪声（简化版）。
        
        公式：
            v_theoretical = noise - x_src（目标轨迹的理论速度）
            delta_v = v_theoretical - v_model
            noise -= delta_v * t
        
        Args:
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度
            v_cfg: [B, seq, C] CFG 速度
            t: 当前时间步 [0, 1]
            dt: 时间步差（本模式不使用，接口一致）
        """
        # 选择 v_model
        v_model = {
            "traj_cond": v_cond,
            "traj_uncond": v_uncond,
            "traj_cfg": v_cfg,
        }[self._mode]  # [B, seq, C]
        
        # 理论速度（直接从 noise 和 x_src 算）
        v_theoretical = self._noise - self._x_src  # [B, seq, C]
        
        # 速度差异
        delta_v = v_theoretical - v_model  # [B, seq, C]
        
        # 更新噪声（权重为 t）
        self._noise = self._noise.to(torch.float32)  # [B, seq, C]
        self._noise -= delta_v.to(torch.float32) * t  # [B, seq, C]
        self._noise = self._noise.to(v_model.dtype)  # [B, seq, C]
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """当前噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def mode(self) -> NoiseMode:
        """当前噪声模式"""
        return self._mode


# =============================================================================
# NaiveInversionMixin - Naive Inversion（Euler 积分）
# =============================================================================

class NaiveInversionMixin:
    """
    Naive Inversion Mixin（Euler 积分更新噪声）。
    
    核心公式：
        v_ideal = ε - x_src
        ε_new = ε + (Δt / t) * (v_pred - v_ideal)
    
    支持三种速度选择：
    - inversion_cond: 用 v_cond 作为 v_pred
    - inversion_uncond: 用 v_uncond 作为 v_pred
    - inversion_cfg: 用 v_cfg 作为 v_pred
    
    使用方法：
        @dataclass
        class MyTracker(BaseStateTracker, NaiveInversionMixin):
            _noise: Optional[torch.Tensor] = None
            _mode: NoiseMode = "inversion_cfg"
            _x_src: Optional[torch.Tensor] = None
            _t_prev: float = 0.0
    """
    
    # 子类需要定义这些字段
    _noise: Optional[torch.Tensor]
    _mode: NoiseMode
    _x_src: Optional[torch.Tensor]
    _t_prev: float
    
    def init(
        self, 
        x_src: torch.Tensor, 
        mode: NoiseMode = "inversion_cfg",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        初始化 Naive Inversion。
        
        Args:
            x_src: [B, seq, C] 源 latent
            mode: inversion_cond / inversion_uncond / inversion_cfg
            seed: 随机种子
        
        Returns:
            [B, seq, C] 初始噪声
        """
        self._mode = mode
        self._x_src = x_src.clone()
        self._t_prev = 0.0
        
        if seed is not None:
            generator = torch.Generator(device=x_src.device).manual_seed(seed)
            self._noise = torch.randn(
                x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
        else:
            self._noise = torch.randn_like(x_src)  # [B, seq, C]
        
        return self._noise
    
    def get_noise(self, x_src: torch.Tensor) -> torch.Tensor:
        """
        获取当前噪声。
        
        Args:
            x_src: [B, seq, C] 参考 tensor（未使用）
        
        Returns:
            [B, seq, C] 噪声
        """
        if self._noise is None:
            raise RuntimeError("请先调用 init() 初始化")
        return self._noise  # [B, seq, C]
    
    def update(
        self,
        v_cond: torch.Tensor,    # 条件速度
        v_uncond: torch.Tensor,  # 无条件速度
        v_cfg: torch.Tensor,     # CFG 速度
        t: float,                # 当前时间步 [0, 1]
        dt: float,               # 时间步差（必传）
        **kwargs,                # 忽略其他参数（接口一致）
    ) -> None:
        """
        Naive Inversion 更新。
        
        公式：ε_new = ε + (dt / t) * (v_pred - v_ideal)
        其中 v_ideal = ε - x_src
        
        Args:
            v_cond: [B, seq, C] 条件速度
            v_uncond: [B, seq, C] 无条件速度
            v_cfg: [B, seq, C] CFG 速度
            t: 当前时间步 [0, 1]
            dt: 时间步差
            **kwargs: 忽略其他参数
        """
        if abs(dt) < 1e-8 or t < 1e-8:
            return
        
        # 根据模式选择 v_pred
        v_pred = {
            "inversion_cond": v_cond,
            "inversion_uncond": v_uncond,
            "inversion_cfg": v_cfg,
        }.get(self._mode, v_cfg)  # [B, seq, C]
        
        # 理想速度场：v_ideal = ε - x_src
        v_ideal = self._noise - self._x_src  # [B, seq, C]
        
        # 速度差
        v_delta = v_pred - v_ideal  # [B, seq, C]
        
        # 更新噪声：ε_new = ε + (dt / t) * v_delta
        self._noise = self._noise.to(torch.float32)
        self._noise += (dt / t) * v_delta.to(torch.float32)
        self._noise = self._noise.to(v_pred.dtype)
    
    @property
    def noise(self) -> Optional[torch.Tensor]:
        """当前噪声 [B, seq, C]"""
        return self._noise
    
    @property
    def mode(self) -> NoiseMode:
        """当前噪声模式"""
        return self._mode


# =============================================================================
# NoiseInversionMixin - DNAEdit 风格（保留但不推荐）
# =============================================================================

class NoiseInversionMixin:
    """
    DNAEdit 风格的 Noise Inversion Mixin（精简版）。
    
    命名与 BaseNoiseMixin 对齐，使用 _inv 后缀区分。
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
            self._noise_inv = torch.randn(
                x_src.shape, generator=generator, device=x_src.device, dtype=x_src.dtype
            )  # [B, seq, C]
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
