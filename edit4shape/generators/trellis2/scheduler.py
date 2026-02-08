# =====================================================================
# Imports
# =====================================================================
from types import SimpleNamespace
from typing import List

import numpy as np
import torch

from trellis2.modules.sparse import SparseTensor

# =====================================================================
# FlowEuler Scheduler（独立类）
# =====================================================================
class FlowEulerScheduler:
    """
    基于 FlowEuler 公式的 Scheduler。
    
    提供 set_timesteps() 和 step() 方法，用于去噪采样。
    
    重要：使用 numpy float64 计算时间步序列，与参考实现保持完全一致。
    这是必要的，因为扩散模型对输入高度敏感，微小的时间步差异会被放大。
    
    使用方法：
        scheduler.set_timesteps(steps, device)
        for idx, t in enumerate(scheduler.get_timesteps_for_loop()):
            t_val = scheduler.get_precise_t(idx)  # 使用精确的 float64 值
            ...
            scheduler.step_by_index(velocity, idx, latents)  # 使用索引而非 t 值
    """
    
    def __init__(self, rescale_t: float = 1.0):
        self.timesteps: torch.Tensor = torch.tensor([])
        self._timesteps_np: np.ndarray = np.array([])  # numpy float64 版本用于精确计算
        self.rescale_t = rescale_t
    
    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        """
        设置时间步序列。
        
        使用 numpy float64 计算（对齐参考实现），然后转换为 torch tensor。
        
        Args:
            num_steps: 采样步数
            device: 目标设备
        
        timesteps: 递减序列 [1.0, ..., 0.0]，长度 num_steps + 1
        """
        # 使用 numpy float64 计算（对齐参考实现 flow_euler.py）
        t_seq = np.linspace(1, 0, num_steps + 1)  # float64
        t_seq = self.rescale_t * t_seq / (1 + (self.rescale_t - 1) * t_seq)  # float64
        
        # 保存 numpy 版本用于精确的 delta 计算
        self._timesteps_np = t_seq
        self._device = device
        
        # 转换为 torch tensor（保持 float64 精度再转到目标设备）
        self.timesteps = torch.from_numpy(t_seq).to(device=device, dtype=torch.float32)  # (steps+1,)
    
    def get_timesteps_for_loop(self) -> List[int]:
        """
        获取用于循环的索引列表（排除最后一个 t=0）。
        
        Returns:
            list[int]: 索引列表 [0, 1, ..., num_steps-1]
        """
        return list(range(len(self._timesteps_np) - 1))
    
    def get_precise_t(self, idx: int) -> float:
        """
        获取精确的时间步值（numpy float64 精度）。
        
        用于传递给模型计算 t_scaled，确保与参考实现完全一致。
        
        Args:
            idx: 时间步索引
        
        Returns:
            float: 精确的时间步值
        """
        return float(self._timesteps_np[idx])
    
    def step_by_index(
        self,
        velocity: SparseTensor,
        idx: int,
        latents: SparseTensor,
    ) -> SimpleNamespace:
        """
        基于索引的 Euler 步进（推荐使用）。
        
        直接使用索引而非时间步值，避免精度损失。
        
        Args:
            velocity: SparseTensor，velocity 预测
            idx: 时间步索引
            latents: SparseTensor，当前 latent
        
        Returns:
            SimpleNamespace: 包含 prev_sample
        """
        assert idx + 1 < len(self._timesteps_np), f"idx={idx} 无后继步"
        
        # 使用 numpy float64 计算 delta（精度对齐参考实现）
        t_np = self._timesteps_np[idx]
        t_prev_np = self._timesteps_np[idx + 1]
        delta = float(t_np - t_prev_np)  # float64 计算后转为 Python float
        
        # Euler 步进（使用 SparseTensor 运算保留 _spatial_cache，对齐参考实现）
        # 参考实现: pred_x_prev = x_t - (t - t_prev) * pred_v
        prev_sample = latents - delta * velocity  # SparseTensor 运算，保留 _spatial_cache
        
        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=None)
    
    def step(
        self,
        velocity: SparseTensor,
        t: torch.Tensor,
        latents: SparseTensor,
    ) -> SimpleNamespace:
        """
        Euler 步进：x_{t-1} = x_t - (t - t_prev) * v
        
        使用 numpy float64 时间步计算 delta，确保与参考实现完全一致。
        
        注意：推荐使用 step_by_index() 避免精度损失。
        
        Args:
            velocity: SparseTensor，velocity 预测
            t: 当前时间步（标量）
            latents: SparseTensor，当前 latent
        
        Returns:
            SimpleNamespace: 包含 prev_sample
        """
        t_val = float(t)
        
        # 在 numpy float64 数组中查找匹配的索引（使用更宽松的容差）
        match_mask = np.isclose(self._timesteps_np, t_val, rtol=1e-5, atol=1e-8)
        match_idx = np.where(match_mask)[0]
        
        assert len(match_idx) > 0, f"t={t_val} 未匹配到 timesteps"
        idx = int(match_idx[0])
        
        return self.step_by_index(velocity, idx, latents)

