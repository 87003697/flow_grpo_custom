"""Flow Matching Scheduler 基类 - 极简版"""
import torch
from typing import Optional


class FlowScheduler:
    """
    Flow Matching 调度器基类，兼容 sde_step_with_logprob。
    
    属性:
        timesteps: 时间步序列，从 1 到 0
        sigmas: Flow Matching 中 sigma = t
    """
    
    def __init__(self):
        self.timesteps: torch.Tensor = torch.tensor([])
    
    @property
    def sigmas(self) -> torch.Tensor:
        """Flow Matching 中 sigma = t"""
        return self.timesteps
    
    def set_timesteps(self, num_steps: int, device: torch.device, rescale_t: float = 1.0):
        """
        设置时间步序列（从1到0，共 num_steps+1 个点）。
        
        Args:
            num_steps: 采样步数
            device: 设备
            rescale_t: 时间步缩放因子（默认1.0，无缩放）
        """
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)  # (num_steps+1,)
        self.timesteps = rescale_t * t / (1 + (rescale_t - 1) * t)  # (num_steps+1,)
    
    def index_for_timestep(self, t: float) -> int:
        """查找时间步索引"""
        diffs = (self.timesteps - t).abs()  # (num_steps+1,)
        return int(diffs.argmin().item())
