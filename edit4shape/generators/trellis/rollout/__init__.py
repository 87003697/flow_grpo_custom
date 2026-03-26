"""
Trellis Rollout 模块

提供 ODE 采样模式:
- ODE: 标准 Euler 采样，用于推理和 ReFL/DRaFT 风格训练

Tracker 类:
- RolloutTracker (rollout.autograd_tracker): 三阶段 Autograd 的 cond-level proxy 记录器
"""

from .base import mix_cfg_sparse, _predict_sparse_cond_velocity, auto_device_sparse, auto_device_dense, predict_sparse_velocity_with_cfg
from .ode import rollout_sparse, rollout_dense
from .autograd_tracker import RolloutTracker, VelocityTracker

__all__ = [
    # Base utilities
    "mix_cfg_sparse",
    "_predict_sparse_cond_velocity",
    "auto_device_sparse",
    "auto_device_dense",
    "predict_sparse_velocity_with_cfg",
    # ODE rollout
    "rollout_sparse",
    "rollout_dense",
    # Autograd tracker
    "RolloutTracker",
    "VelocityTracker",
]
