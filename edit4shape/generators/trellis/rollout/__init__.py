"""
Trellis Rollout 模块

提供 ODE 和 SDE 两种采样模式:
- ODE: 标准 Euler 采样，用于推理和 ReFL/DRaFT 风格训练
- SDE: 随机采样 + 轨迹追踪，用于 Nabla-R2D3 风格 Score Matching 训练

Tracker 类:
- RolloutTracker (rollout.autograd_tracker): 三阶段 Autograd 的 cond-level proxy 记录器
- SDERolloutTracker (state.tracker): SDE 采样轨迹追踪器（Nabla 训练专用）
"""

from .base import mix_cfg_sparse, _predict_cond_velocity, auto_device, predict_velocity_with_cfg
from .ode import rollout_sparse
from .sde import rollout_sparse_sde, compute_score_matching_loss
from .autograd_tracker import RolloutTracker, VelocityTracker

__all__ = [
    # Base utilities
    "mix_cfg_sparse",
    "_predict_cond_velocity",
    "auto_device",
    "predict_velocity_with_cfg",
    # ODE rollout
    "rollout_sparse",
    # SDE rollout + Score Matching
    "rollout_sparse_sde",
    "compute_score_matching_loss",
    # Autograd tracker
    "RolloutTracker",
    "VelocityTracker",
]
