"""
Metric 模块。

提供统一的相似度计算接口，支持：
- MSE: 像素空间均方误差
- SSIM: 结构相似性
- LPIPS: 感知相似性（VGG 特征）
- DINO: DINOv3 特征相似性
- CLIP: CLIP 图像特征相似性

注：latent_csd 和 latent_mse 由 StateTracker.loss() 计算，不在此处注册。

使用方式：
    from edit4shape.guidance.metric import create_metrics
    
    metrics = create_metrics(loss_cfg, device, extra_kwargs={...})
    for name, metric in metrics.items():
        loss = metric.compute(rendered, target)
"""
from typing import Dict, Any

import torch

from .base import BaseMetric
from .mse import MSEMetric
from .ssim import SSIMMetric
from .lpips import LPIPSMetric
from .dino import DINOMetric
from .clip import CLIPMetric


# 注册表：name -> class
# 注：latent_csd/latent_mse 不在此注册，由 Tracker.loss() 处理
METRIC_REGISTRY: Dict[str, type] = {
    "mse": MSEMetric,
    "ssim": SSIMMetric,
    "lpips": LPIPSMetric,
    "dino": DINOMetric,
    "clip": CLIPMetric,
}


def create_metrics(
    loss_cfg: Any,
    device: torch.device,
    extra_kwargs: Dict[str, Dict] = None,
) -> Dict[str, BaseMetric]:
    """
    根据 loss_cfg 中的权重创建 metrics（weight > 0 才创建）。
    
    Args:
        loss_cfg: 配置对象，包含各 metric 的权重（如 loss_cfg.ssim, loss_cfg.lpips）
        device: 计算设备
        extra_kwargs: 额外参数，格式 {"metric_name": {"param": value}}
    
    Returns:
        dict: {name: metric} 仅包含 weight > 0 的 metrics
    
    Example:
        >>> metrics = create_metrics(
        ...     loss_cfg,  # ssim=0.0, lpips=1.0, dino=0.5
        ...     device,
        ...     extra_kwargs={
        ...         "dino": {"model_path": "...", "image_size": 518},
        ...     }
        ... )
        >>> # 返回 {"lpips": LPIPSMetric(...), "dino": DINOMetric(...)}
    """
    extra_kwargs = extra_kwargs or {}
    metrics = {}
    
    for name, cls in METRIC_REGISTRY.items():
        # 获取权重：必须在 config 中显式声明
        weight = loss_cfg[name]
        
        if weight > 0:
            kwargs = extra_kwargs.get(name, {})
            metrics[name] = cls(weight=weight, device=device, **kwargs)
    
    return metrics


__all__ = [
    "BaseMetric",
    "MSEMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "DINOMetric",
    "CLIPMetric",
    "METRIC_REGISTRY",
    "create_metrics",
]








