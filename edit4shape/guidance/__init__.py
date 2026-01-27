"""
Guidance 模块。

提供多种 Guidance 范式，用于 3D 生成训练：
- FlowEdit: 编辑图像 → 计算相似度 loss
- CSD/SDS: Score Distillation 梯度蒸馏

主要接口：
- create_guidance(cfg, train_device, use_pp): 创建 Guidance 实例
- GuidanceResult: 统一的返回结果格式
- BaseGuidance: 抽象基类

设备分配：
- Guidance 模型自动运行在 train_device + 1 的 GPU 上
- 例如：训练在 cuda:0，则 Guidance 在 cuda:1

所有 Guidance 统一使用真 Loss 模式：
- Pipeline 返回 Tracker（包含 x0 / z_edits 等状态）
- 通过 Tracker.loss(src) 计算真 loss
- 直接 loss.backward()，无需 SpecifyGradient

Usage:
    from edit4shape.guidance import create_guidance
    
    guidance = create_guidance(cfg, accelerator.device)
    result = guidance.compute_guidance(comp_rgb, condition_images)
    result.loss.backward()  # 统一使用 result.loss
"""

from edit4shape.guidance.base import (
    create_guidance,
    GuidanceResult,
    BaseGuidance,
)
from edit4shape.guidance.pipeline_parallel import PipelineParallelMixin

__all__ = [
    "create_guidance",
    "GuidanceResult",
    "BaseGuidance",
    "PipelineParallelMixin",
]
