"""
Guidance Paradigms 模块。

提供不同的 Guidance 范式实现：
- FlowEdit: 多步编辑式蒸馏（生成编辑图像 → 计算相似度 loss）
- Distillation: 单步蒸馏（SDS/CSD，通过权重控制）
- BilevelDistillation: 双层蒸馏（VSD，教师-学生双层优化）
"""

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditGuidancePP
from edit4shape.guidance.paradigms.distillation import DistillationGuidance
from edit4shape.guidance.paradigms.bilevel_distillation import BilevelDistillationGuidance
from edit4shape.guidance.paradigms.flowedit_latent_gan import FlowEditLatentGANGuidance

__all__ = [
    "FlowEditGuidance",
    "FlowEditGuidancePP",
    "DistillationGuidance",
    "BilevelDistillationGuidance",
    "FlowEditLatentGANGuidance",
]
