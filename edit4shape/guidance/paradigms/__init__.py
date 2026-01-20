"""
Guidance Paradigms 模块。

提供不同的 Guidance 范式实现：
- FlowEdit: 编辑图像 → 计算相似度 loss
- SDS: Score Distillation Sampling 梯度蒸馏
"""

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditGuidancePP
from edit4shape.guidance.paradigms.sds import SDSGuidance

__all__ = ["FlowEditGuidance", "FlowEditGuidancePP", "SDSGuidance"]
