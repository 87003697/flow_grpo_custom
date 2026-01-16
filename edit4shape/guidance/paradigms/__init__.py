"""
Guidance Paradigms 模块。

提供不同的 Guidance 范式实现：
- FlowEdit: 编辑图像 → 计算相似度 loss
- Distillation: SDS/CSD/VSD 梯度蒸馏（待实现）
"""

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditGuidancePP

__all__ = ["FlowEditGuidance", "FlowEditGuidancePP"]
