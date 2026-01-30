"""
Qwen-Image-Edit Pipeline 模块。

包含：
- FlowEditSimplePipeline: FlowEdit 简化版（Source branch 使用解析式）
- FlowEditFullPipeline: FlowEdit 完整版（双分支模型推理）
- QwenImageDistillationPipeline: 蒸馏 Pipeline
- StateTracker: 统一状态追踪器（random / fixed / aligned 模式）
- InversionStateTracker: Inversion 状态追踪器（inversion_* 模式）
- create_tracker: 工厂函数，根据 noise_mode 创建对应 Tracker
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin

噪声模式：
- "random": 每步随机噪声
- "fixed": 固定噪声
- "aligned": DNAEdit 风格累积补偿
- "inversion_cond": Naive Inversion（用 v_cond）
- "inversion_uncond": Naive Inversion（用 v_uncond）
- "inversion_cfg": Naive Inversion（用 v_cfg）
"""

from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_simple import (
    FlowEditPipeline as FlowEditSimplePipeline,
    FlowEditPipelineOutput,
)
from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_full import (
    FlowEditPipeline as FlowEditFullPipeline,
)
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import (
    StateTracker,
    InversionStateTracker,
    create_tracker,
)
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin
from edit4shape.guidance.pipelines.qwen_image_edit.distillation import (
    QwenImageDistillationPipeline,
    DistillationOutput,
)

__all__ = [
    # Pipeline
    "FlowEditSimplePipeline",
    "FlowEditFullPipeline",
    "FlowEditPipelineOutput",
    "QwenImageDistillationPipeline",
    "DistillationOutput",
    # Tracker
    "StateTracker",
    "InversionStateTracker",
    "create_tracker",
    # Mixin
    "DifferentiableVAEMixin",
]
