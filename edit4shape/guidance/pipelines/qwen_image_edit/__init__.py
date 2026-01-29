"""
Qwen-Image-Edit Pipeline 模块。

包含：
- FlowEditSimplePipeline: FlowEdit 简化版（Source branch 使用解析式，支持 CSD + MSE 混合 loss）
- FlowEditFullPipeline: FlowEdit 完整版（双分支模型推理）
- FlowEditStateTracker: 多步状态追踪器（同时记录 z_edit 和 x0_high/x0_low）
- QwenImageDistillationPipeline: 单步蒸馏 Pipeline（统一 SDS + CSD）
- DistillationStateTracker: 单步蒸馏状态追踪器
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin

Pipeline 类型：
- "simple": FlowEditSimplePipeline（Source branch 使用解析式，速度快）
- "full": FlowEditFullPipeline（双分支都使用模型推理，效果更好）

Loss 类型通过权重控制：
- FlowEdit: csd_weight / mse_weight
- Distillation: sds_weight / csd_weight
"""

from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_simple import (
    FlowEditPipeline as FlowEditSimplePipeline,
    FlowEditPipelineOutput,
)
from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_full import (
    FlowEditPipeline as FlowEditFullPipeline,
)
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import (
    FlowEditStateTracker,
    DistillationStateTracker,
)
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin
from edit4shape.guidance.pipelines.qwen_image_edit.distillation import (
    QwenImageDistillationPipeline,
    DistillationOutput,
)

__all__ = [
    "FlowEditSimplePipeline",
    "FlowEditFullPipeline",
    "FlowEditPipelineOutput",
    "FlowEditStateTracker",
    "DistillationStateTracker",
    "DifferentiableVAEMixin",
    "QwenImageDistillationPipeline",
    "DistillationOutput",
]
