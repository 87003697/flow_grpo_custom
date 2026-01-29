"""
Qwen-Image-Edit Pipeline 模块。

包含：
- FlowEditSimplePipeline: FlowEdit 简化版（Source branch 使用解析式，支持 CSD + MSE 混合 loss）
- FlowEditFullPipeline: FlowEdit 完整版（双分支模型推理）
- FlowEditStateTracker: 统一状态追踪器（同时记录 z_edit 和 x0_high/x0_low）
- QwenImageSDSPipeline: SDS 梯度蒸馏 (grad = noise_pred - noise)
- QwenImageCSDPipeline: CSD 梯度蒸馏 (grad = x0_low - x0_high)
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin

Pipeline 类型：
- "simple": FlowEditSimplePipeline（Source branch 使用解析式，速度快）
- "full": FlowEditFullPipeline（双分支都使用模型推理，效果更好）

Loss 类型通过 csd_weight 和 mse_weight 控制：
- csd_weight=1, mse_weight=0 → 纯 CSD
- csd_weight=0, mse_weight=1 → 纯 MSE
- csd_weight=1, mse_weight=0.5 → 混合模式
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
    SDSStateTracker,
    CSDStateTracker,
)
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin
from edit4shape.guidance.pipelines.qwen_image_edit.sds import QwenImageSDSPipeline, SDSOutput
from edit4shape.guidance.pipelines.qwen_image_edit.csd import QwenImageCSDPipeline, CSDOutput

__all__ = [
    "FlowEditSimplePipeline",
    "FlowEditFullPipeline",
    "FlowEditPipelineOutput",
    "FlowEditStateTracker",
    "SDSStateTracker",
    "CSDStateTracker",
    "DifferentiableVAEMixin",
    "QwenImageSDSPipeline",
    "SDSOutput",
    "QwenImageCSDPipeline",
    "CSDOutput",
]
