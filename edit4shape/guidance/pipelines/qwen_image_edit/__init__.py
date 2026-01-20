"""
Qwen-Image-Edit Pipeline 模块。

包含：
- FlowEditSimplePipeline: FlowEdit 简化版（source branch 解析式）
- FlowEditPipeline: FlowEdit 完整版（双分支模型推理）
- QwenImageSDSPipeline: SDS 梯度蒸馏 (grad = noise_pred - noise)
- QwenImageCSDPipeline: CSD 梯度蒸馏 (grad = x0_low - x0_high)
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin
"""

from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_simple import FlowEditSimplePipeline
from edit4shape.guidance.pipelines.qwen_image_edit.flowedit_full import FlowEditPipeline
from edit4shape.guidance.pipelines.qwen_image_edit.utils import FlowEditStateTracker, DifferentiableVAEMixin
from edit4shape.guidance.pipelines.qwen_image_edit.sds import QwenImageSDSPipeline, SDSOutput
from edit4shape.guidance.pipelines.qwen_image_edit.csd import QwenImageCSDPipeline, CSDOutput

__all__ = [
    "FlowEditSimplePipeline",
    "FlowEditPipeline",
    "FlowEditStateTracker",
    "DifferentiableVAEMixin",
    "QwenImageSDSPipeline",
    "SDSOutput",
    "QwenImageCSDPipeline",
    "CSDOutput",
]
