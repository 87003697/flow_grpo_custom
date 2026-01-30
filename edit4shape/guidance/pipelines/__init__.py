"""
Pipelines 模块。

提供各种 diffusers Pipeline 子类，按基模分组：
- qwen_image_edit/: Qwen-Image-Edit 系列
- flux/: Flux Kontext 系列（待扩展）
"""

from edit4shape.guidance.pipelines.qwen_image_edit import (
    FlowEditSimplePipeline,
    FlowEditFullPipeline,
)
from edit4shape.guidance.pipelines.adapters import (
    create_pipeline_adapter,
    BasePipelineAdapter,
    EditResult,
)

__all__ = [
    "FlowEditSimplePipeline",
    "FlowEditFullPipeline",
    "create_pipeline_adapter",
    "BasePipelineAdapter",
    "EditResult",
]
