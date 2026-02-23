"""
Pipelines 模块。

提供各种 diffusers Pipeline 子类，按基模分组：
- qwen_image_edit/: Qwen-Image-Edit 系列
- flux/: Flux Kontext 系列（待扩展）
"""

from edit4shape.guidance.pipelines.qwen_image_edit import (
    FlowEditFullPipeline,
)

__all__ = [
    "FlowEditFullPipeline",
]
