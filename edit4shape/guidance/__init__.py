"""
Guidance 模块。

提供 FlowEdit 图像编辑 Guidance，用于 3D 生成训练。

主要接口：
- create_guidance(cfg, train_device): 创建 Guidance 实例
- GuidanceResult: 统一的返回结果格式

设备分配：
- Guidance 模型自动运行在 train_device + 1 的 GPU 上
- 例如：训练在 cuda:0，则 FlowEdit 在 cuda:1

Usage:
    from edit4shape.guidance import create_guidance
    
    guidance = create_guidance(cfg, accelerator.device)
    result = guidance.compute_guidance(comp_rgb, condition_images)
    loss = result.loss_ssim + result.loss_lpips
    loss.backward()
"""

from edit4shape.guidance.base import create_guidance, GuidanceResult

__all__ = ["create_guidance", "GuidanceResult"]
