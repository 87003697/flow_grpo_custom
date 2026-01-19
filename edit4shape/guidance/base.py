"""
Guidance 模块基础设施。

提供：
- GuidanceResult: 统一的返回格式
- BaseGuidance: 抽象基类，定义统一接口
- SpecifyGradient: 梯度注入工具（用于 SDS/CSD/VSD）
- create_guidance(): 工厂函数，根据配置创建 Guidance 实例

设备分配：
- Guidance 模型默认运行在 训练设备 + 1 的 GPU 上
- 例如：训练在 cuda:0，则 Guidance 在 cuda:1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, List, Dict
import torch
from torch.autograd import Function
from PIL import Image

if TYPE_CHECKING:
    from edit4shape.guidance.pipelines.qwen_image_edit.utils import FlowEditStateTracker


# =====================================================================
# SpecifyGradient - 梯度注入工具
# =====================================================================

class SpecifyGradient(Function):
    """
    自定义 autograd Function，用于将预计算的梯度注入到反向传播中。
    
    用于 SDS/CSD/VSD：将 noise prediction 差异作为梯度注入，
    使得 loss.backward() 能将梯度穿透 rollout 链回传到参数。
    
    Usage:
        grad = noise_pred - noise  # 预计算的梯度
        loss = SpecifyGradient.apply(latents, grad)  # 返回伪 loss
        loss.backward()  # 梯度会注入到 latents
    
    Reference: threestudio
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, gt_grad: torch.Tensor) -> torch.Tensor:
        """
        前向传播：保存梯度，返回标量 1。
        
        Args:
            input_tensor: 需要注入梯度的张量
            gt_grad: 预计算的梯度（与 input_tensor 形状相同）
        
        Returns:
            标量 tensor（用于 backward 触发）
        """
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)
    
    @staticmethod
    def backward(ctx, grad_scale: torch.Tensor):
        """
        反向传播：返回预计算的梯度。
        
        Args:
            grad_scale: 来自后续层的梯度（通常为 1）
        
        Returns:
            (gt_grad * grad_scale, None): 注入的梯度
        """
        gt_grad, = ctx.saved_tensors
        return gt_grad * grad_scale, None


# =====================================================================
# GuidanceResult - 统一返回格式
# =====================================================================

@dataclass
class GuidanceResult:
    """
    Guidance 计算结果（通用格式）。
    
    Attributes:
        loss: 主 loss（可直接 backward）
        edited_imgs: 编辑后的图像 (B,V,C,H,W)，FlowEdit 专用
        loss_dict: 细分 loss 字典，用于日志记录
        trackers: FlowEdit 中间状态跟踪器列表（用于多步监督）
    """
    loss: torch.Tensor                                                  # 主 loss（必须）
    edited_imgs: Optional[torch.Tensor] = None                          # (B,V,C,H,W) FlowEdit 专用
    loss_dict: Optional[Dict[str, torch.Tensor]] = field(default=None)  # 细分 loss
    trackers: Optional[List["FlowEditStateTracker"]] = None             # FlowEdit 专用


# =====================================================================
# BaseGuidance - 抽象基类
# =====================================================================

class BaseGuidance(ABC):
    """
    Guidance 抽象基类。
    
    所有 Guidance 范式（FlowEdit、CSD、SDS、VSD）都继承此类。
    定义统一的 compute_guidance() 接口。
    """
    
    device: torch.device
    train_device: torch.device
    
    @abstractmethod
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> GuidanceResult:
        """
        计算 Guidance loss。
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C) 或 (B,V,C,H,W)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 额外参数（如 prompt、timestep 等）
        
        Returns:
            GuidanceResult: 包含 loss 和可选的 edited_imgs
        """
        pass


# =====================================================================
# 工厂函数
# =====================================================================

def create_guidance(cfg: Any, train_device: torch.device, use_pp: bool = False) -> BaseGuidance:
    """
    创建 Guidance 实例。
    
    根据 cfg.guidance.type 选择不同的 Guidance 范式：
    - "flowedit": FlowEdit（编辑图像 → 计算相似度 loss）
    - "csd": CSD（Classifier-free Score Distillation）
    - "sds": SDS（Score Distillation Sampling）
    
    Args:
        cfg: 配置对象，需包含 guidance 子配置
        train_device: 训练使用的设备（如 cuda:0）
        use_pp: 是否使用流水线并行版本
    
    Returns:
        BaseGuidance: Guidance 实例
    
    Example:
        >>> guidance = create_guidance(cfg, accelerator.device)
        >>> result = guidance.compute_guidance(comp_rgb, condition_images)
        >>> result.loss.backward()
    """
    paradigm = cfg.guidance.get("type", "flowedit")
    
    if paradigm == "flowedit":
        if use_pp:
            from edit4shape.guidance.paradigms.flowedit import FlowEditGuidancePP
            return FlowEditGuidancePP(cfg, train_device)
        else:
            from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance
            return FlowEditGuidance(cfg, train_device)
    elif paradigm == "csd":
        # TODO: 实现 CSDGuidance
        raise NotImplementedError("CSD guidance not implemented yet")
    elif paradigm == "sds":
        from edit4shape.guidance.paradigms.sds import SDSGuidance
        return SDSGuidance(cfg, train_device)
    else:
        raise ValueError(f"Unknown guidance type: {paradigm}. Choose from: flowedit, csd, sds")
