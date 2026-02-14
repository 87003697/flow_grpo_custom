# =====================================================================
# Imports
# =====================================================================
from typing import Any, Literal, Optional

import torch

from trellis2.modules.sparse import SparseTensor


Stage = Literal["shape", "tex"]


# =====================================================================
# CFG 函数（对齐 TRELLIS.2 参考实现）
# =====================================================================

def _sparse_pred_to_xstart(
    x_t: SparseTensor,
    t: float,
    pred: SparseTensor,
    sigma_min: float,
) -> SparseTensor:
    """
    从 velocity 预测 x0（对齐参考实现 FlowEulerSampler._pred_to_xstart）。
    
    公式: x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
    
    Args:
        x_t: SparseTensor，当前 latent
        t: 时间步 [0, 1]
        pred: SparseTensor，velocity 预测
        sigma_min: flow matching sigma_min 参数
    
    Returns:
        SparseTensor: 预测的 x_0
    """
    return (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred


def _sparse_xstart_to_pred(
    x_t: SparseTensor,
    t: float,
    x_0: SparseTensor,
    sigma_min: float,
) -> SparseTensor:
    """
    从 x0 转换为 velocity（对齐参考实现 FlowEulerSampler._xstart_to_pred）。
    
    公式: v = ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)
    
    Args:
        x_t: SparseTensor，当前 latent
        t: 时间步 [0, 1]
        x_0: SparseTensor，预测的 x_0
        sigma_min: flow matching sigma_min 参数
    
    Returns:
        SparseTensor: velocity 预测
    """
    # 使用 `/` 运算符，对齐参考实现（避免乘法 `* (1.0 / ...)` 的潜在精度差异）
    return ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)


def trellis2_cfg_sparse(
    cond_pred: SparseTensor,
    uncond_pred: SparseTensor,
    guidance_strength: float,
    guidance_rescale: float = 0.0,
    x_t: Optional[SparseTensor] = None,
    t: Optional[float] = None,
    sigma_min: float = 0.0,
) -> SparseTensor:
    """
    Classifier-Free Guidance (CFG) 函数，完全对齐 TRELLIS.2 参考实现。
    
    在 SparseTensor 上进行 CFG 混合，使用 SparseTensor 的运算符和 std 方法，
    确保与参考实现 ClassifierFreeGuidanceSamplerMixin._inference_model 完全一致。
    
    CFG 公式（加权平均）：
        pred = guidance_strength * cond_pred + (1 - guidance_strength) * uncond_pred
    
    CFG Rescale（对齐参考实现）：
        使用 SparseTensor.std(dim=[1], keepdim=True) 进行 std 计算。
    
    Args:
        cond_pred: SparseTensor，条件 velocity 预测
        uncond_pred: SparseTensor，无条件 velocity 预测
        guidance_strength: CFG 强度，通常 > 1.0
        guidance_rescale: CFG rescale 强度，0.0 表示不 rescale
        x_t: SparseTensor，当前 latent（rescale 需要）
        t: 当前时间步 [0, 1]（rescale 需要）
        sigma_min: flow matching sigma_min 参数（rescale 需要）
    
    Returns:
        SparseTensor: CFG 后的 velocity 预测
    """
    if guidance_strength == 1.0:
        return cond_pred  # SparseTensor
    
    if guidance_strength == 0.0:
        return uncond_pred  # SparseTensor
    
    # CFG 加权平均公式（在 SparseTensor 上进行，对齐参考实现）
    # 参考: pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    pred = guidance_strength * cond_pred + (1 - guidance_strength) * uncond_pred  # SparseTensor
    
    # CFG Rescale（对齐参考实现 ClassifierFreeGuidanceSamplerMixin）
    if guidance_rescale > 0 and x_t is not None and t is not None:
        # 从 velocity 预测 x0（在 SparseTensor 上进行）
        x_0_pos = _sparse_pred_to_xstart(x_t, t, cond_pred, sigma_min)  # SparseTensor
        x_0_cfg = _sparse_pred_to_xstart(x_t, t, pred, sigma_min)  # SparseTensor
        
        # 使用 SparseTensor.std（继承自 VarLenTensor.std）
        # 参考实现: x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
        # 对于 SparseTensor，ndim = 2（batch + channels），所以 dim=[1]
        std_pos = x_0_pos.std(dim=[1], keepdim=True)  # (B, 1) 普通 tensor
        std_cfg = x_0_cfg.std(dim=[1], keepdim=True)  # (B, 1) 普通 tensor
        
        # Rescale（SparseTensor * 普通 tensor 会通过 __elemwise__ 正确广播）
        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)  # SparseTensor
        x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg  # SparseTensor
        
        # 转换回 velocity
        pred = _sparse_xstart_to_pred(x_t, t, x_0, sigma_min)  # SparseTensor
    
    return pred  # SparseTensor


def _compute_v_regularization(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    v 正则化 Loss：MSE(v_stu, v_tea)，梯度仅当前步。
    
    Args:
        v_student: (N, C) 学生模型预测的 cond velocity
        v_teacher: (N, C) 教师模型预测的 cond velocity（已 detach）
    
    Returns:
        loss: 标量
    """
    diff = v_student - v_teacher.detach()  # (N, C)
    return (diff ** 2).mean()  # scalar



# =====================================================================
# Rollout 辅助函数
# =====================================================================

def auto_device_trellis2(fn):
    """装饰器：自动处理 Trellis2 跨设备推理。"""
    import functools
    
    @functools.wraps(fn)
    def wrapper(pipeline, x_t, t, cond_emb, stage, resolution, shape_cond=None):
        # 获取模型设备
        model_key = f"{stage}_slat_flow_model_{resolution}"
        model_device = next(pipeline.pipe.models[model_key].parameters()).device
        orig_device = x_t.feats.device
        
        if model_device == orig_device:
            return fn(pipeline, x_t, t, cond_emb, stage, resolution, shape_cond)
        
        # 转移输入
        x_t = SparseTensor(feats=x_t.feats.to(model_device), coords=x_t.coords.to(model_device))
        cond_emb = cond_emb.to(model_device)
        if shape_cond is not None:
            shape_cond = SparseTensor(feats=shape_cond.feats.to(model_device), coords=shape_cond.coords.to(model_device))
        
        out = fn(pipeline, x_t, t, cond_emb, stage, resolution, shape_cond)
        
        # 转回输出
        return SparseTensor(feats=out.feats.to(orig_device), coords=out.coords.to(orig_device))
    
    return wrapper


@auto_device_trellis2
def _predict_velocity(
    pipeline: Any,
    x_t: SparseTensor,
    t: float,
    cond_emb: torch.Tensor,
    stage: Stage,
    resolution: int,
    shape_cond: Optional[SparseTensor] = None,
) -> SparseTensor:
    """
    Velocity 预测（用于 checkpoint 包裹），自动处理跨设备。
    
    返回 SparseTensor 以保持完整的 SparseTensor 流程，对齐参考实现。
    
    Args:
        pipeline: Trellis2RefAdapter
        x_t: SparseTensor，当前 latent
        t: 时间步标量，范围 [0, 1]
        cond_emb: (B, S, C) 条件嵌入
        stage: "shape" 或 "tex"
        resolution: 512 或 1024
        shape_cond: SparseTensor，tex 阶段需要的 shape 条件（已归一化）
    
    Returns:
        SparseTensor: velocity 预测（保持完整的 SparseTensor 类型）
    """
    # 截断输入梯度：避免多步 rollout 的梯度串联导致爆炸/消失
    x_t = x_t.replace(x_t.feats.detach())  # SparseTensor(feats: (N, C), coords: (N, 4))
    # t 已经是 0-1 范围，直接传给 sampling_step（内部会乘 1000）
    out = pipeline.sampling_step(
        x_t, t, cond_emb, stage, resolution, shape_cond=shape_cond
    )  # SparseTensor
    
    return out  # SparseTensor
