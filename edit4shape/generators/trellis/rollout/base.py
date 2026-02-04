"""
Rollout 基础辅助函数

提供 CFG 混合、设备自动转移等通用功能。
"""

from typing import Any, Optional
import torch
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Rollout 辅助函数（模块级，避免嵌套）
# =====================================================================

def mix_cfg_sparse(
    cond_pred: Any,  # SparseTensor
    uncond_pred: Any,  # SparseTensor
    scale: float,
    uncond_mode: str = "detach",
) -> Any:  # SparseTensor
    """
    基于 SparseTensor 的 Classifier-Free Guidance (CFG) 混合函数。
    
    CFG 公式: output = cond_pred + scale * (cond_pred - uncond_pred)
    
    Args:
        cond_pred: SparseTensor，条件预测结果
        uncond_pred: SparseTensor，无条件预测结果，可为 None
        scale: CFG 缩放因子，通常 > 1.0 以增强条件效果
        uncond_mode: 梯度处理模式
            - "detach": 对 uncond_pred 断开梯度（默认）
            - "mirror": 对 cond_pred 断开梯度
            - "none": 保持两者梯度
    
    Returns:
        SparseTensor: 混合后的预测结果
    """
    if uncond_pred is None:
        return cond_pred
    
    # 在 feats 上进行 CFG 混合
    cond_feats = cond_pred.feats  # (N,C)
    uncond_feats = uncond_pred.feats  # (N,C)
    
    if uncond_mode == "detach":
        uncond_feats = uncond_feats.detach()
    elif uncond_mode == "mirror":
        cond_feats = cond_feats.detach()
    # "none" 模式保持原样
    
    mixed_feats = cond_feats + scale * (cond_feats - uncond_feats)  # (N,C)
    
    # 用混合后的 feats 创建新的 SparseTensor
    return cond_pred.replace(mixed_feats)




def auto_device(model_name="slat_flow_model"):
    """
    装饰器：自动将输入转到模型设备，输出转回原设备。
    
    注意：跨设备时必须创建全新的 SparseTensor，不能使用 .to() 方法。
    因为 SparseTensor.to() 不会转移 spatial_cache 和 spconv 内部的 indice_dict，
    这些缓存仍在原设备上，导致 cudaErrorIllegalAddress 错误。
    """
    import functools
    
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(pipeline, x_t, *args, **kwargs):
            model_device = next(pipeline.pipe.models[model_name].parameters()).device
            orig_device = x_t.feats.device
            
            if model_device == orig_device:
                return fn(pipeline, x_t, *args, **kwargs)
            else:
                raise ValueError(f"Model device {model_device} and original device {orig_device} are different.")

            # 创建全新的 SparseTensor，不复用任何缓存（避免跨设备缓存问题）
            x_t_new = SparseTensor(
                feats=x_t.feats.to(model_device),  # (N, C)
                coords=x_t.coords.to(model_device),  # (N, 4)
            )
            inputs = [x_t_new] + [a.to(model_device) for a in args]
            
            result = fn(pipeline, *inputs, **kwargs)
            
            # 返回时也创建全新的 SparseTensor
            return SparseTensor(
                feats=result.feats.to(orig_device),  # (N, C)
                coords=result.coords.to(orig_device),  # (N, 4)
            )
        
        return wrapper
    return decorator


@auto_device("slat_flow_model")
def _predict_cond_velocity(pipeline, x_t, t_batch, cond_emb):
    """Velocity 预测（自动适配设备）。"""
    # ★ 截断输入梯度：避免多步 rollout 的梯度串联导致爆炸/消失
    x_t = x_t.replace(x_t.feats.detach())  # SparseTensor(feats: (N, C), coords: (N, 4))
    return pipeline.sparse_sampling_step(x_t, t_batch, cond_emb, None, 0.0)


def predict_velocity_with_cfg(
    pipeline,
    x_t: SparseTensor,
    t_val: float,
    cond_emb: torch.Tensor,
    uncond_emb: Optional[torch.Tensor],
    slat_guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """
    速度场预测 + CFG 混合（通用版）
    
    注意：Gradient checkpointing 应在模型层面启用（model.blocks[i].use_checkpoint = True），
    而非在此函数中处理。
    
    Args:
        pipeline: 模型 pipeline
        x_t: 当前状态 SparseTensor
        t_val: 时间步
        cond_emb: 条件编码 (B, S, C)
        uncond_emb: 无条件编码 (B, S, C) 或 None
        slat_guidance: CFG scale
        cfg_min, cfg_max: CFG 应用区间
        device: 运行设备
        
    Returns:
        velocity: SparseTensor，速度场预测
    """
    B = cond_emb.shape[0]
    t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
    use_cfg = cfg_min <= t_val <= cfg_max
    
    # 条件预测（checkpointing 由模型内部 block 处理）
    cond_pred = _predict_cond_velocity(pipeline, x_t, t_batch, cond_emb)  # SparseTensor
    
    # CFG 混合
    if use_cfg and uncond_emb is not None:
        with torch.no_grad():
            uncond_pred = _predict_cond_velocity(pipeline, x_t, t_batch, uncond_emb)  # SparseTensor
        return mix_cfg_sparse(cond_pred, uncond_pred, slat_guidance, uncond_mode="detach")  # SparseTensor
    
    return cond_pred
