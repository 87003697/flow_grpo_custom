import os, sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import torch
import numpy as np
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_TRELLIS2_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "TRELLIS.2")
if _TRELLIS2_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS2_ROOT)

from trellis2.modules import sparse as sp
SparseTensor = sp.SparseTensor

# 运行时配置（不含 logprob）
@dataclass
class SparseRuntimeConfig:
    steps: int = 50
    guidance_scale: float = 0.0
    deterministic: bool = False

@dataclass
class ShapeRuntimeConfig:
    guidance_scale: float
    deterministic: bool

@dataclass
class TexRuntimeConfig:
    guidance_scale: float
    deterministic: bool

def set_trellis_timesteps(scheduler: FlowMatchEulerDiscreteScheduler, steps: int, device="cpu", rescale_t: float = 1.0):
    dev = torch.device(device)
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # 形状: (steps+1,)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq / 1000)  # 形状: (steps+1,)
    sigma_seq = t_seq / 1000.0  # 形状: (steps+1,)
    scheduler.set_timesteps(num_inference_steps=steps, device=dev, timesteps=t_seq[:-1].tolist(), sigmas=sigma_seq[:-1].tolist())
    scheduler.timesteps = torch.from_numpy(t_seq).to(device=dev, dtype=torch.float32)  # 形状: (steps+1,)
    scheduler.sigmas = torch.from_numpy(sigma_seq).to(device=dev, dtype=torch.float32)  # 形状: (steps+1,)
    scheduler.num_inference_steps = steps
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler

def create_trellis_scheduler(steps: int, device="cpu", rescale_t: float = 1.0):
    scheduler = FlowMatchEulerDiscreteScheduler()
    return set_trellis_timesteps(scheduler, steps=steps, device=device, rescale_t=rescale_t)

# 基础批处理工具（可从 trellis_sparse_tensor.py 直接复制，省略形状注释）
def sparse_tensor_cat(tensors: List[sp.SparseTensor]) -> sp.SparseTensor:
    """批量拼接 SparseTensor（dim=0）。"""
    if not tensors:
        raise ValueError("输入张量列表为空")
    if len(tensors) == 1:
        return tensors[0]

    start = 0  # 形状: 标量
    coords = []
    for input_tensor in tensors:
        coords.append(input_tensor.coords.clone().to(torch.int32))  # 形状: (N_i, 4)
        coords[-1][:, 0] += start  # 形状: (N_i, 4)
        start += input_tensor.shape[0]  # 形状: 标量

    combined_coords = torch.cat(coords, dim=0)  # 形状: (sum(N_i), 4)
    combined_feats = torch.cat([input_tensor.feats for input_tensor in tensors], dim=0)  # 形状: (sum(N_i), C)

    return sp.SparseTensor(
        coords=combined_coords,  # 形状: (sum(N_i), 4)
        feats=combined_feats,    # 形状: (sum(N_i), C)
    )


def prepare_sparse_tensor_batch(
    sparse_list: List[sp.SparseTensor],
    batch_size: int,
) -> sp.SparseTensor:
    """将单样本 SparseTensor 列表合批。"""
    if len(sparse_list) != batch_size:
        raise ValueError(f"SparseTensor 列表长度 {len(sparse_list)} 与批次大小 {batch_size} 不匹配")

    adjusted_list = []
    for batch_idx, sparse_tensor in enumerate(sparse_list):
        adjusted_coords = sparse_tensor.coords.clone()  # 形状: (N_i, 4)
        adjusted_coords[:, 0] = batch_idx  # 形状: (N_i, 4)
        adjusted_sparse = sp.SparseTensor(
            coords=adjusted_coords,  # 形状: (N_i, 4)
            feats=sparse_tensor.feats,  # 形状: (N_i, C)
        )
        adjusted_list.append(adjusted_sparse)

    return sparse_tensor_cat(adjusted_list)  # 形状: batched 稀疏


def extract_sparse_tensor_from_batch(
    batch_sparse: sp.SparseTensor,
    batch_idx: int,
) -> sp.SparseTensor:
    """从 batched SparseTensor 提取单样本。"""
    mask = (batch_sparse.coords[:, 0] == batch_idx)  # 形状: (N,)
    if not mask.any():
        raise ValueError(f"批次索引 {batch_idx} 在 SparseTensor 中不存在")

    extracted_coords = batch_sparse.coords[mask].clone()  # 形状: (N_i, 4)
    extracted_feats = batch_sparse.feats[mask]  # 形状: (N_i, C)
    extracted_coords[:, 0] = 0  # 形状: (N_i, 4)

    return sp.SparseTensor(
        coords=extracted_coords,  # 形状: (N_i, 4)
        feats=extracted_feats,    # 形状: (N_i, C)
    )


def sparse_clone_with_feats(
    sparse: sp.SparseTensor,
    feats: torch.Tensor,
) -> sp.SparseTensor:
    """用新的 feats 复制 SparseTensor。"""
    if feats.shape != sparse.feats.shape:
        raise ValueError(f"feats 形状不匹配: 预期 {sparse.feats.shape}, 实际 {feats.shape}")
    new_feats = feats.to(dtype=sparse.feats.dtype, device=sparse.feats.device)  # 形状: (N_total, C)
    return sp.SparseTensor(
        coords=sparse.coords.clone(),  # 形状: (N_total, 4)
        feats=new_feats,               # 形状: (N_total, C)
        layout=list(getattr(sparse, "layout", [])),
    )


def sparse_batch_mse(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
    """按 layout 聚合的稀疏 MSE，返回 (B,)。"""
    if len(getattr(pred, "layout", [])) != len(getattr(target, "layout", [])):
        raise ValueError("pred 与 target 的 layout 长度不一致")
    mse_list: List[torch.Tensor] = []
    for sl_pred, sl_tgt in zip(pred.layout, target.layout):
        diff = pred.feats[sl_pred] - target.feats[sl_tgt]  # 形状: (N_b, C)
        mse_val = diff.pow(2).mean() if diff.numel() > 0 else torch.zeros((), device=pred.feats.device, dtype=pred.feats.dtype)  # 形状: ()
        mse_list.append(mse_val.unsqueeze(0))  # 形状: (1,)
    return torch.cat(mse_list, dim=0)  # 形状: (B,)


def dense_batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对稠密张量计算逐样本 MSE，返回 (BK,)。"""
    if pred.shape != target.shape:
        raise ValueError(f"pred 与 target 形状不一致: {pred.shape} vs {target.shape}")
    diff = pred - target  # 形状: (BK, C, R, R, R)
    mse = diff.pow(2).mean(dim=(1, 2, 3, 4))  # 形状: (BK,)
    return mse  # 形状: (BK,)
def compute_sparse_weighted_mse(pred: sp.SparseTensor, target: sp.SparseTensor) -> torch.Tensor:
    return sparse_batch_mse(pred, target)
def compute_dense_weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dense_batch_mse(pred, target)

# ===== 三阶段：直接调用 pipeline 的 sampler，返回采样结果（无 logprob） =====
def stage_sparse(pipeline, cond: Dict, ss_resolution: int, num_samples: int = 1, params: Dict = {}):
    """
    稀疏结构采样：调用 pipeline.sparse_structure_sampler.sample，返回 coords, sampler_out
    """
    flow = pipeline.models["sparse_structure_flow_model"]
    noise = torch.randn(num_samples, flow.in_channels, flow.resolution, flow.resolution, flow.resolution, device=pipeline.device)  # 形状: (B,C,R,R,R)
    out = pipeline.sparse_structure_sampler.sample(flow, noise, **cond, **{**pipeline.sparse_structure_sampler_params, **params}, verbose=True)  # 形状: samples(B,C,R,R,R)
    decoded = pipeline.models["sparse_structure_decoder"](out.samples) > 0  # 形状: (B,1,R,R,R)
    if ss_resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // ss_resolution  # 形状: 标量
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5  # 形状: (B,1,ssR,ssR,ssR)
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # 形状: (N,4)
    return coords, out

def stage_shape(pipeline, cond: Dict, coords: torch.Tensor, resolution: int = 1024, params: Dict = {}):
    """
    形状 SLat 采样：调用 shape_slat_sampler.sample，返回 slat, sampler_out
    """
    flow = pipeline.models[f"shape_slat_flow_model_{resolution}"]
    noise = SparseTensor(feats=torch.randn(coords.shape[0], flow.in_channels, device=pipeline.device), coords=coords)  # 形状: SparseTensor(N,C)
    out = pipeline.shape_slat_sampler.sample(flow, noise, **cond, **{**pipeline.shape_slat_sampler_params, **params}, verbose=True)  # 形状: samples SparseTensor(N,C)
    std = torch.tensor(pipeline.shape_slat_normalization["std"], device=pipeline.device)  # 形状: (C,)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"], device=pipeline.device)  # 形状: (C,)
    slat = out.samples * std + mean  # 形状: SparseTensor(N,C)
    return slat, out

def stage_tex(pipeline, cond: Dict, shape_slat: SparseTensor, params: Dict = {}):
    """
    纹理 SLat 采样：调用 tex_slat_sampler.sample，返回 tex_slat, sampler_out
    """
    std_s = torch.tensor(pipeline.shape_slat_normalization["std"], device=pipeline.device)  # 形状: (C_shape,)
    mean_s = torch.tensor(pipeline.shape_slat_normalization["mean"], device=pipeline.device)  # 形状: (C_shape,)
    shape_norm = (shape_slat - mean_s) / std_s  # 形状: SparseTensor(N,C_shape)
    flow = pipeline.models["tex_slat_flow_model_1024"]
    in_ch = flow.in_channels if hasattr(flow, "in_channels") else flow[0].in_channels  # 形状: 标量
    noise = shape_norm.replace(feats=torch.randn(shape_norm.coords.shape[0], in_ch - shape_norm.feats.shape[1], device=pipeline.device))  # 形状: SparseTensor(N,C_tex_noise)
    out = pipeline.tex_slat_sampler.sample(flow, noise, concat_cond=shape_norm, **cond, **{**pipeline.tex_slat_sampler_params, **params}, verbose=True)  # 形状: samples SparseTensor(N,C_tex)
    std_t = torch.tensor(pipeline.tex_slat_normalization["std"], device=pipeline.device)  # 形状: (C_tex,)
    mean_t = torch.tensor(pipeline.tex_slat_normalization["mean"], device=pipeline.device)  # 形状: (C_tex,)
    tex = out.samples * std_t + mean_t  # 形状: SparseTensor(N,C_tex)
    return tex, out

__all__ = [
    "SparseTensor",
    "SparseRuntimeConfig",
    "ShapeRuntimeConfig",
    "TexRuntimeConfig",
    "set_trellis_timesteps",
    "create_trellis_scheduler",
    "stage_sparse",
    "stage_shape",
    "stage_tex",
    "sparse_tensor_cat",
    "prepare_sparse_tensor_batch",
    "extract_sparse_tensor_from_batch",
    "sparse_clone_with_feats",
    "sparse_batch_mse",
    "dense_batch_mse",
    "compute_sparse_weighted_mse",
    "compute_dense_weighted_mse",
]