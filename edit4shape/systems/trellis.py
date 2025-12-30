"""
Trellis 单 renderer 版（适配 Gen2Turbo Trellis 逻辑）。

本模块实现了基于 TRELLIS 架构的 3D 生成系统，支持从单张图像生成 3D 模型。
核心流程：图像条件编码 -> 结构采样 (Dense Sampling) -> 特征采样 (Sparse Sampling) -> 解码 -> 渲染

特性：
- 单 renderer，训练/推理共用统一 rollout。
- 必需稠密结构 coords，若缺失直接报错。
- 统一步数 num_steps_sparse，训练/推理一致。
- 全程 CFG：每步都跑 cond/uncond，再 mix_cfg。

主要组件：
1. TrellisState: 存储生成状态（坐标、特征、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_sparse: 执行稀疏特征的去噪采样过程
4. train_edit4shape: 训练循环，支持 Flow Matching 训练
5. evaluate: 评估循环，生成 mesh 并保存可视化结果

依赖：
- TRELLIS 参考实现 (_reference_codes/TRELLIS)
- Accelerate 分布式训练库
- nvdiffrast (mesh 渲染) 或 Gaussian Splatting (GS 渲染)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import argparse
import csv
import json
import os
import random
import sys
import importlib.util
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, List

# =====================================================================
# 第三方库导入
# =====================================================================
from PIL import Image
import numpy as np
import requests
import yaml
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点，节省显存
from tqdm import tqdm

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.datasets.trellis import TrellisDataConfig, TrellisDataModule

# 使用 absl 的 config_flags 管理配置文件
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")

# =====================================================================
# TRELLIS 参考实现路径设置
# 将 TRELLIS 参考代码目录加入 Python 路径，以便导入其模块
# =====================================================================
import os
import sys
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
# 包含 coords (坐标) 和 feats (特征) 两个主要属性
from trellis.modules.sparse import SparseTensor

# =====================================================================
# Guidance 模块
# =====================================================================

from edit4shape.guidance import create_guidance
from edit4shape.systems.base import SpecifyGradient


# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    mix_cfg,
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    BaseState,
    TrellisState,
    System,
    CheckpointIO,
    build_run_paths,
)
from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO, LossDict


# TrellisState 和 System 已从 base.py 导入，不再重复定义


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> System:
    """
    构建完整的 Trellis 系统。
    
    根据配置创建所有必要的组件，包括：
    1. Pipeline: 负责条件编码、结构/特征采样、解码的核心生成管道
    2. Renderer: 将 3D 表示渲染为 2D 图像的渲染器
    3. Guidance: 训练时的指导模块（如 SDS loss）
    4. Optimizer: 模型参数优化器
    
    Args:
        cfg: 完整配置对象，包含以下关键配置：
            - cfg.renderer: 渲染器配置（类型、分辨率、近远裁剪面等）
            - cfg.train.optimizer: 优化器配置
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 分布式训练加速器
    
    Returns:
        System: 包含所有组件的系统实例
    """
    # ---- 1. 构建 Pipeline (核心生成管道) ----
    # Pipeline 封装了 TRELLIS 的所有生成逻辑，包括：
    # - 图像条件编码 (DINOv2 等)
    # - 结构采样 (Dense Structure Sampling)
    # - 特征采样 (Sparse Latent Sampling, SLAT)
    # - 解码器 (Mesh/GS 解码)
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    pipeline = build_pipeline_from_reference(cfg, accelerator)

    # ---- 2. 构建 Renderer (3D 渲染器) ----
    # 根据配置选择渲染方式：
    # - "mesh": 基于 nvdiffrast 的可微分网格光栅化
    # - "gs": 基于 3D Gaussian Splatting 的渲染
    renderer_type = cfg.renderer.type  # "mesh" 或 "gs"
    
    if renderer_type == "gs":
        # ---- Gaussian Splatting 渲染器 ----
        # 优势：渲染速度快，支持实时渲染
        # 适用场景：预览、快速迭代
        from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
        rendering_options = {
            "resolution": cfg.renderer.resolution,  # 渲染分辨率 (像素)
            "near": cfg.renderer.near,  # 近裁剪面距离
            "far": cfg.renderer.far,    # 远裁剪面距离
            "ssaa": cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            "bg_color": cfg.renderer.bg_color,  # 背景色模式
        }
        renderer = GaussianRenderer(rendering_options)
    else:
        # ---- Mesh 光栅化渲染器 (nvdiffrast) ----
        # 优势：支持精确的几何渲染，法线/深度图质量高
        # 适用场景：训练、精细渲染
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
        renderer_cfg = TrellisRendererConfig(
            resolution=cfg.renderer.resolution,  # 渲染分辨率 (像素)
            ssaa=cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            near=cfg.renderer.near,  # 近裁剪面距离
            far=cfg.renderer.far,    # 远裁剪面距离
        )
        renderer = TrellisMeshRasterizer(cfg=renderer_cfg, device=str(accelerator.device))

    # ---- 3. 构建 Guidance 和 Optimizer ----
    # 仅在训练模式下创建 Guidance 和优化器
    guidance = None
    optimizer = None

    if not cfg.eval_only:
        # 创建 Guidance（FlowEdit 模型自动放在 train_device + 1）
        guidance = create_guidance(cfg, train_device=accelerator.device)
        
        # 为 SLAT (Sparse Latent) 模型创建优化器
        from edit4shape.generators.trellis.training_adpter import build_optimizer_for_slat
        slat_model = pipeline.pipe.models["slat_flow_model"]  # 获取 SLAT flow 模型
        optimizer = build_optimizer_for_slat(slat_model, cfg.train.optimizer)

    return System(pipeline=pipeline, renderer=renderer, guidance=guidance, optimizer=optimizer)


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练和评估的 DataLoader。
    
    使用 TrellisDataModule 统一管理数据加载，支持：
    - 多视角相机采样（训练时随机，评估时固定）
    - 分布式数据分片
    - 图像预处理和条件准备
    
    Args:
        cfg: 配置对象，需包含：
            - cfg.data.train: 训练数据配置（batch_size, dir, n_view, yaw_range 等）
            - cfg.data.eval: 评估数据配置（batch_size, dir, n_view, yaw 等）
            - cfg.renderer.resolution: 渲染分辨率
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 加速器，提供分布式信息
    
    Returns:
        tuple: (train_loader, eval_loader)
            - train_loader: 训练数据加载器（eval_only 时为 None）
            - eval_loader: 评估数据加载器
    """
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig
    
    # ---- 构建训练相机配置 ----
    # 训练时相机参数在指定范围内随机采样，增加数据多样性
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,                    # 每个样本采样的视角数
        yaw_range=list(cfg.data.train.yaw_range),        # 偏航角范围 [min, max]
        pitch_range=list(cfg.data.train.pitch_range),    # 俯仰角范围 [min, max]
        r_range=list(cfg.data.train.r_range),            # 相机距离范围 [min, max]
        fov_range=list(cfg.data.train.fov_range),        # 视场角范围 [min, max]
    )
    
    # ---- 构建评估相机配置 ----
    # 评估时使用固定相机参数，确保结果可比较
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,    # 评估视角数
        yaw=cfg.data.eval.yaw,          # 固定偏航角
        pitch=cfg.data.eval.pitch,      # 固定俯仰角
        r=cfg.data.eval.r,              # 固定相机距离
        fov=cfg.data.eval.fov,          # 固定视场角
    )
    
    # ---- 构建完整数据配置 ----
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,           # 训练批次大小
        eval_batch_size=cfg.data.eval.batch_size,       # 评估批次大小
        width=cfg.renderer.resolution,   # 渲染宽度
        height=cfg.renderer.resolution,  # 渲染高度
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    # ---- 创建 DataModule 并设置分布式 ----
    dm = TrellisDataModule(
        dm_cfg, 
        num_replicas=accelerator.num_processes,  # 分布式进程数
        rank=accelerator.process_index           # 当前进程排名
    )
    dm.setup()

    # ---- 返回 DataLoader ----
    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader


# =====================================================================
# Rollout 辅助函数（模块级，避免嵌套）
# =====================================================================

def _predict_cond_velocity(pipeline, coords, feats, t_batch, cond_emb):
    """
    纯条件 velocity 预测（用于 checkpoint 包裹）。
    
    Args:
        pipeline: 生成 pipeline
        coords: (N,4) 稀疏坐标
        feats: (N,C) 当前特征
        t_batch: (B,) 时间步 batch
        cond_emb: (B,S,C) 条件嵌入
    
    Returns:
        (N,C) velocity 预测
    """
    x_t = SparseTensor(coords=coords, feats=feats.detach())
    out = pipeline.sparse_sampling_step(x_t, t_batch, cond_emb, None, 0.0)
    return out.feats  # (N,C)


def _compute_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    latents: torch.Tensor,
    t_norm: float,
    reg_type: str,
    weight_mode: str = "uniform",
) -> Tuple[torch.Tensor, float]:
    """
    计算正则化 loss（VSD / KL）。
    
    Args:
        x0_student: (N,C) 学生模型预测的 x0
        x0_teacher: (N,C) 教师模型预测的 x0（无梯度）
        latents: (N,C) 当前步的 x_t，VSD 模式下用于梯度注入
        t_norm: 归一化时间步 (0~1)
        reg_type: "vsd" | "kl"
        weight_mode: "uniform" | "t" | "ada"
    
    Returns:
        (loss, metric): loss 用于反向传播，metric 用于日志
    """
    diff = x0_student - x0_teacher  # (N,C)
    
    # ---- 加权 ----
    if weight_mode == "t":
        diff = t_norm * diff  # (N,C)
    elif weight_mode == "ada":
        diff = diff / (x0_teacher.abs().mean() + 0.01).detach()  # (N,C)
    # else: uniform, 不加权
    
    # ---- 计算 loss ----
    if reg_type == "vsd":
        metric = 0.5 * (diff ** 2).mean().item()
        loss = SpecifyGradient.apply(latents, diff)  # 伪 loss，梯度注入
    elif reg_type == "kl":
        var = t_norm ** 2 + 1e-3
        loss = (0.5 * diff ** 2 / var).mean()
        metric = loss.item()
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")
    
    return loss, metric


# =====================================================================
# Rollout - 核心采样循环（训练/评估共用）
# =====================================================================

def rollout_sparse(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    稀疏特征去噪采样（SLAT Stage 2）。
    
    核心流程: x_T (噪声) → 迭代去噪 → x_0 (特征) → 反归一化
    
    Args:
        state: TrellisState 状态对象，包含条件编码、坐标等
        cfg: 配置对象，cfg.reg.type ("none"|"vsd"|"kl"), cfg.reg.weight_mode
        system: 系统组件（pipeline、renderer 等）
        device: 运行设备
        generator: 随机数生成器（用于可复现性）
        is_training: 是否为训练模式
    
    Returns:
        dict: "latents" (SparseTensor), "coords", "reg_loss", "reg_metric"
    """
    pipeline = system.pipeline
    _, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 1. 初始化 ----
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B,S,C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B,S,C)
    
    assert state.coords is not None, "state.coords 缺失"
    generator = generator or torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    latents = pipeline.init_latents(
        coords=state.coords,
        in_channels=pipeline.pipe.models['slat_flow_model'].in_channels,
        generator=generator
    ).feats  # (N,C)
    
    # ---- 2. Scheduler 配置 ----
    scheduler = pipeline.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    cfg_min, cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]
    
    # ---- 3. 正则化配置 ----
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_enabled = reg_type != "none" and is_training
    
    reg_loss_sum = 0.0
    reg_metric_sum = 0.0
    
    # ---- 4. 去噪循环 ----
    steps = list(scheduler.timesteps)[:-1]
    steps_iter = tqdm(steps, desc="Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for t in steps_iter:
        t_val = float(t.item())
        t_norm = t_val / 1000.0
        use_cfg = cfg_min <= t_val <= cfg_max
        B = cond_emb.shape[0]
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)
        
        # ---- cond 预测（训练时 checkpoint，推理时 no_grad）----
        if is_training:
            cond_pred = checkpoint(
                _predict_cond_velocity, pipeline, state.coords, latents,
                t_batch, cond_emb, use_reentrant=False
            )  # (N,C)
        else:
            with torch.no_grad():
                cond_pred = _predict_cond_velocity(
                    pipeline, state.coords, latents, t_batch, cond_emb
                )  # (N,C)
        
        # ---- uncond 预测（始终 no_grad）+ CFG 混合 ----
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_cond_velocity(
                    pipeline, state.coords, latents, t_batch, uncond_emb
                )  # (N,C)
            velocity = mix_cfg(cond_pred, uncond_pred, slat_guidance, uncond_mode=True)  # (N,C)
        else:
            velocity = cond_pred  # (N,C)
        
        # ---- 正则化（VSD / KL）----
        if reg_enabled:
            with pipeline.disable_lora_context(), torch.no_grad():
                teacher_cond = _predict_cond_velocity(
                    pipeline, state.coords, latents, t_batch, cond_emb
                )  # (N,C)
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_cond_velocity(
                        pipeline, state.coords, latents, t_batch, uncond_emb
                    )  # (N,C)
                    teacher_vel = mix_cfg(teacher_cond, teacher_uncond, slat_guidance, uncond_mode=True)  # (N,C)
                else:
                    teacher_vel = teacher_cond  # (N,C)
            
            x0_stu = latents - t_norm * velocity       # (N,C)
            x0_tea = latents - t_norm * teacher_vel    # (N,C)
            
            reg_loss, reg_metric = _compute_regularization(
                x0_stu, x0_tea, latents, t_norm,
                reg_type=reg_type, weight_mode=weight_mode
            )
            reg_loss_sum = reg_loss_sum + reg_loss
            reg_metric_sum = reg_metric_sum + reg_metric
        
        # ---- Scheduler 步进 ----
        x_t = SparseTensor(coords=state.coords, feats=latents)
        v_pred = SparseTensor(coords=state.coords, feats=velocity)
        latents = scheduler.step(v_pred, t, x_t).prev_sample.feats  # (N,C)
    
    # ---- 5. 反归一化 ----
    norm = pipeline.pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1,C)
    mean = torch.tensor(norm['mean'])[None].to(device) # (1,C)
    latents = latents * std + mean  # (N,C)
    
    # ---- 返回 ----
    num_steps = max(1, len(steps))
    return {
        "latents": SparseTensor(coords=state.coords, feats=latents),
        "coords": state.coords,
        "reg_loss": reg_loss_sum / num_steps if reg_enabled else None,
        "reg_metric": reg_metric_sum / num_steps if reg_enabled else None,
    }


# =====================================================================
# 渲染工具函数 - Mesh 渲染
# =====================================================================

def decode_and_render_mesh(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # TrellisMeshRasterizer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Mesh 并渲染多视角图像。
    
    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: Mesh 渲染器实例
        device: 运行设备
    
    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "normal": (B,V,H,W,3) 法线图
            - "depth": (B,V,H,W,1) 深度图
            - "meshes": list[len=B] of MeshExtractResult
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['mesh'])  # dict
    meshes = outputs['mesh']  # list[len=B] of MeshExtractResult
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    batch_size, num_views = extr_all.shape[:2]  # (), ()
    
    # ---- 逐样本逐视角渲染 ----
    all_renders: Dict[str, List[torch.Tensor]] = {}
    
    for i, mesh in enumerate(meshes):
        view_renders: Dict[str, List[torch.Tensor]] = {}
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)
            
            # Mesh 渲染器返回 dict of (H,W,C)
            render_out = renderer.render(mesh, ext_iv, intr_iv)  # dict
            
            for k, val in render_out.items():
                view_renders.setdefault(k, []).append(val)  # (H,W,C)
        
        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        for k, v_list in view_renders.items():
            stacked = torch.stack(v_list, dim=0)  # (V,H,W,C)
            all_renders.setdefault(k, []).append(stacked)
    
    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {}
    for k, b_list in all_renders.items():
        result[k] = torch.stack(b_list, dim=0)  # (B,V,H,W,C)
    
    result["meshes"] = meshes  # 保留 mesh 供导出
    return result


# =====================================================================
# 渲染工具函数 - Gaussian Splatting 渲染
# =====================================================================

def decode_and_render_gs(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # GaussianRenderer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Gaussian Splatting 并渲染多视角图像。
    
    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: GS 渲染器实例
        device: 运行设备
    
    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "gaussians": list[len=B] of Gaussian 对象
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['gaussian'])  # dict
    gaussians = outputs['gaussian']  # list[len=B] of Gaussian
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    _, num_views = extr_all.shape[:2]  # (), ()
    
    # ---- 逐样本逐视角渲染 ----
    all_colors: List[torch.Tensor] = []
    
    for i, gs in enumerate(gaussians):
        view_colors: List[torch.Tensor] = []
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)
            
            # Detach 一些变量，用于稳定训练
            gs._xyz = gs._xyz.detach()
            gs._rotation = gs._rotation.detach()
            gs._scaling = gs._scaling.detach()
            gs._opacity = gs._opacity.detach()
            gs._features_dc = gs._features_dc.detach()

            # GS 渲染器返回 color: (C,H,W)
            render_out = renderer.render(gs, ext_iv, intr_iv)  # dict
            color = render_out['color']  # (C,H,W)
            color = color.permute(1, 2, 0)  # (H,W,C)
            view_colors.append(color)
        
        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        stacked = torch.stack(view_colors, dim=0)  # (V,H,W,C)
        all_colors.append(stacked)
    
    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {
        "color": torch.stack(all_colors, dim=0),  # (B,V,H,W,C)
        "gaussians": gaussians,  # 保留 GS 供其他用途
    }
    return result




# =====================================================================
# 训练 - 核心训练循环
# =====================================================================

def train_edit4shape(
    system: System,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    单步训练函数（核心训练循环）。
    
    实现了基于 Flow Matching 的 3D 生成训练流程：
    1. 执行 rollout_sparse 生成潜变量（带梯度）
    2. 解码潜变量得到 3D 表示（mesh/GS）
    3. 渲染多视角图像
    4. 计算损失（重建损失 + guidance 损失）
    5. 反向传播更新参数
    
    训练策略：
    - 使用 Gradient Checkpointing 减少显存占用
    - 每步使用不同的随机种子以增加数据多样性
    - 支持梯度累积（通过 accelerator 配置）
    
    Args:
        system: 系统组件（pipeline、renderer、optimizer）
        state: TrellisState 状态对象（已挂载 batch 数据）
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
    
    Returns:
        tuple: (训练日志字典, 渲染输出字典)
    """
    device = accelerator.device
    optimizer = system.optimizer

    # =====================================================
    # 1. 准备阶段
    # =====================================================
    # 注意：optimizer.zero_grad() 移到反向传播后，配合 accelerator.accumulate() 使用
    
    # =====================================================
    # 2. 显式结构生成 (Dense Sampling)
    # 与评估流程保持一致，先生成稠密坐标再进入 SLAT 采样
    # =====================================================
    pipeline = system.pipeline
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()  # () - 解析结构采样步数
    with torch.no_grad():
        cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4) - 稠密采样得到稀疏坐标
    state.coords = coords  # (N,4) - 挂载坐标供后续 rollout 使用
    
    # =====================================================
    # 3. 训练核心逻辑（在 TrainModeGuard 下执行）
    # Pipeline 加载时默认将所有模型设为 eval（见 base.py Pipeline.__init__）
    # 需要将 flow model 和解码器切换到 train 模式以启用可微分路径
    # =====================================================
    pipe_models = pipeline.pipe.models
    with TrainModeGuard(
        pipe_models['slat_flow_model'],      # 我们训练的目标模型
        pipe_models['slat_decoder_mesh'],    # 使 mesh_extractor(x, training=True) 启用可微分 FlexiCubes
        pipe_models['slat_decoder_gs'],      # GS 解码器保持一致性
    ):
        # ---- Rollout：执行稀疏特征采样 ----
        # 每步使用不同的随机种子，确保训练数据多样性
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
        
        rollout_out = rollout_sparse(
            state, cfg, system, device, 
            generator=generator, 
            is_training=True,  # 启用梯度和 Checkpointing
        )
        
        # latents 是 SparseTensor，其 feats 包含完整的计算图
        latents = rollout_out["latents"]
        
        # ---- 解码 & 渲染 ----
        # 根据 renderer 类型选择解码格式并渲染多视角图像
        renderer_type = cfg.renderer.type
        
        if renderer_type == "gs":
            render_out = decode_and_render_gs(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color": (B,V,H,W,C), "gaussians": list
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        
        comp_rgb = render_out["color"]  # (B,V,H,W,C) - 渲染的颜色图
        state.views_generated.image_tensor = comp_rgb  # 挂载生成图用于可视化
        
        # ---- FlowEdit Guidance & 损失计算 ----
        # 使用 system.guidance（在 build_system 中初始化，支持 local/http 后端）
        guidance_result = system.guidance.compute_guidance(
            comp_rgb, 
            state.views_conditioned.image_pils,
            rank=accelerator.process_index,
        )
        state.views_edited.image_tensor = guidance_result.edited_imgs  # 存入 state
        
        # ---- 统一 Loss 管理 ----
        losses = LossDict(device=accelerator.device)  # 统一到训练设备
        guidance_weights = system.guidance.get_loss_weights()
        
        # Guidance losses（权重统一在此应用）
        losses.add("ssim", guidance_result.loss_ssim, weight=guidance_weights["ssim"])
        losses.add("lpips", guidance_result.loss_lpips, weight=guidance_weights["lpips"])
        losses.add("latent_mse", guidance_result.loss_latent_mse, weight=guidance_weights["latent_mse"])
        
        # VSD/KL 正则化 loss（reg_loss 用于梯度注入，reg_metric 用于日志）
        losses.add("reg", rollout_out.get("reg_loss"), weight=cfg.train.loss.reg)
        
        # ---- 反向传播 ----
        total_loss = losses.total()
        accelerator.backward(total_loss)
        
        # 仅在梯度同步时（累积完成）执行优化器步骤
        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()
    # TrainModeGuard 退出后自动恢复模型的原始模式
    
    # 日志自动生成
    logs = losses.to_logs()
    # VSD 的 reg_loss 是伪 loss（始终为 1），使用 reg_metric 记录实际值
    if rollout_out.get("reg_metric") is not None:
        logs["loss/reg"] = rollout_out["reg_metric"]
    return logs


# =====================================================================
# 评估 - 推理与可视化保存
# =====================================================================

@torch.no_grad()
def evaluate(
    system: System,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """
    评估函数：执行推理并保存可视化结果。
    
    完整的评估流程：
    1. 从图像提取条件编码
    2. 执行 Dense Sampling 生成稀疏结构
    3. 执行 Sparse Sampling 生成特征
    4. 解码为 3D 表示（mesh 或 GS）
    5. 渲染多视角图像并保存
    6. 导出 mesh 文件
    
    输出目录结构：
    visuals_eval_dir/
    └── epoch_{N}/
        ├── sample_name_1/
        │   ├── color.png      # 渲染的颜色图
        │   ├── normal.png     # 渲染的法线图（mesh 模式）
        │   └── mesh.obj       # 导出的网格文件
        └── sample_name_2/
            └── ...
    
    Args:
        system: 系统组件
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 可视化输出目录
    
    Returns:
        dict: 评估日志字典
    """
    if eval_loader is None:
        return {}
    
    pipeline = system.pipeline
    # 获取采样参数
    ss_steps, _, slat_steps, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 创建 VisualIO 用于保存 ----
    visual_io = VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution)
        
    # =====================================================
    # 使用 EvalModeGuard 确保所有模型处于评估模式
    # =====================================================
    pipe_models = pipeline.pipe.models
    with EvalModeGuard(
        pipe_models['slat_flow_model'],
        pipe_models['slat_decoder_mesh'],
        pipe_models['slat_decoder_gs'],
    ):
        # =====================================================
        # 遍历评估数据集
        # =====================================================
        for batch_idx, batch in enumerate(eval_loader):
            # 每个 batch 创建独立状态，避免跨 batch 残留
            state = TrellisState()
            
            # ---- 挂载 batch 数据 ----
            state.attach_batch(batch, pipeline=pipeline)  # 自动从 image_pils 生成条件编码并挂载
            
            # =====================================================
            # Step 2: Dense Sampling（结构生成）
            # 根据条件生成稀疏 3D 坐标
            # =====================================================
            cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
            coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
            state.coords = coords  # 保存到 state 供后续使用

            # =====================================================
            # Step 3: Sparse Sampling（特征生成）
            # 在稀疏坐标上执行去噪采样
            # =====================================================
            rollout_out = rollout_sparse(state, cfg, system, accelerator.device)  # dict
            latents = rollout_out["latents"]  # SparseTensor

            # =====================================================
            # Step 4: 解码 & 渲染 & 保存
            # 根据 renderer 类型选择解码格式
            # =====================================================
            renderer_type = cfg.renderer.type
            
            if renderer_type == "gs":
                # ---- Gaussian Splatting 分支 ----
                render_out = decode_and_render_gs(
                    latents, state.cameras, pipeline, system.renderer, accelerator.device
                )  # dict with "color": (B,V,H,W,C), "gaussians": list
            else:
                # ---- Mesh Rasterizer 分支 ----
                render_out = decode_and_render_mesh(
                    latents, state.cameras, pipeline, system.renderer, accelerator.device
                )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
            
            # ---- 保存结果 ----
            state.views_generated.image_tensor = render_out["color"]  # 挂载渲染结果供保存使用
            if accelerator.is_main_process:
                visual_io.save_batch_eval(
                    state=state,
                    epoch=epoch,
                    render_out=render_out,
                    pipeline=pipeline,
                    export_mesh=(renderer_type != "gs"),
                )

    return {"eval_done": 1.0}


# build_run_paths, CheckpointIO, ModeGuard, TrainModeGuard, EvalModeGuard, MetricLogger, VisualIO 
# 已从 base.py / utils.py 导入，此处不再重复定义


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    完整的训练/评估流程：
    1. 解析配置文件
    2. 设置环境与随机种子
    3. 初始化 Accelerator（分布式/混合精度）
    4. 创建运行目录
    5. 构建数据加载器
    6. 构建系统组件（pipeline, renderer, optimizer）
    7. 加载检查点（如有）
    8. 执行训练循环或评估
    
    配置文件示例：
        python -m edit4shape.systems.trellis --config=configs/trellis.py
    
    关键配置项：
        - cfg.eval_only: True 时仅执行评估
        - cfg.num_epochs: 训练总 epoch 数
        - cfg.eval_freq: 评估频率（每 N 个 epoch）
        - cfg.save_freq: 保存检查点频率
        - cfg.checkpoint: 恢复训练的检查点路径
    
    Args:
        argv: 命令行参数（由 absl.app.run 传入，本函数不使用）
    """
    del argv  # absl.app.run 会传入 argv；本函数不使用
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
    # 配置混合精度训练和梯度累积
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,  # "no", "fp16", "bf16"
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq)

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式（仅评估不训练）
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator, 
            epoch=start_epoch, 
            global_step=global_step, 
            eval_loader=eval_loader, 
            visuals_eval_dir=visuals_eval_dir
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环
    # =====================================================
    # 初始化训练日志记录器（自动处理梯度累积）
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        # 设置分布式采样器的 epoch（确保各进程数据不同）
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            
            # 创建新状态并挂载 batch 数据
            state = TrellisState()
            state.attach_batch(batch, pipeline=system.pipeline)  # 挂载所有数据
            
            # 使用 accumulate 上下文管理器处理梯度累积
            with accelerator.accumulate(system.pipeline.pipe.models['slat_flow_model']):
                train_log = train_edit4shape(system, state, cfg, accelerator, epoch, global_step)
            
            # 仅主进程按频率保存三联图
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(
                    state=state,
                    epoch=epoch,
                    step=global_step,
                )
            
            # 自动累积并在 sync_gradients 时发射平均日志
            train_logger.log_step(train_log, len(batch['image_pils']), global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator, 
                epoch=epoch, 
                global_step=global_step, 
                eval_loader=eval_loader, 
                visuals_eval_dir=visuals_eval_dir
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
