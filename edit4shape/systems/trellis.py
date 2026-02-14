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
4. trellis_forward: 共享的前向传播（Dense Sampling → Rollout → Render）
5. evaluate: 评估循环，生成 mesh 并保存可视化结果
6. main: 训练主循环（内联 guidance/loss/backward）

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
from contextlib import nullcontext
from PIL import Image
import numpy as np
import requests
import yaml
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from tqdm import tqdm

# =====================================================================
# TRELLIS 参考实现路径设置（必须在 trellis 相关导入之前）
# 将 TRELLIS 参考代码目录加入 Python 路径，以便导入其模块
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.datasets.trellis import TrellisDataConfig, TrellisDataModule
from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import rollout_sparse, rollout_sparse_sde

# 使用 absl 的 config_flags 管理配置文件
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
# 包含 coords (坐标) 和 feats (特征) 两个主要属性
from trellis.modules.sparse import SparseTensor

# =====================================================================
# Guidance 模块
# =====================================================================

from edit4shape.guidance import create_guidance


# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    BaseState,
    System,
    CheckpointIO,
    build_run_paths,
    compute_guidance_device,
)
from edit4shape.systems.utils import MetricLogger, VisualIO


# TrellisState 从 edit4shape.generators.trellis.state 导入
# rollout_sparse 从 edit4shape.generators.trellis.rollout 导入


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict, 
    accelerator: Accelerator,
    guidance_factory: callable,
) -> System:
    """
    构建完整的 Trellis 系统。
    
    根据配置创建所有必要的组件，包括：
    1. Pipeline: 负责条件编码、结构/特征采样、解码的核心生成管道
    2. Renderer: 将 3D 表示渲染为 2D 图像的渲染器
    3. Guidance: 训练时的指导模块（如 SDS loss）
    4. Strategy: 训练策略（LoRA / Full / Frozen）
    5. Optimizer: 模型参数优化器
    
    Args:
        cfg: 完整配置对象，包含以下关键配置：
            - cfg.renderer: 渲染器配置（类型、分辨率、近远裁剪面等）
            - cfg.train.mode: 训练模式 ("lora" | "full" | "frozen")
            - cfg.train.optimizer: 优化器配置
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数。
                          同步版本传入 create_guidance，流水线并行版本传入 create_guidance_pp。
    
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
        from edit4shape.renderers.base_renderer import RenderConfig
        render_cfg = RenderConfig(
            resolution=cfg.renderer.resolution,  # 渲染分辨率 (像素)
            near=cfg.renderer.near,  # 近裁剪面距离
            far=cfg.renderer.far,    # 远裁剪面距离
            ssaa=cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            bg_color=cfg.renderer.bg_color,  # 背景色模式
        )
        renderer = GaussianRenderer(config=render_cfg, device=str(accelerator.device))
    else:
        # ---- Mesh 光栅化渲染器 (nvdiffrast) ----
        # 优势：支持精确的几何渲染，法线/深度图质量高
        # 适用场景：训练、精细渲染
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer
        from edit4shape.renderers.base_renderer import RenderConfig
        renderer_cfg = RenderConfig(
            resolution=cfg.renderer.resolution,  # 渲染分辨率 (像素)
            ssaa=cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            near=cfg.renderer.near,  # 近裁剪面距离
            far=cfg.renderer.far,    # 远裁剪面距离
            bg_color=cfg.renderer.bg_color,  # 背景色模式
        )
        renderer = TrellisMeshRasterizer(config=renderer_cfg, device=str(accelerator.device))

    # ---- 3. 构建 Strategy（训练 & 推理都需要）----
    # Strategy 负责将模型注册到 accelerator（accelerator.prepare），
    # 这样 accelerator.load_state() 才能正确恢复 checkpoint 的模型权重。
    # 因此即使 eval_only 也必须创建 strategy。
    from edit4shape.generators.trellis.training_adpter import (
        register_sparse_linear_with_peft,
        inject_lora_to_slat,
        build_optimizer_for_slat,
        TrellisFullFinetuneStrategy,
        TrellisLoRAStrategy,
        TrellisFrozenStrategy,
    )
    
    train_mode = cfg.train.get("mode", "full")  # 默认全参微调
    train_device = accelerator.device
    teacher_device = compute_guidance_device(accelerator.device)
    
    # 根据训练模式创建策略
    if train_mode == "full":
        strategy = TrellisFullFinetuneStrategy(
            pipeline, train_device, teacher_device, cfg.pretrained.model
        )
    elif train_mode == "lora":
        register_sparse_linear_with_peft()
        inject_lora_to_slat(pipeline, cfg.lora)
        strategy = TrellisLoRAStrategy(pipeline, train_device, teacher_device)
    else:
        strategy = TrellisFrozenStrategy(pipeline, train_device, teacher_device)
    
    strategy.setup()

    # ---- 4. 构建 Guidance 和 Optimizer（仅训练时需要）----
    guidance = None
    optimizer = None

    if not cfg.eval_only:
        # 4a. 使用工厂函数创建 Guidance
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        # 4b. 启用 slat_flow_model 的 Gradient Checkpointing（节省显存）
        slat_model = pipeline._resolve_slat_flow_module()
        for block in slat_model.blocks:
            block.use_checkpoint = True
        
        # 4b-2. 也为 slat_decoder_gs 启用 Gradient Checkpointing（避免 decode 时 OOM）
        decoder_gs = pipeline.pipe.models.get('slat_decoder_gs')
        if decoder_gs is not None and hasattr(decoder_gs, 'blocks'):
            for block in decoder_gs.blocks:
                block.use_checkpoint = True
        
        # 4c. 为学生模型创建优化器
        optimizer = build_optimizer_for_slat(strategy.student, cfg.train.optimizer)

    return System(
        pipeline=pipeline, 
        renderer=renderer, 
        guidance=guidance, 
        optimizer=optimizer, 
        strategy=strategy,
    )


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
        adaptive_distance=cfg.data.train.adaptive_distance,
    )
    
    # ---- 构建评估相机配置 ----
    # 评估时使用固定相机参数，确保结果可比较
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,    # 评估视角数
        yaw=cfg.data.eval.yaw,          # 固定偏航角
        pitch=cfg.data.eval.pitch,      # 固定俯仰角
        r=cfg.data.eval.r,              # 固定相机距离
        fov=cfg.data.eval.fov,          # 固定视场角
        adaptive_distance=cfg.data.eval.adaptive_distance,
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
            
            # Mesh 渲染器返回 RenderOutput
            render_out = renderer.render(mesh, ext_iv, intr_iv)  # RenderOutput
            render_dict = {
                "color": render_out.color,  # (H,W,3)
                "normal": render_out.normal,  # (H,W,3)
                "depth": render_out.depth,  # (H,W)
                "mask": render_out.mask,  # (H,W)
            }
            
            for k, val in render_dict.items():
                if val is None:
                    continue
                view_renders.setdefault(k, []).append(val)  # (H,W,C) or (H,W)
        
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
            
            # #  限制梯度，用于稳定训练,参考 ml-sharp
            # gs._xyz =  gs._xyz.detach() #(1 - 0.001) * gs._xyz.detach() + 0.001 * gs._xyz
            # gs._rotation =  gs._rotation.detach() #(1 - 0.1) * gs._rotation.detach() + 0.1 * gs._rotation
            # gs._scaling =  gs._scaling.detach() #(1 - 0.1) * gs._scaling.detach() + 0.1 * gs._scaling
            # gs._opacity =  gs._opacity.detach() #(1 - 0.1) * gs._opacity.detach() + 0.1 * gs._opacity

            # GS 渲染器返回 RenderOutput
            render_out = renderer.render(gs, ext_iv, intr_iv)  # RenderOutput
            color = render_out.color  # (H,W,3)
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
# 前向传播 - 共享的 Trellis 前向逻辑
# =====================================================================

def trellis_forward(
    system: System,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Trellis 前向传播：Dense Sampling → Rollout → Decode → Render
    
    抽取共享的前向逻辑，供训练、评估和流水线并行版本复用。
    
    注意：调用此函数时需要在外层使用 TrainModeGuard（训练时）或 EvalModeGuard（评估时）。
    
    Args:
        system: 系统组件（pipeline、renderer）
        state: TrellisState 状态对象（已挂载 batch 数据，含条件编码）
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数（用于随机种子）
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B,V,H,W,C) 渲染图像
            - "meshes" 或 "gaussians": 3D 表示（用于导出）
        
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.regularization: 挂载 reg_loss
        - state.views_generated.image_tensor: 挂载渲染图像
    """
    pipeline = system.pipeline
    
    # ---- 1. Dense Sampling（结构生成）----
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()
    with torch.no_grad():
        cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
    state.coords = coords  # (N,4) - 挂载坐标供后续 rollout 使用
    
    # ---- 2. Rollout：执行稀疏特征采样（挂载 state.features.slat 和 state.regularization）----
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    
    # 根据配置选择 ODE 或 SDE rollout
    # 注意：推理时强制使用 ODE（确定性），训练时可选
    use_sde = is_training and cfg.rollout.type == "sde"
    
    if use_sde:
        rollout_sparse_sde(
            state, cfg, system, device,
            generator=generator,
            is_training=is_training,
            track_trajectory=False,
        )
    else:
        rollout_sparse(
            state, cfg, system, device,
            generator=generator,
            is_training=is_training,
        )
    latents = state.features.slat  # SparseTensor (挂载于 rollout)
    
    # 释放 rollout 阶段产生的显存碎片，为 decode 腾出空间
    torch.cuda.empty_cache()
    
    # ---- 3. 解码 & 渲染 ----
    renderer_type = cfg.renderer.type
    
    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, system.renderer, device
        )  # dict with "color": (B,V,H,W,C), "gaussians": list
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, system.renderer, device
        )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        render_out["color"] = render_out["normal"]
    
    state.views_generated.image_tensor = render_out["color"]  # (B,V,H,W,C) 挂载生成图用于可视化
    
    return render_out


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
    
    注意：
        accelerator.prepare() 会为模型附加 autocast(bf16) 上下文，其中 nn.Linear
        （包括 SparseLinear）的输出会被提升为 bf16。而 spconv 在 eval 模式下走
        ops.implicit_gemm 推理路径，该路径的 ConvTunerSimple 无法为 bf16 输入
        找到合适的 GEMM 算法，导致 RuntimeError。
        因此评估前需要临时卸下 DDP/autocast 包装，使用原始模型推理。
        （参考 TRELLIS 原始代码：训练用 self.training_models，推理用 self.models）
    
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
    visual_io = VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution, accelerator=accelerator)
    
    # =====================================================
    # 使用 EvalModeGuard 确保所有模型处于评估模式
    # =====================================================
    pipe_models = pipeline.pipe.models
    # ★ TRELLIS 风格：推理时换回原始模型（无 DDP / autocast(bf16)）
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
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
            
            # ---- 使用共享的 trellis_forward 执行前向传播 ----
            render_out = trellis_forward(
                system, state, cfg, accelerator.device, global_step, is_training=False
            )
            
            # ---- 保存结果（所有进程都保存各自处理的样本）----
            renderer_type = cfg.renderer.type
            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=(renderer_type != "gs"),
            )

    return {"eval_done": 1.0}


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
    # Step 2: 初始化 Accelerator（含 wandb 日志）
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if cfg.use_wandb else None,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    
    # 初始化 wandb trackers
    if cfg.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(
        visuals_train_dir, 
        target_h=cfg.renderer.resolution, 
        vis_freq=vis_freq,
        accelerator=accelerator,
    )

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
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
    
    pipeline = system.pipeline
    pipe_models = pipeline.pipe.models
    
    def _compute_loss_and_backward(state: TrellisState) -> Dict[str, Any]:
        """
        计算 loss 并反向传播。
        
        所有需要的数据都已挂载在 state 中：
        - state.guidance: 包含 loss（主 loss）和 loss_dict（细分 loss）
        - state.regularization: 包含 reg_loss
        """
        # ---- 计算总 loss ----
        # guidance.loss 在 Guidance 设备上，需要移到训练设备
        guidance_loss = state.guidance.loss.to(accelerator.device) * cfg.train.loss.guidance  # ()
        total = guidance_loss  # ()
        if state.regularization.reg_loss is not None:
            total = total + cfg.train.loss.reg * state.regularization.reg_loss  # ()
        
        # ---- 反向传播 ----
        accelerator.backward(total)
        
        # ---- 构建日志（直接复用 loss_dict）----
        logs = {f"loss/{k}": v.item() for k, v in (state.guidance.loss_dict or {}).items() if v is not None}
        logs["loss/total"] = total.item()

        if state.regularization.reg_loss is not None:
            logs["loss/reg"] = state.regularization.reg_loss.item()
        return logs

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        # 设置分布式采样器的 epoch（确保各进程数据不同）
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            
            # 使用 accumulate 上下文管理器处理梯度累积
            with accelerator.accumulate(pipe_models['slat_flow_model']):
                # ---- 在 TrainModeGuard 下执行训练 ----
                with TrainModeGuard(
                    pipe_models['slat_flow_model'],
                    pipe_models['slat_decoder_mesh'],
                    pipe_models['slat_decoder_gs'],
                ):
                    # 创建新状态并挂载 batch 数据
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=pipeline)  # 挂载所有数据
                    
                    # ---- 前向传播 ----
                    render_out = trellis_forward(
                        system, state, cfg, accelerator.device, global_step, is_training=True
                    )
                    comp_rgb = render_out["color"]  # (B,V,H,W,C)
                    
                    # ---- Guidance ----
                    guidance_result = system.guidance.compute_guidance(
                        comp_rgb, 
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(guidance_result)  # 挂载到 state
                    
                    # ---- Loss & Backward ----
                    train_log = _compute_loss_and_backward(state)
                
                # ---- 优化器步进 ----
                if accelerator.sync_gradients:
                    system.optimizer.step()
                    system.optimizer.zero_grad()
            
            # 仅主进程按频率保存可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(
                    state=state,
                    epoch=epoch,
                    step=global_step,
                    pipe=system.guidance.pipe if system.guidance else None,
                    n_progress_samples=cfg.freq.save.progress_samples,
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
