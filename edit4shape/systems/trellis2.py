"""
Trellis2 训练系统（适配 TRELLIS.2 双阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 生成系统训练，支持从单张图像生成 3D 模型。
核心流程：
- Stage 1 (Shape): 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss
- Stage 2 (Tex): Tex Rollout -> MeshWithVoxel -> RGB 渲染 -> Guidance Loss

特性：
- 双阶段训练：Shape 阶段用 Normal 渲染监督几何，Tex 阶段用 RGB 渲染监督纹理
- 每个 batch 分两步计算 Guidance Loss
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape / rollout_tex: 执行 Shape/Tex 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（生成 Normal）
5. trellis2_tex_forward: Tex 阶段前向传播（生成 RGB）
6. evaluate: 评估循环，生成 mesh 并保存可视化结果
7. main: 训练主循环（依次执行 Shape Guidance 和 Tex Guidance）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (mesh 渲染)
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
from typing import Any, ClassVar, Dict, Optional, Tuple, List, Literal

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
# TRELLIS.2 参考实现路径设置
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

# SparseTensor: TRELLIS.2 中用于表示稀疏 3D 特征的核心数据结构
from trellis2.modules.sparse import SparseTensor

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
    CheckpointIO,
    build_run_paths,
    SpecifyGradient,
)
from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO, LossDict

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape", "tex"]


# =====================================================================
# 从 training_adpter 导入 StageConfig
# =====================================================================
from edit4shape.generators.trellis2.training_adpter import StageConfig


# =====================================================================
# Trellis2 系统组件类
# =====================================================================

@dataclass
class StageSystem:
    """
    单个阶段的系统组件。
    
    封装 Shape 或 Tex 阶段的 model、optimizer、renderer 和配置。
    
    属性:
        model: Flow Model
        optimizer: 优化器
        renderer: 渲染器（Shape 用 Normal 渲染，Tex 用 RGB/PBR 渲染）
        config: StageConfig 配置
    """
    model: Any = None       # Flow Model
    optimizer: Any = None   # Optimizer
    renderer: Any = None    # Renderer（阶段专用）
    config: StageConfig = field(default_factory=StageConfig)


@dataclass
class Trellis2System:
    """
    Trellis2 双阶段系统。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（model, optimizer, renderer, config）
    - tex: Tex 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.shape.model      # Shape Flow Model
        system.shape.renderer   # Shape Renderer (Normal)
        system.tex.renderer     # Tex Renderer (RGB/PBR)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # 分阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    tex: StageSystem = field(default_factory=StageSystem)
    
    # 共享组件
    guidance: Any = None
    
    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """设置随机种子与确定性运行环境。"""
        import random
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def prepare_lora(self, cfg: Any, adapter: str = "base", **kwargs) -> "Trellis2System":
        """准备 LoRA 适配器"""
        for module in [self.pipeline, self.guidance]:
            if module is not None and hasattr(module, "set_adapter"):
                module.set_adapter(adapter)
        return self
    
    def prepare_optimizers(self, accelerator: Accelerator) -> "Trellis2System":
        """准备双阶段优化器（使用 accelerator.prepare）"""
        if self.shape.optimizer is not None:
            self.shape.optimizer = accelerator.prepare(self.shape.optimizer)
        if self.tex.optimizer is not None:
            self.tex.optimizer = accelerator.prepare(self.tex.optimizer)
        return self


# =====================================================================
# Trellis2State - Trellis2 专用状态类
# =====================================================================

@dataclass
class Trellis2State(BaseState):
    """
    Trellis2 生成过程的状态容器。
    
    扩展 BaseState 以支持 TRELLIS.2 的双阶段生成：
    - shape_slat: 几何阶段的稀疏潜变量
    - tex_slat: 纹理阶段的稀疏潜变量
    - subs: 解码中间结果（用于 tex 解码）
    
    属性说明:
        coords (torch.Tensor): 稀疏结构坐标，形状 (N, 4)。
                               N 为总点数 (batch_size * num_points)。
                               第 0 列为 batch 索引，后 3 列为 (x, y, z) 坐标。
        
        features (Trellis2State.Features): 特征容器。
            - shape_slat (SparseTensor): Shape 阶段输出的稀疏特征
            - tex_slat (SparseTensor): Tex 阶段输出的稀疏特征
            - subs (List[SparseTensor]): Shape 解码中间结果
        
        cameras (BaseState.Cameras): 相机参数容器。
            - c2w (torch.Tensor): (B, V, 4, 4) 相机到世界变换矩阵。
            - w2c (torch.Tensor): (B, V, 4, 4) 世界到相机变换矩阵。
            - intrinsics (torch.Tensor): (B, V, 3, 3) 内参矩阵。
            
        views_conditioned (BaseState.ViewsConditioned): 条件信息容器。
            - image_pils (List[PIL.Image]): 输入的条件图像列表。
            - cond_embed (torch.Tensor): (B, S, C) 条件嵌入 (DINOv3)。
            - uncond_embed (torch.Tensor): (B, S, C) 无条件嵌入 (用于 CFG)。
            
        views_generated (BaseState.ViewsGenerated): 生成结果容器。
            - image_tensor (torch.Tensor): (B, V, H, W, C) 渲染出的图像。
            
        views_edited (BaseState.ViewsEdited): 编辑结果容器。
            - image_tensor (torch.Tensor): (B, V, C, H, W) 经过 Guidance 编辑后的图像。
            
        regularization (Trellis2State.Regularization): 正则化信息容器。
            - reg_loss: 正则化 loss（用于反向传播）
            - reg_metric: 正则化 metric（用于日志记录）
            
        guidance (Trellis2State.Guidance): Guidance 结果容器。
            - loss_ssim: SSIM loss
            - loss_lpips: LPIPS loss
            - loss_latent_mse: Latent MSE loss
    """
    
    @dataclass
    class Features:
        """特征容器。存储 Shape 和 Tex 阶段的稀疏特征。"""
        shape_slat: Any = None  # SparseTensor, Shape 阶段输出
        tex_slat: Any = None    # SparseTensor, Tex 阶段输出
        subs: Any = None        # List[SparseTensor], Shape 解码中间结果
    
    @dataclass
    class Regularization:
        """正则化信息容器。存储 VSD/KL 正则化的 loss 和 metric。"""
        reg_loss: Any = None    # 正则化 loss（用于反向传播）
        reg_metric: Any = None  # 正则化 metric（用于日志记录）
    
    @dataclass
    class Guidance:
        """Guidance 结果容器。存储 FlowEdit 的各项 loss。"""
        loss_ssim: Any = None         # SSIM loss（标量张量）
        loss_lpips: Any = None        # LPIPS loss（标量张量）
        loss_latent_mse: Any = None   # Latent MSE loss（标量张量）
    
    # batch key -> state 属性的映射（类常量）
    _CAMERA_KEYS: ClassVar[List[str]] = ["c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions"]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]
    
    # ============== Trellis2 专用子状态容器 ==============
    features: Features = field(default_factory=Features)
    regularization: Regularization = field(default_factory=Regularization)
    guidance: Guidance = field(default_factory=Guidance)

    def attach_batch(self, batch: Dict[str, Any], pipeline: Any = None, resolution: int = 1024) -> "Trellis2State":
        """
        从数据批次中提取并挂载所有数据到 state。
        
        Args:
            batch: DataLoader 返回的批次数据，包含图像、相机参数等。
            pipeline: 必须提供，用于调用 prepare_image_conditions 从 image_pils 生成条件嵌入。
            resolution: 条件编码分辨率（512 或 1024）
        
        Returns:
            self: 支持链式调用
        """
        # ---- 1. views_conditioned（图像、路径、嵌入） ----
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])
        
        # 从 image_pils 生成条件编码
        if "image_pils" in batch and pipeline is not None:
            cond = pipeline.prepare_image_conditions(batch["image_pils"], resolution=resolution)
            self.views_conditioned.cond_embed = cond["cond"]  # (B, S, C)
            self.views_conditioned.uncond_embed = cond["neg_cond"]  # (B, S, C)
        
        # ---- 2. 指导信号 (Guidance 数据) ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]
        
        # ---- 3. 相机参数 ----
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])
        
        return self

    def attach_guidance_result(self, guidance_result: Any) -> "Trellis2State":
        """
        将 GuidanceResult 挂载到 state。
        
        Args:
            guidance_result: GuidanceResult 对象，包含编辑后图像和各项 loss。
        
        Returns:
            self: 支持链式调用
        """
        # Loss 挂载到 guidance
        self.guidance.loss_ssim = guidance_result.loss_ssim
        self.guidance.loss_lpips = guidance_result.loss_lpips
        self.guidance.loss_latent_mse = guidance_result.loss_latent_mse
        # 编辑后图像挂载到 views_edited
        self.views_edited.image_tensor = guidance_result.edited_imgs
        return self


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict, 
    accelerator: Accelerator,
    guidance_factory: callable,
) -> Trellis2System:
    """
    构建完整的 Trellis2 系统。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        Trellis2System: 包含所有组件的系统实例
    """
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, set_stage_trainable, build_optimizer_for_stage, register_sparse_linear_with_peft
    )
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 2. Renderer 配置 ----
    renderer_cfg = TrellisRendererConfig(
        resolution=cfg.renderer.resolution,
        ssaa=cfg.renderer.ssaa,
        near=cfg.renderer.near,
        far=cfg.renderer.far,
    )
    
    # ---- 3. 获取阶段配置（训练和评估都需要） ----
    shape_config = get_stage_config(pipeline_type, "shape")
    tex_config = get_stage_config(pipeline_type, "tex")
    
    # ---- 4. 构建 StageSystem ----
    shape_stage = StageSystem(
        config=shape_config,
        renderer=TrellisMeshRasterizer(cfg=renderer_cfg, device=device),
    )
    tex_stage = StageSystem(
        config=tex_config,
        renderer=TrellisMeshRasterizer(cfg=renderer_cfg, device=device),
    )
    
    # ---- 5. 训练模式：设置 model 和 optimizer ----
    guidance = None
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        register_sparse_linear_with_peft()
        set_stage_trainable(pipeline, pipeline_type, ["shape", "tex"])
        
        # 获取模型
        shape_stage.model = pipeline.get_flow_model(shape_config.model_stage, shape_config.flow_resolution)
        tex_stage.model = pipeline.get_flow_model(tex_config.model_stage, tex_config.flow_resolution)
        
        # 创建优化器
        optimizer_shape, optimizer_tex = build_optimizer_for_stage(
            pipeline, pipeline_type, ["shape", "tex"], cfg.train.optimizer
        )
        shape_stage.optimizer = optimizer_shape
        tex_stage.optimizer = optimizer_tex

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        tex=tex_stage,
        guidance=guidance,
    )


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练和评估的 DataLoader。
    
    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器
    
    Returns:
        tuple: (train_loader, eval_loader)
    """
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig
    
    # ---- 构建训练相机配置 ----
    # 训练时相机参数在指定范围内随机采样，增加数据多样性
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
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
# Rollout 辅助函数
# =====================================================================

def _predict_velocity(
    pipeline: Any,
    coords: torch.Tensor,
    feats: torch.Tensor,
    t: float,
    cond_emb: torch.Tensor,
    stage: Stage,
    resolution: int,
    shape_cond: Optional[SparseTensor] = None,
) -> torch.Tensor:
    """
    Velocity 预测（用于 checkpoint 包裹）。
    
    Args:
        pipeline: Trellis2RefAdapter
        coords: (N, 4) 稀疏坐标
        feats: (N, C) 当前特征
        t: 时间步标量，范围 [0, 1]
        cond_emb: (B, S, C) 条件嵌入
        stage: "shape" 或 "tex"
        resolution: 512 或 1024
        shape_cond: SparseTensor，tex 阶段需要的 shape 条件（已归一化）
    
    Returns:
        (N, C) velocity 预测
    """
    x_t = SparseTensor(coords=coords, feats=feats.detach())
    
    # t 已经是 0-1 范围，直接传给 sampling_step（内部会乘 1000）
    out = pipeline.sampling_step(
        x_t, t, cond_emb, stage, resolution, shape_cond=shape_cond
    )  # SparseTensor
    return out.feats  # (N, C)


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
        x0_student: (N, C) 学生模型预测的 x0
        x0_teacher: (N, C) 教师模型预测的 x0（无梯度）
        latents: (N, C) 当前步的 x_t
        t_norm: 归一化时间步 (0~1)
        reg_type: "vsd" | "kl"
        weight_mode: "uniform" | "t" | "ada"
    
    Returns:
        (loss, metric): loss 用于反向传播，metric 用于日志
    """
    diff = x0_student - x0_teacher  # (N, C)
    
    if weight_mode == "t":
        diff = t_norm * diff  # (N, C)
    elif weight_mode == "ada":
        diff = diff / (x0_teacher.abs().mean() + 0.01).detach()  # (N, C)
    
    if reg_type == "vsd":
        metric = 0.5 * (diff ** 2).mean().item()
        loss = SpecifyGradient.apply(latents, diff)
    elif reg_type == "kl":
        var = t_norm ** 2 + 1e-3
        loss = (0.5 * diff ** 2 / var).mean()
        metric = loss.item()
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")
    
    return loss, metric


# =====================================================================
# Rollout - Shape 阶段
# =====================================================================

def rollout_shape(
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    system: Trellis2System,
    device: torch.device,
    resolution: int = 1024,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> None:
    """
    Shape 阶段去噪采样。
    
    Args:
        state: Trellis2State，包含条件编码、坐标等
        cfg: 配置对象
        system: 系统组件
        device: 运行设备
        resolution: 模型分辨率（512 或 1024）
        generator: 随机数生成器
        is_training: 是否为训练模式
    
    Side Effects:
        - state.features.shape_slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 挂载 reg_loss 和 reg_metric
    """
    pipeline = system.pipeline
    stage = "shape"
    
    # ---- 1. 获取采样参数 ----
    sampler_params = pipeline.get_sampler_params(stage)
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["cfg_strength"])
    cfg_min, cfg_max = pipeline.get_cfg_interval(stage)
    
    # ---- 2. 初始化 ----
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)
    
    assert state.coords is not None, "state.coords 缺失"
    generator = generator or torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    latents_st = pipeline.init_latents(
        coords=state.coords,
        stage=stage,
        resolution=resolution,
        generator=generator,
    )  # SparseTensor
    latents = latents_st.feats  # (N, C)
    
    # ---- 3. Scheduler 配置 ----
    scheduler = pipeline.scheduler(stage)
    scheduler.set_timesteps(steps, device=device)
    
    # ---- 4. 正则化配置 ----
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_enabled = reg_type != "none" and is_training
    
    reg_loss_sum = 0.0
    reg_metric_sum = 0.0
    
    # ---- 5. 去噪循环 ----
    timesteps = list(scheduler.timesteps)[:-1]  # 排除最后一个 t=0
    steps_iter = tqdm(timesteps, desc="Shape Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for t in steps_iter:
        t_val = float(t.item())  # 0-1 范围
        t_norm = t_val  # 直接使用，scheduler.timesteps 已经是 0-1 范围
        use_cfg = cfg_min <= t_norm <= cfg_max
        
        # ---- cond 预测 ----
        if is_training:
            cond_pred = checkpoint(
                _predict_velocity, pipeline, state.coords, latents,
                t_val, cond_emb, stage, resolution, None,
                use_reentrant=False
            )  # (N, C)
        else:
            with torch.no_grad():
                cond_pred = _predict_velocity(
                    pipeline, state.coords, latents, t_val, cond_emb,
                    stage, resolution, None
                )  # (N, C)
        
        # ---- uncond 预测 + CFG 混合 ----
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, state.coords, latents, t_val, uncond_emb,
                    stage, resolution, None
                )  # (N, C)
            velocity = mix_cfg(cond_pred, uncond_pred, cfg_strength, uncond_mode=True)  # (N, C)
        else:
            velocity = cond_pred  # (N, C)
        
        # ---- 正则化（VSD / KL）----
        if reg_enabled:
            with pipeline.disable_lora_context(stage, resolution), torch.no_grad():
                teacher_cond = _predict_velocity(
                    pipeline, state.coords, latents, t_val, cond_emb,
                    stage, resolution, None
                )  # (N, C)
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_velocity(
                        pipeline, state.coords, latents, t_val, uncond_emb,
                        stage, resolution, None
                    )  # (N, C)
                    teacher_vel = mix_cfg(teacher_cond, teacher_uncond, cfg_strength, uncond_mode=True)  # (N, C)
                else:
                    teacher_vel = teacher_cond  # (N, C)
            
            x0_stu = latents - t_norm * velocity  # (N, C)
            x0_tea = latents - t_norm * teacher_vel  # (N, C)
            
            reg_loss, reg_metric = _compute_regularization(
                x0_stu, x0_tea, latents, t_norm,
                reg_type=reg_type, weight_mode=weight_mode
            )
            reg_loss_sum = reg_loss_sum + reg_loss
            reg_metric_sum = reg_metric_sum + reg_metric
        
        # ---- Scheduler 步进 ----
        x_t = SparseTensor(coords=state.coords, feats=latents)
        v_pred = SparseTensor(coords=state.coords, feats=velocity)
        latents = scheduler.step(v_pred, t, x_t).prev_sample.feats  # (N, C)
    
    # ---- 6. 反归一化 ----
    shape_slat_normalized = SparseTensor(coords=state.coords, feats=latents)
    shape_slat = pipeline.denormalize(shape_slat_normalized, stage)  # SparseTensor
    
    # ---- 7. 挂载到 state ----
    state.features.shape_slat = shape_slat
    
    num_steps = max(1, len(timesteps))
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None
    state.regularization.reg_metric = reg_metric_sum / num_steps if reg_enabled else None


# =====================================================================
# Rollout - Tex 阶段
# =====================================================================

def rollout_tex(
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    system: Trellis2System,
    device: torch.device,
    resolution: int = 1024,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> None:
    """
    Tex 阶段去噪采样。
    
    Args:
        state: Trellis2State，包含条件编码、坐标、shape_slat 等
        cfg: 配置对象
        system: 系统组件
        device: 运行设备
        resolution: 模型分辨率（512 或 1024）
        generator: 随机数生成器
        is_training: 是否为训练模式
    
    Side Effects:
        - state.features.tex_slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 更新 reg_loss 和 reg_metric
    """
    pipeline = system.pipeline
    stage = "tex"
    
    # ---- 1. 获取采样参数 ----
    sampler_params = pipeline.get_sampler_params(stage)
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["cfg_strength"])
    cfg_min, cfg_max = pipeline.get_cfg_interval(stage)
    
    # ---- 2. 初始化 ----
    cond_emb, uncond_emb = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)
    
    assert state.coords is not None, "state.coords 缺失"
    assert state.features.shape_slat is not None, "shape_slat 缺失，需先执行 rollout_shape"
    
    generator = generator or torch.Generator(device=device).manual_seed(int(cfg.seed) + 1000)
    
    # 归一化 shape_slat 作为 tex 的条件
    shape_cond = pipeline.normalize(state.features.shape_slat, "shape")  # SparseTensor
    
    latents_st = pipeline.init_latents(
        coords=state.coords,
        stage=stage,
        resolution=resolution,
        generator=generator,
    )  # SparseTensor
    latents = latents_st.feats  # (N, C)
    
    # ---- 3. Scheduler 配置 ----
    scheduler = pipeline.scheduler(stage)
    scheduler.set_timesteps(steps, device=device)
    
    # ---- 4. 正则化配置 ----
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_enabled = reg_type != "none" and is_training
    
    # Tex 阶段独立计算正则化（不累加 shape 阶段的）
    reg_loss_sum = 0.0
    reg_metric_sum = 0.0
    
    # ---- 5. 去噪循环 ----
    timesteps = list(scheduler.timesteps)[:-1]
    steps_iter = tqdm(timesteps, desc="Tex Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for t in steps_iter:
        t_val = float(t.item())  # 0-1 范围
        t_norm = t_val  # 直接使用，scheduler.timesteps 已经是 0-1 范围
        use_cfg = cfg_min <= t_norm <= cfg_max
        
        # ---- cond 预测 ----
        if is_training:
            cond_pred = checkpoint(
                _predict_velocity, pipeline, state.coords, latents,
                t_val, cond_emb, stage, resolution, shape_cond,
                use_reentrant=False
            )  # (N, C)
        else:
            with torch.no_grad():
                cond_pred = _predict_velocity(
                    pipeline, state.coords, latents, t_val, cond_emb,
                    stage, resolution, shape_cond
                )  # (N, C)
        
        # ---- uncond 预测 + CFG 混合 ----
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, state.coords, latents, t_val, uncond_emb,
                    stage, resolution, shape_cond
                )  # (N, C)
            velocity = mix_cfg(cond_pred, uncond_pred, cfg_strength, uncond_mode=True)  # (N, C)
        else:
            velocity = cond_pred  # (N, C)
        
        # ---- 正则化 ----
        if reg_enabled:
            with pipeline.disable_lora_context(stage, resolution), torch.no_grad():
                teacher_cond = _predict_velocity(
                    pipeline, state.coords, latents, t_val, cond_emb,
                    stage, resolution, shape_cond
                )  # (N, C)
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_velocity(
                        pipeline, state.coords, latents, t_val, uncond_emb,
                        stage, resolution, shape_cond
                    )  # (N, C)
                    teacher_vel = mix_cfg(teacher_cond, teacher_uncond, cfg_strength, uncond_mode=True)  # (N, C)
                else:
                    teacher_vel = teacher_cond  # (N, C)
            
            x0_stu = latents - t_norm * velocity  # (N, C)
            x0_tea = latents - t_norm * teacher_vel  # (N, C)
            
            reg_loss, reg_metric = _compute_regularization(
                x0_stu, x0_tea, latents, t_norm,
                reg_type=reg_type, weight_mode=weight_mode
            )
            reg_loss_sum = reg_loss_sum + reg_loss
            reg_metric_sum = reg_metric_sum + reg_metric
        
        # ---- Scheduler 步进 ----
        x_t = SparseTensor(coords=state.coords, feats=latents)
        v_pred = SparseTensor(coords=state.coords, feats=velocity)
        latents = scheduler.step(v_pred, t, x_t).prev_sample.feats  # (N, C)
    
    # ---- 6. 反归一化 ----
    tex_slat_normalized = SparseTensor(coords=state.coords, feats=latents)
    tex_slat = pipeline.denormalize(tex_slat_normalized, stage)  # SparseTensor
    
    # ---- 7. 挂载到 state ----
    state.features.tex_slat = tex_slat
    
    num_steps = max(1, len(timesteps))
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None
    state.regularization.reg_metric = reg_metric_sum / num_steps if reg_enabled else None


# =====================================================================
# 渲染工具函数 - Normal 渲染（Phase 1: Shape 训练）
# =====================================================================

def decode_and_render_normal(
    shape_slat: SparseTensor,
    cameras: Any,  # Trellis2State.Cameras
    pipeline: Any,
    renderer: Any,
    device: torch.device,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """
    解码 shape_slat 为 Mesh 并渲染 Normal 图（用于 Shape 阶段训练）。
    
    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: Mesh 渲染器
        device: 运行设备
        resolution: 输出分辨率
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) Normal 图（作为训练目标）
            "normal": (B, V, H, W, 3) Normal 图
            "meshes": List[Mesh]
            "subs": List[SparseTensor]
        }
    """
    # ---- 解码 Shape ----
    decode_result = pipeline.decode(shape_slat, tex_slat=None, resolution=resolution)
    meshes = decode_result["meshes"]  # List[Mesh]
    subs = decode_result["subs"]  # List[SparseTensor]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 逐样本逐视角渲染 ----
    all_normals: List[torch.Tensor] = []
    
    for i, mesh in enumerate(meshes):
        view_normals: List[torch.Tensor] = []
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            render_out = renderer.render(mesh, ext_iv, intr_iv, return_types=["normal"])
            normal = render_out["normal"]  # (H, W, 3)
            view_normals.append(normal)
        
        stacked = torch.stack(view_normals, dim=0)  # (V, H, W, 3)
        all_normals.append(stacked)
    
    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": normals,  # 用 normal 作为 "color" 供 guidance 使用
        "normal": normals,
        "meshes": meshes,
        "subs": subs,
    }


# =====================================================================
# 渲染工具函数 - RGB/PBR 渲染（Phase 2: Tex 训练）
# =====================================================================

def decode_and_render_pbr(
    shape_slat: SparseTensor,
    tex_slat: SparseTensor,
    subs: List[SparseTensor],
    cameras: Any,
    pipeline: Any,
    renderer: Any,
    device: torch.device,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """
    解码 shape_slat + tex_slat 为 MeshWithVoxel 并渲染 RGB 图（用于 Tex 阶段训练）。
    
    使用简化的渲染方式：查询顶点 PBR 属性，将 base_color 作为 RGB 输出。
    
    Args:
        shape_slat: SparseTensor，shape 特征
        tex_slat: SparseTensor，tex 特征
        subs: List[SparseTensor]，shape 解码中间结果
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: Mesh 渲染器
        device: 运行设备
        resolution: 输出分辨率
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) RGB 图（base_color）
            "mesh_with_voxel": List[MeshWithVoxel]
        }
    """
    # ---- 解码完整 MeshWithVoxel ----
    decode_result = pipeline.decode(shape_slat, tex_slat=tex_slat, resolution=resolution)
    mesh_with_voxels = decode_result["mesh_with_voxel"]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染 base_color 作为 RGB ----
    all_colors: List[torch.Tensor] = []
    
    for i, mesh in enumerate(mesh_with_voxels):
        view_colors: List[torch.Tensor] = []
        mesh = mesh.to(device)
        
        # 查询顶点 PBR 属性并提取 base_color
        vertex_attrs = mesh.query_vertex_attrs()  # (Nv, C), C=6 (base_color:3, metallic:1, roughness:1, alpha:1)
        base_color_slice = mesh.layout['base_color']
        base_color = vertex_attrs[:, base_color_slice]  # (Nv, 3)
        
        # 临时设置 vertex_attrs 供 renderer 使用
        mesh.vertex_attrs = base_color  # (Nv, 3)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            render_out = renderer.render(mesh, ext_iv, intr_iv, return_types=["color"])
            color = render_out["color"]  # (H, W, 3)
            view_colors.append(color)
        
        stacked = torch.stack(view_colors, dim=0)  # (V, H, W, 3)
        all_colors.append(stacked)
    
    colors = torch.stack(all_colors, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": colors,
        "mesh_with_voxel": mesh_with_voxels,
    }


# =====================================================================
# 前向传播 - Shape 阶段
# =====================================================================

def trellis2_shape_forward(
    system: Trellis2System,
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Shape 阶段前向传播: Dense Sampling → Shape Rollout → Normal 渲染
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) Normal 图（作为 guidance 输入）
            - "normal": (B, V, H, W, 3) Normal 图
            - "meshes": List[Mesh]
            - "subs": List[SparseTensor]
    
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.features.shape_slat: 挂载 shape latent
        - state.features.subs: 挂载解码中间结果
        - state.regularization: 挂载 reg_loss 和 reg_metric
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    
    # Dense Sampling
    ss_params = pipeline.get_ss_params()
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_embed,
            "neg_cond": state.views_conditioned.uncond_embed
        }
        coords = pipeline.dense_sampling(
            cond_dict, steps=int(ss_params["steps"]), resolution=stage_config["ss_resolution"]
        )  # (N, 4)
    state.coords = coords
    
    # Shape Rollout
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    rollout_shape(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # 解码 + Normal 渲染（使用 Shape 阶段的 renderer）
    render_out = decode_and_render_normal(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=pipeline.target_resolution,
    )
    
    # 挂载 subs 供后续 tex 阶段使用
    state.features.subs = render_out["subs"]
    
    return render_out


# =====================================================================
# 前向传播 - Tex 阶段
# =====================================================================

def trellis2_tex_forward(
    system: Trellis2System,
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Tex 阶段前向传播: Tex Rollout → RGB 渲染
    
    前置条件: 
        - state.coords 已挂载（由 trellis2_shape_forward 设置）
        - state.features.shape_slat 已挂载（由 trellis2_shape_forward 设置）
        - state.features.subs 已挂载（由 trellis2_shape_forward 设置）
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) RGB 图（base_color）
            - "mesh_with_voxel": List[MeshWithVoxel]
    
    Side Effects:
        - state.features.tex_slat: 挂载 tex latent
        - state.regularization: 更新 reg_loss 和 reg_metric
        - state.views_generated.image_tensor: 挂载渲染图像
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("tex")
    
    # 检查前置条件
    assert state.coords is not None, "state.coords 缺失，请先调用 trellis2_shape_forward"
    assert state.features.shape_slat is not None, "shape_slat 缺失，请先调用 trellis2_shape_forward"
    assert state.features.subs is not None, "subs 缺失，请先调用 trellis2_shape_forward"
    
    # Tex Rollout
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step + 1000)
    rollout_tex(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # RGB 渲染（使用 Tex 阶段的 renderer）
    render_out = decode_and_render_pbr(
        state.features.shape_slat,
        state.features.tex_slat,
        state.features.subs,
        state.cameras,
        pipeline,
        system.tex.renderer,
        device,
        resolution=pipeline.target_resolution,
    )
    
    state.views_generated.image_tensor = render_out["color"]  # (B, V, H, W, C)
    return render_out


# =====================================================================
# 评估
# =====================================================================

@torch.no_grad()
def evaluate(
    system: Trellis2System,
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
        accelerator: Accelerator
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 输出目录
    
    Returns:
        dict: 评估日志
    """
    if eval_loader is None:
        return {}
    
    pipeline = system.pipeline
    visual_io = VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution)
    
    # 获取需要设置为 eval 模式的模型
    models_to_eval = [
        system.shape.model,
        system.tex.model,
        pipeline.pipe.models['shape_slat_decoder'],
        pipeline.pipe.models['tex_slat_decoder'],
    ]
    
    # 过滤 None（eval_only 模式下 model 可能为 None）
    models_to_eval = [m for m in models_to_eval if m is not None]
    
    with EvalModeGuard(*models_to_eval):
        for batch_idx, batch in enumerate(eval_loader):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=pipeline, resolution=system.tex.config.cond_resolution)
            
            # Shape Forward (渲染 Normal)
            _ = trellis2_shape_forward(
                system, state, cfg, accelerator.device, global_step,
                is_training=False
            )
            
            # Tex Forward (渲染 RGB)
            render_out = trellis2_tex_forward(
                system, state, cfg, accelerator.device, global_step,
                is_training=False
            )
            
            if accelerator.is_main_process:
                visual_io.save_batch_eval(
                    state=state,
                    epoch=epoch,
                    render_out=render_out,
                    pipeline=pipeline,
                    export_mesh=True,
                )
    
    return {"eval_done": 1.0}


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    同时训练 Shape 和 Tex 两个 Flow Model，使用 RGB 渲染。
    
    流程: Dense Sampling → Shape Rollout → Tex Rollout → RGB 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2 --config=configs/trellis2.py
    """
    del argv
    cfg = _CONFIG.value
    
    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)
    
    # =====================================================
    # Step 2: 初始化 Accelerator
    # 配置混合精度训练和梯度累积
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
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
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_optimizers(accelerator)
    
    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    
    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return
    
    # =====================================================
    # Step 8: 训练循环
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
        losses = LossDict(device=accelerator.device)
        guidance_weights = system.guidance.get_loss_weights()
        
        losses.add("ssim", state.guidance.loss_ssim, weight=guidance_weights["ssim"])
        losses.add("lpips", state.guidance.loss_lpips, weight=guidance_weights["lpips"])
        losses.add("latent_mse", state.guidance.loss_latent_mse, weight=guidance_weights["latent_mse"])
        losses.add("reg", state.regularization.reg_loss, weight=cfg.train.loss.reg)
        
        # ---- 反向传播 ----
        total_loss = losses.total()
        accelerator.backward(total_loss)
        
        # ---- 构建日志 ----
        logs = losses.to_logs()  # {"loss/ssim": ..., "loss/total": ...}
        if state.regularization.reg_metric is not None:
            logs["loss/reg_metric"] = state.regularization.reg_metric
        return logs
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            # ============================================
            # Stage 1: Shape Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_render_out = trellis2_shape_forward(
                        system, state, cfg, accelerator.device, global_step,
                        is_training=True
                    )
                    shape_normal = shape_render_out["color"]  # (B, V, H, W, 3) - Normal 图
                    
                    # Shape Guidance（使用 Normal 监督几何）
                    shape_guidance_result = system.guidance.compute_guidance(
                        shape_normal,
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(shape_guidance_result)
                    
                    # Shape Loss & Backward
                    shape_log = _compute_loss_and_backward(state)
                
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # ============================================
            # Stage 2: Tex Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_render_out = trellis2_tex_forward(
                        system, state, cfg, accelerator.device, global_step,
                        is_training=True
                    )
                    tex_rgb = tex_render_out["color"]  # (B, V, H, W, 3) - RGB 图
                    
                    # Tex Guidance（使用 RGB 监督纹理）
                    tex_guidance_result = system.guidance.compute_guidance(
                        tex_rgb,
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(tex_guidance_result)
                    
                    # Tex Loss & Backward
                    tex_log = _compute_loss_and_backward(state)
                
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # ============================================
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用最终的 RGB 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(state=state, epoch=epoch, step=global_step)
        
        # 周期性评估
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)
        
        # 周期性保存检查点
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)
