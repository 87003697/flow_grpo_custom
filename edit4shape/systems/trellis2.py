"""
Trellis2 训练系统（适配 TRELLIS.2 双阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 生成系统训练，支持从单张图像生成 3D 模型。
核心流程：
- Stage 1 (Shape): 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss
- Stage 2 (Tex): Tex Rollout -> MeshWithVoxel -> PBR Voxel 渲染 -> Guidance Loss

特性：
- 双阶段训练：Shape 阶段用 Normal 渲染监督几何，Tex 阶段用 PBR Voxel 渲染监督纹理
- 每个 batch 分两步计算 Guidance Loss
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape / rollout_tex: 执行 Shape/Tex 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（渲染 Mesh Normal）
5. trellis2_tex_forward: Tex 阶段前向传播（使用 PbrMeshRenderer 渲染 PBR）
6. evaluate: 评估循环，生成 mesh 并保存可视化结果
7. main: 训练主循环（依次执行 Shape Guidance 和 Tex Guidance）

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- Shape 阶段: MeshRenderer 直接渲染 normal（支持梯度）
- Tex 阶段: PbrMeshRenderer 渲染 PBR + IBL 着色（支持梯度）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import argparse
import csv
import json
import logging
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
from absl import app, flags
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

# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。

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

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    build_run_paths,
)
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import rollout_shape, rollout_tex
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO

# =====================================================================
# Renderer 导入（使用 trellis2 的可微渲染器）
# =====================================================================
from trellis2.renderers import MeshRenderer, PbrMeshRenderer, EnvMap
from trellis2.representations.mesh import Mesh

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
        renderer: 渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）
            - Shape 阶段: MeshRenderer (直接渲染 normal，支持梯度)
            - Tex 阶段: PbrMeshRenderer (渲染 PBR + IBL 着色，支持梯度)
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
    
    渲染器配置（使用 trellis2 的 nvdiffrast 可微渲染器）：
    - shape.renderer: MeshRenderer (直接渲染 normal，支持梯度)
    - tex.renderer: PbrMeshRenderer (渲染 PBR + IBL 着色，支持梯度)
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.shape.model      # Shape Flow Model
        system.shape.renderer   # MeshRenderer (Normal)
        system.tex.renderer     # PbrMeshRenderer (PBR)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # 分阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    tex: StageSystem = field(default_factory=StageSystem)
    
    # 共享组件
    guidance: Any = None
    
    # 训练策略（LoRA 或 全参微调）
    strategy: Any = None
    
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
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, _build_single_optimizer,
    )
    from edit4shape.systems.base import compute_guidance_device
    from edit4shape.systems.utils.strategy import create_trellis2_strategy
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 2. Renderer 配置（Shape 和 Tex 共用） ----
    render_opts = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
        "chunk_size": 8000000,  # 分块渲染：800万面片/chunk，避免 nvdiffrast 2^24 限制，保持可微
    }
    
    # ---- 3. 获取阶段配置（训练和评估都需要） ----
    shape_config = get_stage_config(pipeline_type, "shape")
    tex_config = get_stage_config(pipeline_type, "tex")
    
    # ---- 4. 构建 StageSystem（使用 trellis2 可微渲染器） ----
    shape_renderer = MeshRenderer(rendering_options=render_opts, device=device)
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    tex_renderer = PbrMeshRenderer(rendering_options=render_opts, device=device)
    from edit4shape.renderers.ovoxel_trellis2 import load_envmap
    logging.info(f"[PbrMeshRenderer] 加载环境贴图: {cfg.renderer.envmap_path}")
    tex_renderer.envmap = load_envmap(cfg.renderer.envmap_path, device=device)
    tex_stage = StageSystem(
        config=tex_config,
        renderer=tex_renderer,
    )
    
    # ---- 5. 训练模式：创建 Strategy + 获取模型 + 构建优化器 ----
    guidance = None
    strategy = None
    
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        train_mode = cfg.train.mode  # "lora" | "full" | "frozen"
        train_device = accelerator.device
        teacher_device = compute_guidance_device(accelerator.device)
        
        strategy = create_trellis2_strategy(
            mode=train_mode,
            pipeline=pipeline,
            train_device=train_device,
            teacher_device=teacher_device,
            pipeline_type=pipeline_type,
            stages=["shape", "tex"],
            lora_cfg=cfg.lora,
            pretrained_path=cfg.pretrained.model,
        )
        
        strategy.setup()
        strategy.prepare(accelerator)
        
        # 统一获取学生模型和构建优化器
        shape_model = strategy.get_student("shape", shape_config.flow_resolution)
        optimizer_shape = _build_single_optimizer(shape_model, cfg.train.optimizer)
        shape_stage.model = shape_model
        shape_stage.optimizer = optimizer_shape
        
        tex_model = strategy.get_student("tex", tex_config.flow_resolution)
        optimizer_tex = _build_single_optimizer(tex_model, cfg.train.optimizer)
        tex_stage.model = tex_model
        tex_stage.optimizer = optimizer_tex
        
        # 启用 Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        pipeline._set_decoder_checkpointing("tex_slat_decoder", enable=True)
        pipeline._set_flow_model_checkpointing("shape", shape_config.flow_resolution, enable=True)
        pipeline._set_flow_model_checkpointing("tex", tex_config.flow_resolution, enable=True)
        logging.info("[Trellis2] 已启用 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        tex=tex_stage,
        guidance=guidance,
        strategy=strategy,
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
# 渲染工具函数 - Normal 渲染（Phase 1: Shape 训练）
# =====================================================================

def decode_and_render_normal(
    shape_slat: SparseTensor,
    cameras: Any,  # Trellis2State.Cameras
    pipeline: Any,
    renderer: Any,  # MeshRenderer（nvdiffrast，支持梯度）
    device: torch.device,
    resolution: int = 1024,
    use_checkpointing: bool = True,  # 使用 gradient checkpointing 减少显存
) -> Dict[str, Any]:
    """
    解码 shape_slat 为 Mesh 并使用 MeshRenderer 渲染 Normal 图。
    
    使用 nvdiffrast 可微渲染器直接渲染 normal，支持梯度反向传播。
    只调用 decode_shape，不调用 decode_tex（Normal 渲染不需要纹理信息）。
    支持 gradient checkpointing 以减少显存使用。
    
    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: MeshRenderer（nvdiffrast）
        device: 运行设备
        resolution: 输出分辨率
        use_checkpointing: 是否使用 gradient checkpointing（默认 True）
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) Normal 图
            "subs": List[SparseTensor]
            "meshes": List[Mesh]
        }
    """
    
    # ---- 解码 Shape（不调用 decode_tex，Normal 渲染只需要 Mesh） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    shape_result = pipeline.decode_shape(shape_slat, resolution)
    meshes = shape_result["meshes"]  # List[Mesh]
    subs = shape_result["subs"]  # List[SparseTensor]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染辅助函数 ----
    def _render_normal(mesh, ext, intr):
        out = renderer.render(mesh, ext, intr, return_types=["normal", "mask"])
        return out["normal"].permute(1, 2, 0)  # (H, W, 3)
    
    # ---- 使用 MeshRenderer 渲染 normal（nvdiffrast，支持梯度） ----
    all_normals: List[torch.Tensor] = []
    
    for i, mesh in enumerate(meshes):
        view_normals: List[torch.Tensor] = []
        mesh = mesh.to(device)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            if use_checkpointing:
                normal = checkpoint(_render_normal, mesh, ext_iv, intr_iv, use_reentrant=False)
            else:
                normal = _render_normal(mesh, ext_iv, intr_iv)
            
            view_normals.append(normal)  # (H, W, 3)
        
        stacked = torch.stack(view_normals, dim=0)  # (V, H, W, 3)
        all_normals.append(stacked)
    
    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": normals,   # (B, V, H, W, 3) Normal 图
        "subs": subs,       # List[SparseTensor]
        "meshes": meshes,   # List[Mesh]，用于后续 decode_tex
    }


# =====================================================================
# 渲染工具函数 - RGB/PBR 渲染（Phase 2: Tex 训练）
# =====================================================================

def decode_and_render_pbr(
    meshes: List[Any],  # List[Mesh]，来自 Shape 阶段
    tex_slat: SparseTensor,
    subs: List[SparseTensor],
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # PbrMeshRenderer（nvdiffrast，支持梯度）
    device: torch.device,
    resolution: int = 1024,
    use_checkpointing: bool = False,  # 使用 gradient checkpointing 减少显存
) -> Dict[str, Any]:
    """
    使用已解码的 Mesh 和 tex_slat 渲染 PBR 图。
    
    只调用 decode_tex（不重复调用 decode_shape），复用 Shape 阶段的 meshes。
    使用 nvdiffrast 可微渲染器进行 IBL 着色，支持梯度反向传播。
    支持 gradient checkpointing 以减少显存使用。
    
    注意：为了支持 checkpointing（要求确定性），SSAO 在 checkpointing 模式下被跳过。
    
    Args:
        meshes: List[Mesh]，来自 Shape 阶段的 decode_shape
        tex_slat: SparseTensor，tex 特征
        subs: List[SparseTensor]，shape 解码中间结果
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: PbrMeshRenderer（已挂载 envmap）
        device: 运行设备
        resolution: 输出分辨率
        use_checkpointing: 是否使用 gradient checkpointing（默认 True）
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) PBR shaded 图
            "mesh_with_voxels": List[MeshWithVoxel]
        }
    """
    
    # ★ FIX: Detach envlight specular mipmap 以避免跨 iter 计算图复用
    # renderer.envmap._nvdiffrec_envlight.specular 在 build_mips() 中被修改
    # 如果不 detach，第二次 iter 会尝试访问第一次 iter 已释放的计算图
    # 注意：_nvdiffrec_envlight 是惰性属性，只有在第一次访问 _backend 后才存在
    if hasattr(renderer.envmap, '_nvdiffrec_envlight'):
        envlight = renderer.envmap._nvdiffrec_envlight
        envlight.specular = [s.detach() if s is not None else None for s in envlight.specular]
    
    # ---- 只解码 Tex（复用 Shape 阶段的 meshes） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    # 数值保护（safe_clamp）已在 pipeline.decode_tex 中完成
    tex_result = pipeline.decode_tex(tex_slat, meshes, subs, resolution)
    mesh_with_voxels = tex_result["mesh_with_voxel"]  # List[MeshWithVoxel]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染辅助函数 ----
    # 注意：PbrMeshRenderer 的 SSAO 使用随机采样，checkpointing 时需固定种子
    def _render_pbr(mesh, ext, intr, seed):
        torch.manual_seed(seed)  # 固定种子确保 SSAO 确定性
        out = renderer.render(mesh, ext, intr, envmap=renderer.envmap, use_envmap_bg=False)
        return out['shaded'].permute(1, 2, 0)  # (H, W, 3)
    
    # ---- 使用 PbrMeshRenderer 渲染（nvdiffrast，支持梯度） ----
    all_colors: List[torch.Tensor] = []
    
    for i, voxel in enumerate(mesh_with_voxels):
        view_colors: List[torch.Tensor] = []
        voxel = voxel.to(device)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            seed = torch.tensor(42 + i * num_views + v)  # 作为 tensor 传入 checkpoint
            
            if use_checkpointing:
                shaded = checkpoint(_render_pbr, voxel, ext_iv, intr_iv, seed, use_reentrant=False)
            else:
                shaded = _render_pbr(voxel, ext_iv, intr_iv, seed)
            
            view_colors.append(shaded)  # (H, W, 3)
        
        all_colors.append(torch.stack(view_colors, dim=0))  # (V, H, W, 3)
    
    colors = torch.stack(all_colors, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": colors,                       # (B, V, H, W, 3) PBR shaded 图
        "mesh_with_voxels": mesh_with_voxels,  # List[MeshWithVoxel]，用于 mesh 导出
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
    Shape 阶段前向传播: Dense Sampling → Shape Rollout → Mesh Normal 渲染
    
    使用 MeshRenderer (nvdiffrast) 渲染 MeshWithVoxel，直接获取 normal（支持梯度）。
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) Normal 图
            - "subs": List[SparseTensor]
    
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.features.shape_slat: 挂载 shape latent
        - state.features.subs: 挂载解码中间结果
        - state.regularization: 挂载 reg_loss 和 reg_metric
        - state.views_generated.shape_tensor: 挂载 Normal 渲染图像
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    
    # Dense Sampling
    # Dense Sampling - 始终使用 512 分辨率的条件编码（对齐 TRELLIS.2 参考实现）
    ss_params = pipeline.get_ss_params()
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_512_embed,      # 始终用 512
            "neg_cond": state.views_conditioned.uncond_512_embed  # 始终用 512
        }
        coords = pipeline.dense_sampling(
            cond_dict, steps=int(ss_params["steps"]), resolution=stage_config["ss_resolution"]
        )  # (N, 4)
    state.coords = coords
    
    # Shape Rollout
    # eval 时使用全局种子（对齐参考实现），train 时使用独立 Generator
    generator = None if not is_training else torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
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
    
    # 挂载结果
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]  # List[Mesh]，供 Tex 阶段复用
    state.views_generated.shape_tensor = render_out["color"]  # (B, V, H, W, C) Normal 图
    
    # 简化超大 mesh，避免 nvdiffrast 面片数量限制（Shape 和 Tex 共用同一 mesh）
    state.simplify_meshes()
    
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
    Tex 阶段前向传播: Tex Rollout → PBR Mesh 渲染
    
    前置条件: 
        - state.coords 已挂载（由 trellis2_shape_forward 设置）
        - state.features.shape_slat 已挂载（由 trellis2_shape_forward 设置）
        - state.features.subs 已挂载（由 trellis2_shape_forward 设置）
    
    使用 PbrMeshRenderer (nvdiffrast) 渲染 MeshWithVoxel，进行 IBL 着色（支持梯度）。
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) PBR shaded 图
    
    Side Effects:
        - state.features.tex_slat: 挂载 tex latent
        - state.regularization: 更新 reg_loss 和 reg_metric
        - state.views_generated.pbr_tensor: 挂载 PBR 渲染图像
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("tex")
    
    # 检查前置条件
    assert state.coords is not None, "state.coords 缺失，请先调用 trellis2_shape_forward"
    assert state.features.shape_slat is not None, "shape_slat 缺失，请先调用 trellis2_shape_forward"
    assert state.features.subs is not None, "subs 缺失，请先调用 trellis2_shape_forward"
    assert state.features.meshes is not None, "meshes 缺失，请先调用 trellis2_shape_forward"
    
    # ★ 彻底切断与 Shape 阶段计算图的依赖
    # Shape backward 后计算图已释放，Tex 阶段必须完全切断所有依赖
    # 注意：SparseTensor.detach() 会复制 _spatial_cache，可能导致跨 iter 的计算图污染
    
    # 1. Detach 双分辨率条件嵌入 - 这些嵌入在 Shape/Tex 两阶段共用，必须 detach
    if state.views_conditioned.cond_512_embed is not None:
        state.views_conditioned.cond_512_embed = state.views_conditioned.cond_512_embed.detach()  # (B, S, C)
    if state.views_conditioned.uncond_512_embed is not None:
        state.views_conditioned.uncond_512_embed = state.views_conditioned.uncond_512_embed.detach()  # (B, S, C)
    if state.views_conditioned.cond_1024_embed is not None:
        state.views_conditioned.cond_1024_embed = state.views_conditioned.cond_1024_embed.detach()  # (B, S, C)
    if state.views_conditioned.uncond_1024_embed is not None:
        state.views_conditioned.uncond_1024_embed = state.views_conditioned.uncond_1024_embed.detach()  # (B, S, C)
    
    # 2. Detach coords - 虽然在 no_grad 下创建，但可能在 Shape rollout 中被 SparseTensor 缓存关联
    state.coords = state.coords.detach().clone()  # (N, 4) 创建全新的坐标张量
    
    # 3. Detach shape_slat - 创建全新的 SparseTensor
    state.features.shape_slat = SparseTensor(
        coords=state.features.shape_slat.coords.detach(),
        feats=state.features.shape_slat.feats.detach()
    )
    
    # 4. Detach subs - 创建全新的 SparseTensor，不继承任何缓存
    state.features.subs = [
        SparseTensor(coords=sub.coords.detach(), feats=sub.feats.detach())
        for sub in state.features.subs
    ]
    
    # 5. Detach meshes - vertices 和 vertex_attrs 都来自 shape decoder
    state.features.meshes = [
        Mesh(
            vertices=m.vertices.detach(),  # (V, 3) 顶点坐标
            faces=m.faces,                 # (F, 3) 面索引，整数不需要 detach
            vertex_attrs=m.vertex_attrs.detach() if m.vertex_attrs is not None else None  # 顶点属性
        )
        for m in state.features.meshes
    ]
    
    # Tex Rollout
    # eval 时使用全局种子（对齐参考实现），train 时使用独立 Generator
    generator = None if not is_training else torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step + 1000)
    rollout_tex(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # RGB 渲染（使用 Tex 阶段的 renderer，复用 Shape 阶段的 meshes）
    render_out = decode_and_render_pbr(
        state.features.meshes,   # 使用 Shape 阶段解码的 meshes，避免重复 decode_shape
        state.features.tex_slat,
        state.features.subs,
        state.cameras,
        pipeline,
        system.tex.renderer,
        device,
        resolution=pipeline.target_resolution,
    )
    
    state.views_generated.pbr_tensor = render_out["color"]  # (B, V, H, W, C)
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
    visual_io = Trellis2VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution)
    
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
    # Step 2: 初始化 Accelerator（含 wandb 日志）
    # =====================================================
    use_wandb = cfg.use_wandb #getattr(cfg, 'use_wandb', False)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )
    
    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    
    # 初始化 wandb trackers
    if use_wandb and accelerator.is_main_process:
        run_name = cfg.run_name #getattr(cfg, 'run_name', 'trellis2-distillation')
        accelerator.init_trackers(
            project_name="trellis2-shape+tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
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
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, system, stages=["shape", "tex"], mode="train")
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
    
    def _compute_loss_and_backward(state: Trellis2State, stage_name: str = "unknown") -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
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
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            # # ★ DEBUG: 开启 detect_anomaly 以获取详细的 backward 错误信息
            # with torch.autograd.set_detect_anomaly(True):
            
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
                    shape_log = _compute_loss_and_backward(state, stage_name="shape")
                
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # 保存 Shape 可视化（必须在 tex forward 之前，否则 views_edited 被覆盖）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)
        
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
                    tex_log = _compute_loss_and_backward(state, stage_name="tex")
                
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
        
            # ============================================
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存 Tex 可视化
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_tex_train(state=state, epoch=epoch, step=global_step)

        # ---- 周期性评估（epoch 级别，与 trellis.py 一致）----
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

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, epoch, global_step, stages=["shape", "tex"])


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
