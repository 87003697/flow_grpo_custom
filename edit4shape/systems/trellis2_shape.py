"""
Trellis2 Shape 训练系统（专注于 Shape 阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 几何生成系统训练，支持从单张图像生成 3D 模型。
核心流程：
- 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss

特性：
- 专注 Shape 阶段训练：使用 Normal 渲染监督几何
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape: 执行 Shape 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（渲染 Mesh Normal）
5. evaluate: 评估循环，生成 mesh 并保存可视化结果
6. main: 训练主循环

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- MeshRenderer 直接渲染 normal（支持梯度）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
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
# Chunked Forward 支持（自定义实现，已从 _reference_codes 迁移）
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
from edit4shape.generators.trellis2.chunked import MemoryMonitor

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from edit4shape.systems.base import SpecifyGradient

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    BaseState,
    CheckpointIO,
    build_run_paths,
    SpecifyGradient,
)
from edit4shape.systems.utils import MetricLogger, append_csv_row, Trellis2VisualIO, LossDict
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import rollout_shape

# =====================================================================
# Renderer 导入（使用 trellis2 的可微渲染器）
# =====================================================================
from trellis2.renderers import MeshRenderer
from trellis2.representations.mesh import Mesh

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape"]


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
    
    封装 Shape 阶段的 model、optimizer、renderer 和配置。
    
    属性:
        model: Flow Model
        optimizer: 优化器
        renderer: MeshRenderer（直接渲染 normal，支持梯度）
        config: StageConfig 配置
    """
    model: Any = None       # Flow Model
    optimizer: Any = None   # Optimizer
    renderer: Any = None    # MeshRenderer
    config: StageConfig = field(default_factory=StageConfig)


@dataclass
class Trellis2System:
    """
    Trellis2 Shape 训练系统。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    
    渲染器配置（使用 trellis2 的 nvdiffrast 可微渲染器）：
    - shape.renderer: MeshRenderer (直接渲染 normal，支持梯度)
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.shape.model      # Shape Flow Model
        system.shape.renderer   # MeshRenderer (Normal)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # Shape 阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    
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
        """准备 Shape 优化器（使用 accelerator.prepare）"""
        if self.shape.optimizer is not None:
            self.shape.optimizer = accelerator.prepare(self.shape.optimizer)
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
    构建完整的 Trellis2 Shape 系统。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        Trellis2System: 包含所有组件的系统实例
    """
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, set_stage_trainable, build_optimizer_for_stage, register_sparse_linear_with_peft
    )
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 注入 Chunked Decoder（强制启用自适应显存分块） ----
    shape_decoder = pipeline.pipe.models['shape_slat_decoder']
    ChunkedDecoderMixin.inject_to(shape_decoder)
    print("[Trellis2] Shape decoder 已启用 chunked forward（自适应显存）")
    
    # ---- 2. Renderer 配置 ----
    render_opts = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
        "chunk_size": 8000000,  # 分块渲染：800万面片/chunk，避免 nvdiffrast 2^24 限制，保持可微
    }
    
    # ---- 3. 获取 Shape 阶段配置 ----
    shape_config = get_stage_config(pipeline_type, "shape")
    
    # ---- 4. 构建 StageSystem（使用 trellis2 可微渲染器） ----
    # Shape 阶段：MeshRenderer 渲染 normal（nvdiffrast，支持梯度）
    shape_renderer = MeshRenderer(rendering_options=render_opts, device=device)
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    
    # ---- 5. 训练模式：设置 model 和 optimizer ----
    guidance = None
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        register_sparse_linear_with_peft()
        set_stage_trainable(pipeline, pipeline_type, ["shape"])
        
        # 获取模型
        shape_stage.model = pipeline.get_flow_model(shape_config.model_stage, shape_config.flow_resolution)
        
        # 创建优化器
        optimizer_shape, = build_optimizer_for_stage(
            pipeline, pipeline_type, ["shape"], cfg.train.optimizer
        )
        shape_stage.optimizer = optimizer_shape
        
        # 启用 Decoder Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        print("[Trellis2] 已启用 shape_slat_decoder 的 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
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
# 渲染工具函数 - Normal 渲染
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
    
    使用"伪 GT intersected"方案渲染 Normal（可微 Mesh 路径，强制使用 chunked forward）。
    
    核心思路：
    1. 用模型预测的 h.feats[3:6] > 0 作为 intersected（detach，固定拓扑）
    2. dual_vertices (h.feats[0:3]) 和 quad_lerp (h.feats[6:7]) 参与梯度
    3. 调用 flexible_dual_grid_to_mesh(train=True) 生成可微 Mesh
    4. 使用 MeshRenderer 渲染 Normal
    
    使用 nvdiffrast 可微渲染器直接渲染 normal，支持梯度反向传播。
    只调用 decode_shape（Normal 渲染不需要纹理信息）。
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
    from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
    import torch.nn.functional as F
    
    # ---- 解码 Shape（Normal 渲染只需要 Mesh） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    shape_result = pipeline.decode_shape(shape_slat, resolution)
    meshes = shape_result["meshes"]  # List[Mesh]
    subs = shape_result["subs"]  # List[SparseTensor]
    
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # ★ 自适应 chunk_size 估算
    monitor = MemoryMonitor(target_usage_ratio=0.75, min_chunk_size=32)
    chunk_size = monitor.estimate_chunk_size(
        num_points=shape_slat.coords.shape[0],
        coord_range=resolution,
        bytes_per_point=4096,
    )
    
    # ★ 直接调用 chunked forward
    h, subs = decoder.forward_chunked(shape_slat, chunk_size=chunk_size, axis=3, return_subs=True)  # h.feats: (N, 7)
    
    voxel_margin = decoder.voxel_margin
    
    # ========== 分解 h.feats ==========
    # 1. dual_vertices: sigmoid 变换后的顶点偏移（可微）
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )  # SparseTensor feats: (N, 3)
    
    # 2. intersected: 硬阈值 + detach（伪 GT，不可微）
    # 这是关键：用模型自己的预测作为固定拓扑
    pseudo_gt_intersected = h.replace(
        (h.feats[..., 3:6] > 0).detach()  # detach 切断梯度
    )  # SparseTensor feats: (N, 3)
    
    # 3. quad_lerp: softplus 变换（可微）
    quad_lerp_sp = h.replace(F.softplus(h.feats[..., 6:7]))  # SparseTensor feats: (N, 1)
    
    # ========== 为每个 batch 构建 Mesh ==========
    meshes = []
    for v, i, q in zip(vertices_sp, pseudo_gt_intersected, quad_lerp_sp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:],  # (N, 3) voxel 坐标
            v.feats,          # (N, 3) dual_vertices（可微）
            i.feats,          # (N, 3) intersected（detached bool）
            q.feats,          # (N, 1) quad_lerp（可微）
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=True,       # 启用可微路径
        )
        meshes.append(Mesh(vertices, faces))
    
    # ========== 渲染 Normal ==========
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染辅助函数 ----
    def _render_normal(mesh, ext, intr):
        out = renderer.render(mesh, ext, intr, return_types=["normal", "mask"])  # out["normal"]: (1, H, W, 3)
        return out["normal"].permute(1, 2, 0)  # (H, W, 3)
    
    # ---- 使用 MeshRenderer 渲染 normal（nvdiffrast，支持梯度） ----
    all_normals: List[torch.Tensor] = []
    
    for i, mesh in enumerate(meshes):
        view_normals: List[torch.Tensor] = []
        mesh = mesh.to(device)  # Mesh verts: (Nv, 3), faces: (Nf, 3)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            if use_checkpointing:
                normal = checkpoint(_render_normal, mesh, ext_iv, intr_iv, use_reentrant=False)  # (H, W, 3)
            else:
                normal = _render_normal(mesh, ext_iv, intr_iv)  # (H, W, 3)
            
            view_normals.append(normal)  # (H, W, 3)
        
        stacked = torch.stack(view_normals, dim=0)  # (V, H, W, 3)
        all_normals.append(stacked)
    
    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": normals,  # (B, V, H, W, 3) Normal 图
        "subs": list(subs),  # List[SparseTensor]
        "meshes": meshes,  # List[Mesh]
    }


def decode_and_render_normal_fdg(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    device: torch.device,
    resolution: int = 1024,
    render_resolution: int = 1024,
) -> Dict[str, Any]:
    """
    使用 FDG 模式的可微 Voxel Normal 渲染（强制使用 chunked forward）。

    核心思路：
    1. 调用 FDG Decoder 父类获取原始特征 h.feats (N, 7)
    2. 分解出 dual_vertices (h.feats[0:3]) 和 intersected_logits (h.feats[3:6])
    3. 使用 render_normal_fdg 渲染可微 Normal

    相比 decode_and_render_normal_mesh_pseudo_gt 的改进：
    - ✅ dual_vertices 有梯度（与伪 GT 方案相同）
    - ✅ intersected_logits 也有梯度（伪 GT 方案中 intersected 是 detach 的）

    梯度流：
    Loss → Normal → voxel_normals[voxel_id] → dual_vertices + intersected_logits → Decoder

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        device: 运行设备
        resolution: Decoder 分辨率（grid_size）
        render_resolution: 渲染输出分辨率

    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": None}
    """
    from edit4shape.renderers.diff_voxel_normal import render_normal_fdg, RenderConfig
    import torch.nn.functional as F

    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)

    # ★ 自适应 chunk_size 估算（强制使用 MemoryMonitor）
    monitor = MemoryMonitor(target_usage_ratio=0.75, min_chunk_size=32)
    chunk_size = monitor.estimate_chunk_size(
        num_points=shape_slat.coords.shape[0],
        coord_range=resolution,
        bytes_per_point=4096,  # 每点约 4KB（经验值）
    )
    
    # ★ 直接使用 chunked forward（chunk_size=None 时内部会使用普通 forward）
    # return_subs=False：不保存中间结果，节省显存（纯 Shape 训练不需要 subs）
    h = decoder.forward_chunked(shape_slat, chunk_size=chunk_size, axis=3, return_subs=False)  # h.feats: (N, 7)

    voxel_margin = decoder.voxel_margin

    # ========== 获取相机参数 ==========
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]

    # 中性 Normal 背景（朝向相机，RGB = [0.5, 0.5, 1.0]）
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # (3,)

    # ========== 为每个 batch 渲染 Normal ==========
    all_normals = []

    for i, h_i in enumerate(h):
        coords = h_i.coords[:, 1:]  # (N, 3) voxel 坐标

        # 分解 h.feats
        # dual_vertices: sigmoid 变换后的顶点偏移（可微）
        dual_vertices = (1 + 2 * voxel_margin) * F.sigmoid(h_i.feats[..., 0:3]) - voxel_margin  # (N, 3)

        # intersected_logits: 保持原始 logits（可微，不做硬阈值）
        intersected_logits = h_i.feats[..., 3:6]  # (N, 3)

        view_normals = []
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)

            # 构建渲染配置（简化接口）
            config = RenderConfig(
                extrinsic=ext_iv,
                intrinsic=intr_iv,
                resolution=resolution,
            )

            # 使用 FDG 模式渲染
            normal, mask = render_normal_fdg(coords, dual_vertices, intersected_logits, config)  # (H, W, 3), (H, W)
            normal = (normal + 1.0) * 0.5  # (H, W, 3)

            # 混合背景颜色
            mask_3d = mask.unsqueeze(-1).float()  # (H, W, 1)
            normal = normal * mask_3d + bg_color * (1 - mask_3d)  # (H, W, 3)

            view_normals.append(normal)

        all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)

    return {"color": normals, "subs": None, "meshes": None}


def decode_and_render_normal_neighbor26_soft(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    device: torch.device,
    resolution: int = 1024,
    render_resolution: int = 1024,
) -> Dict[str, Any]:
    """
    使用 26 邻居 soft occupancy 的可微 Normal 渲染（强制使用 chunked forward）。

    核心思路：
    1. 调用 Decoder 获取 subs（4 层 sub logits）
    2. 使用 render_sub_normal_soft 渲染可微 Normal

    梯度流：
    Loss → Normal → neighbor_occupancy(soft) → subs logits → Decoder

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        device: 运行设备
        resolution: Decoder 分辨率（grid_size，必须是 1024）
        render_resolution: 渲染输出分辨率

    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": None}
    """
    from edit4shape.renderers.diff_voxel_normal_neighbor26 import (
        render_sub_normal_soft, RenderConfig
    )

    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)

    # ★ 自适应 chunk_size 估算
    monitor = MemoryMonitor(target_usage_ratio=0.75, min_chunk_size=32)
    chunk_size = monitor.estimate_chunk_size(
        num_points=shape_slat.coords.shape[0],
        coord_range=resolution,
        bytes_per_point=4096,
    )
    
    # ★ 直接调用 chunked forward（需要 return_subs=True）
    h, subs = decoder.forward_chunked(shape_slat, chunk_size=chunk_size, axis=3, return_subs=True)  # h.feats: (N, 7), subs: List[SparseTensor]
    # subs: [sub0(64), sub1(128), sub2(256), sub3(512)]

    # 获取相机参数
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]

    # 中性 Normal 背景（朝向相机，RGB = [0.5, 0.5, 1.0]）
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # (3,)

    # 为每个 batch 渲染 Normal
    all_normals = []

    for i, h_i in enumerate(h):
        # 提取第 i 个 batch 的 subs（每层取第 i 个 batch）
        subs_i = [sub[i] for sub in subs]  # List[SparseTensor]，each feats: (N_i, C)
        
        view_normals = []
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)

            # 构建渲染配置
            config = RenderConfig(
                extrinsic=ext_iv,
                intrinsic=intr_iv,
                resolution=resolution,
            )

            # 渲染（可微）
            normal, mask = render_sub_normal_soft(
                subs=subs_i,
                config=config,
                h=h_i,  # 提供目标层坐标
                voxel_resolution=resolution,
                target_size=(render_resolution, render_resolution) if render_resolution != resolution else None,
            )  # (H, W, 3), (H, W)
            
            # 转换到 [0, 1] 范围
            normal = (normal + 1.0) * 0.5  # (H, W, 3)

            # 混合背景颜色
            mask_3d = mask.unsqueeze(-1).float()  # (H, W, 1)
            normal = normal * mask_3d + bg_color * (1 - mask_3d)  # (H, W, 3)

            view_normals.append(normal)

        all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)

    return {"color": normals, "subs": list(subs), "meshes": None}


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
    state.features.meshes = render_out["meshes"]  # List[Mesh]
    state.views_generated.shape_tensor = render_out["color"]  # (B, V, H, W, C) Normal 图
    
    # 简化超大 mesh，避免 nvdiffrast 面片数量限制
    state.simplify_meshes()
    
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
    3. 执行 Shape Rollout 生成特征
    4. 解码为 Mesh
    5. 渲染 Normal 图并保存
    6. 导出 mesh 文件
    
    输出目录结构：
    visuals_eval_dir/
    └── epoch_{N}/
        ├── sample_name_1/
        │   ├── normal.png     # 渲染的法线图
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
        pipeline.pipe.models['shape_slat_decoder'],
    ]
    
    # 过滤 None（eval_only 模式下 model 可能为 None）
    models_to_eval = [m for m in models_to_eval if m is not None]
    
    with EvalModeGuard(*models_to_eval):
        for batch_idx, batch in enumerate(eval_loader):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=pipeline, resolution=system.shape.config.cond_resolution)
            
            # Shape Forward (渲染 Normal)
            render_out = trellis2_shape_forward(
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
    
    训练 Shape Flow Model，使用 Normal 渲染监督几何。
    
    流程: Dense Sampling → Shape Rollout → Normal 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_shape --config=configs/trellis2_shape.py
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
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq)
    
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
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
        losses = LossDict(device=accelerator.device)
        
        # Guidance loss（统一遍历 loss_dict）
        guidance_weights = system.guidance.get_loss_weights()
        for name, loss in state.guidance.loss_dict.items():
            w = guidance_weights.get(name) * cfg.train.loss.guidance
            losses.add(name, loss, weight=w)
        
        # 正则化 loss
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
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.shape.config.cond_resolution)
            
            # ============================================
            # Shape Forward → Backward → Update
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
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 Normal 渲染结果）
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
