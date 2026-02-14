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
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点，节省显存
from tqdm import tqdm

# =====================================================================
# 项目内部导入
# =====================================================================


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
# Chunked Forward 支持（自定义实现，已从 _reference_codes 迁移）
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin

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
    build_run_paths,
)
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO
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
    
    # 训练策略（LoRA / Full / Frozen）
    strategy: Any = None
    
    # 兼容 autograd 三阶段实现：允许系统对象携带运行时上下文
    cfg: Any = None
    accelerator: Accelerator = None
    
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
        """
        通过 strategy.prepare() 统一做 DDP 包裹 + 回写 pipeline。
        
        与 V1 System.prepare_models_and_optimizers() 对齐：
        模型和优化器一起 prepare → DDP 包裹 + 注册到 accelerator，
        使 save_state/load_state 自动管理模型权重。
        """
        if self.strategy is not None and self.shape.optimizer is not None:
            shape_config = self.shape.config
            self.shape.model, self.shape.optimizer = self.strategy.prepare(
                accelerator, shape_config.model_stage, shape_config.flow_resolution, self.shape.optimizer
            )
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
        get_stage_config, _build_single_optimizer,
    )
    from edit4shape.systems.base import compute_guidance_device
    from edit4shape.generators.trellis2.training_adpter import create_trellis2_strategy
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 注入 Chunked Decoder（强制启用自适应显存分块） ----
    shape_decoder = pipeline.pipe.models['shape_slat_decoder']
    ChunkedDecoderMixin.inject_to(shape_decoder)
    logging.info("[Trellis2] Shape decoder 已启用 chunked forward（自适应显存）")
    
    # ---- 2. Renderer 配置 ----
    render_opts_base = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
    }
    
    # ---- 3. 获取 Shape 阶段配置 ----
    shape_config = get_stage_config(pipeline_type, "shape")
    
    # ---- 4. 构建 StageSystem ----
    normal_mode = cfg.renderer.normal_mode
    if normal_mode == "hybrid26":
        from edit4shape.renderers.hybrid_trellis2 import Hybrid26NormalRenderer
        # Hybrid26NormalRenderer 使用 ProfiledScheduler 自适应分块，无需 chunk_size
        shape_renderer = Hybrid26NormalRenderer(rendering_options=render_opts_base, device=device)
        logging.info("[Trellis2] Shape renderer: Hybrid26NormalRenderer（subs 可微，自适应分块）")
    else:
        # MeshRenderer 需要 chunk_size 控制 nvdiffrast 分块
        mesh_opts = {**render_opts_base, "chunk_size": 8000000}
        shape_renderer = MeshRenderer(rendering_options=mesh_opts, device=device)
        logging.info("[Trellis2] Shape renderer: MeshRenderer（nvdiffrast）")
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    
    # ---- 5. 训练模式：设置 model 和 optimizer ----
    guidance = None
    strategy = None
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        train_mode = cfg.train.mode  # "lora" | "full" | "frozen"
        train_device = accelerator.device
        teacher_device = compute_guidance_device(accelerator.device)
        
        lora_cfg = getattr(cfg, "lora", None)
        strategy = create_trellis2_strategy(
            mode=train_mode,
            pipeline=pipeline,
            train_device=train_device,
            teacher_device=teacher_device,
            pipeline_type=pipeline_type,
            stages=["shape"],
            lora_cfg=lora_cfg,
            pretrained_path=cfg.pretrained.model,
        )
        
        strategy.setup()
        
        # 统一获取学生模型和构建优化器（prepare 在 prepare_optimizers 中完成）
        shape_model = strategy.get_student("shape", shape_config.flow_resolution)
        optimizer_shape = _build_single_optimizer(shape_model, cfg.train.optimizer)
        shape_stage.model = shape_model
        shape_stage.optimizer = optimizer_shape
        
        # 启用 Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        pipeline._set_flow_model_checkpointing("shape", shape_config.flow_resolution, enable=True)
        logging.info("[Trellis2] 已启用 shape_slat_decoder + shape_flow_model 的 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        guidance=guidance,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )


# =====================================================================
# 渲染工具函数 - Normal 渲染
# =====================================================================

def decode_and_render_normal(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,
    device: torch.device,
    resolution: int,
    normal_mode: str = "mesh",
) -> Dict[str, Any]:
    """
    解码 shape_slat 并渲染 Normal 图（统一入口，根据 normal_mode 分发）。

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: MeshRenderer 或 Hybrid26NormalRenderer
        device: 运行设备
        resolution: Decoder 分辨率
        normal_mode: "mesh" | "hybrid26"

    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": List[Mesh]}
    """
    if normal_mode == "hybrid26":
        return decode_and_render_normal_hybrid(
            shape_slat, cameras, pipeline, renderer, device, resolution,
        )
    else:
        return decode_and_render_normal_mesh(
            shape_slat, cameras, pipeline, renderer, device, resolution,
        )


def decode_and_render_normal_mesh(
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
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # ★ 逐层自适应 chunked forward（每层根据实时显存自动估算 chunk_size）
    h, subs = decoder.forward_chunked(shape_slat, axis=3, return_subs=True, use_checkpoint=True)  # h.feats: (N, 7)
    
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


def decode_and_render_normal_hybrid(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # Hybrid26NormalRenderer
    device: torch.device,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """
    解码 shape_slat 并使用 Hybrid26NormalRenderer 渲染 Normal 图。

    单次 forward_chunked 同时获取 h（mesh 构建参数 + voxel 坐标）和 subs（多层 sub logits）。

    梯度路径：
    Loss → pixel_normal → grid_sample_3d → voxel_normal
         → occupancy_diff → sub_logits → Decoder

    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: Hybrid26NormalRenderer
        device: 运行设备
        resolution: Decoder 分辨率

    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": List[Mesh]}
    """
    from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
    import torch.nn.functional as F

    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)

    # ★ 逐层自适应 chunked forward（每层根据实时显存自动估算 chunk_size）
    h, subs = decoder.forward_chunked(shape_slat, axis=3, return_subs=True, use_checkpoint=True)

    voxel_margin = decoder.voxel_margin

    # ★ 归还 PyTorch reserved-but-unallocated 显存给 CUDA，
    #   供 renderer 的 grid_sample_3d 等原生 CUDA 分配使用
    torch.cuda.empty_cache()

    # ========== 分解 h.feats → 构建可微 Mesh ==========
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    intersected = h.replace((h.feats[..., 3:6] > 0).detach())
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

    meshes = []
    for v, i, q in zip(vertices_sp, intersected, quad_lerp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=False,
        )
        meshes.append(Mesh(vertices, faces))

    # ========== 渲染 Normal ==========
    extr_all = cameras.w2c.to(device)          # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)   # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]

    # 中性 Normal 背景（朝向相机，RGB = [0.5, 0.5, 1.0]）
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # (3,)

    all_normals: List[torch.Tensor] = []

    for i, (mesh_i, h_i) in enumerate(zip(meshes, h)):
        subs_i = [sub[i] for sub in subs]   # 提取 per-batch subs
        coords_i = h_i.coords[:, 1:]        # (N, 3) voxel 坐标
        mesh_i = mesh_i.to(device)

        view_normals: List[torch.Tensor] = []
        for v in range(num_views):
            out = renderer.render(
                mesh=mesh_i,
                subs=subs_i,
                coords=coords_i,
                extrinsics=extr_all[i, v],   # (4, 4)
                intrinsics=intr_all[i, v],   # (3, 3)
                voxel_resolution=resolution,
                return_types=["normal", "mask"],
            )
            normal = out["normal"]                          # (H, W, 3)
            mask_3d = out["mask"].unsqueeze(-1).float()     # (H, W, 1)
            normal = normal * mask_3d + bg_color * (1 - mask_3d)  # (H, W, 3)
            view_normals.append(normal)

        all_normals.append(torch.stack(view_normals, dim=0))  # (V, H, W, 3)

    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)

    return {
        "color": normals,       # (B, V, H, W, 3) Normal 图
        "subs": list(subs),     # List[SparseTensor]
        "meshes": meshes,       # List[Mesh]
    }


# =====================================================================
# 前向传播 - Shape 阶段
# =====================================================================

def trellis2_shape_forward(
    system: Trellis2System,
    state: Trellis2State,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Shape 阶段前向传播: Dense Sampling → Shape Rollout → Mesh Normal 渲染
    
    使用 MeshRenderer (nvdiffrast) 渲染 MeshWithVoxel，直接获取 normal（支持梯度）。
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
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
    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    
    if system.accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")
    device = system.accelerator.device
    
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
    
    # 解码 + Normal 渲染
    render_out = decode_and_render_normal(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=pipeline.target_resolution,
        normal_mode=cfg.renderer.normal_mode,
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
    6. 可选导出 mesh 文件（默认关闭）
    
    输出目录结构：
    visuals_eval_dir/
    └── epoch_{N}/
        ├── sample_name_1/
        │   ├── normal.png     # 渲染的法线图
        │   └── mesh.obj       # 导出的网格文件（可选）
        └── sample_name_2/
            └── ...
    
    Args:
        system: 系统组件
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 输出目录
    
    Returns:
        dict: 评估日志
    """
    if eval_loader is None:
        return {}
    
    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    
    accelerator = system.accelerator
    if accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")
    
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
            # 推理阶段复用 build_system() 创建的训练渲染器实例，确保渲染配置一致。
            render_out = trellis2_shape_forward(
                system, state, global_step,
                is_training=False
            )
            
            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=False,
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
    # Step 2: 初始化 Accelerator（含 wandb 日志）
    # =====================================================
    use_wandb = cfg.use_wandb
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
        accelerator.init_trackers(
            project_name="trellis2-shape-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    from edit4shape.systems.trellis2 import build_dataloaders
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
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    
    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system,
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
        # Guidance loss（已在 Guidance 内部加权汇总，直接使用）
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
                            system, state, global_step,
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
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 的计算图和碎片缓存，防止 OOM
            del state, shape_render_out, shape_guidance_result, shape_log
            torch.cuda.empty_cache()

        # ---- 周期性评估（epoch 级别，与 trellis.py 一致）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system,
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
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
