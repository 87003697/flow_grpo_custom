"""
Trellis2 Tex 训练系统（专注于 Tex 阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 纹理生成系统训练，支持从单张图像生成 3D 模型的纹理。
核心流程：
- 图像条件 -> Dense Sampling -> Shape Rollout (frozen) -> Tex Rollout -> PBR 渲染 -> Guidance Loss

特性：
- 专注 Tex 阶段训练：使用 PBR 渲染监督纹理
- Shape 阶段使用冻结的模型生成几何
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_tex: 执行 Tex 阶段的去噪采样
4. trellis2_tex_forward: Tex 阶段前向传播（使用 PbrMeshRenderer 渲染 PBR）
5. evaluate: 评估循环，生成 mesh 并保存可视化结果
6. main: 训练主循环

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- PbrMeshRenderer 渲染 PBR + IBL 着色（支持梯度）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from edit4shape.systems.utils import MetricLogger, VisualIO

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
# 从 trellis2_shape / generators 中导入共用组件（避免代码重复）
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State as Trellis2StateBase
from edit4shape.generators.trellis2.rollout import rollout_shape, rollout_tex
from edit4shape.generators.trellis2.rollout.base import (
    trellis2_cfg_sparse,
    _predict_velocity,
    _compute_regularization,
)
from edit4shape.systems.trellis2_shape import (
    StageSystem,
    decode_and_render_normal,
    trellis2_shape_forward,
    build_dataloaders,
)

# =====================================================================
# 从 training_adpter 导入 StageConfig
# =====================================================================
from edit4shape.generators.trellis2.training_adpter import StageConfig




@dataclass
class Trellis2System:
    """
    Trellis2 Tex 训练系统。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（仅用于生成几何，不训练）
    - tex: Tex 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    
    渲染器配置（使用 trellis2 的 nvdiffrast 可微渲染器）：
    - shape.renderer: MeshRenderer (直接渲染 normal，用于 Shape Forward)
    - tex.renderer: PbrMeshRenderer (渲染 PBR + IBL 着色，支持梯度)
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.tex.model        # Tex Flow Model (可训练)
        system.tex.renderer     # PbrMeshRenderer (PBR)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # 分阶段系统
    shape: StageSystem = field(default_factory=StageSystem)  # Shape 阶段仅用于生成几何
    tex: StageSystem = field(default_factory=StageSystem)    # Tex 阶段可训练
    
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
        """准备 Tex 优化器（使用 accelerator.prepare）"""
        if self.tex.optimizer is not None:
            self.tex.optimizer = accelerator.prepare(self.tex.optimizer)
        return self


# =====================================================================
# Trellis2State - 双阶段状态类（继承自 trellis2_shape 的基类）
# =====================================================================

@dataclass
class Trellis2State(Trellis2StateBase):
    """
    Trellis2 双阶段生成过程的状态容器。
    
    继承自 trellis2_shape.Trellis2State（作为 Trellis2StateBase 导入），
    扩展 Tex 阶段相关字段：
    - features.tex_slat: 纹理阶段的稀疏潜变量
    - features.tex_slat_norm: 纹理阶段的归一化潜变量
    - views_generated.pbr_tensor: Tex 阶段 PBR shaded 图
    
    其他属性继承自基类，参见 trellis2_shape.Trellis2State 的文档。
    """
    
    @dataclass
    class Features:
        """特征容器。扩展基类，增加 Tex 阶段字段。"""
        # Denormalized 版本（用于 decode）
        shape_slat: Any = None      # SparseTensor, Shape 阶段输出（denormalized）
        tex_slat: Any = None        # SparseTensor, Tex 阶段输出（denormalized）
        # Normalized 版本（用于作为条件输入其他模型）
        shape_slat_norm: Any = None # SparseTensor, Shape latent（normalized）
        tex_slat_norm: Any = None   # SparseTensor, Tex latent（normalized）
        # 解码中间结果
        subs: Any = None            # List[SparseTensor], Shape 解码中间结果
        meshes: Any = None          # List[Mesh], Shape 解码输出的 mesh
    
    @dataclass
    class ViewsGenerated:
        """双阶段生成结果容器。扩展基类，增加 pbr_tensor 字段。"""
        shape_tensor: Any = None  # (B, V, H, W, C) Shape 阶段 Normal 图
        pbr_tensor: Any = None    # (B, V, H, W, C) Tex 阶段 PBR shaded 图
    
    # 覆盖父类：可卸载到 CPU 的属性路径（扩展基类，添加 tex 相关字段）
    _OFFLOADABLE: ClassVar[List[str]] = [
        "features.shape_slat_norm",
        "features.tex_slat_norm",
        "features.subs",
        "features.meshes",
        "views_conditioned.cond_512_embed",
        "views_conditioned.uncond_512_embed",
        "views_conditioned.cond_1024_embed",
        "views_conditioned.uncond_1024_embed",
    ]
    
    # ============== 覆盖基类的子状态容器（使用扩展版本） ==============
    features: Features = field(default_factory=Features)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict, 
    accelerator: Accelerator,
    guidance_factory: callable,
) -> Trellis2System:
    """
    构建 Trellis2 Tex 训练系统。
    
    只训练 Tex 阶段，Shape 阶段使用冻结的模型生成几何。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        Trellis2System: 包含所有组件的系统实例
    """
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, register_sparse_linear_with_peft, inject_lora_to_stage,
        Trellis2LoRAStrategy, Trellis2FullFinetuneStrategy, _build_single_optimizer,
    )
    from edit4shape.systems.base import compute_guidance_device
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 注入 Chunked Decoder（强制启用自适应显存分块） ----
    shape_decoder = pipeline.pipe.models['shape_slat_decoder']
    tex_decoder = pipeline.pipe.models['tex_slat_decoder']
    ChunkedDecoderMixin.inject_to(shape_decoder)
    ChunkedDecoderMixin.inject_to(tex_decoder)
    print("[Trellis2Tex] Shape/Tex decoder 已启用 chunked forward（自适应显存）")
    
    # ---- 2. Renderer 配置 ----
    render_opts = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
        "chunk_size": 8000000,  # 分块渲染：800万面片/chunk，避免 nvdiffrast 2^24 限制，保持可微
    }
    
    # ---- 3. 获取阶段配置 ----
    shape_config = get_stage_config(pipeline_type, "shape")
    tex_config = get_stage_config(pipeline_type, "tex")
    
    # ---- 4. 构建 StageSystem ----
    shape_renderer = MeshRenderer(rendering_options=render_opts, device=device)
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    tex_renderer = PbrMeshRenderer(rendering_options=render_opts, device=device)
    from edit4shape.renderers.ovoxel_trellis2 import load_envmap
    print(f"[PbrMeshRenderer] 加载环境贴图: {cfg.renderer.envmap_path}")
    tex_renderer.envmap = load_envmap(cfg.renderer.envmap_path, device=device)
    tex_stage = StageSystem(
        config=tex_config,
        renderer=tex_renderer,
    )
    
    # ---- 5. 训练模式：创建 Strategy + 获取模型 + 构建优化器（只训练 Tex） ----
    guidance = None
    strategy = None
    
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        train_mode = cfg.train.mode  # "lora" 或 "full"
        train_device = accelerator.device
        teacher_device = compute_guidance_device(accelerator.device)
        
        # 根据训练模式创建 Strategy
        if train_mode == "lora":
            register_sparse_linear_with_peft()
            inject_lora_to_stage(pipeline, pipeline_type, "tex", cfg.lora)
            strategy = Trellis2LoRAStrategy(pipeline, train_device, teacher_device)
        elif train_mode == "full":
            strategy = Trellis2FullFinetuneStrategy(
                pipeline, train_device, teacher_device,
                cfg.pretrained.model, pipeline_type, stages=["tex"]
            )
        else:
            raise ValueError(f"Unknown train.mode: {train_mode}. Use 'lora' or 'full'.")
        
        strategy.setup()
        
        # 统一获取学生模型和构建优化器（只训练 Tex）
        tex_model = strategy.get_student("tex", tex_config.flow_resolution)
        optimizer_tex = _build_single_optimizer(tex_model, cfg.train.optimizer)
        tex_stage.model = tex_model
        tex_stage.optimizer = optimizer_tex
        
        # 启用 Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        pipeline._set_decoder_checkpointing("tex_slat_decoder", enable=True)
        pipeline._set_flow_model_checkpointing("shape", shape_config.flow_resolution, enable=True)
        pipeline._set_flow_model_checkpointing("tex", tex_config.flow_resolution, enable=True)
        print("[Trellis2Tex] 已启用 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        tex=tex_stage,
        guidance=guidance,
        strategy=strategy,
    )



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
    使用已解码的 Mesh 和 tex_slat 渲染 PBR 图（强制使用 chunked forward）。
    
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
    
    # ★ 自适应 chunk_size 估算（Tex decoder 更大，每点约 8KB）
    monitor = MemoryMonitor(target_usage_ratio=0.75, min_chunk_size=32)
    chunk_size = monitor.estimate_chunk_size(
        num_points=tex_slat.coords.shape[0],
        coord_range=resolution,
        bytes_per_point=8192,
    )
    
    # ---- 只解码 Tex（复用 Shape 阶段的 meshes） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    # 数值保护（safe_clamp）已在 pipeline.decode_tex 中完成
    # ★ ChunkedDecoderMixin 已注入到 tex_decoder，pipeline.decode_tex 内部会自动使用 chunked forward
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
        "color": colors,           # (B, V, H, W, 3) PBR shaded 图
        "meshes": mesh_with_voxels,  # List[MeshWithVoxel]，用于 mesh 导出
    }




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
                    export_mesh=False,
                )
    
    return {"eval_done": 1.0}


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    只训练 Tex Flow Model，使用 PBR 渲染监督纹理。
    Shape 阶段使用冻结的模型生成几何。
    
    流程: Dense Sampling → Shape Rollout (frozen) → Tex Rollout → PBR 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_tex --config=configs/trellis2_tex.py
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
        run_name = cfg.run_name #getattr(cfg, 'run_name', 'trellis2-tex-distillation')
        accelerator.init_trackers(
            project_name="trellis2-tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": run_name}},
        )
    
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
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, system, stages=["tex"], mode="train")
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
    # Step 8: 训练循环（只训练 Tex 阶段）
    # =====================================================
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
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
            
            # ============================================
            # Shape Forward（冻结，只用于生成几何）
            # ============================================
            with torch.no_grad():
                _ = trellis2_shape_forward(
                    system, state, cfg, accelerator.device, global_step,
                    is_training=False  # Shape 阶段不训练
                )
            
            # ============================================
            # Tex Forward → Backward → Update
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
            
            # 每步结束后：卸载不需要的特征到 CPU + 清理显存缓存
            state.offload_features()
            torch.cuda.empty_cache()
        
            # ============================================
            # Logging
            # ============================================
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 PBR 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(state=state, epoch=epoch, step=global_step)
        
        # ============================================
        # Epoch 结束后：周期性评估和检查点保存
        # ============================================
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
        
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, epoch, global_step, stages=["tex"])


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)


# =====================================================================
# 模块导出列表（供 trellis2_shape+tex.py 等模块复用）
# =====================================================================
__all__ = [
    # 扩展版 State（含 tex 字段）
    "Trellis2State",
    # Tex 阶段核心函数
    "rollout_tex",
    "decode_and_render_pbr",
    "trellis2_tex_forward",
]
