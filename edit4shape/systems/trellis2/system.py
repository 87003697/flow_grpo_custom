"""
Trellis2 系统共享组件。

本模块提供被所有训练入口文件共享的基础组件：
- Trellis2System: 系统基类（共享 pipeline / guidance / strategy）
- ShapeSystem / TexSystem / ShapeTexSystem: 按阶段组合的子类
- StageSystem: 单阶段的 model / optimizer / renderer / config
- build_system: 统一的系统构建工厂（支持 shape / tex / shape_tex 模式）
- build_dataloaders: 构造训练和评估 DataLoader

继承关系：
    Trellis2System (base)
    ├── ShapeSystem      — 仅 Shape 阶段
    ├── TexSystem        — Shape (frozen) + Tex 阶段
    └── ShapeTexSystem   — Shape + Tex 双阶段

注意：
    具体训练逻辑（autograd / async 等）在各自入口模块中实现，
    本文件不包含 main()、evaluate() 等训练循环代码。
"""

from __future__ import annotations

# =====================================================================
# 标准库导入
# =====================================================================
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# =====================================================================
# 第三方库导入
# =====================================================================
import ml_collections
import torch
from accelerate import Accelerator

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.generators.trellis2.training_adpter import (
    StageConfig,
    get_stage_config,
    _build_single_optimizer,
    create_trellis2_strategy,
)

# 数据
from edit4shape.datasets.trellis import (
    TrellisDataConfig,
    TrellisDataModule,
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
)

# Pipeline & Chunked
from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin

# Renderer
from edit4shape.renderers.mesh_peeled_trellis2 import MeshPeeledRenderer
from edit4shape.renderers.hybrid_peeled_trellis2 import Hybrid26NormalRenderer
from edit4shape.renderers.ovoxel_trellis2 import load_envmap

# Base
from edit4shape.systems.base import compute_guidance_device, setup_env_and_seed


# =====================================================================
# Trellis2 系统组件类
# =====================================================================

@dataclass
class StageSystem:
    """
    单个阶段的系统组件。
    
    封装 Shape 或 Tex 阶段的 model、optimizer、renderer 和配置。
    
    属性:
        model: Flow Model（训练时由 strategy.get_student 填充）
        optimizer: 优化器（训练时由 _build_single_optimizer 填充）
        renderer: 渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）
            - Shape 阶段: MeshPeeledRenderer (直接渲染 normal，支持梯度)
            - Tex 阶段: MeshPeeledRenderer (渲染 PBR + IBL 着色，支持梯度)
        config: StageConfig 配置（由 get_stage_config 提供，始终非空）
    """
    model: Any = None       # Flow Model
    optimizer: Any = None   # Optimizer
    renderer: Any = None    # Renderer（阶段专用）
    config: Optional[StageConfig] = None  # StageConfig（构建时必传）


# =====================================================================
# 系统类层次
# =====================================================================

@dataclass
class Trellis2System:
    """
    Trellis2 系统基类。
    
    包含所有训练模式共享的组件：
    - pipeline: 共享的生成管道
    - guidance: 共享 Guidance
    - strategy: 训练策略（LoRA / 全参微调）
    - cfg / accelerator: 运行时上下文

    子类按需添加 shape / tex 阶段：
    - ShapeSystem:    仅 shape
    - TexSystem:      shape (frozen) + tex
    - ShapeTexSystem: shape + tex
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory, mode="shape")
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
    """
    
    pipeline: Any = None
    
    # 共享组件
    guidance: Any = None
    
    # 训练策略（LoRA 或 全参微调）
    strategy: Any = None

    # 运行时上下文
    cfg: Any = None
    accelerator: Accelerator = None
    
    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """设置随机种子与确定性运行环境（委托给 base.setup_env_and_seed）。"""
        setup_env_and_seed(cfg)
    
    def prepare_lora(self, cfg: Any, adapter: str = "base", **kwargs) -> "Trellis2System":
        """准备 LoRA 适配器"""
        for module in [self.pipeline, self.guidance]:
            if module is not None and hasattr(module, "set_adapter"):
                module.set_adapter(adapter)
        return self
    
    @property
    def stages(self) -> Dict[str, StageSystem]:
        """子类覆写，返回自身包含的所有阶段。"""
        return {}
    
    def prepare_optimizers(self, accelerator: Accelerator) -> "Trellis2System":
        """
        通用 prepare：遍历 self.stages，对有 optimizer 的阶段做 DDP 包裹。
        """
        if self.strategy is None:
            return self
        for stage in self.stages.values():
            if stage.optimizer is not None:
                stage.model, stage.optimizer = self.strategy.prepare(
                    accelerator,
                    stage.config.model_stage,
                    stage.config.flow_resolution,
                    stage.optimizer,
                )
        return self


@dataclass
class ShapeSystem(Trellis2System):
    """Shape-only 系统。用于仅训练 Shape 阶段的入口。"""
    shape: Optional[StageSystem] = None

    @property
    def stages(self) -> Dict[str, StageSystem]:
        return {"shape": self.shape}


@dataclass
class TexSystem(Trellis2System):
    """
    Tex 系统（含 frozen Shape）。

    shape 阶段始终存在（renderer 用于 evaluate 中的 shape_forward），
    但 shape.model / shape.optimizer 为 None（不训练）。
    """
    shape: Optional[StageSystem] = None
    tex: Optional[StageSystem] = None

    @property
    def stages(self) -> Dict[str, StageSystem]:
        return {"shape": self.shape, "tex": self.tex}


@dataclass
class ShapeTexSystem(Trellis2System):
    """Shape + Tex 双阶段系统。两个阶段均可训练。"""
    shape: Optional[StageSystem] = None
    tex: Optional[StageSystem] = None

    @property
    def stages(self) -> Dict[str, StageSystem]:
        return {"shape": self.shape, "tex": self.tex}


# =====================================================================
# 数据加载
# =====================================================================

def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple:
    """
    构造训练和评估的 DataLoader。
    
    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器
    
    Returns:
        tuple: (train_loader, eval_loader)
    """
    # ---- 构建训练相机配置 ----
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=cfg.data.train.adaptive_distance,
    )
    
    # ---- 构建评估相机配置 ----
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw_range=list(cfg.data.eval.yaw_range),
        pitch_range=list(cfg.data.eval.pitch_range),
        r_range=list(cfg.data.eval.r_range),
        fov_range=list(cfg.data.eval.fov_range),
        adaptive_distance=cfg.data.eval.adaptive_distance,
    )
    
    # ---- 构建完整数据配置 ----
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,
        eval_batch_size=cfg.data.eval.batch_size,
        width=cfg.render_base.resolution,
        height=cfg.render_base.resolution,
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    # ---- 创建 DataModule 并设置分布式 ----
    dm = TrellisDataModule(
        dm_cfg, 
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dm.setup()

    # ---- 返回 DataLoader ----
    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader


# =====================================================================
# build_system 公共子函数
# =====================================================================

def _build_render_opts_base(cfg: Any) -> Dict[str, Any]:
    """提取渲染器基础配置（resolution, ssaa, near, far, peel_layers）。"""
    return {
        "resolution": cfg.render_base.resolution,
        "ssaa": cfg.render_base.ssaa,
        "near": cfg.render_base.near,
        "far": cfg.render_base.far,
        "peel_layers": cfg.render_base.peel_layers,
    }


def _build_pipeline_and_inject(
    cfg: Any,
    accelerator: Accelerator,
    inject_decoders: List[str] = ("shape",),
) -> Any:
    """
    构建 Pipeline 并注入 ChunkedDecoder。
    
    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器
        inject_decoders: 需要注入 ChunkedDecoder 的阶段列表，
            如 ["shape"] 或 ["shape", "tex"]
    
    Returns:
        构建好的 Pipeline 实例
    """
    pipeline = build_pipeline_from_reference(cfg, accelerator)

    decoder_keys = {"shape": "shape_slat_decoder", "tex": "tex_slat_decoder"}
    for stage in inject_decoders:
        decoder = pipeline.pipe.models[decoder_keys[stage]]
        ChunkedDecoderMixin.inject_to(decoder)

    injected = "/".join(inject_decoders)
    logging.info(f"[Trellis2] {injected} decoder 已启用 chunked forward（自适应显存）")
    
    return pipeline


def _build_hybrid26_renderer(cfg: Any, device: str, trainable: bool = True) -> Any:
    """构建 Hybrid26NormalRenderer（26-neighbor voxel normal 路径）。"""
    render_opts = _build_render_opts_base(cfg)
    if trainable:
        render_opts["grad_checkpoint"] = cfg.shape.renderer.grad_checkpoint
    renderer = Hybrid26NormalRenderer(rendering_options=render_opts, device=device)
    logging.info(f"[Trellis2] Shape renderer: Hybrid26NormalRenderer (trainable={trainable})")
    return renderer


def _build_mesh_peeled_shape_renderer(cfg: Any, device: str, trainable: bool = True) -> Any:
    """构建 MeshPeeledRenderer（face normal 路径）。"""
    render_opts = _build_render_opts_base(cfg)
    if trainable:
        render_opts["grad_checkpoint"] = cfg.shape.renderer.grad_checkpoint
    renderer = MeshPeeledRenderer(rendering_options=render_opts, device=device)
    logging.info(f"[Trellis2] Shape renderer: MeshPeeledRenderer (trainable={trainable})")
    return renderer


def _build_shape_renderer(cfg: Any, device: str, trainable: bool = True) -> Any:
    """根据 cfg.shape.renderer.type 构建 Shape 阶段渲染器。"""
    renderer_type = cfg.shape.renderer.type
    if renderer_type == "hybrid26_peeled":
        return _build_hybrid26_renderer(cfg, device, trainable)
    elif renderer_type == "mesh_peeled":
        return _build_mesh_peeled_shape_renderer(cfg, device, trainable)
    else:
        raise ValueError(f"Unknown shape renderer type: {renderer_type}")


def _build_tex_renderer(cfg: Any, device: str) -> Any:
    """
    构建 Tex 阶段渲染器（含 envmap 加载和冻结）。
    """
    render_opts = _build_render_opts_base(cfg)
    renderer = MeshPeeledRenderer(rendering_options=render_opts, device=device)

    logging.info(f"[Trellis2] 加载环境贴图: {cfg.tex.renderer.envmap_path}")
    renderer.envmap = load_envmap(cfg.tex.renderer.envmap_path, device=device)

    # 冻结 envmap（不优化环境光，只优化纹理）
    # EnvironmentLight 构造函数会强制 base 为 nn.Parameter(requires_grad=True)，
    # 关掉梯度后重建 mips，使 specular/diffuse 从源头就无梯度
    _envlight = renderer.envmap._backend
    _envlight.base.requires_grad_(False)
    _envlight.build_mips()

    return renderer


def _build_training_components(
    cfg: Any,
    pipeline: Any,
    accelerator: Accelerator,
    guidance_factory: callable,
    train_stages: List[str],
    stage_systems: Dict[str, StageSystem],
    checkpoint_stages: List[str],
) -> Tuple[Any, Any]:
    """
    构建训练组件：Guidance、Strategy、Model/Optimizer，并启用 Gradient Checkpointing。

    如果 cfg.eval_only 为 True，返回 (None, None)。
    
    Args:
        cfg: 配置对象
        pipeline: 已构建的 Pipeline
        accelerator: Accelerate 加速器
        guidance_factory: Guidance 工厂函数
        train_stages: 需要训练的阶段列表（如 ["shape"]、["tex"]、["shape", "tex"]）
        stage_systems: 阶段名 → StageSystem 映射，会原地修改（设置 model 和 optimizer）
        checkpoint_stages: 需要启用 gradient checkpointing 的阶段列表
    
    Returns:
        (guidance, strategy) 元组
    """
    if cfg.eval_only:
        return None, None

    guidance = guidance_factory(cfg.guidance_init, train_device=accelerator.device)

    # 训练模式取自第一个可训练阶段的配置
    train_mode = getattr(cfg, train_stages[0]).train.mode
    train_device = accelerator.device
    teacher_device = compute_guidance_device(accelerator.device)

    strategy = create_trellis2_strategy(
        mode=train_mode,
        pipeline=pipeline,
        train_device=train_device,
        teacher_device=teacher_device,
        pipeline_type=cfg.pipeline_type,
        stages=train_stages,
        lora_cfg=getattr(cfg, "lora", None),
        pretrained_path=cfg.pretrained.model,
    )
    strategy.setup()

    # 获取学生模型 + 构建优化器
    for stage_name in train_stages:
        stage_sys = stage_systems[stage_name]
        model = strategy.get_student(stage_name, stage_sys.config.flow_resolution)
        optimizer = _build_single_optimizer(
            model, getattr(cfg, stage_name).train.optimizer
        )
        stage_sys.model = model
        stage_sys.optimizer = optimizer

    # 启用 Gradient Checkpointing
    decoder_keys = {"shape": "shape_slat_decoder", "tex": "tex_slat_decoder"}
    for stage_name in checkpoint_stages:
        stage_sys = stage_systems[stage_name]
        pipeline._set_decoder_checkpointing(decoder_keys[stage_name], enable=True)
        pipeline._set_flow_model_checkpointing(
            stage_name, stage_sys.config.flow_resolution, enable=True
        )

    ckpt_info = "/".join(checkpoint_stages)
    logging.info(f"[Trellis2] 已启用 {ckpt_info} 的 gradient checkpointing")

    return guidance, strategy


# =====================================================================
# 统一系统构建工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
    mode: Literal["shape", "tex", "shape_tex"] = "shape_tex",
) -> Trellis2System:
    """
    构建 Trellis2 训练系统（统一入口）。

    根据 mode 自动决定：
    - 注入哪些 ChunkedDecoder
    - Shape 渲染器是否可训练
    - 是否构建 Tex 渲染器
    - 训练哪些阶段的 Flow Model
    - 启用哪些阶段的 Gradient Checkpointing
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
        mode: 训练模式
            - "shape": 仅训练 Shape → 返回 ShapeSystem
            - "tex": 仅训练 Tex（Shape 冻结）→ 返回 TexSystem
            - "shape_tex": 同时训练 Shape + Tex → 返回 ShapeTexSystem
    
    Returns:
        ShapeSystem / TexSystem / ShapeTexSystem
    """
    # ---- 根据 mode 推导参数 ----
    has_tex = mode in ("tex", "shape_tex")
    shape_trainable = mode in ("shape", "shape_tex")
    inject_decoders = ["shape", "tex"] if has_tex else ["shape"]
    train_stages = {
        "shape": ["shape"],
        "tex": ["tex"],
        "shape_tex": ["shape", "tex"],
    }[mode]
    checkpoint_stages = ["shape", "tex"] if has_tex else ["shape"]

    # ---- 构建 Pipeline ----
    pipeline = _build_pipeline_and_inject(cfg, accelerator, inject_decoders=inject_decoders)
    device = str(accelerator.device)
    pipeline_type = cfg.pipeline_type

    # ---- 构建 StageSystem ----
    shape_config = get_stage_config(pipeline_type, "shape")
    shape_stage = StageSystem(
        config=shape_config,
        renderer=_build_shape_renderer(cfg, device, trainable=shape_trainable),
    )

    tex_stage = None
    if has_tex:
        tex_config = get_stage_config(pipeline_type, "tex")
        tex_stage = StageSystem(
            config=tex_config,
            renderer=_build_tex_renderer(cfg, device),
        )
        
    # ---- 构建训练组件 ----
    stage_systems: Dict[str, StageSystem] = {"shape": shape_stage}
    if tex_stage is not None:
        stage_systems["tex"] = tex_stage

    guidance, strategy = _build_training_components(
        cfg, pipeline, accelerator, guidance_factory,
        train_stages=train_stages,
        stage_systems=stage_systems,
        checkpoint_stages=checkpoint_stages,
    )
        
    # ---- 共享基础参数 ----
    base_kwargs = dict(
        pipeline=pipeline,
        guidance=guidance,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )
                    
    # ---- 按 mode 构建对应子类 ----
    if mode == "shape":
        return ShapeSystem(shape=shape_stage, **base_kwargs)
    elif mode == "tex":
        return TexSystem(shape=shape_stage, tex=tex_stage, **base_kwargs)
    else:  # shape_tex
        return ShapeTexSystem(shape=shape_stage, tex=tex_stage, **base_kwargs)
