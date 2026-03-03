"""
Trellis 系统工厂 — 构建 Pipeline、Renderer、Strategy、Guidance、Optimizer。

本模块仅包含：
1. _CONFIG:           absl config_flags 配置入口
2. build_system:      系统组件工厂函数
3. build_dataloaders: 训练/评估 DataLoader 工厂

前向/渲染/评估逻辑见 forward.py；三阶段 Autograd 见 phases.py；
训练入口见 entries/ 子目录。
"""

# =====================================================================
# 标准库导入
# =====================================================================
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# =====================================================================
# 第三方库导入
# =====================================================================
import numpy as np
import ml_collections
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

# =====================================================================
# TRELLIS / TripoSF 参考实现路径设置（必须在相关导入之前）
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.datasets.trellis import (
    TrellisDataConfig,
    TrellisDataModule,
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
)
from edit4shape.systems.base import compute_guidance_device

# Pipeline
from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference

# Renderer
from edit4shape.renderers.base_renderer import RenderConfig
from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer

# Strategy & Optimizer
from edit4shape.generators.trellis.training_adpter import (
    register_sparse_linear_with_peft,
    inject_lora_to_slat,
    build_optimizer_for_slat,
    TrellisFullFinetuneStrategy,
    TrellisLoRAStrategy,
    TrellisFrozenStrategy,
)

# 使用 absl 的 config_flags 管理配置文件
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")


# =====================================================================
# TrellisSystem — Trellis v1 专用系统容器
# =====================================================================

@dataclass
class TrellisSystem:
    """
    Trellis v1 系统核心组件容器。

    与 Trellis2System 平行的独立设计：
    - pipeline:    共享生成管道
    - renderers:   渲染器字典 {"mesh": ..., "gs": ...}，支持 hybrid 双路渲染
    - guidance:    指导模块
    - optimizer:   优化器（单 flow model，不分 stage）
    - strategy:    训练策略（LoRA / Full / Frozen）
    - cfg:         运行时配置（供 StageOps 查询）
    - accelerator: Accelerate 加速器（供 StageOps 查询）
    """

    pipeline: Any = None
    renderers: Dict[str, Any] = field(default_factory=dict)  # "mesh"/"gs" → renderer
    guidance: Any = None
    optimizer: Any = None
    strategy: Any = None  # TrainingStrategy 实例
    cfg: Any = None
    accelerator: Any = None  # Accelerator

    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """
        设置随机种子与确定性运行环境。

        确保实验可复现性，设置以下随机源的种子：
        - Python random / NumPy / PyTorch CPU+CUDA / cuDNN
        """
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_lora(
        self,
        cfg: Any,
        adapter: str = "base",
        load_path: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> "TrellisSystem":
        """准备 LoRA 适配器。"""
        target_modules = [m for m in [self.pipeline, self.guidance] if hasattr(m, "set_adapter")]
        for module in target_modules:
            if load_path and hasattr(module, "load_adapter"):
                module.load_adapter(load_path, adapter_name=adapter)
            module.set_adapter(adapter)
        return self

    def prepare_models_and_optimizers(self, cfg: Any, accelerator: Accelerator) -> "TrellisSystem":
        """
        通过 strategy.prepare() 注册模型到 DDP 并包装优化器。

        即使 optimizer 为 None（eval_only 模式），也需要调用 prepare
        将模型注册到 accelerator，否则 accelerator.load_state() 无法恢复权重。
        """
        if accelerator is None:
            return self
        if self.strategy is not None:
            self.optimizer = self.strategy.prepare(accelerator, self.optimizer)
        return self


# =====================================================================
# 向后兼容的 re-export
# （外部脚本可能 from edit4shape.systems.trellis.system import trellis_forward 等）
# =====================================================================
from edit4shape.systems.trellis.forward import (  # noqa: F401
    decode_and_render_mesh,
    decode_and_render_gs,
    trellis_forward,
    evaluate,
)


# =====================================================================
# Sparse Mesh Extractor 注入
# =====================================================================

def inject_sparse_mesh_extractor(pipeline, device: str = "cuda") -> None:
    """
    将 SLatMeshDecoder 的 mesh_extractor 替换为基于 TripoSF 稀疏 FlexiCubes 的版本。

    TripoSF 的稀疏实现避免了构建 (res+1)^3 的稠密网格，在高分辨率（如 256^3）下
    显著降低显存占用，同时保持与 TRELLIS decoder 输出的兼容性。

    Args:
        pipeline: TrellisRefAdapter 实例，内部持有 pipe.models['slat_decoder_mesh']
        device: 设备字符串，如 "cuda:0"
    """
    from edit4shape.generators.trellis.sparse_mesh_extractor import (
        SparseFeatures2Mesh as SparseSparseFeatures2Mesh,
    )

    decoder = pipeline.pipe.models.get("slat_decoder_mesh")
    if decoder is None:
        return

    original = decoder.mesh_extractor
    sparse_extractor = SparseSparseFeatures2Mesh(
        device=device,
        res=original.res,
        use_color=False # getattr(original, "use_color", False),
    )
    decoder.mesh_extractor = sparse_extractor
    print(
        f"[SparseMeshExtractor] Injected TripoSF sparse FlexiCubes into "
        f"slat_decoder_mesh (res={original.res})"
    )


# =====================================================================
# build_system 公共子函数
# =====================================================================

def _build_pipeline(cfg: ml_collections.ConfigDict, accelerator: Accelerator):
    """
    构建 Pipeline（核心生成管道）。

    Pipeline 负责：条件编码、结构/特征采样、解码。

    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器

    Returns:
        构建好的 Pipeline 实例
    """
    return build_pipeline_from_reference(cfg, accelerator)


def _build_renderer_of_type(cfg: ml_collections.ConfigDict, device: str, renderer_type: str):
    """
    按指定类型构建单个 Renderer。

    near/far 从 cfg.renderer.{renderer_type}.near/far 读取。

    Args:
        cfg: 完整配置对象（读取 renderer.resolution 等公共参数）
        device: 设备字符串，如 "cuda:0"
        renderer_type: "mesh" 或 "gs"

    Returns:
        渲染器实例
    """
    per_renderer = cfg.renderer[renderer_type]
    render_cfg = RenderConfig(
        resolution=cfg.renderer.resolution,
        near=per_renderer.near,
        far=per_renderer.far,
        ssaa=cfg.renderer.ssaa,
        bg_color=per_renderer.bg_color,
    )

    if renderer_type == "gs":
        return GaussianRenderer(config=render_cfg, device=device)
    else:
        return TrellisMeshRasterizer(config=render_cfg, device=device)


def _build_renderer(cfg: ml_collections.ConfigDict, device: str):
    """
    向后兼容：按 cfg.renderer.type 构建单个 Renderer。
    """
    return _build_renderer_of_type(cfg, device, cfg.renderer.type)


def _build_strategy(
    cfg: ml_collections.ConfigDict,
    pipeline,
    accelerator: Accelerator,
):
    """
    构建 Strategy（训练策略：LoRA / Full / Frozen）。

    Strategy 决定了参数冻结/解冻、学生/教师模型的放置方式。
    训练和推理模式均需要调用。

    Args:
        cfg: 完整配置对象
        pipeline: 已构建的 Pipeline 实例
        accelerator: Accelerate 加速器

    Returns:
        已 setup 的 Strategy 实例
    """
    train_mode = cfg.train.get("mode", "full")
    train_device = accelerator.device
    teacher_device = (
        train_device if cfg.eval_only
        else compute_guidance_device(accelerator.device)
    )

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
    return strategy


def _build_training_components(
    cfg: ml_collections.ConfigDict,
    pipeline,
    strategy,
    accelerator: Accelerator,
    guidance_factory: callable,
):
    """
    构建训练组件：Guidance、Gradient Checkpointing、Optimizer。

    仅在训练模式下调用（cfg.eval_only == False）。
    如果 cfg.eval_only 为 True，返回 (None, None)。

    Args:
        cfg: 完整配置对象
        pipeline: 已构建的 Pipeline 实例
        strategy: 已构建的 Strategy 实例
        accelerator: Accelerate 加速器
        guidance_factory: Guidance 工厂函数

    Returns:
        (guidance, optimizer) 元组
    """
    if cfg.eval_only:
        return None, None

    # 1. 使用工厂函数创建 Guidance
    guidance = guidance_factory(cfg.guidance, train_device=accelerator.device)

    # 2. 启用 slat_flow_model 的 Gradient Checkpointing
    slat_model = pipeline._resolve_slat_flow_module()
    for block in slat_model.blocks:
        block.use_checkpoint = True

    # 3. 也为 slat_decoder_gs 启用 Gradient Checkpointing
    decoder_gs = pipeline.pipe.models.get('slat_decoder_gs')
    if decoder_gs is not None and hasattr(decoder_gs, 'blocks'):
        for block in decoder_gs.blocks:
            block.use_checkpoint = True

    # 4. 为学生模型创建优化器
    optimizer = build_optimizer_for_slat(strategy.student, cfg.train.optimizer)

    return guidance, optimizer


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
) -> TrellisSystem:
    """
    构建完整的 Trellis 系统（编排入口）。

    依次调用各子函数构建：
    1. _build_pipeline   → Pipeline
    2. _build_renderer   → Renderer（放入 renderers dict）
    3. _build_strategy   → Strategy
    4. _build_training_components → Guidance + Optimizer

    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数

    Returns:
        TrellisSystem: 包含所有组件的系统实例
    """
    device = str(accelerator.device)

    pipeline = _build_pipeline(cfg, accelerator)
    renderer = _build_renderer(cfg, device)
    renderer_key = cfg.renderer.type  # "mesh" 或 "gs"
    strategy = _build_strategy(cfg, pipeline, accelerator)
    guidance, optimizer = _build_training_components(
        cfg, pipeline, strategy, accelerator, guidance_factory,
    )

    return TrellisSystem(
        pipeline=pipeline,
        renderers={renderer_key: renderer},
        guidance=guidance,
        optimizer=optimizer,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )


def build_hybrid_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
) -> TrellisSystem:
    """
    构建双路渲染系统 — 同时持有 mesh + gs 渲染器。

    与 build_system 的唯一区别：renderers 字典包含两个渲染器
    {"mesh": TrellisMeshRasterizer, "gs": GaussianRenderer}。

    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数

    Returns:
        TrellisSystem: 包含双路渲染器的系统实例
    """
    device = str(accelerator.device)

    pipeline = _build_pipeline(cfg, accelerator)
    # 硬编码注入 TripoSF 稀疏 FlexiCubes mesh extractor
    inject_sparse_mesh_extractor(pipeline, device=device)
    strategy = _build_strategy(cfg, pipeline, accelerator)
    guidance, optimizer = _build_training_components(
        cfg, pipeline, strategy, accelerator, guidance_factory,
    )

    return TrellisSystem(
        pipeline=pipeline,
        renderers={
            "mesh": _build_renderer_of_type(cfg, device, "mesh"),
            "gs": _build_renderer_of_type(cfg, device, "gs"),
        },
        guidance=guidance,
        optimizer=optimizer,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )


def build_flowedit_system(
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    guidance_factory: callable,
) -> TrellisSystem:
    """
    构建 FlowEdit 训练系统 — Pretrained Rollout + Finetuned 单步去噪。

    与 build_system 相同，但确保：
    - 使用 gs 渲染器（cfg.renderer.type = "gs"）
    - decoder 不参与优化但保留计算图

    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数

    Returns:
        TrellisSystem: 系统实例
    """
    return build_system(cfg, accelerator, guidance_factory)


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练和评估的 DataLoader。

    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器

    Returns:
        tuple: (train_loader, eval_loader)
    """
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=cfg.data.train.adaptive_distance,
    )

    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw_range=list(cfg.data.eval.yaw_range),
        pitch_range=list(cfg.data.eval.pitch_range),
        r_range=list(cfg.data.eval.r_range),
        fov_range=list(cfg.data.eval.fov_range),
        adaptive_distance=cfg.data.eval.adaptive_distance,
    )

    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,
        eval_batch_size=cfg.data.eval.batch_size,
        width=cfg.renderer.resolution,
        height=cfg.renderer.resolution,
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    dm = TrellisDataModule(
        dm_cfg,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dm.setup()

    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader
