#!/usr/bin/env python3
"""
TRELLIS.2 训练适配器。

核心功能：
- StageConfig: 阶段配置数据类
- SparseLinearLora: LoRA 层实现，仅对 feats 路径施加 LoRA
- register_sparse_linear_with_peft: 将 SparseLinear 的 LoRA 注册到 PEFT
- set_stage_trainable: 设置模型冻结/解冻状态
- prepare_stage_for_training: 注入 LoRA + 构建优化器 + 回写 pipeline
- save_stage_lora: 保存单个阶段的 LoRA 权重
- load_stage_lora: 加载单个阶段的 LoRA 权重
- Trellis2CheckpointIO: Trellis2 专属检查点管理类
"""
from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# TRELLIS.2 参考实现路径设置
# =====================================================================
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.linear import SparseLinear

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora.layer import LoraLayer

from .pipeline_adapter import PIPELINE_CONFIGS

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape", "tex", "shape_stage1", "shape_stage2"]


# =====================================================================
# StageConfig - 阶段配置数据类
# =====================================================================
@dataclass
class StageConfig:
    """
    单个阶段的配置信息。
    
    存储 Shape 或 Tex 阶段的分辨率配置。
    由 set_stage_trainable 返回，供 Trellis2System 使用。
    """
    ss_resolution: int        # Dense Sampling 分辨率
    cond_resolution: int      # 条件编码分辨率
    flow_resolution: int      # Flow Model 分辨率
    render_type: str          # 渲染类型："normal" 或 "rgb"
    model_stage: str          # 模型阶段："shape" 或 "tex"


# =====================================================================
# 阶段配置辅助函数
# =====================================================================
def get_render_type(stage_name: str) -> str:
    """获取阶段对应的渲染类型。shape 相关阶段用 normal，tex 阶段用 rgb。"""
    return "normal" if stage_name.startswith("shape") else "rgb"


def get_stage_config(pipeline_type: str, stage_name: str) -> StageConfig:
    """
    从 PIPELINE_CONFIGS 获取阶段配置。
    
    Args:
        pipeline_type: "512" | "1024" | "1024_cascade" | "1536_cascade"
        stage_name: "shape" | "tex" | "shape_stage1" | "shape_stage2"
    
    Returns:
        StageConfig: 阶段配置
    """
    if pipeline_type not in PIPELINE_CONFIGS:
        raise ValueError(f"未知的 pipeline_type: {pipeline_type}")
    
    stages = PIPELINE_CONFIGS[pipeline_type]["stages"]
    if stage_name not in stages:
        raise ValueError(f"pipeline_type={pipeline_type} 不存在 stage={stage_name}，可用: {list(stages.keys())}")
    
    stage_cfg = stages[stage_name]
    return StageConfig(
        ss_resolution=stage_cfg["ss_resolution"],
        cond_resolution=stage_cfg["cond_resolution"],
        flow_resolution=stage_cfg["flow_resolution"],
        render_type=get_render_type(stage_name),
        model_stage="shape" if stage_name.startswith("shape") else "tex",
    )


# =====================================================================
# SparseLinear LoRA 层
# =====================================================================
class SparseLinearLora(nn.Module, LoraLayer):
    """LoRA for TRELLIS.2 SparseLinear.

    注意：forward 接收 SparseTensor，内部仅对 feats 施加 LoRA，再 replace 到 SparseTensor。
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def forward(self, x: SparseTensor, *args: Any, **kwargs: Any) -> SparseTensor:
        """前向传播，仅对 feats 施加 LoRA。"""
        # 基线输出（保持 SparseTensor 结构）
        if self.disable_adapters or self.merged:
            return self.get_base_layer()(x, *args, **kwargs)

        out: SparseTensor = self.get_base_layer()(x, *args, **kwargs)  # feats: (N, Cout)
        out_dtype = out.feats.dtype

        # 逐 adapter 叠加 LoRA 贡献，仅作用于 feats
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            feats = x.feats.to(lora_A.weight.dtype)  # (N, Cin)
            lora_feats = lora_B(lora_A(dropout(feats))) * scaling  # (N, Cout)
            out = out.replace(out.feats + lora_feats.to(out_dtype))  # feats: (N, Cout)

        return out

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """合并 LoRA 权重到基线模型。"""
        adapter_names = list(self.active_adapters) if adapter_names is None else adapter_names
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                delta = self.get_delta_weight(active_adapter)
                base = self.get_base_layer()
                base.weight.data = base.weight.data + delta
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """从基线模型中移除 LoRA 权重。"""
        if not self.merged:
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                delta = self.get_delta_weight(active_adapter)
                base = self.get_base_layer()
                base.weight.data = base.weight.data - delta

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """计算 LoRA 权重增量。"""
        weight_A = self.lora_A[adapter].weight  # (r, in_features)
        weight_B = self.lora_B[adapter].weight  # (out_features, r)
        return (weight_B @ weight_A) * self.scaling[adapter]  # (out_features, in_features)


# =====================================================================
# PEFT LoRA 注册
# =====================================================================
def register_sparse_linear_with_peft() -> None:
    """将 SparseLinear 的 LoRA 注入注册到 PEFT 的 dispatch 函数中。"""
    from peft.tuners.lora import layer as lora_layer_mod

    orig_dispatch = lora_layer_mod.dispatch_default

    def _dispatch(target: torch.nn.Module, adapter_name: str, lora_config, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base = target.get_base_layer()
        else:
            target_base = target

        if isinstance(target_base, SparseLinear):
            return SparseLinearLora(
                target,
                adapter_name,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=False,
                **kwargs,
            )

        return orig_dispatch(target, adapter_name, lora_config, **kwargs)

    lora_layer_mod.dispatch_default = _dispatch


# =====================================================================
# 优化器构建
# =====================================================================
def _build_single_optimizer(model: Any, opt_cfg: Any) -> Optional[optim.Optimizer]:
    """为单个模型创建优化器（对齐 trellis 的多优化器分支）。"""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return None

    opt_type = str(opt_cfg.type).lower()

    if opt_type == 'adam_8bit':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            trainable,
            lr=float(opt_cfg.lr),
            betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )

    from timm.optim.optim_factory import create_optimizer_v2

    if opt_type == "sgd":
        return create_optimizer_v2(trainable, opt="sgd", lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))

    if opt_type == "adan":
        return create_optimizer_v2(
            trainable,
            opt="adan",
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
            betas=(0.98, 0.92, 0.99),
            eps=float(opt_cfg.eps),
        )

    return create_optimizer_v2(trainable, opt=opt_type, lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay), betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)), eps=float(opt_cfg.eps))


# =====================================================================
# 模型冻结/解冻
# =====================================================================
def set_stage_trainable(
    pipeline: Any,
    pipeline_type: str,
    stage_name: Union[str, List[str]],
) -> Union[StageConfig, List[StageConfig]]:
    """
    设置指定阶段的模型为可训练，其他模型冻结。
    
    Args:
        pipeline: Trellis2RefAdapter 实例
        pipeline_type: "512" | "1024" | "1024_cascade" | "1536_cascade"
        stage_name: 阶段名称或列表（如 "shape", "tex", ["shape", "tex"]）
    
    Returns:
        单个阶段返回 StageConfig，多个阶段返回 list[StageConfig]
    """
    stages = [stage_name] if isinstance(stage_name, str) else stage_name
    
    # 冻结所有 flow model
    for name, model in pipeline.pipe.models.items():
        if 'flow_model' in name and hasattr(model, 'parameters'):
            for p in model.parameters():
                p.requires_grad = False
    
    # 解冻目标模型
    configs = []
    for stage in stages:
        config = get_stage_config(pipeline_type, stage)
        target_model = pipeline.get_flow_model(config.model_stage, config.flow_resolution)
        for p in target_model.parameters():
            p.requires_grad = True
        
        model_name = f"{config.model_stage}_slat_flow_model_{config.flow_resolution}"
        logging.info(f"[Training] Stage={stage}: {model_name} 可训练")
        configs.append(config)
    
    return configs[0] if len(configs) == 1 else configs


# =====================================================================
# LoRA 注入（单阶段）
# =====================================================================

def inject_lora_to_stage(
    pipeline: Any,
    pipeline_type: str,
    stage_name: str,
    lora_cfg: Any,
) -> StageConfig:
    """
    为单个阶段注入 LoRA 适配器。
    
    流程：
    1. 获取阶段配置
    2. 获取对应的 Flow Model
    3. 注入 LoRA 适配器
    4. 回写模型到 pipeline（确保 disable_lora_context 能正确获取）
    
    Args:
        pipeline: Trellis2RefAdapter 实例
        pipeline_type: "512" | "1024" | "1024_cascade" | "1536_cascade"
        stage_name: "shape" | "tex"
        lora_cfg: LoRA 配置对象（需含 lora_rank 字段）
    
    Returns:
        StageConfig: 阶段配置
    """
    from peft import LoraConfig, get_peft_model
    
    # 1. 获取阶段配置
    stage_config = get_stage_config(pipeline_type, stage_name)
    
    # 2. 获取模型
    model = pipeline.get_flow_model(stage_config.model_stage, stage_config.flow_resolution)
    
    # 3. 注入 LoRA
    peft_config = LoraConfig(
        r=lora_cfg.lora_rank,
        lora_alpha=lora_cfg.lora_rank,
        target_modules=["to_qkv", "to_out"],  # Trellis2 SparseMultiHeadAttention 的模块名
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    # 4. 回写到 pipeline（关键！确保 disable_lora_context 能获取注入 LoRA 的模型）
    model_key = f"{stage_config.model_stage}_slat_flow_model_{stage_config.flow_resolution}"
    pipeline.pipe.models[model_key] = model
    
    # 日志
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    logging.info(
        f"[LoRA] {stage_name}: 注入 LoRA (rank={lora_cfg.lora_rank}), "
        f"可训练参数={trainable_count}"
    )
    
    return stage_config


# =====================================================================
# 阶段训练准备（LoRA 注入 + 优化器构建 + 回写 pipeline）- 兼容旧代码
# =====================================================================

def prepare_stage_for_training(
    pipeline: Any,
    pipeline_type: str,
    stage_name: str,
    lora_cfg: Any,
    opt_cfg: Any,
) -> Tuple[nn.Module, optim.Optimizer, StageConfig]:
    """
    为单个阶段准备训练：注入 LoRA + 构建优化器 + 回写 pipeline。
    
    注意：此函数保留用于兼容旧代码。新代码建议使用 inject_lora_to_stage + strategy.get_student。
    
    Args:
        pipeline: Trellis2RefAdapter 实例
        pipeline_type: "512" | "1024" | "1024_cascade" | "1536_cascade"
        stage_name: "shape" | "tex"
        lora_cfg: LoRA 配置对象（需含 lora_rank 字段）
        opt_cfg: 优化器配置对象（需含 type/lr/beta1/beta2/eps/weight_decay）
    
    Returns:
        tuple: (model, optimizer, stage_config)
    """
    # 1. 注入 LoRA
    stage_config = inject_lora_to_stage(pipeline, pipeline_type, stage_name, lora_cfg)
    
    # 2. 获取注入 LoRA 后的模型
    model = pipeline.get_flow_model(stage_config.model_stage, stage_config.flow_resolution)
    
    # 3. 构建优化器
    optimizer = _build_single_optimizer(model, opt_cfg)
    
    return model, optimizer, stage_config


# =====================================================================
# LoRA 权重保存/加载
# =====================================================================


def save_stage_lora(
    model: nn.Module,
    save_dir: Path,
    stage_name: str,
) -> None:
    """
    保存单个阶段的 LoRA 权重。
    
    Args:
        model: 注入了 LoRA 的 PEFT 模型（必须支持 save_pretrained）
        save_dir: 保存目录（会创建 lora_{stage_name} 子目录）
        stage_name: "shape" | "tex"
    """
    save_dir = Path(save_dir)
    lora_dir = save_dir / f"lora_{stage_name}"
    model.save_pretrained(lora_dir)
    logging.info(f"[Checkpoint] 已保存 {stage_name} LoRA 到 {lora_dir}")


def load_stage_lora(
    model: nn.Module,
    load_dir: Path,
    stage_name: str,
    adapter_name: str = "default",
) -> None:
    """
    加载单个阶段的 LoRA 权重。
    
    注意：模型必须已经通过 get_peft_model 注入 LoRA 结构。
    此函数只加载权重，不创建 LoRA 结构。
    
    Args:
        model: 注入了 LoRA 的 PEFT 模型（必须支持 load_adapter）
        load_dir: 检查点目录（包含 lora_{stage_name} 子目录）
        stage_name: "shape" | "tex"
        adapter_name: 适配器名称
    """
    load_dir = Path(load_dir)
    lora_dir = load_dir / f"lora_{stage_name}"
    
    if not lora_dir.exists():
        logging.warning(f"[Checkpoint] {stage_name} LoRA 目录不存在: {lora_dir}，跳过")
        return
    
    model.load_adapter(lora_dir, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    logging.info(f"[Checkpoint] 已加载 {stage_name} LoRA 从 {lora_dir}")


# =====================================================================
# Trellis2CheckpointIO - Trellis2 专属检查点管理
# =====================================================================
from accelerate import Accelerator


@dataclass
class Trellis2CheckpointIO:
    """
    Trellis2 检查点管理类（与 V1 CheckpointIO 对齐）。
    
    使用 Accelerate 的 save_state/load_state 进行分布式安全的检查点操作。
    模型权重通过 accelerator.prepare() 注册后，由 Accelerate 统一管理。
    
    前提：模型和优化器已通过 strategy.prepare() → accelerator.prepare(model, optimizer)
    注册到 Accelerate，save_state/load_state 自动包含模型权重。
    """

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(self, epoch: int, global_step: int) -> None:
        """保存检查点（model + optimizer + random_states + meta）。"""
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"
        target.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        # 1. 保存 Accelerate 状态（优化器 + RNG + sampler）
        self.accelerator.save_state(str(target))
        
        # 2. 保存模型权重（仅主进程，委托给 strategy）
        if self.accelerator.is_main_process:
            meta = {"epoch": int(epoch), "global_step": int(global_step)}
            with (target / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        self.accelerator.wait_for_everyone()

    def load(self, path: str, mode: str = "train") -> int:
        """
        加载检查点（model + optimizer + random_states）。
        
        Args:
            path: 检查点目录路径。
            mode: "train" 时 start_epoch = epoch + 1；否则为 0。
        """
        cp = path
        if not (isinstance(cp, str) and cp):
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        root = Path(cp)
        meta_path = root / "meta.json"
        
        # ★ 如果路径是父目录（没有 meta.json），自动查找最新 checkpoint
        if root.is_dir() and not meta_path.exists():
            candidates = sorted(
                [d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()],
                key=lambda d: json.load((d / "meta.json").open("r", encoding="utf-8")).get("global_step", 0),
            )
            if candidates:
                root = candidates[-1]
                meta_path = root / "meta.json"
                logging.info(f"[Trellis2CheckpointIO] 自动选择最新检查点: {root}")
            else:
                logging.warning(
                    f"[Trellis2CheckpointIO] 目录 {cp} 下未找到任何有效检查点（需包含 meta.json）。"
                    f"将从头开始训练。"
                )
                self.start_epoch = 0
                self.start_global_step = 0
                return 0
        
        if not (root.is_dir() and meta_path.exists()):
            logging.warning(
                f"[Trellis2CheckpointIO] 检查点路径无效或不存在: {root} "
                f"(is_dir={root.is_dir()}, meta_exists={meta_path.exists() if root.is_dir() else 'N/A'}). "
                f"将从头开始训练。"
            )
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        # accelerator 一次性恢复：模型权重 + optimizer + random states
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(str(root))
        self.accelerator.wait_for_everyone()
        
        meta = json.load(meta_path.open("r", encoding="utf-8"))
        self.start_epoch = int(meta["epoch"]) + 1 if mode == "train" else 0
        self.start_global_step = int(meta["global_step"])
        logging.info(
            f"[Trellis2CheckpointIO] 检查点已恢复: {root} "
            f"(epoch={meta['epoch']}, step={meta['global_step']}, "
            f"恢复后 start_epoch={self.start_epoch})"
        )
        return self.start_epoch



# =====================================================================
# Trellis2 训练策略（多阶段）— 统一 ABC 基类
# =====================================================================

class Trellis2TrainingStrategy(ABC):
    """
    Trellis2 多阶段训练策略基类。
    
    所有 Trellis2 训练模式（LoRA / Full / Frozen）都继承此基类，
    并实现以下抽象方法。外部代码只需要依赖此接口。
    
    生命周期:
        strategy = create_trellis2_strategy(...)
        strategy.setup()                     # 注入 LoRA / 解冻 / 冻结
        model = strategy.get_student(...)    # 获取学生模型 → 创建优化器
        model, opt = strategy.prepare(accelerator, stage, res, opt)  # DDP + accelerator 注册
        ...
        # 检查点由 accelerator.save_state / load_state 统一管理
        strategy.export_student(path, stages)  # 导出为可推理格式（LoRA merge 等）
    """
    
    def __init__(
        self,
        pipeline: Any,
        train_device: torch.device,
        teacher_device: torch.device,
        pipeline_type: str,
        stages: List[str],
    ):
        self.pipeline = pipeline
        self.train_device = train_device
        self.teacher_device = teacher_device
        self.pipeline_type = pipeline_type
        self.stages = stages
        self._accelerator: Optional[Accelerator] = None
    
    # ----- 抽象方法 -----
    
    @abstractmethod
    def setup(self) -> None:
        """
        设置模型的可训练状态。
        
        - LoRA: 注入 LoRA 适配器
        - Full: 解冻学生 + 加载冻结教师
        - Frozen: 冻结所有参数
        """
        ...
    
    @abstractmethod
    def get_student(self, stage: str, resolution: int) -> nn.Module:
        """获取指定阶段/分辨率的学生模型（用于构建优化器）。"""
        ...
    
    @abstractmethod
    @contextmanager
    def teacher_context(self, stage: str, resolution: int) -> Generator[None, None, None]:
        """
        教师模型预测上下文。
        
        在此上下文中调用 pipeline.sampling_step 使用教师模型。
        """
        ...
    
    @property
    @abstractmethod
    def has_teacher(self) -> bool:
        """是否有教师模型可用（用于正则化）。"""
        ...
    
    # ----- 可选方法（默认实现） -----
    
    def prepare(
        self,
        accelerator: Accelerator,
        stage: str,
        resolution: int,
        optimizer: optim.Optimizer,
    ) -> Tuple[nn.Module, optim.Optimizer]:
        """
        用 accelerator.prepare() 包装模型+优化器，回写到 pipeline。
        
        与 V1 TrainingStrategy.prepare() 对齐：
        模型和优化器一起 prepare → DDP 包裹 + 注册到 accelerator，
        使 save_state/load_state 自动管理模型权重。
        
        Args:
            accelerator: Accelerate 加速器
            stage: 模型阶段名（如 "shape", "tex"）
            resolution: flow model 分辨率
            optimizer: 该阶段的优化器
        
        Returns:
            (model, optimizer): DDP 包裹后的模型和 prepared 优化器
        """
        self._accelerator = accelerator
        model_key = f"{stage}_slat_flow_model_{resolution}"
        model = self.pipeline.pipe.models[model_key]
        model, optimizer = accelerator.prepare(model, optimizer)
        self.pipeline.pipe.models[model_key] = model  # 回写 DDP 包裹后的模型
        return model, optimizer
    
    def _unwrap(self, model: nn.Module) -> nn.Module:
        """解包 DDP / FSDP 包裹，返回底层模型。"""
        if self._accelerator is not None:
            return self._accelerator.unwrap_model(model)
        return model
    
    def _resolve_flow_model(self, stage: str, resolution: int) -> nn.Module:
        """获取 pipeline 中的 flow model，自动解包 DDP。"""
        return self._unwrap(self.pipeline.get_flow_model(stage, resolution))
    
    def save_student(self, save_dir: Union[str, Path], stages: List[str]) -> None:
        """
        手动保存学生模型权重（供导出等场景使用，训练检查点由 accelerator 统一管理）。
        
        默认实现：保存各阶段 state_dict。子类可覆盖以保存 LoRA-only 等。
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for stage_name in stages:
            config = get_stage_config(self.pipeline_type, stage_name)
            model = self._resolve_flow_model(config.model_stage, config.flow_resolution)
            out_path = save_dir / f"{stage_name}_flow_model_{config.flow_resolution}.pt"
            torch.save(model.state_dict(), out_path)
            logging.info(f"[Export] 已保存 {stage_name} 权重到 {out_path}")
    
    def export_student(self, export_dir: Union[str, Path], stages: List[str]) -> None:
        """
        导出为可推理格式（合并 LoRA / 直接拷贝权重）。
        
        默认实现与 save_student 相同，子类可覆盖以执行 LoRA 合并等。
        """
        self.save_student(export_dir, stages)


class Trellis2LoRAStrategy(Trellis2TrainingStrategy):
    """
    Trellis2 LoRA 训练策略。
    
    - 多阶段 LoRA 注入（shape/tex）
    - teacher 通过 disable_lora_context 获取（禁用 LoRA 即可恢复原始权重）
    """
    
    def __init__(
        self,
        pipeline: Any,
        train_device: torch.device,
        teacher_device: torch.device,
        pipeline_type: str,
        stages: List[str],
        lora_cfg: Any,
    ):
        super().__init__(pipeline, train_device, teacher_device, pipeline_type, stages)
        self.lora_cfg = lora_cfg
    
    def setup(self) -> None:
        """LoRA 设置：注入 LoRA 到指定阶段。"""
        from edit4shape.generators.trellis2.training_adpter import (
            register_sparse_linear_with_peft,
            inject_lora_to_stage,
        )
        register_sparse_linear_with_peft()
        for stage in self.stages:
            inject_lora_to_stage(self.pipeline, self.pipeline_type, stage, self.lora_cfg)
    
    def get_student(self, stage: str, resolution: int) -> nn.Module:
        """获取指定阶段的学生模型（注入 LoRA 后）。"""
        return self.pipeline.get_flow_model(stage, resolution)
    
    @contextmanager
    def teacher_context(self, stage: str, resolution: int) -> Generator[None, None, None]:
        """LoRA 模式：临时禁用 LoRA adapters，使用原始权重。"""
        model = self._resolve_flow_model(stage, resolution)
        if hasattr(model, 'disable_adapters'):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            yield
    
    @property
    def has_teacher(self) -> bool:
        return True
    
    def save_student(self, save_dir: Union[str, Path], stages: List[str]) -> None:
        """保存各阶段 LoRA 权重（通过 PEFT save_pretrained）。"""
        save_dir = Path(save_dir)
        for stage_name in stages:
            config = get_stage_config(self.pipeline_type, stage_name)
            model = self._resolve_flow_model(config.model_stage, config.flow_resolution)
            save_stage_lora(model, save_dir, stage_name)
    
    def export_student(self, export_dir: Union[str, Path], stages: List[str]) -> None:
        """导出合并 LoRA 后的全参权重（merge + save state_dict）。"""
        from edit4shape.generators.trellis2.training_adpter import get_stage_config
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for stage_name in stages:
            config = get_stage_config(self.pipeline_type, stage_name)
            model = self._resolve_flow_model(config.model_stage, config.flow_resolution)
            
            # 合并 LoRA 权重到基础模型
            if hasattr(model, 'merge_and_unload'):
                merged = model.merge_and_unload()
            else:
                merged = model
            
            out_path = export_dir / f"{stage_name}_flow_model_{config.flow_resolution}.pt"
            torch.save(merged.state_dict(), out_path)
            logging.info(f"[Export] 已导出 {stage_name} 合并权重到 {out_path}")


class Trellis2FullFinetuneStrategy(Trellis2TrainingStrategy):
    """
    Trellis2 全参微调策略（多阶段）。
    
    - 解冻学生
    - 加载冻结教师（按 stage + resolution）
    """
    
    FLOW_MODEL_PATHS = {
        ("shape", 512): "slat_flow_img2shape_dit_1_3B_512_bf16",
        ("shape", 1024): "slat_flow_img2shape_dit_1_3B_1024_bf16",
        ("tex", 512): "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        ("tex", 1024): "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    }
    
    def __init__(
        self,
        pipeline: Any,
        train_device: torch.device,
        teacher_device: torch.device,
        pretrained_path: str,
        pipeline_type: str,
        stages: List[str],
    ):
        super().__init__(pipeline, train_device, teacher_device, pipeline_type, stages)
        self.pretrained_path = pretrained_path
        self._teacher_models: Dict[Tuple[str, int], nn.Module] = {}
    
    def setup(self) -> None:
        """解冻学生并加载冻结教师。"""
        from trellis2 import models as trellis2_models
        from edit4shape.generators.trellis2.training_adpter import get_stage_config
        
        total_trainable = 0
        total_teacher_mem = 0
        
        for stage in self.stages:
            config = get_stage_config(self.pipeline_type, stage)
            model_stage = config.model_stage
            resolution = config.flow_resolution
            
            student_model = self.pipeline.get_flow_model(model_stage, resolution)
            for p in student_model.parameters():
                p.requires_grad = True
            total_trainable += sum(p.numel() for p in student_model.parameters() if p.requires_grad)
            
            flow_model_name = self.FLOW_MODEL_PATHS.get((model_stage, resolution))
            if flow_model_name is None:
                raise ValueError(f"未知的 stage/resolution 组合: {model_stage}/{resolution}")
            
            model_path = f"{self.pretrained_path}/ckpts/{flow_model_name}"
            teacher = trellis2_models.from_pretrained(model_path)
            teacher.to(self.teacher_device).eval().requires_grad_(False)
            
            self._teacher_models[(model_stage, resolution)] = teacher
            total_teacher_mem += sum(p.numel() * p.element_size() for p in teacher.parameters())
        
        logging.info(f"[Trellis2FullFinetuneStrategy] 全参微调: {total_trainable:,} 参数可训练")
        logging.info(
            f"[Trellis2FullFinetuneStrategy] 教师模型 → {self.teacher_device} "
            f"({total_teacher_mem / 1e6:.0f} MB)"
        )
        if self.teacher_device == self.train_device:
            logging.warning("[Trellis2FullFinetuneStrategy] 教师与学生在同一设备，显存翻倍")
    
    def get_student(self, stage: str, resolution: int) -> nn.Module:
        """获取指定阶段的学生模型（已解冻）。"""
        return self.pipeline.get_flow_model(stage, resolution)
    
    @contextmanager
    def teacher_context(self, stage: str, resolution: int) -> Generator[None, None, None]:
        """临时替换 pipeline 中的模型为冻结教师。"""
        teacher = self._teacher_models.get((stage, resolution))
        if teacher is None:
            yield
            return
        
        model_key = f"{stage}_slat_flow_model_{resolution}"
        original = self.pipeline.pipe.models[model_key]
        self.pipeline.pipe.models[model_key] = teacher
        try:
            yield
        finally:
            self.pipeline.pipe.models[model_key] = original
    
    @property
    def has_teacher(self) -> bool:
        return True
    
class Trellis2FrozenStrategy(Trellis2TrainingStrategy):
    """冻结策略（多阶段），仅推理。"""
    
    def __init__(
        self,
        pipeline: Any,
        train_device: torch.device,
        teacher_device: torch.device,
        pipeline_type: str,
        stages: List[str],
    ):
        super().__init__(pipeline, train_device, teacher_device, pipeline_type, stages)
    
    def setup(self) -> None:
        """冻结所有阶段的模型参数。"""
        for stage in self.stages:
            config = get_stage_config(self.pipeline_type, stage)
            model = self.pipeline.get_flow_model(config.model_stage, config.flow_resolution)
            for p in model.parameters():
                p.requires_grad = False
        logging.info("[Trellis2FrozenStrategy] 模型冻结（推理模式）")
    
    def get_student(self, stage: str, resolution: int) -> nn.Module:
        """返回冻结的学生模型。"""
        return self.pipeline.get_flow_model(stage, resolution)
    
    @contextmanager
    def teacher_context(self, stage: str, resolution: int) -> Generator[None, None, None]:
        """冻结模式：无教师（不应调用）。"""
        raise RuntimeError("Trellis2FrozenStrategy 不支持正则化，请设置 cfg.reg.type = 'none'")
        yield
    
    @property
    def has_teacher(self) -> bool:
        return False
    
    def prepare(
        self,
        accelerator: Accelerator,
        stage: str,
        resolution: int,
        optimizer: optim.Optimizer,
    ) -> Tuple[nn.Module, optim.Optimizer]:
        """冻结模式无需 DDP 包装。"""
        self._accelerator = accelerator
        return self.pipeline.get_flow_model(stage, resolution), optimizer


def create_trellis2_strategy(
    mode: str,
    pipeline: Any,
    train_device: torch.device,
    teacher_device: torch.device,
    pipeline_type: str,
    stages: List[str],
    lora_cfg: Any = None,
    pretrained_path: str = "",
) -> Trellis2TrainingStrategy:
    """
    Trellis2 策略工厂（多阶段）。
    
    Args:
        mode: "lora" | "full" | "frozen"
        pipeline: Trellis2RefAdapter
        train_device: 训练设备
        teacher_device: 教师设备
        pipeline_type: pipeline 类型
        stages: 训练阶段列表（如 ["shape"] 或 ["shape", "tex"]）
        lora_cfg: LoRA 配置
        pretrained_path: 预训练权重路径
    
    Returns:
        Trellis2TrainingStrategy: 对应的策略实例
    """
    if mode == "lora":
        if lora_cfg is None:
            raise ValueError("train.mode='lora' 时必须提供 cfg.lora 配置。")
        return Trellis2LoRAStrategy(
            pipeline, train_device, teacher_device, pipeline_type, stages, lora_cfg
        )
    if mode == "full":
        return Trellis2FullFinetuneStrategy(
            pipeline, train_device, teacher_device, pretrained_path, pipeline_type, stages
        )
    if mode == "frozen":
        return Trellis2FrozenStrategy(
            pipeline, train_device, teacher_device, pipeline_type, stages
        )
    
    raise ValueError(f"Unknown mode: {mode}. Use 'lora' | 'full' | 'frozen'.")
