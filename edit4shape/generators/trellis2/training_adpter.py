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

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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
        return create_optimizer_v2(trainable, opt="adan", lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay), betas=(0.98, 0.92, 0.99))

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
from pathlib import Path


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
import json
from dataclasses import dataclass
from accelerate import Accelerator


@dataclass
class Trellis2CheckpointIO:
    """
    Trellis2 专属检查点管理类。
    
    保存内容：
    - 模型权重（委托给 strategy.save_student / load_student，
      LoRA 模式保存 adapter 权重，全参模式保存 state_dict）
    - Accelerate 状态（优化器 + RNG + DataLoader sampler）
    - 训练元信息（epoch, global_step, stages）
    
    目录结构:
        checkpoint_{epoch}_{global_step}/
        ├── lora_shape/      # LoRA 模式：Shape LoRA 权重
        ├── lora_tex/        # LoRA 模式：Tex LoRA 权重
        ├── full_*.pt        # 全参模式：全参权重
        ├── state.json       # Accelerate 状态索引
        ├── optimizer_*/     # 优化器状态
        ├── random_states_*  # RNG 状态
        └── meta.json        # 训练元信息
    """

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(
        self,
        system: Any,
        epoch: int,
        global_step: int,
        stages: List[str],
    ) -> None:
        """
        保存检查点：模型权重 + Accelerate 状态 + meta。
        
        模型权重保存委托给 system.strategy.save_student()，
        自动适配 LoRA / 全参 / 冻结模式。
        
        Args:
            system: Trellis2System 实例（需有 strategy 属性）
            epoch: 当前 epoch
            global_step: 当前步数
            stages: 要保存的阶段列表，如 ["shape"], ["tex"], ["shape", "tex"]
        """
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"
        target.mkdir(parents=True, exist_ok=True)
        
        self.accelerator.wait_for_everyone()
        
        # 1. 保存 Accelerate 状态（优化器 + RNG + sampler）
        self.accelerator.save_state(str(target))
        
        # 2. 保存模型权重（仅主进程，委托给 strategy）
        if self.accelerator.is_main_process:
            if system.strategy is not None:
                system.strategy.save_student(target, stages)
            
            # 3. Meta（包含保存的阶段信息）
            meta = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "stages": stages,
            }
            with (target / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        self.accelerator.wait_for_everyone()

    def load(
        self,
        path: str,
        system: Any,
        stages: List[str],
        mode: str = "train",
    ) -> int:
        """
        加载检查点：模型权重 + Accelerate 状态。
        
        模型权重加载委托给 system.strategy.load_student()。
        
        Args:
            path: 检查点路径
            system: Trellis2System 实例（需有 strategy 属性）
            stages: 要加载的阶段，如 ["shape"], ["tex"], ["shape", "tex"]
            mode: "train" 返回 epoch+1，"eval" 返回 0
        
        Returns:
            起始 epoch
        """
        cp = path
        if not (isinstance(cp, str) and cp):
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        root = Path(cp)
        if not root.is_dir():
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        self.accelerator.wait_for_everyone()
        
        # 1. 加载 Accelerate 状态（优化器 + RNG + sampler）
        if (root / "state.json").exists():
            self.accelerator.load_state(str(root))
        
        # 2. 读取 meta
        meta_path = root / "meta.json"
        if meta_path.exists():
            meta = json.load(meta_path.open("r", encoding="utf-8")) or {}
            epoch_val = meta.get("epoch", 0)
            step_val = meta.get("global_step", 0)
        else:
            epoch_val, step_val = 0, 0
        
        # 3. 加载模型权重（委托给 strategy）
        if system.strategy is not None:
            system.strategy.load_student(root, stages)
        
        self.accelerator.wait_for_everyone()
        
        self.start_epoch = int(epoch_val) + 1 if mode == "train" else 0
        self.start_global_step = int(step_val)
        return self.start_epoch


