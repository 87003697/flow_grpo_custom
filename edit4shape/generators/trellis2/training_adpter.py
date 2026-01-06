#!/usr/bin/env python3
"""
TRELLIS.2 训练适配器（精简版）。

核心功能：
- StageConfig: 阶段配置数据类
- SparseLinearLora: LoRA 层实现，仅对 feats 路径施加 LoRA
- register_sparse_linear_with_peft: 将 SparseLinear 的 LoRA 注册到 PEFT
- build_optimizer_for_stage: 根据阶段构建优化器
- set_stage_trainable: 设置模型冻结/解冻状态
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

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
    """为单个模型创建优化器。"""
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
    else:
        return optim.AdamW(
            trainable,
            lr=float(opt_cfg.lr),
            betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )


def build_optimizer_for_stage(
    pipeline: Any,
    pipeline_type: str,
    stage_name: Union[str, List[str]],
    opt_cfg: Any,
) -> Union[Optional[optim.Optimizer], List[Optional[optim.Optimizer]]]:
    """
    根据训练阶段构建 optimizer。
    
    Args:
        pipeline: Trellis2RefAdapter 实例
        pipeline_type: "512" | "1024" | "1024_cascade" | "1536_cascade"
        stage_name: 阶段名称或列表（如 "shape", "tex", ["shape", "tex"]）
        opt_cfg: 优化器配置（需含 type/lr/beta1/beta2/eps/weight_decay）
    
    Returns:
        单个阶段返回 Optimizer，多个阶段返回 list[Optimizer]
    """
    stages = [stage_name] if isinstance(stage_name, str) else stage_name
    
    optimizers = []
    for stage in stages:
        config = get_stage_config(pipeline_type, stage)
        model = pipeline.get_flow_model(config.model_stage, config.flow_resolution)
        optimizers.append(_build_single_optimizer(model, opt_cfg))
    
    return optimizers[0] if len(optimizers) == 1 else optimizers


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
        print(f"[Training] Stage={stage}: {model_name} 可训练")
        configs.append(config)
    
    return configs[0] if len(configs) == 1 else configs
