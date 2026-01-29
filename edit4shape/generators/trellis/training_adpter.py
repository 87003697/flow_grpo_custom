#!/usr/bin/env python3
"""
TRELLIS 训练适配器。

核心功能：
- SparseLinearLora: LoRA 层实现，仅对 feats 路径施加 LoRA
- register_sparse_linear_with_peft: 将 SparseLinear 的 LoRA 注册到 PEFT
- build_optimizer_for_slat: 为 slat_flow_model 构建优化器
- set_slat_trainable: 设置 slat_flow_model 为可训练（冻结其他模型）
- inject_lora_to_slat: 向 slat_flow_model 注入 LoRA 层
- TrellisFullFinetuneStrategy: Trellis 全参微调策略
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import torch
import torch.nn as nn

from generators.trellis import sparse as sp  # noqa: E402
from generators.trellis.modules.sparse.linear import SparseLinear  # noqa: E402

from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402
from peft.tuners.lora.layer import LoraLayer  # noqa: E402
import torch.optim as optim


class SparseLinearLora(nn.Module, LoraLayer):
    """LoRA for TRELLIS SparseLinear.

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

    def forward(self, x: sp.SparseTensor, *args: Any, **kwargs: Any) -> sp.SparseTensor:
        # 基线输出（保持 SparseTensor 结构）
        if self.disable_adapters or self.merged:
            return self.get_base_layer()(x, *args, **kwargs)

        out: sp.SparseTensor = self.get_base_layer()(x, *args, **kwargs)  # SparseTensor feats shape (N, Cout)
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
            out = out.replace(out.feats + lora_feats.to(out_dtype))  # SparseTensor feats (N, Cout)

        return out

    # 合并/拆分逻辑：直接对基线 nn.Linear 权重做增量，与 peft Linear 一致
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
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
        if not self.merged:
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                delta = self.get_delta_weight(active_adapter)
                base = self.get_base_layer()
                base.weight.data = base.weight.data - delta

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        return (weight_B @ weight_A) * self.scaling[adapter]


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


def build_optimizer_for_slat(
    slat_model: nn.Module,
    opt_cfg: Any,
) -> Optional[optim.Optimizer]:
    """为 slat_flow_model 构建 optimizer。

    Args:
        slat_model: 可训练的 SLatFlowModel
        opt_cfg: 优化器配置（需含 type/lr/beta1/beta2/eps/weight_decay）

    Returns:
        Optimizer 或 None（无可训练参数时）
    """
    trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    if not trainable_params:
        return None

    opt_type = str(opt_cfg.type).lower()
    
    if opt_type == 'adam_8bit':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            trainable_params,
            lr=float(opt_cfg.lr),
            betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)),
            eps=float(opt_cfg.eps),
            weight_decay=float(opt_cfg.weight_decay),
        )
    
    from timm.optim.optim_factory import create_optimizer_v2

    if opt_type == "sgd":
        return create_optimizer_v2(trainable_params, opt="sgd", lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))

    if opt_type == "adan":
        return create_optimizer_v2(trainable_params, opt="adan", lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay), betas=(0.98, 0.92, 0.99))

    return create_optimizer_v2(trainable_params, opt=opt_type, lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay), betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)), eps=float(opt_cfg.eps))


# =====================================================================
# 模型可训练设置
# =====================================================================

def set_slat_trainable(pipeline: Any, trainable: bool = True) -> None:
    """
    设置 slat_flow_model 的可训练状态。
    
    Args:
        pipeline: TrellisRefAdapter 实例
        trainable: True 解冻，False 冻结
    """
    slat_model = pipeline.pipe.models["slat_flow_model"]
    for p in slat_model.parameters():
        p.requires_grad = trainable
    
    status = "可训练" if trainable else "冻结"
    n_params = sum(p.numel() for p in slat_model.parameters())
    print(f"[set_slat_trainable] slat_flow_model {status} ({n_params:,} 参数)")


def inject_lora_to_slat(pipeline: Any, lora_cfg: Any) -> None:
    """
    向 slat_flow_model 注入 LoRA 层。
    
    注意：调用前需先调用 register_sparse_linear_with_peft()。
    
    Args:
        pipeline: TrellisRefAdapter 实例
        lora_cfg: LoRA 配置，需含以下字段:
            - lora_rank: LoRA 秩
            - lora_alpha: LoRA alpha
            - target_modules: 目标模块列表（如 ["to_q", "to_v"]）
            - lora_dropout: dropout 比例（可选，默认 0.0）
    """
    from peft import LoraConfig, get_peft_model
    
    slat_model = pipeline.pipe.models["slat_flow_model"]
    
    # 构建 LoRA 配置
    config = LoraConfig(
        r=int(lora_cfg.lora_rank),
        lora_alpha=int(lora_cfg.get("lora_alpha", lora_cfg.lora_rank)),
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
    )
    
    # 注入 LoRA
    slat_model_lora = get_peft_model(slat_model, config)
    pipeline.pipe.models["slat_flow_model"] = slat_model_lora
    
    # 统计参数
    trainable = sum(p.numel() for p in slat_model_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in slat_model_lora.parameters())
    print(f"[inject_lora_to_slat] LoRA 注入完成: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# =====================================================================
# Trellis 全参微调策略
# =====================================================================

from edit4shape.systems.utils.strategy import TrainingStrategy


class TrellisFullFinetuneStrategy(TrainingStrategy):
    """
    Trellis 全参微调策略。
    
    - 解冻所有参数
    - 从预训练权重加载冻结教师
    - ⚠️ 强制同设备：spconv 不支持跨设备推理，教师和学生必须在同一 GPU
    """
    
    def __init__(
        self, 
        pipeline: Any, 
        train_device: torch.device, 
        teacher_device: torch.device,  # 忽略，强制同设备
        pretrained_path: str,
    ):
        # ★ 强制同设备，避免 spconv 跨设备 indice_key 缓存问题
        super().__init__(pipeline, train_device, train_device)
        self._teacher: Optional[nn.Module] = None
        self._pretrained_path = pretrained_path
    
    def setup(self) -> None:
        """全参设置：解冻学生，从预训练加载教师（同设备）。"""
        from trellis import models as trellis_models
        
        # 解冻学生
        for p in self._student.parameters():
            p.requires_grad = True
        
        # 加载教师到同一设备（spconv 不支持跨设备）
        slat_model_path = f"{self._pretrained_path}/ckpts/slat_flow_img_dit_L_64l8p2_fp16"
        self._teacher = trellis_models.from_pretrained(slat_model_path)
        self._teacher.to(self.teacher_device).eval().requires_grad_(False)
        
        mem_mb = sum(p.numel() * p.element_size() for p in self._teacher.parameters()) / 1e6
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)
        
        print(f"[TrellisFullFinetuneStrategy] 全参微调: {trainable:,} 参数可训练")
        print(f"[TrellisFullFinetuneStrategy] 教师模型 → {self.teacher_device} ({mem_mb:.0f} MB)")
        print(f"[TrellisFullFinetuneStrategy] ⚠️ 显存翻倍（spconv 不支持跨设备推理）")
    
    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        """临时替换 pipeline 中的模型为冻结教师。"""
        original = self.pipeline.pipe.models["slat_flow_model"]
        self.pipeline.pipe.models["slat_flow_model"] = self._teacher
        try:
            yield
        finally:
            self.pipeline.pipe.models["slat_flow_model"] = original

