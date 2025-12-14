#!/usr/bin/env python3
"""
自定义 PEFT LoRA 层以支持 TRELLIS 的 SparseLinear（输入/输出为 SparseTensor）。

- 仅对 feats 路径施加 LoRA，保持 coords 不变
- API 对齐 peft 的 Linear LoRA：支持 disable_adapter、merge/unmerge/save_pretrained 流程
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional, Union

import torch
import torch.nn as nn

# 注入 TRELLIS 官方代码路径
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_TRELLIS_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "TRELLIS")
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

from trellis.modules import sparse as sp  # type: ignore  # noqa: E402
from trellis.modules.sparse.linear import SparseLinear  # type: ignore  # noqa: E402

from peft.tuners.tuners_utils import BaseTunerLayer  # noqa: E402
from peft.tuners.lora.layer import LoraLayer  # noqa: E402


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

        out: sp.SparseTensor = self.get_base_layer()(x, *args, **kwargs)
        out_dtype = out.feats.dtype

        # 逐 adapter 叠加 LoRA 贡献，仅作用于 feats
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            feats = x.feats.to(lora_A.weight.dtype)
            lora_feats = lora_B(lora_A(dropout(feats))) * scaling
            out = out.replace(out.feats + lora_feats.to(out_dtype))

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


