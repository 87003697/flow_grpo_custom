"""训练策略：基类 + spconv 推理兼容 mixin。

具体策略（TrellisLoRAStrategy 等）定义在各 generator 的 training_adpter 中。
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Generator
import torch
import torch.nn as nn


# =====================================================================
# 基类
# =====================================================================

class TrainingStrategy(ABC):
    """训练策略基类。"""

    def __init__(self, pipeline: Any, train_device: torch.device, teacher_device: torch.device):
        self.pipeline = pipeline
        self.train_device = train_device
        self.teacher_device = teacher_device
        self._student = pipeline.pipe.models["slat_flow_model"]
        self._accelerator = None

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    @contextmanager
    def sparse_teacher_context(self) -> Generator[None, None, None]: ...

    @contextmanager
    def dense_teacher_context(self) -> Generator[None, None, None]:
        """临时替换 pipeline 中的 dense 模型为冻结教师。

        默认 no-op（子类覆写以提供 dense teacher swap）。
        """
        yield

    @property
    def student(self) -> nn.Module:
        return self._student

    @property
    def has_teacher(self) -> bool:
        return True

    # ---- DDP 注册 ----

    def prepare_sparse(self, accelerator, optimizer_sparse=None):
        """用 accelerator.prepare 包装 Sparse 模型(+优化器)，回写到 pipeline。返回 prepared optimizer。
        
        当 optimizer_sparse 为 None 时（eval_only 模式），仅 prepare 模型，
        确保 accelerator.load_state() 能正确恢复 checkpoint 权重。
        """
        self._accelerator = accelerator
        if optimizer_sparse is not None:
            self._student, optimizer_sparse = accelerator.prepare(self._student, optimizer_sparse)
        else:
            self._student = accelerator.prepare(self._student)
        self.pipeline.pipe.models["slat_flow_model"] = self._student
        return optimizer_sparse

    def prepare_dense(self, accelerator, optimizer_dense=None):
        """用 accelerator.prepare 包装 Dense 模型(+优化器)，回写到 pipeline。

        仅 Dual-Stage 入口调用。Sparse-only 入口无需调用。
        """
        self._dense_student = self.pipeline.pipe.models["sparse_structure_flow_model"]
        if optimizer_dense is not None:
            self._dense_student, optimizer_dense = accelerator.prepare(
                self._dense_student, optimizer_dense,
            )
        else:
            self._dense_student = accelerator.prepare(self._dense_student)
        self.pipeline.pipe.models["sparse_structure_flow_model"] = self._dense_student
        return optimizer_dense

    @contextmanager
    def inference_context(self):
        """推理上下文。基类默认 no-op，子类 / mixin 可覆写。"""
        yield

    def _unwrap(self) -> nn.Module:
        """去除 DDP wrapper 获取原始模型。"""
        return self._accelerator.unwrap_model(self._student)

    # ---- 检查点（训练恢复用） ----

    def save_student(self, ckpt_dir: Path) -> None:
        """保存学生模型全部权重。"""
        from safetensors.torch import save_file
        save_file(self._unwrap().state_dict(), str(ckpt_dir / "slat_flow_model.safetensors"))

    def load_student(self, ckpt_dir: Path) -> None:
        """加载学生模型全部权重。"""
        p = ckpt_dir / "slat_flow_model.safetensors"
        if p.exists():
            from safetensors.torch import load_file
            self._unwrap().load_state_dict(load_file(str(p), device="cpu"))

    # ---- 推理导出 ----

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        """
        从训练 checkpoint 导出推理兼容的权重文件到 export_dir。
        
        默认行为：直接 copy slat_flow_model.safetensors。
        LoRA 子类覆写此方法以实现 merge。
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        src = ckpt_dir / "slat_flow_model.safetensors"
        dst = export_dir / "slat_flow_model.safetensors"
        shutil.copy2(src, dst)
        logging.info(f"[export] slat_flow_model: {src} → {dst}")
        return {"slat_flow_model": dst}


# =====================================================================
# Mixin：spconv 推理兼容
# =====================================================================

class SpconvInferenceMixin:
    """
    Mixin: 解决 spconv eval 路径不支持 bf16 的问题。

    accelerator.prepare() 会给 model.forward 注入 autocast(bf16)，
    但 spconv 的推理路径 (ops.implicit_gemm / ConvTunerSimple) 无法为
    bf16 输入找到合适的 GEMM 算法。

    本 mixin 在 prepare() 前保存原始 forward，推理时临时还原，
    仿照 TRELLIS 的 self.models (推理) vs self.training_models (训练) 设计。

    See: bug.md
    """

    _original_forward = None

    def prepare_sparse(self, accelerator, optimizer_sparse):
        # 在 accelerate 注入 autocast(bf16) 之前，保存原始 forward
        self._original_forward = self._student.forward
        return super().prepare_sparse(accelerator, optimizer_sparse)

    @contextmanager
    def inference_context(self):
        """推理时临时换回原始模型（无 DDP / autocast(bf16)）。"""
        if self._accelerator is None or self._original_forward is None:
            yield
            return

        pipe_models = self.pipeline.pipe.models
        saved_model = pipe_models["slat_flow_model"]
        inner = self._accelerator.unwrap_model(self._student)
        patched_forward = inner.forward

        inner.forward = self._original_forward
        pipe_models["slat_flow_model"] = inner
        try:
            yield
        finally:
            inner.forward = patched_forward
            pipe_models["slat_flow_model"] = saved_model
