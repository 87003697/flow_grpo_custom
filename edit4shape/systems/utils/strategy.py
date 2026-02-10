"""训练策略：LoRA / 全参 / 冻结。"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import torch
import torch.nn as nn


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
    def teacher_context(self) -> Generator[None, None, None]: ...

    @property
    def student(self) -> nn.Module:
        return self._student

    @property
    def has_teacher(self) -> bool:
        return True

    # ---- DDP 注册 ----

    def prepare(self, accelerator, optimizer):
        """用 accelerator.prepare 包装模型+优化器，回写到 pipeline。返回 prepared optimizer。"""
        self._accelerator = accelerator
        self._student, optimizer = accelerator.prepare(self._student, optimizer)
        # DDP 包装后的模型回写到 pipeline（forward 走 DDP，属性访问走 _resolve）
        self.pipeline.pipe.models["slat_flow_model"] = self._student
        return optimizer

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
        
        Returns:
            {内部模型名: 导出文件路径}
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        src = ckpt_dir / "slat_flow_model.safetensors"
        dst = export_dir / "slat_flow_model.safetensors"
        shutil.copy2(src, dst)
        logging.info(f"[export] slat_flow_model: {src} → {dst}")
        return {"slat_flow_model": dst}


class LoRAStrategy(TrainingStrategy):
    """LoRA 微调：冻结基础权重，只训练 LoRA 参数。"""

    def setup(self) -> None:
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._student.parameters())
        logging.info(f"[LoRAStrategy] 可训练 {trainable:,} / 总参数 {total:,} ({100*trainable/total:.2f}%)")

    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        with self.pipeline.disable_lora_context():
            yield

    def save_student(self, ckpt_dir: Path) -> None:
        """只保存 LoRA adapter 参数。"""
        from safetensors.torch import save_file
        lora_sd = {k: v for k, v in self._unwrap().state_dict().items() if "lora_" in k}
        save_file(lora_sd, str(ckpt_dir / "slat_flow_model.safetensors"))

    def load_student(self, ckpt_dir: Path) -> None:
        """只加载 LoRA adapter 参数（strict=False）。"""
        p = ckpt_dir / "slat_flow_model.safetensors"
        if p.exists():
            from safetensors.torch import load_file
            self._unwrap().load_state_dict(load_file(str(p), device="cpu"), strict=False)

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        """LoRA 导出：先加载 adapter，merge 到 base，导出完整权重。"""
        from safetensors.torch import load_file, save_file

        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载 adapter 到当前模型
        self.load_student(ckpt_dir)

        # 2. Merge lora → base，生成干净的 state_dict
        merged_sd = self._merge_lora_weights()

        # 3. 导出
        dst = export_dir / "slat_flow_model.safetensors"
        save_file(merged_sd, str(dst))
        logging.info(f"[export] slat_flow_model (LoRA merged): → {dst}")
        return {"slat_flow_model": dst}

    def _merge_lora_weights(self) -> Dict[str, torch.Tensor]:
        """将 lora_A/lora_B 折叠进 base weight，返回无 LoRA key 的干净 state_dict。"""
        model = self._unwrap()
        merged: Dict[str, torch.Tensor] = {}

        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                W = module.weight.data                      # (out_features, in_features)
                A = module.lora_A.weight.data               # (rank, in_features)
                B = module.lora_B.weight.data               # (out_features, rank)
                scaling = getattr(module, "scaling", 1.0)
                merged[f"{name}.weight"] = W + (B @ A) * scaling  # (out_features, in_features)
                if module.bias is not None:
                    merged[f"{name}.bias"] = module.bias.data

        # 补充非 LoRA 参数
        for k, v in model.state_dict().items():
            if "lora_" not in k and k not in merged:
                merged[k] = v

        return merged


class FullFinetuneStrategy(TrainingStrategy):
    """全参微调：解冻所有参数，克隆冻结副本作为教师。"""

    def __init__(self, pipeline: Any, train_device: torch.device, teacher_device: torch.device):
        super().__init__(pipeline, train_device, teacher_device)
        self._teacher: Optional[nn.Module] = None

    def setup(self) -> None:
        for p in self._student.parameters():
            p.requires_grad = True
        self._teacher = copy.deepcopy(self._student).eval().requires_grad_(False)
        self._teacher.to(self.teacher_device)

        mem_mb = sum(p.numel() * p.element_size() for p in self._teacher.parameters()) / 1e6
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)

        if self.teacher_device == self.train_device:
            logging.warning(f"[FullFinetuneStrategy] 教师与学生在同一设备 ({self.train_device})，显存翻倍")
        logging.info(f"[FullFinetuneStrategy] 全参微调: {trainable:,} 参数可训练")
        logging.info(f"[FullFinetuneStrategy] 教师模型 → {self.teacher_device} ({mem_mb:.0f} MB)")

    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        original = self.pipeline.pipe.models["slat_flow_model"]
        self.pipeline.pipe.models["slat_flow_model"] = self._teacher
        try:
            yield
        finally:
            self.pipeline.pipe.models["slat_flow_model"] = original


class FrozenStrategy(TrainingStrategy):
    """冻结策略（推理模式）。"""

    def setup(self) -> None:
        for p in self._student.parameters():
            p.requires_grad = False
        logging.info("[FrozenStrategy] 模型冻结（推理模式）")

    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        raise RuntimeError("FrozenStrategy 不支持正则化")
        yield  # type hint

    @property
    def has_teacher(self) -> bool:
        return False

    def prepare(self, accelerator, optimizer):
        """冻结模式无需 DDP 包装。"""
        self._accelerator = accelerator
        return optimizer

    def save_student(self, ckpt_dir: Path) -> None:
        pass

    def load_student(self, ckpt_dir: Path) -> None:
        pass

    def export_student(self, ckpt_dir: Path, export_dir: Path) -> Dict[str, Path]:
        return {}


# ---- 工厂 ----

_STRATEGIES = {
    "lora": LoRAStrategy,
    "full": FullFinetuneStrategy,
    "frozen": FrozenStrategy,
}


def create_strategy(
    mode: str, pipeline: Any, train_device: torch.device, teacher_device: torch.device,
) -> TrainingStrategy:
    if mode not in _STRATEGIES:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[mode](pipeline, train_device, teacher_device)
