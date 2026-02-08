#!/usr/bin/env python3
"""
TRELLIS 训练策略模块。

提供统一的训练模式抽象，支持：
- LoRA 微调：冻结基础权重，只训练 LoRA 参数
- 全参微调：解冻所有参数，冻结副本作为教师
- 冻结模式：不训练，仅推理

核心抽象：
- TrainingStrategy: 基类，定义模型设置和教师获取的统一接口
- create_strategy(): 工厂函数，根据配置创建对应策略
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
import logging
from typing import Any, Generator, Optional
import torch
import torch.nn as nn


class TrainingStrategy(ABC):
    """
    训练策略基类。
    
    定义模型设置和教师获取的统一接口，子类实现具体逻辑。
    
    Attributes:
        pipeline: TrellisRefAdapter 实例
        train_device: 训练设备（学生模型所在设备）
        teacher_device: 教师设备（教师模型所在设备，可与训练设备不同）
    """
    
    def __init__(
        self, 
        pipeline: Any, 
        train_device: torch.device, 
        teacher_device: torch.device,
    ):
        self.pipeline = pipeline
        self.train_device = train_device
        self.teacher_device = teacher_device
        self._student = pipeline.pipe.models["slat_flow_model"]
    
    @abstractmethod
    def setup(self) -> None:
        """设置模型的可训练状态。"""
        pass
    
    @abstractmethod
    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        """
        教师模型预测的上下文管理器。
        
        在此上下文中调用 pipeline.sparse_sampling_step 会使用教师模型。
        """
        pass
    
    @property
    def student(self) -> nn.Module:
        """获取学生模型（用于优化器创建）。"""
        return self._student
    
    @property
    def has_teacher(self) -> bool:
        """是否有教师模型可用（用于正则化）。"""
        return True


class LoRAStrategy(TrainingStrategy):
    """
    LoRA 微调策略。
    
    - 基础权重冻结
    - 只训练 LoRA 参数
    - 教师通过 disable_adapters 获取
    """
    
    def setup(self) -> None:
        """LoRA 设置：基础权重已冻结，LoRA 参数可训练。"""
        # LoRA 注入应在调用 setup 前完成（通过 PEFT）
        # 这里只打印信息
        trainable = sum(p.numel() for p in self._student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._student.parameters())
        logging.info(f"[LoRAStrategy] LoRA 微调: 可训练 {trainable:,} / 总参数 {total:,} ({100*trainable/total:.2f}%)")
    
    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        """LoRA 模式：临时禁用 LoRA adapters，使用原始权重。"""
        with self.pipeline.disable_lora_context():
            yield


class FullFinetuneStrategy(TrainingStrategy):
    """
    全参微调策略。
    
    - 解冻所有参数
    - 克隆冻结副本作为教师（放到 teacher_device）
    """
    
    def __init__(
        self, 
        pipeline: Any, 
        train_device: torch.device, 
        teacher_device: torch.device,
    ):
        super().__init__(pipeline, train_device, teacher_device)
        self._teacher: Optional[nn.Module] = None
    
    def setup(self) -> None:
        """全参设置：解冻学生，克隆冻结教师。"""
        # 解冻学生
        for p in self._student.parameters():
            p.requires_grad = True
        
        # 克隆冻结教师到 teacher_device
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
        """全参模式：临时替换 pipeline 中的模型为冻结教师。"""
        # 保存原模型
        original_model = self.pipeline.pipe.models["slat_flow_model"]
        
        # 替换为教师模型
        self.pipeline.pipe.models["slat_flow_model"] = self._teacher
        try:
            yield
        finally:
            # 恢复原模型
            self.pipeline.pipe.models["slat_flow_model"] = original_model


class FrozenStrategy(TrainingStrategy):
    """
    冻结策略（推理模式）。
    
    - 所有参数冻结
    - 无教师模型（不支持正则化）
    """
    
    def setup(self) -> None:
        """冻结设置：所有参数不可训练。"""
        for p in self._student.parameters():
            p.requires_grad = False
        logging.info(f"[FrozenStrategy] 模型冻结（推理模式）")
    
    @contextmanager
    def teacher_context(self) -> Generator[None, None, None]:
        """冻结模式：无教师（不应调用）。"""
        raise RuntimeError("FrozenStrategy 不支持正则化，请设置 cfg.reg.type = 'none'")
        yield  # 使 mypy 识别为 Generator
    
    @property
    def has_teacher(self) -> bool:
        """冻结模式无教师。"""
        return False


# =====================================================================
# 工厂函数
# =====================================================================

_STRATEGIES = {
    "lora": LoRAStrategy,
    "full": FullFinetuneStrategy,
    "frozen": FrozenStrategy,
}


def create_strategy(
    mode: str,
    pipeline: Any,
    train_device: torch.device,
    teacher_device: torch.device,
) -> TrainingStrategy:
    """
    根据模式创建对应的训练策略。
    
    Args:
        mode: 训练模式 ("lora" | "full" | "frozen")
        pipeline: TrellisRefAdapter 实例
        train_device: 训练设备
        teacher_device: 教师设备（全参模式时教师模型放置位置）
    
    Returns:
        TrainingStrategy: 对应的训练策略实例
    
    Raises:
        ValueError: 未知的模式
    """
    if mode not in _STRATEGIES:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(_STRATEGIES.keys())}")
    
    return _STRATEGIES[mode](pipeline, train_device, teacher_device)
