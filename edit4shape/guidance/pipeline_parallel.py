"""
流水线并行基础设施。

提供通用的流水线并行 Mixin，支持：
- 双缓冲 CUDA stream
- 异步提交/等待接口（FIFO 队列）
- 可被任意 Guidance paradigm 复用

Usage:
    class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
        pass
    
    class DistillationGuidancePP(PipelineParallelMixin, DistillationGuidance):
        pass
"""

import logging
from collections import deque
from typing import List, Any, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image

from edit4shape.guidance.base import GuidanceResult


T = TypeVar('T')


class PipelineParallelMixin:
    """
    流水线并行 Mixin。
    
    继承此 Mixin 的类需要：
    1. 有 `device` 属性（CUDA 设备）
    2. 有 `compute_guidance(comp_rgb, condition_images, **kwargs)` 方法
    
    提供：
    - submit_async(): 异步提交计算任务
    - wait_and_get(): 等待并获取结果（FIFO）
    - has_pending(): 检查是否有待处理任务
    """
    
    # 声明需要的属性（子类必须提供）
    device: torch.device
    
    def _init_pipeline_parallel(self, num_streams: int = 2):
        """
        初始化流水线并行基础设施。
        
        子类需要在 __init__ 中调用此方法。
        
        Args:
            num_streams: CUDA stream 数量（默认 2，双缓冲）
        """
        self._pp_streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(num_streams)
        ]
        self._pp_queue: deque = deque(maxlen=num_streams)
        self._pp_slot_counter = 0
        self._pp_num_streams = num_streams
        
        logging.info(f"[PipelineParallel] Enabled with {num_streams} streams on {self.device}")
    
    def submit_async(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> None:
        """
        异步提交 guidance 计算任务。
        
        当前实现：同步执行（跨设备 stream 同步问题）。
        未来可改为真正异步。
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 传递给 compute_guidance 的额外参数
        """
        # 同步所有 CUDA 操作，确保跨设备操作正确
        torch.cuda.synchronize()
        
        # 同步执行 guidance 计算
        result = self.compute_guidance(comp_rgb, condition_images, **kwargs)
        
        # 再次同步，确保 guidance 完成
        torch.cuda.synchronize()
        
        # 添加到队列（FIFO）
        self._pp_queue.append({
            "result": result,
            "stream_idx": self._pp_slot_counter % self._pp_num_streams,
        })
        self._pp_slot_counter += 1
    
    def wait_and_get(self) -> GuidanceResult:
        """
        获取最早提交的 submit_async 结果（FIFO）。
        
        Returns:
            GuidanceResult: 最早提交的计算结果
        
        Raises:
            RuntimeError: 如果没有 pending 的异步提交
        """
        if not self._pp_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")
        
        # 从队列头部取出（FIFO）
        slot = self._pp_queue.popleft()
        return slot["result"]
    
    def has_pending(self) -> bool:
        """检查是否有 pending 的异步提交。"""
        return len(self._pp_queue) > 0
    
    @property
    def pending_count(self) -> int:
        """返回 pending 任务数量。"""
        return len(self._pp_queue)
