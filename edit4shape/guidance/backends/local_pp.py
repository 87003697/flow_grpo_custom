"""
同进程多 GPU Guidance（支持流水线并行）。

继承 LocalGuidance，添加异步接口实现流水线并行。

优势：
- 继承 LocalGuidance 的所有功能
- 支持异步接口实现流水线并行
- 双缓冲 CUDA stream 实现真正的重叠执行
"""

from collections import deque
from typing import List, Any
from PIL import Image

import torch

from edit4shape.guidance.base import GuidanceResult
from edit4shape.guidance.backends.local import LocalGuidance as LocalGuidanceBase


class LocalGuidance(LocalGuidanceBase):
    """
    同进程多 GPU Guidance（支持流水线并行）。
    
    继承 LocalGuidanceBase，添加：
    - 双缓冲 CUDA stream
    - 异步提交/等待接口
    """
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 Guidance。
        
        Args:
            cfg: 完整配置对象（需要 cfg.guidance.flowedit 和 cfg.train.loss）
            train_device: 训练使用的设备（用于计算 Guidance 设备）
        """
        # 调用父类初始化
        super().__init__(cfg, train_device)
        
        # ---- 流水线并行支持 ----
        # 两个 CUDA stream 用于双缓冲
        self._guidance_streams = [
            torch.cuda.Stream(device=self.device),
            torch.cuda.Stream(device=self.device),
        ]
        # 使用 deque 实现 FIFO 队列，支持真正的流水线重叠
        self._pending_queue: deque = deque(maxlen=2)  # 双缓冲，最多 2 个 pending 任务
        self._slot_counter = 0
        
        print(f"[LocalGuidance-PP] Pipeline parallelism enabled with 2 streams.")
    
    # =========================================================================
    # 异步接口（流水线并行）
    # =========================================================================
    
    def submit_async(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
    ) -> None:
        """
        异步提交 guidance 计算（不阻塞）。
        
        当前 micro-batch 提交后，调用方可立即开始下一个 micro-batch 的 Trellis。
        使用 wait_and_get() 获取结果。
        
        流水线时序：
        - micro-batch N: Trellis 完成 → submit_async() → 开始 FlowEdit（异步）
        - micro-batch N+1: 同时开始 Trellis → submit_async() → ...
        - wait_and_get(): 等待 micro-batch N 的 FlowEdit 完成
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
        """
        # 选择当前 stream（双缓冲交替）
        slot_idx = self._slot_counter % 2
        self._slot_counter += 1
        stream = self._guidance_streams[slot_idx]
        
        # 记录当前默认 stream 上的事件（等待 Trellis 完成）
        trellis_done = torch.cuda.Event()
        trellis_done.record(torch.cuda.current_stream(self.train_device))
        
        # 创建完成事件
        guidance_done = torch.cuda.Event()
        
        # 在 guidance stream 上异步执行
        with torch.cuda.stream(stream):
            # 等待 Trellis 完成
            stream.wait_event(trellis_done)
            
            # 执行 guidance 计算（使用父类的 compute_guidance）
            result = super().compute_guidance(comp_rgb, condition_images)
            
            # 记录完成事件
            guidance_done.record(stream)
        
        # 添加到队列（FIFO）
        self._pending_queue.append({
            "result": result,
            "done_event": guidance_done,
            "stream": stream,
        })
    
    def wait_and_get(self) -> GuidanceResult:
        """
        等待并获取最早提交的 submit_async 结果（FIFO）。
        
        阻塞直到 FlowEdit 计算完成，并确保训练 stream 同步。
        
        Returns:
            GuidanceResult: 最早提交的异步计算结果
        
        Raises:
            RuntimeError: 如果没有 pending 的异步提交
        """
        if not self._pending_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")
        
        # 从队列头部取出（FIFO）
        slot = self._pending_queue.popleft()
        
        # 等待 guidance stream 完成
        slot["done_event"].synchronize()
        
        # 确保训练 stream 也等待 guidance 完成，避免 backward 时的竞态条件
        torch.cuda.current_stream(self.train_device).wait_event(slot["done_event"])
        
        return slot["result"]
    
    def has_pending(self) -> bool:
        """检查是否有 pending 的异步提交。"""
        return len(self._pending_queue) > 0
