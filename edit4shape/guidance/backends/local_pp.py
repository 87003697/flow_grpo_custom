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
    # 同步接口（暂时禁用异步，确保梯度正确传播）
    # =========================================================================
    
    def submit_async(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
    ) -> None:
        """
        同步执行 guidance 计算（暂时禁用异步，用于调试梯度问题）。
        
        原异步版本存在跨设备 stream 同步问题，导致梯度无法正确传播。
        改为同步执行后，接口保持不变，trellis_pp.py 无需修改。
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
        """
        # 同步所有 CUDA 操作，确保跨设备操作正确
        torch.cuda.synchronize()
        
        # 同步执行 guidance 计算（使用父类的 compute_guidance）
        result = super().compute_guidance(comp_rgb, condition_images)
        
        # 再次同步，确保 guidance 完成
        torch.cuda.synchronize()
        
        # 添加到队列（FIFO）
        self._pending_queue.append({
            "result": result,
            "done_event": None,
            "stream": None,
        })
    
    def wait_and_get(self) -> GuidanceResult:
        """
        获取最早提交的 submit_async 结果（FIFO）。
        
        同步版本直接返回结果，无需等待。
        
        Returns:
            GuidanceResult: 最早提交的计算结果
        
        Raises:
            RuntimeError: 如果没有 pending 的异步提交
        """
        if not self._pending_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")
        
        # 从队列头部取出（FIFO）
        slot = self._pending_queue.popleft()
        
        return slot["result"]
    
    def has_pending(self) -> bool:
        """检查是否有 pending 的异步提交。"""
        return len(self._pending_queue) > 0
