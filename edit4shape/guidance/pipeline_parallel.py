"""
流水线并行基础设施。

提供通用的流水线并行 Mixin，支持：
- 双缓冲 CUDA stream
- 异步提交/等待接口（FIFO 队列）
- 可被任意 Guidance paradigm 复用

两层接口：
- submit_async() / wait_and_get(): comp_rgb 层面 proxy，
  在 guidance 侧独立 backward 得到 rgb_grad，train 侧用 comp_rgb.backward(rgb_grad) 驱动反传。
- has_pending() / pending_count: 队列状态查询。

AsyncGuidanceResult:
  存储 guidance 侧已算好的像素级梯度 (rgb_grad) 和标量日志，
  train GPU 侧无需重复 guidance forward/backward。

Usage:
    class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
        pass
    
    class DistillationGuidancePP(PipelineParallelMixin, DistillationGuidance):
        pass
"""

import logging
from collections import deque
from typing import List, Any, Callable, TypeVar, Generic, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from PIL import Image

from edit4shape.guidance.base import GuidanceResult


T = TypeVar('T')


# =====================================================================
# AsyncGuidanceResult — 异步 Guidance 返回结果
# =====================================================================

@dataclass
class AsyncGuidanceResult:
    """
    异步 Guidance 返回结果。

    与 GuidanceResult 不同，这里存储的是已计算好的 rgb_grad（像素级梯度），
    而非需要 backward 的 loss tensor。用于 train GPU 侧 comp_rgb.backward(rgb_grad)。

    生命周期：
        1. submit_async() 内部：detach comp_rgb → guidance forward → backward → rgb_grad
        2. wait_and_get(target_device) 返回本对象，rgb_grad 已搬到 target_device
        3. 调用方执行 comp_rgb.backward(rgb_grad)

    Attributes:
        rgb_grad: ∂(weight*L)/∂comp_rgb，形状与 comp_rgb 一致 (B, V, H, W, 3)
        loss_scalar: 加权后的标量 loss 值（float，用于日志）
        loss_dict: 细分 loss 字典（str → float，用于日志），可为 None
        edited_imgs: 编辑后图像 (B,V,C,H,W)，FlowEdit 专用，可为 None
        trackers: FlowEdit StateTracker 列表（用于 progress 可视化），可为 None
                  ★ 内部 latent 保留在 guidance GPU（get_progress_grid 需要）
    """
    rgb_grad: torch.Tensor                                  # (B, V, H, W, 3)
    loss_scalar: float = 0.0                                # 加权后标量 loss
    loss_dict: Optional[Dict[str, float]] = None            # 细分 loss（日志用）
    edited_imgs: Optional[torch.Tensor] = None              # (B, V, C, H, W) 编辑后图像
    trackers: Optional[List] = None                         # FlowEdit StateTracker 列表

    @classmethod
    def from_guidance_result(
        cls,
        result: "GuidanceResult",
        rgb_grad: torch.Tensor,
        guidance_weight: float,
    ) -> "AsyncGuidanceResult":
        """
        从 GuidanceResult + rgb_grad 构建异步结果。

        集中处理 detach / scalar 转换，避免 submit_async 内部散落转换逻辑。
        """
        loss_scalar = (result.loss * guidance_weight).item()
        loss_dict_scalar: Optional[Dict[str, float]] = None
        if result.loss_dict:
            loss_dict_scalar = {
                k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in result.loss_dict.items()
                if v is not None
            }
        return cls(
            rgb_grad=rgb_grad,
            loss_scalar=loss_scalar,
            loss_dict=loss_dict_scalar,
            edited_imgs=result.edited_imgs.detach() if result.edited_imgs is not None else None,
            trackers=result.trackers,  # 内部 tensor 已是 detached clone
        )


# =====================================================================
# PipelineParallelMixin
# =====================================================================

class PipelineParallelMixin:
    """
    流水线并行 Mixin。
    
    继承此 Mixin 的类需要：
    1. 有 `device` 属性（CUDA 设备）
    2. 有 `compute_guidance(comp_rgb, condition_images, **kwargs)` 方法
    
    提供：
    - submit_async(): 异步提交计算任务（内部独立 backward → rgb_grad）
    - wait_and_get(): 等待并获取结果（FIFO，返回 AsyncGuidanceResult）
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
        *,
        guidance_weight: float = 1.0,
        rank: int = 0,
        **kwargs,
    ) -> None:
        """
        异步提交 guidance 计算任务。

        流程：
          1. detach comp_rgb → 创建 requires_grad 的 proxy（与 train 侧计算图解耦）
          2. 调用 compute_guidance(proxy, ...) → GuidanceResult
          3. weighted_loss = result.loss * guidance_weight → backward → proxy.grad = rgb_grad
          4. 构建 AsyncGuidanceResult（rgb_grad + 标量日志）入队

        当前实现仍为同步执行（跨设备 stream 同步问题），
        但接口已为真正异步预留：train GPU 侧无需等待即可继续下一个 MB 的前向。

        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]，来自 train 侧（有 autograd 图）
            condition_images: 条件图像列表 [len=B] of PIL.Image
            guidance_weight: guidance loss 权重
            rank: 当前进程的 rank（传递给 compute_guidance）
            **kwargs: 传递给 compute_guidance 的额外参数
        """
        # 同步所有 CUDA 操作，确保跨设备数据就绪
        torch.cuda.synchronize()
        
        # ---- 1. proxy → guidance forward → backward → rgb_grad ----
        proxy_rgb = comp_rgb.detach().requires_grad_(True)  # (B, V, H, W, 3)
        result: GuidanceResult = self.compute_guidance(
            proxy_rgb, condition_images, rank=rank, **kwargs
        )
        (result.loss * guidance_weight).backward()
        rgb_grad = proxy_rgb.grad.detach()  # (B, V, H, W, 3)
        
        # ---- 2. 打包结果（scalar 转换 + vis 数据提取集中在工厂方法） ----
        async_result = AsyncGuidanceResult.from_guidance_result(
            result, rgb_grad, guidance_weight
        )
        
        # ---- 3. 释放 guidance 侧中间产物 + 入队 ----
        del result, proxy_rgb
        torch.cuda.synchronize()
        self._pp_queue.append(async_result)
        self._pp_slot_counter += 1
    
    def wait_and_get(
        self,
        target_device: Optional[torch.device] = None,
    ) -> AsyncGuidanceResult:
        """
        获取最早提交的 submit_async 结果（FIFO）。

        如果指定 target_device，会将 rgb_grad 搬到目标设备
        （处理 guidance GPU ≠ train GPU 的情况）。
        
        Args:
            target_device: rgb_grad 的目标设备（None 则保持原设备）
        
        Returns:
            AsyncGuidanceResult: 包含 rgb_grad + 标量日志
        
        Raises:
            RuntimeError: 如果没有 pending 的异步提交
        """
        if not self._pp_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")
        
        # 从队列头部取出（FIFO）
        async_result: AsyncGuidanceResult = self._pp_queue.popleft()
        
        # 搬到目标设备（如果需要）
        if target_device is not None:
            if async_result.rgb_grad.device != target_device:
                async_result.rgb_grad = async_result.rgb_grad.to(target_device)
            if async_result.edited_imgs is not None and async_result.edited_imgs.device != target_device:
                async_result.edited_imgs = async_result.edited_imgs.to(target_device)
            # ★ trackers 保留在 guidance GPU：get_progress_grid(pipe, n) 需要
            #   pipe (VAE) 和 tracker 内部 latent 都在 guidance GPU 上
        
        return async_result
    
    def has_pending(self) -> bool:
        """检查是否有 pending 的异步提交。"""
        return len(self._pp_queue) > 0
    
    @property
    def pending_count(self) -> int:
        """返回 pending 任务数量。"""
        return len(self._pp_queue)
