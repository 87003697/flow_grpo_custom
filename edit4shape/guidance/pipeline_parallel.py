"""
流水线并行基础设施 — 真异步版本。

提供通用的流水线并行 Mixin，支持：
- 后台线程执行 guidance forward + backward（不阻塞 train GPU）
- FIFO 队列传递 AsyncGuidanceResult
- 可被任意 Guidance paradigm 复用

并行原理：
  Train GPU (cuda:0) 和 Guidance GPU (cuda:2) 是独立硬件。
  PyTorch 的 CUDA 操作执行时释放 GIL，所以用 threading.Thread
  可以让两个 GPU 同时工作：
    - 主线程：在 Train GPU 上做 rollout / decode / render / Phase 3
    - 后台线程：在 Guidance GPU 上做 compute_guidance + backward

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
import threading
import time
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
    # ---- 挂钟计时（双 GPU 利用率分析）----
    guid_wall_start: float = 0.0                            # worker 开始 GPU 计算的 perf_counter
    guid_wall_end: float = 0.0                              # worker 完成 GPU 计算的 perf_counter

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
# PipelineParallelMixin — 真异步版本
# =====================================================================

class PipelineParallelMixin:
    """
    流水线并行 Mixin（真异步版本）。
    
    继承此 Mixin 的类需要：
    1. 有 `device` 属性（Guidance CUDA 设备）
    2. 有 `compute_guidance(comp_rgb, condition_images, **kwargs)` 方法
    
    并行机制：
    - submit_async(): 在后台 Thread 中执行 guidance forward + backward，立即返回
    - wait_and_get(): join 后台线程 → 从队列取出 AsyncGuidanceResult
    - 主线程可在 submit 后立即继续 Train GPU 上的工作（rollout / P3 等）
    
    线程安全：
    - _join_prev_thread(): 每次 submit/wait 前先 join 上一个线程，
      保证 Guidance GPU 串行执行，避免显存竞争
    - 后台线程异常保存在 _pp_error，wait_and_get 时抛出
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
        self._pp_queue: deque = deque()  # 无 maxlen 限制，支持任意 accum_steps
        self._pp_slot_counter = 0
        self._pp_num_streams = num_streams
        self._pp_thread: Optional[threading.Thread] = None  # 后台线程引用
        self._pp_error: Optional[BaseException] = None       # 后台线程异常捕获
        
        logging.info(f"[PipelineParallel] Async mode enabled with {num_streams} streams on {self.device}")
    
    def _join_prev_thread(self) -> None:
        """
        等待上一个后台线程完成，并检查异常。
        
        调用时机：
        - submit_async 开头：保证 Guidance GPU 串行执行（一次只跑一个 guidance）
        - wait_and_get 开头：确保结果已入队
        
        异常传播：后台线程捕获的异常会在此处重新抛出，不会被静默吞掉。
        """
        if self._pp_thread is not None:
            self._pp_thread.join()
            self._pp_thread = None
        if self._pp_error is not None:
            err = self._pp_error
            self._pp_error = None
            raise RuntimeError(f"[PipelineParallel] Background guidance thread failed: {err}") from err
    
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
        真异步提交 guidance 计算任务。

        在后台 Thread 中执行 guidance forward + backward，主线程立即返回。
        Train GPU 可在此期间继续做 drain_prev / build_next 等工作。

        流程：
          主线程:
            0. join 上一个后台线程（保证 Guidance GPU 串行）
            1. detach comp_rgb → 搬到 Guidance GPU → requires_grad proxy
            2. 启动后台线程 → 立即返回
          后台线程:
            3. compute_guidance(proxy, ...) → GuidanceResult
            4. (loss * weight).backward() → proxy.grad = rgb_grad
            5. 打包 AsyncGuidanceResult → 入队

        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]，来自 train 侧（有 autograd 图）
            condition_images: 条件图像列表 [len=B] of PIL.Image
            guidance_weight: guidance loss 权重
            rank: 当前进程的 rank（传递给 compute_guidance）
            **kwargs: 传递给 compute_guidance 的额外参数
        """
        # ---- 0. 等待上一个 submit 完成（Guidance GPU 串行，避免显存竞争） ----
        self._join_prev_thread()
        
        # ---- 1. 在 Train GPU 上 detach → 搬到 Guidance GPU ----
        # 只同步 comp_rgb 所在设备的当前 stream（不全局同步，避免阻塞 Guidance GPU）
        torch.cuda.current_stream(comp_rgb.device).synchronize()
        proxy_rgb = comp_rgb.detach().to(self.device).requires_grad_(True)  # (B, V, H, W, 3)，在 Guidance GPU 上
        
        # ---- 2. 在后台线程中执行 guidance forward + backward ----
        slot_idx = self._pp_slot_counter % self._pp_num_streams
        stream = self._pp_streams[slot_idx]
        
        def _worker():
            try:
                _wall_start = time.perf_counter()
                with torch.cuda.stream(stream):
                    # guidance forward（最耗时的部分，~59s）
                    result: GuidanceResult = self.compute_guidance(
                        proxy_rgb, condition_images, rank=rank, **kwargs
                    )
                    # guidance backward → rgb_grad
                    (result.loss * guidance_weight).backward()
                    rgb_grad = proxy_rgb.grad.detach()  # (B, V, H, W, 3)
                    
                    # 打包结果
                    async_result = AsyncGuidanceResult.from_guidance_result(
                        result, rgb_grad, guidance_weight
                    )
                    
                    # 释放 guidance 侧中间产物
                    # ★ 不要 del proxy_rgb：它是闭包捕获的变量，
                    #   del 会让 Python 将其标记为局部变量 → UnboundLocalError
                    del result
                
                # 确保 stream 中所有 kernel 完成后再入队
                stream.synchronize()
                _wall_end = time.perf_counter()
                async_result.guid_wall_start = _wall_start
                async_result.guid_wall_end = _wall_end
                self._pp_queue.append(async_result)
            except BaseException as e:
                self._pp_error = e
        
        self._pp_thread = threading.Thread(target=_worker, daemon=True)
        self._pp_thread.start()  # ★ 立即返回！主线程可以继续做 Train GPU 上的工作
        self._pp_slot_counter += 1
    
    def wait_and_get(
        self,
        target_device: Optional[torch.device] = None,
    ) -> AsyncGuidanceResult:
        """
        获取最早提交的 submit_async 结果（FIFO）。
        
        正常 case（队列非空）：直接 pop，不 join 当前后台线程。
          → guid(N) 继续在后台线程跑，P3(N-1) 可以并行执行。
        边界 case（队列为空，如 accum 末尾 / epoch 末尾）：
          join 当前后台线程等待结果入队。

        Args:
            target_device: rgb_grad 的目标设备（None 则保持在 Guidance GPU 上）
        
        Returns:
            AsyncGuidanceResult: 包含 rgb_grad + 标量日志
        
        Raises:
            RuntimeError: 后台线程异常 或 队列为空
        """
        # ---- 只在队列为空时阻塞等待（边界 case） ----
        # 正常 case: submit_async(N) 已 join guid(N-1) → 结果在队列 → 直接 pop
        # 边界 case: accum 末尾/epoch 末尾没有下一个 submit → 需手动等
        if not self._pp_queue:
            self._join_prev_thread()
        
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
        return len(self._pp_queue) > 0 or self._pp_thread is not None
    
    @property
    def pending_count(self) -> int:
        """返回 pending 任务数量（含正在执行的后台线程）。"""
        count = len(self._pp_queue)
        if self._pp_thread is not None and self._pp_thread.is_alive():
            count += 1
        return count
