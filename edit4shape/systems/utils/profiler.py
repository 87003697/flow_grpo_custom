"""
PhaseProfiler — 基于 CUDA Event 的 Phase 级别计时器。

用于测量训练步中各 Phase 的 GPU 耗时，辅助异步流水线优化决策。

使用方式（同步版）:
    profiler = PhaseProfiler(enabled=True)
    
    profiler.tick("P1_rollout")
    ...  # Phase 1
    profiler.tick("P2a_decode_render")
    ...  # Phase 2a
    profiler.tick("P2_guidance")
    ...  # Phase 2 (guidance + backward)
    profiler.tick("P3_grad_backward")
    ...  # Phase 3
    profiler.tick("end")
    
    timings = profiler.elapsed()
    # {"P1_rollout": 120.5, "P2a_decode_render": 85.2, ...}  单位 ms
    
    # 自动汇总（每 N 步打印一次）
    profiler.step(timings, global_step, print_freq=10)

使用方式（异步版，双 GPU 利用率分析）:
    profiler = AsyncPhaseProfiler(enabled=True)
    
    profiler.tick("P1_rollout")
    ...  # P1 + P2a + submit
    profiler.tick("P2_wait_backward")
    guidance_log = wait_and_backward(...)
    profiler.set_guid_timing(async_result.guid_wall_start, async_result.guid_wall_end)
    ...  # P3
    profiler.tick("end")
    
    timings = profiler.elapsed()
    # 额外包含: guid_gpu_active, overlap_guid_P3, train_gpu_idle, guid_gpu_idle
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# =====================================================================
# PhaseProfiler — 基础计时器
# =====================================================================

@dataclass
class PhaseProfiler:
    """
    基于 CUDA Event 的 Phase 级别计时器。
    
    特性：
    - 使用 torch.cuda.Event(enable_timing=True) 精确测量 GPU 时间
    - tick() 标记阶段边界，elapsed() 返回各阶段耗时
    - step() 自动累积统计，周期性打印汇总
    - enabled=False 时所有操作为空操作（零开销）
    
    输出示例（print_freq=10 时每 10 步打印一次）：
        [PhaseProfiler] Step 10 | avg over 10 steps (ms):
          dense_sampling:     45.2
          P1_rollout:        312.8
          P2a_decode_render: 185.4
          P2_guidance_bw:    523.1  ← 异步化可隐藏的部分
          P3_grad_backward:  148.9
          total:            1215.4
    """
    enabled: bool = True
    verbose: bool = True    # 控制 step() 是否打印汇总（多 rank 时仅主进程设 True）
    
    # ---- 单步计时 ----
    _events: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)
    
    # ---- 累积统计 ----
    _history: Dict[str, List[float]] = field(default_factory=dict)
    _step_count: int = 0
    
    def tick(self, name: str) -> None:
        """记录一个阶段边界。相邻两个 tick 之间的时间即为该阶段耗时。"""
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name] = event
        self._order.append(name)
    
    def elapsed(self) -> Dict[str, float]:
        """
        返回相邻 tick 之间的耗时（ms）。
        
        需要在所有 tick 之后调用。会调用 torch.cuda.synchronize()
        确保所有 GPU 操作完成后再读取时间。
        
        Returns:
            Dict[str, float]: {phase_name: elapsed_ms}，
            最后一项 "total" 为首尾 tick 之间的总耗时。
        """
        if not self.enabled or len(self._order) < 2:
            return {}
        torch.cuda.synchronize()
        timings: Dict[str, float] = {}
        for i in range(len(self._order) - 1):
            name = self._order[i]
            start = self._events[self._order[i]]
            end = self._events[self._order[i + 1]]
            timings[name] = start.elapsed_time(end)  # ms
        timings["total"] = self._events[self._order[0]].elapsed_time(
            self._events[self._order[-1]]
        )  # ms
        return timings
    
    def reset(self) -> None:
        """重置单步计时（不清除累积统计）。"""
        self._events.clear()
        self._order.clear()
    
    def step(
        self,
        timings: Dict[str, float],
        global_step: int,
        print_freq: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        累积一步计时，并在每 print_freq 步打印汇总。
        
        Args:
            timings: elapsed() 返回的计时字典
            global_step: 当前全局步数
            print_freq: 打印频率（每 N 步打印一次平均值）
        
        Returns:
            如果本步触发打印，返回平均值字典；否则返回 None。
        """
        if not self.enabled or not timings:
            self.reset()
            return None
        
        # 累积
        for k, v in timings.items():
            if k not in self._history:
                self._history[k] = []
            self._history[k].append(v)
        self._step_count += 1
        
        # 重置单步计时
        self.reset()
        
        # 检查是否该打印
        if self._step_count % print_freq != 0:
            return None
        
        # 计算平均值
        avg: Dict[str, float] = {}
        for k, vals in self._history.items():
            avg[k] = sum(vals) / len(vals)
        
        # 打印
        if self.verbose:
            self._print_summary(avg, global_step)
        
        # 清除累积
        self._history.clear()
        self._step_count = 0
        
        return avg
    
    def _print_summary(self, avg: Dict[str, float], global_step: int) -> None:
        """打印汇总（子类可覆盖以扩展格式）。"""
        n = len(next(iter(self._history.values())))
        lines = [f"[PhaseProfiler] Step {global_step} | avg over {n} steps (ms):"]
        
        # 按 key 顺序打印（total 放最后）
        for k in avg:
            if k != "total":
                lines.append(f"  {k:30s} {avg[k]:8.1f}")
        if "total" in avg:
            lines.append(f"  {'total':30s} {avg['total']:8.1f}")
        
        logging.info("\n".join(lines))
    
    def as_log_dict(self, timings: Dict[str, float], prefix: str = "time/") -> Dict[str, float]:
        """
        将计时字典转为适合日志系统的格式。
        
        Args:
            timings: elapsed() 返回的计时字典
            prefix: 键名前缀
        
        Returns:
            Dict[str, float]: {prefix+phase_name: elapsed_ms}
        """
        if not timings:
            return {}
        return {f"{prefix}{k}": v for k, v in timings.items()}
    
    def collect(self, global_step: int, print_freq: int = 10) -> Dict[str, float]:
        """
        一站式收集：elapsed → 合入日志 → 累积汇总 → 重置。
        
        调用方只需在所有 tick 之后调 collect()，拿到 time/* 字典合入日志即可，
        无需手动调用 elapsed / as_log_dict / step。
        
        Args:
            global_step: 当前全局步数
            print_freq: 打印频率（每 N 步打印一次平均值）
        
        Returns:
            Dict[str, float]: {time/phase_name: elapsed_ms}，
            enabled=False 时返回空字典。
        """
        timings = self.elapsed()
        log_dict = self.as_log_dict(timings)
        self.step(timings, global_step, print_freq=print_freq)
        return log_dict


# =====================================================================
# GPUUtilizationMixin — 双 GPU 利用率分析
# =====================================================================

class GPUUtilizationMixin:
    """
    为 PhaseProfiler 添加双 GPU 挂钟时间追踪和利用率分析。
    
    通过 time.perf_counter() 记录各 phase 的挂钟时间，
    结合 Guidance GPU 的活跃时段，计算：
    - guid_gpu_active: Guidance GPU 活跃时间
    - overlap_guid_P3: guid 与 P3 的重叠时间（= 异步收益）
    - train_gpu_idle: Train GPU 空闲时间
    - guid_gpu_idle: Guidance GPU 空闲时间
    
    设计意图：
      CUDA Event 只能测量同一设备上的时间间隔，无法跨设备比较。
      用挂钟时间可以分析 Train GPU 和 Guidance GPU 的时间重叠关系。
    """
    
    # ---- 需要由组合类提供 ----
    enabled: bool
    
    def _init_gpu_util(self) -> None:
        """初始化 GPU 利用率追踪状态。子类在 __post_init__ 中调用。"""
        self._wall_times: Dict[str, float] = {}
        self._guid_wall_start: float = 0.0
        self._guid_wall_end: float = 0.0
    
    def set_guid_timing(self, wall_start: float, wall_end: float) -> None:
        """
        设置 Guidance GPU 的活跃时间段。
        
        由 worker 线程通过 AsyncGuidanceResult 传回，在 drain_prev 中调用。
        
        Args:
            wall_start: guidance 计算开始的 time.perf_counter() 值
            wall_end:   guidance 计算结束的 time.perf_counter() 值
        """
        if not self.enabled:
            return
        self._guid_wall_start = wall_start
        self._guid_wall_end = wall_end
    
    def _tick_wall(self, name: str) -> None:
        """记录一个阶段边界的挂钟时间。在 tick() 中被调用。"""
        self._wall_times[name] = time.perf_counter()
    
    def _compute_gpu_util(self, timings: Dict[str, float]) -> None:
        """
        计算 GPU 利用率指标，原地写入 timings 字典。
        
        在 elapsed() 末尾被调用。需要 _wall_times 中有 "P3_grad_backward" 和 "end"，
        以及 _guid_wall_end > 0（即 set_guid_timing 被调用过）。
        
        Args:
            timings: elapsed() 已计算的 phase timing 字典（会被原地扩充）
        """
        if not (self._guid_wall_end > 0
                and "P3_grad_backward" in self._wall_times
                and "end" in self._wall_times):
            return
        
        guid_s = self._guid_wall_start
        guid_e = self._guid_wall_end
        p3_s = self._wall_times["P3_grad_backward"]
        p3_e = self._wall_times["end"]
        
        # Guidance GPU 活跃时间（ms）
        guid_active_ms = (guid_e - guid_s) * 1000.0
        timings["guid_gpu_active"] = guid_active_ms
        
        # guid 与 P3 的重叠时间 = 真正的异步收益（ms）
        overlap_ms = max(0.0, (min(guid_e, p3_e) - max(guid_s, p3_s))) * 1000.0
        timings["overlap_guid_P3"] = overlap_ms
        
        # Train GPU 空闲 ≈ total - 有 GPU 计算的 phase 之和（ms）
        train_active_phases = ["dense_sampling", "P1_rollout",
                               "P2a_decode_render", "P3_grad_backward"]
        train_active_ms = sum(timings.get(p, 0.0) for p in train_active_phases)
        timings["train_gpu_idle"] = max(0.0, timings.get("total", 0.0) - train_active_ms)
        
        # Guidance GPU 空闲 ≈ 总挂钟时间 - guid 活跃时间（ms）
        # 取 _wall_times 中最早的 tick（由于 dict 覆盖，实际是 iter N 的第一个 tick）
        wall_keys = list(self._wall_times.keys())
        if wall_keys:
            first_wall = self._wall_times[wall_keys[0]]
            total_wall_ms = (self._wall_times["end"] - first_wall) * 1000.0
            timings["guid_gpu_idle"] = max(0.0, total_wall_ms - guid_active_ms)
    
    def _reset_gpu_util(self) -> None:
        """重置挂钟时间状态。在 reset() 中被调用。"""
        self._wall_times.clear()
        self._guid_wall_start = 0.0
        self._guid_wall_end = 0.0


# =====================================================================
# AsyncPhaseProfiler — 异步流水线专用
# =====================================================================

# GPU 利用率指标的 key 集合（用于打印分区）
_GPU_UTIL_KEYS = frozenset({
    "guid_gpu_active", "overlap_guid_P3",
    "train_gpu_idle", "guid_gpu_idle",
})


@dataclass
class AsyncPhaseProfiler(GPUUtilizationMixin, PhaseProfiler):
    """
    PhaseProfiler + 双 GPU 利用率分析，用于异步 Guidance 流水线。
    
    在 PhaseProfiler 的 CUDA Event 计时基础上，额外：
    - 记录每个 tick 的挂钟时间（time.perf_counter）
    - 接收 Guidance GPU 的活跃时段（set_guid_timing）
    - 计算 overlap / idle 指标
    
    输出示例：
        [AsyncPhaseProfiler] Step 10 | avg over 10 steps (ms):
          dense_sampling                     3322.3
          P1_rollout                        11502.9
          ...
          total                            131049.6
          ── GPU utilization ──
          guid_gpu_active                   59123.4
          overlap_guid_P3                   55000.0  ← 异步收益
          train_gpu_idle                     1500.0
          guid_gpu_idle                     16000.0
    """
    
    def __post_init__(self):
        self._init_gpu_util()
    
    def tick(self, name: str) -> None:
        super().tick(name)
        if self.enabled:
            self._tick_wall(name)
    
    def elapsed(self) -> Dict[str, float]:
        timings = super().elapsed()
        if self.enabled:
            self._compute_gpu_util(timings)
        return timings
    
    def reset(self) -> None:
        super().reset()
        self._reset_gpu_util()
    
    def _print_summary(self, avg: Dict[str, float], global_step: int) -> None:
        """覆盖打印：常规 phase + GPU 利用率分区显示。"""
        n = len(next(iter(self._history.values())))
        lines = [f"[AsyncPhaseProfiler] Step {global_step} | avg over {n} steps (ms):"]
        
        # 常规 phase（排除 GPU 利用率指标和 total）
        for k in avg:
            if k != "total" and k not in _GPU_UTIL_KEYS:
                lines.append(f"  {k:30s} {avg[k]:8.1f}")
        if "total" in avg:
            lines.append(f"  {'total':30s} {avg['total']:8.1f}")
        
        # GPU 利用率区域
        if any(k in avg for k in _GPU_UTIL_KEYS):
            lines.append(f"  {'── GPU utilization ──':30s}")
            for k in ["guid_gpu_active", "overlap_guid_P3",
                       "train_gpu_idle", "guid_gpu_idle"]:
                if k in avg:
                    lines.append(f"  {k:30s} {avg[k]:8.1f}")
        
        logging.info("\n".join(lines))
