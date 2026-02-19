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

使用方式（异步版，双 GPU 利用率 + 显存诊断）:
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
    # 显存:    mem@P1_rollout, mem@P2_wait_backward, ..., mem@peak  (GiB)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


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
# MemoryMixin — 每阶段 GPU 显存快照
# =====================================================================

# 显存快照指标的 key 前缀（每个 phase 结束时的 allocated GiB）
_MEM_PREFIX = "mem@"


class MemoryMixin:
    """
    为 PhaseProfiler 添加 GPU 显存追踪。

    在每个 tick 时记录 ``torch.cuda.memory_allocated()``，
    elapsed() 时注入每阶段结束显存 (``mem@{phase}``) 和本步峰值 (``mem@peak``)。

    设计意图：
      与 GPUUtilizationMixin 平行的 Mixin，AsyncPhaseProfiler 同时组合两者。
      每阶段的显存快照可定位 OOM 高水位阶段，峰值可监控整体趋势。

    Key 约定（timings dict 内）：
      ``mem@dense_sampling``  → dense_sampling 结束时的 allocated GiB
      ``mem@peak``            → 本步的 max_memory_allocated GiB
    """

    # ---- 需要由组合类提供 ----
    enabled: bool
    _order: List[str]

    def _init_mem(self) -> None:
        """初始化显存追踪状态。子类在 __post_init__ 中调用。"""
        self._mem_at_tick: Dict[str, float] = {}  # tick_name -> allocated GiB

    def _tick_mem(self, name: str) -> None:
        """记录一个阶段边界的 GPU allocated 显存。在 tick() 中被调用。"""
        self._mem_at_tick[name] = torch.cuda.memory_allocated() / (1024 ** 3)  # GiB

    def _compute_mem(self, timings: Dict[str, float]) -> None:
        """
        计算每阶段结束显存 + 峰值，原地写入 timings 字典。

        在 elapsed() 末尾被调用。

        写入的 key：
          ``mem@{phase_name}`` — 该 phase 结束时的 allocated 显存 (GiB)
          ``mem@peak``         — 本步 max_memory_allocated (GiB)

        Args:
            timings: elapsed() 已计算的 phase timing 字典（会被原地扩充）
        """
        for i in range(len(self._order) - 1):
            phase_name = self._order[i]
            end_tick = self._order[i + 1]
            if end_tick in self._mem_at_tick:
                timings[f"{_MEM_PREFIX}{phase_name}"] = self._mem_at_tick[end_tick]
        timings[f"{_MEM_PREFIX}peak"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

    def _reset_mem(self) -> None:
        """重置显存追踪状态 + 峰值统计。在 reset() 中被调用。"""
        self._mem_at_tick.clear()
        torch.cuda.reset_peak_memory_stats()  # 下一步的 peak 从当前值重新计


# =====================================================================
# AsyncPhaseProfiler — 异步流水线专用
# =====================================================================

# GPU 利用率指标的 key 集合（用于打印分区）
_GPU_UTIL_KEYS = frozenset({
    "guid_gpu_active", "overlap_guid_P3",
    "train_gpu_idle", "guid_gpu_idle",
})


@dataclass
class AsyncPhaseProfiler(MemoryMixin, GPUUtilizationMixin, PhaseProfiler):
    """
    PhaseProfiler + 双 GPU 利用率分析 + 显存诊断，用于异步 Guidance 流水线。
    
    组合三个正交维度：
    - PhaseProfiler:       CUDA Event 计时
    - GPUUtilizationMixin: 双 GPU 挂钟时间 + 利用率分析
    - MemoryMixin:         每阶段 GPU allocated 快照 + 峰值追踪
    
    显存 key 约定：
      timings dict 中以 ``mem@`` 前缀存储（GiB），
      as_log_dict() 转为 ``mem/`` 前缀写入 CSV / wandb。
    
    跨 rank 显存可见性：
      step() 覆写后，每个 rank 独立用 logging.info 打印一行显存摘要，
      配合 _setup_file_logging 的 per-rank 日志文件，可事后对比各 rank 显存。
    
    输出示例（rank 0，verbose=True 时额外输出完整 phase 汇总）：
        [AsyncPhaseProfiler] Step 10 | avg over 10 steps:
          dense_sampling                     3322.3 ms  | mem 45.3 GiB
          P1_rollout                        11502.9 ms  | mem 62.1 GiB
          ...
          total                            131049.6 ms  | peak 92.6 GiB

    每个 rank 都会打印一行显存摘要（写入各自的 run_rank{i}.log）：
        [Rank 0 Step 10] mem(GiB): dense_sampling: 45.3, ..., peak=73.2
        [Rank 2 Step 10] mem(GiB): dense_sampling: 46.1, ..., peak=92.6
    """
    
    def __post_init__(self):
        self._init_gpu_util()
        self._init_mem()
    
    def tick(self, name: str) -> None:
        super().tick(name)
        if self.enabled:
            self._tick_wall(name)
            self._tick_mem(name)
    
    def elapsed(self) -> Dict[str, float]:
        timings = super().elapsed()
        if self.enabled:
            self._compute_gpu_util(timings)
            self._compute_mem(timings)
        return timings
    
    def step(
        self,
        timings: Dict[str, float],
        global_step: int,
        print_freq: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        覆写：print_freq 触发时，所有 rank 独立打印一行显存摘要。

        详细 phase 汇总由 _print_summary 处理（verbose-gated，仅 rank 0）；
        显存摘要由 _log_mem_all_ranks 处理（所有 rank，写入各自的日志文件）。
        """
        avg = super().step(timings, global_step, print_freq)
        if avg is not None:
            self._log_mem_all_ranks(avg, global_step)
        return avg

    def reset(self) -> None:
        super().reset()
        self._reset_gpu_util()
        self._reset_mem()
    
    def as_log_dict(self, timings: Dict[str, float], prefix: str = "time/") -> Dict[str, float]:
        """
        覆写：mem@ 开头的 key 使用 ``mem/`` 前缀，其余沿用 ``time/`` 前缀。

        这样 CSV / wandb 日志中的 key 为：
          time/dense_sampling, time/P1_rollout, ...
          mem/dense_sampling, mem/P1_rollout, ..., mem/peak
        """
        if not timings:
            return {}
        result: Dict[str, float] = {}
        for k, v in timings.items():
            if k.startswith(_MEM_PREFIX):
                # mem@phase_name -> mem/phase_name
                result[f"mem/{k[len(_MEM_PREFIX):]}"] = v
            else:
                result[f"{prefix}{k}"] = v
        return result

    def _log_mem_all_ranks(self, avg: Dict[str, float], global_step: int) -> None:
        """
        每个 rank 独立打印显存摘要（通过 logging.info）。

        配合 _setup_file_logging 为每个 rank 创建的 run_rank{i}.log，
        事后可对比各 rank 的显存用量，定位单 rank OOM。
        不使用 all_reduce，零通信开销。
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        peak_key = f"{_MEM_PREFIX}peak"
        parts: List[str] = []
        for k, v in avg.items():
            if k.startswith(_MEM_PREFIX) and k != peak_key:
                parts.append(f"{k[len(_MEM_PREFIX):]}: {v:.1f}")
        peak = avg.get(peak_key, 0.0)

        logging.info(
            f"[Rank {rank} Step {global_step}] "
            f"mem(GiB): {', '.join(parts)}, peak={peak:.1f}"
        )

    def _print_summary(self, avg: Dict[str, float], global_step: int) -> None:
        """覆盖打印：常规 phase + 显存快照 + GPU 利用率。"""
        n = len(next(iter(self._history.values())))
        lines = [f"[AsyncPhaseProfiler] Step {global_step} | avg over {n} steps:"]

        # 构造 phase -> 显存 的映射
        mem_for_phase: Dict[str, float] = {}
        for k, v in avg.items():
            if k.startswith(_MEM_PREFIX) and k != f"{_MEM_PREFIX}peak":
                mem_for_phase[k[len(_MEM_PREFIX):]] = v

        # 常规 phase（排除 GPU 利用率指标、total 和 mem@ 指标）
        for k in avg:
            if k != "total" and k not in _GPU_UTIL_KEYS and not k.startswith(_MEM_PREFIX):
                mem_str = f"  | mem {mem_for_phase[k]:.1f} GiB" if k in mem_for_phase else ""
                lines.append(f"  {k:30s} {avg[k]:8.1f} ms{mem_str}")
        if "total" in avg:
            peak_key = f"{_MEM_PREFIX}peak"
            peak_str = f"  | peak {avg[peak_key]:.1f} GiB" if peak_key in avg else ""
            lines.append(f"  {'total':30s} {avg['total']:8.1f} ms{peak_str}")

        # GPU 利用率区域
        if any(k in avg for k in _GPU_UTIL_KEYS):
            lines.append(f"  {'── GPU utilization ──':30s}")
            for k in ["guid_gpu_active", "overlap_guid_P3",
                       "train_gpu_idle", "guid_gpu_idle"]:
                if k in avg:
                    lines.append(f"  {k:30s} {avg[k]:8.1f} ms")

        logging.info("\n".join(lines))
