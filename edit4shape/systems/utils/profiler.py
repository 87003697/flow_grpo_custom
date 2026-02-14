"""
PhaseProfiler — 基于 CUDA Event 的 Phase 级别计时器。

用于测量训练步中各 Phase 的 GPU 耗时，辅助异步流水线优化决策。

使用方式:
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
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


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
        
        # 打印（仅 verbose=True 时，避免多 rank 重复输出）
        if self.verbose:
            n = len(next(iter(self._history.values())))
            lines = [f"[PhaseProfiler] Step {global_step} | avg over {n} steps (ms):"]
            
            # 按 _order 顺序打印（total 放最后）
            for k in avg:
                if k != "total":
                    lines.append(f"  {k:30s} {avg[k]:8.1f}")
            if "total" in avg:
                lines.append(f"  {'total':30s} {avg['total']:8.1f}")
            
            logging.info("\n".join(lines))
        
        # 清除累积
        self._history.clear()
        self._step_count = 0
        
        return avg
    
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
