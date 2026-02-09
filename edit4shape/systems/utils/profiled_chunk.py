"""
Profile-based 自适应 chunk 调度器。

核心类：
- ProfiledScheduler: 生成器，自动 profiling 显存消耗并动态计算 chunk_size

原理：
1. 第一次 yield 返回 probe chunk（小），调用方处理后控制权返回
2. 测量 probe 期间的峰值显存增量 → 计算 bytes_per_item
3. 根据剩余显存和 bytes_per_item 计算最优 chunk_size
4. 后续 yield 使用最优 chunk_size

用法：
    # cat 合并（normal 分块）
    results = []
    for start, size in ProfiledScheduler(K, device, probe_size=2000):
        results.append(compute(data[start:start+size]))
    output = torch.cat(results, dim=0)

    # z-buffer 合并（rast 分块）
    for start, size in ProfiledScheduler(num_faces, device, probe_size=50000):
        rast = rasterize(faces[start:start+size])
        zbuffer_merge(...)
"""

import logging
import torch


class ProfiledScheduler:
    """
    Profile-based 自适应 chunk 调度器（生成器）。

    利用 Python generator 的 yield 暂停/恢复机制：
    - 第一次 yield 后，调用方处理 probe chunk
    - 控制权返回时测量显存，计算最优 chunk_size
    - 后续 yield 使用最优 chunk_size

    chunk_size 计算公式::

        bytes_per_item = probe_delta / probe_size × safety_factor
        available      = (GPU总显存 - 当前占用) × target_usage
        chunk_size     = clamp(available / bytes_per_item, min_chunk, max_chunk)

    三道防线：
    1. safety_factor — 单位估算层面的保守（高估 bytes_per_item → 低估 chunk_size）
    2. target_usage  — 全局显存层面的保守（预留给 allocator 碎片 / 其他进程）
    3. min/max_chunk — 硬性钳位兜底

    Args:
        total: 总 item 数量。
        device: CUDA 设备。
        probe_size: 探针大小（第一个 chunk 的大小）。
            - 太小 → 显存 delta 被 CUDA allocator 对齐/碎片噪声淹没，估算不准。
            - 太大 → 显存紧张时 probe 本身 OOM。
            - 经验：取预期 chunk_size 的 1/10 ~ 1/50。
            - 重计算场景（如 26-neighbor normal）用小值（~2000）；
              GPU 并行场景（如 nvdiffrast rasterize）用大值（~50000）以获得稳定信号。
        safety_factor: 安全系数，乘在 bytes_per_item 上。
            - 1.3（30% 余量）：适用于开销均匀的场景（如 voxel normal 计算）。
            - 1.5（50% 余量）：适用于波动较大的场景（如 nvdiffrast 内部 buffer 预分配、
              面片密度不均）。
        target_usage: 目标显存利用率（0~1），默认 0.85。
            - 保留 15% 给 PyTorch allocator 内部碎片和其他进程。
            - 与 safety_factor 形成双重保险。
        min_chunk: 最小 chunk 大小。
            - 防止显存紧张时 chunk_size 过小导致循环次数爆炸、Python 循环 overhead 主导。
            - 重计算场景 ~500；GPU 并行场景 ~10000。
        max_chunk: 最大 chunk 大小。
            - 防止 probe 低估 bytes_per_item 导致下一个 chunk OOM。
            - nvdiffrast 硬限制：单次 rasterize ≤ 2^24（~1677万）面片，
              rast 场景建议 max_chunk ≤ 2000000。
    """

    def __init__(
        self,
        total: int,
        device: torch.device,
        probe_size: int = 2000,
        safety_factor: float = 1.3,
        target_usage: float = 0.85,
        min_chunk: int = 500,
        max_chunk: int = 500000,
    ):
        self.total = total
        self.device = device
        self.probe_size = min(probe_size, total)
        self.safety_factor = safety_factor
        self.target_usage = target_usage
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        # profiling 结果（外部可读）
        self.bytes_per_item: float = 0.0
        self.chunk_size: int = 0

    def __iter__(self):
        dev_idx = self.device.index if self.device.index is not None else 0
        total = self.total
        probe_n = self.probe_size

        # 总量 ≤ probe_size，一次搞定
        if total <= probe_n:
            yield (0, total)
            return

        # ---- Phase 1: Probe ----
        torch.cuda.reset_peak_memory_stats(dev_idx)
        mem_before = torch.cuda.max_memory_allocated(dev_idx)

        yield (0, probe_n)  # 暂停，调用方处理 probe chunk

        # 恢复：probe 已处理完，测量显存
        mem_after = torch.cuda.max_memory_allocated(dev_idx)
        delta = max(mem_after - mem_before, 1)
        self.bytes_per_item = delta / probe_n * self.safety_factor

        # 计算最优 chunk_size
        total_mem = torch.cuda.get_device_properties(dev_idx).total_memory
        allocated_now = torch.cuda.memory_allocated(dev_idx)
        available = int((total_mem - allocated_now) * self.target_usage)
        optimal = int(available / self.bytes_per_item) if self.bytes_per_item > 0 else self.max_chunk
        self.chunk_size = max(self.min_chunk, min(optimal, self.max_chunk))

        logging.debug(
            f"[ProfiledScheduler] probe={probe_n}, "
            f"delta={delta / 1024 / 1024:.1f}MB, "
            f"bytes/item={self.bytes_per_item:.0f}, "
            f"chunk_size={self.chunk_size}, total={total}"
        )

        # ---- Phase 2: 最优分块 ----
        offset = probe_n
        while offset < total:
            size = min(self.chunk_size, total - offset)
            yield (offset, size)
            offset += size
