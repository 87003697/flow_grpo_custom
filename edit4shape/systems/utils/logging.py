"""MetricLogger - 统一的指标记录器"""
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from .mixins import AccumulatorMixin, DistributedMixin, CSVMixin, WandbMixin


# =====================================================================
# MetricLogger - 统一的指标记录器（使用 Mixin 组合）
# =====================================================================

class MetricLogger(AccumulatorMixin, DistributedMixin, CSVMixin, WandbMixin):
    """
    统一的指标记录器（支持分布式聚合、CSV、Wandb）。
    
    继承自多个 Mixin，提供：
    - AccumulatorMixin: 指标累积 (reset, accumulate, average)
    - DistributedMixin: 分布式聚合 (reduce_sum, reduce_mean)
    - CSVMixin: CSV 日志 (log_csv)
    - WandbMixin: Wandb 日志 (log_metrics)
    
    使用模式：
    
    1. 训练模式（自动梯度累积）：
        logger = MetricLogger(accelerator, logs_dir / "train.csv")
        for batch in train_loader:
            with accelerator.accumulate(model):
                train_log = train_step(...)
            logger.log_step(train_log, batch_size, global_step, epoch)
    
    2. 评估模式（手动控制）：
        logger = MetricLogger(accelerator, logs_dir / "eval.csv")
        for batch in eval_loader:
            metrics = evaluate(...)
            logger.accumulate(metrics, batch_size)
        logger.flush(global_step, epoch)
    """

    def __init__(self, accelerator: Accelerator = None, csv_path: Path = None):
        """
        初始化。
        
        Args:
            accelerator: Accelerate 加速器（用于分布式聚合和 wandb）
            csv_path: CSV 日志输出路径
        """
        self.accelerator = accelerator
        self.csv_path = csv_path
        self.reset()

    def _distributed_average(self) -> Optional[Dict[str, float]]:
        """
        计算分布式聚合后的平均值。
        
        安全实现：先 all_gather 所有 rank 的 key 取并集，再打包为单个 tensor
        做一次 all_reduce。即使各 rank 的 log key 不一致（如某个 rank OOM 导致
        guidance key 缺失）也不会死锁。
        """
        if self.count <= 0.0:
            return None
        
        # ★ 收集所有 rank 的 key 集合，取并集 → 任何 rank 有的 key 都不会丢
        if dist.is_initialized():
            local_keys = set(self.sums.keys())
            all_keys_list = [None] * dist.get_world_size()
            dist.all_gather_object(all_keys_list, local_keys)
            keys = sorted(set().union(*all_keys_list))
        else:
            keys = sorted(self.sums.keys())
        
        # 打包 [count, v0, v1, ...] → 缺失 key 补 0.0，单次 all_reduce
        vals = [self.count] + [float(self.sums.get(k, 0.0)) for k in keys]
        t = torch.tensor(vals, device=self.device)  # (len(keys)+1,)
        t = self.accelerator.reduce(t, reduction="sum")  # 仅 1 次 all_reduce
        
        total_count = t[0].item()
        if total_count <= 0.0:
            return None
        return {k: t[i + 1].item() / total_count for i, k in enumerate(keys)}

    def flush(self, global_step: int, epoch: int) -> Optional[Dict[str, float]]:
        """
        发射累积的平均日志并重置。
        
        在分布式训练中会先同步所有进程再聚合。
        """
        self.wait_for_everyone()
        
        avg_log = self._distributed_average()
        if avg_log:
            # CSV 日志（利用 CSVMixin）
            self.log_csv({"global_step": global_step, "epoch": epoch, **avg_log})
            # Wandb 日志（利用 WandbMixin）
            self.log_metrics(avg_log, step=global_step)
        
        self.reset()
        return avg_log

    def log_step(
        self, 
        logs: Dict[str, Any], 
        batch_size: int, 
        global_step: int, 
        epoch: int,
    ) -> None:
        """
        记录单步日志（自动处理梯度累积）。
        
        每步都累积，仅在 sync_gradients 时发射并重置。
        """
        self.accumulate(logs, batch_size)
        
        if self.accelerator is None or self.accelerator.sync_gradients:
            avg_log = self._distributed_average()
            if avg_log:
                self.log_csv({"global_step": global_step, "epoch": epoch, **avg_log})
                self.log_metrics(avg_log, step=global_step)
            self.reset()

