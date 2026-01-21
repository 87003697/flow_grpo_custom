"""MetricLogger - 统一的指标记录器"""
import csv
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict, Optional

from .mixins import AccumulatorMixin, DistributedMixin, CSVMixin, WandbMixin


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """
    追加写入 CSV 日志文件。
    
    如果文件不存在，先写入表头；如果存在，追加数据行。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
        """计算分布式聚合后的平均值（利用 DistributedMixin）。"""
        if self.count <= 0.0:
            return None
        
        # 聚合 count
        total_count = self.reduce_sum(self.count)
        if total_count <= 0.0:
            return None
        
        # 聚合各指标的 sum
        result = {}
        for k, v in self.sums.items():
            total_sum = self.reduce_sum(v)
            result[k] = total_sum / total_count
        
        return result

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

