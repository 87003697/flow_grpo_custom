"""Mixin 基类 - 可复用的功能模块"""
import csv
import torch
from accelerate import Accelerator
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image


# =====================================================================
# Mixin 基类 - 可复用的功能模块
# =====================================================================

class AcceleratorMixin:
    """Accelerator 基础功能封装。
    
    提供 accelerator 相关的通用操作，如主进程判断、设备获取、同步等。
    子类需要提供 self.accelerator 属性。
    """
    
    accelerator: Accelerator = None
    
    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.accelerator is None or self.accelerator.is_main_process
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        return self.accelerator.device if self.accelerator else torch.device("cpu")
    
    def wait_for_everyone(self) -> None:
        """等待所有进程同步"""
        if self.accelerator:
            self.accelerator.wait_for_everyone()


class DistributedMixin(AcceleratorMixin):
    """分布式聚合功能。
    
    提供跨进程的 reduce 操作，用于分布式训练时的指标聚合。
    """
    
    def reduce_sum(self, value: float) -> float:
        """跨进程求和"""
        if self.accelerator is None:
            return value
        t = torch.tensor([value], device=self.device)
        return self.accelerator.reduce(t, reduction="sum").item()
    
    def reduce_mean(self, total_sum: float, total_count: float) -> float:
        """跨进程求均值（先聚合 sum 和 count，再计算均值）"""
        global_sum = self.reduce_sum(total_sum)
        global_count = self.reduce_sum(total_count)
        return global_sum / max(global_count, 1e-8)


class WandbMixin(AcceleratorMixin):
    """Wandb 日志功能。
    
    通过 accelerator.log() 记录指标和图像到 wandb。
    需要在 Accelerator 初始化时设置 log_with=["wandb"]。
    """
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """记录指标到 wandb"""
        if self.is_main_process and self.accelerator:
            self.accelerator.log(metrics, step=step)
    
    def log_images(self, images: Dict[str, Image.Image], step: int, prefix: str = "") -> None:
        """记录图像到 wandb"""
        if self.is_main_process and self.accelerator:
            import wandb
            log_dict = {}
            for k, v in images.items():
                key = f"{prefix}/{k}" if prefix else k
                log_dict[key] = wandb.Image(v)
            self.accelerator.log(log_dict, step=step)


class CSVMixin(AcceleratorMixin):
    """CSV 日志功能。
    
    提供 CSV 文件日志记录，仅主进程写入。
    子类需要提供 self.csv_path 属性。
    """
    
    csv_path: Path = None
    
    def log_csv(self, row: Dict[str, Any]) -> None:
        """追加一行到 CSV 文件（仅主进程）"""
        if not self.is_main_process or not self.csv_path:
            return
        
        path = self.csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        fieldnames = list(row.keys())
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


class AccumulatorMixin:
    """指标累积功能。
    
    用于累积多个步骤的指标，支持 batch 加权平均。
    """
    
    def reset(self) -> None:
        """重置累积器"""
        self.sums: Dict[str, float] = {}
        self.count: float = 0.0
    
    def accumulate(self, logs: Dict[str, Any], batch_size: int) -> None:
        """累积一个步骤的指标"""
        bs = float(batch_size)
        self.count += bs
        for k, v in logs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = float(v.detach().item())
            self.sums.setdefault(k, 0.0)
            self.sums[k] += v * bs
    
    def average(self) -> Optional[Dict[str, float]]:
        """计算本地累积的平均值"""
        if self.count <= 0:
            return None
        return {k: v / self.count for k, v in self.sums.items()}
