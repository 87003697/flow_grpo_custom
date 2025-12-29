"""
edit4shape.systems.utils
========================

通用工具函数和类，供训练/评估系统使用。
"""

import csv
import os
import torch
from accelerate import Accelerator
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional
from pathlib import Path


# =====================================================================
# CSV 日志工具
# =====================================================================

def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """
    追加写入 CSV 日志文件。
    
    如果文件不存在，先写入表头；如果存在，追加数据行。
    
    Args:
        path: CSV 文件路径
        row: 要写入的数据行（字典格式）
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
# MetricLogger - 统一的指标记录器
# =====================================================================

class MetricLogger:
    """
    统一的指标记录器（支持分布式聚合和梯度累积）。
    
    功能特性：
    1. 累积多个步骤的指标
    2. 分布式聚合（多 GPU 时聚合所有进程的值）
    3. 自动处理梯度累积（训练场景）
    4. 发射到 CSV 和实验追踪器
    
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
            accelerator: Accelerate 加速器（用于分布式聚合和梯度同步判断）
            csv_path: CSV 日志输出路径
        """
        self.accelerator = accelerator
        self.csv_path = csv_path
        self.reset()

    def reset(self) -> None:
        """重置累积器。"""
        self.sums: Dict[str, float] = {}
        self.count = 0.0

    def accumulate(self, logs: Dict[str, Any], batch_size: int) -> None:
        """
        累积一个步骤的指标。
        
        Args:
            logs: 指标字典，如 {"loss/total": tensor, "metric/ssim": 0.9}
            batch_size: 当前 batch 大小
        """
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
        """计算本地累积的平均值（不进行分布式聚合）。"""
        if self.count <= 0.0:
            return None
        return {k: v / self.count for k, v in self.sums.items()}

    def _distributed_average(self) -> Optional[Dict[str, float]]:
        """计算分布式聚合后的平均值。"""
        if self.count <= 0.0:
            return None
        
        if self.accelerator is None:
            return {k: v / self.count for k, v in self.sums.items()}
        
        device = self.accelerator.device
        
        # 聚合 count
        count_tensor = torch.tensor([self.count], device=device)
        total_count = self.accelerator.reduce(count_tensor, reduction="sum").item()
        
        if total_count <= 0.0:
            return None
        
        # 聚合各指标的 sum
        result = {}
        for k, v in self.sums.items():
            sum_tensor = torch.tensor([v], device=device)
            total_sum = self.accelerator.reduce(sum_tensor, reduction="sum").item()
            result[k] = total_sum / total_count
        
        return result

    def flush(self, global_step: int, epoch: int) -> Optional[Dict[str, float]]:
        """
        发射累积的平均日志并重置。
        
        在分布式训练中会先同步所有进程再聚合。
        
        Args:
            global_step: 全局步数
            epoch: 当前 epoch
        
        Returns:
            发射的平均日志字典
        """
        if self.accelerator:
            self.accelerator.wait_for_everyone()
        
        avg_log = self._distributed_average()
        if avg_log:
            self._emit(avg_log, global_step, epoch)
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
        
        Args:
            logs: 指标字典
            batch_size: 当前 batch 大小
            global_step: 全局步数
            epoch: 当前 epoch
        """
        self.accumulate(logs, batch_size)
        
        if self.accelerator is None or self.accelerator.sync_gradients:
            avg_log = self._distributed_average()
            if avg_log:
                self._emit(avg_log, global_step, epoch)
            self.reset()

    def _emit(self, log_dict: Dict[str, float], global_step: int, epoch: int) -> None:
        """发射日志到 CSV 和实验追踪器。"""
        if not log_dict:
            return
        
        # 仅主进程写入 CSV
        if self.accelerator is None or self.accelerator.is_main_process:
            if self.csv_path:
                row = {"global_step": global_step, "epoch": epoch, **log_dict}
                append_csv_row(self.csv_path, row)
        
        # 发射到实验追踪器
        if self.accelerator:
            self.accelerator.log(log_dict, step=global_step)


# =====================================================================
# VisualIO - 训练可视化保存
# =====================================================================


class VisualIO:
    """
    负责保存条件/生成/编辑三联图的简易 I/O。
    """

    def __init__(self, root: Path, target_h: int = 512, vis_freq: int = 100):
        self.root = root
        self.target_h = target_h
        self.vis_freq = vis_freq

    @staticmethod
    def _to_pil(x) -> Image.Image:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()  # (H,W,C)
        x = (x * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
        return Image.fromarray(x)

    def _resize_h(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_h / max(1, h)  # ()
        new_w = max(1, int(round(w * scale)))  # ()
        return img.resize((new_w, self.target_h), Image.Resampling.LANCZOS)

    def _save_triptych(self, save_path: Path, cond_pil, gen_tensor, edit_tensor=None) -> None:
        imgs = [
            self._resize_h(cond_pil.convert("RGB")),
            self._resize_h(self._to_pil(gen_tensor)),
        ]
        if edit_tensor is not None:
            imgs.append(self._resize_h(self._to_pil(edit_tensor)))

        margin = 12
        total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
        total_h = max(im.height for im in imgs) + margin * 2
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x = margin
        for im in imgs:
            canvas.paste(im, (x, margin))
            x += im.width + margin
        canvas.save(save_path)

    def save_batch(self, state, epoch: int, step: int) -> None:
        """
        从 TrellisState 中提取并保存一批三联图。
        需 state.views_conditioned.paths/image_pils, state.views_generated.image_tensor, state.views_edited.image_tensor。
        """
        image_paths = state.views_conditioned.paths
        image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

        conditioned = state.views_conditioned.image_pils  # list[len=B] of PIL
        render_color = state.views_generated.image_tensor  # (B,V,H,W,C)
        edited = state.views_edited.image_tensor  # (B,V,C,H,W) or None

        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for b, name in enumerate(image_names):
            cond = conditioned[b]  # PIL
            gen = render_color[b, 0]  # (H,W,C)
            edt = edited[b, 0].permute(1, 2, 0) if edited is not None else None  # (H,W,C)
            self._save_triptych(out_dir / f"{name}.png", cond, gen, edt)

