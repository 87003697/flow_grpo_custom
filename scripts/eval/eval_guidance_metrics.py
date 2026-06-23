"""
独立评估脚本：评估 Guidance 前后 CLIP / DINO / SilhouetteIoU 指标。

不修改任何 system 代码，仅复用已有组件。支持单卡和 DDP（accelerate launch）。

数据流：
    1. trellis_forward → 渲染图（Guidance 前）
    2. guidance.compute_guidance → 编辑图（Guidance 后）
    3. CLIPMetric / DINOMetric / SilhouetteExtractor 计算指标
    4. 增量写 CSV + 最终 JSON

用法（单卡）：
    bash scripts/eval/eval_mesh_scorer_eval3d.sh
用法（DDP 多卡）：
    bash scripts/eval/eval_guidance_metric_DDP.sh
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# TRELLIS 路径（必须在 trellis 导入之前）
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from absl import app
from accelerate import Accelerator

from edit4shape.systems.trellis.system import (
    build_system, trellis_forward, _CONFIG,
)
from edit4shape.systems.base import (
    setup_env_and_seed, EvalModeGuard, CheckpointIO,
)
from edit4shape.datasets.trellis import (
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
    TrellisDataConfig,
    TrellisDataModule,
)
from edit4shape.generators.trellis.state import TrellisState
from edit4shape.guidance import create_guidance
from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.metric.clip import CLIPMetric
from edit4shape.guidance.metric.dino import DINOMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

def build_eval_dataloader(cfg, accelerator: Accelerator):
    """按 range 相机配置构建 eval dataloader，避免依赖旧字段 yaw/pitch/r/fov。"""
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=cfg.data.train.adaptive_distance,
    )
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw_range=list(cfg.data.eval.yaw_range),
        pitch_range=list(cfg.data.eval.pitch_range),
        r_range=list(cfg.data.eval.r_range),
        fov_range=list(cfg.data.eval.fov_range),
        adaptive_distance=cfg.data.eval.adaptive_distance,
    )
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,
        eval_batch_size=cfg.data.eval.batch_size,
        width=cfg.renderer.resolution,
        height=cfg.renderer.resolution,
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )
    dm = TrellisDataModule(
        dm_cfg,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dm.setup(stage="test")
    return dm.eval_dataloader()

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL → (1,3,H,W) float [0,1]，自动 RGBA 合成白底。"""
    return TF.to_tensor(composite_alpha_to_white(img)).unsqueeze(0).to(device)  # (1,3,H,W)


def _to_bchw(view: torch.Tensor, fmt: str = "hwc") -> torch.Tensor:
    """视角 tensor → (1,C,H,W)。fmt='hwc' 对应渲染图，'chw' 对应编辑图。"""
    if fmt == "hwc":
        return view.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) → (1,C,H,W)
    return view.unsqueeze(0)  # (C,H,W) → (1,C,H,W)


def _similarity(metric, a, b) -> float:
    """metric.compute / compute_from_pil 返回 loss=1-sim，这里返回 similarity。"""
    with torch.no_grad():
        if isinstance(a, Image.Image):
            return 1.0 - metric.compute_from_pil([a], [b]).item()
        if isinstance(a, list) and len(a) > 0 and isinstance(a[0], Image.Image):
            return 1.0 - metric.compute_from_pil(a, b).item()
        return 1.0 - metric.compute(a, b).item()


def _to_pil(t: torch.Tensor) -> Image.Image:
    """(1,C,H,W) 或 (C,H,W) float [0,1] → PIL RGB。"""
    if t.dim() == 4:
        t = t.squeeze(0)  # (1,C,H,W) → (C,H,W)
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
    return Image.fromarray((t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))


def _save_images(images_dir: Path, name: str, cond_pil: Image.Image,
                 before: torch.Tensor, after: torch.Tensor, v: int) -> None:
    """保存 condition / before / after / grid 图片。"""
    d = images_dir / name
    d.mkdir(parents=True, exist_ok=True)

    if v == 0:
        cond_pil.save(d / "condition.png")

    bp = _to_pil(before)
    bp.save(d / f"v{v}_before.png")
    ap = _to_pil(after)
    ap.save(d / f"v{v}_after.png")

    # 三图拼接 grid [condition | before | after]
    margin = 12
    h = bp.height
    c = cond_pil.copy()
    if c.height != h:
        s = h / c.height
        c = c.resize((max(1, int(c.width * s)), h), Image.LANCZOS)
    imgs = [c, bp, ap]
    total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
    grid = Image.new("RGB", (total_w, h + margin * 2), (255, 255, 255))
    x = margin
    for im in imgs:
        grid.paste(im, (x, margin))
        x += im.width + margin
    grid.save(d / f"v{v}_grid.png")


# =====================================================================
# SilhouetteExtractor — RMBG-2.0 前景 mask + IoU
# =====================================================================

class SilhouetteExtractor:
    """使用 RMBG-2.0 提取前景 mask 并计算 IoU。"""

    PATH = "pretrained_weights/rmbg2/RMBG-2.0"
    SIZE = 1024
    THRESH = 0.5

    def __init__(self, device: torch.device):
        from transformers import AutoModelForImageSegmentation
        logger.info(f"[SilhouetteExtractor] Loading {self.PATH}")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.PATH, trust_remote_code=True,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)  # (1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)   # (1,3,1,1)

    @torch.no_grad()
    def _mask(self, img: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) float [0,1] → (1,1,H,W) bool mask。"""
        h, w = img.shape[2], img.shape[3]
        x = F.interpolate(img.to(self.device), (self.SIZE, self.SIZE),
                          mode="bilinear", align_corners=False)  # (1,3,1024,1024)
        x = (x - self.mean) / self.std  # (1,3,1024,1024)
        pred = self.model(x)[-1].sigmoid()  # (1,1,1024,1024)
        mask = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)  # (1,1,H,W)
        return mask > self.THRESH  # (1,1,H,W) bool

    def iou(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """计算两张图前景 mask 的 IoU。"""
        ma, mb = self._mask(a), self._mask(b)
        inter = (ma & mb).sum().float()  # scalar
        union = (ma | mb).sum().float()  # scalar
        return 1.0 if union.item() == 0 else (inter / union).item()

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            del self.model


# =====================================================================
# EvalMetricLogger — 增量 CSV + DDP gather + JSON
# =====================================================================

class EvalMetricLogger:
    """增量 CSV 落盘 + 每次 log 时 DDP all_gather 实时写全局数据。"""

    def __init__(self, out_dir: Path, keys: List[str], accelerator: Accelerator):
        self.out_dir = out_dir
        self.keys = keys
        self.fields = ["name", "view"] + keys
        self.rows: List[Dict[str, Any]] = []          # 本 rank 的行
        self._all_rows: List[Dict[str, Any]] = []     # rank 0 累积的全局行
        self._acc = accelerator
        self._is_main = accelerator.is_main_process
        self._ddp = accelerator.num_processes > 1

        # 仅 rank 0 打开增量 CSV
        self.csv_path = out_dir / "guidance_similarity.csv"
        self._f = None
        if self._is_main:
            self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields)
            self._w.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        """所有进程 gather 本行数据 → rank 0 实时追加 CSV。

        注意：DDP 下这是集合操作，所有 rank 必须同步调用相同次数，否则 deadlock。
        """
        self.rows.append(row)

        if self._ddp:
            import torch.distributed as dist
            buf: List[Any] = [None] * self._acc.num_processes
            dist.all_gather_object(buf, row)
            if self._is_main:
                for r in buf:
                    self._all_rows.append(r)
                    self._w.writerow(r)
                self._f.flush()
        else:
            self._all_rows.append(row)
            if self._is_main:
                self._w.writerow(row)
                self._f.flush()

    def finalize(self) -> Optional[Dict[str, float]]:
        """rank 0 重写完整 CSV（加 AVERAGE 行）+ JSON。无需再 gather。"""
        if self._f is not None and not self._f.closed:
            self._f.close()

        if not self._is_main or not self._all_rows:
            return None

        avg = {k: round(float(np.mean([r[k] for r in self._all_rows])), 4) for k in self.keys}

        # 重写完整 CSV（含 AVERAGE）
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in self._all_rows:
                w.writerow(r)
            w.writerow({"name": "AVERAGE", "view": "-", **avg})

        # JSON
        with open(self.out_dir / "guidance_similarity.json", "w", encoding="utf-8") as f:
            json.dump({"samples": self._all_rows, "average": avg}, f, indent=2, ensure_ascii=False)

        return avg


# =====================================================================
# 主流程
# =====================================================================

def main(argv) -> None:
    del argv
    cfg = _CONFIG.value
    cfg.eval_only = False

    # ---- 派生参数同步（支持命令行覆盖 target_prompt / true_cfg_scale_tgt）----
    cfg.train.guidance.source_prompt = cfg.train.guidance.target_prompt
    cfg.train.guidance.true_cfg_scale_src = -1 * cfg.train.guidance.true_cfg_scale_tgt

    # ---- 环境 ----
    setup_env_and_seed(cfg)
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    is_main = accelerator.is_main_process
    logger.info(f"[Rank {accelerator.process_index}/{accelerator.num_processes}] device={device}")

    # ---- 数据 + 系统 ----
    eval_loader = build_eval_dataloader(cfg, accelerator)
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    # 评估无需 DDP 包装，跳过 prepare_lora / prepare_models_and_optimizers

    run_root = Path(cfg.logdir) / (cfg.run_name or "run")
    ckpt_io = CheckpointIO(accelerator, run_root / "checkpoints")
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    logger.info(f"Loaded checkpoint: epoch={start_epoch}, step={global_step}")

    # ---- 输出目录 ----
    out_dir = run_root / "eval_metrics"
    images_dir = out_dir / "images"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(cfg.to_dict(), sort_keys=False))
    accelerator.wait_for_everyone()

    # ---- 指标记录器 ----
    metric_keys = [
        "clip_before", "clip_after", "clip_delta",
        "dino_before", "dino_after", "dino_delta",
        "sil_iou",
    ]
    el = EvalMetricLogger(out_dir, metric_keys, accelerator)

    # ---- 指标模型（延迟初始化）----
    clip_m = dino_m = sil = None

    # ---- 评估循环 ----
    pipe_models = system.pipeline.pipe.models
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models['slat_flow_model'],
        pipe_models['slat_decoder_mesh'],
        pipe_models['slat_decoder_gs'],
    ):
        loader = tqdm(eval_loader, desc="Eval") if is_main else eval_loader
        for batch_idx, batch in enumerate(loader):
            state = TrellisState()
            state.attach_batch(batch, pipeline=system.pipeline)

            # Step 1-2: Trellis 前向 + Guidance
            with torch.no_grad():
                render_out = trellis_forward(
                    system, state, cfg, device,
                    global_step=global_step, is_training=False,
                )
                comp_rgb = render_out["color"]  # (B,V,H,W,C)
                h, w = comp_rgb.shape[2], comp_rgb.shape[3]  # (H, W)
                cond_pils = [
                    im.resize((w, h), Image.LANCZOS)
                    for im in state.views_conditioned.image_pils
                ]
                gr = system.guidance.compute_guidance(
                    comp_rgb, cond_pils,
                    guidance_cfg=cfg.train.guidance,
                    rank=accelerator.process_index,
                )
            state.attach_guidance_result(gr)

            # 延迟初始化指标模型
            if clip_m is None:
                clip_m = CLIPMetric(weight=1.0, device=device)
                dino_m = DINOMetric(weight=1.0, device=device)
                sil = SilhouetteExtractor(device)

            # Step 3: 逐样本逐视角计算指标 + 保存图片
            B, V = comp_rgb.shape[:2]
            for b in range(B):
                name = os.path.splitext(os.path.basename(
                    state.views_conditioned.paths[b]
                ))[0]
                cond_pil = composite_alpha_to_white(cond_pils[b])

                for v in range(V):
                    bef = _to_bchw(state.views_generated.image_tensor[b, v], "hwc")  # (1,3,H,W)
                    aft = _to_bchw(state.views_edited.color_tensor[b, v], "chw")      # (1,3,H,W)

                    _save_images(images_dir, name, cond_pil, bef, aft, v)

                    cb = _similarity(clip_m, _to_pil(bef), cond_pil)
                    ca = _similarity(clip_m, _to_pil(aft), cond_pil)
                    db = _similarity(dino_m, _to_pil(bef), cond_pil)
                    da = _similarity(dino_m, _to_pil(aft), cond_pil)
                    si = sil.iou(bef, aft)

                    el.log({
                        "name": name, "view": v,
                        "clip_before": round(cb, 4), "clip_after": round(ca, 4),
                        "clip_delta":  round(ca - cb, 4),
                        "dino_before": round(db, 4), "dino_after": round(da, 4),
                        "dino_delta":  round(da - db, 4),
                        "sil_iou":     round(si, 4),
                    })

                    if is_main:
                        logger.info(
                            f"[{name} v{v}] "
                            f"CLIP {cb:.4f}→{ca:.4f} Δ{ca-cb:+.4f} | "
                            f"DINO {db:.4f}→{da:.4f} Δ{da-db:+.4f} | "
                            f"SilIoU {si:.4f}"
                        )

            # 释放本批次显存
            del state, render_out, comp_rgb, gr
            torch.cuda.empty_cache()

    # ---- 汇总（DDP gather → rank 0 写文件）----
    accelerator.wait_for_everyone()
    avg = el.finalize()

    if is_main and avg:
        logger.info("=" * 60)
        logger.info(f"CLIP:   {avg['clip_before']:.4f} → {avg['clip_after']:.4f} (Δ={avg['clip_delta']:+.4f})")
        logger.info(f"DINO:   {avg['dino_before']:.4f} → {avg['dino_after']:.4f} (Δ={avg['dino_delta']:+.4f})")
        logger.info(f"SilIoU: {avg['sil_iou']:.4f}")
        logger.info(f"CSV:  {el.csv_path}")
        logger.info(f"JSON: {el.out_dir / 'guidance_similarity.json'}")
        logger.info("=" * 60)

    # ---- 清理 ----
    for m in [clip_m, dino_m, sil]:
        if m is not None:
            m.cleanup()


if __name__ == "__main__":
    app.run(main)
