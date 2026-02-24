"""
Trellis Teacher/Student 对比评估脚本。

复用训练中 Strategy.teacher_context() / inference_context() 机制，
在每个 batch 内切换 pretrained (teacher) 和 finetuned (student) 模型，
渲染多视角图像后使用 CLIP / DINO 计算与输入条件图像的相似度。

数据流（对齐 trellis.py 训练主流程）：
    1. build_system(eval_only=True) → pipeline + renderer + strategy（不加载 guidance）
    2. prepare_lora + prepare_models_and_optimizers → 注册模型到 accelerator
    3. CheckpointIO.load() → 用 accelerator.load_state 恢复 finetuned 权重
    4. 每个 batch（在 inference_context 内）:
       a. student forward → finetuned 渲染
       b. teacher_context() forward → pretrained 渲染（共享 coords）
       c. CLIP / DINO similarity(渲染图, 输入条件图)
    5. 增量写 CSV + 最终 JSON 汇总

用法（单卡）：
    python scripts/eval/eval_trellis.py --config=configs/trellis_eval.py

用法（DDP 多卡）：
    accelerate launch scripts/eval/eval_trellis.py --config=configs/trellis_eval.py
"""

# =====================================================================
# 标准库导入
# =====================================================================
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# =====================================================================
# TRELLIS 参考实现路径设置（必须在 trellis 相关导入之前）
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# =====================================================================
# 第三方库导入
# =====================================================================
import numpy as np
import torch
import torchvision.transforms.functional as TF
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from absl import app
from ml_collections import config_flags
from accelerate import Accelerator

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.systems.trellis.system import (
    build_system, trellis_forward, _CONFIG,
)
from edit4shape.systems.base import (
    setup_env_and_seed, EvalModeGuard,
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

def _to_pil(t: torch.Tensor) -> Image.Image:
    """(H,W,C) 或 (C,H,W) float [0,1] → PIL RGB。"""
    if t.dim() == 4:
        t = t.squeeze(0)  # (1,C,H,W) → (C,H,W)
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):
        # (C,H,W) 格式
        t = t.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
    return Image.fromarray(
        (t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )


def _similarity(metric, rendered_pil: Image.Image, target_pil: Image.Image) -> float:
    """使用 metric.compute_from_pil 计算 similarity（= 1 - loss）。"""
    with torch.no_grad():
        loss = metric.compute_from_pil([rendered_pil], [target_pil])
    return 1.0 - loss.item()


def _save_images(
    images_dir: Path,
    name: str,
    cond_pil: Image.Image,
    stu_pil: Image.Image,
    tea_pil: Image.Image,
    v: int,
) -> None:
    """保存 condition / student / teacher / grid 图片。"""
    d = images_dir / name
    d.mkdir(parents=True, exist_ok=True)

    if v == 0:
        cond_pil.save(d / "condition.png")

    stu_pil.save(d / f"v{v}_student.png")
    tea_pil.save(d / f"v{v}_teacher.png")

    # 三图拼接 grid [condition | teacher | student]
    margin = 12
    h = stu_pil.height
    c = cond_pil.copy()
    if c.height != h:
        s = h / c.height
        c = c.resize((max(1, int(c.width * s)), h), Image.LANCZOS)
    imgs = [c, tea_pil, stu_pil]
    total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
    grid = Image.new("RGB", (total_w, h + margin * 2), (255, 255, 255))
    x = margin
    for im in imgs:
        grid.paste(im, (x, margin))
        x += im.width + margin
    grid.save(d / f"v{v}_grid.png")


# =====================================================================
# EvalMetricLogger — 增量 CSV + DDP gather + JSON
# =====================================================================

class EvalMetricLogger:
    """增量 CSV 落盘 + DDP gather + JSON 汇总。"""

    def __init__(self, out_dir: Path, keys: List[str], accelerator: Accelerator):
        self.out_dir = out_dir
        self.keys = keys
        self.fields = ["name", "view"] + keys
        self.rows: List[Dict[str, Any]] = []
        self._all_rows: List[Dict[str, Any]] = []
        self._acc = accelerator
        self._is_main = accelerator.is_main_process
        self._ddp = accelerator.num_processes > 1

        self.csv_path = out_dir / "teacher_student_similarity.csv"
        self._f = None
        if self._is_main:
            self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields)
            self._w.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        """所有进程 gather 本行数据 → rank 0 实时追加 CSV。"""
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
        """rank 0 重写完整 CSV（加 AVERAGE 行）+ JSON。"""
        if self._f is not None and not self._f.closed:
            self._f.close()

        if not self._is_main or not self._all_rows:
            return None

        avg = {
            k: round(float(np.mean([r[k] for r in self._all_rows])), 4)
            for k in self.keys
        }

        # 重写完整 CSV（含 AVERAGE）
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in self._all_rows:
                w.writerow(r)
            w.writerow({"name": "AVERAGE", "view": "-", **avg})

        # JSON
        json_path = self.out_dir / "teacher_student_similarity.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"samples": self._all_rows, "average": avg},
                f, indent=2, ensure_ascii=False,
            )

        return avg


# =====================================================================
# 主流程
# =====================================================================

def main(argv) -> None:
    del argv
    cfg = _CONFIG.value

    # ---- 强制 eval_only，跳过 guidance 加载 ----
    cfg.eval_only = True

    # ---- 环境 ----
    setup_env_and_seed(cfg)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    device = accelerator.device
    is_main = accelerator.is_main_process
    logger.info(
        f"[Rank {accelerator.process_index}/{accelerator.num_processes}] device={device}"
    )

    # ---- 数据 ----
    eval_loader = build_eval_dataloader(cfg, accelerator)

    # ---- 构建系统（对齐 trellis.py 训练主流程）----
    # build_system 内部已创建 strategy（含 teacher_context 能力）
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    # prepare_lora: 注入/加载 LoRA adapter（与训练一致）
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)

    # ---- 加载 finetuned checkpoint ----
    # eval-only 模式下不调用 accelerator.prepare()（会触发 DDP 包装导致 NCCL 错误），
    # 而是直接从 checkpoint 的 model.safetensors 加载权重到 student 模型。
    ckpt_path = cfg.get("checkpoint", "")
    if ckpt_path:
        from safetensors.torch import load_file
        safetensors_path = Path(ckpt_path) / "model.safetensors"
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path), device="cpu")
            system.strategy.student.load_state_dict(state_dict)
            logger.info(f"[Checkpoint] Student 权重已从 {safetensors_path} 加载")
        else:
            logger.error(f"[Checkpoint] 未找到 {safetensors_path}，student 使用 pretrained 权重")
    else:
        logger.warning("[Checkpoint] 未指定 checkpoint，student 使用 pretrained 权重（与 teacher 相同）")

    # ---- 参数差异检查：确认 student ≠ teacher ----
    if is_main and system.strategy.has_teacher:
        student_params = dict(system.strategy.student.named_parameters())
        teacher_model = system.strategy._teacher  # TrellisFullFinetuneStrategy 的教师模型
        n_diff, n_total, max_diff = 0, 0, 0.0
        for name, t_param in teacher_model.named_parameters():
            s_param = student_params.get(name)
            if s_param is not None:
                n_total += 1
                diff = (s_param.data.float() - t_param.data.float()).abs().max().item()
                if diff > 1e-8:
                    n_diff += 1
                max_diff = max(max_diff, diff)
        if n_diff == 0:
            logger.error(
                f"[ParamCheck] ⚠️ Student 与 Teacher 参数完全相同！"
                f"（{n_total} 层，max_diff={max_diff:.2e}）→ checkpoint 可能未正确加载"
            )
        else:
            logger.info(
                f"[ParamCheck] ✅ Student 与 Teacher 有 {n_diff}/{n_total} 层参数不同，"
                f"max_diff={max_diff:.2e}"
            )

    # ---- 输出目录 ----
    run_root = Path(cfg.logdir) / (cfg.run_name or "run")
    if ckpt_path:
        ckpt_tag = Path(str(ckpt_path).rstrip("/")).name
    else:
        ckpt_tag = "pretrained_baseline"
    out_dir = run_root / "eval_teacher_student" / ckpt_tag
    images_dir = out_dir / "images"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # ---- 指标模型（延迟初始化，节省显存）----
    clip_m: Optional[CLIPMetric] = None
    dino_m: Optional[DINOMetric] = None

    # ---- 指标记录器 ----
    metric_keys = [
        "clip_teacher", "clip_student", "clip_delta",
        "dino_teacher", "dino_student", "dino_delta",
    ]
    el = EvalMetricLogger(out_dir, metric_keys, accelerator)

    # ---- 评估循环 ----
    pipe_models = system.pipeline.pipe.models
    # ★ 推理时换回原始模型（无 DDP / autocast(bf16)），对齐 trellis.py evaluate()
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models["slat_flow_model"],
        pipe_models["slat_decoder_mesh"],
        pipe_models["slat_decoder_gs"],
    ):
        loader = tqdm(eval_loader, desc="Eval") if is_main else eval_loader
        for batch_idx, batch in enumerate(loader):

            with torch.no_grad():
                # === Student (finetuned) forward ===
                state_stu = TrellisState()
                state_stu.attach_batch(batch, pipeline=system.pipeline)
                render_stu = trellis_forward(
                    system, state_stu, cfg, device,
                    global_step=0, is_training=False,
                )
                comp_rgb_stu = render_stu["color"]  # (B,V,H,W,C)

                # === Teacher (pretrained) forward ===
                with system.strategy.teacher_context():
                    state_tea = TrellisState()
                    state_tea.attach_batch(batch, pipeline=system.pipeline)
                    # 复用 student 的 coords（dense_sampling 含随机性，
                    # 共享 coords 确保几何一致，只比较 rollout 差异）
                    state_tea.coords = state_stu.coords
                    render_tea = trellis_forward(
                        system, state_tea, cfg, device,
                        global_step=0, is_training=False,
                    )
                comp_rgb_tea = render_tea["color"]  # (B,V,H,W,C)

            # ---- 延迟初始化指标模型 ----
            if clip_m is None:
                clip_m = CLIPMetric(weight=1.0, device=device)
                dino_m = DINOMetric(weight=1.0, device=device)

            # ---- 逐样本逐视角计算指标 + 保存图片 ----
            B, V = comp_rgb_stu.shape[:2]
            h, w = comp_rgb_stu.shape[2], comp_rgb_stu.shape[3]

            for b in range(B):
                # 获取样本名
                name = os.path.splitext(
                    os.path.basename(state_stu.views_conditioned.paths[b])
                )[0]

                # 条件图（输入图像），合成白底 + resize 到渲染分辨率
                cond_pil = composite_alpha_to_white(
                    state_stu.views_conditioned.image_pils[b]
                ).resize((w, h), Image.LANCZOS)

                for v in range(V):
                    stu_pil = _to_pil(comp_rgb_stu[b, v])  # (H,W,C) → PIL
                    tea_pil = _to_pil(comp_rgb_tea[b, v])  # (H,W,C) → PIL

                    # 保存图片（所有进程都保存自己分到的样本）
                    _save_images(images_dir, name, cond_pil, stu_pil, tea_pil, v)

                    # 计算 CLIP / DINO similarity
                    cs = _similarity(clip_m, stu_pil, cond_pil)
                    ct = _similarity(clip_m, tea_pil, cond_pil)
                    ds = _similarity(dino_m, stu_pil, cond_pil)
                    dt = _similarity(dino_m, tea_pil, cond_pil)

                    el.log({
                        "name": name,
                        "view": v,
                        "clip_teacher": round(ct, 4),
                        "clip_student": round(cs, 4),
                        "clip_delta": round(cs - ct, 4),
                        "dino_teacher": round(dt, 4),
                        "dino_student": round(ds, 4),
                        "dino_delta": round(ds - dt, 4),
                    })

                    if is_main:
                        logger.info(
                            f"[{name} v{v}] "
                            f"CLIP tea={ct:.4f} stu={cs:.4f} Δ{cs - ct:+.4f} | "
                            f"DINO tea={dt:.4f} stu={ds:.4f} Δ{ds - dt:+.4f}"
                        )

            # 释放本批次显存
            del state_stu, state_tea, render_stu, render_tea
            del comp_rgb_stu, comp_rgb_tea
            torch.cuda.empty_cache()

    # ---- 汇总 ----
    accelerator.wait_for_everyone()
    avg = el.finalize()

    if is_main and avg:
        logger.info("=" * 60)
        logger.info(
            f"CLIP:  teacher={avg['clip_teacher']:.4f}  "
            f"student={avg['clip_student']:.4f}  "
            f"Δ={avg['clip_delta']:+.4f}"
        )
        logger.info(
            f"DINO:  teacher={avg['dino_teacher']:.4f}  "
            f"student={avg['dino_student']:.4f}  "
            f"Δ={avg['dino_delta']:+.4f}"
        )
        logger.info(f"CSV:  {el.csv_path}")
        logger.info(f"JSON: {el.out_dir / 'teacher_student_similarity.json'}")
        logger.info("=" * 60)

    # ---- 清理 ----
    for m in [clip_m, dino_m]:
        if m is not None:
            m.cleanup()


if __name__ == "__main__":
    app.run(main)
