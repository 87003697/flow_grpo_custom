"""
Dense FlowEdit 评估脚本。

对比两路渲染结果的 CLIP / DINO 相似度：
  Teacher (pretrained): Dense ODE → decode_to_coords → Sparse ODE
  Student (finetuned) : Dense ODE → Dense FlowEdit → decode_to_coords → Sparse ODE

与 eval_trellis.py 的差异：
  - student 路径在 Stage 1 额外执行 Dense FlowEdit
  - FlowEdit 参数通过 --config.rollout.flowedit.* 覆盖（或使用默认值）
  - 不依赖新 config 文件，直接向现有 config 注入 flowedit 子字段

用法（单卡）：
  python scripts/eval/eval_trellis_dense_flowedit.py \
    --config=config/trellis_stage1+2_contrastive.py \
    --config.checkpoint=/path/to/ckpt \
    --config.rollout.flowedit.n_max=9 \
    --config.rollout.flowedit.cfg_scale_tgt=3.0 \
    --config.rollout.flowedit.cfg_scale_src=-3.0

用法（DDP 多卡）：
  accelerate launch scripts/eval/eval_trellis_dense_flowedit.py \
    --config=config/trellis_stage1+2_contrastive.py ...
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

import numpy as np
import torch
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from absl import app
from ml_collections import config_flags
import ml_collections
from accelerate import Accelerator

from edit4shape.systems.trellis.system import (
    build_system, trellis_forward, _CONFIG,
)
from edit4shape.systems.base import setup_env_and_seed, EvalModeGuard
from edit4shape.datasets.trellis import (
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
    TrellisDataConfig,
    TrellisDataModule,
)
from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import rollout_sparse, rollout_dense
from edit4shape.generators.trellis.rollout.flowedit import rollout_dense_flowedit
from edit4shape.systems.trellis.forward import decode_and_render_gs, decode_and_render_mesh
from edit4shape.guidance import create_guidance
from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.metric.clip import CLIPMetric
from edit4shape.guidance.metric.dino import DINOMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# FlowEdit 默认参数（可通过 --config.rollout.flowedit.* 覆盖）
# =====================================================================

_FE_DEFAULTS = {
    "steps": 12,
    "n_max": 9,
    "cfg_scale_tgt": 3.0,
    "cfg_scale_src": -3.0,
}


def _ensure_flowedit_cfg(cfg: ml_collections.ConfigDict) -> None:
    """若 cfg.rollout.flowedit 不存在则注入默认值。

    ml_collections ConfigDict 在 absl 解析后被锁定，必须先 unlock 再添加新 key。
    """
    if not hasattr(cfg.rollout, "flowedit"):
        fe = ml_collections.ConfigDict({
            "steps": _FE_DEFAULTS["steps"],
            "n_max": _FE_DEFAULTS["n_max"],
            "cfg_scale_tgt": _FE_DEFAULTS["cfg_scale_tgt"],
            "cfg_scale_src": _FE_DEFAULTS["cfg_scale_src"],
        })
        with cfg.rollout.unlocked():
            cfg.rollout.flowedit = fe


# =====================================================================
# 工具函数（与 eval_trellis.py 完全一致）
# =====================================================================

def build_eval_dataloader(cfg, accelerator: Accelerator):
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
        image_dataset_dir=cfg.data.eval.dir,
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
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    return Image.fromarray(
        (t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )


def _similarity(metric, rendered_pil: Image.Image, target_pil: Image.Image) -> float:
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
    d = images_dir / name
    d.mkdir(parents=True, exist_ok=True)
    if v == 0:
        cond_pil.save(d / "condition.png")
    stu_pil.save(d / f"v{v}_student.png")
    tea_pil.save(d / f"v{v}_teacher.png")
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


def _load_existing_eval_rows(
    csv_path: Path,
    fields: List[str],
    expected_views: int,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """读取已有 CSV，返回已写入 rows 和 view 数完整的样本名。"""
    if not csv_path.exists():
        return [], set()

    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    views_by_name: Dict[str, Set[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            view = row.get("view", "")
            if not name or name == "AVERAGE" or view in ("", "-"):
                continue

            clean_row = {field: row.get(field, "") for field in fields}
            key = (name, str(view))
            rows_by_key[key] = clean_row
            views_by_name.setdefault(name, set()).add(str(view))

    completed = {
        name
        for name, views in views_by_name.items()
        if len(views) >= expected_views
    }
    return list(rows_by_key.values()), completed


# =====================================================================
# EvalMetricLogger（与 eval_trellis.py 完全一致）
# =====================================================================

class EvalMetricLogger:
    def __init__(
        self,
        out_dir: Path,
        keys: List[str],
        accelerator: Accelerator,
        existing_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        self.out_dir = out_dir
        self.keys = keys
        self.fields = ["name", "view"] + keys
        self.rows: List[Dict[str, Any]] = []
        self._all_rows: List[Dict[str, Any]] = list(existing_rows or [])
        self._seen_keys = {
            (str(r.get("name", "")), str(r.get("view", "")))
            for r in self._all_rows
        }
        self._acc = accelerator
        self._is_main = accelerator.is_main_process
        self._ddp = accelerator.num_processes > 1
        self.csv_path = out_dir / "teacher_student_similarity.csv"
        self._f = None
        if self._is_main:
            mode = "a" if self._all_rows else "w"
            self._f = open(self.csv_path, mode, newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields)
            if not self._all_rows:
                self._w.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        self.log_many([row])

    def log_many(self, rows: List[Dict[str, Any]]) -> None:
        self.rows.extend(rows)
        if self._ddp:
            import torch.distributed as dist
            buf: List[Any] = [None] * self._acc.num_processes
            dist.all_gather_object(buf, rows)
            if self._is_main:
                for rank_rows in buf:
                    for r in rank_rows:
                        self._write_row(r)
                self._f.flush()
        else:
            if self._is_main:
                for row in rows:
                    self._write_row(row)
                self._f.flush()

    def _write_row(self, row: Dict[str, Any]) -> None:
        key = (str(row.get("name", "")), str(row.get("view", "")))
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        self._all_rows.append(row)
        if self._is_main:
            self._w.writerow(row)

    def finalize(self) -> Optional[Dict[str, float]]:
        if self._f is not None and not self._f.closed:
            self._f.close()
        if not self._is_main or not self._all_rows:
            return None
        avg = {
            k: round(float(np.mean([float(r[k]) for r in self._all_rows])), 4)
            for k in self.keys
        }
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in self._all_rows:
                w.writerow(r)
            w.writerow({"name": "AVERAGE", "view": "-", **avg})
        json_path = self.out_dir / "teacher_student_similarity.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": self._all_rows, "average": avg}, f, indent=2, ensure_ascii=False)
        return avg


# =====================================================================
# Student 前向：Dense ODE + Dense FlowEdit + Sparse ODE
# =====================================================================

def trellis_forward_dense_flowedit(
    system,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Student 专用前向：Dense ODE → Dense FlowEdit → decode_to_coords → Sparse ODE → Render

    FlowEdit 参数从 cfg.rollout.flowedit 读取。
    """
    pipeline = system.pipeline
    _seed = int(cfg.seed)

    # ---- Stage 1: Dense ODE + FlowEdit ----
    # rollout_dense_flowedit 内部先跑 ODE（CPU gen），再跑 FlowEdit（CUDA gen，种子+1）
    gen_dense = torch.Generator(device="cpu").manual_seed(_seed)
    with torch.no_grad():
        rollout_dense_flowedit(state, cfg, system, device, generator=gen_dense)
    # state.stage1.z0 = FlowEdit 编辑后的 dense latent

    batch_size = state.stage1.z0.shape[0]
    state.coords = pipeline.dense.decode_to_coords(state.stage1.z0, batch_size=batch_size)

    # ---- Stage 2: Sparse ODE ----
    gen_sparse = torch.Generator(device=device).manual_seed(_seed)
    with torch.no_grad():
        rollout_sparse(state, cfg, system, device, generator=gen_sparse, is_training=False)
    latents = state.stage2.z0

    torch.cuda.empty_cache()

    # ---- Decode & Render ----
    renderer_type = cfg.renderer.type
    renderer = system.renderers[renderer_type]

    if renderer_type == "gs":
        render_out = decode_and_render_gs(latents, state.cameras, pipeline, renderer, device)
    else:
        render_out = decode_and_render_mesh(latents, state.cameras, pipeline, renderer, device)
        render_out["color"] = render_out["normal"]

    state.views_generated.image_tensor = render_out["color"]
    return render_out


# =====================================================================
# DDP barrier
# =====================================================================

def _dist_barrier(local_rank: int) -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        t = torch.zeros(1, device=f"cuda:{local_rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


# =====================================================================
# 主流程
# =====================================================================

def main(argv) -> None:
    del argv
    cfg = _CONFIG.value
    cfg.eval_only = True

    # 注入 FlowEdit 默认配置（若 config 文件未定义）
    _ensure_flowedit_cfg(cfg)

    # Dense FlowEdit eval 固定使用 360 度 8 视角，避免依赖共享 config 的默认 eval 视角数。
    cfg.data.eval.n_view = 8
    cfg.data.eval.yaw_range = [0.0, 360.0]
    cfg.data.eval.pitch_range = [0.0, 0.0]

    setup_env_and_seed(cfg)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, kwargs_handlers=[])
    local_rank = accelerator.local_process_index
    torch.cuda.set_device(local_rank)
    device = accelerator.device
    is_main = accelerator.is_main_process

    fe_cfg = cfg.rollout.flowedit
    logger.info(
        f"[Rank {accelerator.process_index}] device={device} | "
        f"FlowEdit steps={fe_cfg.steps} n_max={fe_cfg.n_max} "
        f"cfg_tgt={fe_cfg.cfg_scale_tgt} cfg_src={fe_cfg.cfg_scale_src}"
    )
    logger.info(
        f"[Rank {accelerator.process_index}] Eval camera | "
        f"n_view={cfg.data.eval.n_view} "
        f"yaw_range={list(cfg.data.eval.yaw_range)} "
        f"pitch_range={list(cfg.data.eval.pitch_range)}"
    )

    # ---- 数据 ----
    eval_loader = build_eval_dataloader(cfg, accelerator)

    # ---- 构建系统 ----
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)

    # ---- 加载 finetuned checkpoint ----
    ckpt_path = cfg.get("checkpoint", "")
    has_dense_ckpt = False
    if ckpt_path:
        from safetensors.torch import load_file
        root = Path(ckpt_path)

        # 支持三种 checkpoint 格式：
        #   A) 扁平双阶段：root/model.safetensors (sparse) + root/model_1.safetensors (dense)
        #   B) 子目录双阶段：root/model_0/model.safetensors (sparse) + root/model_1/model.safetensors (dense)
        #   C) 扁平单阶段：root/model.safetensors (sparse only)
        subdir_dual = (root / "model_0").is_dir()
        flat_dense = (root / "model_1.safetensors").exists()

        if subdir_dual:
            sparse_path = root / "model_0" / "model.safetensors"
            dense_path: Optional[Path] = root / "model_1" / "model.safetensors"
        elif flat_dense:
            sparse_path = root / "model.safetensors"
            dense_path = root / "model_1.safetensors"
        else:
            sparse_path = root / "model.safetensors"
            dense_path = None

        # 加载 sparse student
        if sparse_path.exists():
            state_dict = load_file(str(sparse_path), device="cpu")
            system.strategy.student.load_state_dict(state_dict)
            logger.info(f"[Checkpoint] Sparse student 权重已从 {sparse_path} 加载")
        else:
            logger.error(f"[Checkpoint] 未找到 {sparse_path}")

        # 加载 dense student（若存在）
        if dense_path is not None and dense_path.exists():
            dense_state = load_file(str(dense_path), device="cpu")
            ss_model = system.pipeline.pipe.models["sparse_structure_flow_model"]
            if hasattr(ss_model, "module"):
                ss_model = ss_model.module
            ss_model.load_state_dict(dense_state)
            has_dense_ckpt = True
            logger.info(f"[Checkpoint] Dense student 权重已从 {dense_path} 加载")
        else:
            logger.warning("[Checkpoint] 未找到 dense 权重，dense student 使用 pretrained 权重")
    else:
        logger.warning("[Checkpoint] 未指定 checkpoint，student 使用 pretrained 权重（与 teacher 相同）")

    # ---- 参数差异检查：确认 student ≠ teacher ----
    def _param_diff_check(label, student_model, teacher_model):
        student_params = dict(student_model.named_parameters())
        n_diff, n_total, max_diff = 0, 0, 0.0
        for pname, t_param in teacher_model.named_parameters():
            s_param = student_params.get(pname)
            if s_param is not None:
                n_total += 1
                diff = (s_param.data.float() - t_param.data.float()).abs().max().item()
                if diff > 1e-8:
                    n_diff += 1
                max_diff = max(max_diff, diff)
        if n_diff == 0:
            logger.error(
                f"[ParamCheck-{label}] Student 与 Teacher 参数完全相同！"
                f"（{n_total} 层，max_diff={max_diff:.2e}）→ checkpoint 可能未正确加载"
            )
        else:
            logger.info(
                f"[ParamCheck-{label}] Student 与 Teacher 有 {n_diff}/{n_total} 层参数不同，"
                f"max_diff={max_diff:.2e}"
            )

    if is_main and system.strategy.has_teacher:
        _param_diff_check("Sparse", system.strategy.student, system.strategy._sparse_teacher)
        if has_dense_ckpt and hasattr(system.strategy, "_dense_teacher"):
            _param_diff_check(
                "Dense",
                system.pipeline.pipe.models["sparse_structure_flow_model"],
                system.strategy._dense_teacher,
            )

    # ---- 输出目录 ----
    run_root = Path(cfg.logdir) / (cfg.run_name or "run")
    ckpt_tag = Path(str(ckpt_path).rstrip("/")).name if ckpt_path else "pretrained_baseline"
    # 区分本脚本与标准 eval
    ckpt_tag = ckpt_tag + "_dense_flowedit"
    out_dir = run_root / "eval_teacher_student" / ckpt_tag
    images_dir = out_dir / "images"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        # 记录 FlowEdit 运行参数
        with open(out_dir / "flowedit_params.json", "w") as f:
            json.dump({
                "steps": int(fe_cfg.steps),
                "n_max": int(fe_cfg.n_max),
                "cfg_scale_tgt": float(fe_cfg.cfg_scale_tgt),
                "cfg_scale_src": float(fe_cfg.cfg_scale_src),
                "checkpoint": ckpt_path,
                "has_dense_ckpt": has_dense_ckpt,
            }, f, indent=2)
    _dist_barrier(local_rank)

    # ---- 指标模型（延迟初始化）----
    clip_m: Optional[CLIPMetric] = None
    dino_m: Optional[DINOMetric] = None

    # ---- 指标记录器 ----
    metric_keys = [
        "clip_teacher", "clip_student", "clip_delta",
        "dino_teacher", "dino_student", "dino_delta",
    ]
    existing_rows, completed_names = _load_existing_eval_rows(
        out_dir / "teacher_student_similarity.csv",
        ["name", "view"] + metric_keys,
        int(cfg.data.eval.n_view),
    )
    if is_main and existing_rows:
        logger.info(
            f"[Resume] 发现已有 CSV rows={len(existing_rows)}，"
            f"完整样本={len(completed_names)}，将跳过完整样本并继续补跑。"
        )
    el = EvalMetricLogger(out_dir, metric_keys, accelerator, existing_rows=existing_rows)

    # ---- 评估循环 ----
    pipe_models = system.pipeline.pipe.models
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models["slat_flow_model"],
        pipe_models["slat_decoder_mesh"],
        pipe_models["slat_decoder_gs"],
    ):
        loader = tqdm(eval_loader, desc="Eval") if is_main else eval_loader
        for batch_idx, batch in enumerate(loader):
            batch_paths = batch.get("paths", [])
            if isinstance(batch_paths, str):
                batch_paths = [batch_paths]
            batch_names = [
                os.path.splitext(os.path.basename(str(p)))[0]
                for p in batch_paths
            ]
            if batch_names and all(name in completed_names for name in batch_names):
                el.log_many([])
                continue

            with torch.no_grad():
                # === Student: Dense ODE + FlowEdit → Sparse ODE ===
                state_stu = TrellisState()
                state_stu.attach_batch(batch, pipeline=system.pipeline)
                render_stu = trellis_forward_dense_flowedit(
                    system, state_stu, cfg, device,
                )
                comp_rgb_stu = render_stu["color"]  # (B,V,H,W,C)

                # === Teacher: 标准 Dense ODE → Sparse ODE（pretrained 权重）===
                with system.strategy.sparse_teacher_context(), \
                     system.strategy.dense_teacher_context():
                    state_tea = TrellisState()
                    state_tea.attach_batch(batch, pipeline=system.pipeline)
                    # Teacher 始终用自己的 dense model 重新生成 coords（不共享 student coords）
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
            batch_rows: List[Dict[str, Any]] = []

            for b in range(B):
                name = os.path.splitext(
                    os.path.basename(state_stu.views_conditioned.paths[b])
                )[0]
                cond_pil = composite_alpha_to_white(
                    state_stu.views_conditioned.image_pils[b]
                ).resize((w, h), Image.LANCZOS)

                for v in range(V):
                    stu_pil = _to_pil(comp_rgb_stu[b, v])
                    tea_pil = _to_pil(comp_rgb_tea[b, v])

                    _save_images(images_dir, name, cond_pil, stu_pil, tea_pil, v)

                    cs = _similarity(clip_m, stu_pil, cond_pil)
                    ct = _similarity(clip_m, tea_pil, cond_pil)
                    ds = _similarity(dino_m, stu_pil, cond_pil)
                    dt = _similarity(dino_m, tea_pil, cond_pil)

                    batch_rows.append({
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

            el.log_many(batch_rows)

            del state_stu, state_tea, render_stu, render_tea
            del comp_rgb_stu, comp_rgb_tea
            torch.cuda.empty_cache()

    # ---- 汇总 ----
    _dist_barrier(local_rank)
    avg = el.finalize()

    if is_main and avg:
        logger.info("=" * 60)
        logger.info(f"[Dense FlowEdit Eval] steps={fe_cfg.steps} n_max={fe_cfg.n_max} "
                    f"cfg_tgt={fe_cfg.cfg_scale_tgt} cfg_src={fe_cfg.cfg_scale_src}")
        logger.info(
            f"CLIP:  teacher={avg['clip_teacher']:.4f}  "
            f"student={avg['clip_student']:.4f}  Δ={avg['clip_delta']:+.4f}"
        )
        logger.info(
            f"DINO:  teacher={avg['dino_teacher']:.4f}  "
            f"student={avg['dino_student']:.4f}  Δ={avg['dino_delta']:+.4f}"
        )
        logger.info(f"CSV:  {el.csv_path}")
        logger.info(f"JSON: {out_dir / 'teacher_student_similarity.json'}")
        logger.info("=" * 60)

    for m in [clip_m, dino_m]:
        if m is not None:
            m.cleanup()

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
