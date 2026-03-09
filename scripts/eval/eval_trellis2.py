"""
Trellis2 Teacher/Student 对比评估脚本（Shape+Tex 全参微调）。

在每个 batch 内切换 pretrained (teacher) 和 finetuned (student) 模型，
执行 Shape→Tex 双阶段 forward 后渲染 PBR 图像，
使用 CLIP / DINO 计算与输入条件图像的相似度。

数据流（对齐 shape_tex.py 训练主流程）：
    1. build_system(eval_only=True) → pipeline + renderer（不加载 guidance）
    2. 手动构建 Trellis2FullFinetuneStrategy → 加载冻结 teacher
    3. 从 accelerator checkpoint 加载 finetuned student 权重
    4. 每个 batch（no_grad）:
       a. Student: shape_forward → tex_forward → PBR 渲染
       b. Teacher: teacher_context(shape) + teacher_context(tex) → PBR 渲染（共享 coords）
       c. CLIP / DINO similarity(PBR 渲染图, 条件图)
    5. 增量写 CSV + 最终 JSON 汇总

用法（单卡）：
    python scripts/eval/eval_trellis2.py --config=config/trellis2_shape_tex_distillation.py

用法（DDP 多卡）：
    accelerate launch scripts/eval/eval_trellis2.py --config=config/trellis2_shape_tex_distillation.py
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
# TRELLIS.2 参考实现路径设置（必须在 trellis2 相关导入之前）
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

# =====================================================================
# 第三方库导入
# =====================================================================
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from absl import app
from ml_collections import config_flags
from accelerate import Accelerator

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.systems.trellis2.system import (
    build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import (
    trellis2_shape_forward,
    trellis2_tex_forward,
)
from edit4shape.systems.base import (
    setup_env_and_seed, EvalModeGuard,
)
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.training_adpter import (
    Trellis2FullFinetuneStrategy,
    get_stage_config,
)
from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.metric.clip import CLIPMetric
from edit4shape.guidance.metric.dino import DINOMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

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
# Student 权重加载
# =====================================================================

def _detect_stage_by_arch(
    state_dict: dict,
    shape_config,
    tex_config,
    pipeline,
) -> Optional[str]:
    """
    通过 input_layer.weight 的形状自动判断 state_dict 属于 shape 还是 tex。

    - shape DiT: input_layer.weight = (hidden, 32)  — sparse latent dim=32
    - tex DiT:   input_layer.weight = (hidden, 64)  — sparse latent dim=64

    Returns:
        "shape" | "tex" | None（无法判断）
    """
    il = state_dict.get("input_layer.weight")
    if il is None:
        return None

    for stage_name, stage_cfg in [("shape", shape_config), ("tex", tex_config)]:
        ref_model = pipeline.get_flow_model(stage_cfg.model_stage, stage_cfg.flow_resolution)
        ref_il = ref_model.input_layer.weight
        if il.shape == ref_il.shape:
            return stage_name
    return None


def _load_student_weights(
    pipeline,
    ckpt_path: str,
    shape_config,
    tex_config,
) -> bool:
    """
    从 accelerator save_state 格式的 checkpoint 加载 student 权重。

    支持三种 checkpoint 布局：
        1. 多模型格式（shape_tex 联合训练）：
            checkpoint_dir/
              model_0/model.safetensors   ← shape flow model
              model_1/model.safetensors   ← tex flow model
              meta.json

        2. 单模型格式（仅训练 shape 或 tex）：
            checkpoint_dir/
              model.safetensors           ← 自动检测 shape/tex
              meta.json

        3. 导出格式：
            checkpoint_dir/
              {stage}_flow_model_{resolution}.pt
              meta.json

    Args:
        pipeline: Trellis2RefAdapter
        ckpt_path: checkpoint 目录路径
        shape_config: shape 阶段的 StageConfig
        tex_config: tex 阶段的 StageConfig

    Returns:
        bool: 是否成功加载
    """
    from safetensors.torch import load_file

    root = Path(ckpt_path)

    # 自动查找最新 checkpoint（如果给的是父目录）
    meta_path = root / "meta.json"
    if root.is_dir() and not meta_path.exists():
        candidates = sorted(
            [d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists()],
            key=lambda d: json.load((d / "meta.json").open("r", encoding="utf-8")).get("global_step", 0),
        )
        if candidates:
            root = candidates[-1]
            logger.info(f"[Checkpoint] 自动选择最新检查点: {root}")
        else:
            logger.error(f"[Checkpoint] 目录 {ckpt_path} 下未找到任何有效检查点")
            return False

    loaded_any = False

    # ---- 格式 1: 多模型子目录 (model_0/, model_1/) ----
    if (root / "model_0").is_dir():
        stage_configs = [("shape", shape_config, 0), ("tex", tex_config, 1)]
        for stage_name, stage_cfg, model_idx in stage_configs:
            safetensors_path = root / f"model_{model_idx}" / "model.safetensors"
            bin_path = root / f"model_{model_idx}" / "pytorch_model.bin"

            if safetensors_path.exists():
                state_dict = load_file(str(safetensors_path), device="cpu")
            elif bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            else:
                logger.warning(f"[Checkpoint] {stage_name} 权重文件不存在: model_{model_idx}/")
                continue

            model = pipeline.get_flow_model(stage_cfg.model_stage, stage_cfg.flow_resolution)
            model.load_state_dict(state_dict)
            loaded_any = True
            logger.info(f"[Checkpoint] {stage_name} student 权重已加载（{len(state_dict)} 个张量）")
        return loaded_any

    # ---- 格式 2: Accelerate 扁平编号 (model.safetensors + model_1.safetensors) ----
    #   Accelerate 的命名规则：第一个 prepare 的模型存为 model.safetensors,
    #   第二个存为 model_1.safetensors。prepare 顺序由 prepare_optimizers 决定
    #   （遍历 stages dict 顺序：shape → tex）。
    flat_first = root / "model.safetensors"
    flat_second = root / "model_1.safetensors"
    if flat_first.exists():
        # 收集所有找到的权重文件
        weight_files = [flat_first]
        if flat_second.exists():
            weight_files.append(flat_second)

        for wf in weight_files:
            state_dict = load_file(str(wf), device="cpu")

            # 自动检测是 shape 还是 tex（通过 input_layer.weight 形状）
            detected = _detect_stage_by_arch(state_dict, shape_config, tex_config, pipeline)
            if detected is None:
                logger.error(f"[Checkpoint] 无法自动判断 {wf.name} 属于 shape 还是 tex")
                continue

            stage_cfg = shape_config if detected == "shape" else tex_config
            model = pipeline.get_flow_model(stage_cfg.model_stage, stage_cfg.flow_resolution)
            model.load_state_dict(state_dict)
            loaded_any = True
            logger.info(
                f"[Checkpoint] {wf.name} → 自动检测为 {detected} 模型，"
                f"已加载 student 权重（{len(state_dict)} 个张量）"
            )
        return loaded_any

    # ---- 格式 3: 导出格式 ({stage}_flow_model_{resolution}.pt) ----
    for stage_name, stage_cfg in [("shape", shape_config), ("tex", tex_config)]:
        export_path = root / f"{stage_name}_flow_model_{stage_cfg.flow_resolution}.pt"
        if export_path.exists():
            state_dict = torch.load(str(export_path), map_location="cpu", weights_only=True)
            model = pipeline.get_flow_model(stage_cfg.model_stage, stage_cfg.flow_resolution)
            model.load_state_dict(state_dict)
            loaded_any = True
            logger.info(f"[Checkpoint] {stage_name} student 权重已加载（导出格式，{len(state_dict)} 个张量）")

    if not loaded_any:
        logger.error(f"[Checkpoint] 目录 {root} 下未找到任何可识别的权重文件")

    return loaded_any


# =====================================================================
# 参数差异检查
# =====================================================================

def _check_param_diff(
    pipeline,
    strategy: Trellis2FullFinetuneStrategy,
    shape_config,
    tex_config,
) -> None:
    """逐层检查 student 与 teacher 参数差异（shape + tex）。"""
    for stage_name, stage_cfg in [("shape", shape_config), ("tex", tex_config)]:
        student = pipeline.get_flow_model(stage_cfg.model_stage, stage_cfg.flow_resolution)
        teacher = strategy._teacher_models.get((stage_cfg.model_stage, stage_cfg.flow_resolution))
        if teacher is None:
            logger.warning(f"[ParamCheck] {stage_name} 无 teacher 模型，跳过检查")
            continue

        student_params = dict(student.named_parameters())
        n_diff, n_total, max_diff = 0, 0, 0.0
        for name, t_param in teacher.named_parameters():
            s_param = student_params.get(name)
            if s_param is not None:
                n_total += 1
                diff = (s_param.data.float() - t_param.data.float()).abs().max().item()
                if diff > 1e-8:
                    n_diff += 1
                max_diff = max(max_diff, diff)

        if n_diff == 0:
            logger.error(
                f"[ParamCheck] ⚠️ {stage_name}: Student 与 Teacher 参数完全相同！"
                f"（{n_total} 层，max_diff={max_diff:.2e}）→ checkpoint 可能未正确加载"
            )
        else:
            logger.info(
                f"[ParamCheck] ✅ {stage_name}: Student 与 Teacher 有 {n_diff}/{n_total} 层参数不同，"
                f"max_diff={max_diff:.2e}"
            )


# =====================================================================
# 主流程
# =====================================================================

_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")


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
    _, eval_loader = build_dataloaders(cfg, accelerator)

    # ---- 构建系统（eval_only=True → pipeline + renderer, strategy=None）----
    system = build_system(
        cfg, accelerator,
        guidance_factory=lambda *a, **kw: None,  # eval 不需要 guidance
        mode="shape_tex",
    )

    # ---- 获取阶段配置 ----
    pipeline_type = cfg.pipeline_type
    shape_config = get_stage_config(pipeline_type, "shape")
    tex_config = get_stage_config(pipeline_type, "tex")

    # ---- 手动构建 strategy（用于 teacher_context 切换）----
    strategy = Trellis2FullFinetuneStrategy(
        pipeline=system.pipeline,
        train_device=device,
        teacher_device=device,  # eval 时 teacher 和 student 共用设备
        pretrained_path=cfg.pretrained.model,
        pipeline_type=pipeline_type,
        stages=["shape", "tex"],
    )
    strategy.setup()  # 加载冻结 teacher 模型
    system.strategy = strategy
    logger.info("[Strategy] Trellis2FullFinetuneStrategy 已创建并加载 teacher 模型")

    # ---- 加载 finetuned checkpoint ----
    ckpt_path = cfg.get("checkpoint", "")
    if ckpt_path:
        success = _load_student_weights(
            system.pipeline, ckpt_path, shape_config, tex_config,
        )
        if not success:
            logger.error("[Checkpoint] Student 权重加载失败，使用 pretrained 权重（与 teacher 相同）")
    else:
        logger.warning(
            "[Checkpoint] 未指定 checkpoint，student 使用 pretrained 权重（与 teacher 相同）"
        )

    # ---- 参数差异检查 ----
    if is_main:
        _check_param_diff(system.pipeline, strategy, shape_config, tex_config)

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

    # ---- 条件编码分辨率（tex 阶段决定）----
    cond_resolution = tex_config.cond_resolution

    # ---- 收集需要 eval mode 的模型 ----
    pipe_models = system.pipeline.pipe.models
    models_to_eval = []
    for key in pipe_models:
        model = pipe_models[key]
        if hasattr(model, "eval"):
            models_to_eval.append(model)
    # 也包含 teacher 模型
    for teacher_model in strategy._teacher_models.values():
        if hasattr(teacher_model, "eval"):
            models_to_eval.append(teacher_model)

    # ---- 评估循环 ----
    with EvalModeGuard(*models_to_eval):
        loader = tqdm(eval_loader, desc="Eval") if is_main else eval_loader
        for batch_idx, batch in enumerate(loader):

            with torch.no_grad():
                # ★ 保存 RNG 状态：trellis2_shape_forward 内部会调用 dense_sampling_no_grad，
                #   其中 dense_sampling 含随机性。通过保存/恢复 RNG 状态，
                #   确保 student 和 teacher 获得完全相同的 coords 和初始 latent noise，
                #   从而只比较 flow model 的 rollout 差异。
                cpu_rng_state = torch.random.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

                # === Student (finetuned) forward: Shape → Tex → PBR ===
                state_stu = Trellis2State()
                state_stu.attach_batch(
                    batch, pipeline=system.pipeline, resolution=cond_resolution
                )
                trellis2_shape_forward(
                    system, state_stu, global_step=0, is_training=False,
                )
                tex_out_stu = trellis2_tex_forward(
                    system, state_stu, global_step=0, is_training=False,
                )
                comp_rgb_stu = tex_out_stu["color"]  # (B, V, H, W, 3)

                # ★ 恢复 RNG 状态，使 teacher 获得与 student 完全相同的 coords/noise
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, device)

                # === Teacher (pretrained) forward: Shape → Tex → PBR ===
                state_tea = Trellis2State()
                state_tea.attach_batch(
                    batch, pipeline=system.pipeline, resolution=cond_resolution
                )

                with strategy.teacher_context(
                    shape_config.model_stage, shape_config.flow_resolution
                ):
                    trellis2_shape_forward(
                        system, state_tea, global_step=0, is_training=False,
                    )

                with strategy.teacher_context(
                    tex_config.model_stage, tex_config.flow_resolution
                ):
                    tex_out_tea = trellis2_tex_forward(
                        system, state_tea, global_step=0, is_training=False,
                    )
                comp_rgb_tea = tex_out_tea["color"]  # (B, V, H, W, 3)

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
            del state_stu, state_tea, tex_out_stu, tex_out_tea
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
