#!/usr/bin/env python3
"""
离线图像质量评估脚本。

对已渲染/生成的图片目录计算以下无参考图像质量指标（via pyiqa）：
- MANIQA↑：基于 ViT 的无参考图像质量评估
- MUSIQ↑：Multi-scale 无参考图像质量评估
- NIMA↑：Neural Image Assessment，预测美学评分分布

支持两种输入目录结构：
  1. eval_trellis.py 输出格式：
       images_dir/
         sample_name/
           condition.png
           v0_student.png, v0_teacher.png
           v1_student.png, v1_teacher.png
           ...
  2. 扁平目录：
       images_dir/
         *.png

用法：
    python scripts/eval/eval_image_quality.py \\
        --images_dir outputs/eval_teacher_student/step_xxx/images \\
        --save_json outputs/image_quality.json
"""

import argparse
import csv
import glob
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import pyiqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# 工具函数
# =====================================================================

def load_with_white_bg(path: str) -> Image.Image:
    """加载图片，RGBA 自动合成白底。"""
    im = Image.open(path)
    if im.mode == "RGBA":
        a = im.split()[-1]
        white = Image.new("RGB", im.size, (255, 255, 255))
        white.paste(im.convert("RGB"), mask=a)
        return white
    return im.convert("RGB")


def collect_images_from_trellis_dir(
    images_dir: str,
    role: str = "student",
) -> Dict[str, List[Image.Image]]:
    """从 eval_trellis.py 输出目录收集渲染图。

    返回: {sample_name: [v0_pil, v1_pil, ...]}
    """
    result: Dict[str, List[Image.Image]] = {}
    if not os.path.isdir(images_dir):
        return result

    for name in sorted(os.listdir(images_dir)):
        sample_dir = os.path.join(images_dir, name)
        if not os.path.isdir(sample_dir):
            continue
        views: List[Tuple[int, Image.Image]] = []
        for fn in sorted(os.listdir(sample_dir)):
            # 匹配 v0_student.png, v1_student.png, ...
            if fn.endswith(f"_{role}.png") and fn.startswith("v"):
                try:
                    v_idx = int(fn.split("_")[0][1:])
                except ValueError:
                    continue
                views.append((v_idx, load_with_white_bg(os.path.join(sample_dir, fn))))
        if views:
            views.sort(key=lambda x: x[0])
            result[name] = [v[1] for v in views]
    return result


def collect_images_flat(images_dir: str) -> List[Image.Image]:
    """从扁平目录收集所有 PNG/JPG 图片。"""
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    paths = set()
    for pat in patterns:
        for fp in glob.glob(os.path.join(images_dir, "**", pat), recursive=True):
            paths.add(fp)
    return [load_with_white_bg(p) for p in sorted(paths)]


# =====================================================================
# NR-IQA 指标（MANIQA / MUSIQ / NIMA）via pyiqa
# =====================================================================

NR_IQA_METRICS = ["maniqa", "musiq", "nima"]


@torch.no_grad()
def compute_nr_iqa(
    pil_images: List[Image.Image],
    device: torch.device,
    metrics: List[str] = NR_IQA_METRICS,
    batch_size: int = 16,
) -> Dict[str, List[float]]:
    """用 pyiqa 计算无参考图像质量指标。

    Args:
        pil_images: 待评估 PIL 图片列表
        device: 计算设备
        metrics: 要计算的指标名列表
        batch_size: 推理 batch size

    Returns:
        {metric_name: [score_per_image, ...]}
    """
    if len(pil_images) == 0:
        return {}

    pp = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 1]
    ])

    results: Dict[str, List[float]] = {}

    for name in metrics:
        try:
            model = pyiqa.create_metric(name, device=device)
        except Exception as e:
            logger.warning(f"pyiqa 加载 {name} 失败: {e}")
            continue
        logger.info(f"✅ pyiqa/{name} 已加载")

        scores_list: List[float] = []
        for i in range(0, len(pil_images), batch_size):
            batch = torch.stack(
                [pp(im) for im in pil_images[i:i + batch_size]],
            ).to(device)  # (B, 3, 224, 224)
            scores = model(batch).flatten()  # (B,)
            scores_list.extend(scores.cpu().tolist())

        results[name] = scores_list
        logger.info(
            f"  {name.upper()}: {np.mean(scores_list):.4f} ± {np.std(scores_list):.4f}"
        )

        del model
        torch.cuda.empty_cache()

    return results


# =====================================================================
# 主流程
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线图像质量评估（MANIQA / MUSIQ / NIMA）",
    )
    parser.add_argument(
        "--images_dir", type=str, required=True,
        help="渲染图目录（eval_trellis.py 输出的 images/ 目录，或扁平图片目录）",
    )
    parser.add_argument(
        "--role", type=str, default="student",
        choices=["student", "teacher"],
        help="eval_trellis.py 目录结构时，评估 student 还是 teacher（默认 student）",
    )
    parser.add_argument(
        "--save_json", type=str, default="image_quality_results.json",
        help="结果输出 JSON 路径",
    )
    parser.add_argument(
        "--save_csv", type=str, default=None,
        help="结果输出 CSV 路径（可选，默认与 JSON 同目录同名）",
    )
    parser.add_argument(
        "--nr_iqa_batch_size", type=int, default=16,
        help="NR-IQA 推理 batch size（默认 16）",
    )
    return parser.parse_args()


def _detect_trellis_format(images_dir: str) -> bool:
    """检测目录是否为 eval_trellis.py 输出格式（子目录下有 v*_student.png）。"""
    for name in os.listdir(images_dir):
        sub = os.path.join(images_dir, name)
        if os.path.isdir(sub):
            for fn in os.listdir(sub):
                if fn.startswith("v") and "_student.png" in fn:
                    return True
                if fn.startswith("v") and "_teacher.png" in fn:
                    return True
    return False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- 检测输入目录格式 ----
    is_trellis_fmt = _detect_trellis_format(args.images_dir)
    logger.info(
        f"目录格式: {'eval_trellis.py 输出' if is_trellis_fmt else '扁平目录'}"
    )

    # ---- 收集图片 ----
    if is_trellis_fmt:
        sample_images = collect_images_from_trellis_dir(
            args.images_dir, role=args.role,
        )
        all_gen_images = []
        for views in sample_images.values():
            all_gen_images.extend(views)
        logger.info(
            f"收集到 {len(sample_images)} 个样本，"
            f"共 {len(all_gen_images)} 张 {args.role} 渲染图"
        )
    else:
        all_gen_images = collect_images_flat(args.images_dir)
        sample_images = {"all": all_gen_images}
        logger.info(f"收集到 {len(all_gen_images)} 张图片")

    if len(all_gen_images) == 0:
        logger.error("未找到任何图片，退出")
        return

    # ==================================================================
    # 1. NR-IQA 指标（MANIQA / MUSIQ / NIMA）
    # ==================================================================
    logger.info("计算 NR-IQA 指标 ...")
    nr_iqa_results = compute_nr_iqa(
        all_gen_images, device, batch_size=args.nr_iqa_batch_size,
    )

    # ==================================================================
    # 2. 汇总 & 保存
    # ==================================================================
    summary: Dict[str, Optional[float]] = {}
    for metric_name, scores in nr_iqa_results.items():
        if scores:
            summary[f"{metric_name}_mean"] = round(float(np.mean(scores)), 4)
            summary[f"{metric_name}_std"] = round(float(np.std(scores)), 4)

    # 将 NR-IQA 全局分数按样本拆分（每样本 = 该样本所有视角的均值）
    per_sample_results: Dict[str, Dict] = {}
    for name, views in sample_images.items():
        per_sample_results[name] = {"num_views": len(views)}

    nr_iqa_per_sample: Dict[str, Dict[str, float]] = {}
    for metric_name, all_scores in nr_iqa_results.items():
        idx = 0
        for name, views in sample_images.items():
            n = len(views)
            sample_scores = all_scores[idx:idx + n]
            idx += n
            sample_mean = round(float(np.mean(sample_scores)), 4)
            nr_iqa_per_sample.setdefault(name, {})[metric_name] = sample_mean
            per_sample_results[name][f"{metric_name}_mean"] = sample_mean

    output = {
        "config": {
            "images_dir": args.images_dir,
            "role": args.role if is_trellis_fmt else "flat",
            "num_gen_images": len(all_gen_images),
            "num_samples": len(per_sample_results),
        },
        "summary": summary,
        "per_sample": per_sample_results,
    }

    # 保存 JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ JSON 已保存: {args.save_json}")

    # 保存 CSV（逐样本）
    csv_path = args.save_csv
    if csv_path is None:
        csv_path = os.path.splitext(args.save_json)[0] + ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    csv_fields = ["name", "num_views"]
    for m in nr_iqa_results:
        csv_fields.append(m)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for name, res in per_sample_results.items():
            row = {"name": name, "num_views": res["num_views"]}
            for m in nr_iqa_results:
                row[m] = nr_iqa_per_sample.get(name, {}).get(m, "")
            writer.writerow(row)
        # AVERAGE 行
        avg_row: Dict[str, object] = {"name": "AVERAGE", "num_views": "-"}
        for m in nr_iqa_results:
            key = f"{m}_mean"
            if key in summary:
                avg_row[m] = summary[key]
        writer.writerow(avg_row)
    logger.info(f"✅ CSV 已保存: {csv_path}")

    # 打印汇总
    logger.info("=" * 60)
    logger.info("汇总指标:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
