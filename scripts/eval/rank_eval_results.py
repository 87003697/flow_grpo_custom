#!/usr/bin/env python3
"""
对 eval_trellis.py 产出的 teacher_student_similarity.csv 按 combined_delta 排序，
并按排名收集 v3_grid.png 到 ranked_grids/ 目录，方便挑选论文定性对比样本。

用法:
    python scripts/eval/rank_eval_results.py <eval_dir>

例:
    python scripts/eval/rank_eval_results.py \
        logs/eval_xxx/eval_teacher_student
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="按 clip_delta + dino_delta 降序排列评估结果，收集 v3 grid 图片"
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="eval_teacher_student 目录（包含 teacher_student_similarity.csv 和 images/）",
    )
    parser.add_argument(
        "--view", type=int, default=3,
        help="要收集的 grid 视角编号（默认 3）",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    csv_path = eval_dir / "teacher_student_similarity.csv"
    images_dir = eval_dir / "images"
    view_idx = args.view

    if not csv_path.exists():
        print(f"[ERROR] CSV 不存在: {csv_path}")
        sys.exit(1)

    # ---- 1. 读取 CSV，按样本名分组 ----
    metric_keys = [
        "clip_teacher", "clip_student", "clip_delta",
        "dino_teacher", "dino_student", "dino_delta",
    ]
    samples: dict[str, list[dict[str, float]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if name == "AVERAGE":
                continue
            samples[name].append({k: float(row[k]) for k in metric_keys})

    if not samples:
        print("[ERROR] CSV 中没有有效数据行")
        sys.exit(1)

    # ---- 2. 逐样本跨 view 取平均 ----
    ranked = []
    for name, rows in samples.items():
        n = len(rows)
        avg = {k: round(sum(r[k] for r in rows) / n, 4) for k in metric_keys}
        avg["combined_delta"] = round(avg["clip_delta"] + avg["dino_delta"], 4)
        avg["name"] = name
        avg["n_views"] = n
        ranked.append(avg)

    # ---- 3. 按 combined_delta 降序排序 ----
    ranked.sort(key=lambda x: x["combined_delta"], reverse=True)

    # ---- 4. 输出排序后的 CSV ----
    out_csv = eval_dir / "teacher_student_similarity_ranked.csv"
    out_fields = [
        "rank", "name", "n_views",
        "clip_teacher", "clip_student", "clip_delta",
        "dino_teacher", "dino_student", "dino_delta",
        "combined_delta",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for i, item in enumerate(ranked, 1):
            item["rank"] = i
            w.writerow({k: item[k] for k in out_fields})

    print(f"[OK] 排序 CSV ({len(ranked)} 个样本): {out_csv}")

    # ---- 5. 收集 v{view_idx}_grid.png 到 ranked_grids/ ----
    grids_dir = eval_dir / f"ranked_grids_v{view_idx}"
    grids_dir.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for i, item in enumerate(ranked, 1):
        name = item["name"]
        src = images_dir / name / f"v{view_idx}_grid.png"
        if src.exists():
            # 文件名: 排名_样本名_combined_delta.png
            delta_str = f"{item['combined_delta']:+.4f}".replace("+", "p").replace("-", "n")
            dst = grids_dir / f"{i:04d}_{name}_{delta_str}.png"
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"[OK] Grid 图片收集到: {grids_dir}")
    print(f"     已复制: {copied}, 跳过（不存在）: {skipped}")

    # ---- 6. 打印 Top-10 ----
    print("\n" + "=" * 80)
    print(f"Top-10 样本 (student 提升最大，按 clip_Δ + dino_Δ 降序)")
    print("=" * 80)
    print(f"{'Rank':<6}{'Name':<10}{'CLIP_tea':>10}{'CLIP_stu':>10}{'CLIP_Δ':>10}"
          f"{'DINO_tea':>10}{'DINO_stu':>10}{'DINO_Δ':>10}{'Combined':>10}")
    print("-" * 80)
    for item in ranked[:10]:
        print(
            f"{item['rank']:<6}{item['name']:<10}"
            f"{item['clip_teacher']:>10.4f}{item['clip_student']:>10.4f}{item['clip_delta']:>+10.4f}"
            f"{item['dino_teacher']:>10.4f}{item['dino_student']:>10.4f}{item['dino_delta']:>+10.4f}"
            f"{item['combined_delta']:>+10.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
