#!/usr/bin/env python3
"""
汇总 GSO 评测结果，打印与论文 Table 2 格式一致的对比表。

读取：
    <eval_dir>/teacher_student_similarity.json  → CLIP Sim / DINO Sim
    <eval_dir>/image_quality_student.json        → OREO MANIQA / MUSIQ
    <eval_dir>/image_quality_teacher.json        → Trellis MANIQA / MUSIQ

用法：
    python scripts/eval/aggregate_gso_results.py --eval_dir <eval_dir>

示例：
    python scripts/eval/aggregate_gso_results.py \\
        --eval_dir logs_for_eval/.../eval_teacher_student/checkpoint_11_3444_gso
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="聚合 GSO 评测结果")
    parser.add_argument(
        "--eval_dir", type=str, required=True,
        help="eval_trellis.py 的输出目录（含 teacher_student_similarity.json）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    # ---- 读取 CLIP / DINO ----
    clip_dino_path = eval_dir / "teacher_student_similarity.json"
    clip_dino = load_json(clip_dino_path)
    avg = clip_dino["average"]

    clip_teacher = avg["clip_teacher"]
    clip_student = avg["clip_student"]
    dino_teacher = avg["dino_teacher"]
    dino_student = avg["dino_student"]
    n_samples = len(clip_dino.get("samples", []))

    # ---- 读取 MANIQA / MUSIQ ----
    student_iqa_path = eval_dir / "image_quality_student.json"
    teacher_iqa_path = eval_dir / "image_quality_teacher.json"

    student_iqa = load_json(student_iqa_path)
    teacher_iqa = load_json(teacher_iqa_path)

    def get_iqa(data: dict, metric: str) -> float:
        return data["summary"].get(f"{metric}_mean", float("nan"))

    maniqa_teacher = get_iqa(teacher_iqa, "maniqa")
    maniqa_student = get_iqa(student_iqa, "maniqa")
    musiq_teacher  = get_iqa(teacher_iqa, "musiq")
    musiq_student  = get_iqa(student_iqa, "musiq")

    # ---- 打印结果表 ----
    header = f"{'Method':<26} {'CLIP Sim':>10} {'DINO Sim':>10} {'MANIQA':>10} {'MUSIQ':>10}"
    sep    = "-" * len(header)
    row_t  = f"{'Trellis (pretrained)':<26} {clip_teacher:>10.4f} {dino_teacher:>10.4f} {maniqa_teacher:>10.4f} {musiq_teacher:>10.4f}"
    row_s  = f"{'OREO (ours)':<26} {clip_student:>10.4f} {dino_student:>10.4f} {maniqa_student:>10.4f} {musiq_student:>10.4f}"

    print()
    print("=" * len(header))
    print("GSO Benchmark Results  (n={})".format(n_samples if n_samples else "?"))
    print("=" * len(header))
    print(header)
    print(sep)
    print(row_t)
    print(row_s)
    print(sep)

    # Delta 行
    delta_clip  = clip_student  - clip_teacher
    delta_dino  = dino_student  - dino_teacher
    delta_maniqa = maniqa_student - maniqa_teacher
    delta_musiq  = musiq_student  - musiq_teacher

    def fmt_delta(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.4f}"

    row_d = (
        f"{'Δ (OREO - Trellis)':<26}"
        f" {fmt_delta(delta_clip):>10}"
        f" {fmt_delta(delta_dino):>10}"
        f" {fmt_delta(delta_maniqa):>10}"
        f" {fmt_delta(delta_musiq):>10}"
    )
    print(row_d)
    print("=" * len(header))
    print()

    # ---- 同时保存 JSON 摘要 ----
    summary = {
        "dataset": "GSO",
        "n_samples": n_samples,
        "eval_dir": str(eval_dir),
        "trellis_pretrained": {
            "clip_sim":  clip_teacher,
            "dino_sim":  dino_teacher,
            "maniqa":    maniqa_teacher,
            "musiq":     musiq_teacher,
        },
        "oreo": {
            "clip_sim":  clip_student,
            "dino_sim":  dino_student,
            "maniqa":    maniqa_student,
            "musiq":     musiq_student,
        },
        "delta": {
            "clip_sim":  round(delta_clip,   4),
            "dino_sim":  round(delta_dino,   4),
            "maniqa":    round(delta_maniqa, 4),
            "musiq":     round(delta_musiq,  4),
        },
    }

    out_json = eval_dir / "gso_final_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已保存: {out_json}")


if __name__ == "__main__":
    main()
