#!/usr/bin/env python3
"""基于 CameraNormalScorer 对 meshes_benchmark_v1 做 VLM 法线验证。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from reward_models.camera_normal_scorer.scorer import CameraNormalScorer
from scripts.eval.utils_camera_normal import (
    load_glb_mesh_as_obj,
    load_normal_pil_from_cache,
    _rotate_meshes_by_source_front,
)


def _resolve_dataset_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (root / path)


def _maybe_prefer_high_res_mesh(mesh_path: Path, pipeline: str) -> Path:
    """direct3d pipeline 优先使用 *_1024.ply，如缺失则回退原路径。"""
    pipeline_lower = pipeline.lower()
    if "direct3d" not in pipeline_lower:
        return mesh_path
    if mesh_path.suffix.lower() != ".ply":
        return mesh_path
    if mesh_path.stem.endswith("_1024"):
        return mesh_path
    high_res_path = mesh_path.with_name(f"{mesh_path.stem}_1024{mesh_path.suffix}")
    if high_res_path.is_file():
        return high_res_path
    print(f"[warn] direct3d mesh 缺少 {high_res_path.name}，使用 {mesh_path.name}")
    return mesh_path


def _build_vis_subdir(base_dir: Path, image_path: str) -> Path:
    """基于 image_path 还原 images 下的相对层级，用于保存可视化结果。"""
    img_path = Path(image_path)
    parts = img_path.parts
    rel_parts: Tuple[str, ...] = ()
    for idx, part in enumerate(parts):
        if part == "images":
            rel_parts = parts[idx + 1 :]
            break
    if len(rel_parts) == 0:
        rel_parts = (img_path.name,)
    rel_path = Path(*rel_parts)
    try:
        rel_path = rel_path.with_suffix("")
    except ValueError:
        pass
    return base_dir / rel_path


def _load_dataset_entries(index_path: Path) -> List[Dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{index_path} 不是列表")
    return data


def _iter_pipeline_filter(filter_str: Optional[str]) -> Optional[Sequence[str]]:
    if filter_str is None or filter_str.strip() == "":
        return None
    return [item.strip() for item in filter_str.split(",") if item.strip()]


def _should_keep_pipeline(pipeline: str, allowed: Optional[Sequence[str]]) -> bool:
    if allowed is None:
        return True
    return pipeline in allowed


def _collect_inputs(
    entries: Sequence[Dict[str, Any]],
    dataset_root: Path,
    cache_dir: str,
    normal_resolution: int,
    allowed_pipelines: Optional[Sequence[str]],
    max_count: int,
) -> Tuple[List[Any], List[Image.Image], List[Dict[str, Any]], List[Dict[str, Any]]]:
    meshes: List[Any] = []
    images: List[Image.Image] = []
    metadata: List[Dict[str, Any]] = []
    record_infos: List[Dict[str, Any]] = []

    selected_entries = entries[: max_count if max_count > 0 else len(entries)]
    for entry_idx, entry in enumerate(selected_entries):
        image_rel = entry.get("image")
        normal_rel = entry.get("normal")
        mesh_records = entry.get("meshes", [])

        if not image_rel or not isinstance(mesh_records, list):
            continue

        image_path = _resolve_dataset_path(dataset_root, image_rel)
        normal_path = _resolve_dataset_path(dataset_root, normal_rel) if normal_rel else None

        if not image_path.is_file():
            print(f"[warn] 跳过 image={image_path} (不存在)")
            continue

        try:
            rgb_pil = Image.open(image_path).convert("RGB")
        except Exception as exc:  # pragma: no cover - PIL 读取失败
            print(f"[warn] 读取 image 失败: {image_path}, err={exc}")
            continue

        normal_pil = None
        if normal_path and normal_path.is_file():
            normal_pil = Image.open(normal_path).convert("RGB")
        else:
            try:
                normal_pil = load_normal_pil_from_cache(str(image_path), cache_dir, normal_resolution)
            except FileNotFoundError as exc:
                print(f"[warn] 跳过 sample={image_path}，因 normal 缺失: {exc}")
                continue

        for mesh_idx, mesh_info in enumerate(mesh_records):
            pipeline = mesh_info.get("pipeline", "unknown")
            if not _should_keep_pipeline(pipeline, allowed_pipelines):
                continue
            mesh_rel = mesh_info.get("path")
            if not mesh_rel:
                continue
            mesh_path = _resolve_dataset_path(dataset_root, mesh_rel)
            mesh_path = _maybe_prefer_high_res_mesh(mesh_path, pipeline)
            if not mesh_path.is_file():
                print(f"[warn] 缺少 mesh={mesh_path}, pipeline={pipeline}")
                continue
            try:
                mesh_obj = load_glb_mesh_as_obj(str(mesh_path))
            except Exception as exc:  # pragma: no cover - trimesh 抛错
                print(f"[warn] trimesh 解析失败: {mesh_path}, err={exc}")
                continue

            meshes.append(mesh_obj)
            images.append(rgb_pil.copy())
            metadata.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "normal_pil": normal_pil.copy(),
                }
            )
            record_infos.append(
                {
                    "sample_index": entry.get("sample_id", str(entry_idx)),
                    "image_path": str(image_path),
                    "mesh_path": str(mesh_path),
                    "pipeline": pipeline,
                    "method": entry.get("method", "unknown"),
                    "candidate_id": mesh_idx,
                }
            )

    return meshes, images, metadata, record_infos


def _build_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "normal_resolution": args.normal_resolution,
        "cache_dir": args.cache_dir,
        "encoder": args.encoder,
        "dino_v2_path": args.dino_v2_path,
        "dino_v3_path": args.dino_v3_path,
        "save_vis": args.save_vis,
        "vis_dir": args.vis_dir,
        "cam_batch_size": args.cam_batch_size,
        "render_batch_size": args.render_batch_size,
        "encoding_batch_size": args.encoding_batch_size,
        "camera_config_py": args.camera_config,
        "camera_ckpt": args.camera_ckpt,
        "camera_param_dim": args.camera_param_dim,
        "img_size": args.img_size,
        "camera_type": args.camera_type,
        "source_front": args.source_front,
        "avg_camera_per_group": args.avg_camera_per_group,
        "use_RGB_for_comparison": args.use_rgb_for_comparison,
        "vlm_api_source": args.vlm_api_source,
        "vlm_max_concurrent": args.vlm_max_concurrent,
        "vlm_timeout": args.vlm_timeout,
        "vlm_prompt_version": args.vlm_prompt_version,
        "vlm_max_tokens": args.vlm_max_tokens,
        "vlm_enable_thinking": args.vlm_enable_thinking,
        "vlm_debug_raw_response": args.vlm_debug_response,
    }


def _save_scores_csv(
    csv_path: Path,
    record_infos: Sequence[Dict[str, Any]],
    scores: Sequence[float],
    pipeline_stats: Dict[str, Dict[str, float]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "method", "pipeline", "candidate_id", "image", "mesh", "score"])
        for info, score in zip(record_infos, scores):
            writer.writerow(
                [
                    info["sample_index"],
                    info["method"],
                    info["pipeline"],
                    info["candidate_id"],
                    info["image_path"],
                    info["mesh_path"],
                    f"{float(score):.6f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["pipeline", "count", "mean", "std"])
        for pipeline, stat in pipeline_stats.items():
            writer.writerow(
                [
                    pipeline,
                    int(stat["count"]),
                    f"{stat['mean']:.6f}",
                    f"{stat['std']:.6f}",
                ]
            )


def _compute_pipeline_stats(record_infos: Sequence[Dict[str, Any]], scores: Sequence[float]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for info, score in zip(record_infos, scores):
        buckets[info["pipeline"]].append(float(score))

    stats: Dict[str, Dict[str, float]] = {}
    for pipeline, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        stats[pipeline] = {
            "count": float(arr.size),
            "mean": float(arr.mean()) if arr.size > 0 else 0.0,
            "std": float(arr.std()) if arr.size > 0 else 0.0,
        }
    return stats


def _save_normal_comparisons(
    grouped_meta: Sequence[Dict[str, Any]],
    record_infos: Sequence[Dict[str, Any]],
    save_dir: Path,
    cols: int = 4,
) -> None:
    if len(grouped_meta) == 0:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    info_lookup = {idx: info for idx, info in enumerate(record_infos)}
    label_height = 28  # 形状: 标量

    for grp in grouped_meta:
        base = _build_vis_subdir(save_dir, grp["image_path"])
        base.mkdir(parents=True, exist_ok=True)

        candidates = grp.get("candidates", [])
        best_by_mesh: Dict[int, Dict[str, Any]] = {}
        for cand in candidates:
            mesh_index = cand.get("mesh_index")
            if mesh_index is None:
                continue
            mesh_index = int(mesh_index)
            score_val = cand.get("score") if cand.get("score") is not None else -1.0
            prev = best_by_mesh.get(mesh_index)
            prev_score = prev.get("score") if (prev and prev.get("score") is not None) else -1.0
            if (prev is None) or (score_val > prev_score):
                best_by_mesh[mesh_index] = cand
        deduped_cands = list(best_by_mesh.values()) if len(best_by_mesh) > 0 else candidates
        sorted_cands = sorted(
            deduped_cands,
            key=lambda c: c.get("score") if c.get("score") is not None else -1.0,
            reverse=True,
        )

        ref_pil = grp.get("image_rgb_pil") or grp.get("image_normal_pil")
        tiles = [
            {"label": "reference", "score": None, "pil": ref_pil},
        ]
        for cand in sorted_cands:
            mesh_index = cand.get("mesh_index")
            info = info_lookup.get(int(mesh_index)) if mesh_index is not None else None
            label = "mesh"
            if info is not None:
                label = f"{info['pipeline']}#{info['candidate_id']}"
            tiles.append(
                {
                    "label": label,
                    "score": cand.get("score"),
                    "pil": cand.get("rendered_normal_pil"),
                }
            )

        if len(tiles) == 0 or tiles[0]["pil"] is None:
            continue

        w, h = tiles[0]["pil"].size
        effective_cols = max(1, min(cols, len(tiles)))
        rows = ceil(len(tiles) / effective_cols)
        canvas_width = effective_cols * w
        canvas_height = rows * (h + label_height)
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        for idx, tile in enumerate(tiles):
            pil = tile.get("pil")
            if pil is None:
                continue
            r, c = divmod(idx, effective_cols)
            x0 = c * w
            y0 = r * (h + label_height)
            canvas.paste(pil, (x0, y0))
            text = tile["label"]
            if tile["score"] is not None:
                text += f" | {tile['score']:.3f}"
            text_y = y0 + h + 4
            draw.text((x0 + 4, text_y), text, fill=(0, 0, 0), font=font)

        canvas.save(base / "normal_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CameraNormalScorer VLM 验证脚本")
    parser.add_argument("--dataset-root", default="dataset/meshes_benchmark_v1", help="数据集根目录")
    parser.add_argument(
        "--dataset-index", default="metadata/dataset_index.json", help="dataset_root 下 dataset_index.json 相对路径"
    )
    parser.add_argument("--device", default="cuda", help="推理 device")
    parser.add_argument("--normal-resolution", type=int, default=512, help="法线渲染分辨率 R")
    parser.add_argument("--cache-dir", default="./tmp_camera_normal_cache", help="法线缓存目录")
    parser.add_argument("--encoder", default="gemini-2.5-flash", help="编码器/评分模型")
    parser.add_argument("--camera-ckpt", required=True, help="VGGT 相机 checkpoint")
    parser.add_argument("--camera-config", default="_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py")
    parser.add_argument("--camera-type", default="search", help="camera_type，支持 search/fixed_v1 等")
    parser.add_argument("--camera-param-dim", type=int, default=9)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--cam-batch-size", type=int, default=64)
    parser.add_argument("--render-batch-size", type=int, default=32)
    parser.add_argument("--encoding-batch-size", type=int, default=64)
    parser.add_argument("--avg-camera-per-group", action="store_true")
    parser.add_argument("--use-rgb-for-comparison", action="store_true")
    parser.add_argument("--vlm-api-source", default="1")
    parser.add_argument("--vlm-max-concurrent", type=int, default=2)
    parser.add_argument("--vlm-timeout", type=float, default=180.0)
    parser.add_argument("--vlm-prompt-version", default="v1")
    parser.add_argument("--vlm-max-tokens", type=int, default=8000)
    parser.add_argument("--vlm-enable-thinking", action="store_true")
    parser.add_argument("--vlm-debug-response", action="store_true", help="打印 Gemini 原始响应")
    parser.add_argument("--dino-v2-path", default="pretrained_weights/dinov2-giant")
    parser.add_argument("--dino-v3-path", default="pretrained_weights/dinov3-vitb14")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-dir", default="outputs/camera_vis")
    parser.add_argument("--source-front", default="+z", help="mesh 输入的朝向，用于旋转对齐")
    parser.add_argument("--pipelines", default=None, help="逗号分隔，仅保留指定 pipeline")
    parser.add_argument("--max-count", type=int, default=-1, help="限制处理的 samples 数量")
    parser.add_argument("--output-csv", default="outputs/meshes_benchmark_v1/vlm_scores.csv")
    parser.add_argument("--save-dir", default="outputs/vlm_vis_demo", help="法线拼图输出目录")
    parser.add_argument("--save-cols", type=int, default=4, help="法线拼图列数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    dataset_index_path = dataset_root / args.dataset_index
    if not dataset_index_path.is_file():
        raise FileNotFoundError(f"dataset_index 不存在: {dataset_index_path}")

    entries = _load_dataset_entries(dataset_index_path)
    allowed_pipelines = _iter_pipeline_filter(args.pipelines)
    meshes, images, metadata, record_infos = _collect_inputs(
        entries=entries,
        dataset_root=dataset_root,
        cache_dir=args.cache_dir,
        normal_resolution=args.normal_resolution,
        allowed_pipelines=allowed_pipelines,
        max_count=args.max_count,
    )

    if len(meshes) == 0:
        print("[error] 没有可用的 mesh")
        return

    _rotate_meshes_by_source_front(meshes, args.source_front)

    cfg = _build_cfg_from_args(args)
    device = torch.device(args.device)
    scorer = CameraNormalScorer(device=device, cfg=cfg)

    result = scorer.compute_scores(meshes, images, metadata)
    if isinstance(result, tuple):
        scores, grouped_meta = result
    else:
        scores, grouped_meta = result, []

    pipeline_stats = _compute_pipeline_stats(record_infos, scores)
    _save_scores_csv(Path(args.output_csv), record_infos, scores, pipeline_stats)

    if len(grouped_meta) > 0:
        _save_normal_comparisons(grouped_meta, record_infos, Path(args.save_dir), cols=max(1, args.save_cols))

    summary = {
        "num_meshes": len(scores),
        "num_groups": len(grouped_meta),
        "output_csv": args.output_csv,
        "save_dir": args.save_dir,
        "pipelines": allowed_pipelines,
        "stats": pipeline_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


