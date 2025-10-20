#!/usr/bin/env python3
"""
从 Hugging Face 数据集导出全部样本，并依据远端 CSV（包含列 id,split）划分为 train/test：

<out_dir>/train/images/*.png
<out_dir>/test/images/*.png

默认数据集：ZhiyuanthePony/AlphaImages_10k。

示例：
python scripts/download/download_alphaimages_1k.py \
    --dataset ZhiyuanthePony/AlphaImages_10k \
    --out dataset/alphaimages_1k \
    --token <HF_TOKEN> \
    --split_repo ZhiyuanthePony/AlphaImages_10k \
    --split_path split.csv
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from huggingface_hub import hf_hub_download
import csv


def _load_dataset(dataset: str, split: str, token: Optional[str]):
    """不再使用 datasets 库加载样本，若被调用则直接报错（严格）。"""
    raise RuntimeError("此脚本按 CSV 中的 path 逐文件下载，不再使用 datasets.load_dataset")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _stem_from_path(path_rel: str) -> str:
    p = Path(path_rel)
    return p.stem


def _load_split_map(repo: str, revision: str, path_in_repo: str, token: Optional[str]) -> list[tuple[str, str]]:
    """下载并解析 (path, split) 列表（仅允许 train/test）。"""
    local_path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=path_in_repo, token=token)
    rows: list[tuple[str, str]] = []
    with open(local_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not ("path" in reader.fieldnames and "split" in reader.fieldnames):
            raise KeyError("split CSV 必须包含列: path, split")
        for row in reader:
            path_val = str(row["path"]) if row["path"] is not None else None
            split_val = str(row["split"]).strip().lower() if row["split"] is not None else None
            if path_val is None or split_val is None:
                raise ValueError("split CSV 存在空 path 或 split")
            if split_val not in ("train", "test"):
                raise ValueError(f"仅支持 split=train/test，发现: {split_val}")
            rows.append((path_val, split_val))
    if len(rows) == 0:
        raise ValueError("split CSV 为空")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export images by split CSV into <out>/train|test/images as .png files")
    parser.add_argument("--dataset", default="ZhiyuanthePony/AlphaImages_10k", help="HF dataset id")
    parser.add_argument("--out", default=str(Path("dataset") / "alphaimages_1k"), help="output root dir; images will go to <out>/(train|test)/images")
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--split_repo", default=None, help="HF repo id where split CSV lives; default to --dataset if not set")
    parser.add_argument("--split_revision", default="main", help="repo revision for split CSV")
    parser.add_argument("--split_path", default="captions_split.csv", help="CSV path in repo, must contain columns: id,split")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files with same stem")
    args = parser.parse_args()

    token = args.token
    split_repo = args.split_repo if args.split_repo is not None else args.dataset
    path_split_rows = _load_split_map(split_repo, args.split_revision, args.split_path, token)

    out_root = Path(args.out)
    out_train = out_root / "train" / "images"
    out_test = out_root / "test" / "images"
    _ensure_dir(out_train)
    _ensure_dir(out_test)

    saved = 0
    skipped_exists = 0
    for path_rel, split in path_split_rows:
        # 按 CSV 的相对路径逐文件下载
        local_src = hf_hub_download(repo_id=split_repo, repo_type="dataset", revision=args.split_revision, filename=path_rel, token=token)
        src = Image.open(local_src)
        # 保留 alpha：若存在 alpha（RGBA/LA/调色板带透明），保存为 RGBA；否则保存为 RGB
        if src.mode in ("RGBA", "LA"):
            pil = src.convert("RGBA")
        elif src.mode == "P" and ("transparency" in src.info):
            pil = src.convert("RGBA")
        else:
            pil = src.convert("RGB")
        stem = _stem_from_path(path_rel)
        out_path = (out_train if split == "train" else out_test) / f"{stem}.png"

        if out_path.exists() and not args.overwrite:
            skipped_exists += 1
            continue

        pil.save(out_path)
        saved += 1
        if saved % 100 == 0:
            print(f"saved {saved} images to {out_root}")

    print(f"✅ Done. saved={saved}, skipped_exists={skipped_exists}, out_dir={out_root}")


if __name__ == "__main__":
    main()


