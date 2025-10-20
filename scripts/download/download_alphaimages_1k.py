#!/usr/bin/env python3
"""
两种方式下载 AlphaImages_1k 数据集：

1) snapshot 模式（默认，推荐）：
     - 从 Hugging Face Hub 直接镜像仓库的目录结构（train/test + images/normals）。
     - 不需要 CSV，适合我们现在的私有数据集 ZhiyuanthePony/AlphaImages_1k。

     示例：
     python scripts/download/download_alphaimages_1k.py \
             --dataset ZhiyuanthePony/AlphaImages_1k \
             --out dataset/alphaimages_1k_local

2) csv 模式（可选，向后兼容）：
     - 依据远端 CSV（包含列 path,split）逐文件下载，并写入 <out_dir>/train|test/images/
     - 仅导出图像（不处理 normals）。

通用参数：
    --token <HF_TOKEN>   # 私有仓库需已登录或提供 token
    --max_files N        # 仅下载前 N 个文件（快速测试）
    --include-images/--include-normals  # 控制 snapshot 模式下载范围
"""

import argparse
import os
from pathlib import Path
import shutil
from typing import Any, Optional

from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
import csv


def _load_dataset(dataset: str, split: str, token: Optional[str]):
    """保留占位，当前未使用。"""
    raise RuntimeError("本脚本不使用 datasets.load_dataset；请用 snapshot/csv 两种模式之一")


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
        if not reader.fieldnames or not ("path" in reader.fieldnames and "split" in reader.fieldnames):
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


def _iter_files(base: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(base.glob(pat))
    return out


def _snapshot_mirror(repo_id: str, out_dir: Path, token: Optional[str], include_images: bool, include_normals: bool, max_files: Optional[int] = None) -> tuple[int, int]:
    """将 Hub 上的数据集目录镜像到 out_dir。返回 (copied, skipped_exists)。"""
    allow_patterns: list[str] = []
    if include_images:
        allow_patterns += ["train/images/*.png", "test/images/*.png"]
    if include_normals:
        # normals 下可能有子目录（如 R518）
        allow_patterns += ["train/normals/**/*.png", "test/normals/**/*.png"]

    snap_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns if allow_patterns else None,
        token=token,
    )
    snap = Path(snap_path)

    # 收集候选文件
    pats = []
    if include_images:
        pats += ["train/images/*.png", "test/images/*.png"]
    if include_normals:
        pats += ["train/normals/**/*.png", "test/normals/**/*.png"]
    files = _iter_files(snap, pats)
    files.sort()

    copied = 0
    skipped = 0
    for i, src in enumerate(files):
        if max_files is not None and i >= max_files:
            break
        rel = src.relative_to(snap)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
        if copied % 100 == 0:
            print(f"copied {copied} files -> {out_dir}")
    return copied, skipped


def _head_download(repo_id: str, out_dir: Path, token: Optional[str], include_images: bool, include_normals: bool, max_files: int, revision: str = "main") -> tuple[int, int]:
    """仅下载前 max_files 个匹配文件，避免 snapshot 下载全部。
    返回 (downloaded, skipped_exists)。
    """
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)

    def _want(path: str) -> bool:
        if not path.endswith(".png"):
            return False
        ok = False
        if include_images and (path.startswith("train/images/") or path.startswith("test/images/")):
            ok = True
        if include_normals and (path.startswith("train/normals/") or path.startswith("test/normals/")):
            ok = True
        return ok

    wanted = [p for p in files if _want(p)]
    wanted.sort()
    wanted = wanted[:max_files]

    downloaded = 0
    skipped = 0
    for rel in wanted:
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        local_src = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel, revision=revision, token=token)
        shutil.copy2(local_src, dst)
        downloaded += 1
        if downloaded % 50 == 0:
            print(f"downloaded {downloaded} files -> {out_dir}")
    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AlphaImages_1k from HF Hub via snapshot (default) or CSV list")
    parser.add_argument("--dataset", default="ZhiyuanthePony/AlphaImages_1k", help="HF dataset id")
    parser.add_argument("--out", default=str(Path("dataset") / "alphaimages_1k"), help="output root dir; will mirror repo structure by default")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    parser.add_argument("--mode", choices=["snapshot", "csv"], default="snapshot", help="download mode")
    # snapshot options
    parser.add_argument("--include-images", action="store_true", help="include train/test images")
    parser.add_argument("--include-normals", action="store_true", help="include train/test normals (may contain subfolders)")
    parser.add_argument("--max_files", type=int, default=None, help="limit number of files for a quick test")
    # csv options (legacy)
    parser.add_argument("--split_repo", default=None, help="HF repo id where split CSV lives; default to --dataset if not set")
    parser.add_argument("--split_revision", default="main", help="repo revision for split CSV")
    parser.add_argument("--split_path", default="captions_split.csv", help="CSV path in repo, must contain columns: path,split")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files with same stem (csv mode only)")
    args = parser.parse_args()

    token = args.token
    out_root = Path(args.out)

    if args.mode == "snapshot":
        # 默认同时下载 images 与 normals；若用户未显式指定，则两者都拉取
        include_images = args.include_images or (not args.include_images and not args.include_normals)
        include_normals = args.include_normals or (not args.include_images and not args.include_normals)
        if args.max_files is not None:
            downloaded, skipped = _head_download(
                repo_id=args.dataset,
                out_dir=out_root,
                token=token,
                include_images=include_images,
                include_normals=include_normals,
                max_files=args.max_files,
            )
            print(f"✅ Head download done. downloaded={downloaded}, skipped_exists={skipped}, out_dir={out_root}")
        else:
            copied, skipped = _snapshot_mirror(
                repo_id=args.dataset,
                out_dir=out_root,
                token=token,
                include_images=include_images,
                include_normals=include_normals,
                max_files=None,
            )
            print(f"✅ Snapshot done. copied={copied}, skipped_exists={skipped}, out_dir={out_root}")
        return

    # csv 模式（仅导出 images 到 <out>/train|test/images）
    split_repo = args.split_repo if args.split_repo is not None else args.dataset
    path_split_rows = _load_split_map(split_repo, args.split_revision, args.split_path, token)

    out_train = out_root / "train" / "images"
    out_test = out_root / "test" / "images"
    _ensure_dir(out_train)
    _ensure_dir(out_test)

    saved = 0
    skipped_exists = 0
    for idx, (path_rel, split) in enumerate(path_split_rows):
        if args.max_files is not None and idx >= args.max_files:
            break
        local_src = hf_hub_download(repo_id=split_repo, repo_type="dataset", revision=args.split_revision, filename=path_rel, token=token)
        src = Image.open(local_src)
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

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
        saved += 1
        if saved % 100 == 0:
            print(f"saved {saved} images to {out_root}")

    print(f"✅ CSV export done. saved={saved}, skipped_exists={skipped_exists}, out_dir={out_root}")


if __name__ == "__main__":
    main()


