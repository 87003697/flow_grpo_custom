#!/usr/bin/env python3
"""
使用 huggingface_hub.snapshot_download 下载 TRELLIS.2-4B 到本地
（默认路径: 项目根/pretrained_weights/TRELLIS.2-4B），保持官方目录结构。

用法:
    python scripts/download/download_trellis2.py [--dest DIR] [--repo-id microsoft/TRELLIS.2-4B] [--local-only]

说明:
- 默认联网下载并利用缓存；如加 --local-only 则仅依赖已有缓存（需缓存完整）。
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TRELLIS.2 weights via snapshot_download.")
    parser.add_argument(
        "--repo-id",
        default="microsoft/TRELLIS.2-4B",
        help="HuggingFace 仓库名，默认 microsoft/TRELLIS.2-4B",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="下载目录，默认项目根/pretrained_weights/TRELLIS.2-4B",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="仅使用本地缓存（不联网）；缓存需完整，否则会报错",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dest = Path(args.dest) if args.dest else project_root / "pretrained_weights" / "TRELLIS.2-4B"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo_id}")
    print(f"Dest: {dest}")

    # 启用 HF transfer 以提升下载速度
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
        local_files_only=bool(args.local_only),
    )
    print(f"✅ Downloaded {args.repo_id} to {dest}")
    print(f"config.pretrained.model 可指向: {dest}")


if __name__ == "__main__":
    main()