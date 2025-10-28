#!/usr/bin/env python3
"""
下载 vggt-camera-search 权重子目录到 pretrained_weights/vggt-camera-search。

示例:
  export HF_TOKEN=你的令牌
  python scripts/download/download_vggt_camera_search.py \
    --repo ZhiyuanthePony/vggt-camera-search_v1 \
    --subdir 2025.08.20_08.56.06 \
    --dest pretrained_weights/vggt-camera-search \
    --revision main
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download vggt-camera-search subdir into local directory"
    )
    parser.add_argument(
        "--repo",
        default="ZhiyuanthePony/vggt-camera-search_v1",
        help="HF repo id",
    )
    parser.add_argument(
        "--subdir",
        default="2025.08.20_08.56.06",
        help="sub directory inside repo to download",
    )
    parser.add_argument(
        "--dest",
        default=str(
            Path(__file__).parent.parent.parent
            / "pretrained_weights"
            / "vggt-camera-search"
        ),
        help="local destination dir",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="repo revision",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        allow_patterns=[f"{args.subdir}/*", ".gitattributes"],
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()


