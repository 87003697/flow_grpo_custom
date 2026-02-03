#!/usr/bin/env python3
"""
使用 snapshot 下载 ZhiyuanthePony/AlphaImages_v2 数据集。

示例：
    python scripts/download/download_alphaimages_v2.py \
            --out dataset/alphaimages_v2
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AlphaImages_v2 from HF Hub via snapshot")
    parser.add_argument("--dataset", default="ZhiyuanthePony/AlphaImages_v2", help="HF dataset id")
    parser.add_argument("--out", default=str(Path("dataset") / "alphaimages_v2"), help="output root dir")
    parser.add_argument("--token", default=None, help="HF token (optional if already logged in)")
    args = parser.parse_args()

    out_root = Path(args.out)
    snap_path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        token=args.token,
    )
    snap = Path(snap_path)

    # 直接镜像 train/test 下的 png 文件到输出目录
    copied = 0
    skipped = 0
    for src in sorted(snap.rglob("*.png")):
        rel = src.relative_to(snap)
        if not (str(rel).startswith("train/") or str(rel).startswith("test/")):
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
            continue
        dst.write_bytes(src.read_bytes())
        copied += 1
        if copied % 100 == 0:
            print(f"copied {copied} files -> {out_root}")

    print(f"✅ Snapshot done. copied={copied}, skipped_exists={skipped}, out_dir={out_root}")


if __name__ == "__main__":
    main()


