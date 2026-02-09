#!/usr/bin/env python3
"""
Download BRIA RMBG-2.0 model to pretrained_weights/rmbg2/RMBG-2.0
for offline loading.

Usage:
    python scripts/download/download_rmbg2.py
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def snapshot_to_dir(repo_id: str, out_dir: Path, revision: str = "main") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download BRIA RMBG-2.0 weights to pretrained_weights/rmbg2 for offline loading."
    )
    parser.add_argument("--repo", default="briaai/RMBG-2.0", help="HF repo id")
    parser.add_argument("--revision", default="main", help="HF repo revision")
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: <project_root>/pretrained_weights/rmbg2/RMBG-2.0)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = (
        Path(args.out)
        if args.out is not None
        else (project_root / "pretrained_weights" / "rmbg2" / "RMBG-2.0")
    )
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading RMBG-2.0: {args.repo} -> {out_root}")
    path = snapshot_to_dir(args.repo, out_root, args.revision)
    print(f"Done: {path}")

    try:
        rel_dir = out_root.relative_to(project_root)
    except ValueError:
        rel_dir = out_root

    print("\nUse the local path for loading:")
    print("    from transformers import AutoModelForImageSegmentation")
    print(f'    model = AutoModelForImageSegmentation.from_pretrained("{rel_dir}", trust_remote_code=True)')
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
