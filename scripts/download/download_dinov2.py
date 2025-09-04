#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download DINOv2 model to a local directory")
    parser.add_argument("--repo", default="facebook/dinov2-base", help="HF repo id or local path")
    parser.add_argument("--dest", default="pretrained_weights/dinov2-base", help="local destination dir")
    parser.add_argument("--revision", default="main", help="repo revision")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()


