#!/usr/bin/env python3
import argparse
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 model from ModelScope to a local directory")
    parser.add_argument("--repo", default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="ModelScope repo id")
    parser.add_argument("--dest", default="pretrained_weights/dinov3-vitl16-pretrain-lvd1689m", help="local destination dir")
    parser.add_argument("--revision", default=None, help="repo revision (optional)")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 下载模型到指定目录
    kwargs = {
        "model_id": args.repo,
        "cache_dir": str(dest),
    }
    if args.revision:
        kwargs["revision"] = args.revision
    
    path = snapshot_download(**kwargs)
    print(f"模型已下载到: {path}")


if __name__ == "__main__":
    main()

