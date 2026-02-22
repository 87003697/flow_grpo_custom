#!/usr/bin/env python3
import argparse
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 model from ModelScope to a local directory")
    parser.add_argument("--repo", default="facebook/dinov3-vith16plus-pretrain-lvd1689m", help="ModelScope repo id")
    parser.add_argument("--dest", default="pretrained_weights/dinov3-vith16plus-pretrain-lvd1689m", help="local destination dir")
    parser.add_argument("--revision", default=None, help="repo revision (optional)")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # 直接下载到目标目录，避免 cache_dir 带来的目录层级混淆。
    kwargs = {
        "model_id": args.repo,
        "local_dir": str(dest),
    }
    if args.revision:
        kwargs["revision"] = args.revision

    downloaded_root = Path(snapshot_download(**kwargs)).resolve()
    nested_model_dir = downloaded_root / args.repo
    model_dir = nested_model_dir if nested_model_dir.is_dir() else downloaded_root

    print(f"下载完成，快照目录: {downloaded_root}")
    print(f"建议在代码中使用的模型目录: {model_dir}")


if __name__ == "__main__":
    main()

