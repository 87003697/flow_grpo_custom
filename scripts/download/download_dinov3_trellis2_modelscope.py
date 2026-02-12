#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 model from ModelScope and move to TRELLIS2 expected layout")
    parser.add_argument("--repo", default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="ModelScope repo id")
    parser.add_argument(
        "--dest",
        default="pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="final destination dir used by TRELLIS2 config",
    )
    parser.add_argument("--revision", default=None, help="repo revision (optional)")
    parser.add_argument("--force", action="store_true", help="overwrite destination if it already exists")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = dest.parents[2] / "_modelscope_cache" / "dinov3-vitl16-pretrain-lvd1689m"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 下载到缓存目录，再移动到最终目录（保证目录结构与项目配置一致）
    kwargs = {
        "model_id": args.repo,
        "cache_dir": str(cache_dir),
    }
    if args.revision:
        kwargs["revision"] = args.revision

    downloaded_path = Path(snapshot_download(**kwargs)).resolve()
    if dest.exists():
        if not args.force:
            print(f"目标目录已存在，跳过移动: {dest}")
            print("如需覆盖请追加 --force")
            print(f"已下载缓存目录: {downloaded_path}")
            return
        shutil.rmtree(dest)

    shutil.move(str(downloaded_path), str(dest))
    print(f"模型已移动到: {dest}")
    print(f"可直接用于 cfg.pretrained.dino_local_path: {dest}")


if __name__ == "__main__":
    main()

