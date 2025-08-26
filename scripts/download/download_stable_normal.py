#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="预下载 Stable Normal 权重到本地 pretrained_weights 目录 (与参考实现一致)")
    parser.add_argument("--weights_dir", type=str, default="./pretrained_weights", help="权重缓存根目录")
    parser.add_argument("--version", type=str, default="yoso-normal-v1-8-1", help="YOSO normal 模型版本")
    args = parser.parse_args()

    repo_id = f"Stable-X/{args.version}"
    local_dir = os.path.join(args.weights_dir, args.version)
    os.makedirs(args.weights_dir, exist_ok=True)

    print(f"将下载 {repo_id} 到 {local_dir}")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, force_download=False)
    print(f"✅ 完成，已缓存于: {path}")


if __name__ == "__main__":
    main()


