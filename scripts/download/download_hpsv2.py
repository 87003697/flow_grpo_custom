#!/usr/bin/env python3
"""
下载 HPSv2 权重到项目根目录下的 pretrained_weights/hpsv2/
默认文件名：
  - v2.1: HPS_v2.1_compressed.pt
  - v2.0: HPS_v2_compressed.pt
"""

import sys
import argparse
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download


HPS_VERSION_MAP = {
    "v2.1": "HPS_v2.1_compressed.pt",
    "v2.0": "HPS_v2_compressed.pt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 HPSv2 权重到 pretrained_weights/hpsv2")
    parser.add_argument("--version", default="v2.1", choices=list(HPS_VERSION_MAP.keys()), help="HPSv2 版本")
    parser.add_argument("--repo", default="xswu/HPSv2", help="HF 仓库")
    parser.add_argument(
        "--out",
        default=None,
        help="输出根目录（默认写入到 <project_root>/pretrained_weights/hpsv2）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = Path(args.out) if args.out is not None else (project_root / "pretrained_weights" / "hpsv2")
    out_root.mkdir(parents=True, exist_ok=True)

    filename = HPS_VERSION_MAP[args.version]
    print(f"➡️  从 {args.repo} 下载 {filename}")
    cached_path = hf_hub_download(repo_id=args.repo, filename=filename)
    dst_path = out_root / filename
    print(f"➡️  复制到 {dst_path}")
    shutil.copy2(cached_path, dst_path)
    print(f"✅ 完成: {dst_path}")

    # 提示配置字段
    try:
        rel_dst = dst_path.relative_to(project_root)
    except ValueError:
        rel_dst = dst_path
    print("\n📌 请在配置中使用以下本地路径：")
    print(f"hpsv2_ckpt_path: {rel_dst}")
    print("\n🎉 全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())


