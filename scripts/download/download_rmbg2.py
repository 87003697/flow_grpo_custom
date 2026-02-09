#!/usr/bin/env python3
"""
下载 BRIA RMBG-2.0 背景移除模型到项目根目录下的 pretrained_weights/rmbg2/RMBG-2.0
便于离线通过本地路径加载：
  - AutoModelForImageSegmentation.from_pretrained("pretrained_weights/rmbg2/RMBG-2.0")

使用方法：
  conda activate grpo3d_trellis
  python scripts/download/download_rmbg2.py
"""

import sys
import argparse
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
    parser = argparse.ArgumentParser(description="下载 BRIA RMBG-2.0 背景移除模型权重到 pretrained_weights/rmbg2 以供离线加载")
    parser.add_argument("--repo", default="briaai/RMBG-2.0", help="HF 仓库 ID")
    parser.add_argument("--revision", default="main", help="HF 仓库分支/修订")
    parser.add_argument(
        "--out",
        default=None,
        help="输出目录（默认写入到 <project_root>/pretrained_weights/rmbg2/RMBG-2.0）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = Path(args.out) if args.out is not None else (project_root / "pretrained_weights" / "rmbg2" / "RMBG-2.0")
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"➡️  下载 RMBG-2.0: {args.repo} -> {out_root}")
    path = snapshot_to_dir(args.repo, out_root, args.revision)
    print(f"✅ 完成: {path}")

    # 显示在配置文件中应设置的相对路径
    try:
        rel_dir = out_root.relative_to(project_root)
    except ValueError:
        rel_dir = out_root

    print(f"\n📌 请在代码中使用以下本地路径加载模型：")
    print(f"    from transformers import AutoModelForImageSegmentation")
    print(f'    model = AutoModelForImageSegmentation.from_pretrained("{rel_dir}", trust_remote_code=True)')
    print(f"\n🎉 全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
