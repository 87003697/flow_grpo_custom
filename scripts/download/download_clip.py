#!/usr/bin/env python3
"""
下载 CLIP 模型与处理器到项目根目录下的 pretrained_weights/clip/clip-vit-large-patch14
便于离线通过本地路径加载（model_id 与 processor_id 指向同一目录）：
  - clip_model_id      -> pretrained_weights/clip/clip-vit-large-patch14
  - clip_processor_id  -> pretrained_weights/clip/clip-vit-large-patch14
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
    parser = argparse.ArgumentParser(description="下载 CLIP 权重到 pretrained_weights/clip 以供离线加载")
    parser.add_argument("--repo", default="openai/clip-vit-large-patch14", help="CLIP 模型与处理器所在仓库")
    parser.add_argument("--revision", default="main", help="HF 仓库分支/修订")
    parser.add_argument(
        "--out",
        default=None,
        help="输出根目录（默认写入到 <project_root>/pretrained_weights/clip/clip-vit-large-patch14）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = Path(args.out) if args.out is not None else (project_root / "pretrained_weights" / "clip" / "clip-vit-large-patch14")
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"➡️  下载 CLIP: {args.repo} -> {out_root}")
    path = snapshot_to_dir(args.repo, out_root, args.revision)
    print(f"✅ 完成: {path}")

    # 显示在配置文件中应设置的相对路径
    try:
        rel_dir = out_root.relative_to(project_root)
    except ValueError:
        rel_dir = out_root

    print("\n📌 请在配置中使用以下本地路径（模型与处理器相同）：")
    print(f"clip_model_id: {rel_dir}")
    print(f"clip_processor_id: {rel_dir}")
    print("\n🎉 全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())


