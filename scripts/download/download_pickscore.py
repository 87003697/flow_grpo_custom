#!/usr/bin/env python3
"""
下载 PickScore 模型与 CLIP 处理器到项目根目录下的 pretrained_weights/pickscore/
便于离线通过本地路径加载：
  - pickscore_model_id      -> pretrained_weights/pickscore/PickScore_v1
  - pickscore_processor_id  -> pretrained_weights/pickscore/CLIP-ViT-H-14-laion2B-s32B-b79K
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
    parser = argparse.ArgumentParser(description="下载 PickScore 相关权重到 pretrained_weights/pickscore 以供离线加载")
    parser.add_argument("--model_repo", default="yuvalkirstain/PickScore_v1", help="PickScore 模型仓库")
    parser.add_argument("--processor_repo", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", help="CLIP 处理器仓库")
    parser.add_argument("--revision", default="main", help="HF 仓库分支/修订")
    parser.add_argument(
        "--out",
        default=None,
        help="输出根目录（默认写入到 <project_root>/pretrained_weights/pickscore）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = Path(args.out) if args.out is not None else (project_root / "pretrained_weights" / "pickscore")
    out_root.mkdir(parents=True, exist_ok=True)

    model_out = out_root / "PickScore_v1"
    proc_out = out_root / "CLIP-ViT-H-14-laion2B-s32B-b79K"

    print(f"➡️  下载 PickScore 模型: {args.model_repo} -> {model_out}")
    model_path = snapshot_to_dir(args.model_repo, model_out, args.revision)
    print(f"✅ 模型完成: {model_path}")

    print(f"➡️  下载 CLIP 处理器: {args.processor_repo} -> {proc_out}")
    proc_path = snapshot_to_dir(args.processor_repo, proc_out, args.revision)
    print(f"✅ 处理器完成: {proc_path}")

    # 显示在配置文件中应设置的相对路径
    try:
        rel_model = model_out.relative_to(project_root)
        rel_proc = proc_out.relative_to(project_root)
    except ValueError:
        # 若 out 不在项目内，则直接显示绝对路径
        rel_model = model_out
        rel_proc = proc_out

    print("\n📌 请在配置中使用以下本地路径：")
    print(f"pickscore_model_id: {rel_model}")
    print(f"pickscore_processor_id: {rel_proc}")
    print("\n🎉 全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())


