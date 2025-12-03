#!/usr/bin/env python3
"""一键从 Hugging Face Hub（默认 ZhiyuanthePony/meshes_benchmark）镜像数据集到 dataset/meshes_benchmark_v1."""

import argparse
import importlib
import shutil
from pathlib import Path
from typing import Iterable


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _copy_files(
    src_root: Path,
    dst_root: Path,
    overwrite: bool,
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for src in _iter_files(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        if copied % 100 == 0:
            print(f"[info] copied {copied} files -> {dst_root}")
    return copied, skipped


def _snapshot_download(**kwargs):
    try:
        hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard for linting envs
        raise ModuleNotFoundError(
            "缺少 huggingface_hub 依赖，请先 `pip install huggingface_hub` 后再运行脚本"
        ) from exc
    snapshot_download = getattr(hub, "snapshot_download")
    return snapshot_download(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download meshes_benchmark snapshot to dataset/meshes_benchmark_v1")
    parser.add_argument("--dataset", default="ZhiyuanthePony/meshes_benchmark", help="HF dataset repo id")
    parser.add_argument("--revision", default="main", help="HF repo revision/tag/commit")
    parser.add_argument("--out", default=str(Path("dataset") / "meshes_benchmark_v1"), help="output directory")
    parser.add_argument("--token", default=None, help="HF token（私有仓库需提供，或提前 huggingface-cli login）")
    parser.add_argument("--overwrite", action="store_true", help="若本地已存在文件则重新覆盖")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] snapshot downloading {args.dataset}@{args.revision} ...")
    snapshot_path = _snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        token=args.token,
    )
    snapshot_root = Path(snapshot_path)
    print(f"[info] snapshot cached at {snapshot_root}")

    copied, skipped = _copy_files(
        src_root=snapshot_root,
        dst_root=out_dir,
        overwrite=args.overwrite,
    )
    print(f"✅ done. copied={copied}, skipped_exists={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()


