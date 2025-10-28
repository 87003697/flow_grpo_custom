#!/usr/bin/env python3
"""Direct3D-S2 weight downloader (512-only).

假设：
1. 固定 subfolder: `direct3d-s2-v-1-1`。
2. 仅下载 512 相关权重（dense + sparse_512 + refiner）。
3. 不下载 1024 分支（如需 1024 评估请手动获取）。

无容错回退；任一必需文件下载失败即退出非零码。
"""

from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

REQUIRED_512_ONLY = [
    "config.yaml",
    "model_dense.ckpt",
    "model_sparse_512.ckpt",
    "model_refiner.ckpt",
]
SUBFOLDER = "direct3d-s2-v-1-1"


def download(repo_id: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_list = REQUIRED_512_ONLY
    print(f"==> Downloading Direct3D-S2 weights (512-only) from {repo_id} to {out}")
    for fname in file_list:
        local_path = hf_hub_download(
            repo_id=repo_id,
            subfolder=SUBFOLDER,
            filename=fname,
            repo_type="model",
            local_dir=str(out),
        )
        src = Path(local_path)
        dst = out / fname
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        print(f"  [OK] {fname}")

    # Final presence check for required 512-only
    missing = [f for f in REQUIRED_512_ONLY if not (out / f).exists()]
    if len(missing) > 0:
        print(f"Missing required files after download: {missing}")
        sys.exit(1)

    # 不清理任何文件（保留 512-only）

    print("==> Done. Files present:")
    for f in file_list:
        print("   ", f, "✓" if (out / f).exists() else "✗")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="wushuang98/Direct3D-S2", help="HF repo id (e.g. wushuang98/Direct3D-S2)")
    parser.add_argument("--out", type=str, default="pretrained_weights/direct3d_s2-v-1-1", help="output dir (will be created)")
    args = parser.parse_args()
    download(args.repo_id, args.out)


if __name__ == "__main__":
    main()


