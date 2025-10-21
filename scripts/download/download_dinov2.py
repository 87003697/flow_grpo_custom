#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def download_hf(repo: str, dest: str, revision: str = "main") -> None:
    from huggingface_hub import snapshot_download
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(dest_p),
        local_dir_use_symlinks=False,
    )
    print(f"[HF] Downloaded to: {path}")


def warm_torchhub(repo: str, entry: str, torch_home: str) -> None:
    os.environ["TORCH_HUB_DISABLE_NETWORK"] = "0"  # 允许联网以便预热
    os.environ["TORCH_HOME"] = os.path.expanduser(torch_home)
    hub_root = Path(os.environ["TORCH_HOME"]) / "hub"
    (hub_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    import torch
    print(f"[TorchHub] Warming cache: repo={repo}, entry={entry}, TORCH_HOME={os.environ['TORCH_HOME']}")
    _ = torch.hub.load(repo, entry, pretrained=True)

    repo_dir = hub_root / "facebookresearch_dinov2_main"
    ckpt_dir = hub_root / "checkpoints"
    ckpts = sorted(ckpt_dir.glob(f"{entry}-*.pth"))
    print(f"[TorchHub] Repo cached at: {repo_dir} ({'OK' if repo_dir.exists() else 'MISSING'})")
    if ckpts:
        print(f"[TorchHub] Found checkpoint: {ckpts[-1]}")
    else:
        print(f"[TorchHub] WARNING: checkpoint for {entry} not found under {ckpt_dir}")
    print("[TorchHub] Done. You can set TORCH_HUB_DISABLE_NETWORK=1 for offline use.")


def main():
    parser = argparse.ArgumentParser(
        description="Download DINOv2 (HF) or warm TorchHub cache for offline use"
    )
    parser.add_argument(
        "--mode",
        choices=["hf", "torchhub"],
        default="torchhub",
        help="hf: HuggingFace snapshot; torchhub: warm torch.hub cache",
    )
    # HF 参数
    parser.add_argument("--repo", default="facebook/dinov2-giant", help="[hf] HF repo id or local path")
    parser.add_argument("--dest", default="pretrained_weights/dinov2-giant", help="[hf] local destination dir")
    parser.add_argument("--revision", default="main", help="[hf] repo revision")
    # TorchHub 预热参数
    parser.add_argument("--hub_repo", default="facebookresearch/dinov2", help="[torchhub] torch.hub repo, e.g. facebookresearch/dinov2")
    parser.add_argument("--entry", default="dinov2_vitl14_reg", help="[torchhub] entry for torch.hub.load")
    parser.add_argument("--torch_home", default="~/.cache/torch", help="[torchhub] TORCH_HOME for hub cache")
    args = parser.parse_args()

    if args.mode == "hf":
        download_hf(args.repo, args.dest, args.revision)
    else:
        warm_torchhub(args.hub_repo, args.entry, args.torch_home)


if __name__ == "__main__":
    main()

