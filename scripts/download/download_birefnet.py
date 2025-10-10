#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BiRefNet weights into HF cache for offline use")
    parser.add_argument("--repo", default="ZhengPeng7/BiRefNet", help="HF repo id")
    parser.add_argument("--revision", default="main", help="repo revision")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory to place downloaded files. If not set, files are kept in HF cache.",
    )
    args = parser.parse_args()

    if args.out is None:
        path = snapshot_download(repo_id=args.repo, revision=args.revision)
        print(f"Downloaded to cache: {path}")
    else:
        path = snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=args.out)
        print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()


