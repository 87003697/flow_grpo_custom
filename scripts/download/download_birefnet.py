#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BiRefNet weights into HF cache for offline use")
    parser.add_argument("--repo", default="zhengpeng7/BiRefNet", help="HF repo id")
    parser.add_argument("--revision", default="main", help="repo revision")
    args = parser.parse_args()

    path = snapshot_download(repo_id=args.repo, revision=args.revision)
    print(f"Downloaded to cache: {path}")


if __name__ == "__main__":
    main()


