#!/usr/bin/env python3
"""
下载 improved-aesthetic-predictor 权重到项目根目录下的
  pretrained_weights/aesthetic/sac+logos+ava1-l14-linearMSE.pth

来源: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import sys
import argparse
import urllib.request
from pathlib import Path


WEIGHT_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
WEIGHT_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


def _download_with_progress(url: str, dst: Path) -> None:
    """带进度条的下载。"""
    import shutil

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dst, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                    print(
                        f"\r  [{bar}] {pct}% ({downloaded}/{total})",
                        end="", flush=True,
                    )
            print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 Aesthetic Predictor (CLIP ViT-L/14 MLP) 权重",
    )
    parser.add_argument(
        "--url", default=WEIGHT_URL,
        help="权重文件 URL",
    )
    parser.add_argument(
        "--out", default=None,
        help="输出根目录（默认 <project_root>/pretrained_weights/aesthetic）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_root = (
        Path(args.out) if args.out is not None
        else (project_root / "pretrained_weights" / "aesthetic")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    dst_path = out_root / WEIGHT_FILENAME

    if dst_path.exists():
        print(f"✅ 已存在，跳过: {dst_path}")
        return 0

    print(f"➡️  下载 {WEIGHT_FILENAME}")
    print(f"   URL: {args.url}")
    _download_with_progress(args.url, dst_path)
    print(f"✅ 完成: {dst_path}")

    try:
        rel_dst = dst_path.relative_to(project_root)
    except ValueError:
        rel_dst = dst_path
    print(f"\n📌 评估时使用:")
    print(f"  --aesthetic_weights {rel_dst}")
    print("\n🎉 全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
