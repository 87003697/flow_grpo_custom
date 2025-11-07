#!/usr/bin/env python3
"""
下载 ULIP-1 PointBERT 预训练权重到项目根目录下的 pretrained_weights/

固定使用 Hugging Face 镜像源（datasets/SFXX/ulip），不再尝试 GCS 或环境变量覆盖。
"""

import os
import sys
from pathlib import Path
import urllib.request


# 固定使用 Hugging Face 镜像源
DEFAULT_ULIP_POINTBERT_URL = (
    "https://huggingface.co/"
    "datasets/SFXX/ulip/resolve/main/"
    "ULIP-1/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt"
)


def download_ulip_pointbert() -> Path:
    """下载 ULIP-1 PointBERT 权重并保存为 pretrained_weights/checkpoint_pointbert.pt。

    返回保存路径。
    """
    print("🔄 正在下载 ULIP-1 PointBERT 预训练权重…")

    # 解析项目根目录与输出路径
    project_root = Path(__file__).resolve().parent.parent.parent
    weights_dir = project_root / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)

    out_path = weights_dir / "checkpoint_pointbert.pt"

    # 现有文件存在性与大小检查（>100MB 认为完整）
    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        print(f"✅ 发现已存在的 ULIP-1 PointBERT 权重: {out_path}")
        return out_path

    # 固定使用 Hugging Face 镜像源
    url = DEFAULT_ULIP_POINTBERT_URL
    print(f"➡️  下载源 (Hugging Face): {url}")

    tmp_path = out_path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    Path(tmp_path).rename(out_path)
    print(f"✅ ULIP-1 PointBERT 权重已保存到: {out_path}")
    return out_path


def main():
    print("🚀 开始下载 ULIP 系列权重…")
    path = download_ulip_pointbert()
    print("\n✅ 全部完成！")
    print(f"ULIP-1 PointBERT: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n📊 文件大小: {size_mb:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())



