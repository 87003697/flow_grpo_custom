#!/bin/bash
# 准备 GSO (Google Scanned Objects) 测试数据集
#
# 功能：
#   1. 从 HuggingFace 下载 Roldbach/google_scanned_objects（单 zip，~4.7 GB）
#   2. 解压到 /data/zhiyuan_ma/data/gso_extracted/
#   3. 采样 100 个物体，提取正面视角参考图（512×512 白底 RGB）
#   4. 输出到 /data/zhiyuan_ma/data/gso_test/
#   5. 在代码目录创建 dataset/gso_test 软链接
#
# 用法：
#   # 完整运行（下载 + 解压 + 提取）
#   conda activate grpo3d_trellis
#   bash scripts/download/prepare_gso_dataset.sh
#
#   # 只检查 zip 内部结构（不提取图片）
#   bash scripts/download/prepare_gso_dataset.sh --inspect
#
#   # 已有 zip 时跳过下载
#   bash scripts/download/prepare_gso_dataset.sh --local_zip /data/zhiyuan_ma/data/gso/google_scanned_objects.zip
#
# 预计耗时：
#   - 下载：约 1–2 分钟（~4.7 GB，取决于网速；校园网约 30–60 s）
#   - 解压：约 3–5 分钟（zip 中含 54590 个文件）
#   - 图片提取：约 1 分钟
#   - 总计：约 5–10 分钟

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

echo "========================================"
echo "GSO 数据集准备"
echo "========================================"
echo "项目根目录: $REPO_ROOT"
echo "========================================"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python scripts/download/prepare_gso_dataset.py "$@"
