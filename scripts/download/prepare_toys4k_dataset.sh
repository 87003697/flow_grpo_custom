#!/bin/bash
# 准备 Toys4k 单视图测试数据集（对齐 GSO 格式）
#
# 功能：
#   1. 从已下载的 toys4k_blend_files.zip 中解压 .blend 文件
#   2. 用 Blender 批量渲染单视图（仰角 30°，方位角 0°，512×512 RGBA）
#   3. 输出 toys4k_0000.png … 到 /data/zhiyuan_ma/data/toys4k_test/
#   4. 在 dataset/toys4k_test 创建软链接
#
# 用法：
#   conda activate grpo3d_trellis
#   bash scripts/download/prepare_toys4k_dataset.sh
#
#   # 调试：只渲染前 10 个
#   bash scripts/download/prepare_toys4k_dataset.sh --max_objects 10
#
#   # 断点续传（跳过已渲染）
#   bash scripts/download/prepare_toys4k_dataset.sh --resume
#
#   # 已解压时跳过解压
#   bash scripts/download/prepare_toys4k_dataset.sh --skip_extract --resume
#
# 预计耗时（4 workers，RTX3090 GPU）：
#   - 解压：约 5 分钟
#   - 渲染 4000 个物体：约 2–4 小时
#   - 总计：约 2–5 小时

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

echo "========================================"
echo "Toys4k 数据集准备"
echo "========================================"
echo "项目根目录: $REPO_ROOT"
echo "========================================"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python scripts/download/prepare_toys4k_dataset.py \
    --workers 4 \
    "$@"
