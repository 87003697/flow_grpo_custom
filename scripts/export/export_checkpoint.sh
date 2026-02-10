#!/bin/bash
# 将训练 checkpoint 导出为 TRELLIS 推理兼容的权重目录
#
# 用法：
#   bash scripts/export/export_checkpoint.sh <checkpoint_path> <output_path>
#
# 示例：
#   bash scripts/export/export_checkpoint.sh \
#       logs/trellis_FlowEdit-mts_sgd_lr-1e-3/checkpoints/checkpoint_0_2296 \
#       exports/trellis_finetuned_2296

set -euo pipefail

CHECKPOINT="${1:?用法: $0 <checkpoint_path> <output_path>}"
OUTPUT="${2:?用法: $0 <checkpoint_path> <output_path>}"
PRETRAINED="${3:-pretrained_weights/TRELLIS-image-large}"

echo "========================================"
echo "Export Checkpoint → TRELLIS 推理格式"
echo "========================================"
echo "Checkpoint:  $CHECKPOINT"
echo "Output:      $OUTPUT"
echo "Pretrained:  $PRETRAINED"
echo "========================================"

python scripts/export/export_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --pretrained "$PRETRAINED"
