#!/usr/bin/env bash
# FlowEdit V2 服务调用脚本
# 使用方法: bash scripts/service/run_flowedit_v2.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Qwen-Image-Edit 项目路径
QWEN_EDIT_ROOT="$PROJECT_ROOT/_reference_codes/Qwen-Image-Edit"

# 默认参数 - 使用参考数据集中的测试图像
SOURCE_IMAGE="${SOURCE_IMAGE:-$QWEN_EDIT_ROOT/@dataset/normals/02_azi-45_dst-2.png}"
TARGET_IMAGE="${TARGET_IMAGE:-$QWEN_EDIT_ROOT/@dataset/images/02.jpg}"
PROMPT="${PROMPT:-Move the camera}"

API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8005}"  # 默认使用 GPU 4 的端口
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/flowedit}"

# FlowEdit 参数
SEED="${SEED:-0}"
STEPS="${STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TRUE_CFG_SCALE_TGT="${TRUE_CFG_SCALE_TGT:-15.0}"
N_MIN="${N_MIN:-0}"
N_MAX="${N_MAX:-25}"

echo "=========================================="
echo "FlowEdit V2 API 调用"
echo "=========================================="
echo "API: http://${API_HOST}:${API_PORT}"
echo "Source: $SOURCE_IMAGE"
echo "Target: $TARGET_IMAGE"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

# 调用 Python API 客户端
PYTHONPATH="$QWEN_EDIT_ROOT" python "$QWEN_EDIT_ROOT/src/flowedit/flowedit_v2_api.py" \
    --source-image "$SOURCE_IMAGE" \
    --target-image "$TARGET_IMAGE" \
    --prompt "$PROMPT" \
    --api-url "http://${API_HOST}:${API_PORT}" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --true-cfg-scale-tgt "$TRUE_CFG_SCALE_TGT" \
    --n-min "$N_MIN" \
    --n-max "$N_MAX"

