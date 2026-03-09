#!/bin/bash
# ============================================================
# Dense 采样一致性测试
#
# 验证 edit4shape/generators/trellis2 的 dense (structure) 接口
# 与 _reference_codes/TRELLIS.2 参考实现产生完全相同的结果。
# ============================================================
set -euo pipefail

# ---- 可配置参数 ----
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-./pretrained_weights/TRELLIS.2-4B}"
DINO_PATH="${DINO_PATH:-./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m}"
PIPELINE_TYPE="${PIPELINE_TYPE:-1024}"
IMAGE="${IMAGE:-dataset/alphaimages_v3/train/00572.png}"
SEED="${SEED:-42}"
LOW_VRAM="${LOW_VRAM:-}"

# ---- 工作目录 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- 构建命令 ----
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/debug/test_dense_sampling_consistency.py"
CMD+=" --model_path ${MODEL_PATH}"
CMD+=" --dino_local_path ${DINO_PATH}"
CMD+=" --pipeline_type ${PIPELINE_TYPE}"
CMD+=" --seed ${SEED}"

if [ -n "${IMAGE}" ]; then
    CMD+=" --input_image ${IMAGE}"
fi

if [ -n "${LOW_VRAM}" ]; then
    CMD+=" --low_vram"
fi

echo "============================================================"
echo "Dense 采样一致性测试"
echo "============================================================"
echo "  GPU:           ${GPU_ID}"
echo "  Model:         ${MODEL_PATH}"
echo "  DINO:          ${DINO_PATH}"
echo "  Pipeline:      ${PIPELINE_TYPE}"
echo "  Image:         ${IMAGE:-<random cond>}"
echo "  Seed:          ${SEED}"
echo "============================================================"
echo ""
echo "Running: ${CMD}"
echo ""

eval "${CMD}"
