#!/bin/bash

# =================================================================
# Generate → FlowEdit Refine 快速启动脚本
#
# 用法:
#   bash scripts/debug/run_generate_and_refine.sh
#
# 可通过环境变量覆盖默认参数，例如:
#   GPU=7 IMAGE=my_image.png PROMPT="Make it red" bash scripts/debug/run_generate_and_refine.sh
#   REFINE_STEPS=10 REFINE_CFG=2.0 bash scripts/debug/run_generate_and_refine.sh
#   ROUNDS=3 bash scripts/debug/run_generate_and_refine.sh
# =================================================================

set -e

# =================================================================
# 参数（均可通过同名环境变量覆盖）
# =================================================================

# GPU
GPU_ID="${GPU:-7}"

# 模型
MODEL_PATH="${MODEL:-Qwen/Qwen-Image-Edit-2511}"

# 输入
IMAGE="${IMAGE:-dataset/alphaimages_1k/test/images/00098.png}"
PROMPT="${PROMPT:-Move the camera}"
NEG_PROMPT="${NEG_PROMPT:-}"

# Stage 1: 生成
GEN_STEPS="${GEN_STEPS:-50}"
CFG="${CFG:-4.0}"
GUIDANCE="${GUIDANCE:-4.0}"

# Stage 2: Refine
REFINE_STEPS="${REFINE_STEPS:-20}"
REFINE_N_MAX="${REFINE_N_MAX:-${REFINE_STEPS}}"
REFINE_CFG="${REFINE_CFG:-${CFG}}"
NOISE_MODE="${NOISE_MODE:-aligned}"
ROUNDS="${ROUNDS:-1}"

# 其他
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/generate_and_refine}"

# =================================================================
# 环境准备
# =================================================================

cd /home/zhiyuan_ma/code/flow_grpo_custom

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =================================================================
# 打印配置
# =================================================================

echo "============================================================"
echo "Generate → FlowEdit Refine"
echo "============================================================"
echo "  GPU:             ${GPU_ID}"
echo "  Image:           ${IMAGE}"
echo "  Prompt:          ${PROMPT}"
echo "  Neg Prompt:      ${NEG_PROMPT}"
echo "  Gen Steps:       ${GEN_STEPS}"
echo "  CFG:             ${CFG}"
echo "  Refine Steps:    ${REFINE_STEPS}"
echo "  Refine n_max:    ${REFINE_N_MAX}"
echo "  Refine CFG:      tgt=+${REFINE_CFG}, src=-${REFINE_CFG}"
echo "  Noise Mode:      ${NOISE_MODE}"
echo "  Rounds:          ${ROUNDS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "============================================================"

# =================================================================
# 运行
# =================================================================

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/debug/test_generate_and_refine.py \
    --model_path "${MODEL_PATH}" \
    --input_image "${IMAGE}" \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEG_PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --gen_steps ${GEN_STEPS} \
    --cfg_scale ${CFG} \
    --guidance_scale ${GUIDANCE} \
    --refine_steps ${REFINE_STEPS} \
    --refine_n_max ${REFINE_N_MAX} \
    --refine_cfg_scale ${REFINE_CFG} \
    --noise_mode ${NOISE_MODE} \
    --num_refine_rounds ${ROUNDS} \
    --seed ${SEED} \
    --dtype ${DTYPE}
