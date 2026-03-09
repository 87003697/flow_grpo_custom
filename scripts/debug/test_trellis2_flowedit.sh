#!/bin/bash

# =================================================================
# TRELLIS.2 FlowEdit Refine 快速启动脚本
#
# 两阶段流程：
#   Stage 1: 标准 TRELLIS.2 推理 → clean latent
#   Stage 2: FlowEdit 差分采样 → refine
#
# 用法:
#   bash scripts/debug/test_trellis2_flowedit.sh
#
# 可通过环境变量覆盖默认参数，例如:
#   GPU=0 IMAGE=my_image.png bash scripts/debug/test_trellis2_flowedit.sh
#   REFINE_STAGES="shape" CFG_TGT=5.0 bash scripts/debug/test_trellis2_flowedit.sh
#   N_MAX=10 ROUNDS=3 bash scripts/debug/test_trellis2_flowedit.sh
# =================================================================

set -e

# =================================================================
# 参数（均可通过同名环境变量覆盖）
# =================================================================

# GPU
GPU_ID="${GPU:-1}"

# 模型
MODEL_PATH="${MODEL:-./pretrained_weights/TRELLIS.2-4B}"
DINO_PATH="${DINO_PATH:-./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m}"
PIPELINE_TYPE="${PIPELINE_TYPE:-1024}"

# 输入
IMAGE="${IMAGE:-dataset/test_images/example.png}"
NO_PREPROCESS="${NO_PREPROCESS:-}"

# FlowEdit 参数
REFINE_STAGES="${REFINE_STAGES:-shape tex}"
REFINE_STEPS="${REFINE_STEPS:-}"
REFINE_N_MAX="${REFINE_N_MAX:-}"
CFG_TGT="${CFG_TGT:-}"
CFG_SRC="${CFG_SRC:-}"
RESCALE_MODE="${RESCALE_MODE:-l2_norm}"
ROUNDS="${ROUNDS:-1}"

# 渲染参数
RENDER_RES="${RENDER_RES:-512}"
NUM_VIEWS="${NUM_VIEWS:-4}"
ENVMAP_PATH="${ENVMAP_PATH:-}"
RENDER_CHANNELS="${RENDER_CHANNELS:-shaded normal}"

# 其他
SEED="${SEED:-42}"
LOW_VRAM="${LOW_VRAM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/trellis2_flowedit}"

# =================================================================
# 环境准备
# =================================================================

cd /home/zhiyuan_ma/code/flow_grpo_custom

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis2

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =================================================================
# 构建命令
# =================================================================

CMD="python scripts/debug/test_trellis2_flowedit.py"
CMD+=" --model_path ${MODEL_PATH}"
CMD+=" --pipeline_type ${PIPELINE_TYPE}"
CMD+=" --input_image ${IMAGE}"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --refine_stages ${REFINE_STAGES}"
CMD+=" --rescale_mode ${RESCALE_MODE}"
CMD+=" --num_refine_rounds ${ROUNDS}"
CMD+=" --render_resolution ${RENDER_RES}"
CMD+=" --num_views ${NUM_VIEWS}"
CMD+=" --render_channels ${RENDER_CHANNELS}"
CMD+=" --seed ${SEED}"

# 可选参数
[[ -n "${DINO_PATH}" ]]      && CMD+=" --dino_local_path ${DINO_PATH}"
[[ -n "${REFINE_STEPS}" ]]   && CMD+=" --refine_steps ${REFINE_STEPS}"
[[ -n "${REFINE_N_MAX}" ]]   && CMD+=" --refine_n_max ${REFINE_N_MAX}"
[[ -n "${CFG_TGT}" ]]        && CMD+=" --cfg_strength_tgt ${CFG_TGT}"
[[ -n "${CFG_SRC}" ]]        && CMD+=" --cfg_strength_src ${CFG_SRC}"
[[ -n "${NO_PREPROCESS}" ]]  && CMD+=" --no_preprocess"
[[ -n "${LOW_VRAM}" ]]       && CMD+=" --low_vram"
[[ -n "${ENVMAP_PATH}" ]]    && CMD+=" --envmap_path ${ENVMAP_PATH}"

# =================================================================
# 打印配置
# =================================================================

echo "============================================================"
echo "TRELLIS.2 FlowEdit Refine"
echo "============================================================"
echo "  GPU:             ${GPU_ID}"
echo "  Model:           ${MODEL_PATH}"
echo "  Pipeline:        ${PIPELINE_TYPE}"
echo "  Image:           ${IMAGE}"
echo "  Refine stages:   ${REFINE_STAGES}"
echo "  Refine steps:    ${REFINE_STEPS:-auto}"
echo "  Refine n_max:    ${REFINE_N_MAX:-auto}"
echo "  CFG tgt:         ${CFG_TGT:-auto}"
echo "  CFG src:         ${CFG_SRC:-auto (-tgt)}"
echo "  Rescale mode:    ${RESCALE_MODE}"
echo "  Rounds:          ${ROUNDS}"
echo "  Seed:            ${SEED}"
echo "  Render:          ${RENDER_RES}px x ${NUM_VIEWS} views"
echo "  Channels:        ${RENDER_CHANNELS}"
echo "  Low VRAM:        ${LOW_VRAM:-no}"
echo "  Output:          ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Command: CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}"
echo ""

# =================================================================
# 运行
# =================================================================

CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}
