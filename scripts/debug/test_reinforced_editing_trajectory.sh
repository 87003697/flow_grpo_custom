#!/bin/bash
# Reinforced Editing Trajectory 可视化对比脚本
#
# 对比 FlowEdit 中有 / 无 negative source guidance 的中间步编辑轨迹。
# 支持单卡和多卡（DDP）模式，每张 GPU 独立处理分配到的图片子集。
#
# 用法：
#   # 单卡（默认 GPU 7）
#   bash scripts/debug/test_reinforced_editing_trajectory.sh
#
#   # 单卡 + 指定目录
#   bash scripts/debug/test_reinforced_editing_trajectory.sh dataset/alphaimages_v3/test
#
#   # 单卡 + 指定单张图片
#   bash scripts/debug/test_reinforced_editing_trajectory.sh dataset/alphaimages_v3/test/test_06.png
#
#   # 多卡（自动检测 CUDA_VISIBLE_DEVICES 中的 GPU 数量）
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/debug/test_reinforced_editing_trajectory.sh

set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"

# 如果没设 CUDA_VISIBLE_DEVICES，默认单卡
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# === 可调参数 ===
STEPS=12
N_MAX=9
CFG_SCALE=4
PROMPT="Rotate the camera."
NEG_PROMPT=" "
NOISE_MODE="aligned"
SEED=42
SAVE_STEPS="0,3,6,8"
CELL_SIZE=256
RENDER_RES=512

TRELLIS_MODEL="pretrained_weights/TRELLIS-image-large"
FLOWEDIT_MODEL="Qwen/Qwen-Image-Edit-2511"

PYTHON="/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis/bin/python"
TORCHRUN="/home/zhiyuan_ma/anaconda3/envs/grpo3d_trellis/bin/torchrun"

# 输入（支持命令行传入目录或单张图片，默认 test_06.png）
INPUT_PATH="${1:-dataset/alphaimages_v3/test/test_06.png}"

RUN_NAME="trajectory_nmax-${N_MAX}_steps-${STEPS}_cfg-${CFG_SCALE}_${NOISE_MODE}"
OUTPUT_DIR="outputs/reinforced_editing_trajectory/${RUN_NAME}"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 判断是文件还是目录
if [ -f "$INPUT_PATH" ]; then
    INPUT_ARG=(--condition_image "$INPUT_PATH")
    INPUT_DISPLAY="$INPUT_PATH (single image)"
else
    INPUT_ARG=(--image_dir "$INPUT_PATH")
    INPUT_DISPLAY="$INPUT_PATH (directory)"
fi

echo "========================================"
echo "Reinforced Editing Trajectory"
echo "========================================"
echo "  GPU:         $CUDA_VISIBLE_DEVICES ($GPU_COUNT card(s))"
echo "  Input:       $INPUT_DISPLAY"
echo "  TRELLIS:     $TRELLIS_MODEL"
echo "  FlowEdit:    $FLOWEDIT_MODEL"
echo "  Steps:       $STEPS, n_max: $N_MAX"
echo "  Noise mode:  $NOISE_MODE"
echo "  CFG scale:   $CFG_SCALE"
echo "  Prompt:      $PROMPT"
echo "  Neg prompt:  $NEG_PROMPT"
echo "  Seed:        $SEED"
echo "  Save steps:  $SAVE_STEPS"
echo "  Output:      $OUTPUT_DIR"
echo "========================================"

COMMON_ARGS=(
    "${INPUT_ARG[@]}"
    --trellis_model "$TRELLIS_MODEL"
    --model_path "$FLOWEDIT_MODEL"
    --output_dir "$OUTPUT_DIR"
    --num_inference_steps $STEPS
    --n_max $N_MAX
    --cfg_scale $CFG_SCALE
    --noise_mode "$NOISE_MODE"
    --seed $SEED
    --save_steps "$SAVE_STEPS"
    --cell_size $CELL_SIZE
    --render_resolution $RENDER_RES
    --prompt "$PROMPT"
    --negative_prompt "$NEG_PROMPT"
)

if [ "$GPU_COUNT" -gt 1 ]; then
    echo ">>> Launching DDP with $GPU_COUNT GPUs"
    PYTHONPATH="$(pwd):$PYTHONPATH" \
    "$TORCHRUN" \
        --nproc_per_node=$GPU_COUNT \
        --master_port=$(shuf -i 29000-30000 -n 1) \
        scripts/debug/test_reinforced_editing_trajectory.py \
        "${COMMON_ARGS[@]}"
else
    echo ">>> Single GPU mode"
    PYTHONPATH="$(pwd):$PYTHONPATH" \
    "$PYTHON" scripts/debug/test_reinforced_editing_trajectory.py \
        "${COMMON_ARGS[@]}"
fi
