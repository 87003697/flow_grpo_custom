#!/bin/bash
# Trellis Teacher/Student 对比评估脚本（DDP 多卡）
#
# 功能：加载 pretrained (teacher) 和 finetuned (student) 模型，
#       渲染多视角图像，使用 CLIP / DINO 计算与输入图像的相似度。
#
# GPU 需求：
#   - 不加载 Guidance 模型，所有卡均用于 DDP 评估
#   - 每卡显存 ~12-14 GB（pipeline + teacher + CLIP + DINO）
#
# 用法：
#   bash scripts/eval/eval_trellis_DDP.sh
#   bash scripts/eval/eval_trellis_DDP.sh <CKPT_INPUT>

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# CKPT_INPUT 支持三种输入：
# 1) "" -> pretrained baseline
# 2) ".../checkpoints/checkpoint_xxx" -> 单个 ckpt
# 3) ".../checkpoints" -> 遍历所有 checkpoint_*
CKPT_INPUT="${1:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EVAL_DIR="dataset/alphaimages_v2/test"
LOGDIR_ROOT="logs_for_eval"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 规范化输入路径（去尾斜杠）
CKPT_INPUT="${CKPT_INPUT%/}"

# 构建待评估 ckpt 列表
CKPT_LIST=()
if [ -z "$CKPT_INPUT" ]; then
    CKPT_LIST+=("")
elif [ "$(basename "$CKPT_INPUT")" = "checkpoints" ]; then
    mapfile -t CKPT_LIST < <(
        find "$CKPT_INPUT" -maxdepth 1 -mindepth 1 -type d -name "checkpoint_*" | sort -V
    )
    if [ "${#CKPT_LIST[@]}" -eq 0 ]; then
        echo "错误: 在 $CKPT_INPUT 下未找到 checkpoint_*"
        exit 1
    fi
else
    CKPT_LIST+=("$CKPT_INPUT")
fi

echo "========================================"
echo "DDP 评估 GPU 分配"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "评估进程数: $GPU_COUNT（无 Guidance，全部用于评估）"
echo "EVAL_DIR: $EVAL_DIR"
echo "LOGDIR_ROOT: $LOGDIR_ROOT"
echo "CKPT_INPUT: ${CKPT_INPUT:-（无，使用 pretrained）}"
echo "待评估数量: ${#CKPT_LIST[@]}"
echo "========================================"

for CKPT in "${CKPT_LIST[@]}"; do
    if [ -n "$CKPT" ]; then
        TRAIN_RUN=$(basename "$(dirname "$(dirname "$CKPT")")")
        CKPT_NAME=$(basename "$CKPT")
        RUN_NAME="eval_${TRAIN_RUN}_${CKPT_NAME}"
    else
        RUN_NAME="eval_pretrained_baseline"
    fi

    echo "------------ 开始评估 ------------"
    echo "RUN_NAME: $RUN_NAME"
    echo "CKPT: ${CKPT:-（无，使用 pretrained）}"
    echo "----------------------------------"

    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    python -m accelerate.commands.launch \
        --num_processes="$GPU_COUNT" \
        --multi_gpu \
        --main_process_port="$(shuf -i 29000-30000 -n 1)" \
        scripts/eval/eval_trellis.py \
        --config=config/trellis_stage2_distillation.py \
        --config.run_name="$RUN_NAME" \
        --config.checkpoint="$CKPT" \
        --config.logdir="$LOGDIR_ROOT" \
        --config.data.eval.dir="$EVAL_DIR"
done
