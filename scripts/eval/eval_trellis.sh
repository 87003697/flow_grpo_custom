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
#   bash scripts/eval/eval_trellis.sh <checkpoints_dir 或 具体 checkpoint 路径>

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Finetuned checkpoint 路径（留空则 student=pretrained，用于 sanity check）
# 支持传入 checkpoints 目录（遍历所有 checkpoint）或具体 checkpoint 路径
INPUT="${1:-logs_for_eval/trellis_x0-01_FlowEdit-ada01-mts_cfg-4_steps-9_12_sgd_lr-1e-3_8GPU/checkpoints/checkpoint_0_574}"

# 如果传入的是 checkpoints 目录，收集所有 checkpoint；否则单个
if [ -d "$INPUT" ]; then
    CKPT_LIST=($(ls -1d "$INPUT"/checkpoint_* 2>/dev/null | sort -t_ -k3 -n))
    if [ ${#CKPT_LIST[@]} -eq 0 ]; then
        echo "错误：目录 $INPUT 中未找到 checkpoint_* 子目录"
        exit 1
    fi
    echo "发现 ${#CKPT_LIST[@]} 个 checkpoint，将依次评估"
else
    CKPT_LIST=("$INPUT")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EVAL_DIR="dataset/alphaimages_v3/test"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

for CKPT in "${CKPT_LIST[@]}"; do
    # checkpoint 路径: {logdir}/{run_name}/checkpoints/{ckpt_name}
    CKPT_NAME=$(basename "$CKPT")
    EXP_ROOT=$(dirname "$(dirname "$CKPT")")   # 实验根目录
    RUN_NAME=$(basename "$EXP_ROOT")            # 实验名
    LOGDIR=$(dirname "$EXP_ROOT")               # logdir（与 checkpoints 同级）

    echo "========================================"
    echo "DDP 评估 GPU 分配"
    echo "========================================"
    echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
    echo "评估进程数: $GPU_COUNT（无 Guidance，全部用于评估）"
    echo "RUN_NAME: $RUN_NAME"
    echo "LOGDIR: $LOGDIR"
    echo "CKPT: ${CKPT:-（无，使用 pretrained）}"
    echo "EVAL_DIR: $EVAL_DIR"
    echo "========================================"

    PYTHONPATH="$(pwd):$PYTHONPATH" \
    python -m accelerate.commands.launch \
        --num_processes=$GPU_COUNT \
        --multi_gpu \
        --main_process_port=$(shuf -i 29000-30000 -n 1) \
        scripts/eval/eval_trellis.py \
        --config=config/trellis_stage2_distillation.py \
        --config.run_name="$RUN_NAME" \
        --config.logdir="$LOGDIR" \
        --config.checkpoint="$CKPT" \
        --config.data.eval.dir="$EVAL_DIR"
done
