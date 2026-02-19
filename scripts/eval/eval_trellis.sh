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
#   bash scripts/eval/eval_trellis.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Finetuned checkpoint 路径（留空则 student=pretrained，用于 sanity check）
CKPT="logs/trellis_x0-01_FlowEdit-mts_cfg-4-rescale_steps-9_12_sgd_lr-5e-3_8GPU/checkpoints/checkpoint_0_2296"

# 自动从 checkpoint 路径提取 RUN_NAME: logs/{train_run_name}/checkpoints/{ckpt_name}
if [ -n "$CKPT" ]; then
    TRAIN_RUN=$(basename "$(dirname "$(dirname "$CKPT")")")
    CKPT_NAME=$(basename "$CKPT")
    RUN_NAME="eval_${TRAIN_RUN}_${CKPT_NAME}"
else
    RUN_NAME="eval_pretrained_baseline"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "========================================"
echo "DDP 评估 GPU 分配"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "评估进程数: $GPU_COUNT（无 Guidance，全部用于评估）"
echo "RUN_NAME: $RUN_NAME"
echo "CKPT: ${CKPT:-（无，使用 pretrained）}"
echo "========================================"

PYTHONPATH="$(pwd):$PYTHONPATH" \
python -m accelerate.commands.launch \
    --num_processes=$GPU_COUNT \
    --multi_gpu \
    --main_process_port=$(shuf -i 29000-30000 -n 1) \
    scripts/eval/eval_trellis.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME" \
    --config.checkpoint="$CKPT"
