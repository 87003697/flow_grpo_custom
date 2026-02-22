#!/bin/bash
# DDP 多卡评估脚本：评估 Guidance 前后 CLIP / DINO / SilhouetteIoU 指标
#
# GPU 分配策略（与训练脚本 main_trellis_distilation.sh 一致）：
# - 前 N 张卡给评估 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 用法：
#   conda activate grpo3d_trellis
#   bash scripts/eval/eval_guidance_metric_DDP.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_NAME="eval_metrics_full-aligned_steps-6-8_cfg-4"

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# RUN_NAME="eval_metrics_full-aligned_steps-9-12_cfg-4"


# 如需加载特定 checkpoint，取消注释并修改路径：
#   --config.checkpoint=path/to/checkpoint

# 计算评估卡数（总卡数 / 2）
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
EVAL_GPU_COUNT=$((GPU_COUNT / 2))

echo "========================================"
echo "DDP 评估 GPU 分配"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "评估进程数: $EVAL_GPU_COUNT"
echo "评估 GPU: cuda:0-$((EVAL_GPU_COUNT-1))"
echo "Guidance GPU: cuda:$EVAL_GPU_COUNT-$((GPU_COUNT-1))"
echo "========================================"

PYTHONPATH="$(pwd):$PYTHONPATH" \
python -m accelerate.commands.launch \
    --num_processes=$EVAL_GPU_COUNT \
    --main_process_port=$(shuf -i 29000-30000 -n 1) \
    scripts/eval/eval_guidance_metrics.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME" \
    --config.guidance.flowedit.use_mts_sampling=false
