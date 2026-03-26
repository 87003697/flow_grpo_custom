#!/bin/bash
# TRELLIS Dual-Stage (Sparse + Dense) Contrastive FlowEdit 训练脚本（单机版）
#
# GPU 分配策略（共享模式）：
# - 全部 N 张卡同时用于 Trellis 训练 (DDP) 和 Guidance (FlowEdit)
#
# 使用示例：
# - 单卡训练：export CUDA_VISIBLE_DEVICES=0
# - 2卡训练：export CUDA_VISIBLE_DEVICES=0,1

# 计算训练卡数
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "训练 + Guidance: cuda:0-$((GPU_COUNT-1))（共享同一设备）"
echo "========================================"

python -m accelerate.commands.launch \
    --num_processes=$TRAIN_GPU_COUNT \
    -m edit4shape.systems.trellis.entries.contrastive_dualstage_autograd \
    --config=config/trellis_stage1+2_contrastive.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME"
