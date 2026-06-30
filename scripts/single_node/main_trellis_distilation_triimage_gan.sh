#!/bin/bash
# TRELLIS Stage 2 蒸馏 + 三图 DINO GAN 训练脚本（单机版）

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$((GPU_COUNT / 2))

echo "========================================"
echo "GPU 分配信息（蒸馏 + 三图 GAN）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "========================================"

python -m accelerate.commands.launch \
    --num_processes=$TRAIN_GPU_COUNT \
    -m edit4shape.systems.trellis.entries.standard \
    --config=config/trellis_stage2_distillation_triimage_gan.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME" \
    "$@"
