#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本（DEBUG 版 —— 测试 DDP checkpoint save/load）
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 当前配置：4,5,6,7 → 2 卡 DDP 训练 + 2 卡 Guidance

export CUDA_VISIBLE_DEVICES=4,5,6,7
RUN_NAME="debug_ddp_ckpt_test"

# 计算训练卡数（总卡数 / 2）
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$((GPU_COUNT / 2))

echo "========================================"
echo "GPU 分配信息"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "训练 GPU: cuda:0-$((TRAIN_GPU_COUNT-1))"
echo "Guidance GPU: cuda:$TRAIN_GPU_COUNT-$((GPU_COUNT-1))"
echo "========================================"

python -m accelerate.commands.launch \
    --num_processes=$TRAIN_GPU_COUNT \
    --multi_gpu \
    -m edit4shape.systems.trellis \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME" \
    --config.data.train.dir="dataset/debug_ddp/train" \
    --config.data.eval.dir="dataset/debug_ddp/test" \
    --config.num_epochs=2 \
    --config.freq.save.ckpt=1 \
    --config.freq.eval=0 \
    --config.guidance.flowedit.pipeline_type=simple
