#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本（单机版）
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 使用示例：
# - 单卡训练：export CUDA_VISIBLE_DEVICES=0,1  (需要 2 张卡)
# - 2卡训练：export CUDA_VISIBLE_DEVICES=0,1,2,3  (需要 4 张卡)

# === 单卡训练 (需要 2 张卡) ===

# export CUDA_VISIBLE_DEVICES=0,1
# RUN_NAME="trellis-full_kl-uni-01_CSD_uni-05_lr-3e-5"

# export CUDA_VISIBLE_DEVICES=2,3
# RUN_NAME="trellis-full_kl-uni-001_CSD_uni-05_lr-3e-5"

# export CUDA_VISIBLE_DEVICES=4,5
# RUN_NAME="trellis-full_kl-uni-01_Edit-mean-05_lr-3e-5"

# export CUDA_VISIBLE_DEVICES=6,7
# RUN_NAME="trellis-full_kl-uni-001_Edit-mean-05_lr-3e-5"



# # === 2卡 DDP 训练 (需要 4 张卡) ===
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# RUN_NAME="trellis_stage2_distill_lr_3e-4_beta1_0.5_reg_none"



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
    -m edit4shape.systems.trellis \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME"
