#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本（标准版）
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 使用示例：
# - 2卡训练：CUDA_VISIBLE_DEVICES=0,1,2,3 ./main_trellis_distilation.sh
# - 4卡训练：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./main_trellis_distilation.sh

RUN_NAME="trellis_stage2_distill-11_reg_kl-1-uni_max-15_8GPU"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"   # 默认 4 张卡（2 训练 + 2 Guidance）
: "${MASTER_PORT:=29510}"

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
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --mixed_precision=bf16 \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis \
  --config=config/trellis_stage2_distillation.py \
  --config.eval_only=False \
  --config.run_name="$RUN_NAME" \
  "$@"
