#!/bin/bash
# 多卡启动 TRELLIS Stage 2 蒸馏（基于 accelerate）

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"   # 覆盖以指定 GPU
: "${MASTER_PORT:=29510}"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

python -m accelerate.commands.launch \
  --num_processes=${GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis \
  --config=config/trellis_stage2_distillation.py \
  --config.eval_only=False \
  "$@"
