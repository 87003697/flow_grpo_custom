#!/bin/bash
# TRELLIS Stage 2 FlowEdit + GAN 训练脚本

if [ -z "${RUN_NAME:-}" ]; then
    echo "ERROR: RUN_NAME not set. Usage: RUN_NAME=<exp_name> bash $0"
    exit 1
fi

if [ -d "logs/${RUN_NAME}" ]; then
    echo "ERROR: logs/${RUN_NAME} already exists. Use a different RUN_NAME to avoid overwriting."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29512}"

export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息（FlowEdit + GAN — Autograd）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.flowedit_autograd \
  --config=config/trellis_stage2_flowedit_gan.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
