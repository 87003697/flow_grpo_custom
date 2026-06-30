#!/bin/bash
# TRELLIS Stage 2 FlowEdit Denoise + 三图 CFGDiff Latent GAN 训练脚本

if [ -z "${RUN_NAME:-}" ]; then
    echo "ERROR: RUN_NAME not set. Usage: RUN_NAME=<exp_name> bash $0"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${MASTER_PORT:=29512}"

export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息（FlowEdit Denoise + 三图 Latent GAN）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "========================================"

MULTI_GPU_FLAG=""
if [ "$TRAIN_GPU_COUNT" -gt 1 ]; then
  MULTI_GPU_FLAG="--multi_gpu"
fi

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  ${MULTI_GPU_FLAG} \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.flowedit_autograd \
  --config=config/trellis_stage2_flowedit_denoise_triimage_latent_gan.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
