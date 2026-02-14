#!/bin/bash
# TRELLIS.2 Shape 阶段蒸馏 —— DEBUG 脚本（checkpoint 保存/恢复验证）
#
# 使用 dataset/debug_ddp 小数据集，3 epochs，eval freq = 1。
# 用于快速验证 checkpoint 保存/加载 是否正确。

: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"
RUN_NAME="trellis2-shape_debug_ckpt"

: "${MASTER_PORT:=29510}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$((GPU_COUNT / 2))

echo "========================================"
echo "DEBUG: Checkpoint 保存/恢复验证"
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
  -m edit4shape.systems.trellis2_shape_autograd \
  --config=config/trellis2_shape_distillation.py \
  --config.eval_only=false \
  --config.use_wandb=false \
  --config.run_name="$RUN_NAME" \
  --config.num_epochs=3 \
  --config.data.train.dir="dataset/debug_ddp/train" \
  --config.data.eval.dir="dataset/debug_ddp/test" \
  --config.freq.eval=1 \
  --config.freq.save.ckpt=1 \
  --config.freq.save.visual=1 \
  "$@"
