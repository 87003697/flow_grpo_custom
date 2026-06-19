#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本（标准版）
#
# GPU 分配策略（H200 共享模式）：
# - 所有卡用于 DDP 训练
# - Guidance 模型自动共享同卡（compute_guidance_device 自动回退）
#
# 使用示例：
# - 单卡调试：CUDA_VISIBLE_DEVICES=0 bash main_trellis_distilation.sh
# - 8卡训练：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash main_trellis_distilation.sh

if [ -z "${RUN_NAME:-}" ]; then
    echo "ERROR: RUN_NAME not set. Usage: RUN_NAME=<exp_name> bash $0"
    exit 1
fi

if [ -d "logs/${RUN_NAME}" ]; then
    echo "ERROR: logs/${RUN_NAME} already exists. Use a different RUN_NAME to avoid overwriting."
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"   # 默认 8 卡（全部训练，Guidance 共享同卡）
: "${MASTER_PORT:=29510}"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "模式: H200 共享（训练 + Guidance 同卡）"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.standard \
  --config=config/trellis_stage2_distillation.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
