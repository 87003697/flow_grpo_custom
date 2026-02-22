#!/bin/bash
# TRELLIS.2 Shape 阶段蒸馏训练脚本（多机/多卡 DDP 版）
#
# 仅训练 Shape Flow Model，使用 Normal 渲染监督几何。
# 使用三阶段 Autograd + 异步 Guidance 流水线策略（显存 O(1)）。
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis2 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 使用示例：
# - 2卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3 bash main_trellis2_shape_distilation.sh
# - 4卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash main_trellis2_shape_distilation.sh

# : "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"   # 默认 4 张卡（2 训练 + 2 Guidance）
# RUN_NAME="trellis2-shape_debug_async"
# RUN_NAME="trellis2-shape_around_x0-01_FlowEdit-ada01_mts_cfg-4_steps-9_12_sgd_lr-1e-3_async_4GPU"

: "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"   # 默认 4 张卡（2 训练 + 2 Guidance）
RUN_NAME="trellis2-shape_debug_async"
# RUN_NAME="trellis2-shape_mesh-peeled_x0-01_FlowEdit-ada01_mts_cfg-4_steps-9_12_sgd_lr-1e-3_async"


: "${MASTER_PORT:=29510}"

export CUDA_VISIBLE_DEVICES
# 避免 PyTorch 内存碎片化导致 OOM（释放 reserved-but-unallocated 内存）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  -m edit4shape.systems.trellis2_shape_autograd_async \
  --config=config/trellis2_shape_distillation.py \
  --config.eval_only=false \
  --config.use_wandb=false \
  --config.run_name="$RUN_NAME" \
  "$@"
