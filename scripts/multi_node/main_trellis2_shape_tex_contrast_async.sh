#!/bin/bash
# TRELLIS.2 Shape+Tex Contrastive 训练脚本 — 异步流水线（多机/多卡 DDP）
#
# 使用 Contrastive（Pretrained Rollout + 对比学习）+ 异步 FlowEdit 流水线。
# - Shape 阶段使用 Normal 渲染
# - Tex 阶段使用 PBR 渲染
# Loss 在 velocity/x0 空间（3-arm contrastive），不在像素空间。
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis2 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# 使用示例：
# - 2卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3 bash main_trellis2_shape_tex_contrast_async.sh
# - 4卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash main_trellis2_shape_tex_contrast_async.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"   # 默认 4 张卡（2 训练 + 2 Guidance）
RUN_NAME="trellis2_shape_tex_contrastive_v-1e0_ada-false_FlowEdit_cfg-4_steps-9-12_promptv3_adan_lr-1e-4_4GPU"
: "${MASTER_PORT:=29511}"

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
  -m edit4shape.systems.trellis2.entries.shape_tex_contrastive_async \
  --config=config/trellis2_shape_tex_contrastive.py \
  --config.eval_only=false \
  --config.use_wandb=true \
  --config.run_name="$RUN_NAME" \
  "$@"
