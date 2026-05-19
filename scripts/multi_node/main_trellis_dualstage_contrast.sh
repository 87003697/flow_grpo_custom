#!/bin/bash
# TRELLIS Dual-Stage (Sparse + Dense) Contrastive FlowEdit 训练脚本（多机版）
#
# 训练流程：
#   Sparse Stage: Pretrained Rollout → 加噪 → Student velocity → Contrastive loss
#   Dense Stage:  复用 Sparse 的 c_src/c_tgt → Dense velocity → Contrastive loss
#
# GPU 分配策略（共享模式）：
# - 全部 N 张卡同时用于 Trellis 训练 (DDP) 和 Guidance (FlowEdit)
#
# 使用示例：
# - 默认 8 张卡：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./main_trellis_dualstage_contrast.sh
# - 自定义：CUDA_VISIBLE_DEVICES=4,5,6,7 ./main_trellis_dualstage_contrast.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
RUN_NAME="trellis_dual_contrast_on_v-1e0_wo-swap_ada-false_FlowEdit_cfg-4_steps-9-12_promptv3_adan_lr-1e-4_8GPU"
: "${MASTER_PORT:=29512}"

export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息（Dual-Stage Contrastive FlowEdit — Latent 空间对比学习）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "训练 + Guidance: cuda:0-$((GPU_COUNT-1))（共享同一设备）"
echo "渲染模式: GS Color"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.contrastive_dualstage_autograd \
  --config=config/trellis_stage1+2_contrastive.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
