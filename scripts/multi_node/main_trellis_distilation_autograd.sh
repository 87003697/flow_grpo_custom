#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本（三阶段 Autograd 显存压缩版）
#
# GPU 分配策略（共享模式）：
# - 全部 N 张卡同时用于 Trellis 训练 (DDP) 和 Guidance (FlowEdit)
# - 三阶段 Autograd 将显存峰值降为 max(guidance, decode_render)，可共享同一 GPU
# - 总需求：N 张卡（相比标准版的 2N 张卡减半）
#
# 使用示例：
# - 默认 4 张卡：CUDA_VISIBLE_DEVICES=4,5,6,7 ./main_trellis_distilation_autograd.sh
# - 自定义：CUDA_VISIBLE_DEVICES=0,1,2,3 ./main_trellis_distilation_autograd.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"   # 默认 4 张卡（训练 + Guidance 共享）
RUN_NAME="trellis_around_x0-1e-3_FlowEdit-dual-ada01-mts_cfg-4_steps-9_12_adan_lr-1e-4_eps-1e-4_autograd_4GPU"

# : "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"   # 默认 4 张卡（训练 + Guidance 共享）
# RUN_NAME="trellis_around_x0-1e-4_FlowEdit-dual-ada01-mts_cfg-4_steps-9_12_adan_lr-1e-4_eps-1e-4_autograd_4GPU"
# RUN_NAME="debug"

: "${MASTER_PORT:=29511}"

export CUDA_VISIBLE_DEVICES                # ★ 必须 export，否则子进程看到全部 GPU
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT  # ★ 全部用于训练，Guidance 自动共享同一设备

echo "========================================"
echo "GPU 分配信息（三阶段 Autograd 版 — 共享模式）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "训练 + Guidance: cuda:0-$((GPU_COUNT-1))（共享同一设备）"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.autograd \
  --config=config/trellis_stage2_distillation.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
