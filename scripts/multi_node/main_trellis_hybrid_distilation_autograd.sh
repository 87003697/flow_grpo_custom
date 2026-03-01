#!/bin/bash
# TRELLIS Stage 2 Hybrid 蒸馏训练脚本（双路渲染 Autograd 版）
#
# 双路渲染：Mesh Normal + GS Color 同时 guidance，梯度在 proxy 上累加。
#
# GPU 分配策略（共享模式）：
# - 全部 N 张卡同时用于 Trellis 训练 (DDP) 和 Guidance (FlowEdit)
# - 三阶段 Autograd 将显存峰值降为 max(guidance, decode_render)，可共享同一 GPU
# - 双路渲染不增加显存峰值（每路 P2c 结束即释放）
# - 总需求：N 张卡
#
# 使用示例：
# - 默认 4 张卡：CUDA_VISIBLE_DEVICES=4,5,6,7 ./main_trellis_hybrid_distilation_autograd.sh
# - 自定义：CUDA_VISIBLE_DEVICES=0,1,2,3 ./main_trellis_hybrid_distilation_autograd.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"   # 默认 4 张卡（训练 + Guidance 共享）
RUN_NAME="trellis_hybrid_v-1e-4_FlowEdit_ada-1e-1_cfg-4_steps-9_12_x0-neg-src_adan_lr-1e-4_eps-1e-4"
: "${MASTER_PORT:=29510}"

# : "${CUDA_VISIBLE_DEVICES:=4,5,6,7}"   # 默认 4 张卡（训练 + Guidance 共享）
# RUN_NAME=trellis_hybrid_v-0_FlowEdit_ada-1e-1_cfg-4_steps-9_12_x0-neg-src_adan_lr-1e-4_eps-1e-4
# : "${MASTER_PORT:=29511}"

export CUDA_VISIBLE_DEVICES                # ★ 必须 export，否则子进程看到全部 GPU
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT  # ★ 全部用于训练，Guidance 自动共享同一设备

echo "========================================"
echo "GPU 分配信息（Hybrid 双路渲染 Autograd 版 — 共享模式）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "训练进程数: $TRAIN_GPU_COUNT"
echo "训练 + Guidance: cuda:0-$((GPU_COUNT-1))（共享同一设备）"
echo "渲染模式: Hybrid（Mesh Normal + GS Color）"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=${TRAIN_GPU_COUNT} \
  --multi_gpu \
  --main_process_port=${MASTER_PORT} \
  -m edit4shape.systems.trellis.entries.hybrid_autograd \
  --config=config/trellis_stage2_hybrid_distillation.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
