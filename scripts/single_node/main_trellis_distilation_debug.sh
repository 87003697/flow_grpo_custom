#!/bin/bash
# TRELLIS Stage 2 Hybrid 蒸馏训练脚本（单 GPU 版 — Autograd 共享模式）
#
# 双路渲染：Mesh Normal + GS Color 同时 guidance，梯度在 proxy 上累加。
#
# GPU 分配策略（共享模式）：
# - 单张卡同时用于 Trellis 训练和 Guidance (FlowEdit)
# - 三阶段 Autograd 将显存峰值降为 max(guidance, decode_render)，可共享同一 GPU
# - 双路渲染不增加显存峰值（每路 P2c 结束即释放）
# - 总需求：1 张卡
#
# 使用示例：
# - 默认 GPU 0：./main_trellis_distilation_debug.sh
# - 自定义：CUDA_VISIBLE_DEVICES=3 ./main_trellis_distilation_debug.sh

: "${CUDA_VISIBLE_DEVICES:=0}"   # 默认 1 张卡（训练 + Guidance 共享）
export CUDA_VISIBLE_DEVICES                # ★ 必须 export，否则子进程看到全部 GPU

RUN_NAME="trellis_hybrid_debug_single_gpu"

echo "========================================"
echo "GPU 分配信息（Hybrid 双路渲染 Autograd 版 — 单 GPU）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES"
echo "训练进程数: 1"
echo "训练 + Guidance: cuda:0（共享同一设备）"
echo "渲染模式: Hybrid（Mesh Normal + GS Color）"
echo "========================================"

python -m accelerate.commands.launch \
  --num_processes=1 \
  -m edit4shape.systems.trellis.entries.hybrid_autograd \
  --config=config/trellis_stage2_hybrid_distillation.py \
  --config.eval_only=False \
  --config.use_wandb=False \
  --config.run_name="$RUN_NAME" \
  "$@"
