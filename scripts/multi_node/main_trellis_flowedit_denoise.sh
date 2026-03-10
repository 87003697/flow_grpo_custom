#!/bin/bash
# TRELLIS Stage 2 FlowEdit 训练脚本（Pretrained Rollout + Finetuned 单步去噪）
#
# 训练流程：
#   Pretrained Rollout (frozen) → clean z₀
#   → 加噪 z₀ → zₜ (随机时间步)
#   → Finetuned 单步去噪 → ẑ₀
#   → Decode + Render → comp_rgb
#   → 2D FlowEdit Guidance → loss → autograd backward
#
# ★ 与 hybrid_autograd 的区别：
#   - 不需要 VJP / proxy chain / no_sync hack
#   - 标准 autograd + accelerator.accumulate()
#   - 3D 端只做单步去噪，显存固定且低
#
# GPU 分配策略（共享模式）：
# - 全部 N 张卡同时用于 Trellis 训练 (DDP) 和 Guidance (FlowEdit)
#
# 使用示例：
# - 默认 4 张卡：CUDA_VISIBLE_DEVICES=4,5,6,7 ./main_trellis_flowedit_denoise.sh
# - 自定义：CUDA_VISIBLE_DEVICES=0,1,2,3 ./main_trellis_flowedit_denoise.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"   # 默认 4 张卡（训练 + Guidance 共享）
RUN_NAME="trellis_step_x1-1e-0_FlowEdit_cfg-4_steps-9_12_mse-1_adan_lr_1e-4_eps-1e-4_acc-1_8GPU"
: "${MASTER_PORT:=29512}"

export CUDA_VISIBLE_DEVICES                # ★ 必须 export，否则子进程看到全部 GPU
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_GPU_COUNT=$GPU_COUNT

echo "========================================"
echo "GPU 分配信息（FlowEdit Denoise — 单步去噪 Autograd）"
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
  -m edit4shape.systems.trellis.entries.flowedit_autograd \
  --config=config/trellis_stage2_flowedit_denoise.py \
  --config.eval_only=False \
  --config.use_wandb=True \
  --config.run_name="$RUN_NAME" \
  "$@"
