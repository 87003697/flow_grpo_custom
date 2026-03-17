#!/bin/bash
# TRELLIS.2 Shape+Tex 双阶段联合蒸馏训练脚本 — 异步 Onestep 版（多机/多卡 DDP）
#
# 同时训练 Shape 和 Tex 两个 Flow Model：
# - Shape 阶段使用 Normal 渲染监督几何
# - Tex 阶段使用 PBR 渲染监督纹理
# 使用 Onestep Autograd + 异步流水线（S/T 交错）：
#   guidance GPU 与 train GPU 全程并行，吞吐量提升 ~30-50%。
#
# GPU 分配策略：
# - 前 N 张卡给 Trellis2 训练 (DDP)
# - 后 N 张卡给 Guidance (FlowEdit)
# - 总需求：2N 张卡
#
# ★ 建议 gradient_accumulation_steps ≥ 2 以获得异步并行收益。
#
# 使用示例：
# - 2卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3 bash main_trellis2_shape_tex_distilation_async.sh
# - 4卡 DDP 训练：CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash main_trellis2_shape_tex_distilation_async.sh

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"   # 默认 4 张卡（2 训练 + 2 Guidance）
# RUN_NAME="trellis2-shape_tex_autograd_async_debug"
RUN_NAME="trellis2_mesh-filled_tex_x1-1e0_FlowEdit_cfg-4_steps-9_12_pix-1_ssim-1_latent-0_adan_lr_1e-4_eps-1e-4_acc-1_8GPU"
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
  -m edit4shape.systems.trellis2.entries.shape_tex_onestep_autograd_async \
  --config=config/trellis2_shape_tex_distillation.py \
  --config.eval_only=false \
  --config.use_wandb=true \
  --config.run_name="$RUN_NAME" \
  "$@"
