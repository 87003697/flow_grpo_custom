#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本 (Pipeline 并行版)
#
# 设备分配规则：每个 worker 占用 2 张连续的卡
# - LOCAL_RANK 0: train=cuda:0, guidance=cuda:1
# - LOCAL_RANK 1: train=cuda:2, guidance=cuda:3
# - LOCAL_RANK 2: train=cuda:4, guidance=cuda:5
# - LOCAL_RANK 3: train=cuda:6, guidance=cuda:7
#
# 8 卡 = 4 workers × 2 GPUs/worker
export CUDA_VISIBLE_DEVICES=2,3
RUN_NAME="trellis_stage2_distill_reg_none_latent_max_25_pp"

# export CUDA_VISIBLE_DEVICES=4,5
# RUN_NAME="trellis_stage2_distill_reg_none_latent_max_20_pp"

# export CUDA_VISIBLE_DEVICES=6,7
# RUN_NAME="trellis_stage2_distill_reg_none_latent_max_15_pp"

torchrun --standalone --nproc_per_node=1 \
    -m edit4shape.systems.trellis_pp \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=false \
    --config.run_name="$RUN_NAME"
