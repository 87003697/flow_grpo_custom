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

torchrun --standalone --nproc_per_node=4 \
    -m edit4shape.systems.trellis_pp \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=False
