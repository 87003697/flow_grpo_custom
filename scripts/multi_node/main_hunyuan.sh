#!/bin/bash

# 多GPU Hunyuan3D训练脚本
# 基于单GPU版本优化，支持多GPU并行训练

echo " Multi-GPU Hunyuan3D Training (多GPU并行训练)"

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用GPU 2和3

# # NCCL优化配置（解决卡住问题）
# export NCCL_TIMEOUT=1800           # 增加超时时间到30分钟
# export NCCL_DEBUG=INFO              # 开启调试信息
# export NCCL_IB_DISABLE=1            # 禁用InfiniBand（如果有网络问题）
# export NCCL_P2P_DISABLE=1           # 禁用P2P通信（降级到更稳定的通信方式）

DATA_DIR="dataset/eval3d"

# 计算实际可用的GPU数量
GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "🔧 检测到 $GPU_COUNT 个GPU: $CUDA_VISIBLE_DEVICES"

accelerate launch \
    --num_processes=$GPU_COUNT \
    --multi_gpu \
    --main_process_port=29505 \
    scripts/train_hunyuan3d.py \
    --config config/hunyuan3d.py \
    --config.data_dir="$DATA_DIR" \
    --config.sample.input_batch_size=1 \
    --config.sample.num_batches_per_epoch=1 \
    --config.sample.num_meshes_per_image=16 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=2 \
    --config.num_epochs=500

echo "✅ 多GPU训练完成! 📊 查看: tensorboard --logdir profiler_logs" 