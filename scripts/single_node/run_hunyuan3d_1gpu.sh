#!/bin/bash

# Hunyuan3D Single GPU Training Script
# 适用于单GPU环境的快速训练脚本

echo "🚀 Starting Hunyuan3D GRPO Training on Single GPU..."
echo "📊 Configuration:"
echo "  - GPU: 1"
echo "  - Batch Size: 1"
echo "  - Gradient Accumulation: 4"
echo "  - Effective Batch Size: 4"
echo "  - Meshes per Image: 2"
echo "  - Batches per Epoch: 2"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4

# 启动训练
accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=1 \
    --main_process_port 29504 \
    scripts/train_hunyuan3d.py \
    --config config/hunyuan3d.py \
    --config.sample.input_batch_size=1 \
    --config.sample.num_meshes_per_image=2 \
    --config.sample.num_batches_per_epoch=2 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=4 \
    --config.num_epochs=100 \
    --config.save_freq=10 \
    --config.eval_freq=50 