#!/bin/bash

# 基于torch.profiler分析的终极内存优化版本
# 发现：训练前向传播从7.77GB→44.33GB，需要激进优化

echo " Memory-Optimized Hunyuan3D Training (基于profiling分析)"

export CUDA_VISIBLE_DEVICES=1

# export USE_SAGEATTN=1        # 3D DiT使用

DATA_DIR="dataset/eval3d"

accelerate launch \
    --config_file scripts/accelerate_configs/single_gpu.yaml \
    --num_processes=0 \
    --main_process_port=29505 \
    scripts/train_hunyuan3d.py \
    --config config/hunyuan3d.py \
    --config.data_dir="$DATA_DIR" \
    --config.sample.input_batch_size=1 \
    --config.sample.num_batches_per_epoch=2 \
    --config.sample.num_meshes_per_image=2 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=2 \
    --config.num_epochs=500

echo "✅ 内存优化完成! 📊 查看: tensorboard --logdir profiler_logs"
