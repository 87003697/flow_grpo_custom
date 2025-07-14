#!/bin/bash

# Hunyuan3D GRPO Training Script
# 本脚本提供不同GPU配置的训练方案

# 1 GPU - 通过gradient accumulation保持相同有效batch size，减少内存使用
# 策略：减少单次batch size，增加gradient_accumulation_steps保持相同的有效batch size
# 有效batch size = train_batch_size * gradient_accumulation_steps = 1 * 4 = 4
# 采样batch size = input_batch_size * num_meshes_per_image = 1 * 2 = 2 meshes per batch
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29504 scripts/train_hunyuan3d.py --config config/hunyuan3d.py --config.sample.input_batch_size=1 --config.sample.num_meshes_per_image=2 --config.sample.num_batches_per_epoch=2 --config.train.batch_size=1 --config.train.gradient_accumulation_steps=4

# 2 GPU - 中等规模配置
# 有效batch size = train_batch_size * gradient_accumulation_steps * num_gpus = 1 * 3 * 2 = 6
# export NCCL_P2P_Disable=1
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=2 --main_process_port 29504 scripts/train_hunyuan3d.py --config config/hunyuan3d.py --config.sample.input_batch_size=1 --config.sample.num_meshes_per_image=2 --config.sample.num_batches_per_epoch=3 --config.train.batch_size=1 --config.train.gradient_accumulation_steps=3

# 4 GPU - 大规模配置
# 有效batch size = train_batch_size * gradient_accumulation_steps * num_gpus = 1 * 2 * 4 = 8
# export NCCL_P2P_Disable=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29504 scripts/train_hunyuan3d.py --config config/hunyuan3d.py --config.sample.input_batch_size=1 --config.sample.num_meshes_per_image=3 --config.sample.num_batches_per_epoch=4 --config.train.batch_size=1 --config.train.gradient_accumulation_steps=2

# 8 GPU - 最大规模配置（需要足够内存）
# 有效batch size = train_batch_size * gradient_accumulation_steps * num_gpus = 1 * 2 * 8 = 16
export NCCL_P2P_Disable=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29504 scripts/train_hunyuan3d.py --config config/hunyuan3d.py --config.sample.input_batch_size=1 --config.sample.num_meshes_per_image=3 --config.sample.num_batches_per_epoch=5 --config.train.batch_size=1 --config.train.gradient_accumulation_steps=2

# 参数说明：
# --config.sample.input_batch_size: 每次处理的图像数量（建议1以节省内存）
# --config.sample.num_meshes_per_image: 每张图像生成的mesh候选数量
# --config.sample.num_batches_per_epoch: 每个epoch的采样批次数
# --config.train.batch_size: 训练时每个GPU的batch size
# --config.train.gradient_accumulation_steps: 梯度累积步数
# --config.num_epochs: 训练轮数（默认100）
# --config.save_freq: 检查点保存频率（默认20）
