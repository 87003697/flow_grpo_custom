#!/bin/bash

# 简化版Hunyuan3D Single GPU Training Script
# 适用于单GPU环境的简化版快速训练脚本

echo "🚀 Starting Simplified Hunyuan3D GRPO Training on Single GPU..."
echo "📊 Configuration:"
echo "  - GPU: 1"
echo "  - Batch Size: 1"
echo "  - Gradient Accumulation: 2"
echo "  - Effective Batch Size: 2"
echo "  - Images per Batch: 1"
echo "  - Batches per Epoch: 2"
echo "  - 简化版: 内存优化 + 代码简洁"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 数据目录检查
DATA_DIR="dataset/eval3d"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误：数据目录 $DATA_DIR 不存在"
    echo "请运行以下命令准备数据："
    echo "  mkdir -p $DATA_DIR"
    echo "  # 然后将你的图像文件放入该目录"
    exit 1
fi

# 检查图像文件
IMAGE_COUNT=$(find "$DATA_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "❌ 错误：在 $DATA_DIR 中没有找到图像文件"
    echo "请确保目录中包含 .jpg, .jpeg, .png 或 .bmp 格式的图像文件"
    exit 1
fi

echo "✅ 找到 $IMAGE_COUNT 张图像文件"

# 创建输出目录
mkdir -p checkpoints/hunyuan3d_grpo_simplified
mkdir -p logs

# 启动训练 - 使用accelerate但配置更简单
accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=1 \
    --main_process_port 29505 \
    scripts/train_hunyuan3d_simplified.py \
    --config config/hunyuan3d_simplified.py \
    --config.data_dir="$DATA_DIR" \
    --config.sample.input_batch_size=1 \
    --config.sample.num_batches_per_epoch=2 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=2 \
    --config.num_epochs=50 \
    --config.save_freq=10 \
    --config.eval_freq=25

echo "✅ 简化版训练完成！"
echo "📁 检查点保存在: checkpoints/hunyuan3d_grpo_simplified/"
echo "📊 相比原版的优势："
echo "  - 内存使用更少"
echo "  - 代码更简洁"
echo "  - 错误处理更直接"
echo "  - 启动更快速" 