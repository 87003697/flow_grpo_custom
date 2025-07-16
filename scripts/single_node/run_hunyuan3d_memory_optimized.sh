#!/bin/bash

# 基于torch.profiler分析结果的内存极度优化版本
# 问题：训练前向传播从7.77GB跳到44.33GB，需要激进的内存优化

echo "🚀 Starting Memory-Optimized Hunyuan3D GRPO Training..."
echo "📊 基于profiling分析的优化:"
echo "  - 发现：训练前向传播需要36GB+内存"
echo "  - 优化：启用所有可能的内存优化技术"
echo "  - 目标：将训练内存控制在可用范围内"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 🔧 关键：基于profiling结果的激进内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=1

# 🚀 更激进的内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.6"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

# 数据目录检查
DATA_DIR="dataset/eval3d"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误：数据目录 $DATA_DIR 不存在"
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" | wc -l)
echo "✅ 找到 $IMAGE_COUNT 张图像文件"

# 创建输出目录
mkdir -p checkpoints/hunyuan3d_grpo_memory_optimized
mkdir -p logs

# 🔧 终极内存优化配置
accelerate launch \
    --config_file scripts/accelerate_configs/single_gpu.yaml \
    --main_process_port 29505 \
    scripts/train_hunyuan3d.py \
    --config config/hunyuan3d.py \
    --config.data_dir="$DATA_DIR" \
    --config.sample.input_batch_size=1 \
    --config.sample.num_batches_per_epoch=1 \
    --config.sample.num_meshes_per_image=2 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=1 \
    --config.mixed_precision="bf16" \
    --config.num_epochs=10 \
    --config.save_freq=5

echo "✅ 内存优化训练完成！"
echo "📊 profiling结果: 查看 profiler_logs/ 目录"
echo "📈 TensorBoard: tensorboard --logdir profiler_logs" 