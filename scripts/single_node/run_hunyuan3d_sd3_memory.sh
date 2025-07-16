#!/bin/bash

# Hunyuan3D Training with SD3-style Memory Management
# 展示不同内存优化策略的效果

echo "🚀 Starting Hunyuan3D GRPO Training with SD3-style Memory Management..."
echo "📊 内存优化策略对比测试"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 数据目录检查
DATA_DIR="dataset/eval3d"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误：数据目录 $DATA_DIR 不存在"
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" | wc -l)
echo "✅ 找到 $IMAGE_COUNT 张图像文件"

# 创建输出目录
mkdir -p checkpoints/hunyuan3d_sd3_memory
mkdir -p logs

echo ""
echo "🎯 可选择的内存优化策略："
echo "1. aggressive  - 激进优化（VAE移到CPU，节省8-12GB显存）"
echo "2. moderate    - 中等优化（VAE在GPU但使用混合精度）"  
echo "3. conservative- 保守优化（SD3默认策略，性能最佳）"
echo ""

# 默认使用激进模式，用户可以通过参数修改
MEMORY_LEVEL=${1:-"aggressive"}
echo "📝 使用内存优化策略: $MEMORY_LEVEL"

case $MEMORY_LEVEL in
    "aggressive")
        echo "🔥 激进内存优化模式:"
        echo "  - VAE移动到CPU（节省8-12GB显存）"
        echo "  - 8bit Adam优化器"
        echo "  - 混合精度训练"
        echo "  - 适合：显存不足(<16GB)的用户"
        ;;
    "moderate")
        echo "⚖️  中等内存优化模式:"
        echo "  - VAE保持GPU但使用混合精度"
        echo "  - 平衡性能和内存使用"
        echo "  - 适合：16-24GB显存的用户"
        ;;
    "conservative")
        echo "🏛️  保守内存策略(SD3默认):"
        echo "  - VAE保持GPU FP32精度"
        echo "  - 性能最佳但内存占用最高"
        echo "  - 适合：24GB+显存的用户"
        ;;
    *)
        echo "❌ 无效的内存优化策略: $MEMORY_LEVEL"
        echo "请使用: aggressive, moderate, 或 conservative"
        exit 1
        ;;
esac

echo ""
echo "🚀 启动训练..."

# 启动训练 - 使用SD3风格的内存管理
accelerate launch \
    --config_file scripts/accelerate_configs/single_gpu.yaml \
    --main_process_port 29506 \
    scripts/train_hunyuan3d.py \
    --config config/hunyuan3d.py \
    --config.data_dir="$DATA_DIR" \
    --config.memory_optimization_level="$MEMORY_LEVEL" \
    --config.sample.input_batch_size=1 \
    --config.sample.num_batches_per_epoch=2 \
    --config.sample.num_meshes_per_image=2 \
    --config.train.batch_size=1 \
    --config.train.gradient_accumulation_steps=2 \
    --config.mixed_precision="bf16" \
    --config.num_epochs=20 \
    --config.save_freq=10

echo ""
echo "✅ 训练完成！"
echo "📊 内存优化效果:"
echo "  - aggressive: 最大内存节省，速度稍慢"
echo "  - moderate: 平衡内存和性能"
echo "  - conservative: 最佳性能，最高内存使用"
echo ""
echo "💡 提示: 可以通过以下命令尝试不同策略:"
echo "  bash scripts/single_node/run_hunyuan3d_sd3_memory.sh aggressive"
echo "  bash scripts/single_node/run_hunyuan3d_sd3_memory.sh moderate"
echo "  bash scripts/single_node/run_hunyuan3d_sd3_memory.sh conservative" 