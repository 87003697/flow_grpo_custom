#!/bin/bash
# TRELLIS模型运行脚本
# 设置正确的环境变量以避免flash_attn和网络问题

echo "🚀 设置TRELLIS运行环境..."

# 设置注意力后端为xformers (避免flash_attn C++编译问题)
export ATTN_BACKEND=xformers

# 设置离线模式 (使用本地下载的模型)
export HF_HUB_OFFLINE=1

# 设置CUDA设备 (使用第一块GPU)
export CUDA_VISIBLE_DEVICES=0

echo "✅ 环境变量设置完成:"
echo "   ATTN_BACKEND=$ATTN_BACKEND"
echo "   HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 执行传入的命令
echo "🔄 执行命令: $@"
exec "$@" 