#!/usr/bin/env bash
# FlowEdit V2 单 GPU 服务后台启动脚本
# 使用方法: bash scripts/service/service_flowedit_v2.sh

set -euo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Qwen-Image-Edit 项目路径
QWEN_EDIT_ROOT="$PROJECT_ROOT/_reference_codes/Qwen-Image-Edit"

# 配置参数
export GPU="${GPU:-0}"
export PORT="${PORT:-8001}"
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FlowEdit V2 服务启动"
echo "=========================================="
echo "GPU: $GPU"
echo "端口: $PORT"
echo "=========================================="

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

# 启动服务
log_file="$LOG_DIR/flowedit_service_gpu${GPU}.log"

CUDA_VISIBLE_DEVICES="$GPU" HOST="$HOST" PORT="$PORT" PYTHONPATH="$QWEN_EDIT_ROOT" \
    nohup python "$QWEN_EDIT_ROOT/src/flowedit/flowedit_v2_service.py" > "$log_file" 2>&1 &

echo "✅ GPU $GPU 已启动 (PID: $!, 端口: $PORT)"
echo ""
echo "=========================================="
echo "查看日志:  tail -f $log_file"
echo "健康检查:  curl http://localhost:$PORT/health"
echo "停止服务:  pkill -f flowedit_v2_service.py"
echo "=========================================="