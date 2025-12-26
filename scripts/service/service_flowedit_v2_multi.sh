#!/usr/bin/env bash
# FlowEdit V2 多 GPU 服务后台启动脚本
# 使用方法: bash scripts/service/service_flowedit_v2_multi.sh

set -euo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Qwen-Image-Edit 项目路径
QWEN_EDIT_ROOT="$PROJECT_ROOT/_reference_codes/Qwen-Image-Edit"

# 配置参数
GPUS=(${GPUS:-4 5 6 7})  # 使用的 GPU 列表，可通过环境变量覆盖
BASE_PORT="${BASE_PORT:-8001}"
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "FlowEdit V2 多 GPU 服务启动"
echo "=========================================="
echo "GPUs: ${GPUS[*]}"
echo "端口: $((BASE_PORT + GPUS[0])) - $((BASE_PORT + GPUS[-1]))"
echo "=========================================="

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate qwen-image-edit

# 启动各 GPU 实例
for gpu in "${GPUS[@]}"; do
    port=$((BASE_PORT + gpu))
    log_file="$LOG_DIR/flowedit_service_gpu${gpu}.log"
    
    CUDA_VISIBLE_DEVICES="$gpu" HOST="$HOST" PORT="$port" PYTHONPATH="$QWEN_EDIT_ROOT" \
        nohup python "$QWEN_EDIT_ROOT/src/flowedit/flowedit_v2_service.py" > "$log_file" 2>&1 &
    
    echo "✅ GPU $gpu 已启动 (PID: $!, 端口: $port)"
done

echo ""
echo "=========================================="
echo "查看日志:  tail -f $LOG_DIR/flowedit_service_gpu*.log"
echo "健康检查:  curl http://localhost:$((BASE_PORT + GPUS[0]))/health"
echo "停止服务:  pkill -f flowedit_v2_service.py"
echo "=========================================="