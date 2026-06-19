#!/bin/bash
# ============================================================================
# Koala 一键提交脚本 — flow_grpo_custom_v2
# ============================================================================
# 用法:
#   bash scripts/submit_koala.sh                # debug pod (1 GPU, SSH)
#   bash scripts/submit_koala.sh --train        # 正式训练 (8 GPU, s3-log)
#
# 工作流：
#   1. 先提交 debug pod，SSH 进去运行 `. scripts/setup_koala.sh --download` 首次初始化
#   2. 验证单卡能跑通
#   3. 确认后 `bash scripts/submit_koala.sh --train` 提交 8 卡 normal pod
# ============================================================================
set -euo pipefail

S3="s3://arcwm-code-us-west-2/$USER/flow_grpo_custom_v2"

echo "Syncing code to S3..."
s5cmd sync \
    --exclude '.git/*' --exclude '.venv/*' --exclude '*/__pycache__/*' \
    --exclude 'graphify-out/*' --exclude '.agents/*' --exclude 'logs/*' \
    --exclude 'wandb/*' --exclude 'pretrained_weights' \
    . "$S3/"
echo "  Done"

if [[ "${1:-}" == "--train" ]]; then
    shift
    echo "Submitting NORMAL pod (8 GPU, s3-log)..."
    LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8 koala submit \
        --code "$S3:/data/work/run_codes" \
        -m normal -g 8 --s3-log \
        -j "trellis-distill" \
        -c "cd /data/work/run_codes && . scripts/setup_koala.sh --fast && RUN_NAME=trellis_stage2_distill bash scripts/multi_node/main_trellis_distilation.sh" \
        "$@"
else
    echo "Submitting DEBUG pod (1 GPU, SSH)..."
    LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8 koala submit \
        --code "$S3:/data/work/run_codes" \
        "$@"
    echo ""
    echo "Next steps:"
    echo "  1. koala ssh --pod <pod> --connect"
    echo "  2. cd /data/work/run_codes"
    echo "  3. . scripts/setup_koala.sh --download   # 首次"
    echo "  4. . scripts/setup_koala.sh --fast       # 日常"
    echo "  5. bash scripts/multi_node/main_trellis_distilation.sh  # 单卡试跑"
fi
