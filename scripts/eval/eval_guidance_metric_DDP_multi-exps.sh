#!/bin/bash
# DDP 多卡评估脚本：评估 Guidance 前后 CLIP / DINO / SilhouetteIoU 指标
#
# GPU 分配策略（共享模式，与 Autograd 训练脚本一致）：
# - 全部 N 张卡同时用于评估 (DDP) 和 Guidance (FlowEdit)
# - Guidance 自动回退到与评估共享同一设备
#
# 用法：
#   conda activate grpo3d_trellis
#   bash scripts/eval/eval_guidance_metric_DDP_multi-exps.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# === 可调参数（多实验串行）===
# 固定参数（不随实验变化）
STEPS=12
N_MAX=9
PROMPT="Rotate the camera. Consistent concept design."
RUN_NAME_PREFIX="eval_metrics_full-rndm_steps-${N_MAX}-${STEPS}"

# 每条实验配置格式：
#   CFG_SCALE
#
# 说明：
# - 多个实验会按列表顺序串行执行（前一个结束后才跑下一个）
# - RUN_NAME 必须唯一，避免日志目录覆盖
EXPERIMENTS=(
  "4"
  "8"
  "12"
)

# 如需加载特定 checkpoint，取消注释并修改路径：
#   --config.checkpoint=path/to/checkpoint

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "========================================"
echo "DDP 评估 GPU 分配（共享模式）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "评估进程数: $GPU_COUNT"
echo "评估 + Guidance: cuda:0-$((GPU_COUNT-1))（共享同一设备）"
echo "实验总数: ${#EXPERIMENTS[@]}（串行执行）"
echo "========================================"

for idx in "${!EXPERIMENTS[@]}"; do
    CFG_SCALE="${EXPERIMENTS[$idx]}"
    RUN_NAME="${RUN_NAME_PREFIX}_cfg-${CFG_SCALE}_prompt_v15"

    echo
    echo ">>> [实验 $((idx + 1))/${#EXPERIMENTS[@]}] 开始"
    echo "RUN_NAME   : $RUN_NAME"
    echo "STEPS      : $STEPS"
    echo "N_MAX      : $N_MAX"
    echo "CFG_SCALE  : $CFG_SCALE"
    echo "PROMPT     : $PROMPT"
    echo "----------------------------------------"

    PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" \
    python -m accelerate.commands.launch \
        --num_processes=$GPU_COUNT \
        --main_process_port=$(shuf -i 29000-30000 -n 1) \
        scripts/eval/eval_guidance_metrics.py \
        --config=config/trellis_stage2_distillation.py \
        --config.run_name="$RUN_NAME" \
        --config.guidance.flowedit.steps=$STEPS \
        --config.guidance.flowedit.n_max=$N_MAX \
        --config.train.guidance.true_cfg_scale_tgt=$CFG_SCALE \
        --config.train.guidance.target_prompt="$PROMPT"

    echo "<<< [实验 $((idx + 1))/${#EXPERIMENTS[@]}] 完成: $RUN_NAME"
done

echo
echo "========================================"
echo "全部实验执行完成。"
echo "========================================"
