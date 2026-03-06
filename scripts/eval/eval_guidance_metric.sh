#!/bin/bash
# 单卡评估脚本：评估 Guidance 前后 CLIP / DINO / SilhouetteIoU 指标
# 评估 + Guidance 共享同一设备

export CUDA_VISIBLE_DEVICES=2

# === 可调参数 ===
STEPS=12
N_MAX=9
CFG_SCALE=4
PROMPT="Rotate the camera."

RUN_NAME="eval_metrics_full-aligned_steps-${N_MAX}-${STEPS}_cfg-${CFG_SCALE}_prompt_v12"

# 如需加载特定 checkpoint，取消注释并修改路径：
# CKPT="logs/trellis_FlowEdit-mts_sgd_lr-1e-3/checkpoints/checkpoint_0_2296"
# --config.checkpoint="$CKPT" \

PYTHONPATH="$(pwd):$PYTHONPATH" \
python scripts/eval/eval_guidance_metrics.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME" \
    --config.guidance.flowedit.steps=$STEPS \
    --config.guidance.flowedit.n_max=$N_MAX \
    --config.train.guidance.true_cfg_scale_tgt=$CFG_SCALE \
    --config.train.guidance.target_prompt="$PROMPT"
