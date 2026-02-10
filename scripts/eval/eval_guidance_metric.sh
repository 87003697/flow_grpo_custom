#!/bin/bash
# 旧脚本，请使用 eval_mesh_scorer_eval3d.sh
# 保留仅作向后兼容

export CUDA_VISIBLE_DEVICES=2,3
RUN_NAME="trellis_stage2_distill"

# 如需加载特定 checkpoint，取消注释并修改路径：
CKPT="logs/trellis_FlowEdit-mts_sgd_lr-1e-3/checkpoints/checkpoint_0_2296"

PYTHONPATH="$(pwd):$PYTHONPATH" \
python scripts/eval/eval_guidance_metrics.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME" \
    --config.checkpoint="$CKPT" \
    --config.guidance.flowedit.use_mts_sampling=false
