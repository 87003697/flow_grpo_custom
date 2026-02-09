#!/bin/bash
# 旧脚本，请使用 eval_mesh_scorer_eval3d.sh
# 保留仅作向后兼容

export CUDA_VISIBLE_DEVICES=2,3
RUN_NAME="trellis_stage2_distill"

PYTHONPATH="$(pwd):$PYTHONPATH" \
python scripts/eval/eval_guidance_metrics.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME"
    # --config.checkpoint=path/to/checkpoint
