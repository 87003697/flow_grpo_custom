#!/bin/bash
# 评估 Guidance 前后 CLIP / DINO 与输入图像的相似度变化
#
# 路径由 --config.run_name 控制，与训练脚本一致：
#   输出目录: {logdir}/{run_name}/eval_metrics/
#
# 用法：
#   bash scripts/eval/eval_mesh_scorer_eval3d.sh

export CUDA_VISIBLE_DEVICES=2,3
RUN_NAME="trellis_stage2_distill"

PYTHONPATH="$(pwd):$PYTHONPATH" \
python scripts/eval/eval_guidance_metrics.py \
    --config=config/trellis_stage2_distillation.py \
    --config.run_name="$RUN_NAME"
    # --config.checkpoint=path/to/checkpoint   # 如需加载特定 checkpoint，取消注释并修改路径
