#!/bin/bash
# TRELLIS Stage 2 蒸馏训练脚本

python -m edit4shape.systems.trellis \
    --config=config/trellis_stage2_distillation.py \
    --config.eval_only=false
