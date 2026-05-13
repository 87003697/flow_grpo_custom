#!/bin/bash
# Dense ODE vs FlowEdit 对比测试脚本（Stage 1 分叉）
# 使用方法: bash scripts/debug/test_trellis_dense_ODE-vs-FlowEdit.sh

eval "$(conda shell.bash hook)"
conda activate grpo3d_trellis

python scripts/debug/test_trellis_dense_ODE-vs-FlowEdit.py \
  --model_path pretrained_weights/TRELLIS-image-large \
  --image dataset/eval3d_hunyuan3d/images/004.png \
  --out outputs/test_runs/dense_ode_vs_flowedit \
  --ss_steps 12 \
  --slat_steps 12 \
  --guidance 3.0 \
  --seed 777 \
  --num_views 4 \
  --render_resolution 512 \
  --fe_steps 12 \
  --fe_n_max 8 \
  --fe_cfg_tgt 3.0 \
  --fe_cfg_src -3.0
