#!/bin/bash
# SDE vs ODE 对比测试脚本

python scripts/debug/test_trellis_SDE-vs-ODE.py \
  --model_path pretrained_weights/TRELLIS-image-large \
  --image dataset/eval3d_hunyuan3d/images/004.png \
  --out outputs/test_runs/sde_vs_ode \
  --steps 50 \
  --guidance 3.0 \
  --seed 777 \
  --num_views 4 \
  --render_resolution 512
