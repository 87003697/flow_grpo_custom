#!/bin/bash
# 简化版：默认从脚本相对路径跳到仓库根目录后直接跑 Python
set -e

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python scripts/eval/eval_mesh_scorer_eval3d.py \
  --source_front=+z \
  --data_root dataset/eval3d_hi3dgen \
  --cache_dir dataset/eval3d_hi3dgen/normals \
  --camera_ckpt pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors \
  --limit 8 \
  --batch_size 4 \
  --save_vis \
  --vis_dir logs/dino_vis_normal_+z \
  --output_csv logs/eval3d_mesh_scores_normal.csv