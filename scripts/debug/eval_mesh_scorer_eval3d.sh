#!/bin/bash
cd /home/zhiyuan_ma/code/flow_grpo_custom && \
export CUDA_VISIBLE_DEVICES=1 && \
PYTHONPATH=/home/zhiyuan_ma/code/flow_grpo_custom python scripts/eval_mesh_scorer_eval3d.py \
    --source_front=+z \
    --data_root dataset/eval3d_hi3dgen \
    --cache_dir dataset/eval3d_hi3dgen/normals \
    --camera_ckpt pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors \
    --save_vis \
    --vis_dir logs/dino_vis_normal_+z \
    --output_csv logs/eval3d_mesh_scores_normal.csv