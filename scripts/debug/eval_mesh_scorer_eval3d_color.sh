#!/bin/bash
cd /home/zhiyuan_ma/code2/flow_grpo_custom && \
source /home/zhiyuan_ma/miniconda3/etc/profile.d/conda.sh && \
conda activate grpo3d && \
PYTHONPATH=/home/zhiyuan_ma/code2/flow_grpo_custom python scripts/eval_mesh_scorer_eval3d_color.py \
    --source_front=+z \
    --data_root dataset/eval3d_hi3dgen \
    --cache_dir dataset/eval3d_hi3dgen/normals \
    --rgb_resolution 256 \
    --normal_resolution 518 \
    --camera_ckpt pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors \
    --save_vis \
    --vis_dir logs/dino_vis_rgb_+z_direct3d \
    --output_csv logs/eval3d_mesh_scores_rgb.csv
