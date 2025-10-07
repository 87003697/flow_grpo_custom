cd /home/zhiyuan_ma/code/flow_grpo_custom && \
source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh && \
conda activate grpo3d && \
PYTHONPATH=/home/zhiyuan_ma/code/flow_grpo_custom python scripts/eval_mesh_scorer_eval3d.py \
    --source_front=-z \
    --data_root dataset/eval3d_direct3d \
    --cache_dir dataset/eval3d_direct3d/normals \
    --camera_ckpt pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors \
    --save_vis \
    --vis_dir logs/dino_vis_-z_direct3d