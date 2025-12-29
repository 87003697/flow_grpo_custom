source ~/anaconda3/etc/profile.d/conda.sh && \
conda activate grpo3d && \
cd /home/zhiyuan_ma/code/flow_grpo_custom && \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/zhiyuan_ma/code/flow_grpo_custom:$PYTHONPATH \
python scripts/debug/test_trellis_infer.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/trellis_validation \
    --steps 50 \
    --guidance 3.0 \
    --sigma_min 0.002 \
    --rescale_t 1.0 \
    --candidates 2 \
    --seed 777 \
    --sde