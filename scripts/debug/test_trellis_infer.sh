if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
else
  echo "conda.sh 未找到，请检查 Conda 安装路径" >&2
  exit 1
fi && \
conda activate grpo3d && \
cd /home/zhiyuan_ma/code2/flow_grpo_custom && \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/zhiyuan_ma/code2/flow_grpo_custom:$PYTHONPATH \
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