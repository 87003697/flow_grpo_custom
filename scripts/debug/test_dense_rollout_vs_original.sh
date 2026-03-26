conda run -n grpo3d_trellis python scripts/debug/test_dense_rollout_vs_original.py \
  --model_path pretrained_weights/TRELLIS-image-large \
  --image dataset/eval3d_hunyuan3d/images/004.png \
  --seed 42