export HF_HUB_OFFLINE=1
python scripts/preprocess/generate_normals_png.py \
  --input_dir dataset/eval3d/images \
  --output_dir dataset/eval3d/normals \
  --model_dir pretrained_weights/stable-normal \
  --resolution 512 \
  --device cuda \
  --batch_size 8 \
  --preprocess_mode crop