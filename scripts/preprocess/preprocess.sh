export HF_HUB_OFFLINE=1
# python scripts/preprocess/generate_normals_png.py \
#   --input_dir ./dataset/eval3d/images \
#   --output_dir ./dataset/eval3d/normals \
#   --resolution 518 \
#   --device cuda \
#   --batch_size 8 \
#   --predictor_module reward_models.camera_normal_scorer.normal_io.stable_normal_predictor \
#   --model_dir ./pretrained_weights/yoso-normal-v1-8-1

python scripts/preprocess/generate_normals_png.py \
  --input_dir ./dataset/eval3d_hunyuan3d/images \
  --output_dir ./dataset/eval3d_hunyuan3d/normals \
  --resolution 518 \
  --device cuda \
  --batch_size 8 \
  --predictor_module reward_models.camera_normal_scorer.normal_io.stable_normal_predictor \
  --model_dir ./pretrained_weights/yoso-normal-v1-8-1