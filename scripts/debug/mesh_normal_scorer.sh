export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/eval_mesh_scorer_eval3d.py \
  --camera_config _reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py \
  --camera_ckpt pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors \
  --save_vis