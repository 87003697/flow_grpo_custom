CUDA_VISIBLE_DEVICES=0 
python scripts/debug/test_trellis2_shape+render.py \
    --image dataset/alphaimages_1k/test/images/00098.png \
    --device cuda:0