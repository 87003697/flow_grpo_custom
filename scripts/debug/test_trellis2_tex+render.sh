CUDA_VISIBLE_DEVICES=0 \
python scripts/debug/test_trellis2_tex+render.py \
    --image dataset/debug_ddp/test/00098.png \
    --device cuda:0