CUDA_VISIBLE_DEVICES=4 python "scripts/debug/test_trellis2_neighbor-26|mesh_render.py" \
    --image dataset/alphaimages_v3/train/15747.png \
    --save_dir ./outputs/hybrid26_vs_mesh \
    --device cuda:0 \
    --resolution 1024 \
    --render_res 512
