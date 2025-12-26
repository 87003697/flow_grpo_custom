# 第一步：运行参考代码，保存中间结果
cd /home/zhiyuan_ma/code/flow_grpo_custom
conda activate grpo3d_trellis
python scripts/debug/debug_trellis_comparison.py --mode ref --ref_gpu 0 --seed 42

# 第二步：运行我们的代码，对比结果
python scripts/debug/debug_trellis_comparison.py --mode ours --our_gpu 0 --seed 42