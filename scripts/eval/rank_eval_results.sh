#!/bin/bash
# 对 eval_trellis.py 产出的评估结果进行排序 + 收集 grid 图片
#
# 功能：
#   1. 按 clip_delta + dino_delta 降序排列样本
#   2. 输出 teacher_student_similarity_ranked.csv
#   3. 收集指定视角的 grid 图片到 ranked_grids_v{VIEW}/ 目录
#
# 用法：
#   bash scripts/eval/rank_eval_results.sh

# ---- 配置 ----
# 评估输出目录（包含 teacher_student_similarity.csv 和 images/）
EVAL_DIR="logs/eval_trellis_x0-01_FlowEdit-ada01-mts_cfg-4_steps-9_12_sgd_lr-1e-3_8GPU_checkpoint_0_574/eval_teacher_student_default"

# 要收集的 grid 视角编号（默认 3）
VIEW=3

# ---- 运行 ----
echo "========================================"
echo "排序评估结果"
echo "========================================"
echo "EVAL_DIR: $EVAL_DIR"
echo "VIEW:     $VIEW"
echo "========================================"

python scripts/eval/rank_eval_results.py "$EVAL_DIR" --view "$VIEW"
