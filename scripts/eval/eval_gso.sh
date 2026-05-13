#!/bin/bash
# GSO Benchmark 评测脚本（OREO Rebuttal）
#
# 功能：
#   1. 用 eval_trellis.py (DDP 8GPU) 在 GSO 测试集上同时评测：
#      - Teacher: Trellis pretrained（frozen pretrained weights）
#      - Student: OREO (checkpoint_11_3344)
#      → 输出 CLIP Sim / DINO Sim
#   2. 用 eval_image_quality.py 在渲染结果上计算：
#      → 输出 MANIQA / MUSIQ（分 teacher / student）
#   3. 调用 aggregate_gso_results.py 打印最终对比表
#
# 输出目录：
#   logs_for_eval/<RUN_NAME>_gso/eval_teacher_student/checkpoint_11_3344/
#
# 用法：
#   conda activate grpo3d_trellis
#   bash scripts/eval/eval_gso.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =====================================================================
# 配置
# =====================================================================

CKPT="logs/trellis_dual_contrast_on_v-1e0_wo-swap_ada-false_FlowEdit_cfg-4_steps-9-12_promptv3_adan_lr-1e-4_8GPU/checkpoints/checkpoint_11_3444"
EVAL_DIR="dataset/gso_test"
LOGDIR_ROOT="logs_for_eval"

# 从 checkpoint 路径推断基础 run_name，加 _gso 后缀区分 GSO 结果与原始结果
BASE_RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
GSO_RUN_NAME="${BASE_RUN_NAME}_gso"

# ckpt_tag 与 eval_trellis.py 内部逻辑保持一致（= checkpoint 目录名）
CKPT_TAG=$(basename "$CKPT")

OUT_DIR="${LOGDIR_ROOT}/${GSO_RUN_NAME}/eval_teacher_student/${CKPT_TAG}"
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "========================================"
echo "GSO Benchmark 评测（OREO Rebuttal）"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "CKPT:        $CKPT"
echo "EVAL_DIR:    $EVAL_DIR"
echo "GSO_RUN_NAME:$GSO_RUN_NAME"
echo "CKPT_TAG:    $CKPT_TAG"
echo "OUT_DIR:     $OUT_DIR"
echo "========================================"

# =====================================================================
# Step 1: CLIP + DINO（teacher vs student 对比）
# =====================================================================

DONE_FLAG="${OUT_DIR}/teacher_student_similarity.csv"

if [ -f "$DONE_FLAG" ]; then
    echo "[INFO] CLIP/DINO 评测已完成，跳过（已有结果: $DONE_FLAG）"
else
    echo "[INFO] 开始 CLIP/DINO 评测 ..."
    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    python -m accelerate.commands.launch \
        --num_processes="$GPU_COUNT" \
        --multi_gpu \
        --main_process_port="$(shuf -i 29000-30000 -n 1)" \
        scripts/eval/eval_trellis.py \
        --config=config/trellis_stage2_distillation.py \
        --config.run_name="$GSO_RUN_NAME" \
        --config.checkpoint="$CKPT" \
        --config.logdir="$LOGDIR_ROOT" \
        --config.data.eval.dir="$EVAL_DIR"

    echo "[INFO] CLIP/DINO 评测完成"
    sleep 5
fi

# =====================================================================
# Step 2: MANIQA + MUSIQ（student = OREO）
# =====================================================================

IQA_STUDENT_JSON="${OUT_DIR}/image_quality_student.json"

if [ -f "$IQA_STUDENT_JSON" ]; then
    echo "[INFO] OREO 图像质量评测已完成，跳过"
else
    echo "[INFO] 开始 OREO (student) MANIQA/MUSIQ 评测 ..."
    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    python scripts/eval/eval_image_quality.py \
        --images_dir "${OUT_DIR}/images" \
        --role student \
        --save_json "$IQA_STUDENT_JSON"
    echo "[INFO] OREO 图像质量评测完成"
fi

# =====================================================================
# Step 3: MANIQA + MUSIQ（teacher = Trellis pretrained）
# =====================================================================

IQA_TEACHER_JSON="${OUT_DIR}/image_quality_teacher.json"

if [ -f "$IQA_TEACHER_JSON" ]; then
    echo "[INFO] Trellis 图像质量评测已完成，跳过"
else
    echo "[INFO] 开始 Trellis pretrained (teacher) MANIQA/MUSIQ 评测 ..."
    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    python scripts/eval/eval_image_quality.py \
        --images_dir "${OUT_DIR}/images" \
        --role teacher \
        --save_json "$IQA_TEACHER_JSON"
    echo "[INFO] Trellis 图像质量评测完成"
fi

# =====================================================================
# Step 4: 聚合并打印结果
# =====================================================================

echo ""
echo "========================================"
echo "聚合结果..."
echo "========================================"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python scripts/eval/aggregate_gso_results.py \
    --eval_dir "$OUT_DIR"

echo ""
echo "========================================"
echo "GSO 评测全部完成！"
echo "结果目录: $OUT_DIR"
echo "========================================"
