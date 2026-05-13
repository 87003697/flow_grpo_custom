#!/bin/bash
# Dense FlowEdit 评估脚本（DDP 多卡）
#
# 对比：
#   Teacher (pretrained): Dense ODE → Sparse ODE
#   Student (finetuned) : Dense ODE → Dense FlowEdit → Sparse ODE
#
# 用法：
#   bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh
#   bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh <CKPT_INPUT>
#   bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh <CKPT_INPUT> <EVAL_DATASET>
#
# CKPT_INPUT 支持三种输入：
#   1) "" -> 不加载 checkpoint（student ≈ teacher，用于验证框架）
#   2) ".../checkpoints/checkpoint_xxx" -> 单个 ckpt
#   3) ".../checkpoints" -> 遍历所有 checkpoint_*
#
# EVAL_DATASET 支持：
#   1) alphaimages   -> dataset/alphaimages_v3/test（默认）
#   2) alphaimage_train -> dataset/alphaimages_v3/train
#   3) gso_selected  -> dataset/gso_test_selected
#   4) toys4k        -> dataset/toys4k_test
#
# FlowEdit 参数可通过附加 --config.rollout.flowedit.* 覆盖默认值，例如：
#   bash scripts/eval/eval_trellis_dense_flowedit_DDP.sh <CKPT> gso_selected \
#     --config.rollout.flowedit.n_max=12 \
#     --config.rollout.flowedit.cfg_scale_tgt=4.0 \
#     --config.rollout.flowedit.cfg_scale_src=-4.0

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CKPT_INPUT="${1:-}"
EVAL_DATASET="${2:-alphaimages}"
# 收集其余参数（FlowEdit 覆盖项）
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift
fi
EXTRA_ARGS=("$@")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "$EVAL_DATASET" in
    alphaimages)
        EVAL_DIR="dataset/alphaimages_v3/test"
        RUN_SUFFIX="_alphaimages_v3_test"
        ;;
    alphaimage_train)
        EVAL_DIR="dataset/alphaimages_v3/train"
        RUN_SUFFIX="_alphaimage_train"
        ;;
    gso_selected)
        EVAL_DIR="dataset/gso_test_selected"
        RUN_SUFFIX="_gso_selected"
        ;;
    toys4k)
        EVAL_DIR="dataset/toys4k_test"
        RUN_SUFFIX="_toys4k"
        ;;
    *)
        echo "错误: 未知 EVAL_DATASET: $EVAL_DATASET"
        echo "支持的 EVAL_DATASET: alphaimages, alphaimage_train, gso_selected, toys4k"
        exit 1
        ;;
esac
LOGDIR_ROOT="logs_for_eval"
CONFIG="config/trellis_stage1+2_contrastive.py"

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
CKPT_INPUT="${CKPT_INPUT%/}"

# 构建待评估 ckpt 列表
CKPT_LIST=()
if [ -z "$CKPT_INPUT" ]; then
    CKPT_LIST+=("")
elif [ "$(basename "$CKPT_INPUT")" = "checkpoints" ]; then
    mapfile -t CKPT_LIST < <(
        find "$CKPT_INPUT" -maxdepth 1 -mindepth 1 -type d -name "checkpoint_*" | sort -V
    )
    if [ "${#CKPT_LIST[@]}" -eq 0 ]; then
        echo "错误: 在 $CKPT_INPUT 下未找到 checkpoint_*"
        exit 1
    fi
else
    CKPT_LIST+=("$CKPT_INPUT")
fi

echo "========================================"
echo "Dense FlowEdit DDP 评估"
echo "========================================"
echo "可见 GPU: $CUDA_VISIBLE_DEVICES ($GPU_COUNT 张)"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "EVAL_DIR: $EVAL_DIR"
echo "LOGDIR_ROOT: $LOGDIR_ROOT"
echo "CONFIG: $CONFIG"
echo "CKPT_INPUT: ${CKPT_INPUT:-（无，使用 pretrained）}"
echo "待评估数量: ${#CKPT_LIST[@]}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo "额外参数: ${EXTRA_ARGS[*]}"
echo "========================================"

for CKPT in "${CKPT_LIST[@]}"; do
    if [ -n "$CKPT" ]; then
        TRAIN_RUN=$(basename "$(dirname "$(dirname "$CKPT")")")
        CKPT_NAME=$(basename "$CKPT")
        RUN_NAME="${TRAIN_RUN}${RUN_SUFFIX}"
    else
        RUN_NAME="eval_pretrained_baseline${RUN_SUFFIX}"
        CKPT_NAME="pretrained_baseline"
    fi

    # 输出目录后缀与脚本中 _dense_flowedit 一致
    OUT_DIR="${LOGDIR_ROOT}/${RUN_NAME}/eval_teacher_student/${CKPT_NAME}_dense_flowedit"
    DONE_FLAG="${OUT_DIR}/teacher_student_similarity.csv"
    IS_COMPLETE=0
    if [ -f "$DONE_FLAG" ]; then
        if python - "$DONE_FLAG" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    sys.exit(0 if any(row.get("name") == "AVERAGE" for row in reader) else 1)
PY
        then
            IS_COMPLETE=1
        fi
    fi

    if [ "$IS_COMPLETE" -eq 1 ]; then
        echo "跳过已完成 ckpt: ${CKPT:-（无，使用 pretrained）}"
        echo "已有结果: $DONE_FLAG"
        continue
    elif [ -f "$DONE_FLAG" ]; then
        echo "检测到未完成 CSV，将跳过已完成样本并继续补跑: $DONE_FLAG"
    fi

    echo "------------ 开始评估 ------------"
    echo "RUN_NAME: $RUN_NAME"
    echo "CKPT: ${CKPT:-（无，使用 pretrained）}"
    echo "OUT_DIR: $OUT_DIR"
    echo "----------------------------------"

    PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    NCCL_TIMEOUT=1800 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    python -m accelerate.commands.launch \
        --num_processes="$GPU_COUNT" \
        --multi_gpu \
        --main_process_port="$(shuf -i 20000-29000 -n 1)" \
        scripts/eval/eval_trellis_dense_flowedit.py \
        --config="$CONFIG" \
        --config.run_name="$RUN_NAME" \
        --config.checkpoint="${CKPT}" \
        --config.logdir="$LOGDIR_ROOT" \
        --config.data.eval.dir="$EVAL_DIR" \
        "${EXTRA_ARGS[@]}" || true

    rm -f /dev/shm/nccl-* 2>/dev/null || true
    sleep 30
done
