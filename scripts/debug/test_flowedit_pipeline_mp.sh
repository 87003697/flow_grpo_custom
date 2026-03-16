#!/bin/bash

# =================================================================
# FlowEditPipeline 多卡并行消融测试
# 消融维度: CFG 强度 (cfg_scale)
# =================================================================

set -e

# 设置环境变量
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =================================================================
# 公共参数（所有实验共享）
# =================================================================

MODEL_PATH="Qwen/Qwen-Image-Edit-2511"
CONDITION_IMAGE="dataset/alphaimages_1k/test/images/00098.png"
PROMPT="Move the camera"
NEGATIVE_PROMPT=""

# FlowEdit 参数
NUM_INFERENCE_STEPS=12
CFG_SCALE=4.0
GUIDANCE_SCALE=4.0
N_MAX=9
REDUCE_MODE="final"

# 优化参数
NUM_OPTIMIZATION_STEPS=50
INNER_STEPS=5
LEARNING_RATE=0.05
OPTIMIZER_TYPE="AdamW"

# 其他
SAVE_INTERVAL=10
SEED=42
DTYPE="bfloat16"

# =================================================================
# 实验配置: (GPU_ID, CFG_SCALE)
# 可按需修改消融维度，例如不同 CFG 强度
# =================================================================

GPUS=( 7   )
CFG_SCALES=(4.0)

NUM_EXPS=${#GPUS[@]}

# 输出根目录
OUTPUT_ROOT="outputs/flowedit_ablation"
mkdir -p "$OUTPUT_ROOT"

# =================================================================
# 环境准备
# =================================================================

cd /home/zhiyuan_ma/code/flow_grpo_custom

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查输入文件
if [ ! -f "$CONDITION_IMAGE" ]; then
    echo "错误: 找不到条件图像文件: $CONDITION_IMAGE"
    exit 1
fi

# =================================================================
# 并行启动所有实验
# =================================================================

echo "============================================================"
echo "FlowEdit CFG 强度消融测试 (${NUM_EXPS} 个实验并行)"
echo "    CFG Scale: tgt=+${CFG_SCALE}, src=-${CFG_SCALE}"
echo "    FlowEdit Steps: ${NUM_INFERENCE_STEPS}, n_max: ${N_MAX}"
echo "    外层步数: $NUM_OPTIMIZATION_STEPS, 内层步数: $INNER_STEPS"
echo "    学习率: $LEARNING_RATE"
echo "    条件图: $CONDITION_IMAGE"
echo "============================================================"
echo ""

PIDS=()

for i in $(seq 0 $((NUM_EXPS - 1))); do
    GPU_ID=${GPUS[$i]}
    EXP_CFG=${CFG_SCALES[$i]}

    EXP_NAME="cfg-${EXP_CFG}"
    EXP_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    LOG_FILE="${OUTPUT_ROOT}/${EXP_NAME}.log"

    mkdir -p "$EXP_OUTPUT_DIR"

    echo "[实验 $((i+1))/${NUM_EXPS}] GPU=$GPU_ID  cfg=${EXP_CFG}"
    echo "   输出: $EXP_OUTPUT_DIR"
    echo "   日志: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/debug/test_flowedit_pipeline.py \
        --model_path "$MODEL_PATH" \
        --condition_image "$CONDITION_IMAGE" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --prompt "$PROMPT" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --cfg_scale $EXP_CFG \
        --guidance_scale $GUIDANCE_SCALE \
        --n_max $N_MAX \
        --reduce_mode $REDUCE_MODE \
        --num_optimization_steps $NUM_OPTIMIZATION_STEPS \
        --inner_steps $INNER_STEPS \
        --learning_rate $LEARNING_RATE \
        --optimizer_type $OPTIMIZER_TYPE \
        --save_interval $SAVE_INTERVAL \
        --seed $SEED \
        --dtype $DTYPE \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "所有实验已启动，等待完成..."
echo "   PIDs: ${PIDS[*]}"
echo ""

# =================================================================
# 等待所有进程完成并汇总结果
# =================================================================

FAILED=0
for i in $(seq 0 $((NUM_EXPS - 1))); do
    PID=${PIDS[$i]}
    EXP_CFG=${CFG_SCALES[$i]}
    EXP_NAME="cfg-${EXP_CFG}"

    if wait $PID; then
        echo "[实验 $((i+1))] ${EXP_NAME} — 成功 (PID=$PID)"
    else
        echo "[实验 $((i+1))] ${EXP_NAME} — 失败 (PID=$PID)"
        echo "   查看日志: ${OUTPUT_ROOT}/${EXP_NAME}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "消融测试汇总"
echo "============================================================"
echo "   输出根目录: $OUTPUT_ROOT"
echo "   总实验数: $NUM_EXPS"
echo "   成功: $((NUM_EXPS - FAILED))"
echo "   失败: $FAILED"
echo ""

# 打印各实验最终 loss
for i in $(seq 0 $((NUM_EXPS - 1))); do
    EXP_CFG=${CFG_SCALES[$i]}
    EXP_NAME="cfg-${EXP_CFG}"
    PARAMS_FILE="${OUTPUT_ROOT}/${EXP_NAME}/parameters.txt"

    if [ -f "$PARAMS_FILE" ]; then
        FINAL_LOSS=$(grep "最终 Loss" "$PARAMS_FILE" | awk '{print $NF}')
        printf "   %-35s  最终 Loss: %s\n" "$EXP_NAME" "$FINAL_LOSS"
    else
        printf "   %-35s  (无结果)\n" "$EXP_NAME"
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "所有实验完成！"
else
    echo "有 $FAILED 个实验失败，请检查对应日志。"
    exit 1
fi
