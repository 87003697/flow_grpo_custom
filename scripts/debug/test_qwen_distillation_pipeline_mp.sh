#!/bin/bash

# =================================================================
# QwenImageDistillationPipeline 多卡并行消融测试
# 消融维度: CSD 正/负样本模式 (csd_pos_mode × csd_neg_mode)
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
NEGATIVE_PROMPT="blurry, low quality, distorted"

HEIGHT=1024
WIDTH=1024

NUM_OPTIMIZATION_STEPS=1000
LEARNING_RATE=0.05
OPTIMIZER_TYPE="Adam"

LOSS_TYPE="csd"
MSE_WEIGHT=0.0
CSD_WEIGHT=1.0
ADA_FLAG="--ada"         # 设为 "--ada" 启用自适应归一化（与 Flux CSD 默认一致）
ADA_EPS=0.01

CFG_SCALE=4

MIN_STEP_PERCENT=0.02
MAX_STEP_PERCENT=0.98
NUM_TIMESTEPS=1
NOISE_MODE="fixed"

INIT_MODE="random"
INIT_NOISE_SCALE=1.0

DEBUG_SAVE_INTERVAL=50
SEED=42
DTYPE="bfloat16"

# =================================================================
# 实验配置: (GPU_ID, CSD_POS_MODE, CSD_NEG_MODE)
# =================================================================

GPUS=(   4               5          6         7          )
POS_MODES=(  "cfg"        "cfg_rescale"   "cfg"      "cond"     )
NEG_MODES=(  "uncond"     "uncond"        "cond"     "uncond"   )

NUM_EXPS=${#GPUS[@]}

# 输出根目录
OUTPUT_ROOT="outputs/qwen_distillation_ablation_csd_modes"
mkdir -p "$OUTPUT_ROOT"

# =================================================================
# 环境准备
# =================================================================

cd /home/zhiyuan_ma/code/flow_grpo_custom_v2

source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查输入文件
if [ ! -f "$CONDITION_IMAGE" ]; then
    echo "❌ 错误: 找不到条件图像文件: $CONDITION_IMAGE"
    exit 1
fi

# =================================================================
# 并行启动所有实验
# =================================================================

echo "============================================================"
echo "▶️  CSD 正/负样本模式消融测试 (${NUM_EXPS} 个实验并行)"
echo "    CFG Scale: $CFG_SCALE"
echo "    优化步数: $NUM_OPTIMIZATION_STEPS"
echo "    学习率: $LEARNING_RATE"
echo "    条件图: $CONDITION_IMAGE"
echo "============================================================"
echo ""

PIDS=()

for i in $(seq 0 $((NUM_EXPS - 1))); do
    GPU_ID=${GPUS[$i]}
    POS_MODE=${POS_MODES[$i]}
    NEG_MODE=${NEG_MODES[$i]}
    
    EXP_NAME="pos-${POS_MODE}_neg-${NEG_MODE}"
    EXP_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    LOG_FILE="${OUTPUT_ROOT}/${EXP_NAME}.log"
    
    mkdir -p "$EXP_OUTPUT_DIR"
    
    echo "🚀 [实验 $((i+1))/${NUM_EXPS}] GPU=$GPU_ID  pos=$POS_MODE  neg=$NEG_MODE"
    echo "   输出: $EXP_OUTPUT_DIR"
    echo "   日志: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/debug/test_qwen_distillation_pipeline.py \
        --model_path "$MODEL_PATH" \
        --condition_image "$CONDITION_IMAGE" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --prompt "$PROMPT" \
        --negative_prompt "$NEGATIVE_PROMPT" \
        --height $HEIGHT \
        --width $WIDTH \
        --num_optimization_steps $NUM_OPTIMIZATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --optimizer_type $OPTIMIZER_TYPE \
        --loss_type $LOSS_TYPE \
        --mse_weight $MSE_WEIGHT \
        --csd_weight $CSD_WEIGHT \
        $ADA_FLAG \
        --ada_eps $ADA_EPS \
        --cfg_scale $CFG_SCALE \
        --min_step_percent $MIN_STEP_PERCENT \
        --max_step_percent $MAX_STEP_PERCENT \
        --num_timesteps $NUM_TIMESTEPS \
        --noise_mode $NOISE_MODE \
        --csd_pos_mode $POS_MODE \
        --csd_neg_mode $NEG_MODE \
        --init_mode $INIT_MODE \
        --init_noise_scale $INIT_NOISE_SCALE \
        --save_debug_images \
        --debug_save_interval $DEBUG_SAVE_INTERVAL \
        --generate_video \
        --seed $SEED \
        --dtype $DTYPE \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "⏳ 所有实验已启动，等待完成..."
echo "   PIDs: ${PIDS[*]}"
echo ""

# =================================================================
# 等待所有进程完成并汇总结果
# =================================================================

FAILED=0
for i in $(seq 0 $((NUM_EXPS - 1))); do
    PID=${PIDS[$i]}
    POS_MODE=${POS_MODES[$i]}
    NEG_MODE=${NEG_MODES[$i]}
    EXP_NAME="pos-${POS_MODE}_neg-${NEG_MODE}"
    
    if wait $PID; then
        echo "✅ [实验 $((i+1))] ${EXP_NAME} — 成功 (PID=$PID)"
    else
        echo "❌ [实验 $((i+1))] ${EXP_NAME} — 失败 (PID=$PID)"
        echo "   查看日志: ${OUTPUT_ROOT}/${EXP_NAME}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "📊 消融测试汇总"
echo "============================================================"
echo "   输出根目录: $OUTPUT_ROOT"
echo "   总实验数: $NUM_EXPS"
echo "   成功: $((NUM_EXPS - FAILED))"
echo "   失败: $FAILED"
echo ""

# 打印各实验最终 loss
for i in $(seq 0 $((NUM_EXPS - 1))); do
    POS_MODE=${POS_MODES[$i]}
    NEG_MODE=${NEG_MODES[$i]}
    EXP_NAME="pos-${POS_MODE}_neg-${NEG_MODE}"
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
    echo "✅ 所有实验完成！"
else
    echo "⚠️  有 $FAILED 个实验失败，请检查对应日志。"
    exit 1
fi