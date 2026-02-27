#!/bin/bash

# =================================================================
# QwenImageDistillationPipeline 测试脚本（迭代优化版）
# =================================================================

# 设置环境变量
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 配置参数 ---
GPU_ID=4

# 模型路径
MODEL_PATH="Qwen/Qwen-Image-Edit-2511"

# 输入输出
CONDITION_IMAGE="dataset/alphaimages_1k/test/images/00098.png"  # 条件图像路径
OUTPUT_DIR="outputs/qwen_distillation_test"
PROMPT="Move the camera"
NEGATIVE_PROMPT="blurry, low quality, distorted"

# 图像参数
HEIGHT=512
WIDTH=512

# 优化参数
NUM_OPTIMIZATION_STEPS=1000
LEARNING_RATE=0.1
OPTIMIZER_TYPE="Adam"  # Adam, AdamW, SGD

# Loss 参数
LOSS_TYPE="csd"         # csd, mse, mixed
MSE_WEIGHT=0.0
CSD_WEIGHT=1.0
ADA_FLAG=""             # 设为 "--ada" 启用自适应归一化
ADA_EPS=0.0001

# CFG 参数
CFG_SCALE=4.0

# 时间步参数
MIN_STEP_PERCENT=0.02
MAX_STEP_PERCENT=0.98
NUM_TIMESTEPS=1         # MTS 时间步数量
NOISE_MODE="fixed"      # random, fixed, aligned, inversion_*

# 初始化参数
INIT_MODE="random"      # random, condition
INIT_NOISE_SCALE=1.0

# 调试参数
SAVE_DEBUG_IMAGES="--save_debug_images"
DEBUG_SAVE_INTERVAL=100
GENERATE_VIDEO="--generate_video"

# 随机种子
SEED=42

# 精度
DTYPE="bfloat16"

echo "▶️  开始 QwenImageDistillationPipeline 测试..."
echo "    GPU: $GPU_ID"
echo "    模型: $MODEL_PATH"
echo "    条件图: $CONDITION_IMAGE"
echo "    提示: '$PROMPT'"
echo "    Loss 类型: $LOSS_TYPE"
echo "    优化步数: $NUM_OPTIMIZATION_STEPS"
echo "    学习率: $LEARNING_RATE"
echo "    CFG 强度: $CFG_SCALE"
echo "    图像尺寸: ${WIDTH}x${HEIGHT}"
echo ""

# 检查输入文件
if [ ! -f "$CONDITION_IMAGE" ]; then
    echo "❌ 错误: 找不到条件图像文件: $CONDITION_IMAGE"
    echo "   请修改 CONDITION_IMAGE 变量指向有效的图像文件"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 切换到项目根目录
cd /home/zhiyuan_ma/code/flow_grpo_custom

# 激活 conda 环境
source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
conda activate grpo3d_trellis

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行测试
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/debug/test_qwen_distillation_pipeline.py \
    --model_path "$MODEL_PATH" \
    --condition_image "$CONDITION_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
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
    --init_mode $INIT_MODE \
    --init_noise_scale $INIT_NOISE_SCALE \
    $SAVE_DEBUG_IMAGES \
    --debug_save_interval $DEBUG_SAVE_INTERVAL \
    $GENERATE_VIDEO \
    --seed $SEED \
    --dtype $DTYPE

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ QwenImageDistillationPipeline 测试完成！"
    echo "📁 结果保存在: $OUTPUT_DIR"
    echo ""
    echo "📊 结果文件:"
    echo "   - final_image.png       (最终生成图像)"
    echo "   - condition_image.png   (条件图像)"
    echo "   - loss_curve.png        (Loss 曲线)"
    echo "   - parameters.txt        (测试参数)"
    if [ -d "$OUTPUT_DIR/debug_images" ]; then
        echo "   - debug_images/         (调试图像序列)"
    fi
    if [ -f "$OUTPUT_DIR/optimization_process.mp4" ]; then
        echo "   - optimization_process.mp4 (优化过程视频)"
    fi
    echo ""
else
    echo ""
    echo "❌ QwenImageDistillationPipeline 测试失败！"
    echo "请检查错误信息并重试。"
    echo ""
    exit 1
fi
