#!/bin/bash
# Test-Time 3D Editing 脚本
#
# 流程：TRELLIS 生成 GS + Mesh → FlowEdit 编辑多视角 → 纹理烘焙 → 导出 GLB
#
# GPU 需求：
#   - 单卡即可（TRELLIS 和 FlowEdit 顺序加载，不同时占用显存）
#   - 建议 40GB+ 显存（A100 / A6000）
#
# 用法：
#   bash scripts/eval/test_time_edit.sh

# ============================================================
# GPU 配置
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ============================================================
# 模型路径（与 config/trellis_stage2_distillation.py 一致）
# ============================================================
TRELLIS_MODEL="${TRELLIS_MODEL:-./pretrained_weights/TRELLIS-image-large}"
FLOWEDIT_MODEL="${FLOWEDIT_MODEL:-Qwen/Qwen-Image-Edit-2511}"

# ============================================================
# 输入 / 输出
# ============================================================
INPUT_IMAGE="${INPUT_IMAGE:-dataset/alphaimages_v3/test/test_01.png}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/test_time_edit}"
OUTPUT_NAME="${OUTPUT_NAME:-edited}"

# ============================================================
# 编辑 Prompt（与 config 中 _flowedit_runtime_config 一致）
# ============================================================
SOURCE_PROMPT="${SOURCE_PROMPT:-Rotate the camera.}"
TARGET_PROMPT="${TARGET_PROMPT:-Rotate the camera.}"
NEGATIVE_PROMPT_SRC="${NEGATIVE_PROMPT_SRC:- }"
NEGATIVE_PROMPT_TGT="${NEGATIVE_PROMPT_TGT:- }"

# ============================================================
# 渲染 & 编辑参数（与 config 中 _flowedit_init_config / _flowedit_runtime_config 一致）
# ============================================================
NVIEWS=${NVIEWS:-16}
RENDER_RESOLUTION=${RENDER_RESOLUTION:-1024}
EDIT_STEPS=${EDIT_STEPS:-12}             # flowedit.steps = 12
CFG_SCALE_SRC=${CFG_SCALE_SRC:--4}       # true_cfg_scale_src = -1 * cfg_scale_tgt
CFG_SCALE_TGT=${CFG_SCALE_TGT:-4}       # true_cfg_scale_tgt = 4
N_MAX=${N_MAX:-9}                        # flowedit.n_max = 9
NOISE_MODE=${NOISE_MODE:-aligned}        # flowedit.noise_mode = "aligned"
EDIT_SEED=${EDIT_SEED:-0}               # seed = 0
TRELLIS_SEED=${TRELLIS_SEED:-1}
BG_COLOR="${BG_COLOR:-1.0 1.0 1.0}"     # bg_color = [1.0, 1.0, 1.0] (白色，与 GS renderer 一致)

# ============================================================
# 纹理烘焙参数
# ============================================================
SIMPLIFY=${SIMPLIFY:-0.95}
TEXTURE_SIZE=${TEXTURE_SIZE:-1024}
BAKE_MODE=${BAKE_MODE:-opt}

# ============================================================
# 输出选项
# ============================================================
SAVE_INTERMEDIATE=${SAVE_INTERMEDIATE:-true}
SAVE_VIDEO=${SAVE_VIDEO:-false}

# ============================================================
# 打印配置
# ============================================================
echo "========================================"
echo "Test-Time 3D Editing"
echo "========================================"
echo "GPU:              $CUDA_VISIBLE_DEVICES"
echo "TRELLIS:          $TRELLIS_MODEL"
echo "FlowEdit:         $FLOWEDIT_MODEL"
echo "Input:            $INPUT_IMAGE"
echo "Output:           $OUTPUT_DIR/$OUTPUT_NAME.glb"
echo "Source Prompt:     $SOURCE_PROMPT"
echo "Target Prompt:     $TARGET_PROMPT"
echo "Views:            $NVIEWS"
echo "Edit Steps:       $EDIT_STEPS"
echo "CFG (src/tgt):    $CFG_SCALE_SRC / $CFG_SCALE_TGT"
echo "n_max:            $N_MAX"
echo "Noise Mode:       $NOISE_MODE"
echo "BG Color:         $BG_COLOR"
echo "Texture Size:     $TEXTURE_SIZE"
echo "========================================"

# ============================================================
# 构建命令
# ============================================================
CMD=(
    python -m edit4shape.experimental.test_time_edit
    --input_image "$INPUT_IMAGE"
    --output_dir "$OUTPUT_DIR"
    --output_name "$OUTPUT_NAME"
    --trellis_model "$TRELLIS_MODEL"
    --flowedit_model "$FLOWEDIT_MODEL"
    --trellis_seed "$TRELLIS_SEED"
    --render_resolution "$RENDER_RESOLUTION"
    --nviews "$NVIEWS"
    --source_prompt "$SOURCE_PROMPT"
    --target_prompt "$TARGET_PROMPT"
    --negative_prompt_src "$NEGATIVE_PROMPT_SRC"
    --negative_prompt_tgt "$NEGATIVE_PROMPT_TGT"
    --edit_steps "$EDIT_STEPS"
    --cfg_scale_src "$CFG_SCALE_SRC"
    --cfg_scale_tgt "$CFG_SCALE_TGT"
    --n_max "$N_MAX"
    --noise_mode "$NOISE_MODE"
    --edit_seed "$EDIT_SEED"
    --bg_color $BG_COLOR
    --simplify "$SIMPLIFY"
    --texture_size "$TEXTURE_SIZE"
    --bake_mode "$BAKE_MODE"
)

# 可选 flag
if [ "$SAVE_INTERMEDIATE" = "true" ]; then
    CMD+=(--save_intermediate)
fi
if [ "$SAVE_VIDEO" = "true" ]; then
    CMD+=(--save_video)
fi

# ============================================================
# 执行
# ============================================================
echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"
