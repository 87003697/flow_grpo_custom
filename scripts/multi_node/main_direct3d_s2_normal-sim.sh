#!/bin/bash
# 多 GPU Direct3D‑S2 GRPO (sparse512) 最小启动脚本
set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

DATA_DIR=${DATA_DIR:-dataset/eval3d_hunyuan3d}
NORMAL_DIR=${NORMAL_DIR:-dataset/eval3d_hunyuan3d/normals}
LOG_DIR=${LOG_DIR:-logs/direct3d_s2_grpo_multi}
RUN_NAME=${RUN_NAME:-direct3d_s2_grpo_multi}

PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

# View 编码器选择：dino_v2 / dino_v3 / pickscore / clip / hpsv2
# 默认 dino_v3；亦已适配 hpsv2（需本地权重与 config.camera_normal.hpsv2_ckpt_path）；设为 pickscore 可走 CLIP 全局特征余弦
VIEW_ENCODER=${VIEW_ENCODER:-dino_v3}

INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-30}
NUM_CAND=${NUM_CAND:-12}
GUIDANCE=${GUIDANCE:-7.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}
EPOCHS=${EPOCHS:-500}
TRAIN_BS=${TRAIN_BS:-6} #-${NUM_CAND}}
GRAD_ACCUM=${GRAD_ACCUM:-$((NUM_CAND / TRAIN_BS))}
SAVE_FREQ=${SAVE_FREQ:-1}
LR=${LR:-3e-4}
OPT_TYPE=${OPT_TYPE:-adam_8bit}

# PPO 裁剪范围（对称）：控制 config.train.clip_range
CLIP_RANGE=${CLIP_RANGE:-0.02}

# 采样噪声强度：控制 config.slat_sampler_params.noise_level（SDE 随机性）
NOISE_LEVEL=${NOISE_LEVEL:-0.7}

# 时序保留比例：config.train.timestep_keep_ratio
KEEP_RATIO=${KEEP_RATIO:-1.0}

# PPO：是否对无条件分支 detach（对应 config.train.detach_uncond）
DETACH_UNCOND=${DETACH_UNCOND:-false}

# 优势类型（默认 similarity）
ADV_TYPE=${ADV_TYPE:-similarity}  # 可选: similarity, winrate_plus
# 优势来源（逐子项 seperate / 加权总分 average）
ADV_FROM=${ADV_FROM:-average}
# RGB 组比较开关（默认 false）
USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON:-false}

# 评测相关（eval-only 开关）
EVAL_ONLY=${EVAL_ONLY:-false}

# 可选：resume 的 checkpoint 根目录（指向包含 checkpoint_*/ 的目录或具体 checkpoint_* 目录）
CHECKPOINT=${CHECKPOINT:-}

ACC_PY=$(which python)
NVRTC_DIR=$($ACC_PY - <<'PY'
import os, inspect, importlib
print(os.path.dirname(inspect.getfile(importlib.import_module('nvidia.cuda_nvrtc'))))
PY
)
NVJITLINK_DIR=$($ACC_PY - <<'PY'
import os, inspect, importlib
print(os.path.dirname(inspect.getfile(importlib.import_module('nvidia.nvjitlink'))))
PY
)
export LD_LIBRARY_PATH=${NVRTC_DIR}:${NVJITLINK_DIR}:${LD_LIBRARY_PATH:-}

echo "[Direct3D-S2 Multi] DEVICES=$CUDA_VISIBLE_DEVICES | GPUs=$GPU_COUNT" 
echo "   ADV_TYPE=${ADV_TYPE}"
echo "   ADV_FROM=${ADV_FROM}"
echo "   LR=${LR}"
echo "   CLIP_RANGE=${CLIP_RANGE}"
echo "   NOISE_LEVEL=${NOISE_LEVEL}"
echo "   KEEP_RATIO=${KEEP_RATIO}"
echo "   DETACH_UNCOND=${DETACH_UNCOND}"
echo "   USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON}"
echo "   VIEW_ENCODER=${VIEW_ENCODER}"

# 组装可选参数（如 CHECKPOINT）
EXTRA_ARGS=()
if [ -n "${CHECKPOINT}" ]; then
  EXTRA_ARGS+=("--config.checkpoint=${CHECKPOINT}")
fi
if [ -n "${OPT_TYPE}" ]; then
  EXTRA_ARGS+=("--config.train.optimizer.type=${OPT_TYPE}")
fi

"${ACC_PY}" -m accelerate.commands.launch \
  --num_processes=${GPU_COUNT} \
  --main_process_port=29612 \
  scripts/train_direct3d_s2.py \
  --config config/direct3d_s2_grpo_normal-sim.py \
  --config.data_dir="${DATA_DIR}" \
  --config.camera_normal.cache_dir="${NORMAL_DIR}" \
  --config.camera_normal.use_RGB_for_comparison=${USE_RGB_FOR_COMPARISON} \
  --config.camera_normal.encoder="${VIEW_ENCODER}" \
  --config.logdir="${LOG_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_steps=${NUM_STEPS} \
  --config.sample.num_meshes_per_image=${NUM_CAND} \
  --config.sample.num_batches_per_epoch=${NUM_BATCHES_PER_EPOCH} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.sample.adv_type="${ADV_TYPE}" \
  --config.sample.adv_from="${ADV_FROM}" \
  --config.pretrained.pipeline_path="${PRETRAIN_DIR}" \
  --config.pretrained.subfolder="${PRETRAIN_SUBFOLDER}" \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.train.optimizer.lr=${LR} \
  --config.train.clip_range=${CLIP_RANGE} \
  --config.slat_sampler_params.noise_level=${NOISE_LEVEL} \
  --config.train.timestep_keep_ratio=${KEEP_RATIO} \
  --config.train.detach_uncond=${DETACH_UNCOND} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true \
  "${EXTRA_ARGS[@]}"

echo "✅ Direct3D‑S2 multi-GPU training skeleton started. Logs: ${LOG_DIR}" 
