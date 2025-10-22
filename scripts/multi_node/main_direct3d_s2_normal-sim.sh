#!/bin/bash
# 多 GPU Direct3D‑S2 GRPO (sparse512) 最小启动脚本
set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse

: "${CUDA_VISIBLE_DEVICES:=1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

DATA_DIR=${DATA_DIR:-dataset/eval3d_hunyuan3d}
NORMAL_DIR=${NORMAL_DIR:-dataset/eval3d_hunyuan3d/normals}
LOG_DIR=${LOG_DIR:-logs/direct3d_s2_grpo_multi}
RUN_NAME=${RUN_NAME:-direct3d_s2_grpo_multi}

PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-30}
NUM_CAND=${NUM_CAND:-12}
GUIDANCE=${GUIDANCE:-7.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}
EPOCHS=${EPOCHS:-500}
TRAIN_BS=${TRAIN_BS:-6} #-${NUM_CAND}}
GRAD_ACCUM=${GRAD_ACCUM:-2}
SAVE_FREQ=${SAVE_FREQ:-1}

# 统计与优势类型（默认关闭/默认 winrate）
PER_IMAGE_STAT_TRACKING=${PER_IMAGE_STAT_TRACKING:-false}
GLOBAL_STD=${GLOBAL_STD:-false}
ADV_TYPE=${ADV_TYPE:-winrate}

# 评测相关（eval-only 开关）
EVAL_ONLY=${EVAL_ONLY:-false}

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
echo "   PER_IMAGE_STAT_TRACKING=${PER_IMAGE_STAT_TRACKING}"
echo "   GLOBAL_STD=${GLOBAL_STD}"
echo "   ADV_TYPE=${ADV_TYPE}"

"${ACC_PY}" -m accelerate.commands.launch \
  --num_processes=${GPU_COUNT} \
  --main_process_port=29612 \
  scripts/train_direct3d_s2.py \
  --config config/direct3d_s2_grpo_normal-sim.py \
  --config.data_dir="${DATA_DIR}" \
  --config.camera_normal.cache_dir="${NORMAL_DIR}" \
  --config.logdir="${LOG_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_steps=${NUM_STEPS} \
  --config.sample.num_meshes_per_image=${NUM_CAND} \
  --config.sample.num_batches_per_epoch=${NUM_BATCHES_PER_EPOCH} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.sample.adv_type="${ADV_TYPE}" \
  --config.sample.global_std=${GLOBAL_STD} \
  --config.per_image_stat_tracking=${PER_IMAGE_STAT_TRACKING} \
  --config.pretrained.pipeline_path="${PRETRAIN_DIR}" \
  --config.pretrained.subfolder="${PRETRAIN_SUBFOLDER}" \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true

echo "✅ Direct3D‑S2 multi-GPU training skeleton started. Logs: ${LOG_DIR}" 
