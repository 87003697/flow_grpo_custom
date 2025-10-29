#!/bin/bash

# 单机单卡：Direct3D‑S2 Stage 2 GRPO 训练启动脚本
# 约束：仅训练 sparse_dit_512；无 try/except；无 fallback；conda 环境应为 grpo3d
#
# 用法示例（建议在 grpo3d 环境中执行）：
#   conda activate grpo3d
#   NORMAL_DIR=dataset/eval3d_hunyuan3d/normals \
#   LOG_DIR=logs/direct3d_stage2_grpo_single \
#   RUN_NAME=direct3d_stage2_grpo_single \
#   PRETRAIN_DIR=pretrained_weights/direct3d_s2-v-1-1 \
#   INPUT_BS=1 NUM_STEPS=20 NUM_CAND=1 GUIDANCE=3.0 \
#   EPOCHS=1 TRAIN_BS=1 GRAD_ACCUM=1 SAVE_FREQ=1 \
#   bash scripts/single_node/main_direct3d_stage-2_alpha-1k.sh

set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse

# 选择 GPU（按需修改）
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# 数据与输出（按需修改）
TRAIN_DIR=${TRAIN_DIR:-dataset/alphaimages_1k/train}
EVAL_DIR=${EVAL_DIR:-dataset/alphaimages_1k/test}
LOG_DIR=${LOG_DIR:-logs/direct3d_stage2_grpo_single}
RUN_NAME=${RUN_NAME:-direct3d_debug_dummy}

# 预训练（Direct3D‑S2 权重路径）
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

# 采样与训练配置（内存友好，符合规则：batch 1-2）
INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-10}
NUM_CAND=${NUM_CAND:-1}
GUIDANCE=${GUIDANCE:-3.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}

EPOCHS=${EPOCHS:-1}
TRAIN_BS=${TRAIN_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
SAVE_FREQ=${SAVE_FREQ:-1}
LR=${LR:-2e-5}

# KL 正则系数（对应 config.train.beta），默认 0 以保持原行为不启用
KL_BETA=${KL_BETA:-0.0}

# PPO：是否对无条件分支 detach（对应 config.train.detach_uncond）
DETACH_UNCOND=${DETACH_UNCOND:-false}

# 优势类型（默认 winrate，可 similarity）
ADV_TYPE=${ADV_TYPE:-winrate}  # 可选: similarity, winrate_plus


# 统一奖励开关（通过环境变量切换 Dummy / Uni3D / CameraNormal）
# Debug 默认使用 Dummy（低显存友好）
REWARD_DUMMY=${REWARD_DUMMY:-1.0}
REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL:-0.0}
REWARD_UNI3D=${REWARD_UNI3D:-0.0}

# 是否启用 EMA（对应 config.train.ema）
USE_EMA=${USE_EMA:-false}

# 打印配置
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "   TRAIN_DIR=${TRAIN_DIR}"
echo "   EVAL_DIR=${EVAL_DIR}"
echo "   NUM_CAND=${NUM_CAND}"
echo "   NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH}"
echo "   EPOCHS=${EPOCHS}"
echo "   TRAIN_BS=${TRAIN_BS}"
echo "   GRAD_ACCUM=${GRAD_ACCUM}"
echo "   SAVE_FREQ=${SAVE_FREQ}"
echo "   LR=${LR}"
echo "   PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "   ADV_TYPE=${ADV_TYPE}"
echo "   REWARD_DUMMY=${REWARD_DUMMY}"
echo "   REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL}"
echo "   REWARD_UNI3D=${REWARD_UNI3D}"
echo "   USE_EMA=${USE_EMA}"
echo "   KL_BETA=${KL_BETA}"
echo "   DETACH_UNCOND=${DETACH_UNCOND}"

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

accelerate launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  --num_processes=1 \
  --main_process_port=29517 \
  scripts/train_direct3d_s2.py \
  --config config/direct3d_s2_stage-2_grpo_normal-sim_alpha-1k.py \
  --config.train_data_dir="${TRAIN_DIR}" \
  --config.eval_data_dir="${EVAL_DIR}" \
  --config.reward_fn.dummy=${REWARD_DUMMY} \
  --config.reward_fn.camera_normal=${REWARD_CAMERA_NORMAL} \
  --config.reward_fn.uni3d=${REWARD_UNI3D} \
  --config.logdir="${LOG_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_steps=${NUM_STEPS} \
  --config.sample.num_meshes_per_image=${NUM_CAND} \
  --config.sample.num_batches_per_epoch=${NUM_BATCHES_PER_EPOCH} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.sample.adv_type="${ADV_TYPE}" \
  --config.pretrained.pipeline_path="${PRETRAIN_DIR}" \
  --config.pretrained.subfolder="${PRETRAIN_SUBFOLDER}" \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.train.learning_rate=${LR} \
  --config.train.beta=${KL_BETA} \
  --config.train.detach_uncond=${DETACH_UNCOND} \
  --config.train.ema=${USE_EMA} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY:-false} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true

echo "✅ Direct3D‑S2 Stage 2 GRPO started. Logs: ${LOG_DIR} | CKPT: ${LOG_DIR}/${RUN_NAME}/checkpoints"


