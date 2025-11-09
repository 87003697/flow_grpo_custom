#!/bin/bash

# 单机单卡：Direct3D‑S2 Stage 2 GRPO 训练启动脚本
# 约束：仅训练 sparse_dit_512；无 try/except；无 fallback；conda 环境应为 grpo3d
#
# 用法示例（建议在 grpo3d 环境中执行）：
#   conda activate grpo3d
#   DATA_DIR=dataset/eval3d_hunyuan3d \
#   NORMAL_DIR=dataset/eval3d_hunyuan3d/normals \
#   LOG_DIR=logs/direct3d_stage2_grpo_single \
#   RUN_NAME=direct3d_stage2_grpo_single \
#   PRETRAIN_DIR=pretrained_weights/direct3d_s2-v-1-1 \
#   INPUT_BS=1 NUM_STEPS=20 NUM_CAND=1 GUIDANCE=3.0 \
#   EPOCHS=1 TRAIN_BS=1 GRAD_ACCUM=1 SAVE_FREQ=1 \
#   bash scripts/single_node/main_direct3d_normal-sim.sh

set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 选择 GPU（按需修改）
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# 数据与输出（按需修改）
DATA_DIR=${DATA_DIR:-dataset/eval3d_hunyuan3d}
NORMAL_DIR=${NORMAL_DIR:-dataset/eval3d_hunyuan3d/normals}
LOG_DIR=${LOG_DIR:-logs/direct3d_stage2_grpo_single}
RUN_NAME=${RUN_NAME:-direct3d_stage2_grpo}

# 预训练（Direct3D‑S2 权重路径）
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

# 采样与训练配置（内存友好，符合规则：batch 1-2）
INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-30}
NUM_CAND=${NUM_CAND:-8}
GUIDANCE=${GUIDANCE:-7.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}

EPOCHS=${EPOCHS:-10}
TRAIN_BS=${TRAIN_BS:-${NUM_CAND}}
GRAD_ACCUM=${GRAD_ACCUM:-$((NUM_CAND / TRAIN_BS))}
SAVE_FREQ=${SAVE_FREQ:-1}
DINO_SIM_TYPE=${DINO_SIM_TYPE:-dense}
LR=${LR:-3e-4}

# View 编码器选择：dino_v2 / dino_v3 / pickscore
# 默认 dino_v3；设为 pickscore 可走 CLIP 全局特征余弦（需已下载本地权重）
VIEW_ENCODER=${VIEW_ENCODER:-dino_v3}

# PPO 裁剪范围（对称）：控制 config.train.clip_range
CLIP_RANGE=${CLIP_RANGE:-0.02}

# 采样噪声强度：控制 config.slat_sampler_params.noise_level（SDE 随机性）
NOISE_LEVEL=${NOISE_LEVEL:-0.7}

# PPO：是否对无条件分支 detach（对应 config.train.detach_uncond）
DETACH_UNCOND=${DETACH_UNCOND:-false}

# 评测相关（eval-only 开关与测试批大小、可选 ckpt）
EVAL_ONLY=${EVAL_ONLY:-false}
TEST_BS=${TEST_BS:-8}
CHECKPOINT=${CHECKPOINT:-}

# 统计与优势类型（默认 similarity）
ADV_TYPE=${ADV_TYPE:-similarity}  # 可选: similarity, winrate, winrate_plus
# 优势来源（逐子项 seperate / 加权总分 average）
ADV_FROM=${ADV_FROM:-average}
# CameraNormal：组内均值相机开关（默认 false）
AVG_CAMERA_PER_GROUP=${AVG_CAMERA_PER_GROUP:-false}

# CameraNormal：是否使用 RGB 组进行比较（默认 false）
USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON:-false}

# SDE/Flow 参数：sigma_min/rescale_t 已移除；仅保留 use_sde/mc_threshold（如需）

echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "   DATA_DIR=${DATA_DIR}"
echo "   NUM_CAND=${NUM_CAND}"
echo "   NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH}"
echo "   EPOCHS=${EPOCHS}"
echo "   TRAIN_BS=${TRAIN_BS}"
echo "   GRAD_ACCUM=${GRAD_ACCUM}"
echo "   SAVE_FREQ=${SAVE_FREQ}"
echo "   LR=${LR}"
echo "   CLIP_RANGE=${CLIP_RANGE}"
echo "   NOISE_LEVEL=${NOISE_LEVEL}"
echo "   PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "   EVAL_ONLY=${EVAL_ONLY} | TEST_BS=${TEST_BS} | CHECKPOINT=${CHECKPOINT}"
echo "   VIEW_ENCODER=${VIEW_ENCODER}"
echo "   ADV_TYPE=${ADV_TYPE}"
echo "   ADV_FROM=${ADV_FROM}"
echo "   DETACH_UNCOND=${DETACH_UNCOND}"
echo "   USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON}"

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

# 可选：仅当 CHECKPOINT 非空时传递覆盖参数
CKPT_ARG=()
if [ -n "${CHECKPOINT}" ]; then
  CKPT_ARG=(--config.checkpoint="${CHECKPOINT}")
fi

"${ACC_PY}" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  --num_processes=1 \
  --main_process_port=29527 \
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
  --config.train.detach_uncond=${DETACH_UNCOND} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true \
  "${CKPT_ARG[@]}"