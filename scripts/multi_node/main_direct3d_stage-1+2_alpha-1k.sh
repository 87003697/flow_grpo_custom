#!/bin/bash

# 多机多卡：Direct3D‑S2 Stage 1+2 GRPO 联训启动脚本（AlphaImages 1k 配置）
# 约束：同时训练 dense_dit 与 sparse_dit_512；无 try/except；无 fallback；严格数据与法线缓存配置
# 使用 accelerate 多机配置：scripts/accelerate_configs/multi_node.yaml（请按集群修改其中 IP/端口/机器数/进程数）
#
# 用法示例（每台机器分别执行；确保多机 YAML 配置一致且可互访）：
#   TRAIN_DIR=dataset/alphaimages_1k/train \
#   EVAL_DIR=dataset/alphaimages_1k/test \
#   TRAIN_NORMAL_DIR=dataset/alphaimages_1k/train/normals \
#   EVAL_NORMAL_DIR=dataset/alphaimages_1k/test/normals \
#   LOG_DIR=logs/direct3d_stage1+2_grpo_multi \
#   RUN_NAME=direct3d_stage1+2_grpo_multi \
#   PRETRAIN_DIR=pretrained_weights/direct3d_s2-v-1-1 \
#   INPUT_BS=1 NUM_STEPS=30 NUM_CAND=8 GUIDANCE=7.0 \
#   EPOCHS=10 TRAIN_BS=8 GRAD_ACCUM=2 SAVE_FREQ=1 \
#   bash scripts/multi_node/main_direct3d_stage-1+2_alpha-1k.sh

set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# 数据与输出（按需覆盖）
TRAIN_DIR=${TRAIN_DIR:-dataset/alphaimages_1k/train}
EVAL_DIR=${EVAL_DIR:-dataset/alphaimages_1k/test}
TRAIN_NORMAL_DIR=${TRAIN_NORMAL_DIR:-dataset/alphaimages_1k/train/normals}
EVAL_NORMAL_DIR=${EVAL_NORMAL_DIR:-dataset/alphaimages_1k/test/normals}
NORMAL_RES=${NORMAL_RES:-518}
LOG_DIR=${LOG_DIR:-logs/direct3d_stage1+2_grpo_multi}
RUN_NAME=${RUN_NAME:-direct3d_stage1+2_grpo_multi}

# DINO 相似度模式（与 CameraNormal 评分器一致；当 camera_normal>0 时生效）
# 可选值：cls, dense, dense_all, match_gird2pixel, match_pixel
# 示例：dense_all 全层 tokens
DINO_SIMILARITY_TYPE=${DINO_SIMILARITY_TYPE:-dense_all}

# 预训练（Direct3D‑S2 权重路径）
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

# 采样与训练配置
INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-30}
NUM_CAND=${NUM_CAND:-8}
GUIDANCE=${GUIDANCE:-7.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}
EPOCHS=${EPOCHS:-500}
TRAIN_BS=${TRAIN_BS:-4} #-${NUM_CAND}}
GRAD_ACCUM=${GRAD_ACCUM:-$((NUM_CAND / TRAIN_BS))}
SAVE_FREQ=${SAVE_FREQ:-1}
LR=${LR:-3e-4}
OPT_TYPE=${OPT_TYPE:-adam_8bit}

# PPO 裁剪范围（对称）：控制 config.train.clip_range
CLIP_RANGE=${CLIP_RANGE:-0.02}

# 采样噪声强度：控制 config.slat_sampler_params.noise_level（SDE 随机性）
NOISE_LEVEL=${NOISE_LEVEL:-0.7}

# KL 正则系数（对应 config.train.beta），默认 0 以保持原行为不启用
KL_BETA=${KL_BETA:-0.0}

# PPO：是否对无条件分支 detach（对应 config.train.detach_uncond）
DETACH_UNCOND=${DETACH_UNCOND:-false}

# 优势类型（默认 similarity，可 winrate）
ADV_TYPE=${ADV_TYPE:-similarity}  # 可选: similarity, winrate_plus

# 优势来源（逐子项 seperate / 加权总分 average）
ADV_FROM=${ADV_FROM:-average}

# 统一奖励权重（通过环境变量切换 Uni3D / CameraNormal）
# 确保至少有一个 > 0，否则训练将报错
REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL:-1.0}
REWARD_UNI3D=${REWARD_UNI3D:-0.0}

# CameraNormal：组内均值相机开关（默认 false）
AVG_CAMERA_PER_GROUP=${AVG_CAMERA_PER_GROUP:-false}

# CameraNormal：是否使用 RGB 组进行比较（默认 false）
USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON:-false}

# 是否启用 EMA（对应 config.train.ema）
USE_EMA=${USE_EMA:-false}


# 评测相关（eval-only 开关）
EVAL_ONLY=${EVAL_ONLY:-false}

# 可选：resume 的 checkpoint 根目录（指向包含 checkpoint_*/ 的目录或具体 checkpoint_* 目录）
CHECKPOINT=${CHECKPOINT:-}

echo "   TRAIN_DIR=${TRAIN_DIR}"
echo "   EVAL_DIR=${EVAL_DIR}"
echo "   TRAIN_NORMAL_DIR=${TRAIN_NORMAL_DIR}"
echo "   EVAL_NORMAL_DIR=${EVAL_NORMAL_DIR}"
echo "   NORMAL_RES=${NORMAL_RES}"
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
echo "   DINO_SIMILARITY_TYPE=${DINO_SIMILARITY_TYPE}"
echo "   ADV_TYPE=${ADV_TYPE}"
echo "   ADV_FROM=${ADV_FROM}"
echo "   REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL}"
echo "   REWARD_UNI3D=${REWARD_UNI3D}"
echo "   AVG_CAMERA_PER_GROUP=${AVG_CAMERA_PER_GROUP}"
echo "   USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON}"
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

echo "[Direct3D-S2 Multi] DEVICES=$CUDA_VISIBLE_DEVICES | GPUs=$GPU_COUNT" 

# 组装可选参数（如 CHECKPOINT）
EXTRA_ARGS=()
if [ -n "${CHECKPOINT}" ]; then
  EXTRA_ARGS+=("--config.checkpoint=${CHECKPOINT}")
fi
if [ -n "${OPT_TYPE}" ]; then
  EXTRA_ARGS+=("--config.train.optimizer.type=${OPT_TYPE}")
fi

accelerate launch \
  --num_processes=${GPU_COUNT} \
  --main_process_port=29612 \
    scripts/train_direct3d_s2_stage-1+2.py \
  --config config/direct3d_s2_stage-2_grpo_normal-sim_alpha-1k.py \
  --config.train_data_dir="${TRAIN_DIR}" \
  --config.eval_data_dir="${EVAL_DIR}" \
  --config.camera_normal_train.cache_dir="${TRAIN_NORMAL_DIR}" \
  --config.camera_normal_train.normal_resolution=${NORMAL_RES} \
  --config.camera_normal_eval.cache_dir="${EVAL_NORMAL_DIR}" \
  --config.camera_normal_eval.normal_resolution=${NORMAL_RES} \
  --config.reward_fn.camera_normal=${REWARD_CAMERA_NORMAL} \
  --config.reward_fn.uni3d=${REWARD_UNI3D} \
  --config.camera_normal.avg_camera_per_group=${AVG_CAMERA_PER_GROUP} \
  --config.camera_normal.use_RGB_for_comparison=${USE_RGB_FOR_COMPARISON} \
  --config.camera_normal.dino_similarity_type="${DINO_SIMILARITY_TYPE}" \
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
  --config.train.beta=${KL_BETA} \
  --config.train.detach_uncond=${DETACH_UNCOND} \
  --config.train.ema=${USE_EMA} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true \
  "${EXTRA_ARGS[@]}"

echo "✅ Direct3D‑S2 Stage 1+2 GRPO (multi-node) started. Logs: ${LOG_DIR} | CKPT: ${LOG_DIR}/${RUN_NAME}/checkpoints"


