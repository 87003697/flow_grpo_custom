#!/bin/bash

# 单机单卡：Direct3D‑S2 Stage 1+2 GRPO 联训启动脚本
# 约束：同时训练 dense_dit 与 sparse_dit_512；无 try/except；无 fallback；conda 环境应为 grpo3d
#
# 用法示例（建议在 grpo3d 环境中执行）：
#   conda activate grpo3d
#   NORMAL_DIR=dataset/eval3d_hunyuan3d/normals \
#   LOG_DIR=logs/direct3d_stage1+2_grpo_single \
#   RUN_NAME=direct3d_stage1+2_grpo_single \
#   PRETRAIN_DIR=pretrained_weights/direct3d_s2-v-1-1 \
#   INPUT_BS=1 NUM_STEPS=20 NUM_CAND=1 GUIDANCE=3.0 \
#   EPOCHS=1 TRAIN_BS=1 GRAD_ACCUM=1 SAVE_FREQ=1 \
#   bash scripts/single_node/main_direct3d_stage-2_alpha-1k.sh

set -euo pipefail

export ATTN_BACKEND=flash_attn
export HF_HUB_OFFLINE=1
export SPCONV_ALGO=implicit_gemm
export SPARSE_BACKEND=torchsparse
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 选择 GPU（按需修改）
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# 数据与输出（按需修改，严格区分训练/评估目录与法线缓存）
TRAIN_DIR=${TRAIN_DIR:-dataset/alphaimages_1k/train}
EVAL_DIR=${EVAL_DIR:-dataset/alphaimages_1k/test}
TRAIN_NORMAL_DIR=${TRAIN_NORMAL_DIR:-dataset/alphaimages_1k/train/normals}
EVAL_NORMAL_DIR=${EVAL_NORMAL_DIR:-dataset/alphaimages_1k/test/normals}
NORMAL_RES=${NORMAL_RES:-518}
LOG_DIR=${LOG_DIR:-logs/direct3d_stage1+2_grpo_single}
RUN_NAME=${RUN_NAME:-direct3d_stage1+2_grpo}

# DINO 相似度模式接口（当 camera_normal>0 时生效）
# 可选：cls, dense, dense_all, match_gird2pixel, match_pixel
# 示例：启用 dense_all（全层 tokens）
#   DINO_SIMILARITY_TYPE=dense_all \
DINO_SIMILARITY_TYPE=${DINO_SIMILARITY_TYPE:-dense_all}

# View 编码器选择：dino_v2 / dino_v3 / pickscore / clip / hpsv2
# 默认 dino_v3；亦已适配 hpsv2（需本地权重与 config.camera_normal.hpsv2_ckpt_path）；设为 pickscore 可走 CLIP 全局特征余弦
VIEW_ENCODER=${VIEW_ENCODER:-dino_v3}

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
TRAIN_BS=${TRAIN_BS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-$((NUM_CAND / TRAIN_BS))}
SAVE_FREQ=${SAVE_FREQ:-1}
LR=${LR:-3e-4}
OPT_TYPE=${OPT_TYPE:-adam_8bit}

# 采样噪声强度：控制 config.slat_sampler_params.noise_level（SDE 随机性）
NOISE_LEVEL=${NOISE_LEVEL:-0.7}

# 时序保留比例：config.train.timestep_keep_ratio
KEEP_RATIO=${KEEP_RATIO:-1.0}

# DiffusionNFT: LoRA adapter 融合的衰减类型（0/1/2）
DECAY_TYPE=${DECAY_TYPE:-2}

# DiffusionNFT：正负样本融合的裁剪系数（config.train.adv_clip_max）
ADV_CLIP_MAX=${ADV_CLIP_MAX:-2.0}

# DiffusionNFT：正负策略混合系数（config.nft_beta，与 KL 系数独立）
NFT_BETA=${NFT_BETA:-1.0}

# KL 正则系数（对应 config.train.beta），默认 0 以保持原行为不启用
KL_BETA=${KL_BETA:-0.0}

# 优势类型（默认 similarity，可 winrate）
ADV_TYPE=${ADV_TYPE:-similarity}  # 可选: similarity, winrate_plus

# 优势来源（逐子项 seperate / 加权总分 average）
ADV_FROM=${ADV_FROM:-average}


# 统一奖励开关（通过环境变量切换 Uni3D / CameraNormal）
REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL:-1.0}
REWARD_UNI3D=${REWARD_UNI3D:-0.0}

# CameraNormal：组内均值相机开关（默认 false）
AVG_CAMERA_PER_GROUP=${AVG_CAMERA_PER_GROUP:-false}

# CameraNormal：是否使用 RGB 组进行比较（默认 false）
USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON:-false}

# CameraNormal：相机模式；search=VGGT 搜索，fixed_v1=固定 4 视角，fixed_v0=单视角；
#               camera_type 包含 "_max" 时奖励改为多视角取最大值
CAMERA_TYPE=${CAMERA_TYPE:-search}

# 是否启用 EMA（对应 config.train.ema）
USE_EMA=${USE_EMA:-false}

# 打印配置
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
echo "   NOISE_LEVEL=${NOISE_LEVEL}"
echo "   KEEP_RATIO=${KEEP_RATIO}"
echo "   DECAY_TYPE=${DECAY_TYPE}"
echo "   ADV_CLIP_MAX=${ADV_CLIP_MAX}"
echo "   PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "   DINO_SIMILARITY_TYPE=${DINO_SIMILARITY_TYPE}"
echo "   VIEW_ENCODER=${VIEW_ENCODER}"
echo "   ADV_TYPE=${ADV_TYPE}"
echo "   ADV_FROM=${ADV_FROM}"
echo "   REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL}"
echo "   REWARD_UNI3D=${REWARD_UNI3D}"
echo "   AVG_CAMERA_PER_GROUP=${AVG_CAMERA_PER_GROUP}"
echo "   USE_RGB_FOR_COMPARISON=${USE_RGB_FOR_COMPARISON}"
echo "   CAMERA_TYPE=${CAMERA_TYPE}"
echo "   USE_EMA=${USE_EMA}"
echo "   KL_BETA=${KL_BETA}"
echo "   NFT_BETA=${NFT_BETA}"

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

EXTRA_ARGS=()
if [ -n "${CHECKPOINT:-}" ]; then
  EXTRA_ARGS+=("--config.checkpoint=${CHECKPOINT}")
fi
if [ -n "${OPT_TYPE}" ]; then
  EXTRA_ARGS+=("--config.train.optimizer.type=${OPT_TYPE}")
fi

$ACC_PY -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  --num_processes=1 \
  --main_process_port=29517 \
  scripts/train_direct3d_s2_stage-1+2_nft.py \
  --config config/direct3d_s2_diffusion-nft_normal-sim_alpha-1k.py \
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
  --config.camera_normal.camera_type="${CAMERA_TYPE}" \
  --config.camera_normal.encoder="${VIEW_ENCODER}" \
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
  --config.slat_sampler_params.noise_level=${NOISE_LEVEL} \
  --config.train.timestep_keep_ratio=${KEEP_RATIO} \
  --config.train.decay_type=${DECAY_TYPE} \
  --config.train.adv_clip_max=${ADV_CLIP_MAX} \
  --config.train.beta=${KL_BETA} \
  --config.nft_beta=${NFT_BETA} \
  --config.train.ema=${USE_EMA} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.eval_only=${EVAL_ONLY:-false} \
  --config.mixed_precision=bf16 \
  --config.deterministic=true \
  "${EXTRA_ARGS[@]}"

echo "✅ Direct3D‑S2 Stage 1+2 GRPO started. Logs: ${LOG_DIR} | CKPT: ${LOG_DIR}/${RUN_NAME}/checkpoints"


