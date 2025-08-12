#!/bin/bash

# 单机单卡：TRELLIS Stage 2 GRPO 训练启动脚本（遵循 TRELLIS_DEV.md）
# 约束：仅训练 SLatFlowModel；无 try/except；无 fallback；conda 环境应为 grpo3d
#
# 用法示例（建议在 grpo3d 环境中执行）：
#   conda activate grpo3d
#   DATA_DIR=dataset/eval3d \
#   SAVE_DIR=checkpoints/trellis_stage2_grpo_single \
#   LOG_DIR=logs/trellis_stage2_grpo_single \
#   INPUT_BS=1 NUM_STEPS=20 NUM_CAND=1 GUIDANCE=3.0 \
#   EPOCHS=1 TRAIN_BS=1 GRAD_ACCUM=1 SAVE_FREQ=1 \
#   SIGMA_MIN=0.002 RESCALE_T=1.0 \
#   bash scripts/single_node/main_trellis.sh

set -euo pipefail

export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export HF_HUB_OFFLINE=1

# 选择 GPU（按需修改）
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# 数据与输出（按需修改）
DATA_DIR=${DATA_DIR:-dataset/eval3d}
SAVE_DIR=${SAVE_DIR:-checkpoints/trellis_stage2_grpo_single}
LOG_DIR=${LOG_DIR:-logs/trellis_stage2_grpo_single}

# 采样与训练配置（内存友好，符合规则：batch 1-2）
INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-20}
NUM_CAND=${NUM_CAND:-16}
GUIDANCE=${GUIDANCE:-3.0}

EPOCHS=${EPOCHS:-10}
TRAIN_BS=${TRAIN_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-2}
SAVE_FREQ=${SAVE_FREQ:-1}

# SDE/Flow 参数
SIGMA_MIN=${SIGMA_MIN:-0.002}
RESCALE_T=${RESCALE_T:-1.0}

echo "🚀 Launch TRELLIS Stage 2 GRPO (single GPU)"
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "   DATA_DIR=${DATA_DIR}"
echo "   NUM_CAND=${NUM_CAND}"
echo "   EPOCHS=${EPOCHS}"
echo "   TRAIN_BS=${TRAIN_BS}"
echo "   GRAD_ACCUM=${GRAD_ACCUM}"
echo "   SAVE_FREQ=${SAVE_FREQ}"

accelerate launch \
  --config_file scripts/accelerate_configs/single_gpu.yaml \
  --num_processes=0 \
  --main_process_port=29507 \
  scripts/train_trellis.py \
  --config config/trellis_stage2_grpo.py \
  --config.data_dir="${DATA_DIR}" \
  --config.logdir="${LOG_DIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_steps=${NUM_STEPS} \
  --config.sample.num_meshes_per_image=${NUM_CAND} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.sparse_structure_sampler_params.max_points=4096 \
  --config.slat_sampler_params.sigma_min=${SIGMA_MIN} \
  --config.slat_sampler_params.rescale_t=${RESCALE_T} \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.mixed_precision=bf16 \
  --config.deterministic=false

echo "✅ TRELLIS Stage 2 GRPO started. Logs: ${LOG_DIR} | CKPT: ${SAVE_DIR}"


