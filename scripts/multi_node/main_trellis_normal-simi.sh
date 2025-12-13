#!/bin/bash


# 多GPU：TRELLIS Stage 2 GRPO 训练启动脚本（参考 scripts/single_node/main_trellis.sh）
# 说明：单机多卡（默认8卡）。权重将保存到：$LOG_DIR/$RUN_NAME/checkpoints/ckpt_<E>

set -euo pipefail

export ATTN_BACKEND=xformers
: "${HF_HUB_OFFLINE:=1}"
export HF_HUB_OFFLINE
export SPCONV_ALGO=native
echo "SPCONV_ALGO=$SPCONV_ALGO"

# W&B 在线配置（与 Hunyuan3D 一致：通过环境变量提供同一密钥）
export WANDB_API_KEY=cd27ab683cc7cc900fd6b8172132c99a35775d73
export WANDB_MODE=online
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[ERROR] WANDB_API_KEY 未设置。请先在外部 export WANDB_API_KEY=... 再运行本脚本。"
  exit 1
fi

# 选择 GPU（默认使用 0-7 共8卡；可在外部覆写 CUDA_VISIBLE_DEVICES）
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

# 数据与输出（可覆写）
DATA_DIR=${DATA_DIR:-dataset/eval3d_hunyuan3d}
NORMAL_DIR=${NORMAL_DIR:-dataset/eval3d_hunyuan3d/normals}
LOG_DIR=${LOG_DIR:-logs/trellis_stage2_grpo_multi}
RUN_NAME=${RUN_NAME:-trellis_stage2_grpo_multi}

# 采样与训练配置（与单卡脚本对齐）
INPUT_BS=${INPUT_BS:-1}
NUM_STEPS=${NUM_STEPS:-50}
NUM_CAND=${NUM_CAND:-16}
GUIDANCE=${GUIDANCE:-3.0}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-1}

EPOCHS=${EPOCHS:-1000}
TRAIN_BS=${TRAIN_BS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
SAVE_FREQ=${SAVE_FREQ:-1}

# Optional optimizer type override
OPT_TYPE=${OPT_TYPE:-adam_8bit}

# SDE/Flow 参数
SIGMA_MIN=${SIGMA_MIN:-0.2}
RESCALE_T=${RESCALE_T:-1.0}

# 计算实际可用 GPU 数
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "🚀 TRELLIS Stage 2 GRPO 多卡启动 | GPUs=${GPU_COUNT} | DEVICES=${CUDA_VISIBLE_DEVICES}"

ACC_PY=$(which python)
"${ACC_PY}" -m accelerate.commands.launch \
  --num_processes=${GPU_COUNT} \
  --multi_gpu \
  --main_process_port=29508 \
  scripts/train_trellis.py \
  --config config/trellis_stage2_grpo_normal-sim.py \
  --config.data_dir="${DATA_DIR}" \
  --config.camera_normal.cache_dir="${NORMAL_DIR}" \
  --config.logdir="${LOG_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_steps=${NUM_STEPS} \
  --config.sample.num_meshes_per_image=${NUM_CAND} \
  --config.sample.num_batches_per_epoch=${NUM_BATCHES_PER_EPOCH} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.slat_sampler_params.sigma_min=${SIGMA_MIN} \
  --config.slat_sampler_params.rescale_t=${RESCALE_T} \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.mixed_precision=no \
  --config.deterministic=true \
  ${OPT_TYPE:+--config.train.optimizer.type=${OPT_TYPE}}

echo "✅ 已启动 | 日志: ${LOG_DIR} | 检查点: ${LOG_DIR}/${RUN_NAME}/checkpoints"

