#!/bin/bash
# 多 GPU Direct3D‑S2 GRPO (sparse512) 最小启动脚本
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0,1}"; export CUDA_VISIBLE_DEVICES
GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

DATA_DIR=${DATA_DIR:-dataset/eval3d_hunyuan3d}
NORMAL_DIR=${NORMAL_DIR:-dataset/eval3d_hunyuan3d/normals}
LOG_DIR=${LOG_DIR:-logs/direct3d_s2_grpo_multi}
RUN_NAME=${RUN_NAME:-direct3d_s2_grpo_multi}

INPUT_BS=${INPUT_BS:-1}
DENSE_STEPS=${DENSE_STEPS:-50}
SPARSE_STEPS=${SPARSE_STEPS:-30}
NUM_CAND=${NUM_CAND:-2}
GUIDANCE=${GUIDANCE:-3.0}
SIGMA_MIN=${SIGMA_MIN:-0.002}
RESCALE_T=${RESCALE_T:-1.0}
EPOCHS=${EPOCHS:-5}
TRAIN_BS=${TRAIN_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAVE_FREQ=${SAVE_FREQ:-1}

echo "[Direct3D-S2 Multi] DEVICES=$CUDA_VISIBLE_DEVICES | GPUs=$GPU_COUNT" 

accelerate launch \
  --multi_gpu \
  --num_processes=${GPU_COUNT} \
  --main_process_port=29612 \
  scripts/train_direct3d_s2.py \
  --config config/direct3d_s2_grpo_normal-sim.py \
  --config.data_dir="${DATA_DIR}" \
  --config.camera_normal.cache_dir="${NORMAL_DIR}" \
  --config.logdir="${LOG_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.input_batch_size=${INPUT_BS} \
  --config.sample.num_inference_steps_dense=${DENSE_STEPS} \
  --config.sample.num_inference_steps_sparse512=${SPARSE_STEPS} \
  --config.sample.num_candidates=${NUM_CAND} \
  --config.sample.guidance_scale=${GUIDANCE} \
  --config.sample.sigma_min=${SIGMA_MIN} \
  --config.sample.rescale_t=${RESCALE_T} \
  --config.train.batch_size=${TRAIN_BS} \
  --config.train.gradient_accumulation_steps=${GRAD_ACCUM} \
  --config.num_epochs=${EPOCHS} \
  --config.save_freq=${SAVE_FREQ} \
  --config.mixed_precision=bf16

echo "✅ Direct3D‑S2 multi-GPU training skeleton started. Logs: ${LOG_DIR}" 
