#!/bin/bash

# Direct3D‑S2 eval-only 最小测试脚本（单机单卡）
# 使用方式（示例）：
#   conda activate grpo3d
#   EVAL_DIR=dataset/eval3d_hunyuan3d/images \
#   EVAL_NORMAL_DIR=dataset/eval3d_hunyuan3d/normals \
#   PRETRAIN_DIR=pretrained_weights/direct3d_s2-v-1-1 \
#   LOG_DIR=logs \
#   RUN_NAME=direct3d_s2_eval \
#   bash scripts/eval/direct3d_s2.sh

set -euo pipefail

# 选择 GPU（按需覆盖）
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# 推断进程数（NUM_PROCS）：优先环境变量；否则按 CUDA_VISIBLE_DEVICES 的数量
NUM_PROCS=${NUM_PROCS:-}
if [ -z "${NUM_PROCS}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_PROCS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
  else
    NUM_PROCS=1
  fi
fi
MAIN_PORT=${MAIN_PORT:-29518}

# 基本路径（可用环境变量覆盖）
EVAL_DIR=${EVAL_DIR:-dataset/alphaimages_1k/test}
EVAL_NORMAL_DIR=${EVAL_NORMAL_DIR:-dataset/alphaimages_1k/test/normals}
LOG_DIR=${LOG_DIR:-logs}
RUN_NAME=${RUN_NAME:-direct3d_s2_eval}
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrained_weights/direct3d_s2-v-1-1}
PRETRAIN_SUBFOLDER=${PRETRAIN_SUBFOLDER:-direct3d-s2-v-1-1}

# 推理参数
TEST_BS=${TEST_BS:-1}
NUM_STEPS=${NUM_STEPS:-30}
GUIDANCE=${GUIDANCE:-7.0}
NORMAL_RES=${NORMAL_RES:-518}

# 奖励开关（默认仅 camera_normal）
REWARD_CAMERA_NORMAL=${REWARD_CAMERA_NORMAL:-1.0}
REWARD_UNI3D=${REWARD_UNI3D:-0.0}

echo "EVAL_DIR=${EVAL_DIR}"
echo "EVAL_NORMAL_DIR=${EVAL_NORMAL_DIR}"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "NUM_STEPS=${NUM_STEPS} GUIDANCE=${GUIDANCE} TEST_BS(per-rank)=${TEST_BS} NUM_PROCS=${NUM_PROCS}"

if [ "${NUM_PROCS}" -gt 1 ]; then
  # 多卡：使用 accelerate 启动多进程；TEST_BS 为每卡 batch size
  accelerate launch \
    --num_processes="${NUM_PROCS}" \
    --main_process_port="${MAIN_PORT}" \
    scripts/train_direct3d_s2.py \
    --config config/direct3d_s2_grpo_normal-sim.py \
    --config.eval_only=true \
    --config.eval_data_dir="${EVAL_DIR}" \
    --config.camera_normal_eval.cache_dir="${EVAL_NORMAL_DIR}" \
    --config.camera_normal_eval.normal_resolution=${NORMAL_RES} \
    --config.pretrained.pipeline_path="${PRETRAIN_DIR}" \
    --config.pretrained.subfolder="${PRETRAIN_SUBFOLDER}" \
    --config.logdir="${LOG_DIR}" \
    --config.run_name="${RUN_NAME}" \
    --config.sample.test_batch_size=${TEST_BS} \
    --config.sample.num_steps=${NUM_STEPS} \
    --config.sample.guidance_scale=${GUIDANCE} \
    --config.reward_fn.camera_normal=${REWARD_CAMERA_NORMAL} \
    --config.reward_fn.uni3d=${REWARD_UNI3D} \
    --config.mixed_precision=bf16 \
    --config.deterministic=true
else
  # 单卡：直接 python 运行
  python scripts/train_direct3d_s2.py \
    --config config/direct3d_s2_grpo_normal-sim.py \
    --config.eval_only=true \
    --config.eval_data_dir="${EVAL_DIR}" \
    --config.camera_normal_eval.cache_dir="${EVAL_NORMAL_DIR}" \
    --config.camera_normal_eval.normal_resolution=${NORMAL_RES} \
    --config.pretrained.pipeline_path="${PRETRAIN_DIR}" \
    --config.pretrained.subfolder="${PRETRAIN_SUBFOLDER}" \
    --config.logdir="${LOG_DIR}" \
    --config.run_name="${RUN_NAME}" \
    --config.sample.test_batch_size=${TEST_BS} \
    --config.sample.num_steps=${NUM_STEPS} \
    --config.sample.guidance_scale=${GUIDANCE} \
    --config.reward_fn.camera_normal=${REWARD_CAMERA_NORMAL} \
    --config.reward_fn.uni3d=${REWARD_UNI3D} \
    --config.mixed_precision=bf16 \
    --config.deterministic=true
fi

echo "✅ Direct3D‑S2 eval-only 完成。可视化与 OBJ 输出在 ${LOG_DIR}/${RUN_NAME}/generated_meshes/eval_epoch_0/<image_stem>/ 下，camera_normal 在同目录 camera_{idx}.png。"


