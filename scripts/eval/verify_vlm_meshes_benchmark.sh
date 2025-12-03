#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "/home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source /home/zhiyuan_ma/anaconda3/etc/profile.d/conda.sh
elif [ -f "/home/zhiyuan_ma/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source /home/zhiyuan_ma/miniconda3/etc/profile.d/conda.sh
else
  echo "未找到 conda.sh，请确认 Anaconda/Miniconda 路径" >&2
  exit 1
fi
conda activate grpo3d

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DATASET_ROOT="${ROOT_DIR}/dataset/meshes_benchmark_v1"
CAMERA_CKPT="${ROOT_DIR}/pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/outputs/meshes_benchmark_v1/${RUN_TAG}"
VIS_DIR="${OUTPUT_DIR}/vis"
mkdir -p "${OUTPUT_DIR}" "${VIS_DIR}"

python "${ROOT_DIR}/scripts/eval/verify_vlm_meshes_benchmark.py" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-index "metadata/dataset_index.json" \
  --device "cuda" \
  --normal-resolution 512 \
  --cache-dir "${DATASET_ROOT}/normals" \
  --encoder "gemini-2.5-pro_group" \
  --camera-ckpt "${CAMERA_CKPT}" \
  --camera-config "_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py" \
  --camera-type "search" \
  --vlm-api-source "1" \
  --vlm-prompt-version "v2" \
  --vlm-enable-thinking \
  --vlm-debug-response \
  --source-front "+z" \
  --pipelines "direct3d_s2,hi3dgen,hunyuan3d,trellis,triposg" \
  --max-count 0 \
  --output-csv "${OUTPUT_DIR}/vlm_scores.csv" \
  --save-dir "${VIS_DIR}" \
  --save-cols 3 \
  --vlm-debug-response \
  "$@"

