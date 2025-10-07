#!/bin/bash
# Wrapper: 激活 triplaneturbo 环境并运行 Direct3D-S2 推理自检脚本
# 用法示例：
  # bash scripts/debug/run_direct3d_s2_infer.sh \
  #   --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
  #   --image dataset/eval3d_hunyuan3d/images/004.png \
  #   --out outputs/test_runs/direct3d_s2_validation \
  #   --candidates 1 --dense_steps 30 --sparse_steps 20 --guidance 0 --do_e2e --use_sde

set -euo pipefail

# 默认使用 grpo3d 环境（用户可通过 ENV_NAME 覆盖）
ENV_NAME=${ENV_NAME:-grpo3d}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda 未找到，请先安装 Anaconda/Miniconda" >&2
  exit 1
fi

# 激活环境
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || { echo "[ERROR] 激活环境 ${ENV_NAME} 失败"; exit 1; }

echo "[INFO] 使用环境: ${CONDA_DEFAULT_ENV}";
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 可选：自动补充 LD_LIBRARY_PATH 指向 PyTorch lib
if python -c "import torch,os;import sys;print(os.path.join(os.path.dirname(torch.__file__),'lib'))" >/dev/null 2>&1; then
  TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

echo "[INFO] 运行 test_direct3d_s2_infer.py $*"
python "${REPO_ROOT}/scripts/debug/test_direct3d_s2_infer_v2.py" "$@"
