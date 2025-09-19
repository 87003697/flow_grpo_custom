#!/usr/bin/env bash
# Direct3D-S2 自动化构建 + 复现性测试脚本
# 功能:
#  1. 激活指定 conda 环境 (默认: grpo3d)
#  2. 校验 torch / CUDA 可用性
#  3. 尝试导入 udf_ext, 若失败自动搜索并构建扩展
#  4. 运行 test_direct3d_s2_infer.py 的单步 + 端到端 (--do_e2e) 测试
#  5. 输出阶段总结并用退出码指示整体成功/失败
#
# 使用示例:
#   bash scripts/debug/build_and_test_direct3d_s2.sh \
#       --env grpo3d \
#       --pipeline pretrained_weights/direct3d_s2-v-1-1 \
#       --image dataset/eval3d_hunyuan3d/images/004.png \
#       --out outputs/test_runs/direct3d_s2_validation \
#       --candidates 1 --dense 50 --sparse 30 --guidance 7.0 --seed 777

# 重要: 需在仓库根目录下运行本脚本。
set -euo pipefail

# 默认参数
COND_ENV="grpo3d"
PIPELINE_PATH="pretrained_weights/direct3d_s2-v-1-1"
IMAGE_PATH="dataset/eval3d_hunyuan3d/images/004.png"
OUT_DIR="outputs/test_runs/direct3d_s2_validation"
NUM_CANDIDATES=1
DENSE_STEPS=50
SPARSE_STEPS=30
SEED=777
GUIDANCE=7.0
SIGMA_MIN=0.002
RESCALE_T=1000.0
DTYPE=fp16
EXTRA_ARGS=""
CLEAN_BUILD=0
RETRIES=2
DO_E2E=1
USE_SDE=0

log(){ echo -e "[BT][$(date +%H:%M:%S)] $*"; }
err(){ echo -e "[BT][ERR] $*" >&2; }

usage(){ sed -n '1,60p' "$0"; exit 0; }

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) COND_ENV=$2; shift 2;;
    --pipeline) PIPELINE_PATH=$2; shift 2;;
    --image) IMAGE_PATH=$2; shift 2;;
    --out) OUT_DIR=$2; shift 2;;
    --candidates) NUM_CANDIDATES=$2; shift 2;;
    --dense) DENSE_STEPS=$2; shift 2;;
    --sparse) SPARSE_STEPS=$2; shift 2;;
    --seed) SEED=$2; shift 2;;
    --guidance) GUIDANCE=$2; shift 2;;
    --sigma_min) SIGMA_MIN=$2; shift 2;;
    --rescale_t) RESCALE_T=$2; shift 2;;
    --dtype) DTYPE=$2; shift 2;;
    --extra) EXTRA_ARGS=$2; shift 2;;
    --use_sde) USE_SDE=1; shift 1;;
  --clean) CLEAN_BUILD=1; shift 1;;
  --retries) RETRIES=$2; shift 2;;
  --no_e2e) DO_E2E=0; shift 1;;
    -h|--help) usage;;
    *) err "未知参数: $1"; usage;;
  esac
done

ROOT_DIR=$(pwd)
SCRIPT_DIR="$ROOT_DIR/scripts/debug"
TEST_SCRIPT="$SCRIPT_DIR/test_direct3d_s2_infer_v2.py"

if [[ ! -f "$TEST_SCRIPT" ]]; then
  err "未找到测试脚本: $TEST_SCRIPT (请在仓库根目录运行)"; exit 2
fi

# 1. 通过 conda run 执行（避免在当前 shell 内 eval 激活导致不兼容）
if ! command -v conda >/dev/null 2>&1; then
  err "未找到 conda 命令，请确认已安装 Anaconda/Miniconda"
  exit 3
fi
RUNNER=(conda run -n "$COND_ENV")
log "Using conda run: ${RUNNER[*]}"

# 2. Torch / CUDA 检查
log "Checking torch environment"
"${RUNNER[@]}" python - <<'PY' || { echo "[BT][ERR] Torch 基础检查失败"; exit 4; }
import torch, sys
print("PYTHON:", sys.executable)
print("TORCH_VERSION:", torch.__version__)
print("CUDA_AVAILABLE:", torch.cuda.is_available())
print("CUDA_DEVICE_COUNT:", torch.cuda.device_count())
cuda_home = None
try:
  from torch.utils.cpp_extension import CUDA_HOME  # type: ignore
  cuda_home = CUDA_HOME
except Exception:
  try:
    import os
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
  except Exception:
    cuda_home = None
print("CUDA_HOME:", cuda_home)
PY

# 2.1 设定 TORCH_LIB_DIR / LD_LIBRARY_PATH 以避免 libc10.so 缺失
TORCH_LIB_DIR=$("${RUNNER[@]}" python - <<'PY'
import torch, pathlib
print(pathlib.Path(torch.__file__).parent/'lib')
PY
)
if [[ -d "$TORCH_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
  log "Set LD_LIBRARY_PATH prepend: $TORCH_LIB_DIR"
fi

# 所有 conda run 调用均带上 LD_LIBRARY_PATH，确保子进程能找到 libc10.so
# 同时追加 CUDA lib64（若存在）
if [[ -d "/usr/local/cuda/lib64" ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  log "Append CUDA lib64 to LD_LIBRARY_PATH: /usr/local/cuda/lib64"
fi
RUNNER_FULL=(env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "${RUNNER[@]}")

# 3. udf_ext 检测与自动构建
UDF_SRC_DIR="/home/zhiyuan_ma/code/Direct3D-S2/third_party/voxelize"
if "${RUNNER_FULL[@]}" python -c 'import torch; import udf_ext' 2>/dev/null; then
  log "udf_ext 已可用"
else
  log "udf_ext 不可用，尝试从 $UDF_SRC_DIR 自动构建"
  if [[ ! -d "$UDF_SRC_DIR" ]]; then
    err "未找到 udf_ext 源码目录: $UDF_SRC_DIR"
    exit 5
  fi
  pushd "$UDF_SRC_DIR" >/dev/null
  if [[ "$CLEAN_BUILD" == "1" ]]; then
    log "执行清理构建 (setup.py clean)"
    "${RUNNER_FULL[@]}" python setup.py clean --all || true
    rm -rf build *.egg-info 2>/dev/null || true
  fi
  ATTEMPT=0
  until [[ $ATTEMPT -ge $RETRIES ]]; do
    ATTEMPT=$((ATTEMPT+1))
    log "[udf_ext] pip install -v . (attempt $ATTEMPT/$RETRIES)"
    if "${RUNNER_FULL[@]}" python -m pip install -v .; then
      break
    fi
    sleep 1
  done
  popd >/dev/null
  if ! "${RUNNER_FULL[@]}" python -c 'import torch; import udf_ext' 2>/dev/null; then
    err "自动构建 udf_ext 失败，请按 README 手动构建后重试"
    exit 5
  fi
  log "udf_ext 构建成功并可用"
fi

# 报告 udf_ext 安装位置，辅助判断是否引用了非本地副本
"${RUNNER_FULL[@]}" python - <<'PY'
try:
    import torch, udf_ext, inspect, sys, os
    import importlib.util
    spec = importlib.util.find_spec('udf_ext')
    print('[BT] udf_ext origin:', spec.origin if spec else 'UNKNOWN')
    print('[BT] torch file:', torch.__file__)
    print('[BT] LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH',''))
except Exception as e:
    print('[BT][WARN] 无法定位 udf_ext origin:', e)
PY

# 5. 运行单步 + 端到端测试
log "Running Direct3D-S2 test script"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
CMD=("${RUNNER_FULL[@]}" python "$TEST_SCRIPT" \
  --pipeline_path "$PIPELINE_PATH" \
  --image "$IMAGE_PATH" \
  --out "$OUT_DIR" \
  --candidates "$NUM_CANDIDATES" \
  --dense_steps "$DENSE_STEPS" \
  --sparse_steps "$SPARSE_STEPS" \
  --guidance "$GUIDANCE" \
  --sigma_min "$SIGMA_MIN" \
  --rescale_t "$RESCALE_T" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  )

# 默认与官方一致：SDE 关闭（传 --no_sde），除非显式传入 --use_sde
if [[ "$USE_SDE" == "1" ]]; then
  :
else
  CMD+=(--no_sde)
fi

if [[ "$DO_E2E" == "1" ]]; then
  CMD+=(--do_e2e)
fi

# 512-only 校验：要求存在 512 权重
if [[ ! -f "$PIPELINE_PATH/model_sparse_512.ckpt" ]]; then
  err "缺少 512 权重: $PIPELINE_PATH/model_sparse_512.ckpt (请先运行下载脚本)"; exit 5
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=($EXTRA_ARGS)
  CMD+=("${EXTRA_SPLIT[@]}")
fi
log "Invoking: ${CMD[*]}"
set +e
"${CMD[@]}"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  err "测试脚本退出码: $RC"
  exit 6
fi

log "全部步骤完成 (udf_ext OK + 测试成功)"
exit 0
