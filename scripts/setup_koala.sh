#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — flow_grpo_custom (Trellis2)
# ============================================================================
# 用法：
#   . scripts/setup_koala.sh [--fast] [--download]
#
# --fast      跳过首次下载/编译，假设 S3 已有所有 tar 缓存（日常恢复模式）
# --download  首次使用：从 HuggingFace 下载权重/数据，编译 CUDA 扩展，打包到 S3
#
# 注意：通过 source 执行（. scripts/setup_koala.sh），以保留 export 的环境变量。
#
# 环境变量（koala 自动注入或 zshrc 中配置）：
#   HF_TOKEN       HuggingFace 认证（首次下载必需）
#   WANDB_API_KEY  WandB 认证（训练上报可选）
#
# S3 布局（首次 --download 完成后自动创建）：
#   /threed-code/$USER/data/flow_grpo/
#     TRELLIS.2-4B.tar           (16 GB)  Shape+Tex Flow Model
#     TRELLIS-image-large.tar    (3 GB)   v1 模型（可选）
#     dinov3-vitl16.tar          (1.2 GB) DINOv3 图像编码器
#     qwen-image-edit-2511.tar   (33 GB)  FlowEdit Guidance 模型
#     alphaimages_v3.tar         (474 MB) 训练数据集
#     trellis2_reference.tar     (361 MB) _reference_codes/TRELLIS.2
#     cuda_site_packages.tar     (145 MB) 预编译 CUDA 扩展
#     flow_grpo_cuda_ext.tar     (640 MB) CUDA 扩展源码（fallback）
# ============================================================================
set -euo pipefail

# --- 参数解析 ---
FAST_MODE=false
DOWNLOAD_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)     FAST_MODE=true; shift ;;
        --download) DOWNLOAD_MODE=true; shift ;;
        *)          echo "Unknown option: $1"; return 1 2>/dev/null || exit 1 ;;
    esac
done

# --- 路径配置 ---
USER="${KOALA_USER:-ericzyma}"
S3_PREFIX="/threed-code/${USER}"
S3_BUCKET="s3://arcwm-code-us-west-2/${USER}"
PROJECT_DIR="/data/work/run_codes"

# S3 tar 路径（每个模型独立打包）
S3_DATA="${S3_PREFIX}/data/flow_grpo"
TRELLIS2_TAR="${S3_DATA}/TRELLIS.2-4B.tar"
TRELLIS1_TAR="${S3_DATA}/TRELLIS-image-large.tar"
DINOV3_TAR="${S3_DATA}/dinov3-vitl16.tar"
QWEN_TAR="${S3_DATA}/qwen-image-edit-2511.tar"
DATASET_TAR="${S3_DATA}/alphaimages_v3.tar"
REFERENCE_TAR="${S3_DATA}/trellis2_reference.tar"
CUDA_EXT_TAR="${S3_DATA}/flow_grpo_cuda_ext.tar"

# 本地目标路径
WEIGHTS_LOCAL="/local-ssd/pretrained_weights"
DATASET_LOCAL="/local-ssd/alphaimages_v3"
REFERENCE_LOCAL="/local-ssd/TRELLIS.2"
EXT_DIR="/local-ssd/extensions"

cd "${PROJECT_DIR}"

# --- 环境变量 ---
export PATH="/tmp/uv-venv/bin:${PATH}"
export HF_HOME="/local-ssd/hf_cache"
export HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN not set. Export it before running setup.}"
export HF_HUB_DISABLE_XET=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/tmp/uv-venv
export ATTN_BACKEND=flash_attn
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# --- 确保 S3 目标目录存在 ---
mkdir -p "${S3_PREFIX}/data/flow_grpo"
mkdir -p "${S3_PREFIX}/tools"

# ============================================================================
# [1/6] Python 依赖
# ============================================================================
echo "=== [1/6] Python dependencies ==="
if [ ! -d /tmp/uv-venv ]; then
    uv venv --python 3.12 /tmp/uv-venv
fi

# torch 2.6.0 + cu124
if ! /tmp/uv-venv/bin/python -c "import torch; assert torch.__version__.startswith('2.6')" 2>/dev/null; then
    echo "  Installing torch 2.6.0+cu124..."
    uv pip install --python /tmp/uv-venv/bin/python \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "  torch 2.6.0 already installed"
fi

# 基础依赖
echo "  Installing Python packages..."
uv pip install --python /tmp/uv-venv/bin/python \
    "transformers==4.57.3" accelerate "peft>=0.17.0" \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@main" \
    ml-collections wandb kiui rembg onnxruntime open3d \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh kornia timm lpips pytorch-msssim einops deepspeed \
    pillow absl-py omegaconf loguru pydantic huggingface-hub zstandard \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# flash-attn
if ! /tmp/uv-venv/bin/python -c "import flash_attn" 2>/dev/null; then
    echo "  Compiling flash-attn 2.7.3 (~3 min)..."
    uv pip install --python /tmp/uv-venv/bin/python wheel setuptools
    uv pip install --python /tmp/uv-venv/bin/python \
        --no-build-isolation flash-attn==2.7.3
else
    echo "  flash-attn already installed"
fi
echo "  Done"

# ============================================================================
# [2/6] CUDA 扩展（预编译 site-packages 恢复 or 首次编译）
# ============================================================================
echo "=== [2/6] CUDA extensions ==="
CUDA_SP_TAR="${S3_DATA}/cuda_site_packages.tar"

if /tmp/uv-venv/bin/python -c "import nvdiffrast, cumesh, flex_gemm" 2>/dev/null; then
    echo "  Already installed, skipping"
elif [ -f "${CUDA_SP_TAR}" ]; then
    # 直接恢复预编译的 .so 到 site-packages（无需 rebuild，~3s）
    echo "  Restoring pre-built packages from S3..."
    cat "${CUDA_SP_TAR}" | tar xf - -C /tmp/uv-venv/lib/python3.12/site-packages/
    echo "  Restored (instant)"
elif [ -f "${CUDA_EXT_TAR}" ]; then
    # 旧方案：恢复源码后 rebuild（~6 min，fallback）
    echo "  Restoring source + rebuilding from S3 tar..."
    cat "${CUDA_EXT_TAR}" | tar xf - -C /local-ssd/
    uv pip install --python /tmp/uv-venv/bin/python --no-build-isolation --no-deps \
        "${EXT_DIR}/nvdiffrast" \
        "${EXT_DIR}/nvdiffrec" \
        "${EXT_DIR}/CuMesh" \
        "${EXT_DIR}/FlexGEMM"
    echo "  Restored ($(du -sh "${EXT_DIR}" | cut -f1))"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Compiling from source (~5-10 min)..."
    mkdir -p "${EXT_DIR}"
    [ -d "${EXT_DIR}/nvdiffrast" ] || git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "${EXT_DIR}/nvdiffrast"
    uv pip install --python /tmp/uv-venv/bin/python "${EXT_DIR}/nvdiffrast" --no-build-isolation
    [ -d "${EXT_DIR}/nvdiffrec" ] || git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "${EXT_DIR}/nvdiffrec"
    uv pip install --python /tmp/uv-venv/bin/python "${EXT_DIR}/nvdiffrec" --no-build-isolation
    [ -d "${EXT_DIR}/CuMesh" ] || git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "${EXT_DIR}/CuMesh"
    uv pip install --python /tmp/uv-venv/bin/python "${EXT_DIR}/CuMesh" --no-build-isolation
    [ -d "${EXT_DIR}/FlexGEMM" ] || git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "${EXT_DIR}/FlexGEMM"
    uv pip install --python /tmp/uv-venv/bin/python "${EXT_DIR}/FlexGEMM" --no-build-isolation
    # 缓存到 S3
    echo "  Saving tar to S3..."
    tar cf "${CUDA_EXT_TAR}" -C /local-ssd extensions/
    echo "  Compiled and cached"
else
    echo "  ERROR: No CUDA ext tar found and --download not specified."
    echo "  First-time setup: run with --download"
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# [3/6] _reference_codes/TRELLIS.2（git clone or tar 恢复）
# ============================================================================
echo "=== [3/6] TRELLIS.2 reference code ==="
if [ -d "${PROJECT_DIR}/_reference_codes/TRELLIS.2/trellis2" ]; then
    echo "  Already present, skipping"
elif [ -f "${REFERENCE_TAR}" ]; then
    echo "  Restoring from S3 tar..."
    mkdir -p "${PROJECT_DIR}/_reference_codes"
    cat "${REFERENCE_TAR}" | tar xf - -C "${PROJECT_DIR}/_reference_codes/"
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Cloning TRELLIS.2 repo..."
    mkdir -p "${PROJECT_DIR}/_reference_codes"
    git clone --recursive https://github.com/87003697/TRELLIS.2.git "${PROJECT_DIR}/_reference_codes/TRELLIS.2"
    # 缓存到 S3
    echo "  Saving tar to S3..."
    tar cf "${REFERENCE_TAR}" -C "${PROJECT_DIR}/_reference_codes" TRELLIS.2/
    echo "  Cloned and cached"
else
    echo "  ERROR: No tar at ${REFERENCE_TAR} and --download not specified."
    return 1 2>/dev/null || exit 1
fi

# o-voxel 安装
OVOXEL_DIR="${PROJECT_DIR}/_reference_codes/TRELLIS.2/o-voxel"
if /tmp/uv-venv/bin/python -c "import o_voxel" 2>/dev/null; then
    echo "  o-voxel already installed"
elif [ -d "${OVOXEL_DIR}" ]; then
    echo "  Installing o-voxel..."
    uv pip install --python /tmp/uv-venv/bin/python \
        "${OVOXEL_DIR}" --no-build-isolation --no-deps
    echo "  Done"
fi

# ============================================================================
# [4/6] 预训练权重（每个模型单独恢复 → /local-ssd/pretrained_weights/）
# ============================================================================
echo "=== [4/6] Pretrained weights ==="
mkdir -p "${WEIGHTS_LOCAL}"

# TRELLIS.2-4B (Shape + Tex Flow Model, ~16 GB)
if [ -d "${WEIGHTS_LOCAL}/TRELLIS.2-4B" ]; then
    echo "  TRELLIS.2-4B: already present"
elif [ -f "${TRELLIS2_TAR}" ]; then
    echo "  TRELLIS.2-4B: restoring..."
    cat "${TRELLIS2_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  TRELLIS.2-4B: done"
elif [ "$DOWNLOAD_MODE" = true ]; then
    /tmp/uv-venv/bin/python scripts/download/download_trellis2.py \
        --dest "${WEIGHTS_LOCAL}/TRELLIS.2-4B"
    tar cf "${TRELLIS2_TAR}" -C "${WEIGHTS_LOCAL}" TRELLIS.2-4B/
fi

# DINOv3-ViT-L/16 (Image Encoder, ~1.2 GB)
if [ -d "${WEIGHTS_LOCAL}/dinov3-vitl16-pretrain-lvd1689m" ]; then
    echo "  DINOv3: already present"
elif [ -f "${DINOV3_TAR}" ]; then
    echo "  DINOv3: restoring..."
    cat "${DINOV3_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  DINOv3: done"
elif [ "$DOWNLOAD_MODE" = true ]; then
    uv pip install --python /tmp/uv-venv/bin/python modelscope 2>/dev/null
    /tmp/uv-venv/bin/python scripts/download/download_dinov3_trellis2_modelscope.py \
        --dest "${WEIGHTS_LOCAL}/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    tar cf "${DINOV3_TAR}" -C "${WEIGHTS_LOCAL}" dinov3-vitl16-pretrain-lvd1689m/
fi

# TRELLIS-image-large (v1 model, ~3 GB, optional for Trellis1 configs)
if [ -d "${WEIGHTS_LOCAL}/TRELLIS-image-large" ]; then
    echo "  TRELLIS-image-large: already present"
elif [ -f "${TRELLIS1_TAR}" ]; then
    echo "  TRELLIS-image-large: restoring..."
    cat "${TRELLIS1_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  TRELLIS-image-large: done"
fi

# 软链接回项目目录
ln -sfn "${WEIGHTS_LOCAL}" "${PROJECT_DIR}/pretrained_weights"

# ============================================================================
# [5/6] 数据集（S3 → /local-ssd/ → 软链接）
# ============================================================================
echo "=== [5/6] Dataset (alphaimages_v3) ==="
if [ -d "${DATASET_LOCAL}/train" ]; then
    echo "  Already present on local-ssd"
elif [ -f "${DATASET_TAR}" ]; then
    echo "  Restoring from S3 tar..."
    cat "${DATASET_TAR}" | tar xf - -C /local-ssd/
    echo "  Restored ($(du -sh "${DATASET_LOCAL}" | cut -f1))"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading from HuggingFace..."
    /tmp/uv-venv/bin/python scripts/download/download_alphaimages_v3.py \
        --out "${DATASET_LOCAL}" --token "${HF_TOKEN:-}"
    # 缓存到 S3
    echo "  Saving tar to S3..."
    tar cf "${DATASET_TAR}" -C /local-ssd alphaimages_v3/
    echo "  Downloaded and cached"
else
    echo "  ERROR: No tar at ${DATASET_TAR} and --download not specified."
    return 1 2>/dev/null || exit 1
fi
# 软链接回项目目录
mkdir -p "${PROJECT_DIR}/dataset"
ln -sfn "${DATASET_LOCAL}" "${PROJECT_DIR}/dataset/alphaimages_v3"

# ============================================================================
# [6/6] Qwen Guidance 模型（HF cache）
# ============================================================================
echo "=== [6/6] Qwen-Image-Edit-2511 (HF cache) ==="
if [ -d "${HF_HOME}/hub" ] && find "${HF_HOME}/hub" -maxdepth 1 -name "*Qwen*" 2>/dev/null | grep -q .; then
    echo "  Already in HF cache"
elif [ -f "${QWEN_TAR}" ]; then
    echo "  Restoring from tar..."
    cat "${QWEN_TAR}" | tar xf - -C /local-ssd/
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading..."
    mkdir -p "${HF_HOME}"
    /tmp/uv-venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-Edit-2511', cache_dir='${HF_HOME}', resume_download=True)
print('Qwen-Image-Edit-2511 done')
"
    # 缓存到 S3
    echo "  Saving tar to S3..."
    tar cf "${QWEN_TAR}" -C /local-ssd hf_cache/
    echo "  Downloaded and cached"
else
    echo "  No tar found, will download on first training run"
    mkdir -p "${HF_HOME}"
fi

# ============================================================================
# [7/7] 后台 S3 同步（训练产出持久化）
# ============================================================================
echo "=== [7/7] Background S3 sync ==="
LOGS_LOCAL="${PROJECT_DIR}/logs"
LOGS_S3="${S3_BUCKET}/experiments/flow_grpo"

sync_logs() {
    if [ -d "${LOGS_LOCAL}" ]; then
        aws s3 sync "${LOGS_LOCAL}/" "${LOGS_S3}/" \
            --exclude '*.bin' \
            --quiet >> /tmp/s3_sync.log 2>&1 || true
    fi
}

(while true; do sleep 300; sync_logs; done) &
SYNC_PID=$!
trap "kill ${SYNC_PID} 2>/dev/null || true; sync_logs" EXIT

echo "  PID: ${SYNC_PID} (every 5 min)"
echo "  ${LOGS_LOCAL}/ -> ${LOGS_S3}/"
echo "  Mac 查看: rclone rc vfs/refresh dir=\"ericzyma/experiments/flow_grpo\" recursive=true"
echo "            ls ~/threed-code/ericzyma/experiments/flow_grpo/"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================="
echo "  flow_grpo_custom — Setup Complete"
echo "========================================="
echo "Python:   $(/tmp/uv-venv/bin/python --version 2>&1)"
echo "Torch:    $(/tmp/uv-venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')"
echo "GPUs:     $(/tmp/uv-venv/bin/python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "Weights:  ${PROJECT_DIR}/pretrained_weights -> ${WEIGHTS_LOCAL}"
echo "Dataset:  ${PROJECT_DIR}/dataset/alphaimages_v3 -> ${DATASET_LOCAL}"
echo ""
echo "Quick start:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/multi_node/main_trellis2_shape_distilation_async.sh"
echo ""
echo "First time? Run:  . scripts/setup_koala.sh --download"
echo "Daily restore:    . scripts/setup_koala.sh --fast"
echo "========================================="
