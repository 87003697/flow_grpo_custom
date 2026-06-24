#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — flow_grpo_custom_v2 (Trellis Stage 2 Distillation)
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
#   s3://arcwm-code-us-west-2/$USER/data/
#     flow_grpo/                        ← 共享（与 flow_grpo_custom 复用）
#       qwen-image-edit-2511.tar        (33 GB)
#       alphaimages_v3.tar              (474 MB)
#       cuda_site_packages.tar          (150 MB)
#     flow_grpo_v2/                     ← 本项目独有
#       TRELLIS-image-large.tar         (3 GB)
#       trellis_reference.tar           (300 MB)
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
S3_BUCKET="s3://arcwm-code-us-west-2/${USER}"
PROJECT_DIR="/data/work/run_codes"

# S3 tar URI
S3_SHARED="${S3_BUCKET}/data/flow_grpo"
S3_V2="${S3_BUCKET}/data/flow_grpo_v2"
CUDA_SP_TAR="${S3_SHARED}/cuda_site_packages.tar"
VENV_TAR="${S3_V2}/uv-venv.tar"
REFERENCE_TAR="${S3_V2}/trellis_reference.tar"
TRELLIS_WEIGHTS_TAR="${S3_V2}/TRELLIS-image-large.tar"
DATASET_TAR="${S3_SHARED}/alphaimages_v3.tar"
QWEN_TAR="${S3_SHARED}/qwen-image-edit-2511.tar"

# 本地目标路径
WEIGHTS_LOCAL="/local-ssd/pretrained_weights"
DATASET_LOCAL="/local-ssd/alphaimages_v3"
REFERENCE_LOCAL="${PROJECT_DIR}/_reference_codes/TRELLIS"

cd "${PROJECT_DIR}"

# --- 环境变量 ---
export PATH="/tmp/uv-venv/bin:${PATH}"
export HF_HOME="/local-ssd/hf_cache"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="/local-ssd/torch_home"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_PROJECT_ENVIRONMENT=/tmp/uv-venv
export UV_FROZEN=1
export ATTN_BACKEND=flash_attn
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ============================================================================
# [1/6] Python 依赖 + CUDA 扩展（venv 整包恢复 or 逐步安装）
# ============================================================================
echo "=== [1/6] Python dependencies ==="

# 快速路径：从 S3 恢复完整 venv（含所有 pip 包 + CUDA 扩展 + kaolin + flash-attn）
if /tmp/uv-venv/bin/python -c "import torch; import flash_attn; import diff_gaussian_rasterization" 2>/dev/null; then
    echo "  venv already complete, skipping"
elif s5cmd ls "${VENV_TAR}" &>/dev/null; then
    echo "  Restoring pre-built venv from S3 (~8.9 GB)..."
    s5cmd cat "${VENV_TAR}" | tar xf - -C /tmp/
    echo "  venv restored"
else
    echo "  No venv tar found, installing from scratch..."

if [ ! -d /tmp/uv-venv ]; then
    uv venv --python 3.12 /tmp/uv-venv
fi

if ! /tmp/uv-venv/bin/python -c "import torch; assert torch.__version__.startswith('2.6')" 2>/dev/null; then
    echo "  Installing torch 2.6.0+cu124..."
    uv pip install --python /tmp/uv-venv/bin/python \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "  torch 2.6.0 already installed"
fi

echo "  Installing Python packages..."
uv pip install --python /tmp/uv-venv/bin/python \
    "transformers>=4.57.0" "accelerate==1.4.0" "peft>=0.10.0" \
    "diffusers==0.38.0" \
    ml-collections wandb kiui rembg onnxruntime open3d \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh kornia timm lpips pytorch-msssim einops deepspeed \
    pillow absl-py omegaconf loguru pydantic huggingface-hub zstandard \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# spconv
if ! /tmp/uv-venv/bin/python -c "import spconv" 2>/dev/null; then
    echo "  Installing spconv-cu124..."
    uv pip install --python /tmp/uv-venv/bin/python spconv-cu124==2.3.8
else
    echo "  spconv already installed"
fi

# kaolin (required by flexicubes)
if ! /tmp/uv-venv/bin/python -c "import kaolin" 2>/dev/null; then
    echo "  Installing kaolin..."
    uv pip install --python /tmp/uv-venv/bin/python \
        kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
else
    echo "  kaolin already installed"
fi

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
EXT_DIR="/local-ssd/extensions"

if /tmp/uv-venv/bin/python -c "import nvdiffrast; import diff_gaussian_rasterization; import kaolin" 2>/dev/null; then
    echo "  Already installed, skipping"
elif s5cmd ls "${CUDA_SP_TAR}" &>/dev/null; then
    echo "  Restoring pre-built packages from S3..."
    s5cmd cat "${CUDA_SP_TAR}" | tar xf - -C /tmp/uv-venv/lib/python3.12/site-packages/
    echo "  Restored"
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

    # mip-splatting diff-gaussian-rasterization (supports kernel_size)
    if [ ! -d "${EXT_DIR}/mip-splatting" ]; then
        git clone --recursive https://github.com/autonomousvision/mip-splatting.git "${EXT_DIR}/mip-splatting"
    fi
    MIP_RAST="${EXT_DIR}/mip-splatting/submodules/diff-gaussian-rasterization"
    # CUDA 12.x requires explicit cstdint include
    sed -i '1i #include <cstdint>' "${MIP_RAST}/cuda_rasterizer/rasterizer_impl.h" 2>/dev/null || true
    sed -i '1i #include <cstdint>' "${MIP_RAST}/cuda_rasterizer/rasterizer.h" 2>/dev/null || true
    uv pip install --python /tmp/uv-venv/bin/python "${MIP_RAST}" --no-build-isolation

    echo "  Saving tar to S3..."
    cd /tmp/uv-venv/lib/python3.12/site-packages/
    tar cf - nvdiffrast* cumesh* flex_gemm* nvdiffrec_render* diff_gaussian_rasterization* kaolin* | aws s3 cp - "${CUDA_SP_TAR}"
    cd "${PROJECT_DIR}"
    echo "  Compiled and cached"
else
    echo "  ERROR: No CUDA ext tar found and --download not specified."
    echo "  First-time setup: run with --download"
    return 1 2>/dev/null || exit 1
fi

fi  # end of "No venv tar found, installing from scratch"

# ============================================================================
# [3/6] _reference_codes/TRELLIS（git clone or tar 恢复）
# ============================================================================
echo "=== [3/6] TRELLIS reference code ==="
if [ -d "${REFERENCE_LOCAL}/trellis" ]; then
    echo "  Already present, skipping"
elif s5cmd ls "${REFERENCE_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    mkdir -p "${PROJECT_DIR}/_reference_codes"
    s5cmd cat "${REFERENCE_TAR}" | tar xf - -C "${PROJECT_DIR}/_reference_codes/"
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Cloning TRELLIS repo..."
    mkdir -p "${PROJECT_DIR}/_reference_codes"
    git clone --recursive https://github.com/87003697/TRELLIS.git "${REFERENCE_LOCAL}"
    echo "  Saving tar to S3..."
    tar cf - -C "${PROJECT_DIR}/_reference_codes" TRELLIS/ | aws s3 cp - "${REFERENCE_TAR}"
    echo "  Cloned and cached"
else
    echo "  ERROR: No tar at ${REFERENCE_TAR} and --download not specified."
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# [4/6] 预训练权重（TRELLIS-image-large → /local-ssd/ → 软链接）
# ============================================================================
echo "=== [4/6] Pretrained weights (TRELLIS-image-large) ==="
mkdir -p "${WEIGHTS_LOCAL}"

if [ -d "${WEIGHTS_LOCAL}/TRELLIS-image-large" ]; then
    echo "  Already present"
elif s5cmd ls "${TRELLIS_WEIGHTS_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    s5cmd cat "${TRELLIS_WEIGHTS_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading from HuggingFace..."
    /tmp/uv-venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/TRELLIS-image-large', local_dir='${WEIGHTS_LOCAL}/TRELLIS-image-large')
print('TRELLIS-image-large done')
"
    echo "  Saving tar to S3..."
    tar cf - -C "${WEIGHTS_LOCAL}" TRELLIS-image-large/ | aws s3 cp - "${TRELLIS_WEIGHTS_TAR}"
    echo "  Downloaded and cached"
else
    echo "  ERROR: No weights tar and --download not specified."
    return 1 2>/dev/null || exit 1
fi

ln -sfn "${WEIGHTS_LOCAL}" "${PROJECT_DIR}/pretrained_weights"

# ============================================================================
# [5/6] 数据集（S3 → /local-ssd/ → 软链接）
# ============================================================================
echo "=== [5/6] Dataset (alphaimages_v3) ==="
if [ -d "${DATASET_LOCAL}/train" ]; then
    echo "  Already present on local-ssd"
elif s5cmd ls "${DATASET_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    s5cmd cat "${DATASET_TAR}" | tar xf - -C /local-ssd/
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading from HuggingFace..."
    /tmp/uv-venv/bin/python scripts/download/download_alphaimages_v3.py \
        --out "${DATASET_LOCAL}" --token "${HF_TOKEN:-}"
    echo "  Saving tar to S3..."
    tar cf - -C /local-ssd alphaimages_v3/ | aws s3 cp - "${DATASET_TAR}"
    echo "  Downloaded and cached"
else
    echo "  ERROR: No tar at ${DATASET_TAR} and --download not specified."
    return 1 2>/dev/null || exit 1
fi
mkdir -p "${PROJECT_DIR}/dataset"
ln -sfn "${DATASET_LOCAL}" "${PROJECT_DIR}/dataset/alphaimages_v3"

# ============================================================================
# [6/6] Qwen Guidance 模型（HF cache）
# ============================================================================
echo "=== [6/6] Qwen-Image-Edit-2511 (HF cache) ==="
if [ -d "${HF_HOME}/hub/models--Qwen--Qwen-Image-Edit-2511" ]; then
    echo "  Already in HF cache"
elif s5cmd ls "${QWEN_TAR}" &>/dev/null; then
    echo "  Restoring from tar..."
    s5cmd cat "${QWEN_TAR}" | tar xf - -C /local-ssd/
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading (~33 GB, may take a while)..."
    mkdir -p "${HF_HOME}"
    /tmp/uv-venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-Edit-2511', cache_dir='${HF_HOME}', resume_download=True)
print('Qwen-Image-Edit-2511 done')
"
    echo "  Saving tar to S3..."
    tar cf - -C /local-ssd hf_cache/ | aws s3 cp - "${QWEN_TAR}"
    echo "  Downloaded and cached"
else
    echo "  No tar found, will download on first training run"
    mkdir -p "${HF_HOME}"
fi

# ============================================================================
# [6.5/6] DINOv2 torch.hub cache（避免多进程下载竞争）
# ============================================================================
echo "=== [6.5/6] DINOv2 (torch.hub cache) ==="
DINOV2_TAR="${S3_V2}/torch_home_dinov2.tar"
DINOV2_HUB_DIR="${TORCH_HOME}/hub/facebookresearch_dinov2_main"

if [ -d "${DINOV2_HUB_DIR}" ]; then
    echo "  Already in torch.hub cache"
elif s5cmd ls "${DINOV2_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    mkdir -p "${TORCH_HOME}"
    s5cmd cat "${DINOV2_TAR}" | tar xf - -C /local-ssd/
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Pre-downloading DINOv2 via torch.hub..."
    mkdir -p "${TORCH_HOME}"
    /tmp/uv-venv/bin/python -c "
import torch
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
print('DINOv2 hub cache warmed')
"
    echo "  Saving tar to S3..."
    tar cf - -C /local-ssd torch_home/ | aws s3 cp - "${DINOV2_TAR}"
    echo "  Downloaded and cached"
else
    echo "  WARNING: No DINOv2 tar found. Multi-GPU training may fail due to download race."
    echo "  Run with --download to pre-cache, or ensure single-process downloads first."
fi

# ============================================================================
# [6.6/6] DINOv3-S (Discriminator for GAN loss, ~82 MB)
# ============================================================================
echo "=== [6.6/6] DINOv3-S (Discriminator) ==="
DINOV3S_TAR="${S3_V2}/dinov3-vits16.tar"
DINOV3S_DIR="${WEIGHTS_LOCAL}/dinov3-vits16-pretrain-lvd1689m"

if [ -d "${DINOV3S_DIR}" ]; then
    echo "  Already present"
elif s5cmd ls "${DINOV3S_TAR}" &>/dev/null; then
    echo "  Restoring from S3..."
    s5cmd cat "${DINOV3S_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Restored"
else
    echo "  WARNING: No DINOv3-S tar found at ${DINOV3S_TAR}"
    echo "  GAN training will fail. Upload weights first."
fi

# ============================================================================
# [6.7/6] RMBG-2.0 (SilhouetteExtractor for eval, ~200 MB)
# ============================================================================
echo "=== [6.7/6] RMBG-2.0 (Silhouette eval) ==="
RMBG_TAR="${S3_V2}/rmbg2.tar"
RMBG_DIR="${WEIGHTS_LOCAL}/rmbg2/RMBG-2.0"

if [ -d "${RMBG_DIR}" ]; then
    echo "  Already present"
elif s5cmd ls "${RMBG_TAR}" &>/dev/null; then
    echo "  Restoring from S3..."
    s5cmd cat "${RMBG_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Restored"
elif [ "$DOWNLOAD_MODE" = true ]; then
    echo "  Downloading RMBG-2.0 from HuggingFace..."
    /tmp/uv-venv/bin/python scripts/download/download_rmbg2.py
    echo "  Saving tar to S3..."
    tar cf - -C "${WEIGHTS_LOCAL}" rmbg2/ | aws s3 cp - "${RMBG_TAR}"
    echo "  Downloaded and cached"
else
    echo "  WARNING: No RMBG-2.0 tar found at ${RMBG_TAR}. SilhouetteIoU eval will fail."
fi

# ============================================================================
# [6.8/6] CLIP (eval metric, ~1.7 GB)
# ============================================================================
echo "=== [6.8/6] CLIP-ViT-L/14 (eval metric) ==="
CLIP_TAR="${S3_V2}/clip-vit-large-patch14.tar"
CLIP_DIR="${WEIGHTS_LOCAL}/clip/clip-vit-large-patch14"

if [ -d "${CLIP_DIR}" ]; then
    echo "  Already present"
elif s5cmd ls "${CLIP_TAR}" &>/dev/null; then
    echo "  Restoring from S3..."
    s5cmd cat "${CLIP_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Restored"
else
    echo "  WARNING: No CLIP tar found at ${CLIP_TAR}. CLIPMetric eval will fail."
fi

# ============================================================================
# [6.9/6] DINOv3-L (eval metric, ~1.2 GB)
# ============================================================================
echo "=== [6.9/6] DINOv3-ViT-L/16 (eval metric) ==="
DINOV3L_TAR="${S3_SHARED}/dinov3-vitl16.tar"
DINOV3L_DIR="${WEIGHTS_LOCAL}/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"

if [ -d "${DINOV3L_DIR}" ] && [ -f "${DINOV3L_DIR}/model.safetensors" ]; then
    echo "  Already present"
elif s5cmd ls "${DINOV3L_TAR}" &>/dev/null; then
    echo "  Restoring from S3..."
    s5cmd cat "${DINOV3L_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Restored"
else
    echo "  WARNING: No DINOv3-L tar found at ${DINOV3L_TAR}. DINOMetric eval will fail."
fi

# ============================================================================
# 后台 S3 同步（训练产出持久化）
# ============================================================================
echo "=== Background S3 sync ==="
LOGS_LOCAL="${PROJECT_DIR}/logs"
LOGS_S3="${S3_BUCKET}/experiments/flow_grpo_v2"

sync_all() {
    if [ -d "${LOGS_LOCAL}" ]; then
        aws s3 sync "${LOGS_LOCAL}/" "${LOGS_S3}/" \
            --quiet >> /tmp/s3_sync.log 2>&1 || true
    fi
}

(while true; do sleep 300; sync_all; done) &
SYNC_PID=$!
trap "kill ${SYNC_PID} 2>/dev/null || true; sync_all" EXIT

echo "  PID: ${SYNC_PID} (every 5 min, incremental)"
echo "  ${LOGS_LOCAL}/ -> ${LOGS_S3}/"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================="
echo "  flow_grpo_custom_v2 — Setup Complete"
echo "========================================="
echo "Python:   $(/tmp/uv-venv/bin/python --version 2>&1)"
echo "Torch:    $(/tmp/uv-venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')"
echo "GPUs:     $(/tmp/uv-venv/bin/python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "Weights:  ${PROJECT_DIR}/pretrained_weights -> ${WEIGHTS_LOCAL}"
echo "Dataset:  ${PROJECT_DIR}/dataset/alphaimages_v3 -> ${DATASET_LOCAL}"
echo ""
echo "Quick start:"
echo "  bash scripts/multi_node/main_trellis_distilation.sh"
echo ""
echo "First time? Run:  . scripts/setup_koala.sh --download"
echo "Daily restore:    . scripts/setup_koala.sh --fast"
echo "========================================="
