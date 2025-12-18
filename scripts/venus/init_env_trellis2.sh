#!/bin/bash
# 初始化/更新 TRELLIS.2 相关依赖到 grpo3d 环境（无 sudo）
set -euo pipefail

# 激活 conda（优先 anaconda3，其次 miniconda3）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "未找到 conda 初始化脚本" >&2
  exit 1
fi

conda activate grpo3d

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT_DIR="${EXT_DIR:-/tmp/extensions}"
mkdir -p "$EXT_DIR"

# 1) 升级核心框架到 TORCH 2.6.0 + CUDA 12.4
python -m pip install --upgrade \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 2) flash-attn 与 torch 2.6.0 对齐
python -m pip install --upgrade --no-build-isolation flash-attn==2.7.3

# 3) nvdiffrast v0.4.0
rm -rf "$EXT_DIR/nvdiffrast"
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXT_DIR/nvdiffrast"
python -m pip install "$EXT_DIR/nvdiffrast" --no-build-isolation

# 4) nvdiffrec（renderutils 分支）
rm -rf "$EXT_DIR/nvdiffrec"
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXT_DIR/nvdiffrec"
python -m pip install "$EXT_DIR/nvdiffrec" --no-build-isolation

# 5) FlexGEMM
rm -rf "$EXT_DIR/FlexGEMM"
git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXT_DIR/FlexGEMM"
python -m pip install "$EXT_DIR/FlexGEMM" --no-build-isolation

# 6) CuMesh
rm -rf "$EXT_DIR/CuMesh"
git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXT_DIR/CuMesh"
python -m pip install "$EXT_DIR/CuMesh" --no-build-isolation

# 7) 本地安装 o-voxel（使用已装 CuMesh/FlexGEMM，显式 GPU 架构列表）
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
python -m pip install "$REPO_ROOT/_reference_codes/TRELLIS.2/o-voxel" \
  --no-build-isolation --no-deps

echo "TRELLIS.2 依赖安装完成（grpo3d 环境）"
