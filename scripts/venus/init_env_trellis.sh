#!/bin/bash
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# 创建grpo3d环境（与 TRELLIS 兼容的 torch/kaolin 版本）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda remove -n grpo3d_trellis --all -y || true
conda create -n grpo3d_trellis python=3.10 -y
conda activate grpo3d_trellis

# 安装 torch 2.4.0 + cu118（官方 TRELLIS 训练脚本默认）
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# 3D 依赖（匹配 cu118）
python -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html
python -m pip install spconv-cu118==2.3.8 cumm-cu118==0.7.11

# 安装其他依赖
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3

# flash-attn（与 torch2.4/cu118 兼容）
python -m pip install --no-build-isolation flash-attn==2.7.3

# Gaussian Splatting 渲染依赖（mip-splatting 的 diff-gaussian-rasterization）
mkdir -p /tmp/extensions
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting 2>/dev/null || true
python -m pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
