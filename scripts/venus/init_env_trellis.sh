#!/bin/bash
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# 创建grpo3d环境
conda create -n grpo3d_trellis python=3.10 -y && conda activate grpo3d_trellis

# 安装torch
python -m pip install torch==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 3D 依赖
python -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
python -m pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11


# 安装其他依赖
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3


# flash-attn（Hopper 用 90）
TORCH_CUDA_ARCH_LIST=90 python -m pip install --no-build-isolation flash-attn==2.8.3
