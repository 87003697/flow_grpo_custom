#!/bin/bash
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# 创建grpo3d环境
conda create -n grpo3d_direct3d python=3.10 -y && conda activate grpo3d_direct3d

# 安装torch
python -m pip install torch==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 3D 依赖
python -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
python -m pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11


# 安装其他依赖
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3

### 安装 Direct3D‑S2 参考代码与 CUDA 扩展（udf_ext）
# 编译并安装 Direct3D‑S2 的 CUDA 扩展（udf_ext）
python -m pip install -v --no-build-isolation \
  ./_reference_codes/Direct3D-S2/third_party/voxelize

# 以可编辑模式安装 Direct3D‑S2 包（供训练脚本导入）
python -m pip install -v -e \
  ./_reference_codes/Direct3D-S2

### 可选：安装 torchsparse（源码编译，无需 root）
# 1) 安装 sparsehash 头文件（conda，无需 root）
conda install -y -c bioconda google-sparsehash


# 2) 确保编译器能找到头文件（当前会话）
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH"
export CPATH="$CONDA_PREFIX/include:$CPATH"

# 3) 可选：安装 ninja 提升 C++/CUDA 编译速度
python -m pip install ninja

# 4) 源码安装 torchsparse（与 torch 2.5.1+cu124 组合已实测）
python -m pip install -v git+https://github.com/mit-han-lab/torchsparse.git

# flash-attn（Hopper 用 90）
TORCH_CUDA_ARCH_LIST=90 python -m pip install --no-build-isolation flash-attn==2.8.3
