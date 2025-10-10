## 安装与预训练权重准备

## 安装速览（最短路径）

目标环境（已验证）：Python 3.10，torch 2.5.1+cu124，flash‑attn 2.8.3（sm90），kaolin 0.18.0，spconv‑cu124 2.3.8，cumm‑cu124 0.7.11，nvdiffrast 0.3.3。

### 一次安装（复制即可）
```bash
# 进入项目
cd /home/zhiyuan_ma/code/flow_grpo_custom

# 创建并激活环境
conda create -n grpo3d python=3.10 -y && \
source /home/zhiyuan_ma/miniconda3/etc/profile.d/conda.sh && conda activate grpo3d

# PyTorch（cu124）
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 3D 依赖
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11

# flash-attn（Hopper 用 90）
export TORCH_CUDA_ARCH_LIST=90 && \
pip install --no-build-isolation flash-attn==2.8.3

# 其余依赖（若 torchsparse 装不上可忽略；nvdiffrast 无轮子则用源码）
pip install -r requirements.txt || true
pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
```

### 快速验证（Python）
```python
import torch, flash_attn, spconv, kaolin, nvdiffrast
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
```

### 备注
- 如 `torchsparse==2.1.0` 报错，可忽略；不影响核心训练链路。
- 需 xformers 时请另建环境并与 torch 2.6.0 搭配。


### Direct3D‑S2 预训练权重（512-only）

本节记录在本机完成的 Direct3D‑S2（仅 512 分支）权重下载步骤与校验信息，便于复现与核验。

#### 前置条件
- 已准备好 `grpo3d` 环境，并安装 `huggingface_hub`（随 `requirements.txt` 一同安装）。
- 无需激活环境，直接使用环境内的 Python 绝对路径执行。

#### 下载命令（已实测）
```bash
/home/zhiyuan_ma/miniconda3/envs/grpo3d/bin/python \
/home/zhiyuan_ma/code/flow_grpo_custom/scripts/download/download_direct3d_s2.py \
--out /home/zhiyuan_ma/code/flow_grpo_custom/pretrained_weights/direct3d_s2-v-1-1
```