# Hunyuan3D Flow-GRPO

基于 Flow-GRPO 框架的 Hunyuan3D 训练代码。

## 环境配置

### 1. 一键安装（推荐）
```bash
conda create -n grpo3d python=3.10 -y && conda activate grpo3d
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 可选：加速注意力（按需）
# export TORCH_CUDA_ARCH_LIST="80;86;89;90"
# pip install --no-build-isolation flash-attn==2.8.3
```

### 2. 下载预训练模型（精简）
```bash
# 可选：登录 Hugging Face（如需从 Hub 下载私有/受限资源）
# huggingface-cli login

# Hunyuan3D 权重
python scripts/download/download_hunyuan3d_weights.py

# EVA Giant（评分模型）
python scripts/download/download_eva_weights.py

# Direct3D‑S2 Stage1（可选，用于最小集成/测试）
python scripts/download/download_direct3d_s2.py --out pretrained_weights/direct3d_s2
```

权重默认存放：
- Hunyuan3D：`pretrained_weights/tencent/Hunyuan3D-2.1/`
  - DiT：`hunyuan3d-dit-v2-1/`
  - VAE：`hunyuan3d-vae-v2-1/`
- EVA Giant：`pretrained_weights/eva/`
- Direct3D‑S2（可选）：`pretrained_weights/direct3d_s2/`
### 注意力后端兼容性说明
| 组合 | 状态 | 备注 |
|------|------|------|
| torch 2.5.1+cu124 + flash-attn 2.8.3 | ✅ 稳定 | 已验证前向 (fp16) 正常 |
| torch 2.6.0 + flash-attn 2.8.x | ❌ 失败 | C++ undefined symbol (c10::Error) |
| torch 2.6.0 + xformers 0.0.29.post3 | ✅ 可用 | 若需此组合请单独新环境 |

@misc{liu2025flowgrpo,
      title={Flow-GRPO: Training Flow Matching Models via Online RL}, 
      author={Jie Liu and Gongye Liu and Jiajun Liang and Yangguang Li and Jiaheng Liu and Xintao Wang and Pengfei Wan and Di Zhang and Wanli Ouyang},
      year={2025},
      eprint={2505.05470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05470}, 
}
```