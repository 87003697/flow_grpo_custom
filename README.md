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

### 2. 下载预训练模型

#### 2.1 3D Generators

##### Hunyuan3D
```bash
# 可选登录（如需访问私有/受限资源）
# huggingface-cli login

python scripts/download/download_hunyuan3d_weights.py
```
- 默认路径：`pretrained_weights/tencent/Hunyuan3D-2.1/`
  - DiT：`hunyuan3d-dit-v2-1/`
  - VAE：`hunyuan3d-vae-v2-1/`

##### Trellis
- 当前实现无需单独下载权重（随代码使用内置/引用模型）。

##### Direct3D‑S2（Stage1，可选）
```bash
python scripts/download/download_direct3d_s2.py --out pretrained_weights/direct3d_s2
```
- 默认路径：`pretrained_weights/direct3d_s2/`

#### 2.2 Reward Models

##### EVA Giant（默认评分）
```bash
python scripts/download/download_eva_weights.py
```
- 默认路径：`pretrained_weights/eva/`

##### Uni3D（可选）
- 若使用 Uni3D 相关评分，请按本仓库 `reward_models/uni3d_scorer` 说明或上游项目指引准备对应权重。
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