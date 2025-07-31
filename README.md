# Hunyuan3D Flow-GRPO

基于 Flow-GRPO 框架的 Hunyuan3D 训练代码。

## 环境配置

### 1. 创建环境
```bash
# 创建并激活环境
conda create -n grpo3d python=3.10.16
conda activate grpo3d

# 安装基础依赖
pip install torch==2.6.0 torchvision==0.21.0
pip install transformers==4.40.0 diffusers==0.33.1 accelerate==1.4.0
pip install numpy==1.26.4 scipy==1.15.2 matplotlib==3.10.0
pip install scikit-learn==1.6.1 scikit-image==0.25.2
pip install opencv-python-headless==4.11.0.86 pillow==10.4.0

# 安装性能优化相关
pip install peft==0.10.0 deepspeed==0.16.4 safetensors==0.5.3
pip install huggingface-hub==0.29.1 tokenizers==0.19.1

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型
```bash
# 登录 Hugging Face（如果需要）
huggingface-cli login

# 下载 Hunyuan3D 模型
python scripts/download/download_hunyuan3d_weights.py

# 下载 EVA Giant 模型（用于评分）
python scripts/download/download_eva_weights.py
```

下载的模型将被保存在以下位置：
- Hunyuan3D 模型：`pretrained_weights/tencent/Hunyuan3D-2.1/`
  - DiT 模型：`hunyuan3d-dit-v2-1/`
  - VAE 模型：`hunyuan3d-vae-v2-1/`
- EVA Giant 模型：`pretrained_weights/eva/`

### 3. 硬件要求
- GPU 显存 ≥ 16GB
- CUDA 12.4 或更高版本
- Python 3.10.16

## 开始训练

推荐使用内存优化版本的训练脚本：
```bash
bash scripts/single_node/run_memory_optimized.sh
```

## 主要配置

配置文件位于 `config/hunyuan3d.py`，包含以下主要参数：

```python
# 采样参数
input_batch_size = 1          # 每次处理图像数
num_meshes_per_image = 2      # 每张图像生成的 mesh 数量
num_batches_per_epoch = 1     # 每轮采样批次数

# 训练参数
batch_size = 1               # 训练批次大小
num_epochs = 5               # 训练轮数
save_freq = 5               # 保存检查点频率
```

## 引用
```
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