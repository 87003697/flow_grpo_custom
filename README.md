# Hunyuan3D Flow-GRPO

基于 Flow-GRPO 框架的 Hunyuan3D 训练代码。

## 环境配置

### 1. 创建环境
```bash
# 创建并激活环境
conda create -n grpo3d python=3.10.16 -y
conda activate grpo3d

# 安装基础 (CUDA 12.4 对应官方 PyTorch wheel)
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# 关键深度学习与扩散/训练组件
pip install transformers==4.40.0 diffusers==0.33.1 accelerate==1.4.0 peft==0.10.0

# 数学 / 科学计算与图像
pip install numpy==1.26.4 scipy==1.15.2 matplotlib==3.10.0 \
            scikit-learn==1.6.1 scikit-image==0.25.2 \
            opencv-python-headless==4.11.0.86 pillow==10.4.0

# 性能与序列化
pip install deepspeed==0.16.4 safetensors==0.5.3 huggingface-hub==0.29.1 tokenizers==0.19.1

# 稀疏卷积 (会自动拉取匹配的 cumm-cu124)  -- 建议先于其余依赖安装，避免编译链冲突
pip install spconv-cu124==2.3.8

# (可选, 推荐) 安装 flash-attn 加速注意力
export TORCH_CUDA_ARCH_LIST="80;86;89;90"   # 根据 GPU 精简
pip install --no-build-isolation flash-attn==2.8.3

# 其余依赖一次性安装
pip install -r requirements.txt

# 安装后快速验证
python - <<'PY'
import torch, flash_attn, spconv
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Flash-Attn OK, SpConv OK')
PY
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

### 2.1 下载 Direct3D‑S2 权重（用于 Stage1 最小集成与测试）
```bash
# 指定输出目录（默认指向 /home/zhiyuan_ma/code/Direct3D-S2/direct3d_s2-v-1-1）
python scripts/download/download_direct3d_s2.py \
  --repo_id JeffreyXiang/Direct3D-S2 \
  --out /home/zhiyuan_ma/code/Direct3D-S2/direct3d_s2-v-1-1

# 如需 1024 分辨率与 refiner_1024 权重
python scripts/download/download_direct3d_s2.py \
  --repo_id JeffreyXiang/Direct3D-S2 \
### 注意力后端兼容性说明
| 组合 | 状态 | 备注 |
|------|------|------|
| torch 2.5.1+cu124 + flash-attn 2.8.3 | ✅ 稳定 | 已验证前向 (fp16) 正常 |
| torch 2.6.0 + flash-attn 2.8.x | ❌ 失败 | C++ undefined symbol (c10::Error) |
| torch 2.6.0 + xformers 0.0.29.post3 | ✅ 可用 | 若需此组合请单独新环境 |

切换方式：
```bash
# 使用 flash-attn (默认)
export ATTN_BACKEND=flash_attn
# 或强制 xformers (需满足其 torch 版本依赖并卸载 flash-attn)
# export ATTN_BACKEND=xformers
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

运行前建议设置下列环境变量（与项目脚本一致）：
```bash
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=auto
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

## Direct3D‑S2 阶段1（最小）集成与测试

### 1) 构建 CUDA 扩展 `udf_ext`

参考 `_reference_codes/Direct3D-S2` 的第三方模块，需要先构建 `udf_ext`（CUDA）以支持网格 UDF 相关操作。

```bash
conda activate grpo3d
# 在 Direct3D-S2 仓库下构建并安装
cd /home/zhiyuan_ma/code/Direct3D-S2/third_party/voxelize
pip install -v .

# 如果导入报 libc10.so 找不到，运行时临时添加 LD_LIBRARY_PATH（PyTorch lib 路径）
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
python -c "import udf_ext; print('UDF_EXT_OK')"
```

### 2) 运行最小单测与端到端推理

```bash
# 仅单测（验证 log_prob 常数差与固定种子可复现）
env PYTHONPATH=/home/zhiyuan_ma/code2/flow_grpo_custom \
conda run -n grpo3d python /home/zhiyuan_ma/code2/flow_grpo_custom/scripts/debug/test_direct3d_s2_stage1_minimal.py

# 端到端（需要准备权重目录 direct3d_s2-v-1-1 与输入图像）
env PYTHONPATH=/home/zhiyuan_ma/code2/flow_grpo_custom \
LD_LIBRARY_PATH=/home/zhiyuan_ma/miniconda3/envs/grpo3d/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
conda run -n grpo3d python /home/zhiyuan_ma/code2/flow_grpo_custom/scripts/debug/test_direct3d_s2_stage1_minimal.py \
  --do_e2e \
  --pipeline_path /home/zhiyuan_ma/code/Direct3D-S2/direct3d_s2-v-1-1 \
  --image /home/zhiyuan_ma/code2/flow_grpo_custom/dataset/eval3d_hunyuan3d/images/1772.png \
  --out /home/zhiyuan_ma/code2/flow_grpo_custom/outputs/test_runs/direct3d_s2_minimal \
  --device cuda \
  --candidates 1
```

说明：
- 权重目录需要包含 `config.yaml` 与 `model_*.ckpt` 文件。
- 若权重在其他路径，可建立软链接至 `/home/zhiyuan_ma/code/Direct3D-S2/direct3d_s2-v-1-1`。
- 端到端脚本会导出 PLY 网格并打印 log_prob 统计。

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