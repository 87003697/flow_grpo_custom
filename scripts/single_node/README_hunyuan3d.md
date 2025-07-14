# Hunyuan3D GRPO Training Scripts

本目录包含了用于启动Hunyuan3D GRPO训练的各种脚本配置。

## 📋 可用脚本

### 1. `main_hunyuan3d.sh` - 多配置训练脚本
包含1/2/4/8 GPU的不同配置选项，需要手动取消注释相应的配置。

### 2. `run_hunyuan3d_1gpu.sh` - 单GPU快速训练脚本
适用于单GPU环境的即用型训练脚本。

## 🚀 快速开始

### 单GPU训练（推荐开始）
```bash
./scripts/single_node/run_hunyuan3d_1gpu.sh
```

### 多GPU训练
编辑 `main_hunyuan3d.sh`，取消注释相应的GPU配置：
```bash
# 取消注释相应的配置行
vim scripts/single_node/main_hunyuan3d.sh

# 运行脚本
./scripts/single_node/main_hunyuan3d.sh
```

## ⚙️ 配置参数说明

### 采样参数
- `input_batch_size`: 每次处理的图像数量（建议1以节省内存）
- `num_meshes_per_image`: 每张图像生成的mesh候选数量
- `num_batches_per_epoch`: 每个epoch的采样批次数

### 训练参数
- `batch_size`: 训练时每个GPU的batch size
- `gradient_accumulation_steps`: 梯度累积步数
- `num_epochs`: 训练轮数（默认100）
- `save_freq`: 检查点保存频率（默认20）

### 有效批次大小计算
```
有效批次大小 = batch_size × gradient_accumulation_steps × num_gpus
```

## 💾 内存优化建议

### 单GPU (24GB VRAM)
- `batch_size=1`, `gradient_accumulation_steps=4`
- `num_meshes_per_image=2`
- `num_batches_per_epoch=2`

### 多GPU (48GB+ VRAM)
- `batch_size=1`, `gradient_accumulation_steps=2-3`
- `num_meshes_per_image=3`
- `num_batches_per_epoch=3-5`

## 📊 监控训练

训练日志将保存在 `logs/` 目录下，检查点保存在 `checkpoints/hunyuan3d_grpo/` 目录下。

可以通过以下方式监控训练进度：
- 查看终端输出的训练日志
- 检查 `logs/` 目录下的详细日志文件
- 使用 `nvidia-smi` 监控GPU内存使用情况

## 🔧 故障排除

### 内存不足 (OOM)
- 减少 `batch_size` 或 `num_meshes_per_image`
- 增加 `gradient_accumulation_steps` 保持有效批次大小
- 启用 `mixed_precision="fp16"`

### 训练速度慢
- 增加 `num_batches_per_epoch` 以获得更多训练样本
- 考虑使用多GPU配置
- 检查数据加载是否成为瓶颈

### 检查点错误
- 确保 `checkpoints/hunyuan3d_grpo/` 目录存在
- 检查磁盘空间是否充足
- 验证写入权限 