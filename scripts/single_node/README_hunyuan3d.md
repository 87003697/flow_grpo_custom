# Hunyuan3D GRPO Training Scripts

本目录包含了用于启动Hunyuan3D GRPO训练的脚本。

## 📋 可用脚本

### 1. `run_memory_optimized.sh` - 推荐使用 ⭐
最新的、已验证的单GPU训练脚本，包含所有必要的内存优化。

```bash
bash scripts/single_node/run_memory_optimized.sh
```

### 2. `run_hunyuan3d_sd3_memory.sh` - 高级选项
展示不同内存优化策略的脚本（aggressive/moderate/conservative）。

```bash
# 激进内存优化（推荐）
bash scripts/single_node/run_hunyuan3d_sd3_memory.sh aggressive

# 中等内存优化
bash scripts/single_node/run_hunyuan3d_sd3_memory.sh moderate

# 保守策略
bash scripts/single_node/run_hunyuan3d_sd3_memory.sh conservative
```

## 🚀 快速开始

### 单GPU训练（推荐）
```bash
bash scripts/single_node/run_memory_optimized.sh
```

## ⚙️ 配置参数说明

### 采样参数
- `input_batch_size=1`: 每次处理1张图像（内存优化）
- `num_meshes_per_image=2`: 每张图像生成2个mesh候选
- `num_batches_per_epoch=1`: 每个epoch的采样批次数

### 训练参数
- `batch_size=1`: 训练批次大小
- `gradient_accumulation_steps=1`: 梯度累积步数
- `num_epochs=5`: 训练轮数
- `save_freq=5`: 检查点保存频率

## 💾 内存优化

当前配置已针对单GPU环境优化：
- 使用激进内存优化策略
- 8bit Adam优化器
- BF16混合精度训练

## 📊 监控训练

- 训练日志：终端输出
- 检查点：`checkpoints/` 目录
- GPU监控：`nvidia-smi`

## 🔧 故障排除

### 内存不足 (OOM)
- 确保使用了正确的配置文件
- 检查GPU内存是否充足（建议16GB+）
- 减少batch_size或num_meshes_per_image 