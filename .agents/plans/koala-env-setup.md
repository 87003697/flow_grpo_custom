# Plan: Koala 集群环境配置 — flow_grpo_custom

## 目标

编写 `scripts/setup_koala.sh`，让项目在 Koala 集群上一键恢复环境并可运行训练。

## 状态

**当前阶段**: Done

---

## 使用方式

### 首次（一个 debug pod 上执行一次，之后所有 pod 复用）

```bash
# 本地 Mac：上传代码
cd ~/Desktop/codes/flow_grpo_custom
s5cmd sync --exclude '.git/*' --exclude '.venv/*' --exclude '*/__pycache__/*' \
    --exclude 'wandb/*' --exclude 'logs*' . "s3://arcwm-code-us-west-2/ericzyma/flow_grpo_custom/"

# 提交 debug pod
koala submit --code "s3://arcwm-code-us-west-2/ericzyma/flow_grpo_custom:/data/work/run_codes"

# SSH 进 pod，首次 setup（下载 + 编译 + 缓存到 S3，约 20-30 min）
ssh koala
cd /data/work/run_codes
. scripts/setup_koala.sh --download
```

### 日常恢复（后续 pod 重启或新 pod）

```bash
. scripts/setup_koala.sh --fast    # ~1-2 min（pip 缓存 + tar 恢复）
```

### S3 布局（--download 自动创建）

```
/threed-code/ericzyma/
├── data/flow_grpo/
│   ├── pretrained_weights.tar      # TRELLIS.2-4B + DINOv3 (~15 GB)
│   ├── alphaimages_v3.tar          # 训练数据集
│   └── trellis2_reference.tar      # _reference_codes/TRELLIS.2
└── tools/
    ├── flow_grpo_cuda_ext.tar      # 编译好的 nvdiffrast/CuMesh/FlexGEMM
    └── hf_cache_qwen-image-edit-2511.tar  # Guidance 模型
```

## 凭证

| 变量 | 值 | 位置 |
|------|-----|------|
| HF_TOKEN | (see ~/.zshrc) | `~/.zshrc`（已配置） |
| WANDB_API_KEY | (see ~/.zshrc) | `~/.zshrc`（已配置） |

Koala 容器自动注入这两个环境变量（从本地 shell 读取）。
