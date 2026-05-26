# CODE_INDEX.md — flow_grpo_custom 模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| **flow_grpo** | `flow_grpo/` | 核心训练框架（GRPO + Flow Matching） |
| **edit4shape** | `edit4shape/` | 3D 形状编辑管线（数据集、生成器、渲染器、引导系统） |
| **reward_models** | `reward_models/` | 奖励模型（PICKScore、相机法线、Uni3D 等） |
| **generators** | `generators/` | 3D 生成后端（Hunyuan3D） |
| **config** | `config/` | 训练配置（Trellis 各阶段蒸馏/对比学习） |
| **scripts** | `scripts/` | 集群脚本 |
| **dataset** | `dataset/` | 数据集 |
| **docs** | `docs/` | 文档 |

## 核心子模块

### flow_grpo/
| 文件 | 说明 |
|------|------|
| `diffusers_patch/` | Diffusers 定制补丁 |
| `peft_sparse/` | 稀疏 LoRA / PEFT |
| `ema.py` | EMA 权重更新 |
| `prompts.py` | Prompt 管理 |
| `stat_tracking.py` | 训练统计跟踪 |

### config/
| 配置 | 说明 |
|------|------|
| `trellis_stage2_distillation.py` | Stage2 蒸馏 |
| `trellis_stage2_contrastive.py` | Stage2 对比学习 |
| `trellis2_shape_distillation.py` | Trellis2 形状蒸馏 |
| `trellis2_shape_tex_contrastive.py` | Trellis2 形状+纹理对比 |
| `dgx.py` | DGX 集群配置 |
