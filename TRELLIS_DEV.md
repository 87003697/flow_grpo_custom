# TRELLIS Stage 2 GRPO 训练实施计划

## 🎯 核心策略

**简化架构设计**：
- **Stage 1 (稀疏结构)**: 预训练权重固定，GRPO训练中在线推理
- **Stage 2 (SLAT生成)**: 使用GRPO进行强化学习训练
- **避免端到端复杂性**: 专注优化SLAT生成质量

## 🏗️ 技术架构

### 关键简化点
- Stage 1在线推理生成稀疏结构坐标
- 只训练SLatFlowModel，冻结其他组件
- 复用Hunyuan3D的GRPO训练框架
- 适配SparseTensor数据结构

### 核心组件
```
TrellisStage2Pipeline
├── 冻结Stage 1模型（稀疏结构在线生成）
├── 训练SLatFlowModel（SLAT生成）
└── 端到端推理流程

trellis_stage2_with_logprob
├── Stage 1在线推理（稀疏结构）
├── SLAT Flow采样 + LogProb计算
└── 解码为mesh输出
```

## 🔧 核心接口设计

### 1. 主要生成器接口 `generators/trellis/pipeline.py`
```python
class TrellisStage2Pipeline:
    """TRELLIS Stage 2训练管道包装类"""
    # 参考: generators/hunyuan3d/pipeline.py:21-56
    
    def __init__(self, model_path='JeffreyXiang/TRELLIS-image-large'):
        """初始化pipeline，加载预训练模型，Stage 1固定推理"""
        # 参考: generators/hunyuan3d/pipeline.py:24-27
        # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:46-68
        
    def _freeze_stage1(self):
        """冻结Stage 1相关模型，只训练SLatFlowModel"""
        # 设置sparse_structure_flow_model.requires_grad_(False)
        # 设置sparse_structure_encoder/decoder.requires_grad_(False)
        
    def forward_stage1(self, image_cond: Dict) -> torch.Tensor:
        """Stage 1在线推理生成稀疏结构坐标"""
        # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:162-197
        # 使用sample_sparse_structure方法
        
    def prepare_image_conditions(self, images: List[Image]) -> Dict[str, torch.Tensor]:
        """准备TRELLIS图像条件，使用DINOv2特征提取"""
        # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:119-140
        # 使用get_cond方法
        
    def forward_stage2_with_logprob(self, coords: torch.Tensor, image_cond: Dict, **kwargs) -> Tuple:
        """Stage 2推理+LogProb计算，基于在线生成的稀疏结构"""
        # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:219-252
        # 使用sample_slat方法，添加LogProb计算
```

### 2. GRPO训练补丁 `flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py`
```python
def trellis_stage2_with_logprob(pipeline, image_conds: Dict, **kwargs) -> Tuple[List, List, List, List]:
    """TRELLIS完整推理+LogProb计算，返回 (meshes, all_latents, all_log_probs, all_kl)"""
    # 参考: flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py:38-142
    # Stage 1: 冻结推理生成coords
    # Stage 2: SLAT采样 + LogProb记录
    # 解码: 转换为mesh输出
    
def decode_slat_to_mesh(pipeline, slat: sp.SparseTensor) -> List[trimesh.Trimesh]:
    """将SLAT解码为mesh格式"""
    # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:200-217
    # 使用decode_slat方法中的mesh分支
```

### 3. Flow LogProb计算 `flow_grpo/diffusers_patch/trellis_flow_with_logprob.py`
```python
def trellis_flow_step_with_logprob(flow_model, noise_pred, timestep, sample, prev_sample) -> Tuple:
    """Flow matching步骤+LogProb计算，返回 (prev_sample, log_prob, prev_sample_mean, std_dev)"""
    # 参考: flow_grpo/diffusers_patch/hunyuan3d_sde_with_logprob.py:全文
    # 适配TRELLIS的Flow matching调度器
    # 处理SparseTensor格式的概率密度计算
```

### 4. SparseTensor适配 `flow_grpo/diffusers_patch/sparse_tensor_grpo.py`
```python
def compute_log_prob_trellis_stage2(pipeline, sample: Dict, j: int, image_conds: Dict, config) -> Tuple:
    """Stage 2对数概率计算，处理SparseTensor格式"""
    # 参考: scripts/train_hunyuan3d.py:181-232 (compute_log_prob_3d)
    # 提取SparseTensor的coords和feats
    # SLatFlowModel前向传播
    # CFG处理和Flow matching步骤
    
def sparse_tensor_cat(tensors: List[sp.SparseTensor]) -> sp.SparseTensor:
    """SparseTensor的批量拼接操作，用于CFG处理"""
    # 基于TRELLIS SparseTensor.cat方法
    # 处理coords对齐和feats拼接
    
def sparse_tensor_chunk(tensor: sp.SparseTensor, chunks: int) -> List[sp.SparseTensor]:
    """SparseTensor的分块操作，用于CFG分离正负条件"""
    # 基于torch.chunk逻辑适配SparseTensor
    # 保持coords不变，分割feats维度
```

### 5. 主训练函数 `scripts/train_trellis.py`
```python
def main(argv):
    """主训练函数，包含完整GRPO训练循环"""
    # 参考: scripts/train_hunyuan3d.py:447-1231 (main函数)
    # 初始化TrellisStage2Pipeline
    # 设置LoRA目标为SLatFlowModel
    # GRPO训练循环：采样-奖励-训练
    
def eval_trellis_stage2(pipeline, test_dataloader, config, accelerator, epoch, mesh_scorer):
    """TRELLIS Stage 2评估函数，生成mesh并计算奖励"""
    # 参考: scripts/train_hunyuan3d.py:317-421 (eval_hunyuan3d)
    # 使用EMA权重评估
    # Stage 1/2完整推理
    # mesh质量评估和可视化
    
def save_ckpt_trellis(model, ema, optimizer, epoch, global_step, save_dir, accelerator):
    """checkpoint保存函数，支持LoRA权重保存"""
    # 参考: scripts/train_hunyuan3d.py:262-315 (save_ckpt_hunyuan3d)
    # 只保存SLatFlowModel的LoRA权重
    # 保存EMA状态和训练元数据
    
def calculate_zero_std_ratio_trellis(image_names, gathered_rewards):
    """计算图像组奖励标准差为零的比例"""
    # 参考: scripts/train_hunyuan3d.py:422-444 (calculate_zero_std_ratio_images)
    # 复用相同逻辑，无需修改
```

### 6. 数据处理 - 复用现有组件
```python
# 直接复用以下类，无需重新实现
class Image3DDataset(Dataset):
    """图像数据集类，TRELLIS使用相同的图像输入格式"""
    # 参考: scripts/train_hunyuan3d.py:56-101
    
class DistributedImageRepeatSampler(Sampler):
    """分布式重复采样器，确保图像间的group比较"""
    # 参考: scripts/train_hunyuan3d.py:103-179
```

### 7. 工具函数 `generators/trellis/utils.py`
```python
def trellis_preprocess_image(image: Image) -> Image:
    """TRELLIS图像预处理，包含背景移除等"""
    # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:82-118
    # 使用preprocess_image方法
    
def convert_trellis_to_trimesh(slat_outputs: Dict) -> List[trimesh.Trimesh]:
    """将TRELLIS输出转换为trimesh格式，用于奖励计算"""
    # 参考: _reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:211-217
    # 使用decode_slat方法的mesh分支
```

### 8. 配置相关 `config/trellis_stage2_grpo.py`
```python
def get_trellis_grpo_config():
    """TRELLIS Stage 2 GRPO训练配置"""
    # 参考: 现有config文件结构
    # 添加TRELLIS特有参数:
    # - sparse_structure_sampler_params: Stage 1采样参数
    # - slat_sampler_params: Stage 2采样参数
    # - stage1_frozen: True
    # - target_modules: ["attn.to_q", "attn.to_k", "attn.to_v", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
```

## 📁 文件路径架构

### 1. 主训练脚本
```
scripts/
└── train_trellis.py                    # 主训练脚本，仿照train_hunyuan3d.py
```
### 2. TRELLIS生成器封装
```
generators/
└── trellis/
    ├── __init__.py
    ├── pipeline.py                     # TrellisStage2Pipeline包装类
    ├── utils.py                        # 工具函数
    └── patches/
        └── sparse_tensor_utils.py      # SparseTensor工具函数
```
### 3. GRPO训练补丁
```
flow_grpo/
└── diffusers_patch/
    ├── trellis_stage2_with_logprob.py  # Stage 2 + LogProb推理函数
    ├── trellis_flow_with_logprob.py    # Flow步骤+LogProb计算
    └── sparse_tensor_grpo.py           # SparseTensor的GRPO适配
```
### 4. 配置文件
```
config/
└── trellis_stage2_grpo.py             # TRELLIS Stage 2 GRPO训练配置
```


## 📊 实施时间表

| 阶段 | 时间 | 核心任务 |
|------|------|----------|
| **架构设计** | 1天 | 确定Stage 1/2在线推理流程，设计Pipeline接口 |
| **核心开发** | 3天 | TrellisStage2Pipeline + LogProb计算 |
| **训练逻辑** | 2天 | train_trellis.py + GRPO训练循环 |
| **配置调试** | 1天 | 配置文件 + 奖励函数 + 测试 |
| **总计** | **7天** | 完整的train_trellis.py |

## 📝 代码文件创建顺序

### Day 1: 基础架构 (4个文件)
```bash
# 1. 创建基础目录结构
mkdir -p generators/trellis/patches
mkdir -p flow_grpo/diffusers_patch

# 2. 基础包装器 - 参考 generators/hunyuan3d/pipeline.py
touch generators/trellis/__init__.py
touch generators/trellis/pipeline.py        # TrellisStage2Pipeline核心类

# 3. SparseTensor工具函数
touch generators/trellis/patches/sparse_tensor_utils.py  # SparseTensor处理工具
touch generators/trellis/utils.py           # 图像预处理和条件编码
```

### Day 2: GRPO补丁核心 (3个文件)  
```bash
# 4-6. GRPO训练补丁 - 参考 flow_grpo/diffusers_patch/hunyuan3d_*
touch flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py   # 完整推理+LogProb
touch flow_grpo/diffusers_patch/trellis_flow_with_logprob.py     # Flow步骤+LogProb
touch flow_grpo/diffusers_patch/sparse_tensor_grpo.py            # SparseTensor GRPO适配
```

### Day 3-4: 核心计算逻辑 (2个文件)
```bash
# 7. SparseTensor LogProb计算 - 参考 scripts/train_hunyuan3d.py:181-232
# 在 sparse_tensor_grpo.py 中实现 compute_log_prob_trellis_stage2

# 8. 完整Pipeline测试
touch scripts/test_trellis_pipeline.py      # 单元测试脚本，验证推理流程
```

### Day 5: 主训练脚本 (1个文件)
```bash
# 9. 主训练脚本 - 参考 scripts/train_hunyuan3d.py 完整结构
touch scripts/train_trellis.py              # 1200+行主训练脚本
```

### Day 6: 评估和工具 (2个文件)
```bash
# 10-11. 评估和保存函数
# 在 train_trellis.py 中实现:
# - eval_trellis_stage2()          # 参考 train_hunyuan3d.py:317-421
# - save_ckpt_trellis()            # 参考 train_hunyuan3d.py:262-315
```

### Day 7: 配置和测试 (2个文件)
```bash
# 12. 配置文件 - 参考现有config结构
touch config/trellis_stage2_grpo.py         # GRPO训练配置

# 13. 端到端测试
touch scripts/test_trellis_e2e.py           # 端到端训练测试
```

## 📋 文件依赖关系

```
generators/trellis/pipeline.py
├── 依赖: generators/trellis/utils.py
├── 依赖: generators/trellis/patches/sparse_tensor_utils.py
└── 被依赖: scripts/train_trellis.py

flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py  
├── 依赖: trellis_flow_with_logprob.py
├── 依赖: sparse_tensor_grpo.py
└── 被依赖: scripts/train_trellis.py

scripts/train_trellis.py
├── 依赖: generators/trellis/pipeline.py
├── 依赖: flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py
├── 依赖: sparse_tensor_grpo.py (compute_log_prob_trellis_stage2)
└── 依赖: config/trellis_stage2_grpo.py
```

**总计**: ~2400行代码，13个文件

## 🔧 技术难点

### 1. SparseTensor LogProb计算
- 适配TRELLIS的稀疏张量格式 (coords + feats)
- 基于Flow matching重新推导概率密度计算

### 2. Stage 1在线推理
- 保持Stage 1冻结，确保梯度不回传
- 优化Stage 1推理速度，避免训练瓶颈

### 3. LoRA配置
- 只对SLatFlowModel应用LoRA
- 目标模块：["attn.to_q", "attn.to_k", "attn.to_v", "attn.out_proj", "mlp.fc1", "mlp.fc2"]

## 📋 开发检查清单

- [ ] **Day 1**: 目录结构创建 + pipeline.py 框架
- [ ] **Day 2**: GRPO补丁文件创建 + 基础接口
- [ ] **Day 3**: SparseTensor LogProb 核心逻辑
- [ ] **Day 4**: Pipeline完善 + 单元测试
- [ ] **Day 5**: train_trellis.py 主脚本完成
- [ ] **Day 6**: 评估函数 + checkpoint保存
- [ ] **Day 7**: 配置文件 + 端到端测试

**目标**: 7天内完成可运行的 `train_trellis.py`，支持TRELLIS Stage 2的GRPO训练
