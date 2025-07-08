# Hunyuan3D Shape Generator 开发方案

## 概述

将 Hunyuan3D 2.1 Image-to-3D Shape Generator 集成到 Flow-GRPO 训练框架的**最简化**方案。

**目标**：用强化学习训练 Hunyuan3D 形状生成器，从单张图像生成高质量的3D网格。

**原则**：先做最简单能跑的版本，后续按需迭代。

---

## 验证方案 📊

### 目标验证内容
1. **基础集成验证**
   - 能加载 Hunyuan3D 模型
   - 能处理图像-3D网格数据
   - Flow-GRPO 训练循环能正常运行

2. **端到端测试**
   - 数据 → 模型 → 奖励 → 更新的完整流程
   - 能生成3D网格并计算质量分数

### 成功标准
- ✅ 训练流程不报错
- ✅ 能生成3D mesh文件
- ✅ 奖励函数能正常计算
- ✅ 模型参数能正常更新

### 当前可用验证脚本
```bash
# 测试Hunyuan3D集成和渲染
python scripts/test_hunyuan3d.py

# 测试不同体积解码器性能
python scripts/test_volume_decoders_simple.py

# 测试训练脚本（2D图像生成）
python scripts/train_sd3.py --config config/dgx.py:pickscore_sd3
```

---

## 当前架构设计 🏗️

### 目录结构
```
flow_grpo_custom/
├── generators/                   # 生成器模块
│   ├── __init__.py
│   └── hunyuan3d/               # Hunyuan3D集成模块
│       ├── __init__.py
│       ├── pipeline.py          # 我们的推理管道封装
│       ├── hy3dshape/           # 原始Hunyuan3D模块
│       │   ├── pipelines.py     # 核心推理管道
│       │   ├── preprocessors.py # 预处理器
│       │   ├── postprocessors.py # 后处理器
│       │   ├── rembg.py         # 背景移除
│       │   ├── schedulers.py    # 调度器
│       │   ├── surface_loaders.py # 表面加载器
│       │   ├── models/          # 模型代码
│       │   ├── utils/           # 工具代码
│       │   └── data/            # 数据目录
│       └── patches/             # 补丁文件
│           ├── pytorch_rmsnorm_patch.py
│           └── torchvision_fix.py
├── reward_models/               # 奖励函数模块
│   ├── __init__.py
│   ├── rewards.py               # 2D图像奖励函数（已有）
│   ├── pickscore_scorer.py      # PickScore评分器（已有）
│   ├── uclip_scorer.py          # UCLIP 3D奖励函数（待实现）
│   └── uni3d_scorer.py          # Uni3D 3D奖励函数（待实现）
├── flow_grpo/                   # 原有框架
│   ├── stat_tracking.py         # 统计跟踪
│   ├── ema.py                   # 指数移动平均
│   ├── prompts.py               # 提示词处理
│   ├── diffusers_patch/         # Diffusers补丁
│   └── assets/                  # 资源文件
├── config/                      # 配置文件
│   └── dgx.py                   # 训练配置
├── scripts/                     # 脚本文件
│   ├── train_sd3.py             # 2D图像训练脚本（已有）
│   ├── test_hunyuan3d.py        # Hunyuan3D集成测试（已有）
│   ├── test_volume_decoders_simple.py # 体积解码器测试（已有）
│   ├── train_hunyuan3d.py       # 3D训练脚本（待实现）
│   └── test_integration_3d.py   # 3D端到端测试（待实现）
├── dataset/                     # 数据集
└── requirements.txt             # 依赖文件
```

### 核心代码设计

#### 1. 当前已实现的Hunyuan3D管道
```python
# generators/hunyuan3d/pipeline.py
class Hunyuan3DPipeline:
    """Hunyuan3D推理管道的封装"""
    
    def __init__(self, model_path='tencent/Hunyuan3D-2.1'):
        print(f"🚀 正在加载Hunyuan3D模型: {model_path}")
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        self.rembg = BackgroundRemover()
        print("✅ Hunyuan3D模型加载成功")
    
    def generate_mesh(self, image_path_or_pil):
        """从图像生成3D mesh"""
        # 实现细节已完成...
        return mesh
```

#### 2. 当前已实现的奖励函数系统
```python
# reward_models/rewards.py
def multi_score(device, score_dict):
    """多奖励函数组合器"""
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
    }
    # 实现细节已完成...
```

#### 3. 待实现的3D训练适配器
```python
# flow_grpo/trainer_3d.py
class FlowGRPOHunyuan3DTrainer:
    def __init__(self):
        # 加载Hunyuan3D模型
        from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
        self.model = Hunyuan3DPipeline()
        
        # 使用原有的GRPO训练逻辑
        self.grpo_trainer = FlowGRPOTrainer(...)
        
        # 添加渲染器（已有）
        from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import SimpleKiuiRenderer
        self.renderer = SimpleKiuiRenderer()
    
    def train_step(self, batch):
        images, target_meshes = batch
        generated_meshes = self.model.generate_mesh(images[0])
        rewards = compute_mesh_quality(generated_meshes, target_meshes)
        
        # 每100步保存一次可视化
        if self.step % 100 == 0:
            rendered_image = self.renderer.render_single_view(generated_meshes)
            # 保存渲染图像...
        
        return self.grpo_trainer.update(generated_meshes, rewards)
```

#### 4. 待实现的3D奖励评分器
```python
# reward_models/uclip_scorer.py
class UCLIPScorer:
    """基于UCLIP的3D mesh质量评估"""
    
    def __init__(self, device="cuda"):
        # 加载UCLIP预训练模型
        self.device = device
        self.load_model()
    
    def score_mesh(self, mesh, text_prompt):
        """评估mesh与文本提示的一致性"""
        # UCLIP评分逻辑...
        return score

# reward_models/uni3d_scorer.py  
class Uni3DScorer:
    """基于Uni3D的3D mesh质量评估"""
    
    def __init__(self, device="cuda"):
        # 加载Uni3D预训练模型
        self.device = device
        self.load_model()
    
    def score_mesh(self, mesh, reference_features):
        """评估mesh的语义质量"""
        # Uni3D评分逻辑...
        return score

# 组合评分函数
def compute_mesh_quality(generated_meshes, prompts):
    """综合3D mesh质量评估"""
    uclip_scorer = UCLIPScorer()
    uni3d_scorer = Uni3DScorer()
    
    scores = []
    for mesh, prompt in zip(generated_meshes, prompts):
        # UCLIP语义一致性评分
        uclip_score = uclip_scorer.score_mesh(mesh, prompt)
        # Uni3D语义质量评分
        uni3d_score = uni3d_scorer.score_mesh(mesh, None)
        # 基础几何质量指标
        geometric_score = compute_geometric_quality(mesh)
        
        total_score = uclip_score + uni3d_score + geometric_score
        scores.append(total_score)
    return scores
```

#### 5. 当前已有的渲染器 ✅
```python
# generators/hunyuan3d/hy3dshape/utils/visualizers/renderer.py
class SimpleKiuiRenderer:
    """已实现的Kiui mesh渲染器"""
    
    def __init__(self, width=512, height=512, device="cuda"):
        # 渲染器已完全实现...
        
    def render_single_view(self, elevation=30.0, azimuth=45.0, distance=2.0):
        """渲染单个视图 - 已实现"""
        return rendered_image

def simple_render_mesh(mesh_path, save_path, device="cuda"):
    """简单的mesh渲染函数 - 已实现"""
    # 完整实现已存在...
```

---

## 分阶段实现计划 🚀

### 第一步：集成Hunyuan3D并验证一致性 ✅
**目标**：确保Hunyuan3D模型能正常工作，输出与官方一致

#### **✅ 已完成任务**：
1. **集成Hunyuan3D核心代码**
   - ✅ 复制`hy3dshape`模块到`generators/hunyuan3d/`
   - ✅ 创建`generators/hunyuan3d/pipeline.py`封装推理
   - ✅ 实现基础的mesh输出处理

2. **验证一致性**
   - ✅ 创建`scripts/test_hunyuan3d.py`进行集成测试
   - ✅ 能够加载模型并生成mesh
   - ✅ 确保生成的mesh能正常保存

3. **基础可视化**
   - ✅ 实现基础的mesh渲染功能
   - ✅ 能够生成多视角渲染图

4. **📊 额外完成：三种解码器性能验证**
   - ✅ VanillaVolumeDecoder: 稳定基准 (49.89秒)
   - ✅ HierarchicalVolumeDecoding: 智能回退修复，最快 (23.35秒) 
   - ✅ FlashVDMVolumeDecoding: 最高质量 (25.77秒)

5. **📊 已完成重构**：
   - ✅ 代码模块化重构完成
   - ✅ `generators/hunyuan3d/` 目录结构完善
   - ✅ `reward_models/` 奖励函数模块独立
   - ✅ 导入路径和依赖关系修复
   - ✅ 所有验证测试通过

**🎯 第一阶段状态：✅ 完全完成**

### 第二步：集成先进的3D奖励函数 🔄
**目标**：选择Uni3D或ULIP预训练模型实现高质量3D奖励函数

#### **选择方案**：
- **方案A：Uni3D** - 语义一致性更强（推荐）
- **方案B：ULIP** - 多模态对齐更全面

#### **具体任务**：
1. **选择并实现3D奖励函数**
   - 创建 `reward_models/mesh_basic_scorer.py` 基础几何质量评分器
   - ✅ 实现基础几何质量指标（顶点面数比、面积分布、边长分布、几何复杂度）
   - ✅ 集成 kiui mesh 格式支持
   - 🔄 创建 `reward_models/uni3d_scorer.py` 基于Uni3D（进行中）
   - ⏳ 创建 `reward_models/ulip_scorer.py` 基于ULIP

2. **验证一致性**
   - 创建 `scripts/test_3d_scorers.py`
   - 验证我们的评分器与 `_reference_codes` 官方效果保持一致

3. **成功标准**
   - UCLIP和Uni3D评分器能正常计算
   - 与官方代码效果一致
   - 能够区分不同质量的3D mesh
   - 多模态评分系统工作正常

#### **✅ 已完成验证**：
1. **基础几何质量评分器**
   - ✅ 实现了 `MeshBasicScorer` 类
   - ✅ 支持 kiui mesh 格式处理
   - ✅ 在真实数据集（25个.glb文件）上测试通过
   - ✅ 评分范围：0.7242 - 0.8428，平均：0.7809
   - ✅ 能有效区分不同质量的3D mesh

2. **Kiui Mesh 格式支持**
   - ✅ Hunyuan3D 管道支持输出 kiui 格式
   - ✅ 评分器支持 kiui mesh 输入
   - ✅ 提供 GPU 加速的 mesh 处理能力

**🎯 第二阶段状态：🔄 部分完成**

### 第三步：适配GRPO训练到3D生成 ⏳
**目标**：将Hunyuan3D集成到GRPO训练框架

#### **具体任务**：
1. **训练适配器**
   - 创建`flow_grpo/trainer_3d.py`
   - 适配GRPO训练逻辑到3D生成
   - 实现梯度更新和参数优化

2. **端到端训练**
   - 创建`scripts/train_hunyuan3d.py`和`config/train_3d.py`
   - 实现完整的3D训练循环
   - 添加checkpoint保存/恢复

3. **训练验证**
   - 创建`scripts/test_integration_3d.py`
   - 验证完整3D训练流程
   - 确保训练loss正常下降

4. **成功标准**
   - 3D训练流程不报错
   - 训练loss稳定下降
   - 生成mesh质量有改善
   - 完整3D训练循环正常工作

**🎯 第三阶段状态：⏳ 等待中**

---

## 📋 当前文件优先级

### 第一步重点文件 ✅
1. `generators/hunyuan3d/pipeline.py` - 核心推理封装 ✅
2. `scripts/test_hunyuan3d.py` - 一致性验证 ✅
3. `scripts/test_volume_decoders_simple.py` - 性能验证 ✅

### 第二步重点文件 ��
1. `reward_models/mesh_basic_scorer.py` - 基础几何质量评分器 ✅
2. `scripts/mesh_basic_scorer_test.py` - 3D评分器批量测试 ✅
3. `reward_models/uni3d_scorer.py` - Uni3D语义评分器 🔄
4. `reward_models/ulip_scorer.py` - ULIP语义评分器 ⏳

### 第三步重点文件 ⏳
1. `flow_grpo/trainer_3d.py` - 3D训练适配器
2. `scripts/train_hunyuan3d.py` - 3D训练脚本
3. `config/train_3d.py` - 3D训练配置
4. `scripts/test_integration_3d.py` - 3D端到端测试


### 依赖安装
```bash
# 当前已安装依赖
pip install trimesh matplotlib scipy torch transformers diffusers accelerate
pip install open_clip_torch loguru

# 额外需要的3D依赖
pip install pyrender pyglet PyOpenGL PyOpenGL_accelerate

# 新增的3D mesh处理依赖
pip install kiui  # 3D mesh处理和GPU加速
```

### 环境搭建
```bash
# 目录结构已创建
generators/hunyuan3d/hy3dshape/    # ✅ 已完成
generators/hunyuan3d/patches/      # ✅ 已完成

# 核心模块已复制
# ✅ Hunyuan3D核心模块已就位
# ✅ 补丁文件已应用

# 安装基础依赖
pip install -r requirements.txt
```

---

## 注意事项

### 与现有2D框架的区别
- **输入类型**：单张图像 → 3D网格（而非文本 → 图像）
- **奖励函数**：3D几何质量+语义一致性（而非2D图像质量）
- **输出格式**：3D mesh文件(.glb/.obj)（而非图像文件）
- **训练策略**：需要适应3D生成的特殊性

### 硬件要求
- **GPU**: 16GB+ VRAM（已验证）
- **内存**: 32GB+ RAM
- **存储**: 10GB+

### 当前可用的快速验证
```bash
# 测试Hunyuan3D核心功能
python scripts/test_hunyuan3d.py

# 测试不同解码器性能
python scripts/test_volume_decoders_simple.py

# 测试2D训练脚本（参考）
python scripts/train_sd3.py --config config/dgx.py:pickscore_sd3
```

---

## 成功标准

### 第一阶段完成标准 ✅
- ✅ 加载Hunyuan3D模型
- ✅ 处理图像输入并生成3D mesh
- ✅ 保存3D mesh文件
- ✅ 基础渲染可视化
- ✅ 性能基准测试完成

### 第二阶段完成标准 🔄
- ✅ 实现基础几何质量评分器
- ✅ 验证评分器在真实数据集上的效果
- ✅ 集成 kiui mesh 格式支持
- 🔄 实现 Uni3D 语义评分器
- ⏳ 实现 ULIP 语义评分器
- ⏳ 验证与官方代码一致性

### 第三阶段完成标准 🎯
- ⏳ 完整3D训练流程
- ⏳ 训练loss稳定下降
- ⏳ 生成mesh质量有改善
- ⏳ 端到端训练循环正常工作
