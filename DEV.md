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

### 验证脚本
```bash
python scripts/test_integration.py  # 一个脚本测试所有功能
```

---

## 最简化架构设计 🏗️

### 目录结构
```
flow_grpo_3d/
├── flow_grpo/                    # 原有框架，稍作修改
│   ├── trainer.py                # 原有训练器
│   ├── trainer_3d.py             # 新增：3D训练适配器  
│   ├── rewards_3d.py             # 新增：3D奖励函数
│   └── datasets_3d.py            # 新增：3D数据集加载
├── hunyuan3d/                    # 直接复制Hunyuan3D核心代码
│   ├── models/                   # 复制模型代码
│   ├── utils/                    # 复制工具代码
│   └── pipeline.py               # 复制推理管道
├── utils/
│   ├── mesh_utils.py             # 简单的mesh处理工具
│   └── render_utils.py           # 训练时mesh可视化
├── config/
│   └── train_3d.py               # 一个配置文件
├── scripts/
│   ├── test_integration.py       # 集成测试
│   └── train.py                  # 训练脚本
└── requirements_3d.txt           # 额外依赖
```

### 核心代码设计

#### 1. 3D训练适配器
```python
# flow_grpo/trainer_3d.py
class FlowGRPO3DTrainer:
    def __init__(self):
        # 加载Hunyuan3D模型
        from hunyuan3d.pipeline import Hunyuan3DPipeline
        self.model = Hunyuan3DPipeline.from_pretrained("...")
        
        # 使用原有的GRPO训练逻辑
        self.grpo_trainer = FlowGRPOTrainer(...)
        
        # 添加可视化器
        from utils.render_utils import simple_render_mesh
        self.render_fn = simple_render_mesh
    
    def train_step(self, batch):
        images, target_meshes = batch
        generated_meshes = self.model(images)
        rewards = compute_mesh_quality(generated_meshes, target_meshes)
        
        # 每100步保存一次可视化
        if self.step % 100 == 0:
            self.render_fn(generated_meshes[0], f"outputs/mesh_{self.step}.png")
        
        return self.grpo_trainer.update(generated_meshes, rewards)
```

#### 2. 3D奖励函数
```python
# flow_grpo/rewards_3d.py
def compute_mesh_quality(generated_meshes, target_meshes):
    """简单的mesh质量评估"""
    scores = []
    for gen_mesh, target_mesh in zip(generated_meshes, target_meshes):
        # 基础几何质量指标
        geometric_score = mesh_geometric_quality(gen_mesh)
        # 与目标的相似度
        similarity_score = mesh_similarity(gen_mesh, target_mesh)
        scores.append(geometric_score + similarity_score)
    return scores
```

#### 3. 简单的mesh处理
```python
# utils/mesh_utils.py
class SimpleMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices  # numpy数组
        self.faces = faces       # numpy数组
    
    def save_obj(self, path):
        """保存为OBJ文件"""
        pass
    
    @classmethod
    def from_hunyuan3d(cls, hunyuan_output):
        """从Hunyuan3D输出创建mesh"""
        pass
```

#### 4. 简单的mesh渲染
```python
# utils/render_utils.py
import trimesh
import matplotlib.pyplot as plt

def simple_render_mesh(mesh, save_path):
    """简单的mesh渲染 - 训练时可视化"""
    # 转换为trimesh格式
    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    else:
        trimesh_obj = mesh
    
    # 渲染4个视角
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    angles = [0, 90, 180, 270]
    
    for i, angle in enumerate(angles):
        # 旋转mesh
        rotated = trimesh_obj.copy()
        rotated.apply_transform(trimesh.transformations.rotation_matrix(
            angle * 3.14159 / 180, [0, 1, 0]))
        
        # 简单渲染
        axes[i].imshow(rotated.vertices[:, [0, 2]], cmap='viridis')
        axes[i].set_title(f'{angle}°')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 分阶段实现计划 🚀

### 第一步：集成Hunyuan3D并验证一致性
**目标**：确保Hunyuan3D模型能正常工作，输出与官方一致

#### **具体任务**：
1. **集成Hunyuan3D核心代码**
   - 复制`hy3dgen`模块到`hunyuan3d/`
   - 创建`hunyuan3d/pipeline.py`封装推理
   - 实现`utils/mesh_utils.py`处理输出mesh

2. **验证一致性**
   - 创建`scripts/test_hunyuan3d.py`对比官方输出
   - 用相同输入图像测试
   - 确保生成的mesh与官方完全一致

3. **基础可视化**
   - 实现`utils/render_utils.py`
   - 能渲染生成的mesh

4. **成功标准**
   - 能加载Hunyuan3D模型 ✅
   - 输出mesh与官方代码一致 ✅
   - 能保存.obj文件 ✅
   - 能生成可视化图像 ✅

### 第二步：集成reward代码
**目标**：实现3D质量评估，能给mesh打分

#### **具体任务**：
1. **实现奖励函数**
   - 创建`flow_grpo/rewards_3d.py`
   - 实现几何质量评估（面积、体积、曲率）
   - 实现mesh相似度计算

2. **奖励函数验证**
   - 创建`scripts/test_rewards.py`
   - 用好坏mesh样本验证奖励函数合理性
   - 确保奖励分数有区分度

3. **数据管道**
   - 实现`flow_grpo/datasets_3d.py`
   - 能加载图像-3D配对数据

4. **成功标准**
   - 奖励函数能给mesh打分 ✅
   - 好mesh比坏mesh分数高 ✅
   - 数据加载管道正常工作 ✅
   - 奖励计算速度可接受 ✅

### 第三步：适配GRPO训练
**目标**：将Hunyuan3D集成到GRPO训练框架

#### **具体任务**：
1. **训练适配器**
   - 创建`flow_grpo/trainer_3d.py`
   - 适配GRPO训练逻辑到3D生成
   - 实现梯度更新和参数优化

2. **端到端训练**
   - 创建`scripts/train.py`和`config/train_3d.py`
   - 实现完整的训练循环
   - 添加checkpoint保存/恢复

3. **训练验证**
   - 创建`scripts/test_integration.py`
   - 验证完整训练流程
   - 确保训练loss正常下降

4. **成功标准**
   - 训练流程不报错 ✅
   - 训练loss稳定下降 ✅
   - 生成mesh质量有改善 ✅
   - 完整训练循环正常工作 ✅

---

## 📋 修正后的文件优先级

### 第一步重点文件
1. `hunyuan3d/pipeline.py` - 核心推理封装
2. `utils/mesh_utils.py` - mesh处理工具
3. `utils/render_utils.py` - 可视化工具
4. `scripts/test_hunyuan3d.py` - 一致性验证

### 第二步重点文件
1. `flow_grpo/rewards_3d.py` - 奖励函数
2. `flow_grpo/datasets_3d.py` - 数据加载
3. `scripts/test_rewards.py` - 奖励函数验证

### 第三步重点文件
1. `flow_grpo/trainer_3d.py` - 训练适配器
2. `scripts/train.py` - 训练脚本
3. `config/train_3d.py` - 训练配置
4. `scripts/test_integration.py` - 端到端测试

---

## 🎯 这样划分的优势

1. **渐进式验证**：每一步都有明确的验证标准
2. **风险隔离**：问题更容易定位（是模型问题、奖励问题还是训练问题）
3. **并行开发**：后续步骤可以在前面基础上并行开发
4. **更现实**：避免一次性集成太多模块导致调试困难

---

## 具体开发任务

### 必须完成的文件
1. `hunyuan3d/pipeline.py` - Hunyuan3D推理封装
2. `flow_grpo/trainer_3d.py` - 3D训练适配器
3. `flow_grpo/rewards_3d.py` - 3D奖励函数
4. `flow_grpo/datasets_3d.py` - 3D数据加载
5. `utils/mesh_utils.py` - 基础mesh处理
6. `utils/render_utils.py` - 训练时mesh可视化
7. `config/train_3d.py` - 训练配置
8. `scripts/test_hunyuan3d.py` - 一致性验证
9. `scripts/test_rewards.py` - 奖励函数验证
10. `scripts/test_integration.py` - 端到端测试
11. `scripts/train.py` - 训练脚本

### 依赖安装
```bash
# requirements_3d.txt
trimesh>=4.0.0
matplotlib>=3.5.0
scipy>=1.9.0
torch>=2.0.0
# 其他Hunyuan3D依赖
```

### 环境搭建
```bash
# 从本地参考代码复制Hunyuan3D核心模块
cp -r _reference_codes/Hunyuan3D-2.1/hy3dgen ./hunyuan3d/
# 安装依赖
pip install -r requirements_3d.txt
```

---

## 注意事项

### 与现有框架的区别
- **数据类型**：图像 → 3D网格（而非文本 → 图像）
- **奖励函数**：3D几何质量（而非图像质量）
- **输出格式**：3D mesh文件（而非图像文件）

### 硬件要求
- **GPU**: 16GB+ VRAM
- **内存**: 32GB+ RAM
- **存储**: 10GB+

### 快速验证
```bash
# 一键测试
python scripts/test_integration.py

# 开始训练
python scripts/train.py --config config/train_3d.py
```

---

## 成功标准

**第一周结束时应该能够**：
- 加载Hunyuan3D模型 ✅
- 处理一个图像-3D配对 ✅  
- 计算奖励分数 ✅
- 完成一次训练更新 ✅
- 生成一个3D mesh文件 ✅

**如果以上都能做到，项目就算成功了！**后续的优化都是锦上添花。
