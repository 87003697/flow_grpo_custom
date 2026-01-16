# edit4shape/guidance 重构计划

## 一、背景与目标

当前 guidance 模块仅支持 FlowEdit 范式（编辑图像 → 计算相似度 loss）。

目标：
- 支持多种 Guidance 范式：FlowEdit、SDS、CSD、VSD
- 支持多种基础模型：Qwen-Image-Edit、Flux Kontext
- 统一接口，通过配置切换

## 二、当前架构问题

1. **范式单一**：只有 FlowEdit，无法支持 Score Distillation 类方法
2. **返回不一致**：FlowEdit 返回 edited_imgs + 多个 loss；SDS/CSD 只返回 loss
3. **耦合严重**：FlowEditGuidance 同时处理 Pipeline 调用和 Loss 计算

## 三、新架构设计

### 3.1 目录结构

```
edit4shape/guidance/
├── __init__.py              # create_guidance() 工厂
├── base.py                  # GuidanceResult + SpecifyGradient
├── pipeline_parallel.py     # 流水线并行 Mixin（通用）
├── utils.py
│
├── pipelines/               # diffusers Pipeline 子类
│   ├── __init__.py
│   ├── adapters.py          # Pipeline 适配器
│   ├── qwen_image_edit/     # Qwen-Image-Edit 系列
│   │   ├── flowedit_simple.py
│   │   ├── flowedit_full.py
│   │   ├── state_tracker.py
│   │   ├── csd.py           # 待实现
│   │   └── sds.py           # 待实现
│   └── flux/                # Flux Kontext 系列（待扩展）
│       └── ...
│
├── paradigms/               # Guidance 实现
│   ├── flowedit.py          # FlowEditGuidance + FlowEditGuidancePP
│   └── distillation.py      # SDS/CSD/VSD Guidance（待实现）
│
└── metric/                  # 相似度 Metric（FlowEdit 用）
```

### 3.2 Pipeline 层

- 按基模分目录：`qwen/`、`flux/`
- 继承 diffusers 的 Pipeline（如 `QwenImageEditPlusPipeline`、`FluxPipeline`）
- 只重写 `__call__` 中的采样逻辑
- 复用父类的 `encode_prompt`、`prepare_latents`、`vae` 等

### 3.3 Paradigm 层

- `BaseGuidance`：抽象基类，定义 `compute_guidance()` 接口
- `FlowEditGuidance`：调用 FlowEdit Pipeline → 返回 edited_imgs + 相似度 loss
- `DistillationGuidance`：调用 CSD/SDS Pipeline → 返回梯度 loss（通过 SpecifyGradient 注入）

### 3.4 Metric 层

保持不变，供 FlowEditGuidance 使用。

### 3.5 流水线并行（PipelineParallelMixin）

通用 Mixin，任意 Guidance paradigm 可通过继承获得流水线并行能力：

```python
class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
    def __init__(self, ...):
        super().__init__(...)
        self._init_pipeline_parallel(num_streams=2)

class CSDGuidancePP(PipelineParallelMixin, CSDGuidance):
    def __init__(self, ...):
        super().__init__(...)
        self._init_pipeline_parallel(num_streams=2)
```

提供接口：
- `submit_async(comp_rgb, condition_images)`: 异步提交
- `wait_and_get() -> GuidanceResult`: 获取结果（FIFO）
- `has_pending() -> bool`: 检查待处理任务

### 3.6 统一返回格式 GuidanceResult

```python
@dataclass
class GuidanceResult:
    loss: torch.Tensor                              # 必须（可直接 backward）
    edited_imgs: Optional[torch.Tensor] = None      # FlowEdit 专用
    loss_dict: Optional[Dict[str, torch.Tensor]] = None  # 细分 loss
```

## 四、支持的组合矩阵

| 范式 \ 基模 | Qwen-Image-Edit | Flux Kontext |
|------------|-----------------|--------------|
| FlowEdit   | ✅ 已有         | 🔲 待扩展    |
| SDS        | 🔲 待实现       | 🔲 待扩展    |
| CSD        | 🔲 待实现       | 🔲 待扩展    |
| VSD        | ❌ 暂不支持     | 🔲 待扩展    |

说明：
- ✅ 已有：当前代码已支持
- 🔲 待实现/扩展：本次重构目标
- ❌ 暂不支持：需要 LoRA 学生，复杂度较高

## 五、配置接口设计

```python
guidance = dict(
    type="flowedit",                  # flowedit | csd | sds
    base_model="qwen",                # qwen | flux
    model_path="Qwen/Qwen-Image-Edit-2509",
    
    # FlowEdit 参数
    flowedit=dict(
        pipeline_type="simple",       # simple | full
        n_max=20,
        true_cfg_scale=5.5,
    ),
    
    # Distillation 参数
    distillation=dict(
        # 基础参数
        target_prompt="a 3D model",
        negative_prompt="",           # Qwen: uncond prompt
        
        # 时间步采样
        min_step_percent=0.02,
        max_step_percent=0.98,
        
        # 权重策略
        weight_type="ada",            # t | ada | uniform
        weight_eps=0.0,
        
        # CFG
        cfg_scale=5.5,                # Qwen: true_cfg_scale, Flux: guidance_scale
        
        # 噪声控制
        fixed_noise=False,
    ),
)
```

## 六、迁移计划

### Phase 1: 重组目录
- 创建 `pipelines/qwen/` 和 `pipelines/flux/` 目录
- 移动现有 FlowEdit Pipeline 到 `pipelines/qwen/`
- 更新 import 路径

### Phase 2: 实现 CSD Pipeline
- 基于 `flowedit_simple.py` 创建 `csd.py`
- 修改采样逻辑：source/target → uncond/cond
- 返回 grad 而非 edited_imgs

### Phase 3: 重构 Paradigm 层
- 创建 `paradigms/distillation.py`
- 实现 `DistillationGuidance` 类
- 集成 `SpecifyGradient` 梯度注入

### Phase 4: 测试验证
- FlowEdit 功能回归测试
- CSD 训练效果验证

## 七、待定问题