### Camera RGB Scorer（基于相机搜索与 RGB 特征相似度）

基于相机搜索与 RGB 特征相似度的 3D 网格打分，输出约 [0,1]。

- **核心流程**:
  - 使用法线进行 VGGT Camera-Search 估计最优相机姿态（几何对齐）
  - 按估计相机渲染 RGB 图像（而非法线）
  - 与输入的原始 RGB 图像做 DINO 特征余弦相似度，聚合为得分
- **主入口**: `reward_models/camera_rgb_scorer/scorer.py` 的 `CameraRGBScorer`

### 与 camera_normal_scorer 的区别

| 维度 | camera_normal_scorer | camera_rgb_scorer |
|------|---------------------|-------------------|
| **相机搜索输入** | 法线图 (normal_pil) | 法线图 (normal_pil) ✅ 相同 |
| **渲染目标** | 法线 [-1,1] | **RGB [0,1]** ❌ 不同 |
| **图像侧输入** | 法线图 | **原始 RGB 图像** ❌ 不同 |
| **DINO 编码** | DINO(法线) | **DINO(RGB)** ❌ 不同 |
| **评估重点** | 几何结构一致性 | **外观与纹理一致性** ❌ 不同 |

**核心思想**：用法线的几何信息找到最优相机视角，用 RGB 的外观信息评估渲染质量。

### 依赖与准备

- Python 3.10+，PyTorch/torchvision（匹配 CUDA），Transformers，nvdiffrast，kiui，safetensors，Pillow，numpy
- 参考代码目录：`_reference_codes/VGGTObj`（运行时自动注入 `sys.path`）
- 相机搜索权重：`camera_ckpt` 指向目录或 `model.safetensors`
- DINO 权重：`dino_v2_path` 或 `dino_v3_path`（亦可传 HF 模型名）
- 需要 `metadata` 中提供 `normal_pil`（用于相机搜索）

示例安装（按需调整 CUDA/PyTorch 版本）:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors huggingface-hub pillow numpy nvdiffrast kiui
```

### 快速上手（代码调用）

```python
import torch
from PIL import Image
from reward_models.camera_rgb_scorer import CameraRGBScorer

cfg = {
  'rgb_resolution': 256,
  'cache_dir': './rgb_cache',
  'encoder': 'dino_v2',
  'dino_v2_path': 'pretrained_weights/dinov2-base',
  'camera_config_py': '_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py',
  'camera_ckpt': '/path/to/vggt_camera_search_ckpt',
  'img_size': 518,
  'save_vis': False,
  'vis_dir': 'logs/dino_vis_rgb',
}

scorer = CameraRGBScorer(torch.device('cuda'), cfg)

meshes = [...]    # List[Any]，支持 .v/.f 或 .vertices/.faces
images = [...]    # List[PIL.Image]，原始 RGB 图像
metadata = [ 
    {
        'image_path': '/abs/path/to/img.png',
        'normal_pil': normal_image,  # PIL.Image，用于相机搜索
    }, 
    ... 
]

scores, grouped_meta = scorer.compute_scores(meshes, images, metadata)
print(scores)  # List[float]
```

### 关键配置（常用）

- **rgb_resolution**: RGB 渲染分辨率 R（正方形）
- **encoder**: `dino_v2` | `dino_v3`; 对应 `dino_v2_path`/`dino_v3_path`
- **camera_config_py**: 固定多视角配置脚本（需提供 `predefined_poses`）
- **camera_ckpt**: VGGT checkpoint（目录或 `.safetensors`）
- **img_size**: VGGT 训练/推理尺寸（默认 518）
- **camera_param_dim**: 9（姿态编码）或 12（展平外参）
- **cam_batch_size / render_batch_size / dino_batch_size**: 三阶段批大小

### 输入要求

1. **meshes**: 3D 网格列表，支持 `.v/.f` 或 `.vertices/.faces` 属性
2. **images**: PIL.Image 列表，原始 RGB 图像（用于特征比较）
3. **metadata**: 字典列表，每个需包含：
   - `image_path` 或 `image_name`: 图像标识（用于分组）
   - `normal_pil`: PIL.Image，法线图（用于相机搜索）

### 代码复用

本模块直接 import 复用 `camera_normal_scorer` 的以下组件：
- `camera/vggt_estimator.py`: 相机估计器
- `camera/support.py`: 支持视图构建
- `camera/estimate_utils.py`: 相机估计工具
- `render/adapter.py`: Mesh 适配器

仅新增/修改：
- `render/render_rgb.py`: RGB 渲染（vs 法线渲染）
- `encoders/rgb_encoder.py`: RGB 特征编码（vs 法线特征编码）
- `scorer.py`: 主逻辑（修改渲染和编码部分）

### 目录速览

- `scorer.py`: `CameraRGBScorer` 主流程
- `config.py`: 配置类
- `encoders/rgb_encoder.py`: RGB 图像 DINO 特征编码
- `render/render_rgb.py`: RGB 渲染
- （其他模块直接 import 自 `camera_normal_scorer`）

### 常见问题

- 找不到 `_reference_codes/VGGTObj`：确认目录存在且结构完整
- 未找到 VGGT 权重：`camera_ckpt` 指向包含 `model.safetensors` 的目录或文件
- metadata 缺少 `normal_pil`：相机搜索需要法线信息，需提前预测或提供
- 渲染异常：检查网格顶点/面有效性与尺度

### 应用场景

- **图生3D 质量评估**：评估生成的 3D 模型与输入 RGB 图像的外观一致性
- **纹理质量评分**：关注渲染结果的颜色、纹理细节
- **多模态评分**：结合 `camera_normal_scorer`（几何）和 `camera_rgb_scorer`（外观）
- **强化学习奖励**：在 GRPO 等算法中作为外观质量奖励信号

### 组合使用示例

```python
# 同时使用法线和 RGB 评分
from reward_models.camera_normal_scorer import CameraNormalScorer
from reward_models.camera_rgb_scorer import CameraRGBScorer

normal_scorer = CameraNormalScorer(device, normal_cfg)
rgb_scorer = CameraRGBScorer(device, rgb_cfg)

normal_scores, _ = normal_scorer.compute_scores(meshes, images, metadata)
rgb_scores, _ = rgb_scorer.compute_scores(meshes, images, metadata)

# 组合得分：几何 + 外观
final_scores = [0.5 * n + 0.5 * r for n, r in zip(normal_scores, rgb_scores)]
```
