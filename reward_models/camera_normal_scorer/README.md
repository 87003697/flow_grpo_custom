## Camera Normal Scorer 使用说明

该模块提供一种基于“相机搜索 + 法线相似度”的 3D 网格打分方式：
- 使用参考渲染器为每个网格生成固定 support 视角；
- 使用 VGGT Camera-Search 估计与输入图像对齐的 query 视角相机；
- 按估计相机渲染法线图，与图像侧法线特征做余弦相似度，得到 [0,1] 分数。

### 目录结构（关键文件）
- `scorer.py`: 主入口，`CameraNormalScorer` 实现完整打分流程
- `config.py`: `ScorerConfig` 配置定义
- `camera/`: 相机搜索封装与 support 构建
  - `vggt_estimator.py`: VGGT 相机估计器封装
  - `support.py`: 固定视角加载、support 批构建
  - `estimate_utils.py`: 相机分批估计工具（含内参归一化）
- `normal_io/`: 法线 I/O
  - `stable_normal_predictor.py`: 通过 torch.hub 的 StableNormal 预测器工厂
- `encoders/dino_encoder.py`: 法线特征编码（DINO）
- `render/`: 渲染适配与法线渲染
  - `adapter.py`: mesh 适配工具（含 `KiuiMeshLike`）
  - `render_normals.py`: 参考渲染器批量渲染法线
- `vis/save.py`: 可视化输出

## 环境依赖
- Python 3.10+
- PyTorch 与 torchvision（匹配本机 CUDA）
- Transformers（用于 DINO）
- nvdiffrast、kiui（参考渲染器）
- safetensors、huggingface-hub、Pillow、numpy

示例安装（请按实际 CUDA/PyTorch 版本调整）:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors huggingface-hub pillow numpy
pip install nvdiffrast kiui
```

## 参考代码与权重准备
- 将参考工程放置于项目根目录的 `_reference_codes/VGGTObj`，本模块已在运行时注入其路径。
- 准备 VGGT Camera-Search 权重：将 `model.safetensors` 放在某个目录，配置项 `camera_ckpt` 指向该目录或文件。
- 准备 DINO 权重目录（或 HF 模型名）：`dino_v2_path`、`dino_v3_path`。
- 若使用 `query_input=normal_pred`，需在 `normal_weights_dir` 下缓存 `yoso-normal-v1-8-1` 等版本权重（或允许联网自动下载）。

## 配置说明（ScorerConfig）
- **resolution**: 评分渲染分辨率 R（正方形）
- **cache_dir**: 图像侧法线缓存目录（已提前生成的法线 PNG）
- **encoder**: `dino_v2` | `dino_v3`
- **dino_v2_path / dino_v3_path**: DINO 权重路径或模型名
- **save_vis / vis_dir**: 是否保存可视化及输出目录
- **cam_batch_size / render_batch_size / dino_batch_size**: 三个阶段的批量大小
- **camera_config_py**: 固定多视角配置脚本（需含 `get_camera_search_seven_view_config()` 并提供 `predefined_poses`）
- **use_mesh_support**: 是否使用 mesh 生成 support（默认 True）
- **camera_param_dim**: VGGT 支持 9 维姿态编码或 12 维展平外参
- **img_size**: VGGT 训练/推理的输入尺寸（默认 518）
- **camera_ckpt**: VGGT Camera-Search checkpoint 目录或 `.safetensors` 路径
- **query_input**: `rgb` | `normal_pred` | `normal_image`
- **normal_weights_dir / normal_version**: 法线预测权重路径与版本（用于 `normal_pred`）

## 快速开始
### 方式一：直接脚本运行
仓库提供示例脚本 `scripts/eval_mesh_scorer_eval3d.py` 与 `scripts/debug/mesh_normal_scorer.sh`。根据你的数据路径与权重路径修改配置后执行：
```bash
python scripts/eval_mesh_scorer_eval3d.py \
  --cache_dir ./normal_cache \
  --resolution 256 \
  --camera_ckpt /path/to/vggt_camera_search_ckpt \
  --query_input rgb
```

脚本会读取 `FLOW_GRPO_DATA_DIR` 环境变量来定位图像：
```bash
export FLOW_GRPO_DATA_DIR=dataset/eval3d
```

### 方式二：在代码中调用
```python
import torch
from reward_models.camera_normal_scorer import CameraNormalScorer

cfg = {
  'resolution': 256,
  'cache_dir': './normal_cache',
  'encoder': 'dino_v2',
  'dino_v2_path': 'pretrained_weights/dinov2-base',
  'camera_config_py': '_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py',
  'camera_ckpt': '/path/to/vggt_camera_search_ckpt',
  'img_size': 518,
  'query_input': 'rgb',  # or 'normal_pred' | 'normal_image'
  'save_vis': True,
  'vis_dir': 'logs/dino_vis',
}

scorer = CameraNormalScorer(torch.device('cuda'), cfg)

# meshes: List[Any]，可包含 .v/.f 或 .vertices/.faces；
# images: List[PIL.Image]（当前未直接使用）；
# metadata: List[Dict]，每项必须提供 'image_path' 或 'image_name'；
meshes = [...]
images = [...]
metadata = [ {'image_path': '/abs/path/to/img.png'}, ... ]

scores = scorer.compute_scores(meshes, images, metadata, renderer=None)
print(scores)  # List[float]，范围约 [0,1]
```

## 法线缓存与可视化
- 图像侧法线由 `normal_io/cache.py` 的 `load_normal_from_cache` 读取，缓存命名为 `cache_dir/R{R}/{stem}.png`（值域 [-1,1] 映射到 PNG）。
- 若 `save_vis=True`，将输出两张图到 `vis_dir`：
  - `pred_normal_{tag}.png`：图像侧法线
  - `render_normal_{tag}.png`：按估计相机渲染的法线

## 关于 query_input
- **rgb**: 直接用 RGB 做相机搜索的 query 输入
- **normal_pred**: 先通过本地 Normal Predictor 将 RGB→Normal，再作为 query
- **normal_image**: 已有法线图作为 query（RGB 形式读入）

## 常见问题
- 依赖导入警告（linter 无法解析 torch/torchvision/transformers 等）
  - 在安装相应包后即可消失，不影响运行。
- 找不到 `_reference_codes/VGGTObj` 模块
  - 确保该目录存在且结构完整；本模块已在运行时向 `sys.path` 注入该目录。
- 找不到 VGGT 权重
  - 设置 `camera_ckpt` 为包含 `model.safetensors` 的目录或文件；若键名不完全匹配将宽松加载。
- 估计相机后渲染异常
  - 模块内部使用 OpenCV W2C → OpenGL C2W 的转换，再调用参考渲染器渲染；请确认你的网格顶点/面索引有效，且没有单位/尺度异常。

## 开发者提示
- 所有 import 与类定义均位于模块顶端，方便静态分析与依赖审查。
- 关键形状在代码中以注释标注，便于快速排查 tensor 对齐问题。


