### Camera Normal Scorer（精简版）

基于相机搜索与法线特征相似度的 3D 网格打分，输出约 [0,1]。

- **核心流程**:
  - 固定多视角 support 渲染
  - VGGT Camera-Search 估计与图像对齐的 query 相机
  - 按估计相机渲染法线，与图像侧法线做特征余弦相似度（DINO），聚合为得分
- **主入口**: `reward_models/camera_normal_scorer/scorer.py` 的 `CameraNormalScorer`

### 依赖与准备
- Python 3.10+，PyTorch/torchvision（匹配 CUDA），Transformers，nvdiffrast，kiui，safetensors，Pillow，numpy
- 参考代码目录：`_reference_codes/VGGTObj`（运行时自动注入 `sys.path`）
- 相机搜索权重：`camera_ckpt` 指向目录或 `model.safetensors`
- DINO 权重：`dino_v2_path` 或 `dino_v3_path`（亦可传 HF 模型名）
- 若使用 `query_input=normal_pred`：准备 `normal_weights_dir`（本地 StableNormal 权重目录）

示例安装（按需调整 CUDA/PyTorch 版本）:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors huggingface-hub pillow numpy nvdiffrast kiui
```

### 快速上手（代码调用）
```python
import torch
from reward_models.camera_normal_scorer import CameraNormalScorer

cfg = {
  'resolution': 256,
  'cache_dir': './normal_cache',
  'encoder': 'dino_v2',
  'dino_v2_path': 'pretrained_weights/dinov2-giant',
  'camera_config_py': 'training.config.camera_search_seven_view_fixed:get_camera_search_seven_view_config',
  'camera_ckpt': '/path/to/vggt_camera_search_ckpt',
  'img_size': 518,
  'query_input': 'rgb',  # 'rgb' | 'normal_pred' | 'normal_image'
  'save_vis': True,
  'vis_dir': 'logs/dino_vis',
}

scorer = CameraNormalScorer(torch.device('cuda'), cfg)

meshes = [...]    # List[Any]，支持 .v/.f 或 .vertices/.faces
images = [...]    # List[PIL.Image]
metadata = [ {'image_path': '/abs/path/to/img.png'}, ... ]

scores = scorer.compute_scores(meshes, images, metadata, renderer=None)
print(scores)  # List[float]
```

### 关键配置（常用）
- **resolution**: 渲染分辨率 R（正方形）
- **encoder**: `dino_v2` | `dino_v3`; 对应 `dino_v2_path`/`dino_v3_path`
- **camera_config_py**: `module.path:function_name`，将被动态导入并读取 `render.predefined_poses`
- **camera_ckpt**: VGGT checkpoint（目录或 `.safetensors`）
- **img_size**: VGGT 训练/推理尺寸（默认 518）
- **camera_param_dim**: 9（姿态编码）或 12（展平外参）
- **cam_batch_size / render_batch_size / dino_batch_size**: 三阶段批大小
- **query_input**: `rgb` | `normal_pred` | `normal_image`

### 可视化与缓存
- 法线缓存：`cache_dir/R{R}/{stem}.png`（[-1,1] 映射 PNG）
- 若 `save_vis=True`，输出至 `vis_dir`：
  - `pred_normal_{tag}.png`（图像侧法线）
  - `render_normal_{tag}.png`（估计相机渲染法线）

### 常见问题
- 找不到 `_reference_codes/VGGTObj`：确认目录存在且结构完整
- 未找到 VGGT 权重：`camera_ckpt` 指向包含 `model.safetensors` 的目录或文件
- 渲染异常：检查网格顶点/面有效性与尺度

### 目录速览（关键）
- `scorer.py`: `CameraNormalScorer` 主流程
- `camera/`: 相机估计与 support 构建（`vggt_estimator.py`, `support.py`, `estimate_utils.py`）
- `encoders/dino_encoder.py`: 法线特征编码
- `normal_io/stable_normal_predictor.py`: 法线预测（可选）
- `render/`: 网格适配与法线渲染

### 预训练权重下载与准备
- **VGGT Camera-Search**（`camera_ckpt`）
  - 需要一个包含 `model.safetensors` 的目录。
  - 若已获得内部/私有权重，直接放置：`/path/to/vggt_camera_search_ckpt/model.safetensors`，并将配置项 `camera_ckpt` 指向该目录或文件。
  - 如使用自有 HF 仓库存储（示例，需替换占位）：
    ```bash
    huggingface-cli download <your-org>/<your-vggt-camera-search-repo> model.safetensors \
      --local-dir pretrained_weights/vggt_camera_search
    ```

- **DINO（法线特征编码）**（`dino_v2_path` / `dino_v3_path`）
  - 推荐直接填写 HF 模型名，运行时自动下载：如 `facebook/dinov2-giant`。
  - 也可预下载至本地，离线使用：
    ```bash
    # DINOv2（示例：giant 版本）
    huggingface-cli download facebook/dinov2-giant --local-dir pretrained_weights/dinov2-giant
    # DINOv3（如需，替换为你的模型 ID）
    huggingface-cli download <your-dino-v3-id> --local-dir pretrained_weights/dinov3
    ```
  - 配置示例：
    ```python
    cfg.update({
      'encoder': 'dino_v2',
      'dino_v2_path': 'pretrained_weights/dinov2-giant',  # 或 'facebook/dinov2-giant'
    })
    ```

- **StableNormal（可选：图像→法线预测）**（`normal_weights_dir`, `normal_version`）
  - 代码通过 `torch.hub` 使用 `hugoycj/StableNormal` 的 `StableNormal_turbo`，参数 `yoso_version` 取自目录名。
  - 在线环境：首次调用会自动下载缓存到 `local_cache_dir`（目录父级）。
  - 离线环境：准备完整的 diffusers 权重目录，例如 `pretrained_weights/normal/yoso-normal-v1-8-1/`，目录名需与版本一致（如 `yoso-normal-v1-8-1`）。
  - 配置示例：
    ```python
    cfg.update({
      'query_input': 'normal_pred',
      'normal_weights_dir': 'pretrained_weights/normal/yoso-normal-v1-8-1',
      'normal_version': 'yoso-normal-v1-8-1',
    })
    ```

提示：如使用 HF，请先登录以下载私有模型：`huggingface-cli login`。
