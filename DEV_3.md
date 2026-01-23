# 可微 Voxel Normal 渲染：12-Quad 两级加权方案

## 1. 背景与动机

### 1.1 DEV_2 方案回顾
- 当前 FDG：3 轴简化，每轴 8 邻居 → 三角形均值 → intersected_logits 加权
- 问题1：没有区分同一轴内的 4 个 quad
- 问题2：只有最高分辨率 sub_logit 有梯度，低分辨率无法更新

### 1.2 改进目标
- 精确建模 12 个 quad 的几何关系
- 多分辨率梯度传播
- 法线连续性（避免抖动）

## 2. 几何分析

### 2.1 12 个 Quad 与 26 邻居
- 6 面邻居：每个参与 4 个 quad
- 12 边邻居：每个参与 1 个 quad
- 8 角邻居：不参与

### 2.2 12 条边的 Corner 对应

sub_logit 的 8 个 corner 编码：`(x, y, z) → idx = x + 2*y + 4*z`

| 轴 | 边索引 | Corner 对 |
|----|--------|-----------|
| X | 0-3 | (0,1), (2,3), (4,5), (6,7) |
| Y | 4-7 | (0,2), (1,3), (4,6), (5,7) |
| Z | 8-11 | (0,4), (1,5), (2,6), (3,7) |

### 2.3 邻居不存在的处理
- 邻居不在 `h` 中 → quad 无效（weight=0）
- 中心 voxel 的 `sub_logit` 一定存在（来自 `subs[-1]` 的 parent）
- 若 `intersected_logits` 不存在，则邻居的 `dual_vertices` 也不存在，无法计算 quad 法线

## 3. 两级加权算法

### 3.1 层级1：Crossing Weight（sub_logit）

crossing 边 = 两端 occupancy 不同 = 有表面穿过

```python
# 可微的 crossing probability
occ = sigmoid(sub_logit)  # (N, 8)
occ_i, occ_j = occ[:, edge_pairs[:, 0]], occ[:, edge_pairs[:, 1]]
p_cross = occ_i * (1 - occ_j) + occ_j * (1 - occ_i)  # (N, 12)
```

### 3.2 层级2：Axis Weight（intersected_logits）

```python
axis_weights = sigmoid(intersected_logits)  # (N, 3)
```

### 3.3 一致性翻转

由于同一 voxel 的不同 quad 法线可能指向相反方向，直接加权会导致相消，产生抖动和不连续。

**问题示例**：
```
quad1: n1 = [0.9, 0.3, 0]   ↗
quad2: n2 = [0.8, 0.4, 0.1] ↗
quad3: n3 = [-0.7, -0.5, 0] ↙  ← 方向相反！
quad4: n4 = [0.85, 0.35, 0] ↗

直接加权 → 相消 → 结果不稳定
```

**解决方案：一致性翻转**

1. 选择权重最大的法线作为参考方向
2. 其他法线与参考点积 < 0 时翻转
3. 翻转后再加权求和 + 归一化

```python
def consistent_weighted_normal(
    normals: Tensor,    # (N, K, 3) — K 个法线候选
    weights: Tensor,    # (N, K) — 权重
    eps: float = 1e-6,
) -> Tensor:
    """一致性翻转 + 加权求和"""
    N, K, _ = normals.shape
    device = normals.device
    
    # 1. 选择参考方向：权重最大的法线
    max_idx = weights.argmax(dim=-1)  # (N,)
    ref = normals[torch.arange(N, device=device), max_idx]  # (N, 3)
    
    # 2. 计算每个法线与参考的点积
    dots = (normals * ref.unsqueeze(1)).sum(dim=-1)  # (N, K)
    
    # 3. 翻转反向的法线（点积 < 0）
    flip_mask = (dots < 0).unsqueeze(-1)  # (N, K, 1)
    aligned = torch.where(flip_mask, -normals, normals)  # (N, K, 3)
    
    # 4. 加权求和 + 归一化
    weighted = (weights.unsqueeze(-1) * aligned).sum(dim=1)  # (N, 3)
    result = F.normalize(weighted, dim=-1, eps=eps)  # (N, 3)
    
    return result
```

### 3.3.1 梯度分析

**Q: `argmax` 和硬阈值翻转不可微，是否影响训练？**

**A: 影响有限**。分析梯度流：

aligned = torch.where(flip_mask, -normals, normals)
weighted = (weights.unsqueeze(-1) * aligned).sum(dim=1)

# 反向传播：
# ∂L/∂normals = ±weights * ∂L/∂weighted  ✅ 梯度可传递
# ∂L/∂weights = aligned · ∂L/∂weighted   ✅ 梯度可传递


### 3.4 完整融合流程

```python
# 层级1：每个轴方向（4 个 edge → 1 个 axis_normal）
axis_normals = []
for axis in range(3):
    edge_ids = (EDGE_TO_AXIS == axis).nonzero()[0]  # (4,)
    normals_ax = quad_normals[:, edge_ids]    # (N, 4, 3)
    weights_ax = crossing_weights[:, edge_ids]  # (N, 4)
    
    axis_normal = consistent_weighted_normal(
        normals_ax, weights_ax
    )  # (N, 3)
    axis_normals.append(axis_normal)

axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)

# 层级2：3 个方向 → 最终法线
axis_weights = torch.sigmoid(intersected_logits)  # (N, 3)
final_normal = consistent_weighted_normal(
    axis_normals,   # (N, 3, 3)
    axis_weights    # (N, 3)
)  # (N, 3)
```

## 4. 多分辨率融合

### 4.1 问题：低分辨率无梯度
- 原方案只用 `subs[-1]`（最高分辨率 parent）
- 低分辨率层（控制大形状）无法通过 loss 更新
- 大形状变化缓慢

### 4.2 解决方案：软与融合

让 crossing weight 由多层 sub_logit 联合决定：

```python
def compute_crossing_weight_soft_and(
    subs: List[SparseTensor],
    center_coords: Tensor,    # (N, 3)
    voxel_resolution: int,
    temperature: float = 2.0,
    min_prob: float = 0.01,
) -> Tensor:
    """
    软与融合：各层投票，对数域平均
    
    公式：P = exp(mean(log(p_i)) / T)
    """
    log_probs = []
    
    for level, sub in enumerate(subs):
        level_resolution = voxel_resolution // (2 ** (len(subs) - level))
        p_cross = _crossing_one_level(sub, center_coords, level_resolution)
        p_cross = p_cross.clamp(min=min_prob, max=1.0 - min_prob)
        log_probs.append(torch.log(p_cross))
    
    # 软与：对数域平均 + 温度缩放
    mean_log = torch.stack(log_probs, dim=0).mean(dim=0)  # (N, 12)
    soft_and_prob = torch.exp(mean_log / temperature)     # (N, 12)
    
    return soft_and_prob
```

### 4.3 温度参数

| T 值 | 效果 | 梯度特性 |
|------|------|---------|
| T = 1 | 硬与（纯乘法） | 梯度易消失 |
| **T = 2~4** | **软与（推荐）** | **梯度稳定** |
| T → ∞ | 算术平均 | 失去"与"语义 |

### 4.4 梯度稳定性对比

**链式乘法的问题**：

假设各层 p = [0.1, 0.8, 0.9, 0.7]：
- ∂P/∂p₁ = 0.8×0.9×0.7 = 0.504（最大）
- ∂P/∂p₂ = 0.1×0.9×0.7 = 0.063（被 p₁ 压制）

最小值层"主导"梯度，其他层梯度消失。

**软与的优势**：
- 对数域平均 + 温度缩放
- 避免极端值主导
- 所有层都有合理梯度

### 4.5 数值稳定性分析

**Q: `log(p_cross)` 是否会数值爆炸？**

**A: 不会**。`p_cross` 被 clamp 到 `[min_prob, 1-min_prob]`：

| min_prob | log 范围 | exp(log/T) 范围 (T=2) |
|----------|---------|----------------------|
| 0.01 | [-4.6, -0.01] | [0.1, 1.0] |

数值完全可控，`min_prob=0.01` 保护足够。

## 5. 实现架构

### 5.1 Quad12NormalRenderer 类结构

```
render()
├── _extract_fdg_data()        # 提取 coords/dual_vertices/logits
├── _get_visible_voxels()      # 硬渲染 + 可见性筛选
├── _compute_visible_normals() # 核心计算
│   ├── compute_crossing_weight_soft_and()  # 多分辨率融合
│   ├── _compute_quad_normals_for_visible() # 12-Quad 几何
│   ├── aggregate_to_final_normal()         # 两级聚合
│   └── _flip_normals_to_camera()           # 相机空间变换
└── _sample_to_pixels()        # 索引采样
```

### 5.2 Gradient Checkpointing

```python
p_cross = checkpoint(
    _crossing_one_level,
    sub.feats.float(),
    sub.coords,
    center_coords,
    edge_pairs,
    use_reentrant=False,
)
```

- 前向不保存中间激活，反向重算
- 显存开销 ≈ 原方案
- 计算时间 ×2

### 5.3 常量定义

```python
# 12 边的 corner 索引对
EDGE_CORNER_PAIRS = torch.tensor([
    [0, 1], [2, 3], [4, 5], [6, 7],  # X 轴
    [0, 2], [1, 3], [4, 6], [5, 7],  # Y 轴
    [0, 4], [1, 5], [2, 6], [3, 7],  # Z 轴
])  # (12, 2)

# 边到轴的映射
EDGE_TO_AXIS = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2])  # (12,)

# 每边的 3 个邻居偏移 [面邻居1, 面邻居2, 边邻居]
EDGE_NEIGHBOR_OFFSETS = ...  # (12, 3, 3)
```

## 6. 梯度流向

```
Loss
  ↓
final_normal
  ├─→ intersected_logits (h.feats[3:6])    ✅ FDG 参数
  └─→ axis_normals
        ├─→ quad_normals ← dual_vertices   ✅ FDG 参数
        └─→ crossing_weights
              └─→ subs[0..n]               ✅ 所有分辨率层
```

**关键改进**：通过多层融合，低分辨率的 `sub_logits` 也能获得梯度，大形状变化更快。

## 7. 超参数建议

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `temperature` | 2.0 ~ 4.0 | 软与温度，越大越平滑 |
| `min_prob` | 0.01 | 防止 log(0) |
| `use_checkpoint` | True | 控制显存 |
| `voxel_margin` | 0.0 | dual_vertices 边距 |

## 8. 使用示例

```python
from edit4shape.renderers.diff_voxel_normal_quad12 import (
    render_normal_12quad, RenderConfig
)

# 单视角渲染
config = RenderConfig(
    extrinsic=cameras.w2c[0, v],
    intrinsic=cameras.intrinsic[0, v],
    resolution=512,
)
normal, mask = render_normal_12quad(
    h[0], subs, config,
    voxel_margin=decoder.voxel_margin,
    temperature=2.0,
)

# 多视角渲染
normals = []
for v in range(num_views):
    config = RenderConfig(
        extrinsic=cameras.w2c[0, v],
        intrinsic=cameras.intrinsic[0, v],
        resolution=render_resolution,
    )
    normal, mask = render_normal_12quad(h[0], subs, config, ...)
    normals.append(normal)
normals = torch.stack(normals, dim=0)  # (V, H, W, 3)
```

## 9. 文件位置

- 实现：`edit4shape/renderers/diff_voxel_normal_quad12.py`
- 集成：`edit4shape/systems/trellis2_shape.py` 中调用 `render_normal_12quad`
