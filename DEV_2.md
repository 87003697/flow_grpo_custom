# 可微 Voxel Normal 渲染方案

## 1. 背景与动机

### 1.1 问题

当前 TRELLIS.2 的 VoxelRenderer 采用硬光栅化，渲染过程不可微：

| 模式 | 可微参数 | 问题 |
|------|----------|------|
| FDG | `dual_vertices`, `intersected_logits` | 无法通过渲染 loss 获得梯度 |
| Sub | `sub_logits` | 无法通过渲染 loss 获得梯度 |

当前使用 MeshRenderer 的问题：
- **显存占用大**：Mesh 生成 + nvdiffrast 需要存储大量中间变量
- **训练不稳定**：梯度链路长，Mesh 拓扑随 intersected 变化

### 1.2 核心思路

**关键洞察**：硬渲染只决定"哪个 voxel 被击中"（索引），索引操作对被索引的 tensor 是可微的。

```
voxel_normals[voxel_id]  ← voxel_id 无梯度，但 voxel_normals 有梯度
```

**方案：Per-Voxel Normal 预计算 + 索引**

1. **预计算 Per-Voxel Normal**：用可微参数计算每个 voxel 的 normal
2. **硬渲染**：获取每个像素击中的 voxel_id
3. **索引**：`pixel_normal = voxel_normals[voxel_id]`（可微）

两种模式的 normal 计算方式不同：
- **FDG**：邻居 dual_vertices → axis_normals → intersected 加权
- **Sub**：sub_logits → occupancy → 梯度场 → normal

### 1.3 适用范围

- **FDG 模式**（FlexiDualGrid）：用于 Mesh 重建
- **Sub 模式**（Subdivision）：用于 Occupancy 监督

---

## 2. 统一框架

### 2.1 两阶段流程

```
┌─────────────────────────────────────────────────────────────┐
│  阶段 1：Per-Voxel Normal 预计算（可微）                     │
│                                                             │
│  FDG: dual_vertices → axis_normals ─┐                       │
│       intersected_logits → weights ─┴→ voxel_normals        │
│                                                             │
│  Sub: sub_logits → occupancy → gradient → voxel_normals     │
└──────────────────────────┬──────────────────────────────────┘
                           │ voxel_normals (N, 3)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 2：硬渲染 + 索引                                       │
│  VoxelRenderer → voxel_id → pixel_normal = voxel_normals[id]│
│                              (索引操作对 voxel_normals 可微) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 3：World → Camera 变换 + 方向翻转                      │
│  normal_cam = normal_world @ R.T                            │
│  normal_cam = flip_to_camera(normal_cam)  # 确保朝向相机    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心公式

**索引操作的可微性**：
```python
# voxel_id 是整数索引（无梯度）
# 但 voxel_normals 是可微 tensor
pixel_normal = voxel_normals[voxel_id]  # 对 voxel_normals 可微！
```

**World → Camera 变换 + 方向翻转**：
```python
R = extrinsics[:3, :3]              # (3, 3) 旋转矩阵
t = extrinsics[:3, 3]               # (3,) 平移向量

# 变换到 Camera Space
voxel_normals_cam = voxel_normals_world @ R.T  # (N, 3) 法线变换
surface_pos_cam = surface_pos @ R.T + t        # (N, 3) 顶点位置变换

# 用点积判断翻转：确保法线朝向相机
# 如果 normal · pos > 0，说明法线指向远离相机的方向，需要翻转
dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N, 1)
voxel_normals_cam = torch.where(dot_product > 0, -voxel_normals_cam, voxel_normals_cam)  # (N, 3)
```

**为什么用点积判断**：
- 在 Camera Space 中，`surface_pos_cam` 是从相机原点指向表面的向量
- 如果 `normal · pos > 0`，说明法线和视线方向**同向**（法线指向远离相机）
- 我们希望可见面的法线指向相机，所以需要翻转

**参考代码** (`trellis2/renderers/mesh_renderer.py` 第 127-135 行)：
```python
# MeshRenderer 在 Camera Space 中计算并翻转法线
v0 = vertices_camera[0, mesh.faces[:, 0], :3]  # Camera Space 中的顶点位置
e0 = v1 - v0
e1 = v2 - v0
face_normal = torch.cross(e0, e1, dim=1)
face_normal = F.normalize(face_normal, dim=1)
# 用法线与顶点位置的点积判断翻转
face_normal = torch.where(torch.sum(face_normal * v0, dim=1, keepdim=True) > 0, face_normal, -face_normal)
```

### 2.3 模式差异总结

| | FDG 模式 | Sub 模式 |
|---|----------|----------|
| **Normal 来源** | 邻居 dual_vertices 构成的面 | occupancy 梯度场 |
| **加权/方向** | intersected_logits 加权 | 梯度方向即法线 |
| **梯度流向** | dual_vertices + intersected | sub_logits |
| **voxel_id 含义** | voxel 索引 | 子 voxel 全局索引 |
| **适用场景** | Mesh 重建 | Occupancy 监督 |

---

## 3. FDG 模式详解

### 3.1 Axis Face Normals 计算

每个 voxel 最多生成 3 个面（x/y/z 轴方向）。每个面由 **4 个相邻 voxel 的 dual_vertices** 决定：

**参考代码** (`o-voxel/o_voxel/convert/flexible_dual_grid.py`):
```python
# edge_neighbor_voxel_offset: 每个轴的 4 个邻居偏移
edge_neighbor_voxel_offset = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # axis=0: YZ 平面 → 法线沿 X 轴
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # axis=1: XZ 平面 → 法线沿 Y 轴
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # axis=2: XY 平面 → 法线沿 Z 轴
])  # (3, 4, 3)

# 使用方式：intersected_flag[:, axis] = True 时，使用 offset[axis] 的邻居生成面
edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + edge_neighbor_voxel_offset  # (N, 3, 4, 3)
connected_voxel = edge_neighbor_voxel[intersected_flag]  # (M, 4, 3)
```

**索引与平面对应关系**：

| axis 索引 | `intersected[axis]` | 邻居偏移特征 | 生成的面 | 面法线方向 |
|-----------|---------------------|--------------|----------|-----------|
| 0 | X 轴边相交 | x 坐标都是 0 | YZ 平面 | X 轴 |
| 1 | Y 轴边相交 | y 坐标都是 0 | XZ 平面 | Y 轴 |
| 2 | Z 轴边相交 | z 坐标都是 0 | XY 平面 | Z 轴 |

计算每个轴的 face normal：

```python
# surface_pos: 每个 voxel 的表面位置
surface_pos = (coords + dual_vertices) * voxel_size + origin  # (N, 3)

# 构建 coord → index 的哈希映射
coord_to_idx = _build_coord_hash(coords)  # 不可微，但只是索引

# 对每个轴，获取 4 个邻居 voxel 的 surface_pos
neighbor_coords = coords.unsqueeze(1) + offsets[axis]  # (N, 4, 3)
neighbor_idx = _lookup_hash(coord_to_idx, neighbor_coords)  # (N, 4), -1 表示不存在

# 检测邻居有效性：4 个邻居都存在
axis_valid = (neighbor_idx != -1).all(dim=1)  # (N,)

# 安全索引（无效位置用 0 替代，后续会被 mask 掉）
neighbor_pos = surface_pos[neighbor_idx.clamp(min=0)]  # (N, 4, 3)，可微！

# 4 个顶点 → face normal
v0, v1, v2, v3 = neighbor_pos.unbind(dim=1)
axis_normal = F.normalize(torch.cross(v1 - v0, v3 - v0), dim=-1)  # (N, 3)
```

结果：
- `axis_normals: (N, 3, 3)` — 每个 voxel 的 x/y/z 三个轴的 face normal
- `axis_valid_mask: (N, 3)` — 每个轴的邻居是否完整

**注**：边界 voxel 的邻居可能不存在，需要**显式检测**并用 `axis_valid_mask` 过滤。这与原始 `flexible_dual_grid_to_mesh` 的逻辑一致：只有当 4 个邻居都存在时，才生成该轴的面。

**参考代码** (`trellis2/models/sc_vaes/fdg_vae.py`):
```python
# h.feats: (N, 7) - FDG Decoder 输出
vertices = h.replace((1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin)
intersected_logits = h.replace(h.feats[..., 3:6])
quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
```

### 3.2 Intersected 加权

用 `intersected_logits` 作为各轴 normal 的混合权重，同时用 `axis_valid_mask` 过滤邻居缺失的轴：

```python
# 基础权重
weights = torch.sigmoid(intersected_logits)  # (N, 3)

# 邻居缺失的轴，权重强制为 0
effective_weights = weights * axis_valid_mask.float()  # (N, 3)

# 加权平均
weighted = (effective_weights.unsqueeze(-1) * axis_normals).sum(dim=1)  # (N, 3)
voxel_normals = F.normalize(weighted, dim=-1, eps=1e-6)  # (N, 3)
```

**`intersected` 的几何意义**（参考 `o-voxel/src/convert/flexible_dual_grid.cpp`）：

`intersected` 是一个 `bool3`，表示 **voxel 的边是否与 mesh 表面相交**：

| 通道 | 含义 | 生成的面 | 面法线方向 |
|------|------|----------|-----------|
| `intersected[0]` | X 轴方向的边与 mesh 相交 | YZ 平面的 quad | **X 轴方向** |
| `intersected[1]` | Y 轴方向的边与 mesh 相交 | XZ 平面的 quad | **Y 轴方向** |
| `intersected[2]` | Z 轴方向的边与 mesh 相交 | XY 平面的 quad | **Z 轴方向** |

**参考代码** (`o-voxel/src/convert/flexible_dual_grid.cpp` 第 439-456 行)：
```cpp
// face_from_dual_vertices: 根据 intersected 决定生成哪些面
// xy-plane (法线沿 Z 轴)
if (is_intersected[2] && neighbors_exist) {
    // 生成 XY 平面的 quad
}
// yz-plane (法线沿 X 轴)
if (is_intersected[0] && neighbors_exist) {
    // 生成 YZ 平面的 quad
}
// xz-plane (法线沿 Y 轴)
if (is_intersected[1] && neighbors_exist) {
    // 生成 XZ 平面的 quad
}
```

**加权的几何直觉**：
- `intersected[0]` 高 且 x 轴邻居完整 → YZ 平面存在 → X 轴方向 normal 权重大
- `intersected[1]` 高 且 y 轴邻居完整 → XZ 平面存在 → Y 轴方向 normal 权重大
- `intersected[2]` 高 且 z 轴邻居完整 → XY 平面存在 → Z 轴方向 normal 权重大

**优势**（相比原 tanh 方案）：
- 梯度自然流向 intersected_logits
- 不需要复杂的符号近似
- 多面 voxel 时自动融合多个方向
- **邻居缺失时自动过滤**，不会产生错误的 normal

### 3.3 梯度流

```
                    ┌─────────────────────────────────────────┐
dual_vertices ──→ surface_pos ──→ axis_normals ──┐            │
                                                 ├→ voxel_normals ──→ pixel_normal ──→ loss
intersected_logits ──→ sigmoid ──→ weights ──────┘            │
                    └─────────────────────────────────────────┘
```

两个参数都有梯度：
- `dual_vertices`：控制每个轴 face normal 的方向（几何细节）
- `intersected_logits`：控制各轴 normal 的混合权重（面选择）

---

## 4. 多分辨率 Sub 模式详解

TRELLIS.2 的 Decoder 有多层，每层都输出 `sub_logits`。多分辨率 Sub 模式让 normal loss 的梯度能够流向各层的 `sub_logits`，从而实现**大形状修改**。

### 4.1 Occupancy 梯度场

每个父 voxel 包含 8 个子 voxel（2×2×2）。把 `sub_logits` 看作离散的占用场，其**梯度方向**即为表面法线。

**子 voxel 索引布局**：
```
    z=0 面          z=1 面
   ┌───┬───┐      ┌───┬───┐
   │ 2 │ 3 │      │ 6 │ 7 │
   ├───┼───┤      ├───┼───┤
   │ 0 │ 1 │      │ 4 │ 5 │
   └───┴───┘      └───┴───┘
     y              y
     ↑              ↑
     └→ x           └→ x
```

**梯度计算**：
```python
occupancy = torch.sigmoid(sub_logits)  # (N, 8)

# x 方向梯度：右边 - 左边 (索引 1,3,5,7 vs 0,2,4,6)
grad_x = (occupancy[:, [1,3,5,7]] - occupancy[:, [0,2,4,6]]).mean(dim=1)  # (N,)

# y 方向梯度：上边 - 下边 (索引 2,3,6,7 vs 0,1,4,5)
grad_y = (occupancy[:, [2,3,6,7]] - occupancy[:, [0,1,4,5]]).mean(dim=1)  # (N,)

# z 方向梯度：前边 - 后边 (索引 4,5,6,7 vs 0,1,2,3)
grad_z = (occupancy[:, [4,5,6,7]] - occupancy[:, [0,1,2,3]]).mean(dim=1)  # (N,)

# 梯度向量
gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (N, 3)

# 法线 = 梯度反方向（从内部指向外部）
voxel_normals = -F.normalize(gradient, dim=-1)  # (N, 3)
```

**几何意义**：
- 占用概率高 → 物体内部
- 占用概率低 → 物体外部
- 梯度方向：从高到低 = 从内到外 = 表面法线

**注**：梯度为零时（完全在物体内部/外部），用 `F.normalize(..., eps=1e-6)` 自动处理，返回零向量。这些 voxel 通常不可见（被遮挡或在物体外），normal 值无所谓。

**参考代码** (`trellis2/models/sc_vaes/sparse_unet_vae.py`):
```python
# SparseResBlockUpsample3d - Decoder 上采样层
self.to_subdiv = sp.SparseLinear(channels, 8)  # 预测 8 个子 voxel logits
subdiv = self.to_subdiv(x)                      # (N, 8) sub_logits
subdiv_binarized = subdiv.replace(subdiv.feats > 0)
self.updown(h, subdiv_binarized)                # 2x 上采样 → 8 个子 voxel
```

### 4.2 多层渲染策略

**核心思路**：每层 Sub 都渲染父 voxel，用 occupancy 梯度作为 normal，resize 到统一分辨率后与 GT 比较。

```python
# 多分辨率 Sub 处理流程
sub_losses = []
for i, sub in enumerate(subs):
    # sub.feats: (N_parent_i, 8) 该层的 sub_logits
    # sub.coords: (N_parent_i, 4) 该层的父 voxel 坐标
    
    # 1. 计算 occupancy 梯度 → voxel_normals
    voxel_normals = compute_occupancy_gradient(sub.feats)  # (N_parent_i, 3)
    
    # 2. 渲染父 voxel（可以用较低的渲染分辨率）
    voxel_id = hard_render(sub.coords[:, 1:], voxel_size_i, ...)  # (H_i, W_i)
    mask = voxel_id >= 0
    
    # 3. 索引获取 normal（voxel_id 与 sub_logits 索引一一对应）
    pixel_normal = voxel_normals[voxel_id.clamp(min=0)]  # (H_i, W_i, 3)
    
    # 4. World → Camera + 翻转 + mask
    normal_cam = transform_and_flip(pixel_normal, surface_pos, extrinsics) * mask.unsqueeze(-1)
    
    # 5. Resize 到统一分辨率 + 重新归一化
    normal_resized = F.interpolate(normal_cam, size=(H, W), mode='bilinear')
    normal_resized = F.normalize(normal_resized, dim=-1, eps=1e-6)  # 插值后重新归一化
    
    # 6. 计算 loss
    sub_losses.append(image_loss(normal_resized, gt_normal))
```

**渲染父 voxel 的优点**：
- **简单**：不需要 child→parent 映射
- **高效**：低分辨率渲染，resize 到目标分辨率
- **直接对应**：voxel_id 与 sub_logits 索引一一对应

**几何近似**：
- 父 voxel 是立方体，渲染的是立方体表面
- 但赋予的 normal 是 sub_logits 的 occupancy 梯度
- 这是一个近似，但对于"大形状修改"足够——精确细节由 FDG 模式处理

### 4.3 梯度流

```
sub_logits → sigmoid → occupancy → 差分 → voxel_normals → render → resize → loss
```

每层独立计算，最终加权求和。每层的 `sub_logits` 都有梯度。

### 4.4 与 FDG 模式的配合

| 模式 | 可微参数 | 作用 | 修改范围 |
|------|---------|------|---------|
| **多分辨率 Sub** | sub_logits (每层) | 修改稀疏结构 | 大形状（哪里有表面） |
| **FDG** | dual_vertices + intersected_logits | 修改几何细节 | 细节（表面位置/朝向） |

**为什么需要两者配合**：
- 如果 Sub 模式提供的稀疏结构错误（某区域应该有 voxel 但没有），FDG 模式无法修复
- Sub 模式让梯度流向 `sub_logits`，可以"增加/删除" voxel
- FDG 模式让梯度流向 `dual_vertices`，可以"移动/调整"表面

**训练时的配合**：
```python
# 1. 多分辨率 Sub（大形状）
sub_loss = sum(sub_losses) * lambda_sub

# 2. FDG（几何细节）
fdg_normal = diff_normal_fdg(voxel_id, coords, dual_vertices, intersected_logits, ...)
fdg_loss = image_loss(fdg_normal, gt_normal) * lambda_fdg

# 3. 总 loss
total_loss = sub_loss + fdg_loss
```

---

## 5. 实现细节

### 5.1 硬渲染器修改

需要额外输出 voxel 索引：

| 模式 | 输出 | 说明 |
|------|------|------|
| FDG | `voxel_id: (H, W)` | 击中的 voxel 索引，-1 表示背景 |
| Sub | `voxel_id: (H, W)` | 击中的**父 voxel** 索引，-1 表示背景 |

**注**：Sub 模式直接渲染父 voxel 级别的稀疏结构，`voxel_id` 与 `sub_logits` 的索引一一对应。

#### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `o-voxel/src/rasterize/rasterize.cu` | `render` kernel 输出 `out_voxel_id`（复用已有的 `hit` 变量） |
| `o-voxel/src/rasterize/api.h` | 更新函数签名 |
| `o-voxel/o_voxel/rasterize.py` | 返回 `voxel_id` |

#### 注意事项

- CUDA kernel 中 `hit` 变量已记录击中的 voxel 索引，只需输出
- Sub 模式需要先展开子 voxel 列表再渲染
- **不使用 SSAA**：避免 voxel_id 降采样的复杂性

**参考代码** (`o-voxel/src/rasterize/rasterize.cu`):
```cpp
// render kernel
int hit = -1;           // 第 215 行：初始化为 -1（背景）
hit = collected_id[j];  // 第 253 行：记录击中的 voxel 索引
// 只需在最后输出 hit 到 out_voxel_id[pix_id]
```

### 5.2 FDG 模式输入输出

**输入**：

| 参数 | Shape | 说明 |
|------|-------|------|
| `voxel_id` | (H, W) | 硬渲染输出 |
| `coords` | (N, 3) | voxel 整数坐标（注：SparseTensor.coords 是 (N,4)，需 `[:, 1:]`） |
| `dual_vertices` | (N, 3) | 顶点偏移，**可微** |
| `intersected_logits` | (N, 3) | 边相交 logits，**可微** |
| `extrinsics` | (4, 4) | W2C 相机外参 |
| `voxel_size` | float | 体素尺寸 |
| `origin` | (3,) | 网格原点 |

**中间变量**：

| 变量 | Shape | 说明 |
|------|-------|------|
| `surface_pos` | (N, 3) | 表面位置 |
| `axis_normals` | (N, 3, 3) | 三轴 face normal |
| `axis_valid_mask` | (N, 3) | bool，每个轴的 4 个邻居是否都存在 |
| `voxel_normals` | (N, 3) | per-voxel 加权 normal |

**输出**：

| 参数 | Shape | 说明 |
|------|-------|------|
| `normal` | (H, W, 3) | 归一化法向量，**Camera Space**，朝向相机（Z < 0） |
| `mask` | (H, W) | bool，True = 前景 |

### 5.3 多分辨率 Sub 模式输入输出

**输入**：

| 参数 | Shape | 说明 |
|------|-------|------|
| `subs` | List[SparseTensor] | 多层 sub_logits，每层 `(N_i, 8)` |
| `extrinsics` | (4, 4) | W2C 相机外参 |
| `target_size` | (H, W) | resize 目标分辨率 |

**每层处理**：

| 步骤 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 1 | `sub.feats: (N_i, 8)` | `voxel_normals: (N_i, 3)` | occupancy 梯度 |
| 2 | `sub.coords: (N_i, 4)` | `voxel_id: (H_i, W_i)` | 渲染父 voxel |
| 3 | `voxel_normals, voxel_id` | `pixel_normal: (H_i, W_i, 3)` | 索引 |
| 4 | `pixel_normal` | `normal_cam: (H_i, W_i, 3)` | 变换 + 翻转 |
| 5 | `normal_cam` | `normal_resized: (H, W, 3)` | resize |

**输出**：

| 参数 | Shape | 说明 |
|------|-------|------|
| `normals` | List[(H, W, 3)] | 每层的 normal，已 resize 到目标分辨率，**Camera Space** |
| `masks` | List[(H, W)] | 每层的 mask，已 resize 到目标分辨率 |

### 5.4 边界处理

| 情况 | 处理 | 说明 |
|------|------|------|
| 背景像素 | `voxel_id.clamp(min=0)` + `mask` | mask 会置零，索引值无所谓 |
| FDG 邻居缺失 | `axis_valid_mask` 显式过滤 | 邻居不完整的轴权重强制为 0 |
| FDG 所有轴都无效 | `F.normalize(..., eps=1e-6)` | 返回零向量，这种 voxel 通常不可见 |
| Sub 梯度为零 | `F.normalize(..., eps=1e-6)` | 完全内/外的 voxel 不可见 |

**核心原则**：
- 邻居缺失必须**显式检测**，不能依赖 intersected weight 的隐式假设
- 不可见的像素不需要正确的 normal

---

## 6. 使用示例

### 6.1 FDG 模式

```python
# ===== Per-Voxel Face Normal + Intersected 加权 =====

# 1. 计算 surface_pos
surface_pos = (coords + dual_vertices) * voxel_size + origin  # (N, 3)

# 2. 构建 coord → index 哈希
coord_to_idx = _build_coord_hash(coords)

# 3. 计算每个轴的 face normal + 邻居有效性
axis_normals = []
axis_valid_list = []
for axis in range(3):
    neighbor_coords = coords.unsqueeze(1) + offsets[axis]  # (N, 4, 3)
    neighbor_idx = _lookup_hash(coord_to_idx, neighbor_coords)  # (N, 4), -1 表示不存在
    
    # 检测邻居有效性
    axis_valid = (neighbor_idx != -1).all(dim=1)  # (N,)
    axis_valid_list.append(axis_valid)
    
    # 安全索引
    neighbor_pos = surface_pos[neighbor_idx.clamp(min=0)]  # (N, 4, 3)
    v0, v1, v2, v3 = neighbor_pos.unbind(dim=1)
    axis_normal = F.normalize(torch.cross(v1 - v0, v3 - v0), dim=-1)
    axis_normals.append(axis_normal)

axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)
axis_valid_mask = torch.stack(axis_valid_list, dim=1)  # (N, 3)

# 4. Intersected 加权（邻居缺失的轴权重强制为 0）
weights = torch.sigmoid(intersected_logits)  # (N, 3)
effective_weights = weights * axis_valid_mask.float()  # (N, 3)
weighted = (effective_weights.unsqueeze(-1) * axis_normals).sum(dim=1)  # (N, 3)
voxel_normals = F.normalize(weighted, dim=-1, eps=1e-6)  # (N, 3)

# 5. 硬渲染
voxel_id = hard_render(...)  # (H, W)
mask = voxel_id >= 0

# 6. 变换到 Camera Space
R = extrinsics[:3, :3]  # (3, 3)
t = extrinsics[:3, 3]   # (3,)
voxel_normals_cam = voxel_normals @ R.T      # (N, 3)
surface_pos_cam = surface_pos @ R.T + t      # (N, 3)

# 7. 用点积判断翻转：确保法线朝向相机
dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N, 1)
voxel_normals_cam = torch.where(dot_product > 0, -voxel_normals_cam, voxel_normals_cam)  # (N, 3)

# 8. 索引获取 normal（可微！）+ mask
pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H, W, 3)
```

### 6.2 多分辨率 Sub 模式

```python
# ===== 多分辨率 Occupancy 梯度场 =====

def compute_occupancy_gradient(sub_logits):
    """计算单层的 occupancy 梯度"""
    occupancy = torch.sigmoid(sub_logits)  # (N, 8)
    grad_x = (occupancy[:, [1,3,5,7]] - occupancy[:, [0,2,4,6]]).mean(dim=1)
    grad_y = (occupancy[:, [2,3,6,7]] - occupancy[:, [0,1,4,5]]).mean(dim=1)
    grad_z = (occupancy[:, [4,5,6,7]] - occupancy[:, [0,1,2,3]]).mean(dim=1)
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (N, 3)
    return -F.normalize(gradient, dim=-1, eps=1e-6)  # (N, 3)

# ===== 多层处理 =====
target_size = (H, W)  # 目标分辨率
sub_losses = []

for i, sub in enumerate(subs):
    # sub.feats: (N_i, 8)
    # sub.coords: (N_i, 4) - 第一列是 batch_idx
    
    # 1. 计算 occupancy 梯度 + 父 voxel 中心位置
    voxel_normals = compute_occupancy_gradient(sub.feats)  # (N_i, 3)，已归一化
    surface_pos = sub.coords[:, 1:].float() * voxel_size_i + origin  # (N_i, 3) 父 voxel 中心
    
    # 2. 渲染父 voxel（低分辨率）
    voxel_id = hard_render(
        coords=sub.coords[:, 1:],  # (N_i, 3)
        voxel_size=voxel_size_i,
        ...
    )  # (H_i, W_i)
    mask = voxel_id >= 0
    
    # 3. 变换到 Camera Space + 翻转
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    voxel_normals_cam = voxel_normals @ R.T      # (N_i, 3)
    surface_pos_cam = surface_pos @ R.T + t      # (N_i, 3)
    dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)  # (N_i, 1)
    voxel_normals_cam = torch.where(dot_product > 0, -voxel_normals_cam, voxel_normals_cam)  # (N_i, 3)
    
    # 4. 索引获取 normal + mask
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H_i, W_i, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)  # (H_i, W_i, 3)
    
    # 5. Resize 到统一分辨率
    normal_resized = F.interpolate(
        pixel_normal.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H_i, W_i)
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)
    # 注：双线性插值会改变向量长度，需要重新归一化
    normal_resized = F.normalize(normal_resized, dim=-1, eps=1e-6)
    
    mask_resized = F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),  # (1, 1, H_i, W_i)
        size=target_size,
        mode='nearest'
    ).squeeze() > 0.5  # (H, W)
    
    # 6. 计算 loss
    sub_losses.append(image_loss(normal_resized, gt_normal, mask_resized))

# ===== 与 FDG 配合 =====
total_sub_loss = sum(sub_losses) * lambda_sub
fdg_loss = image_loss(fdg_normal, gt_normal) * lambda_fdg
total_loss = total_sub_loss + fdg_loss
```

---

## 7. 方案对比

### 7.1 FDG vs 多分辨率 Sub

| | FDG | 多分辨率 Sub |
|---|-----|-------------|
| **Normal 来源** | 邻居 dual_vertices | occupancy 梯度 |
| **可微参数** | dual_vertices + intersected_logits | sub_logits (每层) |
| **分辨率** | 单层（最终分辨率） | 多层（低→高） |
| **修改范围** | 几何细节（表面位置/朝向） | 大形状（哪里有表面） |
| **精度** | 高（几何面） | 中（2×2×2 离散） |
| **配合关系** | 依赖 Sub 提供的稀疏结构 | 为 FDG 提供稀疏结构 |

### 7.2 vs MeshRenderer（当前实现）

| | Per-voxel Normal | MeshRenderer |
|---|-----------------|--------------|
| 显存 | **小**（per-voxel tensor） | 大（Mesh + nvdiffrast） |
| 稳定性 | **好**（固定结构） | 差（拓扑变化） |
| dual_vertices 梯度 | ✅ | ✅ |
| intersected 梯度 | **✅ 通过权重** | ❌ detach |
| 精度 | 中等（per-voxel） | 高（per-face） |

### 7.3 vs 全软渲染

| | 本方案 | 全软渲染 |
|---|--------|----------|
| 渲染 | 硬渲染 | 软 alpha 合成 |
| 速度 | 快 | 慢 |
| 实现 | 中等 | 复杂（需要 CUDA） |
| 梯度质量 | 只有 Normal | 完整（depth, alpha, normal） |

### 7.4 vs 屏幕空间梯度

| | Per-voxel Normal | 屏幕空间梯度 |
|---|-----------------|--------------|
| 边界问题 | **无** | 严重（跨 voxel 错误） |
| 分辨率依赖 | **无** | 严重（阶梯状） |
| 几何精度 | 中等 | 差 |
| 实现复杂度 | 中等 | 看似简单，坑多 |

---

## 8. 代码架构

### 8.1 文件结构

```
flow_grpo_custom/
├── _reference_codes/TRELLIS.2/o-voxel/     # 底层渲染器修改
│   ├── src/rasterize/
│   │   ├── rasterize.cu                    # [修改] 输出 voxel_id
│   │   └── api.h                           # [修改] 函数签名
│   └── o_voxel/
│       └── rasterize.py                    # [修改] Python 接口
│
└── edit4shape/
    ├── renderers/
    │   └── diff_voxel_normal.py            # [新建] 可微 Normal 模块
    │
    └── systems/
        └── trellis2_shape.py               # [修改] 系统集成
```

### 8.2 修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `o-voxel/src/rasterize/rasterize.cu` | 修改 | render kernel 输出 `out_voxel_id` |
| `o-voxel/src/rasterize/api.h` | 修改 | 更新函数签名 |
| `o-voxel/o_voxel/rasterize.py` | 修改 | 返回 `voxel_id`，SSAA 用 `nearest` |
| `edit4shape/renderers/diff_voxel_normal.py` | 新建 | FDG + Sub 统一模块 |
| `edit4shape/systems/trellis2_shape.py` | 修改 | 集成新渲染方案 |

### 8.3 diff_voxel_normal.py 函数接口

```python
"""
可微 Voxel Normal 渲染模块

设计原则：
- 每个主函数都是"端到端"的，包含渲染 + normal 计算
- 公共参数用 dataclass 封装
- 使用 o-voxel 原生 CUDA 哈希映射，不自己实现
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
from torch import Tensor
import torch.nn.functional as F

# 复用 o-voxel 的 CUDA 哈希映射和渲染器
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap
from o_voxel.rasterize import render as hard_render_raw  # 硬渲染器

# ============ 公共配置 ============
@dataclass
class RenderConfig:
    """渲染配置"""
    intrinsics: Tensor      # (3, 3) 相机内参
    extrinsics: Tensor      # (4, 4) W2C 外参
    resolution: int         # 渲染分辨率
    voxel_size: float       # 体素尺寸
    origin: Tensor          # (3,) 网格原点
    grid_size: Tensor       # (3,) 网格尺寸，用于哈希映射

# ============ 硬渲染（使用 o-voxel 原生渲染器）============
def hard_render(coords: Tensor, config: RenderConfig) -> Tensor:
    """
    硬渲染获取 voxel_id（需要修改 o-voxel 以输出 voxel_id）
    
    Returns:
        voxel_id: (H, W) 击中的 voxel 索引，-1 表示背景
    """
    # TODO: 调用修改后的 o-voxel 渲染器
    # 返回格式：voxel_id (H, W)，其中 -1 表示背景
    ...

# ============ 邻居查找（使用 o-voxel 原生哈希）============
def find_neighbor_indices(
    coords: Tensor,              # (N, 3) voxel 坐标
    neighbor_offsets: Tensor,    # (3, 4, 3) 每个轴的 4 个邻居偏移
    grid_size: Tensor,           # (3,) 网格尺寸
) -> Tuple[Tensor, Tensor]:
    """
    使用 o-voxel 原生 CUDA 哈希映射查找邻居索引
    
    参考代码 (flexible_dual_grid.py 第 225-236 行):
        hashmap = _init_hashmap(grid_size, 2 * N, device)
        _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords, *grid_size)
        indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size)
    
    Returns:
        neighbor_idx: (N, 3, 4) 每个轴的 4 个邻居索引，无效为 -1
        axis_valid_mask: (N, 3) bool，每个轴的 4 个邻居是否都存在
    """
    N = coords.shape[0]
    device = coords.device
    
    # 构建哈希表
    hashmap = _init_hashmap(grid_size, 2 * N, device)
    coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())
    
    # 查找每个轴的邻居
    neighbor_idx_list = []
    axis_valid_list = []
    
    for axis in range(3):
        # 计算邻居坐标
        offsets = neighbor_offsets[axis]  # (4, 3)
        neighbor_coords = coords.unsqueeze(1) + offsets  # (N, 4, 3)
        neighbor_coords_flat = neighbor_coords.reshape(-1, 3)  # (N*4, 3)
        
        # 查询哈希表
        query = torch.cat([
            torch.zeros((N * 4, 1), dtype=torch.int, device=device),
            neighbor_coords_flat
        ], dim=-1)
        indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())
        indices = indices.reshape(N, 4)  # (N, 4)
        
        # 检查有效性（0xffffffff 表示不存在）
        INVALID = 0xffffffff
        valid = (indices != INVALID).all(dim=1)  # (N,)
        indices = indices.int()
        indices[indices == INVALID] = 0  # 无效位置用 0 替代，后续被 mask
        
        neighbor_idx_list.append(indices)
        axis_valid_list.append(valid)
    
    neighbor_idx = torch.stack(neighbor_idx_list, dim=1)  # (N, 3, 4)
    axis_valid_mask = torch.stack(axis_valid_list, dim=1)  # (N, 3)
    
    return neighbor_idx, axis_valid_mask

# ============ 内部辅助函数 ============
def _compute_axis_face_normals(
    coords: Tensor,           # (N, 3)
    dual_vertices: Tensor,    # (N, 3) 可微
    voxel_size: float,
    origin: Tensor,           # (3,)
    grid_size: Tensor,        # (3,)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    计算每个 voxel 的 3 个轴方向 face normal
    
    Returns:
        axis_normals: (N, 3, 3) 每个轴的 face normal
        axis_valid_mask: (N, 3) bool
        surface_pos: (N, 3) 用于翻转判断
    """
    # edge_neighbor_voxel_offset 来自 flexible_dual_grid.py
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # axis=0: YZ
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # axis=1: XZ
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # axis=2: XY
    ], dtype=torch.int, device=coords.device)  # (3, 4, 3)
    
    # 计算表面位置
    surface_pos = (coords.float() + dual_vertices) * voxel_size + origin  # (N, 3)
    
    # 查找邻居索引
    neighbor_idx, axis_valid_mask = find_neighbor_indices(
        coords, edge_neighbor_voxel_offset, grid_size
    )  # (N, 3, 4), (N, 3)
    
    # 计算每个轴的 face normal
    axis_normals = []
    for axis in range(3):
        # 获取邻居的 surface_pos
        idx = neighbor_idx[:, axis, :]  # (N, 4)
        neighbor_pos = surface_pos[idx.clamp(min=0)]  # (N, 4, 3)，可微！
        
        # 4 个顶点 → face normal
        v0, v1, v2, v3 = neighbor_pos.unbind(dim=1)
        axis_normal = F.normalize(torch.cross(v1 - v0, v3 - v0, dim=-1), dim=-1)
        axis_normals.append(axis_normal)
    
    axis_normals = torch.stack(axis_normals, dim=1)  # (N, 3, 3)
    
    return axis_normals, axis_valid_mask, surface_pos

def _compute_occupancy_gradient(sub_logits: Tensor) -> Tensor:
    """
    计算 occupancy 梯度作为法线方向
    
    Returns:
        voxel_normals: (N, 3) World Space，已归一化
    """
    occupancy = torch.sigmoid(sub_logits)  # (N, 8)
    grad_x = (occupancy[:, [1,3,5,7]] - occupancy[:, [0,2,4,6]]).mean(dim=1)
    grad_y = (occupancy[:, [2,3,6,7]] - occupancy[:, [0,1,4,5]]).mean(dim=1)
    grad_z = (occupancy[:, [4,5,6,7]] - occupancy[:, [0,1,2,3]]).mean(dim=1)
    gradient = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (N, 3)
    return -F.normalize(gradient, dim=-1, eps=1e-6)  # (N, 3)

def _flip_normals_to_camera(
    voxel_normals: Tensor,    # (N, 3) World Space
    surface_pos: Tensor,      # (N, 3) World Space
    extrinsics: Tensor,       # (4, 4) W2C
) -> Tensor:
    """
    变换到 Camera Space + 用点积翻转
    
    Returns:
        voxel_normals_cam: (N, 3) Camera Space，朝向相机
    """
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    
    voxel_normals_cam = voxel_normals @ R.T      # (N, 3)
    surface_pos_cam = surface_pos @ R.T + t      # (N, 3)
    
    dot_product = (voxel_normals_cam * surface_pos_cam).sum(dim=-1, keepdim=True)
    voxel_normals_cam = torch.where(dot_product > 0, -voxel_normals_cam, voxel_normals_cam)
    
    return voxel_normals_cam

# ============ FDG 模式（主函数）============
def render_normal_fdg(
    coords: Tensor,                # (N, 3) voxel 整数坐标
    dual_vertices: Tensor,         # (N, 3) 可微
    intersected_logits: Tensor,    # (N, 3) 可微
    config: RenderConfig,
) -> Tuple[Tensor, Tensor]:
    """
    FDG 模式：渲染 + 计算可微 normal
    
    Returns:
        normal: (H, W, 3) Camera Space
        mask: (H, W) bool
    """
    # 1. 计算 axis_normals + surface_pos
    axis_normals, axis_valid_mask, surface_pos = _compute_axis_face_normals(
        coords, dual_vertices, config.voxel_size, config.origin, config.grid_size
    )
    
    # 2. intersected 加权
    weights = torch.sigmoid(intersected_logits)  # (N, 3)
    effective_weights = weights * axis_valid_mask.float()  # (N, 3)
    weighted = (effective_weights.unsqueeze(-1) * axis_normals).sum(dim=1)  # (N, 3)
    voxel_normals = F.normalize(weighted, dim=-1, eps=1e-6)  # (N, 3)
    
    # 3. 硬渲染 → voxel_id
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0
    
    # 4. 变换 + 翻转
    voxel_normals_cam = _flip_normals_to_camera(
        voxel_normals, surface_pos, config.extrinsics
    )
    
    # 5. 索引 → pixel_normal
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)
    
    return pixel_normal, mask

# ============ Sub 模式（主函数）============
def render_normal_sub(
    sub: SparseTensor,             # feats: (N, 8), coords: (N, 4)
    config: RenderConfig,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    单层 Sub 模式：渲染 + 计算可微 normal
    
    Returns:
        normal: (H, W, 3) 或 (target_H, target_W, 3) Camera Space
        mask: (H, W) 或 (target_H, target_W) bool
    """
    coords = sub.coords[:, 1:]  # (N, 3) 去掉 batch_idx
    
    # 1. 计算 occupancy 梯度
    voxel_normals = _compute_occupancy_gradient(sub.feats)  # (N, 3)
    
    # 2. 计算 surface_pos（父 voxel 中心）
    surface_pos = coords.float() * config.voxel_size + config.origin  # (N, 3)
    
    # 3. 硬渲染 → voxel_id
    voxel_id = hard_render(coords, config)  # (H, W)
    mask = voxel_id >= 0
    
    # 4. 变换 + 翻转
    voxel_normals_cam = _flip_normals_to_camera(
        voxel_normals, surface_pos, config.extrinsics
    )
    
    # 5. 索引 → pixel_normal
    pixel_normal = voxel_normals_cam[voxel_id.clamp(min=0)]  # (H, W, 3)
    pixel_normal = pixel_normal * mask.unsqueeze(-1)
    
    # 6. (可选) resize + 归一化
    if target_size is not None:
        pixel_normal = F.interpolate(
            pixel_normal.permute(2, 0, 1).unsqueeze(0),
            size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        pixel_normal = F.normalize(pixel_normal, dim=-1, eps=1e-6)
        
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=target_size, mode='nearest'
        ).squeeze() > 0.5
    
    return pixel_normal, mask

def render_normal_sub_multi(
    subs: List[SparseTensor],
    configs: List[RenderConfig],   # 每层配置（voxel_size 不同）
    target_size: Tuple[int, int],
) -> List[Tuple[Tensor, Tensor]]:
    """
    多分辨率 Sub 模式
    
    Returns:
        List of (normal, mask)，每层已 resize 到 target_size
    """
    results = []
    for sub, config in zip(subs, configs):
        normal, mask = render_normal_sub(sub, config, target_size)
        results.append((normal, mask))
    return results
```

### 8.4 依赖关系

```
rasterize.cu (CUDA)
    ↓
rasterize.py (Python 接口)
    ↓
diff_voxel_normal.py (可微 Normal)
    ↓
trellis2_shape.py (系统集成)
```

---

## 9. 已确认的设计决策

本节记录经过分析确认的设计决策，以及一些"看似问题但实际上合理"的设计点。

### 9.1 Cross Product 顶点顺序

**结论**：✅ Python 版本的 `edge_neighbor_voxel_offset` 顶点顺序是一致的，可以直接使用。

**分析**：
- Python 实现 (`flexible_dual_grid.py`) 与 C++ 实现 (`flexible_dual_grid.cpp`) 的顶点顺序在 axis=0 和 axis=2 上有差异
- 但由于我们使用的是 Python 版本，只要内部一致即可
- 所有轴的法线都一致地指向负方向（-X, -Y, -Z），通过翻转机制统一处理

### 9.2 Sub 模式的离散梯度精度

**结论**：✅ 2×2×2 的离散梯度虽然粗糙，但对于"大形状修改"足够。

**分析**：
- Sub 模式的目标是**修改稀疏结构**（哪里有表面），而不是精确的几何细节
- 精确的法线由 FDG 模式负责
- 粗糙的梯度足以提供"方向信号"，告诉网络往哪个方向调整 `sub_logits`

### 9.3 Sub 模式的"渲染 vs Normal 不匹配"

**结论**：✅ 这是合理的设计，从训练角度可能更有效。

**分析**：
- 硬渲染击中的是**父 voxel 立方体**的面，但赋予的 normal 是 **occupancy 梯度方向**
- 几何上不匹配，但从训练角度合理：
  - occupancy 梯度**直接由 sub_logits 控制**
  - 改变 sub_logits → 改变 occupancy 分布 → 改变梯度方向
  - 如果用立方体面的 normal（固定的 ±X/±Y/±Z），梯度信号无法有效优化 sub_logits
- 本质是让"梯度信号和可控参数对齐"

### 9.4 axis_valid_mask 与 intersected_logits 的关系

**结论**：✅ 两者**不冗余**，各自负责不同的过滤。

**分析**：
- `intersected_logits`：控制"这个轴是否有表面"（可微参数）
- `axis_valid_mask`：检测"邻居是否存在"（稀疏结构约束）
- 两者是独立的条件，需要同时满足才生成有效的 axis_normal
- 这与原始 FDG 的逻辑一致：`is_intersected[axis] && neighbors_exist`

### 9.5 边界情况处理

**结论**：✅ 边界情况的处理是合理的。

| 情况 | 处理方式 | 说明 |
|------|---------|------|
| 所有轴无效 | `F.normalize(..., eps=1e-6)` 返回零向量 | 这种 voxel 通常在边界，不可见 |
| 背景像素 | `voxel_id.clamp(min=0)` + `mask` 置零 | 索引安全，结果被 mask 屏蔽 |
| Sub 梯度为零 | `F.normalize(..., eps=1e-6)` 返回零向量 | 完全内/外的 voxel 不可见 |
| Cross product 退化 | 被其他轴覆盖或返回零向量 | 不常见，影响有限 |

### 9.6 多分辨率层权重

**结论**：⚠️ 先用等权重，如果效果不好再调整。

**当前方案**：
```python
total_sub_loss = sum(sub_losses) * lambda_sub
```

**如需调整的备选方案**：
- 按层设置不同权重（高分辨率层权重更大）
- 只用最后几层
- 按分辨率 resize 到不同目标

### 9.7 SSAA（超采样抗锯齿）

**结论**：⚠️ 先不使用 SSAA，如果边缘问题明显再考虑。

**原因**：
- `voxel_id` 是整数索引，无法用双线性插值降采样
- 锯齿影响主要在边缘，物体内部的 normal 不受影响

**备选方案**（如需要）：
- 对 normal（而不是 voxel_id）做 SSAA
- 在 loss 中忽略边缘像素

### 9.8 实现建议

| 问题 | 建议方案 |
|------|---------|
| Batch 处理 | 每个 batch 单独处理，复用单 batch 代码 |
| 哈希效率 | ✅ 已解决：使用 o-voxel 原生 CUDA 哈希表 |
| 归一化 | 计算时归一化 + resize 后再归一化 |
| 函数设计 | ✅ 已解决：端到端设计 + `RenderConfig` 封装参数 |

**关于 o-voxel 原生哈希的使用**：

```python
# 参考 flexible_dual_grid.py 第 225-236 行
from o_voxel import _C
from o_voxel.convert.flexible_dual_grid import _init_hashmap

# 构建哈希表
hashmap = _init_hashmap(grid_size, 2 * N, device)
coords_with_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
_C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords_with_batch, *grid_size.tolist())

# 查询
indices = _C.hashmap_lookup_3d_cuda(*hashmap, query, *grid_size.tolist())
# 0xffffffff 表示不存在
```
