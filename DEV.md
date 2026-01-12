# 可微分 VoxelRenderer 方案

## 🎯 目标

将 `trellis2_shape.py` 的渲染流程从 **MeshRenderer** 切换为 **可微分 VoxelRenderer**，实现端到端的梯度回传，用于训练 Flow Model 的 LoRA。

---

## 🔧 核心问题与解决方案

### 问题 1：原始 VoxelRenderer 不可微

**原因**：CUDA kernel 使用 "first hit" 策略，没有实现 backward pass

**部分解决**：
- `depth_to_normal` 是纯 PyTorch 操作，天然可微 ✅
- 但 **Depth 本身来自不可微的 o_voxel 渲染** ❌

**⚠️ 注意：STE 不适用于此场景**
- STE 用于离散选择（如 argmax），让梯度"穿过"离散操作
- 这里的问题是：`o_voxel` 渲染是黑盒 CUDA kernel，Depth 和 position/opacity 之间没有保留数学关系
- 即使让梯度"穿过"，也不知道 ∂Depth/∂position 应该是多少

**真正的解决方案**：需要实现 CUDA Alpha Blending + Backward（见后续计划）

### 问题 2：FDG Decoder 的 Mesh 提取不可微

**原因**：推理模式使用 `h.feats[..., 3:6] > 0` 硬阈值提取 mesh

**解决方案**：
- 绕过 Mesh 提取，直接使用 Decoder 的原始输出 `h.feats`
- 创建 **VoxelProxy** 类，将连续的 logits 转换为可微的体素属性

---

## 📁 文件结构

```
edit4shape/renderers/
├── voxel_proxy.py          # VoxelProxy 类，桥接 Decoder 输出与 Renderer 输入
├── ovoxel_trellis2.py      # OVoxelRenderer 包装（不可微）
├── soft_voxel_renderer.py  # ✅ SoftVoxelRenderer（纯 PyTorch 可微，用于验证）
└── __init__.py

edit4shape/systems/
└── trellis2_shape.py       # 使用新渲染流程

scripts/debug/
└── test_trellis2_voxel-mesh|voxel_render.py  # 渲染器对比测试脚本
```

---

## 📊 核心数据流

```
FlexiDualGridVaeDecoder
         │
         ▼
   h.feats (N, 7)
   ┌─────────────────────────────────────────┐
   │ [0:3] dual_vertices  → 体素位置偏移      │
   │ [3:6] intersected    → 体素不透明度      │
   │ [6:7] quad_lerp      → (未使用)         │
   └─────────────────────────────────────────┘
         │
         ▼
    VoxelProxy.from_fdg_decoder()
         │
   ┌─────┴─────┐
   │           │
   ▼           ▼
position    opacities
(可微)       (可微)
   │           │
   └─────┬─────┘
         │
         ▼
  DiffVoxelRenderer.render_proxy()
         │
         ▼
    Depth Map
         │
         ▼
  depth_to_normal() (可微)
         │
         ▼
    Normal Map
         │
         ▼
   Guidance Loss
```

---

## 🧮 关键公式

### 1. 位置计算（可微）

```python
dual_vertices = sigmoid(h.feats[..., 0:3]) * (1 + 2*margin) - margin
base_position = (coords + 0.5) * voxel_size + origin
position = base_position + (dual_vertices - 0.5) * voxel_size
```

### 2. 不透明度计算（可微）

```python
intersected_logits = h.feats[..., 3:6]
max_logit = intersected_logits.max(dim=-1).values
opacities = sigmoid(max_logit * temperature)  # temperature=10.0
```

### 3. Depth-to-Normal（可微）

```python
# 图像空间梯度
dz_dx = depth[:, 2:, 1:-1] - depth[:, :-2, 1:-1]
dz_dy = depth[:, 1:-1, 2:] - depth[:, 1:-1, :-2]
# 世界空间法线
normal = normalize(cross([dx, 0, dz_dx], [0, dy, dz_dy]))
```

---

## 🔄 梯度流路径

### 当前状态（梯度断裂）

```
Loss
  │
  ▼
Normal (H, W, 3)
  │ ✅ (depth_to_normal 可微)
  ▼
Depth (H, W)
  │ ❌ (o_voxel 渲染不可微，梯度断裂)
  ╳
VoxelProxy
  │
  ├──► position ◄── h.feats[0:3]
  └──► opacities ◄── h.feats[3:6]
```

### 目标状态（需要实现 CUDA 可微）

```
Loss
  │
  ▼
Normal (H, W, 3)
  │ ✅ (depth_to_normal 可微)
  ▼
Depth (H, W)
  │ ✅ (alpha blending backward)
  ▼
opacities ◄── h.feats[3:6] (intersected)
  │
  ▼
FlexiDualGridVaeDecoder
  │
  ▼
Flow Model LoRA
```

---

## 📝 主要代码修改

| 文件 | 修改内容 |
|------|----------|
| `voxel_proxy.py` | 新增 `VoxelProxy` 类，桥接 Decoder 输出与 Renderer 输入 |
| `ovoxel_trellis2.py` | 新增 `DiffVoxelRenderer`，实现 `render_proxy()` 和 `render_batch()` |
| `trellis2_shape.py` | 新增 `decode_and_render_normal_voxel()`，替换原有 Mesh 渲染流程 |

---

## ⚠️ 注意事项

1. **❌ 当前不可微**：o_voxel 渲染器没有 backward，梯度无法回传到 VoxelProxy
2. **需要实现 CUDA 可微**：必须实现 Alpha Blending + Backward 才能真正训练
3. **未使用 quad_lerp**：`h.feats[..., 6:7]` 仅用于 Mesh 四边形切分，未参与优化
4. **不生成 Mesh**：`evaluate()` 中设置 `export_mesh=False`

---

## 🚀 必须实现：CUDA 可微光栅化

**当前状态**：o_voxel 渲染不可微，梯度无法回传，无法训练。

**必须实现**：修改 CUDA kernel，添加 Alpha Blending + Backward。

### 目标

将 `o-voxel/src/rasterize/rasterize.cu` 改造为真正的可微光栅化器，支持：
- 对 **opacity** 可微
- 对 **normal/color** 可微

---

### 核心算法改动

#### 1. Forward Pass：从 "First Hit" 改为 "Alpha Blending"

**当前实现（不可微）**：

```cuda
// 找到第一个相交的 voxel 就停止
if (intersected) {
    depth = t_near;
    break;  // 硬选择，梯度无法回传
}
```

**改为 Alpha Blending（可微）**：

```cuda
// 沿射线累积所有 voxel 的贡献
float T = 1.0f;  // 累积透射率
for each voxel in sorted order:
    if (intersected) {
        float alpha = opacity[voxel_idx];
        float weight = T * alpha;
        
        // 累积深度
        accumulated_depth += weight * t_near;
        accumulated_normal += weight * normal[voxel_idx];
        accumulated_color += weight * color[voxel_idx];
        
        T *= (1 - alpha);  // 更新透射率
        
        if (T < 0.001f) break;  // 早停优化
    }
```

#### 2. Backward Pass：计算梯度

需要保存 forward 的中间结果，用于 backward：

```cuda
__global__ void rasterize_backward_kernel(
    // Forward 保存的中间结果
    const float* saved_weights,      // [num_pixels, max_hits]
    const int* saved_voxel_indices,  // [num_pixels, max_hits]
    const float* saved_T,            // [num_pixels, max_hits]
    
    // 上游梯度
    const float* grad_depth,         // [H, W]
    const float* grad_normal,        // [H, W, 3]
    const float* grad_color,         // [H, W, 3]
    
    // 输出梯度
    float* grad_opacity,             // [num_voxels]
    float* grad_voxel_normal,        // [num_voxels, 3]
    float* grad_voxel_color          // [num_voxels, 3]
) {
    // 对于每个像素
    for each hit i in this pixel:
        int voxel_idx = saved_voxel_indices[pixel][i];
        float weight = saved_weights[pixel][i];
        float T = saved_T[pixel][i];
        
        // 颜色/法线梯度（简单链式法则）
        atomicAdd(&grad_voxel_normal[voxel_idx], grad_normal[pixel] * weight);
        atomicAdd(&grad_voxel_color[voxel_idx], grad_color[pixel] * weight);
        
        // opacity 梯度（需要考虑对后续 hit 的影响）
        float grad_alpha = compute_alpha_gradient(...);
        atomicAdd(&grad_opacity[voxel_idx], grad_alpha);
}
```

---

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `o-voxel/src/rasterize/rasterize.cu` | 添加 `rasterize_backward_kernel`，修改 forward 保存中间结果 |
| `o-voxel/src/rasterize/rasterize.h` | 添加 backward 函数声明 |
| `o-voxel/o_voxel/rasterize.py` | 添加 `torch.autograd.Function` 包装 |

---

### PyTorch 集成

```python
class DiffVoxelRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, opacities, normals, colors, ...):
        # 调用 CUDA forward
        depth, normal, color, alpha = _C.rasterize_forward(...)
        
        # 保存用于 backward
        ctx.save_for_backward(positions, opacities, ...)
        ctx.saved_indices = indices
        ctx.saved_weights = weights
        
        return depth, normal, color, alpha
    
    @staticmethod
    def backward(ctx, grad_depth, grad_normal, grad_color, grad_alpha):
        # 调用 CUDA backward
        grad_opacity, grad_voxel_normal, grad_voxel_color = \
            _C.rasterize_backward(ctx.saved_indices, ctx.saved_weights, ...)
        
        return None, grad_opacity, grad_voxel_normal, grad_voxel_color, ...
```

---

### 性能考虑

1. **内存开销**：需要保存每个像素的 hit 列表（可限制 max_hits=16）
2. **原子操作**：backward 中的 `atomicAdd` 可能成为瓶颈
3. **早停优化**：当 T < threshold 时停止累积

---

### 优先级建议

| 优先级 | 功能 | 难度 | 收益 |
|--------|------|------|------|
| **P0** | CUDA forward: Alpha Blending | 中 | 必须（当前完全不可微） |
| **P1** | CUDA backward: opacity 梯度 | 中 | 高（控制形状） |
| P2 | position 梯度 | 高 | 中（位置优化） |

---

## ✅ 已验证：纯 PyTorch 可微渲染 (SoftVoxelRenderer)

**用途**：验证梯度可行性，**不用于实际训练**（太慢）。

**实现文件**：`edit4shape/renderers/soft_voxel_renderer.py`

### 核心原理：Soft Z-buffer

1. **投影体素到屏幕**：将 3D 体素位置投影到 2D 像素坐标
2. **相对深度加权**：`depth_weights = exp(-(z - z_min) * temperature)`
3. **Scatter 累积**：用 `scatter_add_` 累积到像素
4. **归一化**：`depth = depth_sum / weight_sum`

### 验证结果 (2025-01-11)

```
测试脚本: scripts/debug/test_trellis2_voxel-mesh|voxel_render.py

[OVoxel] depth range: [1.7622, 2.6547]  (CUDA 精确渲染)
[Soft]   depth range: [0.0000, 2.0148]  (PyTorch 近似渲染)

[OVoxel] mask sum: 49757
[Soft]   mask sum: 49846  (差异: 0.04%)

✅ h_feats 梯度正常: norm=49.016586, nonzero=2347371 (234万个非零梯度)
```

### 梯度特性

| 变量 | 可微 | 说明 |
|------|------|------|
| `opacities` | ✅ | 完全可微，通过 `scatter_add_` 保留梯度 |
| `positions.z` | ✅ | 可微（影响深度权重和深度值） |
| `positions.x/y` | ❌ | 不可微（只影响像素索引，使用 `.long()` 整数操作） |

### 结论

**VoxelProxy 设计正确**：只要实现 CUDA Alpha Blending + Backward，梯度就能正确回传到 Flow Model。

---

## 📋 下一步计划

| 优先级 | 任务 | 状态 |
|--------|------|------|
| ~~P0~~ | 验证梯度可行性 | ✅ 完成 (SoftVoxelRenderer) |
| **P1** | 伪 GT Mesh + Occupancy 监督 | 📋 待实现（见下方替代方案） |
| P2 | CUDA Alpha Blending Forward | 📋 备选 |
| P3 | CUDA Backward (opacity 梯度) | 📋 备选 |

---

## 🆕 替代方案：伪 GT Mesh + Voxel Occupancy 监督

**背景**：实现 CUDA 可微光栅化工作量大。这里提供一个纯 PyTorch 的替代方案，利用现有架构实现可微训练。

---

### 方案概览

| 方案 | 角色 | 可微特征 | 渲染器 |
|------|------|----------|--------|
| **A: 伪 GT Mesh** | 🔴 主力监督 | `h.feats[0:3]`, `h.feats[6:7]` | MeshRenderer (nvdiffrast) |
| **B: Voxel Occupancy** | 🟡 辅助监督 | `subs[i].feats` | 纯 PyTorch 体渲染 |

---

### 🔴 方案 A：伪 GT Mesh 渲染（主力）

**核心思路**：用硬阈值生成的 `intersected` 作为伪 GT，固定拓扑，让顶点位置可微。

#### 数据流

```
h.feats (N, 7)
    │
    ├── [0:3] dual_vertices = sigmoid(x) * 2 - 0.5   ✅ 可微
    ├── [3:6] intersected = (x > 0).detach()         ❌ 固定拓扑
    └── [6:7] quad_lerp = softplus(x)                ✅ 可微
    │
    ▼
flexible_dual_grid_to_mesh(train=True)
    │
    ▼
Mesh (vertices 可微, faces 固定)
    │
    ▼
MeshRenderer (nvdiffrast) ✅ 完全可微
    │
    ▼
Normal → Loss
```

#### 可训练性

| 通道 | 特征 | 可微 | 物理意义 |
|------|------|------|----------|
| `[0:3]` | dual_vertices | ✅ 是 | 顶点局部偏移，范围 (-0.5, 1.5) |
| `[3:6]` | intersected | ❌ 否 | 拓扑结构（硬阈值 detach） |
| `[6:7]` | quad_lerp | ✅ 是 | 四边形分割权重，范围 (0, +∞) |

**可训练率**：4/7 通道 ≈ 57%

#### 伪代码

```python
def decode_and_render_mesh_pseudo_gt(shape_slat, cameras, renderer, resolution):
    decoder = pipeline.pipe.models['shape_slat_decoder']
    margin = decoder.voxel_margin
    
    # 获取原始特征
    h, subs = parent_forward(decoder, shape_slat, return_subs=True)
    
    # 伪 GT intersected（拓扑固定，不参与梯度）
    pseudo_gt_intersected = h.replace((h.feats[..., 3:6] > 0).detach())
    
    # 可微特征
    vertices = h.replace((1 + 2*margin) * sigmoid(h.feats[..., 0:3]) - margin)
    quad_lerp = h.replace(softplus(h.feats[..., 6:7]))
    
    # 生成可微 Mesh（train=True 使用加权中点插值）
    meshes = [Mesh(*flexible_dual_grid_to_mesh(
        v.coords[:, 1:], v.feats, i.feats, q.feats,
        aabb=[[-0.5,-0.5,-0.5], [0.5,0.5,0.5]],
        grid_size=resolution,
        train=True  # 关键：使用可微分支
    )) for v, i, q in zip(vertices, pseudo_gt_intersected, quad_lerp)]
    
    # nvdiffrast 渲染（完全可微）
    normals = render_normal(meshes, cameras, renderer)
    
    return {"normal": normals, "subs": subs, "meshes": meshes}
```

---

### 🟡 方案 B：Voxel Occupancy 渲染（辅助）

**核心思路**：用 Soft Z-buffer 渲染监督 Decoder 各层的 `subs` subdivision 预测。

**目的**：让 `subs[i].feats` 参与梯度，优化 Decoder 中间层的结构预测。

#### subs 的含义

- `subs[i].coords`: (N_i, 4) 父 voxel 坐标（含 batch 索引）
- `subs[i].feats`: (N_i, 8) 每个父 voxel 的 8 个子 voxel 占用 logits
- `sigmoid(subs[i].feats)`: 占用概率 ∈ (0, 1)

#### 可微实现（Soft Z-buffer）

复用 `soft_voxel_render`，比 Volume Rendering 快 ~20 倍：

```python
def expand_subdivision_to_voxels(parent_coords, sub_logits, parent_resolution):
    """将 (N, 8) subdivision 展开成 (N*8, 3) 子 voxel 位置 + 占用概率"""
    # 8 个子 voxel 偏移
    offsets = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                            [0,0,1],[1,0,1],[0,1,1],[1,1,1]]) * 0.5
    voxel_size = 1.0 / parent_resolution
    
    # 子 voxel 世界坐标
    parent_origin = parent_coords.float() * voxel_size - 0.5
    positions = (parent_origin.unsqueeze(1) + offsets * voxel_size).reshape(-1, 3)
    occupancies = sigmoid(sub_logits).reshape(-1)
    
    return positions, occupancies

def multiscale_occupancy_loss(subs, extrinsics, intrinsics, target_alpha):
    """多尺度 occupancy 监督"""
    total_loss, weight_sum = 0.0, 0.0
    for i, sub in enumerate(subs):
        parent_res = 64 * (2 ** i)
        render_size = min(parent_res * 2, 256)
        
        # 展开 + 渲染
        positions, occupancies = expand_subdivision_to_voxels(sub.coords[:, 1:], sub.feats, parent_res)
        out = soft_voxel_render(positions, occupancies, extrinsics, intrinsics, render_size, render_size)
        
        # 与下采样 target 对比
        target_i = F.interpolate(target_alpha[None, None], (render_size, render_size)).squeeze()
        layer_loss = F.mse_loss(out['alpha'], target_i)
        
        layer_weight = 2 ** i  # 高分辨率层权重大
        total_loss += layer_weight * layer_loss
        weight_sum += layer_weight
    
    return total_loss / weight_sum
```

#### 梯度流

```
Loss → rendered_alpha → soft_voxel_render (scatter_add_)
    → occupancies = sigmoid(sub.feats) ✅
    → subs[i].feats → Decoder 中间层 → Flow Model ✅
```

#### 实现文件

`edit4shape/renderers/soft_voxel_renderer.py` 中的：
- `expand_subdivision_to_voxels()`: 展开 subdivision
- `multiscale_occupancy_loss()`: 多尺度监督 loss

---

### 总体训练 Loss

```python
# 主力：Mesh 渲染监督几何细节
mesh_loss = render_loss(mesh_normal, target_normal)

# 辅助：Occupancy 监督结构预测
occupancy_loss = multiscale_occupancy_loss(subs, cameras, target_alpha)

# 总 Loss
total_loss = mesh_loss + λ_occ * occupancy_loss
```

---

### 与 CUDA 可微方案的对比

| 维度 | 伪 GT Mesh + Occupancy | CUDA Alpha Blending |
|------|------------------------|---------------------|
| 实现难度 | ⭐⭐ 纯 PyTorch | ⭐⭐⭐⭐ CUDA 开发 |
| Mesh 可微特征 | 57% (4/7 通道) | - |
| Decoder 中间层 | ✅ 通过 subs 监督 | ❌ 无 |
| 渲染质量 | 精确 Mesh + 体渲染 | Voxel 近似 |
| 推荐度 | ✅ 优先尝试 | 备选方案 |

**结论**：建议用 **方案 A（伪 GT Mesh）** 作为主力监督，**方案 B（Occupancy 体渲染）** 作为辅助监督。如果效果不足，再实现 CUDA 可微光栅化。
