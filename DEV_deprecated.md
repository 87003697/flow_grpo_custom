# 可微 Voxel Renderer 开发方案

## 1. 背景与动机

### 1.1 当前 VoxelRenderer 的局限性

当前 `o-voxel` 的 `VoxelRenderer` 采用硬光栅化：
- 找到第一个相交的 voxel，直接输出颜色
- 使用 voxel 中心位置，不使用 `dual_vertices`
- 渲染过程不可微

### 1.2 TRELLIS.2 中 intersected 的训练方式

当前 TRELLIS.2 对 `intersected` 的训练：
- 解码器输出 `intersected_logits`（连续值）
- 用 **BCE Loss 直接监督**，不通过渲染 loss
- 推理时用 `> 0` 阈值二值化

**问题**：Mesh 重建使用 GT 或二值化的 intersected，梯度无法回传。

### 1.3 目标与设计原则

**目标**：构建统一的可微 Voxel 渲染框架，支持两种模式：

| 模式 | 应用场景 | Alpha 来源 |
|------|---------|-----------|
| **FDG** | Mesh 重建（dual_vertices + intersected） | `soft_intersected (N, 3)` + per-pixel ray_dir 加权 |
| **Sub** | Occupancy 监督（subdivision） | `sigmoid(sub_logit)` 直接作为 alpha |

**设计原则**：
- 分层架构：Alpha 计算层（模式相关）+ Compositing 层（共享）
- 不引入新的可学习参数
- 两个独立的 Python 接口，底层共享 CUDA 核心

## 2. 代码架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Python 层                                                       │
├─────────────────────────────┬───────────────────────────────────┤
│  DiffRasterizeSub           │  DiffRasterizeFDG                 │
│  • alpha: (N,) 直接传入      │  • soft_intersected: (N, 3)       │
│  • position: (N, 3)         │  • aabb_centers / surface_pos     │
└─────────────┬───────────────┴─────────────┬─────────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  CUDA 层                                                         │
├─────────────────────────────┬───────────────────────────────────┤
│  diff_rasterize_sub.cu      │  diff_rasterize_fdg.cu            │
│  • alpha 直接使用            │  • Alpha 计算层（per-pixel ray）   │
└─────────────┬───────────────┴─────────────┬─────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  diff_compositing.cuh        │
              │  共享 Alpha Compositing 核心  │
              └──────────────────────────────┘
```

### 2.2 目录结构

在 `o-voxel` 内扩展，新增可微渲染模块：

```
o-voxel/
├── o_voxel/
│   ├── __init__.py                  # 添加导出
│   ├── rasterize.py                 # 原有 VoxelRenderer (不可微)
│   ├── diff_rasterize_sub.py        # [新增] Sub 模式接口
│   └── diff_rasterize_fdg.py        # [新增] FDG 模式接口
│
├── src/
│   ├── ext.cpp                      # [修改] 注册新函数
│   └── rasterize/
│       ├── rasterize.cu             # 原有
│       ├── auxiliary.h              # 原有，复用辅助函数
│       ├── config.h                 # 原有
│       ├── diff_compositing.cuh     # [新增] 共享 Compositing 核心
│       ├── diff_alpha_fdg.cuh       # [新增] FDG Alpha 计算
│       ├── diff_rasterize_sub.cu    # [新增] Sub 渲染实现
│       └── diff_rasterize_fdg.cu    # [新增] FDG 渲染实现
│
└── setup.py                         # 无需修改，自动编译新增 .cu 文件
```

### 2.3 各模块来源标注

| 模块 | 来源 | 说明 |
|------|------|------|
| Tile-based 预处理 | 复用 o-voxel | `preprocess`, `duplicateWithKeys`, `identifyTileRanges` |
| Ray-Box Intersection | 复用 o-voxel | `auxiliary.h` 中的 `get_ray_voxel_intersection` |
| **Alpha Compositing** | 借鉴 3DGS | `diff_compositing.cuh`，**FDG/Sub 共享** |
| **Alpha 计算 (FDG)** | 新增 | `diff_alpha_fdg.cuh`，per-pixel ray_dir 加权 |
| **Alpha 计算 (Sub)** | 新增 | 直接使用传入的 alpha |
| Backward 遍历 | 借鉴 3DGS | back-to-front 梯度累积，atomicAdd |
| Position 梯度 | 新增 | 基于 depth 的解析导数 |

### 2.4 核心文件清单

| 文件 | 状态 | 功能 |
|------|------|------|
| `src/rasterize/diff_compositing.cuh` | 新增 | 共享 Compositing 核心（header-only） |
| `src/rasterize/diff_alpha_fdg.cuh` | 新增 | FDG Alpha 计算（header-only） |
| `src/rasterize/diff_rasterize_sub.cu` | 新增 | Sub 模式 Forward + Backward |
| `src/rasterize/diff_rasterize_fdg.cu` | 新增 | FDG 模式 Forward + Backward |
| `src/ext.cpp` | 修改 | 注册 Sub/FDG 的 forward/backward |
| `o_voxel/diff_rasterize_sub.py` | 新增 | Sub 模式 `torch.autograd.Function` |
| `o_voxel/diff_rasterize_fdg.py` | 新增 | FDG 模式 `torch.autograd.Function` |
| `o_voxel/__init__.py` | 修改 | 添加导出 |

## 3. 核心算法

### 3.1 Alpha 计算

#### 3.1.1 Sub 模式

Alpha 直接由调用方提供，与视角无关：

```python
alpha = torch.sigmoid(sub_logit)  # (N,) 每个 voxel 一个固定 alpha
```

#### 3.1.2 FDG 模式

Alpha 依赖 per-pixel 射线方向，**在 CUDA kernel 内计算**：

```
α = Σᵢ (paraᵢ × sᵢ) / Σᵢ paraᵢ
```

其中：
- `sᵢ = sigmoid(intersected_logitsᵢ)` — 软化的边相交概率，i ∈ {x, y, z}
- `paraᵢ = |ray_dirᵢ|` — 射线与第 i 轴的平行程度（**每个像素不同**）

**几何意义**：
- `intersected_i = 1` 表示表面穿过第 i 轴边 → 表面大致垂直于第 i 轴
- 当射线与第 i 轴平行时，会正面击中该表面 → `para_i` 大，`s_i` 权重高

**为什么 FDG 不能预计算 Alpha**：
- 透视投影下，每个像素的 `ray_dir` 不同
- 同一个 voxel 从不同像素看，alpha 不同
- 因此必须在渲染 kernel 内实时计算

### 3.2 前向渲染

**输入**：aabb_centers, surface_positions, attrs, soft_intersected, camera

**Depth 计算**：

ray 与 AABB 相交判断使用 `aabb_centers`，但 depth 使用 `surface_pos` 在射线方向的投影：

```
depthₖ = (surface_posₖ - ray_origin) · ray_dir
```

其中 `ray_dir` 是归一化的射线方向。这样 depth 仍沿射线方向测量，物理意义清晰。

**Alpha Compositing 公式**（按深度从近到远遍历）：

```
Tₖ = ∏ⱼ₌₁ᵏ⁻¹ (1 - αⱼ)           # 第 k 个 voxel 之前的累积透过率
wₖ = αₖ × Tₖ                     # 第 k 个 voxel 的权重

C = Σₖ wₖ × colorₖ               # 最终颜色
D = Σₖ wₖ × depthₖ               # 最终深度  
A = 1 - T_final                  # 最终 alpha
```

**伪代码**：
```python
T = 1.0
for voxel_k in sorted_by_depth:
    α = compute_alpha(ray_dir, soft_intersected[k])
    depth_k = dot(surface_pos[k] - ray_origin, ray_dir)
    w = α * T
    C += w * color[k]
    D += w * depth_k
    T *= (1 - α)
```

### 3.3 反向传播

**目标**：计算 `∂L/∂soft_intersected` 和 `∂L/∂position`

**链式法则**（从后向前遍历）：

```
∂L/∂αₖ = ∂L/∂C × (Tₖ × colorₖ - Cₖ₊₁/(1-αₖ))
       + ∂L/∂D × (Tₖ × depthₖ - Dₖ₊₁/(1-αₖ))
       + ∂L/∂A × Tₖ
```

其中 `Cₖ₊₁` 和 `Dₖ₊₁` 分别是第 k 个之后所有 voxel 的累积颜色和深度贡献。

**梯度传播**：

```
∂L/∂colorₖ = wₖ × ∂L/∂C

∂L/∂sᵢ = ∂L/∂α × paraᵢ / Σpara           # → intersected_logits

∂depthₖ/∂surface_posₖ = ray_dir          # depth 对 surface_pos 的导数
∂L/∂surface_posₖ = ∂L/∂D × wₖ × ray_dir  # → dual_vertices
```

**伪代码**：
```python
T = T_final
for voxel_k in reversed(saved_voxels):
    T /= (1 - α[k])
    dL_dα = compute_alpha_grad(dL_dC, dL_dA, T, color[k], C_accum)
    dL_d_intersected[k] = dL_dα * para / sum(para)
    dL_d_surface_pos[k] = dL_dD * w[k] * ray_dir  # 梯度传到 dual_vertices
```

## 4. intersected 通道定义

### 4.1 存储格式

原始存储为 uint8，通过位解包得到 3 个 bool 通道：

```python
# flexi_dual_grid.py
intersected = torch.cat([
    attr['intersected'] % 2,        # 通道 0 → X 轴边
    attr['intersected'] // 2 % 2,   # 通道 1 → Y 轴边
    attr['intersected'] // 4 % 2,   # 通道 2 → Z 轴边
], dim=-1).bool()
```

### 4.2 通道含义

| 通道 | 含义 | 几何意义 |
|------|------|----------|
| `intersected[:, 0]` | X 轴边 | 表面穿过 voxel 沿 X 方向的边 |
| `intersected[:, 1]` | Y 轴边 | 表面穿过 voxel 沿 Y 方向的边 |
| `intersected[:, 2]` | Z 轴边 | 表面穿过 voxel 沿 Z 方向的边 |

### 4.3 代码来源

**定义**（`flexible_dual_grid.cpp` 第148-149行）：
```cpp
// 当沿 ax2 方向扫描检测到边与三角形相交时
if (dx == 0 && dy == 0)
    intersected.back()[ax2] = true;
```

**Mesh 重建使用**（`flexible_dual_grid.py` 第173-177行）：
```python
edge_neighbor_voxel_offset = torch.tensor([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # x-axis (索引 0)
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # y-axis (索引 1)
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # z-axis (索引 2)
])
# 每条边连接 4 个相邻 voxel，形成一个 quad
```

### 4.4 与 Alpha 计算的对应

在可微渲染中，射线方向决定哪个通道的 intersected 更重要：

```
射线沿 X 方向 → 与 X 轴平行 → paraₓ 大 → sₓ 权重高（表面垂直于 X 轴时被击中）
射线沿 Y 方向 → 与 Y 轴平行 → paraᵧ 大 → sᵧ 权重高（表面垂直于 Y 轴时被击中）
射线沿 Z 方向 → 与 Z 轴平行 → paraᵤ 大 → sᵤ 权重高（表面垂直于 Z 轴时被击中）
```

**多视角梯度效果**：
- 从 X 方向看：主要给 s_x 提供梯度
- 从 Y 方向看：主要给 s_y 提供梯度
- 从 Z 方向看：主要给 s_z 提供梯度

这样不同视角可以区分"应该增大哪个边的 intersected"。

### 4.5 代码验证

**证据 1：intersected 的设置**（`flexible_dual_grid.cpp` 第96-167行）

扫描算法从三个方向检测三角形与 voxel 边的相交：

```cpp
auto scan_line_fill = [&] (const int ax2) {
    // ax2 是扫描方向：0=X, 1=Y, 2=Z
    ...
    if (dx == 0 && dy == 0)
        intersected.back()[ax2] = true;  // 从 ax2 方向扫描检测到相交
};
scan_line_fill(0);  // X 方向扫描 → intersected[0]
scan_line_fill(1);  // Y 方向扫描 → intersected[1]
scan_line_fill(2);  // Z 方向扫描 → intersected[2]
```

**证据 2：Mesh 重建使用 intersected**（第439-456行）

```cpp
// intersected[2] → 生成 xy-plane 的面（垂直于 Z 轴）
if (is_intersected[2] ...) { /* 生成 xy-plane quad */ }

// intersected[0] → 生成 yz-plane 的面（垂直于 X 轴）
if (is_intersected[0] ...) { /* 生成 yz-plane quad */ }

// intersected[1] → 生成 xz-plane 的面（垂直于 Y 轴）
if (is_intersected[1] ...) { /* 生成 xz-plane quad */ }
```

**结论**：

| intersected | 生成的面 | 面的朝向 | 正面击中的射线方向 |
|-------------|---------|---------|------------------|
| `[0] = 1` | yz-plane | 垂直于 X 轴 | 沿 X 方向（para_x 大） |
| `[1] = 1` | xz-plane | 垂直于 Y 轴 | 沿 Y 方向（para_y 大） |
| `[2] = 1` | xy-plane | 垂直于 Z 轴 | 沿 Z 方向（para_z 大） |

这验证了 Alpha 公式 `α = Σ(paraᵢ × sᵢ) / Σparaᵢ` 的合理性。

## 5. 实现细节

### 5.1 新增接口

#### Sub 模式接口

```python
# o_voxel/diff_rasterize_sub.py
class DiffRasterizeSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, attrs, alpha, voxel_size, extrinsics, intrinsics):
        """
        Sub 模式：alpha 直接传入，与视角无关。
        
        输入：
            positions: (N, 3)    # voxel 中心位置（同时用于 AABB 和 depth）
            attrs: (N, C)        # 属性（颜色等）
            alpha: (N,)          # 预计算的不透明度 sigmoid(sub_logit)
            voxel_size, extrinsics, intrinsics: 相机参数
        输出：
            color: (C, H, W), depth: (H, W), alpha: (H, W)
        """
    
    @staticmethod
    def backward(ctx, dL_dcolor, dL_ddepth, dL_dalpha):
        # 返回: dL_dpos, dL_dattr, dL_dalpha_in, None, None, None
```

**CUDA 接口**：
```cpp
std::tuple<Tensor, Tensor, Tensor, RenderState>
diff_rasterize_sub_forward(positions, attrs, alpha, voxel_size, ...);

std::tuple<Tensor, Tensor, Tensor>  // dL_dpos, dL_dattr, dL_dalpha
diff_rasterize_sub_backward(dL_dcolor, dL_ddepth, dL_dalpha, state, ...);
```

#### FDG 模式接口

```python
# o_voxel/diff_rasterize_fdg.py
class DiffRasterizeFDG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, aabb_centers, surface_positions, attrs, soft_intersected,
                voxel_size, extrinsics, intrinsics):
        """
        FDG 模式：alpha 依赖 per-pixel ray_dir，在 kernel 内计算。
        
        输入：
            aabb_centers: (N, 3)         # 用于 ray-AABB 碰撞检测
            surface_positions: (N, 3)    # 用于 depth 计算
            attrs: (N, C)                # 属性
            soft_intersected: (N, 3)     # sigmoid(intersected_logits)
            voxel_size, extrinsics, intrinsics: 相机参数
        输出：
            color: (C, H, W), depth: (H, W), alpha: (H, W)
        
        注意：Alpha 在 kernel 内通过 per-pixel ray_dir 计算
        """
    
    @staticmethod
    def backward(ctx, dL_dcolor, dL_ddepth, dL_dalpha):
        # 返回: None, dL_dsurface, dL_dattr, dL_dintersected, None, None, None
```

**CUDA 接口**：
```cpp
std::tuple<Tensor, Tensor, Tensor, RenderState>
diff_rasterize_fdg_forward(aabb_centers, surface_positions, attrs, soft_intersected, ...);

std::tuple<Tensor, Tensor, Tensor>  // dL_dsurface, dL_dattr, dL_dintersected
diff_rasterize_fdg_backward(dL_dcolor, dL_ddepth, dL_dalpha, state, ...);
```

### 5.2 接口对比总结

| 项目 | Sub | FDG |
|------|-----|-----|
| Alpha 来源 | `(N,)` 直接传入 | `(N, 3)` + ray_dir 加权 |
| Alpha 计算位置 | Python 层 | CUDA kernel 内 |
| Position | 单一 `positions` | `aabb_centers` + `surface_positions` |
| 梯度目标 | alpha, attrs, positions | soft_intersected, surface_pos, attrs |
| 共享 Compositing | ✅ `diff_compositing.cuh` | ✅ `diff_compositing.cuh` |

### 5.3 使用示例

#### Sub 模式

```python
# 展开 subdivision
positions, occupancies = expand_subdivision_to_voxels(coords, sub_logits, res)
alpha = occupancies  # 已经是 sigmoid(logit)

# 可微渲染
color, depth, alpha_out = DiffRasterizeSub.apply(
    positions, attrs, alpha, voxel_size, extrinsics, intrinsics
)

# 反向传播
loss = F.mse_loss(alpha_out, target_alpha)
loss.backward()
# → sub_logits.grad 通过 alpha → occupancies → sigmoid 获得梯度
```

#### FDG 模式

```python
# 准备输入
aabb_centers = (coords + 0.5) * voxel_size + origin
surface_pos = (coords + dual_vertices) * voxel_size + origin
soft_int = torch.sigmoid(intersected_logits)

# 可微渲染
color, depth, alpha_out = DiffRasterizeFDG.apply(
    aabb_centers, surface_pos, attrs, soft_int, voxel_size, extrinsics, intrinsics
)

# 反向传播
loss = F.mse_loss(depth, target_depth)
loss.backward()
# → dual_vertices.grad 通过 surface_pos → depth 获得梯度
# → intersected_logits.grad 通过 soft_int → alpha 获得梯度
```

## 6. CUDA 共享核心

### 6.1 分层设计

| 层 | 文件 | 功能 | 共享 |
|----|------|------|------|
| **Compositing 层** | `diff_compositing.cuh` | Alpha blending、早停、状态保存 | ✅ Sub/FDG 共享 |
| **Alpha 计算层** | `diff_alpha_fdg.cuh` | `α = Σ(para × s) / Σpara` | ❌ FDG 专用 |

### 6.2 组装方式

- **Sub Kernel**：直接使用传入的 alpha，调用共享 Compositing
- **FDG Kernel**：先调用 Alpha 计算层，再调用共享 Compositing

详细实现参考第 3 节的算法公式和第 7 节的 3DGS 代码参考。

## 7. 与 3DGS 对比

### 7.1 架构相似点

| 模块 | 3DGS 代码参考 | 可复用程度 |
|------|--------------|-----------|
| Tile-based 预处理 | `cuda_rasterizer/forward.cu` L70-140 `preprocessCUDA` | 参考思路，复用 o-voxel |
| Alpha Compositing | `cuda_rasterizer/forward.cu` L270-320 `renderCUDA` | 核心循环结构可参考 |
| 中间状态保存 | `cuda_rasterizer/forward.cu` L305-310 | 保存 `T`, `n_contrib` |
| Backward 遍历 | `cuda_rasterizer/backward.cu` L200-300 | back-to-front + atomicAdd |
| Python Binding | `rasterize_points.cu` L120-180 | autograd.Function 结构 |

### 7.2 3DGS 关键代码参考

**前向 Alpha Compositing**（`cuda_rasterizer/forward.cu`）：
```cpp
// L280-310: 核心渲染循环
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
{
    float alpha = min(0.99f, con_o.w * exp(power));  // alpha 计算
    float test_T = T * (1 - alpha);
    if (test_T < 0.0001f) { done = true; continue; }  // 早停
    
    for (int ch = 0; ch < CHANNELS; ch++)
        C[ch] += features[...] * alpha * T;  // 颜色累积
    T = test_T;
    contributor++;
}
// 保存用于 backward
n_contrib[pix_id] = contributor;
```

**反向梯度累积**（`cuda_rasterizer/backward.cu`）：
```cpp
// L220-280: 从后向前遍历
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
{
    T = T / (1.f - alpha);  // 恢复 T
    float dL_dalpha = 0.0f;
    for (int ch = 0; ch < C; ch++)
        dL_dalpha += (c[ch] - accum_rec[ch]) * dL_dout[ch] / (1.f - alpha);
    
    atomicAdd(&dL_dcolors[global_id * C + ch], dL_dcolor);  // 梯度累加
}
```

### 7.3 Alpha 计算的本质区别

| 项目 | 3DGS | 本方案 |
|------|------|--------|
| 几何原语 | 3D Gaussian | Voxel (AABB) |
| Alpha 来源 | `exp(-σ × distance)` | `Σ(para × s) / Σpara` |
| 可学习参数 | opacity, covariance | 复用 intersected_logits |
| 视角依赖 | 2D 协方差投影 | 射线-轴平行度 |