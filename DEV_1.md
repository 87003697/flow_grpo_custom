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

### 1.4 两种模式的意义与互补

#### Sub 模式：结构修改

Sub 模式让渲染 loss 的梯度能够流向 `sub_logits`，实现**稀疏结构修改**：

```
sub_logits → sigmoid → alpha → compositing → render → loss
     ↑                                              |
     └──────────────── 梯度回传 ────────────────────┘
```

**核心能力**：控制"哪里有表面"（增加/删除 voxel）

TRELLIS.2 的 Decoder 有多层，每层都输出 `sub_logits`。多分辨率 Sub 模式让各层都能接收渲染梯度，支持从粗到细的形状优化。

#### FDG 模式：几何细节

FDG 模式让梯度流向 `dual_vertices` 和 `intersected_logits`：

**核心能力**：控制"表面位置/朝向"（移动/调整表面）

#### 互补关系

| 模式 | 可微参数 | 作用 | 修改范围 |
|------|---------|------|---------|
| **Sub** | sub_logits (每层) | 修改稀疏结构 | 大形状（哪里有表面） |
| **FDG** | dual_vertices + intersected_logits | 修改几何细节 | 细节（表面位置/朝向） |

**关键洞察**：FDG 模式依赖于 Sub 模式提供的稀疏结构。如果某个区域本来就没有 voxel，FDG 无论如何调整 `dual_vertices` 都无法在那里生成表面。

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
│   ├── __init__.py                  # [修改] 添加导出
│   ├── rasterize.py                 # 原有 VoxelRenderer (不可微)
│   └── diff_rasterize.py            # [新增] Sub + FDG 模式接口
│
├── src/
│   ├── ext.cpp                      # [修改] 注册新函数
│   └── rasterize/
│       ├── rasterize.cu             # 原有
│       ├── auxiliary.h              # 原有，复用辅助函数
│       ├── config.h                 # 原有
│       └── diff_rasterize.cu        # [新增] Sub + FDG 渲染实现
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
| `src/rasterize/diff_rasterize.cu` | 新增 | Sub + FDG 渲染实现（含共享 Compositing 核心） |
| `src/ext.cpp` | 修改 | 注册 Sub/FDG 的 forward/backward |
| `o_voxel/diff_rasterize.py` | 新增 | Sub + FDG 模式 `torch.autograd.Function` |
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
# o_voxel/diff_rasterize.py
class DiffRasterizeSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, attrs, alpha, voxel_size, extrinsics, intrinsics, eps=1e-4):
        """
        Sub 模式：alpha 直接传入，与视角无关。
        
        输入：
            positions: (N, 3)    # voxel 中心位置（固定，用于 AABB 和 depth）
            attrs: (N, C)        # 属性（颜色等）
            alpha: (N,)          # 预计算的不透明度 sigmoid(sub_logit)
            voxel_size, extrinsics, intrinsics: 相机参数
            eps: float           # 数值稳定性参数，用于 1-α 的保护
        输出：
            color: (C, H, W), depth: (H, W), alpha: (H, W)
        """
        ctx.eps = eps
    
    @staticmethod
    def backward(ctx, dL_dcolor, dL_ddepth, dL_dalpha):
        eps = ctx.eps
        # 返回: None, dL_dattr, dL_dalpha_in, None, None, None, None
        #       ↑ positions 是固定的离散坐标，不需要梯度
```

**CUDA 接口**：
```cpp
std::tuple<Tensor, Tensor, Tensor, RenderState>
diff_rasterize_sub_forward(positions, attrs, alpha, voxel_size, ..., float eps = 1e-4f);

std::tuple<Tensor, Tensor>  // dL_dattr, dL_dalpha（无 dL_dpos）
diff_rasterize_sub_backward(dL_dcolor, dL_ddepth, dL_dalpha, state, ..., float eps = 1e-4f);
```

#### FDG 模式接口

```python
# o_voxel/diff_rasterize.py
class DiffRasterizeFDG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, aabb_centers, surface_positions, attrs, soft_intersected,
                voxel_size, extrinsics, intrinsics, eps=1e-4):
        """
        FDG 模式：alpha 依赖 per-pixel ray_dir，在 kernel 内计算。
        
        输入：
            aabb_centers: (N, 3)         # 用于 ray-AABB 碰撞检测
            surface_positions: (N, 3)    # 用于 depth 计算
            attrs: (N, C)                # 属性
            soft_intersected: (N, 3)     # sigmoid(intersected_logits)
            voxel_size, extrinsics, intrinsics: 相机参数
            eps: float                   # 数值稳定性参数，用于 1-α 和 Σpara 的保护
        输出：
            color: (C, H, W), depth: (H, W), alpha: (H, W)
        
        注意：Alpha 在 kernel 内通过 per-pixel ray_dir 计算
        """
        ctx.eps = eps
    
    @staticmethod
    def backward(ctx, dL_dcolor, dL_ddepth, dL_dalpha):
        eps = ctx.eps
        # 返回: None, dL_dsurface, dL_dattr, dL_dintersected, None, None, None, None
```

**CUDA 接口**：
```cpp
std::tuple<Tensor, Tensor, Tensor, RenderState>
diff_rasterize_fdg_forward(aabb_centers, surface_positions, attrs, soft_intersected, ..., float eps = 1e-4f);

std::tuple<Tensor, Tensor, Tensor>  // dL_dsurface, dL_dattr, dL_dintersected
diff_rasterize_fdg_backward(dL_dcolor, dL_ddepth, dL_dalpha, state, ..., float eps = 1e-4f);
```

### 5.2 接口对比总结

| 项目 | Sub | FDG |
|------|-----|-----|
| Alpha 来源 | `(N,)` 直接传入 | `(N, 3)` + ray_dir 加权 |
| Alpha 计算位置 | Python 层 | CUDA kernel 内 |
| Position | 单一 `positions`（固定） | `aabb_centers` + `surface_positions`（可微） |
| 梯度目标 | alpha, attrs | soft_intersected, surface_pos, attrs |
| 共享 Compositing | ✅ `diff_rasterize.cu` 内 | ✅ `diff_rasterize.cu` 内 |

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

## 8. 方案合理性验证

### 8.1 Alpha 公式验证

**公式**：`α = Σᵢ (paraᵢ × sᵢ) / (Σᵢ paraᵢ + eps)`

| 场景 | 验证结果 |
|------|---------|
| 单边相交 `(1,0,0)` | ✅ 正面射线 α=1，侧面射线 α=0 |
| 多边相交 `(1,1,0)` | ✅ 加权平均，符合几何直觉 |

### 8.2 Depth 排序一致性

**问题**：排序用 `aabb_center` 的 view-space Z，depth 计算用 `surface_pos`

**代码参考**：

| 位置 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 排序深度 | `o-voxel/src/rasterize/rasterize.cu` | L59 | `depths[idx] = p_view.z` |
| QEF 边界定义 | `o-voxel/src/convert/flexible_dual_grid.cpp` | L567-576 | `min_corner = coord * voxel_size` |
| 边界约束检查 | `o-voxel/src/convert/flexible_dual_grid.cpp` | L597-601 | 检查解是否在 voxel 内 |
| 约束求解 | `o-voxel/src/convert/flexible_dual_grid.cpp` | L602-761 | 超出边界时枚举约束解 |

**分析**：
- GT `dual_vertices` 被约束在 voxel 边界内（QEF 求解有边界约束）
- 预测值经过 sigmoid 变换，收敛到相似范围
- `surface_pos` 相对于 `aabb_center` 的偏移 ≤ 0.5 voxel_size

**结论**：偏移量不足以导致相邻 voxel 排序反转，使用原始排序方法是安全的。

### 8.3 数值稳定性

通过 `eps` 参数（默认 `1e-4`）保护所有可能的除零位置：

```cpp
// Alpha 计算 (FDG)
float alpha = (para_x * s_x + para_y * s_y + para_z * s_z) / (sum_para + eps);

// Forward: 透过率更新
float test_T = T * (1.0f - alpha + eps);

// Backward: 恢复 T 和梯度计算
T = T / (1.0f - alpha + eps);
float dL_dalpha = ... / (1.0f - alpha + eps);
```

### 8.4 内存与性能

#### 中间状态存储

采用 3DGS 的策略，只保存最小必要信息：

| 保存内容 | 大小 | 说明 |
|---------|------|------|
| `n_contrib[H×W]` | H×W×4 bytes | 每像素参与的 voxel 数量 |
| `T_final[H×W]` | H×W×4 bytes | 每像素最终透过率 |

**总内存**：1024×1024 分辨率约 8MB

**反向时**：重新遍历 tile 内的排序列表，用 `n_contrib` 作为终止条件，无需保存 voxel 索引。

#### atomicAdd 竞争

多个像素可能同时更新同一个 voxel 的梯度：

```cpp
atomicAdd(&dL_dattr[voxel_id * C + ch], dL_dattr_local);
```

**评估**：TRELLIS.2 可达 1024³ 分辨率，表面 voxel 数量可能达百万级。但由于 voxel 很小，每个 voxel 覆盖像素有限，单 voxel 竞争不严重；主要开销是总写入次数多。

**策略**：先直接使用 atomicAdd，如有性能瓶颈再优化（warp-level 合并或 shared memory 累加）。

### 8.5 训练信号覆盖

#### 无 GT Mesh 的约束

本方案基于预训练权重进行后训练，无 GT mesh 用于 BCE 直接监督。

**代码参考**（`edit4shape/systems/trellis2_shape.py`）：

| 行号 | 说明 |
|------|------|
| L6 | 核心流程：`图像条件 -> ... -> Normal 渲染 -> Guidance Loss` |
| L1547-1549 | Loss 组成：`ssim + lpips + latent_mse + reg`，无 BCE |
| L1588 | Guidance 计算：`system.guidance.compute_guidance(shape_normal, ...)` |

**梯度来源**：仅依赖渲染 loss，无直接 intersected 监督。

**缓解策略**：
- **多视角采样**：编辑时采样多样视角，确保各方向 intersected 通道都能获得梯度
- **预训练初始化**：logits 已收敛到合理范围（通常 [-3, 3]），sigmoid 饱和不是问题

#### 早停与遮挡

早停阈值 `T < 1e-4` 后的 voxel 不参与反向传播。

**参考**（3DGS `cuda_rasterizer/forward.cu` L481）：
```cpp
if (test_T < 0.0001f) { done = true; continue; }
```

**影响评估**：
- 被完全遮挡的 voxel 无渲染梯度（T 累积透过率 < 0.01%）
- 多视角训练可缓解：不同视角下遮挡关系不同，同一 voxel 在其他视角可能可见

#### 多视角梯度平衡

**代码参考**（`edit4shape/datasets/trellis.py`）：

| 行号 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| L291-292 | `yaw_range` | [0°, 360°] | 全方位均匀采样 |
| L293-294 | `pitch_range` | [-15°, 45°] | 偏向上半球 |

**对 intersected 通道的影响**：

| 通道 | 主要梯度来源 | 覆盖情况 |
|------|-------------|---------|
| `s_x` | yaw ≈ 90°/270° | ✅ 全覆盖 |
| `s_y` | yaw ≈ 0°/180° | ✅ 全覆盖 |
| `s_z` | pitch ≈ ±90° | ⚠️ 梯度较弱 |

**建议**：如观察到顶面/底面几何效果差，可增加俯视角度（pitch > 60°）或在编辑阶段额外采样极端视角。