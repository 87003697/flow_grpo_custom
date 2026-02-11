# 三阶段 Autograd 架构设计

## 核心思想

将 training step 的梯度计算拆为三个阶段，使 rollout / decoder+renderer / rollout backward 的计算图**互不叠加**，
任意时刻只有一个阶段的计算图驻留显存。

```
当前（单阶段）:
  loss → renderer → decoder → rollout (12 steps) → flow model θ
  ─────────── 整个 autograd graph 同时驻留显存 ───────────

三阶段:
  Phase 1:  rollout no_grad → shape_slat（零计算图）
  Phase 2:  proxy → decoder → renderer → guidance → loss.backward()
            → proxy.grad = ∂L/∂slat → 释放 decode/render graph
  Phase 3:  逐步 forward（detach x_t）+ 即时 backward
            → 每步: f_θ(x_t.detach(), t) → step_loss.backward() → θ.grad +=
            → 无跨步 autograd 依赖，显存 O(1)
```

数学近似（一阶展开，detach x_t）：

Euler 展开：$\text{slat} = x_0 + \sum_t \Delta t_t \cdot v_t$，其中 $v_t = f_\theta(x_t, t)$。

完整链式法则：$\frac{d\,\text{slat}}{d\,\theta} = \sum_t \Delta t_t \left(\frac{\partial v_t}{\partial \theta} + \frac{\partial v_t}{\partial x_t}\frac{dx_t}{d\theta}\right)$

Detach $x_t$ 后（丢弃跨步 Jacobian $\frac{\partial v_t}{\partial x_t}\frac{dx_t}{d\theta}$）：

$$\frac{\partial L}{\partial \theta} \approx \underbrace{\frac{\partial L}{\partial \text{slat}}}_{\text{Phase 2: slat\_grad}} \cdot \sum_t \Delta t_t \cdot \underbrace{\frac{\partial v_t}{\partial \theta}\bigg|_{x_t=\text{const}}}_{\text{Phase 3: 逐步即时 backward}}$$

等价形式（实际实现）：$\theta.\text{grad} \mathrel{+}= \sum_t \nabla_\theta \left[\text{slat\_grad}^T \cdot \Delta t_t \cdot v_t\right]$

这与流匹配原始训练（$x_t$ 是采样得到的常量）和 SDS/VSD 使用的近似一致。
当步数较多（$\Delta t$ 小）时，被丢弃的二阶项 $O(\Delta t^2)$ 可忽略。

## Phase 原语

### Phase 1: Rollout no_grad
- 输入: state (coords, cond_emb), generator seed
- 操作: `torch.no_grad()` 下跑完整去噪 → shape_slat
- 输出: shape_slat (SparseTensor, 无 autograd 图)
- 显存: 零额外图开销

### Phase 2a: Decode + Render（train GPU）
- 输入: shape_slat (detached), cameras
- 操作:
  1. `proxy_feats = slat.feats.detach().clone().requires_grad_(True)`
  2. `proxy_slat → decoder → mesh → renderer → normals`
  3. 挂载 state 可视化数据（detach）
- 输出: Phase2aResult (proxy_feats, normals, render_out)
- 显存: decode/render 前向图驻留（等待 backward）

### Phase 2b: Guidance Backward + 提取梯度
- 输入: Phase2aResult, GuidanceResult
- 操作:
  1. `loss = guidance_result.loss * weight`
  2. `accelerator.backward(loss)` → 填充 `proxy_feats.grad`
  3. `slat_grad = proxy_feats.grad.clone()`
  4. 构建日志
  5. 释放 decode/render graph + `empty_cache()`
- 输出: Phase2bResult (slat_grad, logs, guidance_loss_scalar)
- 显存: decode/render graph → backward 后释放

### Phase 3: 逐步 Forward + 即时 Backward（detach x_t）
- 输入: state, generator seed (同 Phase 1), Phase2bResult (slat_grad)
- 操作:
  1. 用相同 seed 初始化 x_t（与 Phase 1 一致的初始噪声）
  2. 逐步循环（T 步）：
     a. `x_t_det = x_t.detach()` — 切断跨步梯度链
     b. `v_t = f_θ(x_t_det, t, cond_emb)` — 仅对 θ 有梯度
     c. CFG: `velocity = cfg_mix(v_t, uncond_v)` （uncond 无梯度）
     d. `step_loss = (slat_grad * dt * velocity.feats).sum()`
     e. `step_loss.backward()` — **即时** 累积 θ.grad，本步计算图立即释放
     f. 正则化: `(λ * reg_loss_t).backward()` （如果启用）
     g. `x_t = scheduler.step(velocity.detach(), step_idx, x_t)` — 无梯度推进轨迹
  3. `empty_cache()`
- 输出: flow model 参数梯度 (已累积在 θ.grad), 完整日志字典
- 显存: **永远只有 1 步的激活驻留**（O(1)，不随步数增长）
- 不需要 step-level gradient checkpoint

### 确定性保证
- Phase 1 和 Phase 3 使用**相同 generator seed** → 相同初始噪声
- Phase 3 中 `x_t` 的推进使用 `velocity.detach()`，轨迹更新是纯数值运算
- 因此 Phase 3 的 x_t 序列与 Phase 1 **数值一致**（同样的 seed + 同样的数值路径）

## 显存对比

| 时刻 | 原方案 (单阶段) | 三阶段 |
|---|---|---|
| Rollout 中 | rollout graph (T 步叠加) | **零** (Phase 1 no_grad) |
| Decode+Render 中 | rollout + decoder + renderer | **仅 decoder + renderer** |
| Phase 3 中 | — | **仅 1 步激活** (detach x_t, O(1)) |
| 峰值 | ~3 个图叠加 | **单个图（Phase 2 的 decode+render 图）** |

## 代码架构：调用栈

### 数据结构

```python
@dataclass
class TrainContext:
    """训练上下文：预计算的配置和组件引用，消灭 Phase 函数中的冗长参数。"""
    system: Trellis2System
    cfg: ml_collections.ConfigDict
    accelerator: Accelerator
    device: torch.device
    pipeline: Any                  # Trellis2RefAdapter
    ss_params: Dict[str, Any]      # dense sampling 参数
    ss_resolution: int             # dense sampling 分辨率
    flow_res: int                  # flow model 分辨率
    target_res: int                # decoder 输出分辨率
    normal_mode: str               # "mesh" | "hybrid26"
    reg_weight: float              # 正则化权重
    guidance_weight: float         # guidance 权重

@dataclass
class Phase2aResult:
    """Phase 2a: decode+render 的输出（尚未 backward）。"""
    proxy_feats: torch.Tensor       # (N, C), requires_grad=True 的叶变量
    normals: torch.Tensor           # (B, V, H, W, 3), 渲染输出（有 autograd 图）
    render_out: Dict[str, Any]      # subs, meshes 等

@dataclass
class Phase2bResult:
    """Phase 2b: guidance backward 的输出。"""
    slat_grad: torch.Tensor         # (N, C), ∂L/∂slat
    logs: Dict[str, Any]            # guidance loss 细分
    guidance_loss_scalar: float     # 加权后的 guidance loss 标量
```

### 调用栈

```
main()
├── 初始化 (env, accelerator, dataloaders, system, ckpt)
├── 构建 TrainContext（循环外一次性初始化）
│
├── evaluate()                                   # eval 分支（不变）
│
└── 训练循环
    └── three_phase_shape_step(state, ctx, global_step) → Dict[str, Any]
        │
        ├── dense_sampling_no_grad(state, ctx)
        │   └── pipeline.dense_sampling()                 # no_grad
        │
        ├── phase1_rollout_no_grad(state, ctx, gen_seed)
        │   └── rollout_shape(is_training=False)          # no_grad
        │
        ├── phase2a_decode_render(state, ctx) → Phase2aResult
        │   ├── _create_proxy(slat) → proxy_feats, proxy_slat
        │   ├── decode_and_render_normal(proxy_slat, ...)
        │   └── 挂载 state 可视化数据
        │
        ├── compute_guidance_sync(state, ctx, p2a) → GuidanceResult
        │   └── guidance.compute_guidance(normals, ...)
        │
        ├── phase2b_guidance_backward(state, ctx, p2a, guidance) → Phase2bResult
        │   ├── accelerator.backward(loss)
        │   ├── slat_grad = proxy_feats.grad.clone()
        │   └── del + empty_cache()
        │
        └── phase3_rollout_grad_backward(state, ctx, gen_seed, p2b) → logs
            ├── x_t = init_latents(same seed)
            └── for step_idx in steps:              # 逐步即时 backward
                ├── v = f_θ(x_t.detach(), t)        # 仅对 θ 有梯度
                ├── velocity = cfg_mix(v, uncond_v)
                ├── (slat_grad * dt * velocity).sum().backward()  # 即时释放
                ├── (λ * reg_loss_t).backward()     # 正则化（可选）
                └── x_t = scheduler.step(velocity.detach(), ...)  # 无梯度推进
```

### Phase 函数签名

```python
def dense_sampling_no_grad(state: Trellis2State, ctx: TrainContext) -> None:
    """Dense Sampling（no_grad）。填充 state.coords。"""

def phase1_rollout_no_grad(
    state: Trellis2State, ctx: TrainContext, gen_seed: int,
) -> None:
    """Phase 1: 无梯度 rollout。填充 state.features.shape_slat。"""

def phase2a_decode_render(
    state: Trellis2State, ctx: TrainContext,
) -> Phase2aResult:
    """Phase 2a: proxy → decode → render。不执行 guidance，允许异步插入。"""

def phase2b_guidance_backward(
    state: Trellis2State, ctx: TrainContext,
    p2a: Phase2aResult, guidance_result: GuidanceResult,
) -> Phase2bResult:
    """Phase 2b: guidance loss → backward → 提取 slat_grad → 释放图。"""

def phase3_rollout_grad_backward(
    state: Trellis2State, ctx: TrainContext,
    gen_seed: int, p2b: Phase2bResult,
) -> Dict[str, Any]:
    """Phase 3: 逐步 forward（detach x_t）+ 即时 backward。θ.grad 逐步累积。"""

def compute_guidance_sync(
    state: Trellis2State, ctx: TrainContext, p2a: Phase2aResult,
) -> GuidanceResult:
    """同步 guidance：阻塞式计算。"""
```

### 编排函数

```python
def three_phase_shape_step(state, ctx, global_step) -> Dict[str, Any]:
    gen_seed = int(ctx.cfg.seed) + global_step
    dense_sampling_no_grad(state, ctx)
    phase1_rollout_no_grad(state, ctx, gen_seed)
    p2a = phase2a_decode_render(state, ctx)
    guidance = compute_guidance_sync(state, ctx, p2a)
    p2b = phase2b_guidance_backward(state, ctx, p2a, guidance)
    return phase3_rollout_grad_backward(state, ctx, gen_seed, p2b)
```

### 训练循环（~20 行）

```python
ctx = TrainContext(system=system, cfg=cfg, accelerator=accelerator, ...)

for epoch in range(start_epoch, int(cfg.num_epochs)):
    train_loader.sampler.set_epoch(epoch)
    for batch in train_loader:
        global_step += 1
        state = Trellis2State()
        state.attach_batch(batch, pipeline=ctx.pipeline, ...)
        with accelerator.accumulate(system.shape.model):
            with TrainModeGuard(system.shape.model):
                shape_log = three_phase_shape_step(state, ctx, global_step)
            if accelerator.sync_gradients:
                system.shape.optimizer.step()
                system.shape.optimizer.zero_grad()
        shape_logger.log_step(shape_log, ...)
        del state, shape_log; torch.cuda.empty_cache()
```

## 异步 Guidance：跨 GPU Proxy 技巧

### 核心思路

在 **comp_rgb（normals）层面再做一次 proxy**，使 train GPU 和 guidance GPU 的 autograd 图完全解耦，
从而实现真正的异步并行。与三阶段在 slat 层面做 proxy 是同一个思想，用了两次：

| 层级 | proxy 位置 | 目的 |
|---|---|---|
| 三阶段 Phase 2/3 | slat 层 | 解耦 rollout 和 decoder 的 autograd 图 → **显存隔离** |
| 异步 guidance | comp_rgb 层 | 解耦 train GPU 和 guidance GPU 的 autograd 图 → **计算并行** |

```
当前（单 autograd 图横跨两个 GPU）:

Train GPU                                    Guidance GPU
─────────                                    ────────────
proxy_slat → decoder → renderer → normals ─→ encode → pipeline → loss
                                          ↑
                              autograd 图跨两个 GPU（无法并行）


改进（两个 GPU 各自独立的 autograd 图）:

Train GPU                                    Guidance GPU
─────────                                    ────────────
proxy_slat → decoder → renderer → normals    rgb_proxy → encode → pipeline → loss
                                  │                ↑                           │
                                  │     .detach().to(guidance).requires_grad_  │
                                  │                                  loss.backward()
                                  │                                           │
                                  ◄── rgb_grad = rgb_proxy.grad.to(train) ────┘
                                  │
                     normals.backward(rgb_grad)
                                  │
                     proxy_feats.grad = ∂L/∂slat ✓
```

数学等价性：
$$\frac{\partial L}{\partial \text{proxy\_feats}} = \underbrace{\frac{\partial L}{\partial \text{normals}}}_{\text{rgb\_grad（guidance GPU 算出）}} \cdot \underbrace{\frac{\partial \text{normals}}{\partial \text{proxy\_feats}}}_{\text{train GPU backward}}$$

### 改造 PipelineParallelMixin

```python
@dataclass
class AsyncGuidanceResult:
    """异步 guidance 结果：梯度 + 标量日志。"""
    rgb_grad: torch.Tensor          # guidance GPU 上的 ∂L/∂rgb
    loss_scalar: float
    loss_dict: Dict[str, float]

class PipelineParallelMixin:
    def submit_async(self, comp_rgb, condition_images, **kwargs):
        """
        真异步提交：
        1. comp_rgb → detach → copy to guidance GPU → requires_grad
        2. 在 guidance stream 上执行 compute_guidance + backward
        3. 记录 Event，立即返回（不阻塞 train GPU）
        """
        rgb_proxy = comp_rgb.detach().to(self.device, non_blocking=True).requires_grad_(True)
        stream = self._pp_streams[self._pp_slot_counter % self._pp_num_streams]

        with torch.cuda.stream(stream):
            result = self.compute_guidance(rgb_proxy, condition_images, **kwargs)
            result.loss.backward()                # backward 全部在 guidance GPU
            rgb_grad = rgb_proxy.grad.clone()

        done_event = torch.cuda.Event()
        done_event.record(stream)

        self._pp_queue.append({
            "rgb_grad": rgb_grad,
            "event": done_event,
            "loss_scalar": result.loss.item(),
            "loss_dict": {k: v.item() for k, v in (result.loss_dict or {}).items()},
        })
        self._pp_slot_counter += 1

    def wait_and_get(self, target_device=None) -> AsyncGuidanceResult:
        """等待最早提交的结果，将梯度搬回 train GPU。"""
        slot = self._pp_queue.popleft()
        slot["event"].synchronize()
        rgb_grad = slot["rgb_grad"]
        if target_device is not None:
            rgb_grad = rgb_grad.to(target_device)
        return AsyncGuidanceResult(
            rgb_grad=rgb_grad,
            loss_scalar=slot["loss_scalar"],
            loss_dict=slot["loss_dict"],
        )
```

### 异步版 Phase 2 编排（Shape+Tex 示例）

```python
# Shape Phase 2a: decode + render（train GPU）
shape_p2a = phase2a_decode_render(state, ctx_shape)

# ★ 异步提交 shape guidance（立即返回，不阻塞）
guidance.submit_async(shape_p2a.normals, condition_images)

# ======== 插入其他工作（与 shape guidance 并行）========
phase1_rollout_no_grad(state, ctx_tex, gen_seed_tex)  # tex rollout
tex_p2a = phase2a_decode_render(state, ctx_tex)        # tex decode + render ← 也在 guidance 等待期间完成!

# ★ 等待 shape guidance 完成，取回 rgb_grad
async_result = guidance.wait_and_get(target_device=device)

# Shape Phase 2b: 用 rgb_grad 在 train GPU 上 backward
shape_p2a.normals.backward(async_result.rgb_grad)
slat_grad = shape_p2a.proxy_feats.grad.detach().clone()
```

### 时序对比（Shape+Tex）

```
同步（guidance 阻塞）:                          异步（guidance 并行）:
 T0  shape decode+render                        T0  shape decode+render
 T1  |等待| shape guidance                       T1  tex rollout + decode + render
 T2  shape backward                                  ←→ shape guidance (并行!)
 T3  tex rollout                                 T2  shape backward
 T4  tex decode+render                           总耗时: T0 + max(T1_rollout+render, T_guidance) + T2
 总耗时: T0 + T1 + T2 + T3 + T4                  省掉了 guidance 等待 + tex P1+P2a 时间
```

## 文件组织

```
edit4shape/systems/
├── trellis2_shape.py              # 原版 Shape-only（单阶段，保留不动）
├── trellis2_shape_autograd.py     # ★ Shape-only 三阶段版本
├── trellis2_tex.py                # 原版 Tex-only（单阶段，保留不动）
├── trellis2_tex_autograd.py       # ★ Tex-only 三阶段版本
├── trellis2_shape+tex.py          # 原版 Shape+Tex（单阶段）
└── trellis2_shape+tex_autograd.py # Shape+Tex 三阶段 + 异步 guidance
```

## 扩展路线

### Step 1: Shape-only 同步 ✅
```python
Phase 1 → Phase 2 (sync guidance) → Phase 3 → shape_optimizer.step()
```
文件: `trellis2_shape_autograd.py`

### Step 2: Tex-only 同步
```python
# Shape rollout (no_grad, 提供 mesh + subs 给 Tex)
shape_rollout_no_grad → shape_decode (no_grad, 获取 mesh/subs)

# Tex 三阶段
Phase 1: tex_rollout_no_grad → tex_slat
Phase 2: proxy → tex_decode → PBR render → guidance → backward → tex_proxy.grad
Phase 3: tex_rollout_with_grad → backward(tex_proxy.grad) → tex_optimizer.step()
```
文件: `trellis2_tex_autograd.py`

注意: Tex-only 训练中 Shape 阶段只提供 mesh 和 subs 作为几何条件，
Shape flow model 不训练，所有 Shape 相关计算都在 no_grad 下进行。

### Step 3: Shape+Tex 同步
```python
# Shape 三阶段
Phase 1 → Phase 2 (sync) → Phase 3 → shape_optimizer.step()
# Tex 三阶段
Phase 1 → Phase 2 (sync) → Phase 3 → tex_optimizer.step()
```
文件: `trellis2_shape+tex_autograd.py`

### Step 4: Shape+Tex 异步 Guidance（流水线并行）

```
Train GPU (cuda:0)                         Guidance GPU (cuda:1)
──────────────────                         ──────────────────
T0  shape: dense_sampling
T1  shape: Phase 1 (rollout no_grad)
T2  shape: Phase 2a (decode+render)
T3  ───── submit_async(normals) ─────────► shape guidance 计算中...
T4  tex: Phase 1 + Phase 2a               ← rollout + decode + render 全部
    (rollout + decode + render)              与 shape guidance 重叠！
T5  ◄───── wait_and_get() ───────────────  shape guidance 完成, rgb_grad
T6  shape: Phase 2b (normals.backward(rgb_grad))
T7  shape: Phase 3 (rollout+backward)
T8  shape: optimizer.step()
T9  ───── submit_async(tex rgb) ─────────► tex guidance 计算中...
T10 next batch: prefetch + dense_sampling   ← 与 tex guidance 重叠！
    + shape Phase 1 (weights_{N+1} 已就绪)
T11 ◄───── wait_and_get() ───────────────  tex guidance 完成, rgb_grad
T12 tex: Phase 2b (rgb.backward(rgb_grad))
T13 tex: Phase 3 (rollout+backward)
T14 tex: optimizer.step()
```

注意：T4 中 tex P2a 需要 shape mesh/subs 作为几何条件，
这些在 shape P2a (T2) 中已经产出并 detach 存于 state，安全可用。

文件: `trellis2_shape+tex_autograd.py`（异步模式开关）

### 异步编排伪代码

```python
def async_shape_tex_step(state, ctx_s, ctx_t, global_step):
    guidance = ctx_s.system.guidance  # PP 版本

    dense_sampling_no_grad(state, ctx_s)

    # Shape Phase 1 + 2a
    phase1_rollout_no_grad(state, ctx_s, gen_seed_s)
    shape_p2a = phase2a_decode_render(state, ctx_s)
    guidance.submit_async(shape_p2a.normals, condition_images)  # 非阻塞

    # ★ Tex Phase 1 + 2a（与 shape guidance 并行！）
    phase1_rollout_no_grad(state, ctx_t, gen_seed_t)            # tex rollout
    tex_p2a = phase2a_decode_render(state, ctx_t)               # tex decode + render

    # Shape: wait guidance → Phase 2b + 3
    shape_async = guidance.wait_and_get(target_device=device)
    shape_p2a.normals.backward(shape_async.rgb_grad)
    shape_p2b = _extract_slat_grad_and_cleanup(shape_p2a, shape_async)
    shape_log = phase3_rollout_grad_backward(state, ctx_s, gen_seed_s, shape_p2b)
    ctx_s.system.shape.optimizer.step()

    # Tex: submit guidance（P2a 已在上面完成）
    guidance.submit_async(tex_p2a.rendered_rgb, condition_images)  # 非阻塞

    # ★ 可选：下一 batch 预取 / dense_sampling / shape Phase 1（与 tex guidance 并行）
    # next_state = prepare_next_batch(...)  # 与 tex guidance 重叠

    # Tex: wait guidance → Phase 2b + 3
    tex_async = guidance.wait_and_get(target_device=device)
    tex_p2a.rendered_rgb.backward(tex_async.rgb_grad)
    tex_p2b = _extract_slat_grad_and_cleanup(tex_p2a, tex_async)
    tex_log = phase3_rollout_grad_backward(state, ctx_t, gen_seed_t, tex_p2b)
    ctx_t.system.tex.optimizer.step()

    return shape_log, tex_log
```

## 异步收益分析

### 各操作典型耗时（估算）

| 操作 | GPU | 耗时 | 说明 |
|---|---|---|---|
| Dense Sampling | train | ~1-2s | 数十步小模型推理 |
| Phase 1: Rollout no_grad | train | ~2-5s | 12-40 步 flow model forward，无 autograd |
| Phase 2a: Decode + Render | train | ~1-3s | chunked decoder + nvdiffrast |
| **Guidance 计算** | **guidance** | **~5-15s** | VAE encode + FlowEdit pipeline (20-40步) + loss + backward |
| Phase 2b: Backward | train | ~1-3s | renderer + decoder 的 backward |
| Phase 3: Rollout 逐步 bwd | train | ~3-7s | 12-40 步 forward + 即时 backward（无 ckpt 重算开销） |
| Optimizer step | train | ~0.01s | 几乎瞬时 |

**Guidance 是最耗时的单项操作**（~5-15s），跑在 guidance GPU 上。同步模式下 train GPU 在此期间**完全空闲**。

### 场景 1: Shape-only, grad_accum=1

```
Batch N:
  P1 → P2a → [submit guidance] → 空闲(~10s) → [wait] → P2b → P3 → opt.step → Batch N+1

约束: Phase 1 和 Phase 3 必须用**相同权重**（确定性保证）。
      opt.step 在 guidance 等待之后，所以 Batch N+1 的 P1 不能提前开始。

空闲期间可做的工作:
  ✅ 下一 batch 数据加载 (DataLoader prefetch)
  ✅ dense_sampling（frozen sampler，不依赖 flow model 权重）
  ✅ 条件编码（frozen image encoder）
  ❌ 下一 batch Phase 1（权重未更新）
  ❌ 下一 batch Phase 2a（P1 未完成，无 slat）

时序:
  Train GPU  ████ P1+P2a ░░░ prefetch+dense+cond_enc ░░░ ████ P2b+P3+opt █
  Guid GPU                ██████████ guidance ██████████

  省时: ~2-3s / ~10s 空闲 ≈ 25% 利用（收益有限）
```

### 场景 2: Shape-only, grad_accum > 1 ✅ 大幅收益

当 `gradient_accumulation_steps > 1` 时，**多个 micro-batch 之间权重不变**（optimizer.step 在所有 micro-batch 之后才执行）。
因此 micro-batch i 的 guidance 等待期间可以安全执行 micro-batch i+1 的 **Phase 1 + Phase 2a（rollout + decode + render）**。

```
核心流水线:
  micro 0: P1₀ → P2a₀ → [submit₀] ──────────────────────────► [wait₀] → P2b₀ → P3₀
                                    P1₁ → P2a₁ → [submit₁]    ↑ guidance₀ 完成
                                    ↑ 权重没变，安全！           ↑ P2a₁ 也完成，
                                    ↑ P1 + decode + render       直接可用！
                                      全部在 guidance 期间完成

  micro 1:                                      ──────────────► [wait₁] → P2b₁ → P3₁
                                                                ↑ guidance₁ 完成

Guid GPU:                           [guid₀...........] [guid₁...........]

时序图 (grad_accum=2):
  Train GPU  ████ P1₀+P2a₀ ████ P1₁+P2a₁ ████ P2b₀+P3₀ ████ P2b₁+P3₁ █ opt.step
  Guid GPU                 ██████ guid₀ ██████ ██████ guid₁ ██████

  Guidance 等待完全被 P1+P2a 填满（rollout ~5s + decode+render ~3s ≈ 8s ≤ guidance ~10s）
  → 接近零空闲！
```

显存注意：两个 micro-batch 的 P2a 前向图可能短暂共存。
micro i 的 P2a 图在 P2b backward 后立即释放，不会累积。

编排伪代码（Shape-only, grad_accum > 1）：

```python
def async_shape_step_accum(states, ctx, global_step):
    """支持 micro-batch 级流水线的 shape-only 训练步。"""
    n = len(states)  # micro-batch 数量 = grad_accum_steps
    p2a_results = [None] * n

    for i in range(n):
        gen_seed_i = int(ctx.cfg.seed) + global_step * n + i
        dense_sampling_no_grad(states[i], ctx)

        # ★ Phase 1 + Phase 2a：rollout + decode + render（全部在 guidance 等待期间完成）
        phase1_rollout_no_grad(states[i], ctx, gen_seed_i)
        p2a_results[i] = phase2a_decode_render(states[i], ctx)
        guidance.submit_async(p2a_results[i].normals, states[i].condition_images)  # 非阻塞

        # 如果有前一个 micro-batch 的 guidance 结果，取回并完成 P2b + P3
        if i > 0:
            prev = i - 1
            async_result = guidance.wait_and_get(target_device=device)
            p2a_results[prev].normals.backward(async_result.rgb_grad)
            p2b = _extract_slat_grad_and_cleanup(p2a_results[prev], async_result)
            phase3_rollout_grad_backward(states[prev], ctx, gen_seed_prev, p2b)

    # 处理最后一个 micro-batch
    async_result = guidance.wait_and_get(target_device=device)
    p2a_results[-1].normals.backward(async_result.rgb_grad)
    p2b = _extract_slat_grad_and_cleanup(p2a_results[-1], async_result)
    phase3_rollout_grad_backward(states[-1], ctx, gen_seed_last, p2b)
```

### 场景 3: Tex-only, grad_accum=1

```
Tex-only 训练中 Shape 完全 frozen（不训练）。
→ Shape 的 rollout + decode 可以为任意 batch 提前执行（权重永远不变）。

Batch N:
  shape_decode(no_grad) → tex P1 → tex P2a → [submit]
  → next batch: shape_decode(no_grad) + dense_sampling   ← shape frozen，安全！
  → [wait] → tex P2b → tex P3 → opt.step

时序:
  Train GPU  ██ S:dec ██ T:P1+P2a ████ next:S:dec+dense ████ T:P2b+P3+opt █
  Guid GPU                        ████████ T:guid ████████

  Shape frozen → 下一 batch 的 shape decode 可以提前执行
  Guidance 等待期间 ~4-5s 有用工作 / ~10s ≈ 45% 利用
```

### 场景 4: Tex-only, grad_accum > 1 ✅ 大幅收益

和 Shape-only accum>1 类似，micro-batch 间权重不变（tex 权重不变 + shape frozen），
guidance 等待期间可以执行下一个 micro-batch 的 **shape decode + tex P1 + tex P2a**。

```
  micro 0: S:dec₀ → T:P1₀ → T:P2a₀ → [submit₀]
           ──────────────────────────────────────► guidance₀ 计算中...
           S:dec₁ → T:P1₁ → T:P2a₁ → [submit₁]  ← 全部与 guidance₀ 重叠!
                                        [wait₀] → T:P2b₀ → T:P3₀
  micro 1:                                       ──────────────► [wait₁] → ...

时序:
  Train GPU  ██ S+T:P1₀+P2a₀ ██ S+T:P1₁+P2a₁ ██ T:P2b₀+P3₀ ██ T:P2b₁+P3₁ █ opt
  Guid GPU                    ██████ guid₀ ██████ ██████ guid₁ ██████

  接近零空闲！
```

### 场景 5: Shape+Tex, grad_accum=1 ✅ 天然适合异步

Shape+Tex 训练天然具有异步插入窗口：

- **Shape guidance 等待期间** → 执行 Tex **Phase 1 + Phase 2a**（rollout + decode + render 全部完成）
- **Tex guidance 等待期间** → 执行下一 batch 的 prefetch + Dense Sampling + Shape Phase 1（shape optimizer 已 step）

```
时序图 (Shape+Tex 异步):
  Train GPU  ██ S:P1+P2a ████ T:P1+P2a ████ S:P2b+P3+opt ██ T:submit ░░ T:P2b+P3+opt ██
  Guid GPU                ██████ S:guid ██████              ██████ T:guid ██████

  S=Shape, T=Tex

  Shape guidance (~10s) 被 Tex P1+P2a (~8s) 充分填充
  Tex guidance (~10s) 期间做下一 batch 预取 + dense_sampling + shape P1

  节省时间: ~15-20s / step (两次 guidance 等待基本消除)
```

进一步优化（跨 batch 流水线）:

```
Batch N                                              Batch N+1
──────                                               ──────────

Train GPU:
 ┌──────────────── Batch N ──────────────────────┐┌─── Batch N+1 ─────
 │ S:P1+P2a  T:P1+P2a  S:P2b+P3+opt  T:P2b+P3  ││ prefetch+dense+S:P1
 └──↓──────────────────────────────────────↓─────┘└────────────────────
     ↓submit                               ↓submit
 Guid GPU:
     █████ S:guid █████          █████ T:guid █████
```

### 场景 6: Shape+Tex, grad_accum > 1 ✅ 极致流水线

双重流水线：阶段间流水 + micro-batch 间流水。

```
  micro 0: S:P1₀+P2a₀ → [submit_S₀] → T:P1₀+P2a₀ → [wait_S₀] → S:P2b₀+P3₀
                                                       → [submit_T₀] → micro 1: S:P1₁+P2a₁ → ...
                                                                        [wait_T₀] → T:P2b₀+P3₀

  Guid GPU: [S:guid₀.......] [T:guid₀.......] [S:guid₁.......] ...

  几乎每一秒都在做有用计算！
```

### 收益总结

| 场景 | 异步收益 | guidance 等待利用率 | 空闲期间工作 |
|---|---|---|---|
| Shape-only, accum=1 | 低 | ~25% | prefetch + dense_sampling + cond_enc |
| Shape-only, accum>1 | **高** | ~90% | 下一 micro-batch 的 P1 + P2a (rollout + render) |
| Tex-only, accum=1 | 中 | ~45% | next batch shape_decode(frozen) + dense_sampling |
| Tex-only, accum>1 | **高** | ~90% | 下一 micro-batch 的 shape_dec + P1 + P2a |
| Shape+Tex, accum=1 | **高** | ~80% | S guid 期间: T P1+P2a; T guid 期间: next batch prefetch |
| Shape+Tex, accum>1 | **极高** | ~95% | 双重流水线（阶段间 + micro-batch 间） |
