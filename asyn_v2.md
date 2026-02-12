# 三阶段 Autograd 架构设计

## 核心思想

将 training step 的梯度计算拆为三个阶段，使 rollout / decoder+renderer / rollout backward 的计算图**互不叠加**，
任意时刻只有一个阶段的计算图驻留显存。

```
当前（单阶段）:
  loss → renderer → decoder → rollout (12 steps) → flow model θ
  ─────────── 整个 autograd graph 同时驻留显存 ───────────

三阶段（cond-level proxy）:
  Phase 1:  rollout no_grad，每步记录 cond_proxy[t] = cond_pred.detach().requires_grad_(True)
            → cond_proxy 经 CFG 混合 → 推进 scheduler → slat（proxy chain，不含模型图）
  Phase 2:  slat(proxy chain) → decoder → renderer → guidance → loss.backward()
            → 一路反传穿过 renderer → decoder → slat → scheduler → CFG → cond_proxy
            → cond_proxy[t].grad 自动填充（已包含 CFG 缩放因子）→ 释放所有图
  Phase 3:  逐步重算 f_θ(input[t], t) → (cond_proxy[t].grad * cond_pred_t).sum().backward()
            → θ.grad +=，显存 O(1)
            ★ 只需 cond forward，无需 uncond / CFG 混合
```

数学近似（cond-level proxy，detach x_t）：

记 $v_t^{cond} = f_\theta(x_t, t, c)$ 为模型在第 $t$ 步的**条件**预测，$\hat{v}_t^{cond}$ 为其 proxy。
CFG 混合：$v_t^{cfg} = g(\hat{v}_t^{cond}, v_t^{uncond})$（含 guidance_strength、rescale 等）。
Scheduler 用 CFG 结果推进：$x_{t+1} = \text{step}(v_t^{cfg}, x_t)$，最终 $\text{slat} = x_T$。

链式法则分两段：
$$\frac{\partial L}{\partial \theta} = \sum_t \underbrace{\frac{\partial L}{\partial \hat{v}_t^{cond}}}_{\text{Phase 2: autograd 通过 scheduler→CFG chain 自动算出}} \cdot \underbrace{\frac{\partial v_t^{cond}}{\partial \theta}\bigg|_{x_t=\text{const}}}_{\text{Phase 3: 逐步重算 f\_θ 并 backward}}$$

关键洞察：$\frac{\partial L}{\partial \hat{v}_t^{cond}} = \frac{\partial L}{\partial v_t^{cfg}} \cdot \frac{\partial v_t^{cfg}}{\partial \hat{v}_t^{cond}}$，
即 **CFG 的 Jacobian 被 autograd 自动包含在 cond_proxy.grad 中**。
对于简单 CFG（$v^{cfg} = w \cdot v^{cond} + (1-w) \cdot v^{uncond}$），$\frac{\partial v^{cfg}}{\partial v^{cond}} = w$。
对于含 rescale 的 CFG，autograd 自动处理复杂的 Jacobian，无需手动推导。

等价形式（实际实现）：$\theta.\text{grad} \mathrel{+}= \sum_t \nabla_\theta \left[\hat{v}_t^{cond}.\text{grad}^T \cdot v_t^{cond}\right]$

其中 $\hat{v}_t^{cond}.\text{grad}$ 由 Phase 2 的 `loss.backward()` 沿 scheduler→CFG chain 自动填充，**无需手动计算 $\Delta t$ 或 CFG 系数**。

Detach $x_t$ 的近似同之前：丢弃跨步 Jacobian $\frac{\partial v_t}{\partial x_t}\frac{dx_t}{d\theta}$（二阶项 $O(\Delta t^2)$），
与流匹配原始训练和 SDS/VSD 使用的近似一致。

## Phase 原语

> 以下用 **`slat`** 泛指当前 stage 的 latent（shape 时为 `shape_slat`，tex 时为 `tex_slat`），
> **`comp_rgb`** 泛指渲染输出（shape 时为 normals，tex 时为 PBR shaded RGB）。

### Phase 1: Rollout no_grad + 记录 RolloutTracker（stage-specific）
- 输入: state (coords, cond_emb), generator seed
- 操作:
  1. `torch.no_grad()` 下逐步推理 `cond_pred = f_θ(x_t, t, cond)` + uncond + CFG 混合
  2. 每步记录到 tracker（**proxy 建在 cond_pred 上，CFG 之前**）:
     - `tracker.timesteps[t] = t_val` — 精确时间步（float64）
     - `tracker.input_trajectory[t] = x_t.feats.detach().clone()` — 模型输入快照
     - `tracker.output_trajectory[t] = cond_pred.feats.detach().clone().requires_grad_(True)` — **条件** velocity proxy
     - ★ proxy 替换 cond_pred 后再做 CFG 混合 → velocity 依赖 cond_proxy → scheduler chain 包含 CFG
  3. 如果 `reg_enabled`（见下方 ⚠️ 陷阱警告），预计算 teacher **条件** velocity 并记录:
     - `tracker.teacher_trajectory[t] = teacher_cond.feats.detach().clone()` — 仅条件预测，无 CFG
     - Phase 3 直接读取，无需再跑 teacher 模型
  4. 用 CFG 混合结果推进 scheduler: `x_t = scheduler.step(velocity, step_idx, x_t)`
     → slat 依赖所有 `output_trajectory[t]`（经 CFG chain），构建 proxy chain（不含模型计算图）
- 输出: slat (SparseTensor, 有 proxy chain), RolloutTracker
- 显存: proxy chain（T 个 proxy tensor + CFG 算术图 + scheduler 算术图），**不含模型激活**
- Shape: `shape_phase1_rollout()` → `state.features.shape_slat`，调用 `rollout_shape()`
- Tex: `tex_phase1_rollout()` → `state.features.tex_slat`，调用 `rollout_tex()`

### Phase 2a: Decode + Render（stage-specific，train GPU）
- 输入: slat (含 proxy chain), cameras
- 操作:
  1. 直接使用带 proxy chain 的 slat（**无需 slat_proxy 中间层**）
  2. `slat → decoder → renderer → comp_rgb`
  3. 挂载 state 可视化数据（detach）
- 输出: comp_rgb (有 autograd 图，连接到 slat 的 proxy chain)
- 显存: decode/render 前向图 + proxy chain 驻留（等待 backward）
- Shape: `shape_phase2a_decode_render()` → `decode_and_render_normal()` → normals
- Tex: `tex_phase2a_decode_render()` → `decode_and_render_pbr()` → PBR RGB

### Phase 2: Guidance + Backward → 填充 tracker 梯度（通用）
- 输入: tracker, comp_rgb
- 操作:
  1. `guidance_result = compute_guidance(comp_rgb, ...)`
  2. `loss = guidance_result.loss * weight`
  3. `accelerator.backward(loss)` → **一路反传** renderer → decoder → slat → scheduler → CFG → cond_proxy
     → 自动填充 `tracker.output_trajectory[t].grad`（**已包含 CFG 缩放因子**）
  4. 构建日志
  5. 释放所有计算图（decode/render + proxy chain 一次性释放）+ `empty_cache()`
- 输出: 日志字典（梯度已在 tracker.output_trajectory[t].grad 上，含 CFG 因子，无需返回 tensor）
- 显存: 所有计算图 → backward 后全部释放
- ★ **无 slat_proxy**：单次 backward 穿过整条链，代码更简洁，sync/async 均兼容

### Phase 3: 逐步重算 + 即时 Backward（通用）
- 输入: state, system, tracker
- 正则化判断: `reg_enabled = len(tracker.teacher_trajectory) > 0`
  （Phase 1 已根据条件填充 teacher_trajectory，Phase 3 只需检查非空）
- 操作:
  1. 逐步循环（T 步）：
     a. `t_val = tracker.timesteps[t]` — 精确时间步（无需重建 scheduler）
     b. `x_t = tracker.input_trajectory[t]` — 直接从 tracker 读取（无需重放 rollout）
     c. `cond_pred = f_θ(x_t, t_val, cond_emb)` — 仅 cond forward，对 θ 有梯度
     d. `v_grad = tracker.output_trajectory[t].grad` — Phase 2 沿 CFG chain 反传的梯度（含 CFG 因子）
     e. `combined = (v_grad * cond_pred.feats).sum()` — guidance 梯度项
     f. 如果 `reg_enabled`: `teacher_feats = tracker.teacher_trajectory[t]`
        `combined += λ * mse_loss(cond_pred.feats, teacher_feats.detach())` — v 正则化
     g. `combined.backward()` — **即时** 累积 θ.grad，本步计算图立即释放
  2. `empty_cache()`
- 输出: flow model 参数梯度 (已累积在 θ.grad), 完整日志字典
- 显存: **永远只有 1 步的激活驻留**（O(1)，不随步数增长）
- ★ **只需 cond forward，不需要 uncond 计算或 CFG 混合**（CFG 因子已在 v_grad 中）
- 不需要 step-level gradient checkpoint
- 不需要 gen_seed / 重放 rollout（input_trajectory + timesteps 已记录）
- 不需要重跑 teacher 模型（teacher_trajectory 已在 Phase 1 预计算）

### RolloutTracker 的优势
- **Phase 3 不再需要 seed 确定性**：input_trajectory 已显式记录每步输入，无需用相同 seed 重跑 scheduler
- **Phase 3 不再需要手动算 Δt**：output_trajectory[t].grad 由 autograd 沿 scheduler chain 自动计算，正确处理任意 scheduler
- **Phase 3 不再需要 uncond 计算或 CFG 混合**：proxy 建在 cond_pred 上，CFG 的 Jacobian 已自动包含在 .grad 中
- **Phase 3 不再需要重跑 teacher 模型**：teacher_trajectory 在 Phase 1 预计算，Phase 3 直接读取
- **无 slat_proxy 中间层**：decode/render 直连 proxy chain，loss.backward() 一路到底，无需手动分段 backward
- **消除 Phase2aResult / Phase2bResult**：tracker 既是 Phase 间的数据传递载体，也是梯度的自然存储位置

### ⚠️ 陷阱警告：rollout 函数中 `reg_enabled` 的判断条件

**问题**：`rollout_shape` / `rollout_tex` 中的 `reg_enabled` 控制了 teacher velocity 的计算。
三阶段路径调用 rollout 时传入 `is_training=False`（因为模型推理走 `torch.no_grad()`），
但**仍然需要预计算 teacher velocity** 写入 `tracker.teacher_trajectory`。

**错误写法**（会导致三阶段路径永远不记录 teacher trajectory）：
```python
# ❌ 三阶段路径 is_training=False，reg_enabled 恒为 False
reg_enabled = reg_type == "v" and is_training
```

**正确写法**：
```python
# ✅ tracker is not None 代表三阶段路径，也需要计算 teacher
reg_enabled = reg_type == "v" and (is_training or tracker is not None)
```

同理，rollout 末尾的 `state.regularization.reg_loss` 也要区分：
```python
# 单阶段路径：在 rollout 内直接算 reg loss
# 三阶段路径：reg loss 在 Phase 3 算，rollout 只记录 teacher trajectory
state.regularization.reg_loss = reg_loss_sum / num_steps if (reg_enabled and tracker is None) else None
```

Phase 3 中的 reg 判断则简单检查 teacher_trajectory 是否非空：
```python
reg_enabled = len(tracker.teacher_trajectory) > 0
```

**Tex-only / Shape+Tex 实现时请遵循相同模式，避免重蹈此坑。**

### ★ 关键设计：proxy 建在 cond_pred 上（不是 CFG 后的 velocity）

**动机**：如果 proxy 建在 CFG 后的 velocity 上，Phase 3 就需要重算 uncond + CFG 混合（额外一次模型 forward）。
将 proxy 提前到 cond_pred 上，CFG 混合在 proxy 之后进行，autograd 沿 scheduler → CFG → cond_proxy chain 反传时
自动将 CFG 的 Jacobian 包含在 `cond_proxy.grad` 中。

**Phase 1 中的 proxy chain 构建顺序**：
```python
# 1. cond_pred = f_θ(x_t, t, cond)    # no_grad
# 2. cond_proxy = cond_pred.feats.detach().clone().requires_grad_(True)   # proxy
# 3. cond_pred = cond_pred.replace(cond_proxy)                            # 替换
# 4. velocity = cfg_mix(cond_pred, uncond_pred)   # CFG 在 proxy 之后 → velocity 依赖 cond_proxy
# 5. scheduler.step(velocity, ...)                # scheduler 依赖 velocity → 依赖 cond_proxy
#
# → proxy chain: slat ← scheduler ← velocity ← CFG ← cond_proxy
# → Phase 2 backward: loss → renderer → decoder → slat → ... → cond_proxy.grad（自动含 CFG 因子）
```

**Phase 3 的简化**：
```python
# 只需 cond forward：
cond_pred = f_θ(x_t, t, cond)                    # 重算，有 θ 梯度
v_grad = tracker.output_trajectory[t].grad         # 已含 CFG 因子
combined = (v_grad * cond_pred.feats).sum()        # 无需 uncond / CFG
combined.backward()                                 # θ.grad +=
```

**数学等价性**：对于简单 CFG（`v_cfg = w * v_cond + (1-w) * v_uncond`），
`cond_proxy.grad = w * ∂L/∂v_cfg`，即 CFG 权重 w 被自动吸收。
对于含 rescale 的复杂 CFG，autograd 自动处理完整的 Jacobian。

**收益**：Phase 3 每步省去 1 次 uncond forward + CFG 混合，**计算量减半**。

## 显存对比

| 时刻 | 原方案 (单阶段) | 三阶段 (cond-level proxy) |
|---|---|---|
| Rollout 中 | rollout graph (T 步叠加) | proxy chain（T 个 proxy tensor + scheduler 算术图，**无模型激活**） |
| Decode+Render+Guidance 中 | rollout + decoder + renderer | decoder + renderer + proxy chain（proxy chain 很轻量） |
| Phase 2 Backward 中 | — | backward 一路穿过 renderer→decoder→slat→scheduler→CFG→cond_proxy，之后全部释放 |
| Phase 3 中 | — | **仅 1 步激活** (O(1)) |
| 峰值 | ~3 个图叠加 | **decode+render 图 + proxy chain**（proxy chain ≈ T×12MB ≈ 500MB） |

## 代码架构：调用栈

### 数据结构

`Trellis2System` 扩展：将 `cfg` 和 `accelerator` 直接挂在 system 上，省去独立的 TrainContext。
Phase 函数只需 `(state, system)` 即可访问所有训练配置和组件。

```python
@dataclass
class Trellis2System:
    """训练系统：包含所有组件、配置和加速器引用。"""
    pipeline: Any = None            # Trellis2RefAdapter
    shape: StageSystem = ...        # model, optimizer, renderer, config
    guidance: Any = None            # 共享 Guidance
    strategy: Any = None            # LoRA / Full / Frozen
    cfg: ml_collections.ConfigDict = None   # ★ 新增：训练配置
    accelerator: Accelerator = None         # ★ 新增：加速器

    # 便捷属性（衍生值，运行时从 pipeline/cfg 查询即可）
    # system.pipeline.get_stage_config("shape")["flow_resolution"]  → flow_res
    # system.pipeline.target_resolution                             → target_res
    # system.pipeline.get_ss_params()                               → ss_params
    # system.cfg.renderer.normal_mode                               → normal_mode
    # system.cfg.train.loss.reg                                     → reg_weight
    # system.cfg.train.loss.guidance                                → guidance_weight
    # system.accelerator.device                                     → device

@dataclass
class RolloutTracker:
    """
    Rollout 过程中的 proxy 记录器 — Phase 间的自包含数据传递载体。
    Phase 1 写入轨迹，Phase 2 backward 自动填充 .grad，Phase 3 消费 .grad。
    无 slat_proxy 中间层：decode/render 直连 proxy chain，loss.backward() 一路到底。

    数据流:
      Phase 1  → 写入 input_trajectory / output_trajectory / timesteps / teacher_trajectory
      Phase 2a → slat(含 proxy chain) → decoder → renderer → comp_rgb
      Phase 2  → loss.backward() → 一路反传到 output_trajectory[t].grad（含 CFG 因子）→ 释放所有图
      Phase 3  → 读取 timesteps[t] + input_trajectory[t] + output_trajectory[t].grad
                + teacher_trajectory[t] → 逐步重算 f_θ 并即时 backward
    """
    # Phase 1 写入：rollout 每步的输入/输出快照 + 时间步
    input_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   T × (N, C), 每步 x_t.feats.detach().clone()（Phase 3 重算 f_θ 的输入）
    output_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   T × (N, C), 每步 cond_pred.feats.detach().clone().requires_grad_(True)
    #   ★ 条件 velocity proxy（不是 CFG 后的 velocity）
    #   proxy 替换 cond_pred 后再做 CFG → scheduler chain 包含 CFG 算术图
    #   → Phase 2 backward 后 .grad 自动包含 CFG 缩放因子
    #   → Phase 3 只需重算 cond_pred，无需 uncond / CFG
    timesteps: List[float] = field(default_factory=list)
    #   T × float64, 每步精确 t_val（Phase 3 直接读取，无需重建 scheduler）

    # Phase 1 写入（可选）：teacher cond velocity feats，仅 reg_type="v" 时填充
    teacher_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   T × (N, C), 每步 teacher_cond.feats.detach().clone()（仅条件预测，无 CFG）
    #   ★ 默认空 list（不是 None）— Phase 3 用 len() > 0 判断是否启用 reg

    # ---- Phase 2 backward 后，以下 .grad 自动可用 ----
    # output_trajectory[t].grad  → ∂L/∂v_t^cond（autograd 沿 renderer→decoder→slat→scheduler→CFG chain 自动算出，含 CFG 因子）
```

### 调用栈（以 Shape-only 为例）

```
main()
├── 初始化 (env, accelerator, dataloaders, system, ckpt)
│   └── system.cfg = cfg; system.accelerator = accelerator  # 挂载到 system
│
├── evaluate()                                   # eval 分支（不变）
│
└── 训练循环
    └── three_phase_shape_step(state, system, global_step) → Dict[str, Any]
        │
        ├── dense_sampling_no_grad(state, system)
        │   └── pipeline.dense_sampling()                 # no_grad
        │
        ├── tracker = shape_phase1_rollout(state, system, gen_seed) → RolloutTracker
        │   └── rollout_shape() + 记录 input/output_trajectory → proxy 推进 scheduler
        │
        ├── comp_rgb = shape_phase2a_decode_render(state, system) → Tensor
        │   ├── ★ 直接使用 slat(含 proxy chain)，无 slat_proxy
        │   ├── decode_and_render_normal(slat, ...)
        │   └── 挂载 state 可视化数据
        │
        ├── guidance_log = phase2_guidance_and_backward(state, system, tracker, comp_rgb) → Dict
        │   ├── guidance.compute_guidance(comp_rgb, ...)
        │   ├── accelerator.backward(loss)  → 一路反传到 tracker.output_trajectory[t].grad
        │   └── del + empty_cache()
        │
        └── phase3_log = phase3_rollout_grad_backward(state, system, tracker) → Dict
            ├── reg_enabled = len(tracker.teacher_trajectory) > 0
            └── for t in range(T):                    # 逐步即时 backward
                ├── t_val = tracker.timesteps[t]       # 精确时间步
                ├── x_t = tracker.input_trajectory[t]  # 直接读取
                ├── cond_pred = f_θ(x_t, t_val, cond)  # 仅 cond forward，带 θ 梯度
                │   ★ 无需 uncond / CFG（CFG 因子已在 v_grad 中）
                ├── v_grad = tracker.output_trajectory[t].grad  # autograd 沿 CFG chain 填充
                ├── combined = (v_grad * cond_pred).sum()        # guidance 梯度项
                ├── if reg_enabled:                              # v 正则化（Phase 1 预计算）
                │   combined += λ * mse(cond_pred, tracker.teacher_trajectory[t])
                └── combined.backward()                          # 单次 backward，即时释放
```

### Phase 函数签名

```python
# ==================== 通用函数 ====================

def dense_sampling_no_grad(state: Trellis2State, system: Trellis2System) -> None:
    """Dense Sampling（no_grad）。填充 state.coords。"""

def phase2_guidance_and_backward(
    state: Trellis2State, system: Trellis2System,
    tracker: RolloutTracker, comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """Phase 2 同步版: guidance 计算 + accelerator.backward(loss)
    → 一路反传到 output_trajectory[t].grad（含 CFG 因子）→ 释放所有图。
    无 slat_proxy，单次 backward 穿过整条链。返回日志字典。"""

def phase2_async_guidance_backward(
    state: Trellis2State, system: Trellis2System,
    tracker: RolloutTracker, comp_rgb: torch.Tensor, async_result: AsyncGuidanceResult,
) -> Dict[str, Any]:
    """Phase 2 异步版: comp_rgb.backward(rgb_grad)
    → 一路反传到 output_trajectory[t].grad（含 CFG 因子）→ 释放所有图。
    无 slat_proxy，单次 backward 穿过整条链。返回日志字典。"""

def phase3_rollout_grad_backward(
    state: Trellis2State, system: Trellis2System, tracker: RolloutTracker,
) -> Dict[str, Any]:
    """Phase 3（通用）: 逐步从 tracker 读取 input/grad，仅重算 cond f_θ 并即时 backward。
    ★ 无需 uncond / CFG（proxy 建在 cond_pred 上，.grad 已含 CFG 因子）。
    θ.grad 逐步累积。返回日志字典。"""

# ==================== Shape stage-specific ====================

def shape_phase1_rollout(
    state: Trellis2State, system: Trellis2System, gen_seed: int,
) -> RolloutTracker:
    """Shape Phase 1: 无梯度 rollout_shape + 记录 proxy 轨迹。
    填充 state.features.shape_slat（有 proxy chain），返回 tracker。"""

def shape_phase2a_decode_render(
    state: Trellis2State, system: Trellis2System,
) -> torch.Tensor:
    """Shape Phase 2a: slat(含 proxy chain) → decode_and_render_normal → comp_rgb (normals)。
    直接使用 slat，无 slat_proxy。返回 comp_rgb。"""

# ==================== Tex stage-specific ====================

def tex_phase1_rollout(
    state: Trellis2State, system: Trellis2System, gen_seed: int,
) -> RolloutTracker:
    """Tex Phase 1: 无梯度 rollout_tex + 记录 proxy 轨迹。
    填充 state.features.tex_slat（有 proxy chain），返回 tracker。"""

def tex_phase2a_decode_render(
    state: Trellis2State, system: Trellis2System,
) -> torch.Tensor:
    """Tex Phase 2a: slat(含 proxy chain) → decode_and_render_pbr → comp_rgb (PBR RGB)。
    直接使用 slat，无 slat_proxy。返回 comp_rgb。"""
```

### 编排函数（Shape-only 示例）

```python
def three_phase_shape_step(state, system, global_step) -> Dict[str, Any]:
    gen_seed = int(system.cfg.seed) + global_step
    dense_sampling_no_grad(state, system)
    tracker = shape_phase1_rollout(state, system, gen_seed)
    comp_rgb = shape_phase2a_decode_render(state, system)           # 直接用 slat，无 slat_proxy
    guidance_log = phase2_guidance_and_backward(state, system, tracker, comp_rgb)  # guidance + 一路 backward
    phase3_log = phase3_rollout_grad_backward(state, system, tracker)
    return {**guidance_log, **phase3_log}
```

### 训练循环（~20 行）

```python
# system.cfg 和 system.accelerator 在 build_system() 后已挂载
accelerator = system.accelerator

for epoch in range(start_epoch, int(cfg.num_epochs)):
    train_loader.sampler.set_epoch(epoch)
    for batch in train_loader:
        global_step += 1
        state = Trellis2State()
        state.attach_batch(batch, pipeline=system.pipeline, ...)
        with accelerator.accumulate(system.shape.model):
            with TrainModeGuard(system.shape.model):
                shape_log = three_phase_shape_step(state, system, global_step)
            if accelerator.sync_gradients:
                system.shape.optimizer.step()
                system.shape.optimizer.zero_grad()
        shape_logger.log_step(shape_log, ...)
        del state, shape_log; torch.cuda.empty_cache()
```

训练循环本身不变——`RolloutTracker` 的生命周期完全封装在 `three_phase_shape_step` 内部。

## 异步 Guidance：跨 GPU Proxy 技巧

### 核心思路

在 **comp_rgb 层面做一次 proxy**，使 train GPU 和 guidance GPU 的 autograd 图完全解耦，
从而实现真正的异步并行：

| 层级 | proxy 位置 | 目的 |
|---|---|---|
| 三阶段 Phase 1→2→3 | cond_pred 层 | 解耦 rollout 中 flow model 计算图与 proxy chain → **显存隔离** |
| 异步 guidance | comp_rgb 层 | 解耦 train GPU 和 guidance GPU 的 autograd 图 → **计算并行** |

```
同步（单 autograd 图横跨两个 GPU）:

Train GPU                                    Guidance GPU
─────────                                    ────────────
slat(proxy chain) → decoder → renderer → comp_rgb ─→ encode → pipeline → loss
                                               ↑
                                   autograd 图跨两个 GPU（无法并行）


异步（两个 GPU 各自独立的 autograd 图）:

Train GPU                                    Guidance GPU
─────────                                    ────────────
slat(proxy chain) → decoder → renderer → comp_rgb   rgb_proxy → encode → pipeline → loss
                                           │              ↑                           │
                                           │   .detach().to(guidance).requires_grad_  │
                                           │                                loss.backward()
                                           │                                          │
                                           ◄── rgb_grad = rgb_proxy.grad.to(train) ───┘
                                           │
                              comp_rgb.backward(rgb_grad)
                                           │
                              → 一路反传到 output_trajectory[t].grad ✓
```

数学等价：`∂L/∂cond_proxy = (∂L/∂comp_rgb) · (∂comp_rgb/∂cond_proxy)`，
其中 `∂L/∂comp_rgb` 在 guidance GPU 算出，`∂comp_rgb/∂cond_proxy` 在 train GPU 上 backward 自动完成。

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
# Shape Phase 2a: decode + render（train GPU，直接用 slat 含 proxy chain）
shape_comp_rgb = shape_phase2a_decode_render(state, system)

# ★ 异步提交 shape guidance（立即返回，不阻塞）
system.guidance.submit_async(shape_comp_rgb, state.views_conditioned.image_pils)

# ======== 插入其他工作（与 shape guidance 并行）========
tex_tracker = tex_phase1_rollout(state, system, gen_seed_tex)       # tex rollout
tex_comp_rgb = tex_phase2a_decode_render(state, system)             # tex decode + render ← 也在 guidance 等待期间完成!

# ★ 等待 shape guidance 完成，取回 rgb_grad
async_result = system.guidance.wait_and_get(target_device=device)

# Shape Phase 2: 用 rgb_grad 在 train GPU 上 backward → 一路填充 shape_tracker
shape_comp_rgb.backward(async_result.rgb_grad)    # → 一路反传到 shape_tracker.output_trajectory[t].grad
# ★ 无需 slat_proxy，单次 backward 穿过整条链
```

### 时序对比（Shape+Tex）

```
同步（guidance 阻塞）:                           异步（guidance 并行）:
 T0  shape decode+render (→ comp_rgb)           T0  shape decode+render (→ comp_rgb)
 T1  |等待| shape guidance                       T1  tex rollout + decode + render
 T2  shape backward                                   ←→ shape guidance (并行!)
 T3  tex rollout                                 T2  shape backward
 T4  tex decode+render                           总耗时: T0 + max(T1_rollout+render, T_guidance) + T2
 总耗时: T0 + T1 + T2 + T3 + T4                  省掉了 guidance 等待 + tex P1+P2a 时间
```

## 三种训练模式

三阶段架构支持三种训练模式。核心 Phase 原语（P1/P2/P3）和 RolloutTracker 是**通用**的，
区别仅在于：哪些 stage 走三阶段、哪些 stage frozen、编排顺序。

### Shape-only

- 训练对象：Shape flow model
- 三阶段：Shape P1 → P2a(decode normals) → P2(guidance+backward) → P3 → `shape_opt.step()`
- Tex：不涉及
- 编排函数：`three_phase_shape_step()`

### Tex-only

- 训练对象：Tex flow model（Shape frozen，不训练）
- Shape 前置：`shape_rollout(no_grad)` → `shape_decode(no_grad)` → 获取 mesh + subs（几何条件）
- 三阶段：Tex P1 → P2(decode PBR + render) → P3 → `tex_opt.step()`
- 注意：Shape 所有计算都在 `no_grad` 下（frozen），mesh/subs 对 state 的 detach 挂载
- ⚠️ rollout_tex 中 `reg_enabled` 同样需要 `(is_training or tracker is not None)` 模式

### Shape+Tex

- 训练对象：Shape + Tex flow model（两个 optimizer）
- 编排（同步）：Shape 三阶段完整执行 → Tex 三阶段完整执行
- 编排（异步）：Shape P2a 后 submit guidance → 等待期间执行 Tex P1+P2a → wait → Shape P2(backward)+P3 → ...
- Tex 需要 Shape 产出的 mesh/subs，这些在 Shape P2a 中已 detach 存于 state，安全可用
- 两个 optimizer 各自 step

### 模式差异汇总

| | Shape-only | Tex-only | Shape+Tex |
|---|---|---|---|
| Shape flow model | ★ 训练 | frozen (no_grad) | ★ 训练 |
| Tex flow model | — | ★ 训练 | ★ 训练 |
| Shape rollout | 三阶段 P1 | no_grad（提供几何条件） | 三阶段 P1 |
| Shape decode | P2a (normals) | no_grad（获取 mesh/subs） | P2a (normals) |
| Tex rollout | — | 三阶段 P1 | 三阶段 P1 |
| Tex decode | — | P2a (PBR render) | P2a (PBR render) |
| 异步 guidance 收益 | accum>1 时高 | accum>1 时高 | 天然高（双 stage 交错） |

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

| Step | 模式 | 描述 | 文件 |
|---|---|---|---|
| 1 ✅ | Shape-only 同步 | `P1 → P2a(decode) → P2(guid+bwd) → P3 → shape_opt.step()` | `trellis2_shape_autograd.py` |
| 2 | Tex-only 同步 | Shape frozen(no_grad, 提供 mesh/subs) → Tex 三阶段 | `trellis2_tex_autograd.py` |
| 3 | Shape+Tex 同步 | Shape 三阶段 → Tex 三阶段，各自 opt.step() | `trellis2_shape+tex_autograd.py` |
| 4 | Shape+Tex 异步 | Shape guid 期间做 Tex P1+P2a（流水线并行） | 同上（异步开关） |

Step 2 注意：Tex-only 中 Shape 只提供几何条件（mesh + subs），不训练，全部 no_grad。
Step 4 注意：Tex P2a 需要 shape mesh/subs，这些在 Shape P2a 中已产出并 detach 存于 state，安全可用。

## 异步收益分析

### 各操作典型耗时（估算）

| 操作 | GPU | 耗时 | 说明 |
|---|---|---|---|
| Dense Sampling | train | ~1-2s | 数十步小模型推理 |
| Phase 1: Rollout no_grad | train | ~2-5s | 12-40 步 flow model forward，无 autograd |
| Phase 2a: Decode + Render | train | ~1-3s | chunked decoder + nvdiffrast |
| **Guidance 计算** | **guidance** | **~5-15s** | VAE encode + FlowEdit pipeline (20-40步) + loss + backward |
| Phase 2 Backward | train | ~1-3s | loss.backward() 一路穿过 renderer → decoder → slat → proxy chain |
| Phase 3: Rollout 逐步 bwd | train | ~3-7s | 12-40 步 forward + 即时 backward（无 ckpt 重算开销） |
| Optimizer step | train | ~0.01s | 几乎瞬时 |

**Guidance 是最耗时的单项操作**（~5-15s），跑在 guidance GPU 上。同步模式下 train GPU 在此期间**完全空闲**。

异步核心思路：guidance 等待期间插入其他有用计算（下一 micro-batch 的 P1+P2a，或另一 stage 的 P1+P2a）。
关键约束：Phase 1 和 Phase 3 必须用**相同权重**，所以 accum=1 时 P1 不能跨 batch 提前（opt.step 还没执行）。
accum>1 时 micro-batch 间权重不变 → 可以安全流水。Shape+Tex 天然有异步窗口（S guid 期间做 T P1+P2a）。

### 收益总结

| 场景 | 异步收益 | guidance 等待利用率 | 空闲期间工作 |
|---|---|---|---|
| Shape-only, accum=1 | 低 | ~25% | prefetch + dense_sampling + cond_enc |
| Shape-only, accum>1 | **高** | ~90% | 下一 micro-batch 的 P1 + P2a (rollout + render) |
| Tex-only, accum=1 | 中 | ~45% | next batch shape_decode(frozen) + dense_sampling |
| Tex-only, accum>1 | **高** | ~90% | 下一 micro-batch 的 shape_dec + P1 + P2a |
| Shape+Tex, accum=1 | **高** | ~80% | S guid 期间: T P1+P2a; T guid 期间: next batch prefetch |
| Shape+Tex, accum>1 | **极高** | ~95% | 双重流水线（阶段间 + micro-batch 间） |

---

## 设计备忘

### Phase 3 中 reg loss 需除以 T

Phase 3 逐步 backward 时，guidance 梯度项 `(v_grad * cond_pred).sum()` 是 chain rule 对单个 guidance loss 的分步展开，T 步加起来恰好等于一次完整 backward——**不需要除以 T**。

但 reg loss 不同：每步独立计算 `MSE(cond_pred, teacher)`，T 步累积后 reg 梯度总量 ∝ T。若改变采样步数，reg 与 guidance 的相对强度会变化。因此 **reg loss 需除以 T**：

```python
combined = combined + reg_weight * reg_loss / T  # 步平均，与 T 解耦
```

这样 `reg_weight` 的含义与步数无关，调参更稳定。
