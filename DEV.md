# Reg Loss 计算原理

## 概述

本项目实现了两种正则化方法用于扩散模型蒸馏：**VSD (Variational Score Distillation)** 和 **KL 正则化**。
核心设计支持**梯度穿透整个 rollout 链**，结合 gradient checkpoint 节省显存。

---

## 1. threestudio 原始 VSD/SDS 实现

### 应用场景
threestudio 的 VSD/SDS 用于**优化 3D 表示**（如 NeRF、Gaussian Splatting），目标是让渲染图像符合扩散模型的先验。

### 核心流程
```
3D 表示 (θ) → 渲染图像 (x) → 加噪 (x_t) → 扩散模型预测 → SDS/VSD 梯度 → 更新 θ
```

### 梯度计算公式

**SDS (Score Distillation Sampling)**:
$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon}\left[ w(t) \cdot (\hat{\epsilon}_\phi(x_t, t, c) - \epsilon) \cdot \frac{\partial x}{\partial \theta} \right]$$

**VSD (Variational Score Distillation)**:
$$\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\epsilon}\left[ w(t) \cdot (\hat{\epsilon}_\phi(x_t, t, c) - \hat{\epsilon}_{\text{LoRA}}(x_t, t, c)) \cdot \frac{\partial x}{\partial \theta} \right]$$

### 实现方式：SpecifyGradient
threestudio 使用 `SpecifyGradient` 将预计算的梯度绑定到渲染图像：

```python
class SpecifyGradient(Function):
    @staticmethod
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device)  # 返回标量伪损失
    
    @staticmethod
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        return gt_grad * grad_scale, None  # 将预计算梯度回传
```

**使用方式**：
```python
grad = w(t) * (noise_pred_teacher - noise_pred_student)
loss = SpecifyGradient.apply(rendered_image, grad)
loss.backward()  # 梯度回传到 3D 表示
```

---

## 2. 本项目的 VSD 实现

### 应用场景
本项目用于**扩散模型 LoRA 蒸馏**，目标是训练一个轻量级 LoRA 模型逼近教师模型的行为。

### 核心流程
```
x_T (噪声) → rollout 去噪 → x_0 → 解码 → 渲染 → FlowEdit Loss
     ↓
每步 VSD 正则：Student vs Teacher 对齐
     ↓
梯度穿透整个 rollout → 更新 LoRA 参数
```

### 关键设计：梯度穿透 Rollout

传统实现会在每一步 `detach()` 中间状态，只优化当前步的模型输出。
**本项目允许梯度穿透整个 rollout 链**，类似 DRaFT/DDPO 的轨迹优化。

#### 修改点 1：移除 detach
```python
# 修改前（阻断梯度）
x0 = current_feats.detach() - t_norm * velocity

# 修改后（梯度穿透）
x0 = current_feats - t_norm * velocity
```

#### 修改点 2：使用 SpecifyGradient 注入梯度
```python
def compute_reg_loss(method, x0_student, x0_teacher, t, latents_feats):
    diff = x0_student - x0_teacher  # (N,C)
    
    if method == "vsd":
        grad = weight_diff(diff, t_norm, weight_mode)  # (N,C)
        # 将梯度注入到 latents_feats (x_t)
        loss = SpecifyGradient.apply(latents_feats, grad)
    elif method == "kl":
        var = t_norm ** 2 + 1e-4
        loss = (0.5 * diff ** 2 / var).mean()
    
    return loss, metric
```

---

## 3. 对比总结

| 特性 | threestudio VSD | 本项目 VSD |
|------|-----------------|------------|
| **优化目标** | 3D 表示 (NeRF/GS) | 扩散模型 LoRA |
| **x_t 含义** | 渲染图像 + 噪声 | Diffusion latent |
| **梯度注入点** | 渲染图像 | 每步 x_t (latents_feats) |
| **梯度穿透 rollout** | ❌ 不需要（单次加噪） | ✅ 需要（完整 rollout） |
| **显存优化** | 不需要 checkpoint | gradient checkpoint |
| **正则化公式** | `ε_teacher - ε_student` | `x0_student - x0_teacher` |

---

## 4. 梯度流图

### 每步 VSD 梯度注入
```
x_T (噪声)
 │
 ├─────────────────────────────────┐
 ▼                                 │
Student(x_T) ──► x0_student       │
 │                   │             │
 │                   ▼             │
 │            diff = x0_s - x0_t   │
 │                   │             │
 │                   ▼             │
 │            SpecifyGradient      │
 │            (x_T, grad)  ────────┘
 │                                 ▲ 梯度从这里注入
 ▼
x_{T-1} ◄─────────────────────────── 梯度穿透
 │
 ├─────────────────────────────────┐
 ▼                                 │
Student(x_{T-1}) ──► x0_student   │
 │                   │             │
 ...                ...           ...
 │
 ▼
x_0 (最终结果)
 │
 ▼
[解码 → 渲染 → FlowEdit Loss]
 │
 ▼
全链回传到 LoRA 参数 ✅
```

---

## 5. 权重模式 (weight_mode)

| 模式 | 公式 | 说明 |
|------|------|------|
| `uniform` | `grad = diff` | 所有时间步权重相同 |
| `t` | `grad = t * diff` | 按时间步加权（早期步权重大） |
| `ada` | `grad = diff / (\|x0_teacher\|.mean() + ε)` | 自适应归一化 |

---

## 6. 配置示例

```python
cfg.reg = ml_collections.ConfigDict({
    "type": "vsd",         # "vsd" | "kl" | "none"
    "weight_mode": "ada",  # "uniform" | "t" | "ada"
})
```

---

## 7. 与 DDPO / DRaFT 的对比

本项目的"梯度穿透 rollout"设计与 DDPO、DRaFT 有相似之处，但也有关键区别。

### 7.1 DDPO (Denoising Diffusion Policy Optimization)

**论文**: Training Diffusion Models with Reinforcement Learning (NeurIPS 2023)

**核心思想**：
- 将扩散模型的去噪过程视为**马尔可夫决策过程 (MDP)**
- 使用**策略梯度 (Policy Gradient)** 方法优化模型
- 适用于**不可微的奖励函数**（如人类偏好、CLIP 分数等）

**梯度计算**：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]$$

其中 $\tau$ 是完整的去噪轨迹，$R(\tau)$ 是最终奖励。

**特点**：
- ✅ 支持不可微奖励（CLIP、人类反馈等）
- ❌ 高方差（需要大量样本）
- ❌ 需要多次采样估计梯度
- ❌ 训练效率较低

---

### 7.2 DRaFT (Differentiable Reward for Accelerating Finetuning)

**论文**: Direct Reward Fine-Tuning (arXiv 2024)

**核心思想**：
- 假设奖励函数是**可微的**
- 通过**反向传播穿透整个 rollout** 直接计算梯度
- 只在最后 K 步反向传播以节省显存

**梯度计算**：
$$\nabla_\theta \mathcal{L} = \nabla_\theta R(x_0) = \frac{\partial R}{\partial x_0} \cdot \frac{\partial x_0}{\partial x_1} \cdots \frac{\partial x_{T-K}}{\partial \theta}$$

**特点**：
- ✅ 低方差（直接梯度计算）
- ✅ 高效训练
- ❌ 要求奖励函数可微
- ❌ 完整 rollout 反传需要大量显存

---

### 7.3 本项目的方法

**设计定位**：结合 VSD 蒸馏 + DRaFT 风格的梯度穿透

**核心设计**：
1. **VSD 正则**：每步计算 Student-Teacher 差异，用 `SpecifyGradient` 注入
2. **梯度穿透**：不 detach 中间状态，梯度可以沿 rollout 链回传
3. **显存优化**：使用 `gradient checkpoint` 避免显存爆炸
4. **双重损失**：VSD 正则 + FlowEdit 渲染损失

**梯度计算**：
$$\nabla_\theta \mathcal{L} = \underbrace{\sum_{t} \nabla_\theta \mathcal{L}_{\text{VSD}}^{(t)}}_{\text{每步 VSD 正则}} + \underbrace{\nabla_\theta \mathcal{L}_{\text{render}}}_{\text{渲染损失}}$$

---

### 7.4 对比表格

| 特性 | DDPO | DRaFT | 本项目 |
|------|------|-------|--------|
| **优化目标** | 扩散模型参数 | 扩散模型参数 | LoRA 参数 |
| **奖励类型** | 不可微（CLIP 等） | 可微 | 可微（SSIM/LPIPS） |
| **梯度计算** | 策略梯度（采样估计） | 直接反传 | 直接反传 + SpecifyGradient |
| **梯度穿透 rollout** | ✅（完整轨迹） | ✅（最后 K 步） | ✅（完整 + checkpoint） |
| **方差** | 高 | 低 | 低 |
| **训练效率** | 低（需多次采样） | 高 | 高 |
| **显存优化** | 不需要 | 截断反传 | gradient checkpoint |
| **额外正则** | KL penalty | 无 | VSD 逐步对齐 |

---

### 7.5 梯度流对比图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           DDPO                                      │
├─────────────────────────────────────────────────────────────────────┤
│  x_T → x_{T-1} → ... → x_0 → R(x_0)                                │
│                              │                                      │
│                              ▼                                      │
│                    策略梯度估计（采样）                               │
│                              │                                      │
│                              ▼                                      │
│                    ∇log π(a|s) · R ──► 更新 θ                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                           DRaFT                                     │
├─────────────────────────────────────────────────────────────────────┤
│  x_T → x_{T-1} → ... → x_K → x_{K-1} → ... → x_0 → R(x_0)          │
│                         │                          │                │
│                         │◄───── 反向传播 ──────────┘                │
│                         │                                           │
│                         ▼                                           │
│                    直接梯度 ──► 更新 θ                               │
│  (只反传最后 K 步)                                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         本项目                                       │
├─────────────────────────────────────────────────────────────────────┤
│  x_T ──────────────► x_{T-1} ──────────────► ... ──► x_0            │
│   │                    │                              │             │
│   ▼                    ▼                              ▼             │
│  VSD(t=T)            VSD(t=T-1)                   渲染 Loss          │
│   │                    │                              │             │
│   │◄───────────────────│◄─────────────────────────────│             │
│   │                                                                 │
│   ▼                                                                 │
│  SpecifyGradient 累积 + 直接反传 ──► 更新 LoRA                       │
│  (gradient checkpoint 节省显存)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 7.6 为什么选择这种设计？

1. **兼顾效率与灵活性**：
   - 可微奖励 → 直接反传（比 DDPO 高效）
   - gradient checkpoint → 比 DRaFT 更省显存

2. **逐步对齐**：
   - VSD 正则在每一步都约束 Student ≈ Teacher
   - 避免 DRaFT 只看最终输出导致的中间步偏移

3. **双重监督**：
   - VSD 正则：保持生成轨迹合理性
   - 渲染损失：保证最终 3D 质量

