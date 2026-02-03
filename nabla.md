# Nabla-R2D3: 基于 Score Function Matching 的 3D 扩散模型对齐

## 1. 概述

### 1.1 问题背景：如何用 2D 奖励对齐 3D 扩散模型

3D 原生扩散模型（如Trellis）可以从文本或图像条件直接生成 3D 表示（Gaussian Splatting 或 Mesh）。然而，预训练模型生成的结果往往不能完全满足人类偏好，直观的解决方案是使用 **2D 奖励函数**（如 HPS、Aesthetic Score、法线一致性评分）来评估渲染后的 2D 图像，并以此信号微调 3D 扩散模型。但这带来一个核心挑战：

> **如何将 2D 奖励信号有效地传递回 3D 扩散模型的参数更新？**

传统的强化学习方法（如 REINFORCE、PPO）存在高方差、样本效率低的问题。Nabla-R2D3 提出了一种基于 **Score Function Matching** 的新范式来解决这一问题。

### 1.2 核心思想：Transition Score Matching

Nabla-R2D3 的核心思想是：**不直接优化奖励期望，而是修改扩散模型的采样轨迹方向（transition score）来隐式提升奖励。**

具体而言，算法定义一个"理想的目标 transition score"：

$$
\mathbf{s}^*(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) = \mathbf{s}_{\text{ref}}(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) + \gamma_t \beta \nabla_{\mathbf{x}_{t-1}} \log r(\hat{\mathbf{x}}_0)
$$

其中：
- $\mathbf{s}_{\text{ref}}$：预训练模型（参考模型）的 transition score，代表"生成真实样本"的采样方向
- $\nabla_{\mathbf{x}_{t-1}} \log r(\hat{\mathbf{x}}_0)$：奖励函数对下一时间步 latent 的梯度，代表"提升奖励"的方向
- $\gamma_t$：时间步衰减因子（decay factor），用于调整不同时间步的奖励梯度强度
- $\beta$：控制奖励影响强度的温度系数（论文中称为 reward temperature）

训练目标是让学生模型的 transition score 匹配这个目标：

$$
\mathcal{L} = \left\| \mathbf{s}_\theta(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) - \mathbf{s}^*(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) \right\|^2
$$

这种方法的优势在于：
1. **低方差**：直接匹配 score，避免策略梯度的采样噪声
2. **保持先验**：通过加法形式，自然保持预训练模型的生成能力
3. **可微传递**：利用可微渲染，奖励梯度可以精确计算

## 2. 理论基础

### 2.1 Score Function 通用定义

在生成模型理论中，**Score Function** 定义为数据分布对数概率密度的梯度：

$$
\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

直观理解：Score 指向数据密度增加最快的方向。在采样过程中，沿着 score 的方向移动，可以将噪声样本逐步引导到高概率密度区域（即真实数据分布）。

### 2.2 Flow Matching 模型的 Score Function

#### Velocity Field 与 Score 的关系

Flow Matching 模型通过学习 velocity field $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 来生成样本，采样过程为：

$$
\dot{\mathbf{x}}_t = \mathbf{v}_\theta(\mathbf{x}_t, t), \quad \mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

对于 **Rectified Flow**（$\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\mathbf{x}_0$ 为干净样本），velocity field 与概率流的 score function 存在以下关系：

$$
\mathbf{s}(\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t, t) = -\left[\frac{1}{t} \mathbf{x}_t + \frac{1-t}{t} \mathbf{v}_\theta(\mathbf{x}_t, t)\right]
$$

这个公式建立了 Flow Matching 模型与 score-based 模型之间的桥梁：**velocity field 可以直接转换为 score function**。

#### Flow Matching 的等价 SDE 形式

Rectified Flow 的 ODE 采样可以转换为等价的 SDE 形式。对于反向采样过程（从 $t=1$ 到 $t=0$）：

$$
\mathrm{d}\mathbf{x}_t = \left[\mathbf{v}_\theta(\mathbf{x}_t, t) + \frac{\sigma_t^2}{2t}\left(\mathbf{x}_t + (1-t) \mathbf{v}_\theta(\mathbf{x}_t, t)\right)\right]\mathrm{d}t + \sigma_t \mathrm{d}\mathbf{w}
$$

其中 $\sigma_t$ 是可调节的扩散系数，控制采样过程的随机性。

#### Transition Score

> **重要说明**：在实际代码实现中，使用的是 **Transition Score** 而非标准的 score function。

从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-1}$ 的转移可以建模为高斯分布：

$$
p(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I})
$$

Transition score 定义为这个转移分布的对数概率梯度：

$$
\mathbf{s}_{\text{transition}} = \nabla_{\mathbf{x}_{t-1}} \log p(\mathbf{x}_{t-1} | \mathbf{x}_t) = -\frac{\mathbf{x}_{t-1} - \boldsymbol{\mu}_t}{\sigma_t^2}
$$

#### 为什么需要 SDE 采样

> **重要说明**：Transition score 的计算**必须使用 SDE**，而非纯 ODE。

Transition score 的定义依赖于 $\sigma_t > 0$。如果使用纯 ODE 采样（$\sigma_t = 0$），则：
- 转移分布退化为 Dirac delta 函数 $p(\mathbf{x}_{t-1} | \mathbf{x}_t) = \delta(\mathbf{x}_{t-1} - \boldsymbol{\mu}_t)$
- 无法定义有意义的 transition score

#### SDE 采样的代码实现

基于 Rectified Flow 的 SDE 采样，选择扩散系数为：

$$
\sigma_t = \sqrt{\frac{t}{1-t}} \cdot \text{noise\_level}
$$

离散化后的更新公式：

$$
\mathbf{x}_{t-1} = \mathbf{x}_t \left(1 + \frac{\sigma_t^2}{2t}\Delta t\right) + \mathbf{v}_\theta\left(1 + \frac{\sigma_t^2(1-t)}{2t}\right)\Delta t + \sigma_t\sqrt{|\Delta t|} \cdot \mathbf{z}
$$

**代码实现**（`sde_step_with_logprob` 函数）：

```python
def sde_step_with_logprob(
    scheduler,
    model_output: torch.FloatTensor,  # velocity field 预测 v_θ
    timestep: torch.FloatTensor,       # 当前时间步 t
    sample: torch.FloatTensor,         # 当前样本 x_t
    noise_level: float = 0.7,          # 控制 SDE 随机性（0 = ODE）
    prev_sample: torch.FloatTensor = None,
    generator: torch.Generator = None,
):
    """
    Rectified Flow 的 SDE 采样步，同时返回 log probability。
    """
    # 获取时间步信息
    step_index = scheduler.index_for_timestep(timestep)
    sigma = scheduler.sigmas[step_index]          # 当前时间 t
    sigma_prev = scheduler.sigmas[step_index + 1] # 下一时间 t-1
    dt = sigma_prev - sigma                       # 负值（从 t=1 到 t=0）
    
    # 计算扩散系数 σ_t = sqrt(t/(1-t)) * noise_level
    std_dev_t = torch.sqrt(sigma / (1 - sigma)) * noise_level  # [B, 1, 1, 1]
    
    # 计算转移均值 μ_t
    # x_{t-1} = x_t * (1 + σ²/(2t) * dt) + v * (1 + σ²(1-t)/(2t)) * dt
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + 
        model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )  # [B, C, H, W]
    
    # 添加噪声
    if prev_sample is None:
        noise = torch.randn_like(model_output, generator=generator)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise  # [B, C, H, W]
    
    # 计算 log probability（高斯分布）
    variance = (std_dev_t * torch.sqrt(-dt)) ** 2  # [B, 1, 1, 1]
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * variance)
        - torch.log(std_dev_t * torch.sqrt(-dt))
        - 0.5 * torch.log(2 * torch.pi)
    )  # [B, C, H, W]
    log_prob = log_prob.mean(dim=(1, 2, 3))  # [B]
    
    # Transition score: s = -(x_{t-1} - μ_t) / σ_t²
    # 可通过 -(prev_sample - prev_sample_mean) / variance 计算
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

**关键参数说明**：
- `noise_level = 0`：退化为 ODE 采样，无法计算 transition score
- `noise_level > 0`：SDE 采样，可以计算 transition score
- 典型值：`noise_level = 0.7`

### 2.3 Score 的加法性质

Score function 有一个重要的**加法性质**。如果我们想从一个调制后的分布采样：

$$
p^*(\mathbf{x}) \propto p_{\text{ref}}(\mathbf{x}) \cdot r(\mathbf{x})^\beta
$$

其中 $r(\mathbf{x})$ 是奖励函数，$\beta$ 控制奖励的影响强度。

对两边取对数：

$$
\log p^*(\mathbf{x}) = \log p_{\text{ref}}(\mathbf{x}) + \beta \log r(\mathbf{x}) + \text{const}
$$

再对 $\mathbf{x}$ 求梯度，得到目标分布的 score：

$$
\nabla_\mathbf{x} \log p^*(\mathbf{x}) = \nabla_\mathbf{x} \log p_{\text{ref}}(\mathbf{x}) + \beta \nabla_\mathbf{x} \log r(\mathbf{x})
$$

即：

$$
\boxed{\mathbf{s}^* = \mathbf{s}_{\text{ref}} + \beta \nabla \log r}
$$

这个性质说明：
- **目标 score = 参考 score + 奖励梯度**
- 无需重新训练整个模型，只需让模型学会在原有 score 基础上"叠加"奖励方向
- 奖励信息以**梯度形式**注入，而非直接修改分布

## 3. 算法核心

### 3.1 目标分布

Nabla-R2D3 的目标是让模型采样自一个**被奖励调制后的分布**：

$$
p^*(\mathbf{x}) \propto p_{\text{ref}}(\mathbf{x}) \cdot r(\mathbf{x})^\beta
$$

其中：
- $p_{\text{ref}}(\mathbf{x})$：预训练扩散模型定义的原始分布
- $r(\mathbf{x})$：奖励函数（如 HPS、Aesthetic Score），值越大表示样本质量越高
- $\beta$：温度系数，控制奖励对分布的影响强度
  - $\beta \to 0$：退化为原始分布
  - $\beta \to \infty$：退化为奖励最大化（可能丢失多样性）

这个目标分布的含义是：**在保持原始分布的基础上，提升高奖励样本的概率密度**。

### 3.2 目标 Transition Score

根据 Score 的加法性质，目标分布的 transition score 为：

$$
\mathbf{s}^*(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) = \mathbf{s}_{\text{ref}}(\mathbf{x}_t \to \mathbf{x}_{t-1}, t) + \gamma_t \beta \nabla_{\mathbf{x}_{t-1}} \log r(\hat{\mathbf{x}}_0)
$$

实际计算时：

1. **参考 Transition Score** $\mathbf{s}_{\text{ref}}$：
   - 使用冻结的预训练模型进行 velocity 预测
   - 通过 `sde_step_with_logprob()` 函数计算 transition score：
     $$\mathbf{s}_{\text{ref}} = -\frac{\mathbf{x}_{t-1} - \boldsymbol{\mu}_{\text{ref}}}{\sigma_t^2}$$

2. **奖励梯度** $\nabla_{\mathbf{x}_{t-1}} \log r$：
   - 从下一时间步的 latent $\mathbf{x}_{t-1}$ 开始，启用梯度追踪
   - 使用当前模型（带 LoRA）预测 velocity，估计干净样本 $\hat{\mathbf{x}}_0 = \mathbf{x}_t - t \cdot \mathbf{v}_\theta$
   - 解码为 3D 表示，可微渲染为 2D 图像
   - 计算奖励 $r$ 并使用 `torch.autograd.grad()` 显式计算梯度

3. **目标 Score 构建**：
   ```python
   # 参考模型的 transition score
   _, _, mu_ref, std_dev_t = sde_step_with_logprob(
       scheduler, v_ref, t, x_t, noise_level=noise_level, prev_sample=x_prev
   )
   score_ref = -(x_prev - mu_ref) / (std_dev_t ** 2)
   
   # 奖励梯度
   score_reward = torch.autograd.grad(log_reward.sum(), x_prev)[0]
   
   # 目标 score
   score_target = score_ref + gamma_t * reward_scale * score_reward
   ```

### 3.3 训练目标：Score Matching Loss

训练目标是让学生模型的 score 匹配目标 score。论文中的损失函数为：

$$
\mathcal{L}(\mathbf{x}_{t-1:t}) = \mathbb{E}_{c \sim \mathcal{C}} \left\| \nabla_{\mathbf{x}_{t-1}} \log \tilde{p}_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) - \gamma_t \beta \, \text{sg}\left[\nabla_{\mathbf{x}_{t-1}} \log R(g(\hat{\mathbf{x}}_0, c))\right] \right\|^2
$$

其中：
- $\log \tilde{p}_\theta = \log p_\theta - \log p_{\text{base}}$ 是 **finetuned 模型与 base 模型的 log-density ratio**
- $\gamma_t$ 是时间步衰减因子（decay factor）
- $\text{sg}[\cdot]$ 表示 stop-gradient 操作
- $g(\hat{\mathbf{x}}_0, c)$ 是将预测的干净样本渲染到相机视角 $c$ 的图像

由于 $\nabla \log \tilde{p}_\theta = \nabla \log p_\theta - \nabla \log p_{\text{base}} = \mathbf{s}_\theta - \mathbf{s}_{\text{ref}}$，上式等价于：

$$
\mathcal{L} = \left\| \mathbf{s}_\theta - \mathbf{s}_{\text{ref}} - \gamma_t \beta \nabla \log r \right\|^2
$$

训练时，目标 score（即 $\gamma_t \beta \nabla \log R$ 部分）被视为常量（stop-gradient），只有学生模型的 score 参与梯度计算。

### 3.4 关键技术设计

#### 3.4.1 奖励梯度的可微计算

奖励梯度 $\nabla_{\mathbf{x}_t} \log r(\mathbf{x}_0)$ 的计算依赖于完整的**可微渲染管线**：


关键点：
- 3D Gaussian Splatting 的渲染是可微的，梯度可以从像素传回 latent
- 使用 `torch.autograd.grad` 显式计算梯度，而非通过 loss.backward()

#### 3.4.2 Timestep Fraction

只在部分时间步上进行微调，而非所有时间步：

$$
\text{训练时间步} = \text{随机采样 } \lfloor T \times \text{fraction} \rfloor \text{ 个步骤}
$$

例如 `timestep_fraction = 0.4` 表示只训练 40% 的时间步。

设计动机：
1. **效率**：减少每个 iteration 的计算量
2. **稳定性**：避免过度修改早期（高噪声）时间步的行为
3. **经验发现**：晚期时间步对最终生成质量影响更大

#### 3.4.3 其他技术细节

##### Low Variance Subsampling

一种特殊的时间步采样策略，将时间步分成多个区间，从每个区间中随机采样，确保时间步分布更均匀：

```python
perms = torch.stack([
    torch.cat([
        torch.randperm(5) + 0,   # 区间 [0, 5)
        torch.randperm(5) + 5,   # 区间 [5, 10)
        torch.randperm(5) + 10,  # 区间 [10, 15)
        torch.randperm(5) + 15,  # 区间 [15, 20)
    ])
])
```

配置：`sampling.low_var_subsampling`

##### Reward Aggregation

多视角奖励的聚合方式：
- `mean`：取平均（默认）
- `max`：取最大值

配置：`training.reward_aggregate_func`

## 4. 与其他方法的对比

### 4.1 vs DDPO（策略梯度方法）

DDPO 将扩散模型的去噪过程视为 MDP，使用 vanilla policy gradient 进行微调。

| 方面 | DDPO | Nabla-R2D3 |
|------|------|-----------|
| **梯度信息** | 不利用奖励模型的一阶梯度 | 直接利用奖励梯度 $\nabla \log R$ |
| **方差** | 高方差（策略梯度估计） | 低方差（score matching） |
| **样本效率** | 需要大量样本 | 样本效率高 |
| **理论基础** | 策略梯度定理 | GFlowNet / Soft Q-learning |

**核心区别**：DDPO 没有利用奖励模型的可微性，而 Nabla-R2D3 通过可微渲染直接计算奖励梯度，更高效。

### 4.2 vs ReFL / DRaFT（截断反向传播方法）

ReFL 和 DRaFT 都通过截断的计算图直接优化奖励：
- **ReFL**：使用截断计算图 $z_{t+1} \to z_t \to \hat{z}_0$，随机采样时间步 $t$
- **DRaFT**：使用截断计算图 $z_K \to z_{K-1} \to ... \to z_0$，直接优化 $R(z_0)$

| 方面 | ReFL / DRaFT | Nabla-R2D3 |
|------|--------------|-----------|
| **理论基础** | 无概率理论基础 | 近似 reward-weighted 分布 |
| **过拟合风险** | 容易过拟合奖励模型 | 通过 score matching 保持先验 |
| **3D 一致性** | 可能出现 Janus 问题 | 更好的 3D 一致性 |
| **训练目标** | 直接最大化 $R(z_0)$ | 匹配目标 score |

**核心区别**：ReFL/DRaFT 的训练目标不是近似 reward-weighted 分布，因此容易过拟合；论文实验显示 DRaFT 微调的模型会出现严重的 **Janus 问题**（多视角不一致），而 Nabla-R2D3 不会。

### 4.3 vs 蒸馏方法 (DMD/KL 正则化)

| 方面 | 蒸馏方法 | Nabla-R2D3 |
|------|--------|-----------|
| **知识来源** | 教师模型输出 | 参考模型 score + 奖励梯度 |
| **目标** | 模仿教师行为 | 调制分布以提升奖励 |
| **计算开销** | 需要教师模型推理 | 需要奖励计算和梯度回传 |

**核心区别**：蒸馏方法的目标是让学生模仿教师，而 Nabla-R2D3 的目标是在保持预训练能力的同时提升奖励。

### 4.4 对比总结

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        3D 扩散模型对齐方法对比                              │
├──────────────┬─────────────┬─────────────┬──────────────┬────────────────┤
│    方法       │  方差/稳定性  │   样本效率   │   过拟合风险   │    理论基础     │
├──────────────┼─────────────┼─────────────┼──────────────┼────────────────┤
│ DDPO         │    高方差     │     低      │     中等      │   策略梯度      │
│ ReFL         │    中等       │     中      │     高       │   无概率基础    │
│ DRaFT        │    中等       │     中      │     高       │   无概率基础    │
│ Nabla-R2D3   │    低方差     │     高      │     低       │   Score matching│
└──────────────┴─────────────┴─────────────┴──────────────┴────────────────┘
```

**Nabla-R2D3 的独特优势**：
1. **端到端可微**：通过可微渲染，奖励梯度可以精确传回 3D latent
2. **保持先验**：通过加法 score 组合，自然保持预训练模型的生成能力
3. **理论基础**：基于 GFlowNet / Soft Q-learning，有概率理论支撑

### 4.5 当前 Trellis 实现分析

当前 `edit4shape/systems/trellis.py` 的实现采用的是 **ReFL/DRaFT + DMD 蒸馏正则化** 的组合方法，而非 Nabla-R2D3 的 Score Matching 方法。

#### 4.5.1 当前实现的训练流程

```
Dense Sampling → Rollout (ODE) → Decode → Render → Guidance Loss → Backward
                    ↓
              DMD/KL 正则化（与教师模型对齐）
```

核心 loss 计算：

```python
# 直接优化 guidance loss（类似 ReFL/DRaFT）
guidance_loss = state.guidance.loss * cfg.train.loss.guidance
total = guidance_loss

# 添加 DMD/KL 正则化（缓解过拟合）
if state.regularization.reg_loss is not None:
    total = total + cfg.train.loss.reg * state.regularization.reg_loss

accelerator.backward(total)
```

#### 4.5.2 与 Nabla-R2D3 的关键区别

| 特性 | 当前 Trellis 实现 | Nabla-R2D3 |
|------|------------------|-----------|
| **训练目标** | 直接优化 `guidance_loss` | 匹配目标 transition score |
| **梯度来源** | 渲染图像 → latent 反向传播 | score matching loss |
| **采样方式** | ODE（确定性） | SDE（随机性，必须） |
| **正则化** | DMD/KL 蒸馏 loss | 通过 score 加法天然保持先验 |
| **transition score** | ❌ 不计算 | ✅ 核心概念 |
| **log probability** | ❌ 不需要 | ✅ 用于计算 score |

#### 4.5.3 当前实现的优缺点

**优点**：
- 实现简单，直接优化奖励
- 通过 DMD/KL 正则化缓解过拟合
- 计算效率高（无需 SDE 采样）

**缺点**：
- 缺乏概率理论基础
- 可能出现 Janus 问题（多视角不一致）
- 正则化权重需要仔细调参

#### 4.5.4 改进为 Score Matching 训练

若要将当前实现改为 Nabla-R2D3 风格的 Score Matching 训练，需要以下修改：

1. **将 rollout 改为 SDE 采样**：
   - 使用 `sde_step_with_logprob` 替换当前的 ODE scheduler step
   - 设置 `noise_level > 0`（如 0.7）

2. **计算 transition score**：
   ```python
   # 学生模型的 transition score
   _, _, mu_stu, std_dev_t = sde_step_with_logprob(
       scheduler, v_stu, t, x_t, noise_level=noise_level
   )
   score_stu = -(x_prev - mu_stu) / (std_dev_t ** 2)
   
   # 参考模型的 transition score
   with strategy.teacher_context():
       _, _, mu_ref, _ = sde_step_with_logprob(
           scheduler, v_ref, t, x_t, noise_level=noise_level, prev_sample=x_prev
       )
   score_ref = -(x_prev - mu_ref) / (std_dev_t ** 2)
   ```

3. **替换 loss 为 score matching**：
   ```python
   # 计算奖励梯度
   score_reward = torch.autograd.grad(log_reward.sum(), x_prev)[0]
   
   # 目标 score = 参考 score + 奖励梯度
   score_target = score_ref + gamma_t * reward_scale * score_reward
   
   # Score Matching Loss
   loss = ((score_stu - score_target.detach()) ** 2).mean()
   ```

4. **移除 DMD/KL 正则化**：
   - Score 加法形式天然保持先验，无需额外正则化

## 5. 核心配置参数速查

### 5.1 论文实验参数

根据论文附录的 Implementation Details：

| 参数 | Aesthetic Score | HPSv2 | Geometry Reward | 说明 |
|------|----------------|-------|-----------------|------|
| 奖励温度 $\beta$ | 1e7 | 2e6 | 1e6 | 奖励梯度缩放 |
| Timestep fraction | 0.4 | 0.4 | 0.4 | 训练时间步比例 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 学习率 |
| LoRA rank | 16/8 | 16/8 | 16/8 | LoRA 秩（Pixart-Σ/SD1.5） |
| CFG scale | 7.5/3.5 | 7.5/3.5 | 7.5/3.5 | CFG 引导强度（DiffSplat/GaussianCube） |

### 5.2 代码配置参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `model.reward_scale` | 1.0 | 奖励梯度缩放系数 $\beta$（论文中根据 reward 类型设置为 1e6~1e7） |
| `model.timestep_fraction` | 0.4 | 训练时间步比例 |
| `sampling.guidance_scale` | 5.0 | CFG 引导强度 |
| `sampling.num_steps` | 20 | 采样步数 |