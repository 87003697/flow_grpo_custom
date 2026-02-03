# Flow Matching 图像编辑方法对比

## 符号约定

| 符号 | 含义 |
|------|------|
| $z_0$ | 干净图像 (原图 latent) |
| $\epsilon$ | 目标噪声 |
| $z_t$ | t 时刻的 latent |
| $v_\theta(z_t, t)$ | 模型预测的速度场 |
| $y_{src}$ | 源 prompt (描述原图) |
| $y_{tgt}$ | 目标 prompt (描述编辑目标) |
| $v_{src}$ | 用 $y_{src}$ 预测的速度 |
| $v_{tgt}$ | 用 $y_{tgt}$ 预测的速度 |
| $t \in [0, 1]$ | 时间步 (0=干净, 1=纯噪声) |
| $\Delta t$ | 时间步长 |

---

## 1. 引言

### 1.1 图像编辑的两种范式

在基于 Flow Matching 的图像编辑中，存在两种主要的技术范式：

**反演-编辑 (Invert-then-Edit)**

这类方法采用两阶段流程：
1. **反演阶段**：将真实图像 $z_0$ 映射到噪声空间，得到对应的 $z_T$
2. **编辑阶段**：从 $z_T$ 出发，使用 $y_{tgt}$ 重新生成

代表方法：RF Inversion、DNAEdit

## 2. Flow Matching 基础

### 2.1 前向过程

Flow Matching 采用**直线插值**定义从数据到噪声的路径：

$$z_t = (1-t) \cdot z_0 + t \cdot \epsilon, \quad t \in [0, 1]$$

其中：
- $t=0$ 时，$z_0$ 是干净图像
- $t=1$ 时，$z_1 = \epsilon$ 是纯高斯噪声
- 中间状态是两者的线性插值

### 2.2 速度场定义

对 $z_t$ 关于 $t$ 求导，得到**速度场**：

$$v = \frac{dz_t}{dt} = \epsilon - z_0$$

模型 $v_\theta$ 学习预测这个速度场：

$$v_\theta(z_t, t, y) \approx \epsilon - z_0$$

其中 $y$ 是条件（如 prompt）。

### 2.3 ODE 采样（生成）

从噪声 $z_1 = \epsilon$ 出发，沿速度场反向积分到 $z_0$：

$$z_{t-\Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t, y)$$

注意：这里 $\Delta t < 0$（从 $t=1$ 积分到 $t=0$），也可写作：

$$z_{t-|\Delta t|} = z_t - |\Delta t| \cdot v_\theta(z_t, t, y)$$

### 2.4 朴素反演 (Naive Inversion)

**基本思想**：反转采样过程，从 $z_0$ 积分到 $z_1$。

$$z_{t+\Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t, y_{src})$$

从 $t=0$ 逐步积分到 $t=1$，得到反演噪声 $z_1^{inv}$。

**理想情况**

如果模型完美，应有：
$$z_0 \xrightarrow{\text{反演}} z_1^{inv} \xrightarrow{\text{生成}} z_0$$

**实际问题：累积误差**

模型预测存在误差 $\delta$：
$$v_\theta(z_t, t) = v_{true}(z_t, t) + \delta(z_t, t)$$

经过 $N$ 步积分，误差累积：
$$\text{总误差} \approx \sum_{i=1}^{N} \Delta t \cdot \delta(z_{t_i}, t_i)$$

**CFG 放大问题**

使用 Classifier-Free Guidance 时：
$$v_{cfg} = v_{uncond} + w \cdot (v_{cond} - v_{uncond})$$

CFG 将误差放大约 $w$ 倍（通常 $w = 3.5 \sim 7.5$），导致：
- 反演结果 $z_1^{inv}$ 偏离真实噪声分布
- 重建失败：$z_0^{rec} \neq z_0$

**可视化**

```
理想:   z_0 ═══════════════> z_1 ═══════════════> z_0  ✓

朴素:   z_0 ────────╮
                    ╰──────> z_1' (漂移)
                               │
                    ╭──────────┘
                    ╰──────> z_0' ≠ z_0  ✗
```

**这引出了核心问题**：如何修正反演/生成过程中的误差？

三种方法的解决思路：
- **RF Inversion**: 控制项 $\gamma(v_{cond} - v)$ 显式修正轨迹
- **FlowEdit**: 差分 $v_{tgt} - v_{src}$ 隐式抵消误差
- **DNAEdit**: 补偿向量 $\delta_v$ 记录并复用误差
## 3. RF Inversion

### 3.1 核心思想：受控 ODE

RF Inversion 的核心洞察：**朴素反演失败是因为轨迹偏离理想直线**。

解决方案：引入控制项，将轨迹"拉回"理想路径 $z_t^{ideal} = (1-t) \cdot z_0 + t \cdot \epsilon$。

### 3.2 Controlled Forward ODE (反演)

$$\frac{dz_t}{dt} = v_\theta(z_t) + \gamma \cdot \left(\frac{\epsilon - z_t}{1-t} - v_\theta(z_t)\right)$$

| 项 | 含义 |
|----|------|
| $v_\theta(z_t)$ | 模型预测速度（有误差） |
| $\frac{\epsilon - z_t}{1-t}$ | 理想速度：指向目标 $\epsilon$ |
| $\gamma$ | 控制强度（0 = 纯模型，1 = 纯解析） |

### 3.3 Controlled Reverse ODE (生成)

$$\frac{dz_t}{dt} = v_\theta(z_t) + \eta \cdot \left(\frac{z_0 - z_t}{t} - v_\theta(z_t)\right)$$

| 项 | 含义 |
|----|------|
| $v_\theta(z_t)$ | 模型预测速度（使用 $y_{tgt}$） |
| $\frac{z_0 - z_t}{t}$ | 保真速度：指向原图 $z_0$ |
| $\eta$ | 保真强度（0 = 纯编辑，1 = 纯重建） |

### 3.4 关键参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| $\gamma$ | 反演控制强度 | 0.5 ~ 1.0 |
| $\eta$ | 生成保真强度 | 0.3 ~ 0.7 |

## 4. FlowEdit

### 4.1 核心思想：双分支差分 (无需反演)

FlowEdit 的核心洞察是：**用差分消除误差，无需反演**。

通过同时运行 Source 和 Target 两个分支，利用 $v_{tgt} - v_{src}$ 差分自动抵消模型的共同误差。

### 4.2 Source Branch

**加噪**（解析式，无需模型推理）：
$$z_t^{src} = (1-t) \cdot z_0 + t \cdot \epsilon$$

**速度预测**：
$$v_{src} = v_\theta(z_t^{src}, t, y_{src})$$

### 4.3 Target Branch

**传输**（将编辑结果推到相同噪声水平）：
$$z_t^{tgt} = z_t^{edit} + z_t^{src} - z_0$$

**速度预测**：
$$v_{tgt} = v_\theta(z_t^{tgt}, t, y_{tgt})$$

### 4.4 差分更新

$$z_t^{edit} = z_{t+\Delta t}^{edit} + \Delta t \cdot (v_{tgt} - v_{src})$$

**为什么有效？** 若模型存在共同误差 $\delta$，差分自动消除：
$$v_{tgt} - v_{src} = (v_{tgt}^{true} + \delta) - (v_{src}^{true} + \delta) = v_{tgt}^{true} - v_{src}^{true}$$

### 4.5 关键参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| $n_{max}$ | 最大步数，控制编辑强度 | N/2 |
| $cfg_{src}$ | Source CFG，通常较低 | 1.0 ~ 4.0 |
| $cfg_{tgt}$ | Target CFG，越高编辑越强 | 7.5 ~ 12.0 |

## 5. DNAEdit

### 5.1 核心思想：补偿向量策略

DNAEdit 的核心洞察：**记录反演时的误差，生成时复用来抵消**。

通过保存 Forward 阶段的补偿向量 $\delta_v$，在 Reverse 阶段实现精确重建或可控编辑。

### 5.2 Forward 阶段 (反演)

**预测下一步位置**（解析式）：
$$z_t^{pred} = z_{t-\Delta t} + \frac{\Delta t}{1-t+\Delta t} \cdot (\epsilon - z_{t-\Delta t})$$

**计算补偿向量**（误差 = 解析速度 - 模型速度）：
$$\delta_v = \frac{z_t^{pred} - z_{t-\Delta t}}{\Delta t} - v_{src}$$

**修正轨迹**：
$$z_t = z_t^{pred} - \delta_v \cdot \Delta t$$

**保存**：每步的 $\delta_v$ 和 $v_{src}$，供 Reverse 阶段使用。

### 5.3 Reverse 阶段 (生成)

**混合速度**（编辑方向 + 保真方向）：
$$v = \alpha \cdot v_{tgt} + (1-\alpha) \cdot \frac{z_t - z_{ref}}{t}$$

| 项 | 含义 |
|----|------|
| $v_{tgt}$ | 目标方向（使用 $y_{tgt}$ 预测） |
| $\frac{z_t - z_{ref}}{t}$ | 保真方向：指向参考轨迹 |
| $\alpha$ | 混合系数（0 = 纯重建，1 = 纯编辑） |

**更新**：
$$z_{t-\Delta t} = z_t - \Delta t \cdot v$$

### 5.4 关键参数

| 参数 | 作用 | 建议值 |
|------|------|--------|
| $T_{start}$ | 跳过前几步（高噪声区无需补偿） | 5 ~ 10 |
| $\alpha$ (mvg) | 编辑 vs 保真混合系数 | 0.5 ~ 0.8 |

## 6. 方法对比
   6.1 统一视角：三种修正策略
   6.2 完整对比表
   6.3 选择指南

| 对比维度 | RF Inversion | FlowEdit | DNAEdit |
|---------|--------------|----------|---------|
| **范式** | 反演-编辑 | 直接编辑 | 反演-编辑 |
| **反演方式** | Controlled ODE | 无需反演 | 补偿向量 $\delta_v$ |
| **反演公式** | $v + \gamma(v_{cond} - v)$ | — | $z_t = z_t^{pred} - \delta_v \cdot \Delta t$ |
| **生成公式** | $v + \eta(v_{cond} - v)$ | $v_{tgt} - v_{src}$ | $\alpha \cdot v_{tgt} + (1-\alpha) \cdot \frac{z_t - z_{ref}}{t}$ |
| **分支数** | 1 | 2 (并行) | 1 |
| **存储** | $z_1$, $z_0$ | $z_t^{edit}$ | $\delta_v$ 列表, $v_{src}$ 列表 |
| **控制参数** | γ, η, start/stop_t | n_max, cfg_src/tgt | T_start, α (mvg) |
| **理论基础** | LQR 最优控制 | OT 传输 | 经验补偿 |
| **计算量** | 1N + 1N | 2N (每步双推理) | 1N + 1N |

---

## 核心公式汇总

### RF Inversion

**反演**：
$$\hat{v} = v_\theta(z_t) + \gamma \cdot \left(\frac{\epsilon - z_t}{1-t} - v_\theta(z_t)\right)$$

**生成**：
$$\hat{v} = v_\theta(z_t) + \eta \cdot \left(\frac{z_0 - z_t}{t} - v_\theta(z_t)\right)$$

### FlowEdit

**加噪**：
$$z_t^{src} = (1-t) \cdot z_0 + t \cdot \epsilon$$

**传输**：
$$z_t^{tgt} = z_t^{edit} + z_t^{src} - z_0$$

**更新**：
$$z_t^{edit} = z_{t+\Delta t}^{edit} + \Delta t \cdot (v_{tgt} - v_{src})$$

其中 $v_{src} = v_\theta(z_t^{src}, y_{src})$，$v_{tgt} = v_\theta(z_t^{tgt}, y_{tgt})$。

### DNAEdit

**反演**：
$$\delta_v = \frac{z_t^{pred} - z_{t-\Delta t}}{\Delta t} - v_{src}$$
$$z_t = z_t^{pred} - \delta_v \cdot \Delta t$$

**生成**：
$$v = \alpha \cdot v_{tgt} + (1-\alpha) \cdot \frac{z_t - z_{ref}}{t}$$

其中 $v_{src} = v_\theta(z_t, y_{src})$，$v_{tgt} = v_\theta(z_t, y_{tgt})$。
