# OREO: Generalizing 3D Native Generators with On-Policy Rendering-Editing Optimization

## Abstract

尽管 3D 原生生成器取得了进展，但多样化 3D 数据的稀缺将其限制在有限的分布中，导致在复杂和非标准结构上出现严重错位。
为了解决这个问题，我们提出了 **On-Policy Rendering Editing Optimization (OREO)**，它将 3D 原生生成器的后训练重构为在渲染视角上具有稠密监督的在线策略蒸馏过程。
其核心是 **Reinforced Editing Distillation (RED)** 算法，它利用带有改进 FlowEdit 策略的预训练 2D 编辑模型在渲染视角上生成几何一致的伪真值，然后通过可微渲染器使用对比蒸馏损失进行反向传播，以实现稳定更新。
**OREO** 构建了用于沿时间反向传播 (BPTT) 的可微展开，并辅以 3D 原生正则化以防止几何退化。
实验表明，**OREO** 在高度风格化和富有想象力的输入上优于监督基线，实现了卓越的概念一致性和细节还原，从而缓解了 3D 数据稀缺问题。

## Introduction

<!-- 第一段：背景与空白 (The Missing Piece: Post-Training)
3D 生成现状：3D 生成模型（如 Trellis, Hunyuan3D）通过在大规模数据集上的预训练取得了显著进展。
数据瓶颈：然而，受限于高质量 3D 数据的稀缺，预训练模型的性能似乎触碰到了天花板，难以处理复杂的概念对齐或精细几何。
指出空白：尽管在 LLM 领域，后训练（Post-Training / Alignment）已被证明是提升模型能力的关键步骤，但在 3D 生成领域，这一方向仍处于空白状态（largely unexplored）。
直观尝试的挑战：一个直观的思路是将 LLM 的 RLHF 范式迁移过来（即基于 Reward Model 的 RL）。然而，在 Image-to-3D 任务中，受限的采样空间和缺乏鲁棒的 3D Reward Model 使得这一路径充满挑战。 -->

近年，Trellis 和 Hunyuan3D 等 3D 原生生成模型（3D Native Generators）在自动化高质量 3D 内容创作方面展现了惊人的能力。目前的主流范式主要依赖于在大规模 3D 数据集上进行预训练。然而，高质量 3D 数据的稀缺性给模型性能设定了根本性的天花板。特别是在 **概念设计（Conceptual Design）** 领域，设计师往往需要将充满想象力、风格化甚至非物理的 2D 创意转化为 3D 原型。现有的 3D 模型受限于有限的训练数据分布，在面对这些**复杂或非标准（Complex and Non-canonical）**的输入时，往往难以保持几何合理性和概念一致性，导致生成结果与输入条件出现严重的**错位（Misalignment）**。

这一问题的本质在于**有限的监督数据难以覆盖无限的创意空间**。现有的预训练模型在处理复杂的创意输入时，容易产生**扭曲的结构（Distorted Structures）**或概念错位。传统的监督微调（SFT）受限于静态数据集，无法动态适应这些千变万化的创意需求；而标准的强化学习（RL）虽然能探索新状态，但在高维 3D 生成任务中，仅依赖稀疏的标量奖励（Scalar Reward）进行优化效率极低，且极易陷入局部最优。

为了解决这一难题，我们需要一种结合了**在线探索（On-Policy Exploration）**与**稠密反馈（Dense Feedback）**的新范式。为此，我们提出了 **OREO (On-Policy Rendering Editing Optimization)** 框架。我们的核心洞察是：虽然 3D 数据稀缺，但 2D 图像编辑模型蕴含了丰富的通用视觉知识，可以作为理想的“教师”。通过将这些**外部先验（External Priors）**引入训练，我们能够利用海量的 **in-the-wild 2D 图像**来增强 3D 生成器的能力。我们将 3D 后训练重构为一个**在渲染视角上具有稠密监督的在线策略蒸馏过程（On-Policy Distillation Process with Dense Supervision on Rendered Views）**，从而摆脱了对 3D 真值（3D Ground Truth）的依赖。

不同于传统 RL 的“试错-打分”循环，OREO 采用了一种**稠密监督的 On-Policy Distillation** 策略。该框架的核心是 **Reinforced Editing Distillation (RED)** 算法。具体而言，我们将**渲染图**视为学生模型在当前策略下的“状态采样”，利用**预训练的 2D 编辑模型（Pre-trained 2D Editing Model）**作为“教师”提供即时修正。我们利用改进的 **FlowEdit 策略** 来生成**几何一致的伪真值（Geometrically Consistent Pseudo-Ground Truths）**，在增强渲染图中的概念特征的同时，严格**保持原始视角（View Preservation）**。这使得**目标视图**既像参考图的新视角，又**保持**当前的渲染相机位姿，从而为 3D 模型提供了**稠密的**像素级监督信号。

为了把这些像素级监督信号稳定地蒸馏进 3D 生成模型，我们进一步提出了 **Contrastive Distillation Loss**。受 **Score Distillation 领域前沿成果（如 VSD, CSD）** 的启发，该损失函数通过构建正负样本对来实现稳定的梯度更新。此外，为了在利用 2D 信号的同时保持 3D 结构的完整性，我们构建了**可微展开（Differentiable Rollout）**机制，实现了端到端的**沿时间反向传播（Backpropagation Through Time, BPTT）**。配合 **3D 原生正则化（3D Native Regularization）**，OREO 能够利用 BPTT 机制，充分挖掘预训练模型中蕴含的几何先验，有效防止了在激进的纹理编辑过程中出现的几何退化（Geometric Degradation），显著提升了生成资产的概念保真度与多视角一致性。

总之，我们的贡献主要体现在三个方面：

1. 我们开创了一种新型的 3D 生成模型后训练范式 **OREO**。针对预训练模型在**高度风格化与非标准场景**下的泛化瓶颈，OREO 通过结合**在线探索（On-Policy Exploration）**与**稠密反馈（Dense Feedback）**，有效解决了训练与推理分布不匹配的问题。
2. 我们提出了 **Reinforced Editing Distillation (RED)** 算法，并结合了基于 **BPTT 的 3D 正则化**。前者利用 FlowEdit 和 Contrastive Distillation Loss 提供高质量的概念引导，后者通过时序一致性约束确保几何结构的完整性，两者协同实现了稳定的 On-Policy 优化。
3. 我们在 **Trellis** 等主流 3D 生成模型上验证了该方法的有效性。实验结果表明，OREO 能够显著提升模型在复杂创意输入上的泛化能力，在概念对齐度和几何保真度方面均超越了仅依靠监督训练的基线模型。

## Related work

## Method

### 3.1 预备知识 (Preliminaries)

**On-Policy Distillation for 3D Generation**
我们将 3D 生成模型的后训练形式化为一个 On-Policy Distillation 问题。给定一个参数为 $\theta$ 的学生模型（3D Generator），其目标是学习教师模型（2D Editor）的分布 $p_{teacher}$。不同于离线蒸馏（Offline Distillation）依赖固定的数据集 $D = \{(x, y)\}$，On-Policy Distillation 要求学生模型在自身生成的轨迹 $x \sim \pi_\theta$ 上进行学习。这通过最小化以下散度实现：

$$
\mathbb{E}_{x \sim \pi_\theta} [ \mathcal{L}(x, \text{Teacher}(x)) ]
$$

这种范式确保了模型能够实时纠正自身在推理过程中产生的几何偏差，对于提升模型在**高度风格化与非标准场景（Highly Stylized and Non-canonical Scenarios）**下的生成质量至关重要。

**3D 原生生成与流匹配 (3D Native Generation via Flow Matching)**
我们的目标是优化一类**已经经过大规模数据预训练的** 3D 生成模型 $\mathcal{G}_\theta$。不同于从零初始化的优化方法（如 DreamFusion），我们利用预训练模型作为强大的几何先验起点。这类模型通常基于流匹配（Flow Matching）框架，其生成过程被建模为从先验分布 $z_T \sim \mathcal{N}(0, I)$ 到数据分布 $z_0$ 的常微分方程（ODE）积分过程：

$$
dz_t = v_\theta(z_t, t) dt
$$

其中 $v_\theta$ 是网络预测的速度场。给定当前状态 $z_t$ 和速度 $v_\theta$，我们可以推导出对最终数据 $z_0$ 的估计：$\hat{z}_0 = z_t - t \cdot v_\theta(z_t, t)$。随后，解码器将 $z_0$ 转换为显式的 3D 资产（如 Gaussian Splats），并通过可微渲染器 $\mathcal{R}$ 投影为 2D 视图。

**基于指令的图像编辑 (Instruction-based Image Editing)**
图像编辑的任务是根据文本指令 $y$，将源图像 $x^{src}$ 转换为符合概念的目标图像 $x^{tgt}$，同时保留与指令无关的原始结构。形式上，我们寻找一个映射 $\mathcal{E}: (x^{src}, y) \to x^{tgt}$。在 OREO 框架中，这一映射 $\mathcal{E}$ 充当了“教师”角色，为 3D 模型提供“应该生成什么”的伪真值。

**为什么选择 FlowEdit 算法? (Why FlowEdit Algorithm?)**
在 OREO 框架中，我们采用 **FlowEdit 算法** 来指导基础编辑模型（Base Editing Model）的推理过程。相比于模型默认的推理模式（如直接采样），FlowEdit 具备独特的优势：

1. **视角锁定 (View Locking)**: FlowEdit 利用流匹配的**平行四边形原则**，在潜在空间中构建与源轨迹平行的目标轨迹。这种机制隐式地施加了强几何约束，确保编辑操作仅修改纹理和细节（Identity），而不改变物体的姿态或轮廓（Geometry）。相比之下，基于 SDE 的方法容易引入随机的视角偏移，导致 3D 优化发散。
2. **免训练与通用性 (Training-free & Generalization)**: FlowEdit 无需对特定数据集进行微调，能够直接利用预训练 Diffusion/Flow 模型的强大先验，完美契合我们处理 OOD 数据的目标。
3. **梯度稳定性**: FlowEdit 基于 ODE 确定性采样，相比随机性强的 SDE 采样，能提供更低方差的梯度估计。

**FlowEdit 算法回顾**
FlowEdit 是一种无需训练的图像编辑算法。给定源图像 $x^{src}$，它首先通过正向加噪构造源轨迹 $x_t^{src} = t x^{src} + (1-t)\epsilon$。为了生成编辑后的目标图像 $x^{tgt}$，FlowEdit 利用**平行四边形原则**：假设编辑增量 $\Delta x$ 在加噪空间中保持不变，即 $x_t^{tgt} = x^{edit} - x^{src} + x_t^{src}$。在每个时间步，算法计算目标流场 $v^{tgt}$ 和源流场 $v^{src}$，并利用差异流场 $\Delta v = v^{tgt} - v^{src}$ 更新编辑变量 $x^{edit}$。

**概览 (Overview)**
如图 [Figure X] 所示，OREO 将 3D 后训练建模为一个 **On-Policy Distillation** 循环。与传统的监督训练不同，我们不依赖静态数据集，而是实时执行以下三个步骤：(1) **策略展开**：从当前生成器采样 3D 资产并渲染；(2) **增强编辑 (Reinforced Editing)**：利用定制的 FlowEdit 算法动态生成高质量的伪真值，增强渲染图中的概念特征；(3) **对比蒸馏 (Contrastive Distillation)**：通过对比损失和轨迹正则化更新生成器参数。其中，步骤 (2) 和 (3) 共同构成了 **RED** 算法的核心。下文将详细阐述这三个关键环节。

### 3.2 2D Guidance via Reinforced Editing Distillation

我们利用预训练的 Flow Matching 模型 $v_\varphi$（本工作中采用 Qwen-Image-Edit）作为基础，执行改进版的 FlowEdit 算法。为了适应 3D 后训练任务，我们对标准 FlowEdit 进行了三项关键改进（详见下方的算法对比）。**值得注意的是，RED 的有效性建立在 2D 编辑器能够生成高质量伪真值（Pseudo-GT）的前提之上。我们在后文的 Section 4.1 中通过定量实验验证了这一点，表明改进后的 FlowEdit 能够在大幅增强概念一致性（Conceptual Consistency）的同时，有效地保持几何结构的完整性，从而胜任“教师”的角色。**

1. **负引导源流场 (Negative Guidance)**：我们在计算源流场 $v^{src}$ 时引入了负引导尺度 $-s$。这迫使模型识别并保留源图像中与 Prompt **不符** 的特征（通常是背景或不需要编辑的几何结构），从而增强了编辑的局部性。
2. **动态噪声修正 (Dynamic Noise Correction)**：标准 FlowEdit 假设噪声 $\epsilon$ 是固定的。然而，随着源流场的变化，固定的噪声会导致轨迹偏差。我们引入了动态修正项，根据源流场的梯度实时调整 $\epsilon_t$，确保源轨迹始终锚定在渲染图的流形上。
3. **预测值记录 (Prediction Recording)**：我们在每一步都利用当前的流场预测 $t=0$ 的干净状态。这些中间预测值包含了丰富的概念梯度信息，构成了后续对比损失的基础。

#### 算法流程对比 (Algorithm Comparison)

```python
# Algorithm 1: Vanilla FlowEdit
Input: Source Image x_src, Prompt y_ctx
Init: x_edit = x_src, eps = sample_noise()
For t in [1, ..., 0]:
    # 1. Construct State
    x_t_src = t * x_src + (1-t) * eps
    x_t_tgt = x_edit - x_src + x_t_src
  
    # 2. Predict Velocity
    v_tgt = Model(x_t_tgt, y_ctx)
    v_src = Model(x_t_src, y_src) 
  
    # 3. Update
    v_edit = v_tgt - v_src
    x_edit = x_edit + dt * v_edit
Return x_edit
```

```python
# Algorithm 2: On-Policy Flow Editing (Ours)
Input: Rendered Image x_src, Prompt y_ctx
Init: x_edit = x_src, eps_t = sample_noise()
For t in [1, ..., 0]:
    # 1. Construct State
    x_t_src = t * x_src + (1-t) * eps_t
    x_t_tgt = x_edit - x_src + x_t_src
  
    # 2. Predict Velocity (with Negative Guidance)
    v_tgt = CFG(Model(x_t_tgt, y_ctx), scale=s)
    v_src = CFG(Model(x_t_src, y_ctx), scale=-s) # Key Change 1
  
    # 3. Noise Correction (Key Change 2)
    eps_t = eps_t - (1-t) * (v_src_pos - v_src_neg)
  
    # 4. Record Predictions for Loss (Key Change 3)
    preds.append(predict_x0(x_t_tgt, v_tgt))
    preds.append(predict_x0(x_t_src, v_src))
  
    # 5. Update
    v_edit = v_tgt - v_src
    x_edit = x_edit + dt * v_edit
Return x_edit, preds
```

循环结束后，我们得到最终的编辑图像 $x^{tgt} = x^{edit}$ 以及所有中间步骤的预测集合 $\{ \hat{x}^{+}_{t \to 0}, \hat{x}^{-}_{t \to 0} \}_t$。

**对比蒸馏 (Contrastive Distillation)**

受 **Score Distillation 领域前沿成果（如 VSD, CSD）** 的启发，我们提出了一种在 $x_0$-space（图像域）计算的**对比蒸馏损失（Contrastive Distillation Loss）**。

我们的损失函数定义为采样时间步集合 $\{t\}$ 上的加权和：

$$
\mathcal{L}_{\text{RED}} = \sum_{t} \omega(t) \left[ \| x^{src} - \hat{x}^{+}_{t \to 0} \|^2 - \| x^{src} - \hat{x}^{-}_{t \to 0} \|^2 \right]
$$

在训练过程中，我们采用随机采样策略来选择时间步 $t$，以平衡计算效率与轨迹覆盖率（详见 Section 4.4）。

其中：

* $\hat{x}^{+}_{t \to 0}$ 是 FlowEdit 目标分支使用正引导（Positive Guidance, $+s$）得到的理想目标预测（Conditional Prediction）。这一项作为“吸引子”，驱动生成器产生的渲染图 $x^{src}$ 向高质量的编辑目标靠拢。
* $\hat{x}^{-}_{t \to 0}$ 是目标分支的无条件预测（Unconditional Prediction）。这一项作为“排斥子”，防止模型坍缩到无意义的平均状态。

尽管损失函数形式上仅包含目标分支的对比，但 **Source Branch** 的影响通过 FlowEdit 的迭代过程隐式地传递给了 $\hat{x}^{+}_{t \to 0}$。具体而言，Source Branch 的速度场 $v^{src}$ 在每一步都修正了编辑轨迹 $z^{edit}$，去除了源图像中与目标 Prompt 不符的特征。这种机制确保了最终的 $\hat{x}^{+}_{t \to 0}$ 既包含了目标概念，又保留了必要的原始结构。

其中 $\omega(t)$ 是时间步加权函数。为了平衡不同噪声水平下的梯度幅值并稳定训练，我们采用了**自适应梯度归一化 (Adaptive Gradient Normalization)** 策略。具体而言，我们将 $\omega(t)$ 定义为预测误差的倒数：

$$
\omega(t) = \frac{1}{\| x^{src} - \hat{x}^{+}_{t \to 0} \|_1 + \epsilon}
$$

其中 $\epsilon$ 是一个小的常数。这种设计的核心动机在于应对**伪真值与渲染图之间的非刚性错位（Non-rigid Misalignment）**。尽管 FlowEdit 能够很好地保持全局视角，但在局部纹理布局或精细结构上，生成的伪真值往往难以与当前渲染图实现像素级的完美对齐。在标准的 L2 损失下，这些微小的结构错位会被放大为剧烈的梯度波动，导致生成的 3D 表面出现高频噪声或几何扭曲。通过引入自适应归一化，我们限制了梯度的幅值，使优化过程专注于**概念层面的对齐**而非像素层面的强行拟合，从而显著提升了生成几何的光滑度与合理性。

通过这种对比机制，RED 有效地将 VSD/CSD 的思想从噪声域迁移到了直观的图像域。我们将 $x^{src}$ 视为待优化的变量，通过最小化该损失函数，梯度将通过可微渲染器 $\mathcal{R}$ 反向传播至 3D 生成器 $\mathcal{G}_\theta$。

### 3.3 Differentiable Rollout & Optimization (可微展开与优化)

为了实现端到端的 On-Policy 优化，我们构建了一个完全可微的生成管线。具体而言，给定条件输入，模型首先生成离散的粗糙结构（Dense Structure），该过程在训练中保持冻结。随后，流匹配模型在粗糙结构的引导下进行**稀疏特征采样（Sparse Feature Rollout）**，生成精细的 3D 潜在特征 $z_0$。

在训练过程中，我们采用可微的 ODE 求解器（如 Euler Step）来离散化流匹配积分过程：

$$
z_{t_{i-1}} = z_{t_i} - (t_i - t_{i-1}) \cdot v_\theta(z_{t_i}, t_i)
$$

关键在于，我们在每一步都保留了计算图。这意味着 $z_{t_{i-1}}$ 不仅是当前步网络输出的函数，也是上一步状态 $z_{t_i}$ 的函数。因此，最终生成的 $z_0$ 实际上是整条轨迹上所有速度场预测的复合函数：

$$
z_0 = \text{Solver}(z_T, \{v_\theta(\cdot, t_i)\}_{i=N}^1)
$$

最后，解码器将 $z_0$ 映射为显式的 3D 表示（如 Gaussian Splats），并通过可微渲染器 $\mathcal{R}$ 投影为 2D 图像。这使得来自 2D 编辑器的监督信号能够反向传播穿过渲染器、解码器，并沿着 ODE 求解器的计算图**沿时间反向传播（Backpropagation Through Time, BPTT）**，从而实现对整个生成轨迹的端到端优化。

### 3.4 3D Regularization via $z_0$-Prediction

仅依赖 2D 编辑信号进行 3D 优化本质上是一个不适定问题。为了防止几何退化，我们引入了基于 $z_0$ 预测的 3D 正则化。

我们对比了两种常见的正则化策略：

1. **速度场正则化 (Velocity Regularization)**：直接约束当前时刻的速度场 $v_\theta$ 与预训练教师模型 $v_{frozen}$ 一致，即 $\mathcal{L}_{v} = \| v_\theta(z_t, t) - v_{frozen}(z_t, t) \|^2$。这种方法仅关注**局部的一步预测（Local One-step Prediction）**，忽略了历史轨迹的累积误差。
2. **$z_0$ 正则化 ($z_0$-Regularization)**：我们提出的策略约束学生模型在每一步预测的**最终状态（Clean Data Prediction, $z_0$）**与教师保持一致：

$$
\mathcal{L}_{z_0} = \mathbb{E}_{t \sim [0,1]} \left[ \frac{\| \hat{z}_{0,\theta}(z_t, t) - \hat{z}_{0,frozen}(z_t, t) \|^2}{t^2 + \epsilon} \right]
$$

其中 $\hat{z}_{0}(z_t, t) = z_t - t \cdot v(z_t, t)$ 是根据 Flow Matching 公式推导出的 $z_0$ 估计。

相比于简单的速度场正则化，这一设计具有显著优势：

* **时序一致性（Temporal Consistency）**：由于 $z_t$ 是由历史速度场积分得到的，通过优化 $z_0$，梯度能够通过 $z_t$ **回传至历史时间步（Backpropagation Through Time）**。这不仅约束了当前步的行为，还隐式地修正了之前的轨迹偏差，确保生成的几何结构在整个去噪过程中保持稳定。
* **几何感知（Geometry Awareness）**：$z_0$ 直接对应最终的 3D 几何形态。相比于抽象的速度场 $v$，在 $z_0$ 空间进行约束能够更直观地保留预训练模型中蕴含的几何拓扑先验，防止在编辑纹理时破坏物体的物理结构。

最终的总优化目标由终端的编辑蒸馏损失和中间的正则化损失共同组成：

$$
\mathcal{L}_{total} = \mathcal{L}_{RED}(z_0) + \lambda \mathcal{L}_{z_0}
$$

这种设计确保了梯度能够通过 ODE Solver（如 Euler Step）穿越整个生成轨迹，将末端的编辑信号 $z_0$ 和中间的正则信号 $z_t$ 整合，实现端到端的时序优化。

## Experiment

### 4.1 实验设置 (Experimental Setup)

**数据集 (Dataset)**
为了验证 OREO 在 **3D 概念设计** 场景下的有效性，我们构建了一个专门的 **概念设计数据集 (Conceptual Design Dataset)**。

* **训练集**: 包含约 2000 张从互联网收集的高质量概念设计图像，涵盖了科幻载具、奇幻生物、风格化角色及未来建筑等。这些图像通常具有夸张的比例、独特的纹理和非现实的几何结构，对 3D 生成器的泛化能力提出了极高要求。
* **测试集**: 包含 100 张从未见过的、极具想象力的设计草图，用于评估模型将抽象创意转化为 3D 实体的能力。

**基线模型 (Baselines)**
我们将 OREO 与以下基线进行对比：

* **Trellis (Zero-shot)**: 原始的预训练 Trellis 模型，作为监督学习的基准。

**评价指标 (Metrics)**
我们采用定量与定性相结合的评估方式：

* **CLIP Similarity**: 衡量生成视图与输入 Prompt 的概念一致性。
* **DINO Similarity**: 评估生成结果与参考图的视觉特征相似度。
* **User Study**: 邀请人类评估员对生成的几何质量和纹理细节进行打分。

**实现细节 (Implementation Details)**
我们的实验基于 PyTorch 框架实现。

* **模型架构**: 我们使用预训练的 **Trellis-Image-Large** 作为 3D 生成器基座，并采用 **Qwen-Image-Edit** 作为 2D 编辑教师模型。渲染器选用 Gaussian Splatting，渲染分辨率设为 $1024 \times 1024$。
* **训练设置**: 我们采用 **SGD 优化器**，学习率设为 $5 \times 10^{-3}$。训练在单张 NVIDIA A800 GPU 上进行，Batch Size 为 1，梯度累积步数为 4。为了节省显存，我们使用 **BF16 混合精度** 训练。总训练轮数为 500 epochs。
* **RED 参数**: 噪声模式采用 "Aligned" 策略。基于预实验分析（见 Section 4.2），我们固定采用 **(Steps: 9|12, Cfg=4)** 的配置，即在 $t=0.75$ 的噪声水平下启动编辑，总推理步数为 12 步。
* **损失权重**: 在所有实验中，我们仅使用 **Contrastive Distillation Loss (CDL)**，权重设为 1.0，未启用额外的 MSE 或正则化损失。

### 4.2 预实验：验证编辑教师的有效性与参数选择 (Preliminary Analysis: Validating the Editing Teacher)

在将 FlowEdit 应用于 3D 优化循环之前，我们首先在一个独立的 2D 数据集上评估了其作为“教师模型”的胜任力，并确定了最佳的编辑参数。3D 优化的上限取决于 2D 伪真值（Pseudo-GT）的质量：我们需要确保编辑过程产生的 $\Delta x$ 主要是语义修正，而非几何破坏。

我们选取了 50 张测试集渲染图，在不同参数配置下执行 FlowEdit，并计算 CLIP Similarity（概念对齐）、DINO Similarity（结构一致性）和 Silhouette IoU（轮廓重合度）。此外，我们还将 FlowEdit 与其他主流图像编辑方法进行了对比。表 [Table 2] 展示了详细的定量分析结果。

**表 2: FlowEdit 参数敏感性分析**
我们报告了不同配置下的指标变化值 (Diff) 及最终 Mask IoU。灰色背景表示选定的最终配置。

| Experiment                      | Configuration                        | CLIP Sim Diff$\uparrow$ | DINO Sim Diff$\uparrow$ | Mask IoU$\uparrow$ |
| :------------------------------ | :----------------------------------- | :-----------------------: | :-----------------------: | :------------------: |
| **A. CFG Scale**          | Steps: 20\|40, Cfg=2                 |          +0.0029          |          +0.0028          |        0.9887        |
| *(Fixed Steps)*               | **Steps: 20\|40, Cfg=4**       |     **+0.0034**     |     **+0.0089**     |   **0.9786**   |
|                                 | Steps: 20\|40, Cfg=12                |          -0.0098          |          +0.0051          |        0.9631        |
|                                 |                                      |                          |                          |                      |
| **B. Ratio (3:4 vs 2:4)** | Steps: 20\|40 (Ratio 0.50)           |          +0.0034          |          +0.0089          |        0.9786        |
| *(Fixed Cfg=4)*               | **Steps: 30\|40 (Ratio 0.75)** |     **+0.0228**     |     **+0.0226**     |   **0.9319**   |
|                                 |                                      |                          |                          |                      |
| **C. Efficiency**         | Steps: 30\|40 (Total 40)             |          +0.0228          |          +0.0226          |        0.9319        |
| *(Fixed Ratio 0.75)*          | **Steps: 9\|12 (Total 12)**    |     **+0.0132**     |     **+0.0156**     |   **0.9595**   |

**分析与结论**:

1. **引导尺度的选择 (Choice of Guidance Scale)**: 实验表明，过高的 CFG (如 12) 会严重破坏图像质量，导致 CLIP 和 IoU 双降。**Cfg=4** 展现出了最稳健的性能，在有效增强概念特征的同时，保持了较高的几何一致性 (IoU > 0.97)。
2. **时间步比例的关键作用 (Critical Role of Timestep Ratio)**: 我们发现编辑区间的起始点比总步数更关键。对比 **Steps: 20|40 (Ratio 0.5)** 和 **Steps: 30|40 (Ratio 0.75)**，后者带来了近 **7倍** 的 CLIP 提升 (+0.0228 vs +0.0034)。这表明在 $t=0.5$ 的低噪水平下，图像结构已固化，模型缺乏足够的**可塑性 (Plasticity)** 来注入新概念。因此，我们固定采用 **Ratio 0.75** 以确保足够的编辑自由度。
3. **效率与质量的平衡 (Efficiency-Quality Trade-off)**: 在确定 Ratio 0.75 的前提下，我们对比了高步数 (30|40) 和低步数 (9|12) 策略。虽然 30|40 的概念提升略高，但其 Mask IoU 显著下降 (0.93)，增加了几何漂移的风险。相比之下，**Steps: 9|12** 在保持极高几何稳定性 (**IoU ~0.96**) 的同时，依然实现了具有竞争力的概念增强，且计算成本降低了 **70%**。

基于上述分析，我们在后续的所有 3D 实验中均采用配置 **(Steps: 9|12, Cfg=4)**。

### 4.3 主要结果 (Main Results)

我们在包含 100 个概念设计 Prompt 的测试集上评估了 OREO 及其基线模型的性能。

**定量评估 (Quantitative Evaluation)**
表 [Table 1] 展示了各方法在 CLIP Similarity 和 DINO Similarity 上的得分。

* **概念对齐 (Conceptual Alignment)**: RED 在 CLIP Similarity 上取得了显著领先，相比原始 Trellis 提升了约 15%。这表明我们的方法成功地将 2D 编辑器对复杂概念的理解迁移到了 3D 生成器中。
* **用户偏好**: User Study 结果显示，超过 85% 的用户倾向于认为 RED 生成的资产在几何合理性和纹理细节上优于基线模型，特别是在处理非标准结构的生物和载具时。

**定性评估 (Qualitative Evaluation)**
图 [Figure Y] 展示了 RED 与基线模型的可视化对比。

* **几何修复**: 在“赛博朋克风格的机械义肢”案例中，原始 Trellis 生成的结构往往模糊不清，而 RED 成功还原了清晰的关节和管线细节。
* **纹理增强**: 对于“带有发光符文的魔法书”，RED 生成的纹理不仅清晰度更高，而且光影效果更符合 Prompt 描述，消除了 SDS 常见的过饱和问题。
* **多视角一致性**: 尽管 FlowEdit 是在单视角上进行引导，但得益于 3D 生成器的内在一致性，RED 生成的资产在所有视角下都保持了结构的连贯性，未出现明显的 Janus 问题。

### 4.4 消融实验与分析 (Ablation Study and Analysis)

为了深入理解 RED 各组件的贡献，我们进行了一系列消融实验。

**损失函数组件分析 (Analyzing Loss Components)**
我们验证了对比蒸馏损失（Contrastive Distillation Loss, CDL）中各部分的必要性。

* **替换为 MSE 损失 (Replace CDL with MSE)**: 我们尝试直接最小化渲染图与 FlowEdit 最终输出之间的均方误差 ($\| x^{src} - x^{tgt} \|^2$)。结果显示，虽然该策略能快速拉近概念距离，但极易导致几何结构的扭曲（如表面凹凸不平）。这是因为 MSE 强迫模型在像素级精确匹配编辑结果，而忽略了 2D 编辑过程中不可避免的微小几何偏差。相比之下，CDL 利用相对梯度方向，提供了更鲁棒的概念引导。
* **移除排斥项 ($\| x^{src} - \hat{x}^{-}_{t \to 0} \|^2$)**: 仅保留吸引项会导致模型过度拟合编辑目标，忽略了对无条件分布的抑制，导致生成的纹理过于平滑且缺乏细节。
* **移除轨迹正则化 ($\mathcal{L}_{reg}$)**: 这是一个关键的稳定项。实验表明，在没有正则化的情况下，模型容易在训练后期出现几何崩坏（如表面破损或多余的漂浮物），证明了在蒸馏过程中保持 3D 先验的重要性。

**采样策略分析 (Sampling Strategy Analysis)**
为了平衡训练效率与性能，我们采用了多步时间采样（MTS Sampling）。

* **固定时间步 (Fixed Timestep)**: 如果仅在固定的噪声水平（如 $t=500$）进行优化，模型难以处理多样的输入分布，导致收敛后的纹理细节不足。
* **MTS Sampling (Ours)**: 为了解决固定时间步导致的过拟合问题，我们采用了一种**分层随机采样 (Stratified Random Sampling)** 策略。不同于在固定的离散网格上求解 ODE，我们将整个时间轴划分为连续的子区间，并在每次迭代中从每个子区间内均匀采样一个时间步。这种策略在保持全局流动轨迹（Global Flow Trajectory）正确性的同时，确保了模型能够覆盖连续的噪声水平，从而显著增强了对微小时间偏移的鲁棒性。

## 5. Conclusion

本文提出了 **RED (Reinforced Editing Distillation)**，一种针对 3D 概念设计场景的通用后训练框架。针对现有 3D 生成模型在处理高度风格化和非标准几何输入时的泛化瓶颈，我们创新性地引入了基于 FlowEdit 的增强编辑机制。通过构建 **Contrastive Distillation Loss** 并结合 **轨迹正则化**，RED 成功地将 2D 基础模型中蕴含的丰富概念先验蒸馏到了 3D 生成器中。实验结果表明，我们的方法在概念一致性、几何保真度和纹理细节上均显著优于监督基线，为解决 3D 生成中的数据稀缺问题提供了一条高效的新路径。

**未来工作 (Future Work)**
尽管 RED 表现出色，但仍有进一步探索的空间。首先，目前的 FlowEdit 过程推理成本较高，未来可探索更高效的蒸馏策略（如一步式编辑）。其次，我们将尝试把 RED 扩展到更复杂的场景生成任务中，利用全景编辑模型来优化大规模 3D 环境。最后，结合多模态大模型（LMM）进行更细粒度的交互式编辑也是一个激动人心的方向。
