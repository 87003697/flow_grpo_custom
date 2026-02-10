# Post-Training 3D Native Generators with On-Policy Feedback from Image Editting

## Abstract

尽管 3D 生成模型（如 Trellis, Hunyuan3D）展现出巨大潜力，但高质量 3D 数据的极度稀缺导致依赖监督训练的模型难以理解复杂语义，生成的 3D 资产经常出现与输入条件严重不符（Misalignment）的问题。

为了突破这一限制，我们提出了 **Editing Enforcement Matching (EEM)**，一种通用的 3D 生成模型后训练（Post-Training）方法。EEM 利用**视角投影（View Projection）与可微渲染**作为桥梁，将 3D 优化问题转化为 2D 空间的 **On-Policy Distillation** 过程，从而能够直接利用强大的 2D 视觉先验来指导 3D 生成。具体而言，我们利用图像编辑模型作为语义校准器，对 3D 渲染图像执行基于 FlowEdit 的编辑，构建出符合 Prompt 描述的目标状态。受 Score Distillation 的启发，我们将这一 2D 修正信号转化为一种**基于编辑的对比损失（Edit-based Contrastive Loss）**：该损失函数不仅驱动 3D 模型向编辑后的目标状态靠拢，同时将其推离原始的未对齐状态。这种双向对比信号通过可微渲染管线反向传播，高效地校准 3D 模型在 Latent 空间中的生成行为。

实验结果表明，EEM 能够显著提升生成结果与输入条件的对齐度，在语义一致性和细节还原上均超越了仅依靠 Supervised Training 的基线模型，为解决 3D 生成中的对齐难题提供了一条高效的新路径。


## Introduction

<!-- 第一段：背景与空白 (The Missing Piece: Post-Training)
3D 生成现状：3D 生成模型（如 Trellis, Hunyuan3D）通过在大规模数据集上的预训练取得了显著进展。
数据瓶颈：然而，受限于高质量 3D 数据的稀缺，预训练模型的性能似乎触碰到了天花板，难以处理复杂的语义对齐或精细几何。
指出空白：尽管在 LLM 领域，后训练（Post-Training / Alignment）已被证明是提升模型能力的关键步骤，但在 3D 生成领域，这一方向仍处于空白状态（largely unexplored）。
直观尝试的挑战：一个直观的思路是将 LLM 的 RLHF 范式迁移过来（即基于 Reward Model 的 RL）。然而，在 Image-to-3D 任务中，受限的采样空间和缺乏鲁棒的 3D Reward Model 使得这一路径充满挑战。 -->

近年来，Trellis 和 Hunyuan3D 等 3D 生成模型在自动化高质量 3D 内容创作方面展现了惊人的能力。目前的主流范式主要依赖于在大规模 3D 数据集上进行预训练。然而，高质量 3D 数据的稀缺性给模型性能设定了根本性的天花板。具体而言，3D 训练数据有限的多样性导致了泛化能力（Generalization）的匮乏：虽然模型在分布内样本上表现尚可，但在适应现实世界中广泛且多样的输入图像分布时，往往难以保持高质量的生成效果。虽然后训练策略（如 RLHF）在 LLM 中已被证明有效，但在 3D 领域仍处于空白。直接迁移强化学习（RL）面临一个概念上的错位：RL 通常擅长挖掘模型在预训练阶段已经获取的潜在先验，但难以注入全新的知识。 在 3D 生成的语境下，由于数据稀缺，基座模型往往根本缺乏必要的几何或纹理先验，仅靠 RL 去“发现”正确的 3D 结构是远远不够的。

<!-- 第二段：提出 EEM 与 On-Policy Distillation
核心逻辑：为了解决泛化问题（Inject New Knowledge），我们提出 EEM。
机制：利用图像编辑模型（Teacher）对 3D 渲染图（Student）进行实时修正。
范式转变：从 RL 的“标量奖励最大化”转变为“On-Policy Distillation”，利用稠密的像素级监督信号进行高效知识迁移。 -->

为了弥合这一泛化差距，我们将目光投向了 2D 扩散模型中蕴含的丰富知识。我们的核心洞察是，通过利用 Qwen-Image-Edit 等先进图像编辑器并配合免训练（Training-free）的引导策略，我们可以有效地修正 3D 渲染图中的视觉瑕疵，同时保留其原始结构。这实现了 3D 后训练的范式转变——从最大化模糊的标量奖励转变为 On-Policy Distillation（在线策略蒸馏）。在这个框架中，当 3D 模型生成渲染图时，我们的引导编辑过程将其修正为更优的视觉状态，作为动态的“伪真实值”。为了充分利用这种稠密的像素级监督，我们提出了 Editing Enforcement Matching (EEM)。与标准蒸馏不同，EEM 制定了一种受 Score Distillation 启发的 Edit-based Contrastive Loss。该损失函数构建了一个双向优化场：它显式地强制（Enforce） 3D 模型匹配高质量的编辑目标（吸引），同时将其从原始的未对齐渲染中推离（Repel）（排斥）。这种机制提供了清晰且稳定的梯度信号，使 3D 模型能够快速内化 2D 编辑器的泛化能力

<!-- 第三段：核心技术与 EEM 命名
逻辑流：为了实现范式 -> 设计优化目标 -> 提出 Edit-based Contrastive Loss -> 解释双向机制 -> 命名为 EEM。
重点：强调 Loss 的设计（受 SDS 启发，双向对比）是核心，EEM 是这一整套算法的名称。 -->
为了有效地实现这一范式，我们受 Score Distillation Sampling (SDS) 的启发，制定了一个鲁棒的优化目标。我们不是简单地最小化与编辑目标的距离（这可能是不稳定的），而是构建了一个 Edit-based Contrastive Loss。该损失函数创建了一个双向优化场：它显式地强制（Enforce） 3D 模型匹配高质量的编辑目标（吸引），同时将其从原始的未对齐渲染中推离（Repel）（排斥）。通过利用编辑过程“前”与“后”状态之间的对比，这种机制提供了比朴素蒸馏更清晰、更稳定的梯度信号。我们将这一完整的算法称为 Editing Enforcement Matching (EEM)，它使 3D 模型能够快速修正几何畸变和纹理幻觉，即使在具有挑战性的分布外输入上也能实现卓越的对齐度和保真度。

<!-- 第四段：贡献总结
1. 范式创新：提出利用 2D 编辑先验进行 3D Post-Training。
2. 算法创新：提出 EEM 和 Edit-based Contrastive Loss。
3. 实验验证：在 Image-to-3D 任务上显著提升泛化性和生成质量。 -->

总之，我们的贡献主要体现在三个方面：
我们开创了一种新型的 3D 生成模型后训练范式，通过 On-Policy Distillation 利用 2D 图像编辑模型的丰富先验，绕过了对 3D 数据的需求。
我们提出了 Editing Enforcement Matching (EEM) 算法，配备了双向 Edit-based Contrastive Loss，能高效地将稠密编辑信号转化为用于 3D 优化的稳定梯度。
我们在具有挑战性的 Image-to-3D 任务上证明了该方法的有效性，表明 EEM 在泛化性、语义对齐度和几何保真度方面均显著优于监督基线。

## Related work

## Method

## Experiment
