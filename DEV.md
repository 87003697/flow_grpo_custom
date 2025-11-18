- DiffusionNFT vs. Flow-GRPO（算法差异概要）
  - 公共流程：两者均采用“采样 → 打分 → 优势归一化 → 策略更新 + KL 约束”的两阶段训练范式，区别在于优势统计域、策略表示与损失形式。
  - DiffusionNFT：
    - 核心是“LoRA 双适配器”（`default`/`old`）结构：一个分支参与训练，另一个作为历史策略，通过 `return_decay` 以 EMA 方式更新旧分支，可移植到任意 transformer。
    - 策略更新不显式使用 log-prob，而是构造正/负预测：`v^+ = β v_new + (1-β) v_old`、`v^- = (1+β) v_old - β v_new`，对应 `\hat{x}_0^± = x_t - τ v^±`。
    - 将样本权重映射到 `[0,1]` 后在正/负 `(x_0 - \hat{x}_0)` 回归之间软路由，整体 loss 由正回归、负回归与 KL 三部分组成，偏向“直接回归 + 参考模型约束”的训练路径。
  - Flow-GRPO：
    - 保留标准 PPO 管线：采样阶段显式记录每步 `old_log_probs`，训练阶段计算 `ratio = exp(logπ_θ - logπ_old)`，无特定于某模型的假设，可用于任意带 log-prob 的生成器。
    - 损失为标准 PPO：`L = -E[max(-A·ratio, -A·clip(ratio, 1-ε, 1+ε))] + β·KL`，同时跟踪 `approx_kl`、`clipfrac` 等诊断指标。

- 关键差异速览
  - 策略采样：DiffusionNFT 只需保存基础输入条件、对应时间步与最终干净 latent，再配合奖励 future 供正/负 `x_0` 回归使用；Flow-GRPO 额外保留整个 latent 轨迹及逐步 `log_probs`，以便计算 PPO ratio 与 KL 诊断。
  - 训练路径：DiffusionNFT 以正/负 `x_0` 回归为主、KL 只约束参考模型；Flow-GRPO 采用 PPO ratio + clip 的对数概率损失并跟踪 `clipfrac/approx_kl`。
  - 模型结构：DiffusionNFT 需要能切换/禁用 LoRA adapter 的 transformer；Flow-GRPO 只依赖模型能返回 log-prob，适配单阶段或多阶段生成器。
  - 阶段划分：Stage1/Stage2 仅来自 Direct3D 模型本身的流水线设计，与选择 Flow-GRPO 或 DiffusionNFT 无关，两种训练范式都能沿用相同步骤。
  - Direct3D 改造：将 `scripts/train_direct3d_s2_stage-1+2.py` 切换到 DiffusionNFT 时，需要为 Stage1/Stage2 引入 LoRA `default/old` 双适配器与 `return_decay` EMA，采样阶段仅缓存 prompt 条件/时间步/`x_0` 与奖励 future，优势可以用简单 z-score 或 prompt 级聚合实现，训练 loss 用优势映射 `[0,1]` 后的正/负 `x_0` 加权回归 + 参考 KL，并移除 `compute_log_prob_direct3d_stage*`、PPO ratio、`clipfrac/approx_kl` 等依赖，保持 Stage 间几何/渲染耦合不变。
  - Pipeline 封装：为 Direct3D 的 Stage1/Stage2 增加 `stage1_forward_step`、`stage2_forward_step` 等统一接口，屏蔽噪声调度与输入差异，使训练循环能像 DiffusionNFT 那样在旧/新 adapter 间切换，并复用正/负 `x_0` 回归和参考 KL 逻辑。


- 参考文件（DiffusionNFT）：
  - `_reference_codes/DiffusionNFT/scripts/train_nft_sd3.py`
    - `return_decay`（L169-L189）：旧策略 EMA 系数。
    - 采样循环（L586-L704）：old/default adapter 切换、`samples_data_list` 收集。
    - 优势聚合（L744-L783）与批处理（L793-L828）：prompt 级归一化及 rebatch。
    - Loss 计算（L927-L980）：正/负 `x_0` 加权回归 + 参考 KL。
    - EMA 回写（L1028-L1039）：更新 old adapter。
  - `_reference_codes/DiffusionNFT/flow_grpo/diffusers_patch/pipeline_with_logprob.py`
    - `pipeline_with_logprob`（L25-L269）：采样阶段返回图像、Latent 序列与 `log_probs` 的入口。
  - `_reference_codes/DiffusionNFT/flow_grpo/stat_tracking.py`
    - `PerPromptStatTracker.update/clear`（L5-L41）：prompt 级均值/方差与优势生成。
  - `_reference_codes/DiffusionNFT/flow_grpo/ema.py`
    - `EMAModuleWrapper`（L8-L91）：DiffusionNFT 使用的 EMA 封装，支持 copy/sync。

