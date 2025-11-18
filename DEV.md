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
-  - 策略采样：DiffusionNFT 只需保存基础输入条件、对应时间步与最终干净 latent，再配合奖励 future 供正/负 `x_0` 回归使用；Flow-GRPO 额外保留整个 latent 轨迹及逐步 `log_probs`，以便计算 PPO ratio 与 KL 诊断。
  - 训练路径：DiffusionNFT 以正/负 `x_0` 回归为主、KL 只约束参考模型；Flow-GRPO 采用 PPO ratio + clip 的对数概率损失并跟踪 `clipfrac/approx_kl`。
  - 模型结构：DiffusionNFT 需要能切换/禁用 LoRA adapter 的 transformer；Flow-GRPO 只依赖模型能返回 log-prob，适配单阶段或多阶段生成器。

