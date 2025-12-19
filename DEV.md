## Trellis 适配 Gen2Turbo（单 renderer，必稠密结构，统一步数，CFG 全程开启）

目标：将 `_reference_codes/TriplaneTurbo_v2/custom/gen2turbo/systems/gen2turbo_system_trellis.py` 的算法骨架抽象到 `edit4shape/systems/trellis.py`，只保留单 renderer，训练/推理共用一套 rollout 逻辑。

### 配置与超参
- 单步数：`num_steps_sparse`（训练与推理同用）。
- 稠密结构：`num_steps_dense`，`generate_structure` 必须返回有效 coords，失败直接报错。
- CFG/正则：`guidance_scale`、`uncond_mode_rollout`、`uncond_mode_reg`、`reg_type`（"kl"/sds/csd 系列），`lambda_reg`、`lambda_distill`（无 lambda_sparsity）。
- 其余：种子、混精度、LoRA、optimizer/renderer/guidance/pipeline 子配置、日志/保存频率等沿用。

### Batch 命名与状态
- 批数据键：`Conditions`（原 condition_utils）、`Guidances`（原 guidance_utils），dataloader/组 batch 时需同步。
- `TrellisState` 保留视角占位类，包含 `conditions`/`guidance` 子对象与 `space_cache`。

### 系统构建
- `setup_env_and_seed`：同步 torch/cuda、np、random 的种子与确定性。
- `build_system`：实例化 pipeline（原 geometry）、renderer、guidance、optimizer（单 renderer），挂到 `System`。
- `prepare_lora`/`prepare_models_and_optimizers`：只包装 pipeline 与 optimizer 进入 accelerator。

### rollout（训练/评估共用的去噪函数）
输入：`batch`（含 `Conditions`）、`cfg`、`pipeline`、CFG 相关参数；可接受外部传入的 coords。
流程：
1) 结构：`coords = pipeline.generate_structure(Conditions, steps=num_steps_dense)`，强制存在，必要时 `unsqueeze(0)` 并转设备/类型。
2) embeddings：`cond_embeddings`、`uncond_embeddings` 均准备好（全程启用 CFG）。
3) 初始化与调度：`latents = pipeline.init_latents(batch_size, coords=coords, generator=seeded)`；`scheduler.set_timesteps(num_steps_sparse, device=...)`。
4) 循环（每步必跑 CFG）：cond/uncond 两路 `pipeline.denoise` → `mix_cfg(cond_pred, uncond_pred, guidance_scale, uncond_mode_rollout)` → `latents = scheduler.step(...).prev_sample`。
5) 返回：`{"latents": latents, "coords": coords}`。

### 训练专用补充
- 可在 `rollout_train`（或参数开关）里加入梯度检查点与逐步正则：
  - 单步函数经 checkpoint，输出 `(next_latents, final_pred, final_latent_ft)`，用 `scheduler_step_at_index` 保持时序一致。
  - 若 `lambda_reg > 0`，按 `reg_type` 选 `compute_kl_step_regularization` 或 `compute_score_distillation_step_regularization`，传入 `coords_for_training`、`guidance_scale`、`uncond_mode_reg`，累积伪损失和梯度范数。
- 渲染与损失：
  - 最终 latents → 稀疏（若有 `tokens_to_sparse`）→ `space_cache = pipeline.precompute_cache(...)`。
  - `out = renderer(**batch)`。
  - `guidance_loss = compute_guidance(out, batch, step=...)`（仅一个标量，内部可按子项加权）。
  - 总损失：`loss = guidance_loss + lambda_reg * reg_loss_mean + lambda_distill * distill_loss(如有)`。
- 反传（遵循 HF Accelerator）：
  - 如使用梯度累积，先 `loss /= grad_accum_steps`；`accelerator.backward(loss)`；在 `sync_gradients` 为真时 step/zero_grad，否则仅累积梯度。
  - 返回日志：`loss_total`、`loss_guidance`、`loss_reg_geom`、`grad_norm_reg` 等。

### 评估流程
- 使用通用 rollout（无正则、无梯度）得到 `latents/coords` → `space_cache` → `renderer(**batch)`。
- 不计算 guidance_eval，不调用旧的 `compute_guidance_n_loss`；可按需要汇总/保存渲染结果或简单指标。

### compute_guidance（替换 compute_guidance_n_loss）
- 输入：`out`（含 `comp_rgb`）、`batch`，可选 step。
- 操作：`guidance_rgb = out["comp_rgb"].permute(0,3,1,2)`；调用 guidance，聚合子项为单一 `guidance_loss`（可含蒸馏子项，用 `lambda_distill`）。
- 不再拆 fidelity/regularization，也不再有 sparsity 项。

### 形状注释约定
- 所有张量操作行添加 ASCII 形状注释，例如 `(B,T,C)`、`(1,T,4)`、`(B,S,C)`，保持简短一致。
