## Direct3D S2：Stage1+Stage2 联训改造方案（SDE+独立优化器）

### 背景与目标
- 当前 `scripts/train_direct3d_s2_stage-1+2.py` 仅训练 Stage 2（稀疏流 `sparse_dit_512`），Stage 1（稠密分支 `dense_dit`）冻结，仅用于生成候选稀疏坐标。
- 目标：同时训练 Stage 1 和 Stage 2。
  - 为 Stage 1 适配 SDE rollout 与 logprob，接口对齐 Stage 2。
  - 在训练脚本中为 Stage 1 增设独立优化器（超参复用 Stage 2）。

---

### 变更一：在 `direct3d_s2_pipeline_with_logprob.py` 为 Stage 1 适配 SDE rollout

目标：在现有 Stage 2 接口基础上，为 Stage 1（稠密分支）提供与 Stage 2 类似的带 logprob 的 SDE 采样/回放能力，便于计算每步 logprob 和（可选）KL 奖励，支撑 GRPO/ppo 风格训练。

现状参考点：
- 稀疏/Stage 2 的整批 SDE 循环在：
  - `Direct3DS2PipelineWithLogProb.stage2_with_logprob(...)`
  - 单步：`direct3d_flow_step_with_logprob(...)`（在 `flow_grpo/diffusers_patch/direct3d_s2_sparse_tensor.py`）
- 稠密/Stage 1 目前仅用于生成坐标，入口：
  - `Direct3DS2PipelineWithLogProb.forward_stage1(...)`（内部调用参考管线 `ref.inference(..., mode='dense')` 返回每图 `coords`）

改造要点：
1) 新增稠密分支 SDE+logprob 的单步函数（Dense 版）：
   - 新增函数（放在 `direct3d_s2_pipeline_with_logprob.py` 内部）：
     - `def flow_step_with_logprob_dense(scheduler, sample, model_output, timestep, prev_timestep, generator, deterministic, noise_level=0.7) -> (prev_sample, log_prob_vec, prev_sample_mean, std_vec)`
   - 数学与接口对齐 `direct3d_flow_step_with_logprob`，但作用于稠密特征张量：`sample: Tensor(BK, C, R, R, R)`、`model_output: Tensor(BK, C, R, R, R)`，其中 `BK = B × K`；返回 `prev_sample/prev_sample_mean` 同形状；`log_prob_vec: Tensor(BK,)`。
   - 关键公式（与 Stage 2 相同的 sigma 域 SDE 推导）：
     - `sigma, sigma_prev = scheduler.sigmas[idx_t], scheduler.sigmas[idx_t+1]`
     - `dt = sigma_prev - sigma`
     - `std_dev_t = sqrt(sigma / (1 - sigma_cmp)) * noise_level`
     - `step_std = std_dev_t * sqrt(max(-dt, eps))`
     - `prev_mean = x * (1 + (std^2/(2*sigma))*dt) + v * (1 + std^2*(1-sigma)/(2*sigma)) * dt`
     - 随机分支：`prev = prev_mean + step_std * N(0, I)`；确定性分支直接取 `prev_mean`
    - `logprob = 按 BK 聚合(mean over spatial/channel)`，维度为 `(BK,)`

2) 新增稠密分支的整批 SDE 循环接口：
   - 新增 `def stage1_with_logprob(self, cond_batched, neg_batched, num_inference_steps, guidance_scale, generator=None, deterministic=False) -> Tuple[coords_list, latents_seq_dense, log_prob_seq_dense, t_seq]`
   - 输入：
     - `cond_batched: Tensor(BK,P,C)`，`neg_batched: Optional[Tensor(BK,P,C)]`（与 Stage 2 对齐）；BK=B×K 由上游已 repeat_interleave 就绪。
   - 流程：
     - 初始化稠密 latent：`noise ∼ N(0,I)` 形状 `(BK,C,R,R,R)`；设置 `self.ref.dense_scheduler.set_timesteps(...)`。
     - 时间步循环：调用 `flow_step_with_logprob_dense`，得到 `latents_seq_dense`（长度 steps+1）、`log_prob_seq_dense`（堆叠后形状 `(steps, BK)`）。
     - 结束后按参考管线 indexing/掩码策略将最终 latent 转为 `coords_list`（长度 BK，每项 `(N_i,4)`），接口与原 `forward_stage1` 对齐。
  - 输出建议：
    - `coords_list`: List[Tensor[(N_i,4)]] 与当前一致
    - `latents_seq_dense`: List[Tensor]（稠密 latent 的批序列，用于训练对齐/teacher forcing）
    - `log_prob_seq_dense`: Tensor[(steps, BK)]
    - `t_seq`: Tensor[(steps+1,)]

3) 说明：此函数对齐 Stage 2 的 `direct3d_flow_step_with_logprob` 数学与接口，作用于稠密张量，返回 `(prev_sample, log_prob_vec, prev_sample_mean, std_vec)`，其中 `log_prob_vec` 维度为 `(BK,)`。

#### Stage 1 vs Stage 2 接口差异对比（关键点）
- 单步函数对比：
  - Stage 2（已存在）：`direct3d_flow_step_with_logprob`
    - 输入：
      - `sample: SparseTensor`  // 稀疏，带 `coords(N_total,4)` 与 `layout` 切片
      - `model_output: SparseTensor`  // 稀疏，形状与 `sample` 对齐
      - `timestep, prev_timestep: float`  // 标量
      - 其余：`scheduler, generator, deterministic, noise_level`
    - 输出：
      - `prev_sample: SparseTensor`  // 稀疏，逐候选共享 `coords`
      - `log_prob_vec: Tensor(BK,)`  // 按 `layout` 聚合，BK=候选数之和
      - `prev_sample_mean: SparseTensor`  // 稀疏
      - `std_vec: Tensor(BK,)`
  - Stage 1（新增）：`flow_step_with_logprob_dense`
    - 输入：
      - `sample: Tensor(BK,C,R,R,R)`  // 稠密 latent 特征（3D 体素），BK=B×K
      - `model_output: Tensor(BK,C,R,R,R)`  // 稠密模型输出速度场
      - `timestep, prev_timestep: float`  // 标量
      - 其余：`scheduler, generator, deterministic, noise_level`
    - 输出：
      - `prev_sample: Tensor(BK,C,R,R,R)`  // 稠密
      - `log_prob_vec: Tensor(BK,)`  // 按 batch 维聚合（无 layout）
      - `prev_sample_mean: Tensor(BK,C,R,R,R)`  // 稠密
      - `std_vec: Tensor(BK,)`

- 批循环接口对比：
  - Stage 2：`stage2_with_logprob(cond/neg_cond:(BK,P,C), coords: SparseTensor[batched]) → (meshes: List[Any], latents_seq: List[SparseTensor], log_prob_seq: Tensor(steps,BK), t_seq: Tensor(steps+1))`
  - Stage 1：`stage1_with_logprob(cond_batched: Tensor(BK,P,C), neg_batched: Optional[Tensor(BK,P,C)], ...) → (coords_list: List[Tensor(N_i,4)], latents_seq_dense: List[Tensor(BK,C,R,R,R)], log_prob_seq_dense: Tensor(steps,BK), t_seq: Tensor(steps+1))`


##### compute_log_prob 接口详细对比
- Stage 2：`compute_log_prob_direct3d_stage2(pipeline, samples: List[dict], j: int, config: Stage2RuntimeConfig) -> Tuple[SparseTensor, Tensor(BK,), Tensor(BK,)>`
  - 输入样本 `samples[k]`（k∈[0,BK)）字段：
    - `latents_seq`: List[SparseTensor]，长度 `T+1`；取 `j` 与 `j+1` 作为 `batched_current` 与 `observed_prev` 的来源
    - `cond_patches`: Tensor(1,P,C)，`neg_patches`: Optional[Tensor(1,P,C)]
    - `t_seq`: np.ndarray(T+1,) 或 Tensor(T+1,)
  - 内部：
    - 将 `latents_seq[j]` 与 `latents_seq[j+1]` 合批为 `SparseTensor`（BK 批），拼接 `cond/neg` 为 `(BK,P,C)`
    - `scheduler = pipeline.ref.sparse_scheduler_512`
    - 调 `direct3d_flow_step_with_logprob(..., observed_prev_sample=batched_prev, deterministic=config.deterministic)` 得 `log_prob_vec: (BK,)`、返回 `prev_sample_batched`
    - `kl_vec` 当前为零向量（保留扩展点）
  - 输出：
    - `prev_sample_batched: SparseTensor(batched)`、`log_prob_vec: Tensor(BK,)`、`kl_vec: Tensor(BK,)`

- Stage 1：`compute_log_prob_direct3d_stage1(pipeline, samples: List[dict], j: int, config: Stage1RuntimeConfig) -> Tuple[Tensor(BK,C,R,R,R), Tensor(BK,), Tensor(BK,)>`（新增）
  - 输入样本 `samples[k]`（k∈[0,BK)）字段：
    - `latents_seq_dense`: List[Tensor(C,R,R,R)]，长度 `T+1`；取 `j` 与 `j+1` 作为当前与观测前一帧
    - `cond_patches`: Tensor(1,P,C)，`neg_patches`: Optional[Tensor(1,P,C)]
    - `t_seq`: np.ndarray(T+1,) 或 Tensor(T+1,)
  - 内部：
    - 将 `latents_seq_dense[j]` 与 `latents_seq_dense[j+1]` 沿 batch 合并为 `(BK,C,R,R,R)`；拼接 `cond/neg` 为 `(BK,P,C)`
    - `scheduler = pipeline.ref.dense_scheduler`
    - 调 `flow_step_with_logprob_dense(..., observed_prev_sample=batched_prev, deterministic=config.deterministic)` 得 `log_prob_vec: (BK,)`、返回 `prev_sample_batched: (BK,C,R,R,R)`
    - `kl_vec` 当前为零向量（保留扩展点）
  - 输出：
    - `prev_sample_batched: Tensor(BK,C,R,R,R)`、`log_prob_vec: Tensor(BK,)`、`kl_vec: Tensor(BK,)`
    
- B×K 处理规范（Stage 1）：
  - 条件输入：上游已构造成 `cond/neg_cond: (BK,P,C)`（与 Stage 2 相同）。
  - 初始噪声：直接按 `BK` 构造 `(BK,C,R,R,R)`。
  - 输出坐标：`coords_list` 为长度 `BK` 的列表（每图 K 个候选，各自独立索引/排序形成 `(N_i,4)`）。
  - 聚合维：`log_prob_vec` 与 `std_vec` 在 `BK` 维；无 `layout`（稠密）。

关于 k 参数：
- 不作为 `stage1_with_logprob` 的显式输入。K 由 `cond/neg_cond` 的第一维（BK）隐式确定，和 B 共同决定 BK=B×K；上游负责将条件扩展为 BK。

- CFG 与条件：
  - Stage 2：对稀疏张量使用 `sparse_tensor_cfg_guidance`，cond/neg 为 patch 级 `(BK,P,C)`。
  - Stage 1：对稠密张量按通道/空间逐元线性合成：`neg + s*(pos-neg)`，cond/neg 为稠密条件（由 `ref.dense_image_encoder` 输出，形状与模型期望一致），聚合维度为 `(B,)`。

- 布局与聚合：
  - Stage 2：通过 `SparseTensor.layout: List[slice]` 按候选聚合，产生 `BK` 维度的 logprob 与 std。
  - Stage 1：按 BK 维度聚合（无 layout），直接得到 `BK` 维度的 logprob 与 std。

- 时间步与调度器：
  - 二者均使用 `FlowMatchEulerDiscreteScheduler` 与同一 `timesteps/sigmas` 序列，差异仅在数据形态（稀疏 vs 稠密）。

改动位置（文件/函数名）：
- `flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`
  - 新增：`flow_step_with_logprob_dense(...)`
  - 新增：`stage1_with_logprob(...)`
  - 保留：`forward_stage1(...)`（作为旧路径参考）

---

### 变更二：在 `scripts/train_direct3d_s2_stage-1+2.py` 为 SDE 增加 Stage 1 的独立优化器

目标：与 Stage 2 分离的优化器/梯度流，便于分时/交替回传以降低峰值显存；超参设置与 Stage 2 保持一致（不新增独立配置项）。

配置与超参约定：
- 不新增任何 `stage1` 独立配置项。
- Stage 1 的学习率、权重衰减、梯度累计、EMA 开关与衰减等，全部复用 Stage 2 的相同字段（如 `config.train.learning_rate/weight_decay/gradient_accumulation_steps/ema/ema_decay`）。
- 稠密分支的步数与 Stage 2 保持一致：使用 `config.sample.num_steps`。

改造步骤：
1) 挑选可训练参数：
   - `dense_model = pipeline.ref.dense_dit`
   - 若使用 LoRA，复用 `apply_lora_if_needed` 逻辑创建 `dense_model_lora`（与 Stage 2 同步策略）。

2) 构建独立优化器并通过 `accelerator.prepare` 同步（超参取自 Stage 2 的相同字段）：
   - 参考当前包装：
     - 现有 Stage 2 路径：`prepare_optimizer_and_wrap(...)` 返回 `slat_model, optimizer`
   - 新增 Stage 1：
     - `dense_trainable_params = [p for p in dense_model.parameters() if p.requires_grad]`
     - `optimizer_stage1 = build_optimizer(dense_trainable_params, config)`（内部读取与 Stage 2 相同字段）
     - 包装顺序示例：`dense_model, optimizer_stage1, slat_model, optimizer_stage2 = accelerator.prepare(dense_model, optimizer_stage1, slat_model, optimizer_stage2)`
   - 回写到 `pipeline.ref.dense_dit = dense_model`，确保推理与训练一致。

3) 训练循环中交替/分时步骤（默认交替）：
  - 采样阶段（两阶段均使用 SDE）：
    - Stage 1：调用 `stage1_with_logprob(..., deterministic=False)`，内部 `flow_step_with_logprob_dense` 使用 SDE，得到 `coords_list` 与 `latents_seq_dense/log_prob_seq_dense`。
    - Stage 2：调用 `stage2_with_logprob(..., deterministic=False, slat_sampler_params=SlatSamplerParams(use_sde=True, ...))`，得到 mesh 与 `latents_seq/log_prob_seq`。
   - 训练阶段：
     - Stage 2：维持现有 `compute_log_prob_direct3d_stage2(...)` 与 PPO 损失计算、回传、`optimizer_stage2.step()`。
     - Stage 1：新增 `compute_log_prob_direct3d_stage1(...)`（接口形似 stage2 版本，接受 `latents_seq_dense/t_seq/cond`），计算 logprob/kl 与 PPO 损失，回传、`optimizer_stage1.step()`。
       - 注：`compute_log_prob_direct3d_stage1` 可先作为 `pipeline` 的内部方法实现，直接使用 `_dense_flow_step_with_logprob` 和 `self.ref.dense_scheduler`，避免引入新的模块依赖。
     - 两阶段的反传应串行执行，并在两者之间执行 `optimizer.zero_grad(set_to_none=True)` 与 `torch.cuda.empty_cache()`，降低峰值显存。
   - EMA：若启用，为两套可训练参数分别维护 EMA，衰减与开关读取与 Stage 2 相同字段。

4) 日志与监控：
   - 新增指标前缀：`stage1/`（如 `stage1/train_loss`、`stage1/kl_mean`、`stage1/approx_kl`、`stage1/policy_loss`）。
   - 采样可视化沿用当前机制（mesh 由 Stage 2 产生；Stage 1 可选记录稠密中间可视化开关）。

改动位置（文件/函数名）：
- `scripts/train_direct3d_s2_stage-1+2.py`
  - 复用：`build_optimizer(...)`、`apply_lora_if_needed(...)`、`prepare_optimizer_and_wrap(...)`（可拆分或新增 `prepare_optimizer_stage1(...)`）
  - 新增：构建 `dense_dit` 的优化器与 `accelerator.prepare` 包装；交替/分时的训练分支（仅当 `train.stage1.enable`）。
  - 新增：`compute_log_prob_direct3d_stage1(...)` 的调用（先作为 `pipeline` 方法；如后续复用可抽出到 `direct3d_s2_sparse_tensor.py` 的密集版实现）。

---

### 变更三：验证与调试脚本（Stage2 对照 + Stage1 方案）

以下以 `scripts/debug/test_direct3d_s2_infer_v2.py` 的 Stage 2 测试流程为基准，给出对应的 Stage 1 测试方案与一一对照。Stage 1 的验证与调试将新建独立脚本：`scripts/debug/test_direct3d_s1_infer_v2.py`（专注稠密分支，但在最后一步调用 Stage 2 的 ODE 解码导出 mesh 以便人工检查）。

- Stage 2（现有脚本关键步骤）
  - 单步一致性（SDE/ODE）
    - 使用 `direct3d_flow_step_with_logprob` 在相邻 `(t, t_prev)` 上前进一步，返回 `(prev_sample, log_prob_vec, prev_sample_mean, std_vec)`。
    - 手工按高斯公式计算并核对 `log_prob_vec`（聚合维度 BK）。
  - 管线采样
    - 通过 `_build_stage1_for_image` 得到 `stage1_cond_dict = {cond/neg_cond:(BK,P,C), coords: SparseTensor(batched)}`。
    - 调用 `pipeline.stage2_with_logprob(...)`，得到 `meshes`, `latents_seq: List[SparseTensor](len=steps)`, `log_prob_seq: Tensor(steps,BK)`, `t_seq: Tensor(steps+1)`。
  - 结果校验与复现
    - `validate_sampling_outputs` 断言 `log_prob_seq` 步数与 BK 维，SDE 非零/ODE 为零；`latents_seq` 长度等于 steps。
    - `reproducibility_check` 固定 `coords` 与种子，两次运行 `log_prob_seq` 完全一致。
  - 差异分析与策略复算
    - `compare_single_step` 对比同步长下 SDE/ODE 的 `prev/mean/std/log_prob` 差异量级。
    - `check_grpo_policy_sampling` 使用 `compute_log_prob_direct3d_stage2`（teacher forcing: `observed_prev_sample`）逐步复算并与采样记录对比（容差 1e-4）。
  - 导出
    - `export_meshes` 将输出 mesh（必要时用 `_decode_sparse_mesh`）保存到目录。

- Stage 1（新增脚本思路，对齐替换）
  - 单步一致性（SDE/ODE）
    - 使用 `flow_step_with_logprob_dense` 在相邻 `(t, t_prev)` 上前进一步，输入 `sample/model_output: Tensor(BK,C,R,R,R)`；返回 `(prev_sample, log_prob_vec, prev_sample_mean, std_vec)`，`log_prob_vec: (BK,)`。
    - 同样按高斯公式核对 `log_prob_vec`（聚合维 BK）。
  - 管线采样（稠密分支）
    - 上游构造 `cond_batched/neg_batched: (BK,P,C)`；生成初始噪声 `noise: (BK,C,R,R,R)`；`self.ref.dense_scheduler.set_timesteps(steps)`。
    - 调用 `pipeline.stage1_with_logprob(cond_batched, neg_batched, steps, guidance_scale, generator, deterministic)`，得到：
      - `coords_list: List[Tensor(N_i,4)]`（长度 BK）
      - `latents_seq_dense: List[Tensor(BK,C,R,R,R)]`（长度 steps+1）
      - `log_prob_seq_dense: Tensor(steps,BK)`，`t_seq: Tensor(steps+1)`
    - 使用 `coords_list` 与同一批 `cond/neg_cond` 调用 Stage 2 的解码（ODE）：
      - `pipeline.stage2_with_logprob(num_inference_steps=sparse_steps, guidance_scale, generator, deterministic=True, slat_sampler_params=SlatSamplerParams(use_sde=False, mc_threshold=...), stage1_cond_dict={cond/neg_cond, coords=batched_from_coords_list})` → 得到 `meshes`；将 `meshes` 导出到指定目录。
  - 结果校验与复现
    - 断言 `log_prob_seq_dense.shape == (steps-1, BK)`；SDE 非零、ODE 为零；`latents_seq_dense` 长度等于 steps+1。
    - 复现性建议两种方式（二选一）：
      - 固定初始噪声 `noise` 与种子，两次运行 `log_prob_seq_dense` 一致；
      - 或在函数内仅使用传入 `generator` 采样噪声，固定种子复现。
  - 策略复算（GRPO 对齐）
    - 计划新增 `compute_log_prob_direct3d_stage1(...)`：输入 `samples`（包含 `latents_seq_dense[j], latents_seq_dense[j+1], cond/neg 切片 (1,P,C), t_seq`），内部用 `flow_step_with_logprob_dense(..., observed_prev_sample=latents_seq_dense[j+1])` 逐步复算 `(BK,)`；与 `log_prob_seq_dense[j]` 比较（容差 1e-4）。
  - 对照映射表（Stage2 → Stage1）
    - 单步：`direct3d_flow_step_with_logprob` → `flow_step_with_logprob_dense`
    - 批采样：`stage2_with_logprob` → `stage1_with_logprob`
    - 复算：`compute_log_prob_direct3d_stage2` → `compute_log_prob_direct3d_stage1`（新增）
    - 条件：`cond/neg_cond: (BK,P,C)`（两者相同）
    - 聚合维：`BK`（两者相同）


最小运行（训练验证不依赖此脚本，仅用于推理/自检）：
```bash
# Stage 2 验证（现有脚本）
python -u scripts/debug/test_direct3d_s2_infer_v2.py \
  --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
  --image dataset/eval3d_hunyuan3d/images/004.png \
  --out outputs/test_runs/direct3d_s2_validation \
  --candidates 2 --dense_steps 50 --sparse_steps 30 --guidance 7.0 \
  --seed 777 --dtype fp16 --use_sde --do_e2e

# Stage 1 验证（新脚本，稠密 + 使用 Stage2 ODE 解码导出 mesh）
python -u scripts/debug/test_direct3d_s1_infer_v2.py \
  --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
  --image dataset/eval3d_hunyuan3d/images/004.png \
  --out outputs/test_runs/direct3d_s1_validation \
  --candidates 2 --dense_steps 50 --sparse_steps 30 --guidance 7.0 \
  --seed 777 --dtype fp16 --use_sde --do_e2e --ode_decode
```

说明：`test_direct3d_s1_infer_v2.py` 实现 Stage 1 单步一致性、整批 SDE/ODE、复现与策略复算（基于 `compute_log_prob_direct3d_stage1`），并在末尾使用 Stage 2 的 ODE（`deterministic=True` 且 `use_sde=False`）对 `coords_list` 进行解码生成 `meshes` 并导出，便于人工检查几何质量。



### 实施要点（精简）
- 稠密分支：实现 `direct3d_flow_step_with_logprob_dense` 与 `stage1_with_logprob`，对齐 Stage 2 的接口与时间步循环。
- 训练脚本：为 `dense_dit` 构建独立优化器；超参数与步数复用 Stage 2；交替反传 Stage 2/Stage 1。

---

### 代码参考位置
- 稀疏/Stage 2 采样入口（SDE 批处理与 logprob）：
  - `flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py:408-523`
- 稀疏单步（SDE + logprob）：
  - `flow_grpo/diffusers_patch/direct3d_s2_sparse_tensor.py:41-118`
- 训练脚本（构建/包装/训练 Stage 2）：
  - 优化器准备与回写：`scripts/train_direct3d_s2_stage-1+2.py:919-933, 1084-1092`
  - 采样与 PPO 主循环：`scripts/train_direct3d_s2_stage-1+2.py:1122-1474`

以上改造完成后，即可在不破坏默认行为的前提下，开启 Stage 1+2 联训；若后续需要将 `compute_log_prob_direct3d_stage1` 提升为通用模块，再抽取为 `flow_grpo/diffusers_patch/direct3d_s2_dense_tensor.py` 等独立文件也可。



