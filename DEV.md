## Direct3D‑S2 x GRPO（简明版 TL;DR）

### 当前范围（已完成）
- **稀疏阶段 sparse512 + log_prob**：支持 SDE 采样与逐步 log_prob 累积；默认 eps 形式，可切换 x 形式。
- **最小训练骨架**：`scripts/train_direct3d_s2.py`（LoRA 只作用于 `sparse_dit_512`）。
- **配置与脚本**：`config/direct3d_s2_grpo_normal-sim.py`、单/多机与 lowmem 启动脚本。
- **依赖方式**：暂时直接引用 `_reference_codes/Direct3D-S2`（尚未迁移到 `generators/`）。

### 一条命令跑通
```bash
python scripts/train_direct3d_s2.py --config=config/direct3d_s2_grpo_normal-sim.py
```

### 关键配置项（只保留需要改的）
- **采样**：
  - `sample.num_inference_steps_dense` / `sample.num_inference_steps_sparse512`
  - `sample.guidance_scale`，`sample.use_sde`，`sample.sigma_min`，`sample.rescale_t`
  - `sample.num_candidates`
- **预训练**：`pretrained.pipeline_path`（包含 `config.yaml` 与 `model_*.ckpt`），`pretrained.minimal_512_only=True`
- **LoRA**：`use_lora=True`，`lora.lora_rank`

### 主要接口（仅记函数名与目的）
- `Direct3DS2PipelineWithLogProb.from_pretrained(...)`：加载 dense/sparse512 与调度器。
- `Direct3DS2PipelineWithLogProb.sample_candidates_with_logprob(...)`：Dense 得 `latent_index`，在 sparse512 上生成 K 个候选并返回 `(meshes, all_latents, all_log_probs, all_kl)`。
- `sde_step_with_logprob(...)`：单步 SDE 更新与 log_prob（默认 eps 形式）。

### 已知限制 / 后续
- 未迁移 `generators/direct3d_s2/`；训练循环为“最小可用”，后续再对齐 TRELLIS 全量指标与日志。
- 1024 阶段训练未接入（可用于评估）。

### 参考
- 细节与实现差异请见 `TRELLIS_n_DIRECT3D.md`。

---

## Direct3D‑S2 集成 GRPO 训练开发说明（DEV）

本文件描述如何将 Direct3D‑S2 集成到现有 GRPO 训练框架中。内容包括新增文件列表、接口与函数设计、参考代码映射、配置与启动脚本、风险与验证清单。所有实现需遵循仓库规范：

- 仅训练阶段 2（稀疏生成），不改动/微调阶段 1 的权重。
- 代码中避免使用 try/except 或任何 fallback。
- 每行张量运算需附带形状注释（例如：`# (B,C,H,W)` 或 `# (N_tokens,C)`）。


### 目标概述

- 将 Direct3D‑S2 的两阶段采样流程（dense → sparse512）接入 GRPO 训练闭环。
- 在稀疏阶段实现带对数概率累计的采样（with_logprob），与现有 `SD3/Hunyuan3D/TRELLIS` 的 GRPO 接口保持一致。
- 训练脚本提供与现有 `scripts/train_trellis.py` 相同的使用体验与日志/指标输出。


## 新增与修改的文件清单

- 新增：`flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`
- 新增：`flow_grpo/diffusers_patch/direct3d_s2_sde_with_logprob.py`
- 新增：`scripts/train_direct3d_s2.py`
- 新增：`config/direct3d_s2_grpo_normal-sim.py`
- 新增：`scripts/single_node/main_direct3d_s2_normal-sim.sh`
- 新增：`scripts/multi_node/main_direct3d_s2_normal-sim.sh`
-（如需 LoRA 适配）可修改：`flow_grpo/peft_sparse/sparse_lora.py`（注册 Direct3D‑S2 模块路径匹配）


## 接口与函数设计

### 1) `flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`

用途：包装 Direct3D‑S2 的三阶段采样，提供带 log_prob 的候选采样接口，供 GRPO 训练脚本调用。

建议结构：

```python
class Direct3DS2PipelineWithLogProb:
    """
    参考实现与对照：
    - Direct3D‑S2 参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py`
      - 类与构造/加载：`Direct3DS2Pipeline`、`from_pretrained`（1-172 行）
      - 设备迁移：`to(self, device)`（54-66 行）
      - 条件编码/稀疏封装：`encode_image`（194-217 行）
      - 采样主循环（CFG + scheduler.step + 可选 SDE）：`inference`（260-314 行）
      - 解码与后处理：`vae.decode_mesh` + `refiner_*`（320-341 行）

    - 现有 GRPO 管线参考：
      - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`
        - 采样与 logprob：294-352 行、341-347 行
      - `flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py`
      - TRELLIS 两阶段：`flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py`

    约束：
    - 不使用 try/except；每行张量操作附形状注释。
    """

    @classmethod
    def from_pretrained(cls, pipeline_path: str, subfolder: str = "direct3d-s2-v-1-1") -> "Direct3DS2PipelineWithLogProb":
        """
        参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py: from_pretrained`（68-172 行）
        行为：加载 dense/sparse 组件、encoders 与 schedulers；设置默认 dtype；保持 eval；不含 try/except。
        """

    def to(self, device: str) -> None:
        """
        参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py: to(self, device)`（54-66 行）
        行为：将所有子模块迁移到目标设备。
        """

    def sample_candidates_with_logprob(
        self,
        image,  # 输入图像（路径/ PIL / Tensor），外层调用负责批处理
        num_candidates: int,
        dense_params: dict,
        sparse_params_512: dict,
        # 仅支持 sparse512 阶段；暂不考虑 1024 阶段
        guidance_scale: float,
        use_sde: bool,
        sigma_min: float,
        rescale_t: float,
        generator,
    ) -> tuple[list, list, list, list]:
        """
        参考：
        - Dense 阶段：`inference(..., mode='dense')`（247-316 行）
        - Sparse 阶段：`inference(..., mode='sparse512')`（264-314 行）
        - CFG 合成：`noise_pred = uncond + w * (cond - uncond)`（279-286 行）
        - SDE 注入：291-313 行 的 `noise_strength` 与 eps 采样逻辑
        - SD3 单步对照：`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py: 341-347` 调 sde with logprob

        返回（四元组，对齐 TRELLIS）：
        - meshes: list[mesh]，长度= num_candidates
        - all_latents: List[Tensor]，展平后的时序，长度 = num_candidates*(steps+1)
        - all_log_probs: List[Tensor]，长度 = num_candidates*steps，每项形状 (1,)
        - all_kl: List[Tensor]，长度 = num_candidates*steps，每项形状 (1,)（Direct3D‑S2 阶段1占位 0）

        说明：
        - Stage1（dense → latent_index）仅执行一次；Stage2（sparse512）重复 num_candidates 次并累积 log_prob。
        - 张量操作需标注形状，例如：`latents = prev_mean + noise_strength * eps  # 同 prev_mean 形状`。
        """

    def _run_dense_stage(...):
        """
        内部函数；参考 `inference(..., mode='dense')`（247-316 行），输出 latent_index（返回索引）。
        """

    def _run_sparse_stage(...):
        """
        内部函数；参考 `inference(..., mode='sparse512')`（264-314、320-341 行），
        在每个扩散步进行 ODE 均值与可选 SDE 采样，并累计 log_prob。
        """

    @staticmethod
    def _gaussian_log_prob(eps, std):
        """
        计算 eps ~ N(0, std^2) 的对数概率；
        参考 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` 的实现风格与数值稳定性处理（17-80 行）。
        """
```

### Repro / P0+P1 选项说明（新增）

为解决 SDE 同种子 log_prob 不可复现的问题，管线新增以下选项（`PipelineOptions`）：

| 选项 | 默认 | 作用 |
|------|------|------|
| `logprob_reduction` | `mean_per_dim` | 避免不同 token 数量导致的大小偏置（原为 sum）。|
| `cache_dense_latent_index` | False | 首次 dense 阶段计算后缓存 latent_index。|
| `reuse_cached_dense_latent_index` | False | 强制复用已缓存 latent_index（Phase B 验证使用）。|
| `repro_print` | True | 打印 latent_index hash/唯一行数、noise_strength 统计。|
| `record_noise_strength` | True | 记录每步 SDE `noise_strength` 并输出统计。|

插桩输出示例：
```
[REPRO] dense latent_index cached shape=(N,3)
[REPRO] latent_index mode=new shape=(N,3) unique_rows=... hash=...
[REPRO] latent_index mode=reused shape=(N,3) unique_rows=... hash=...
[REPRO] noise_strength count=K*T min=... max=... mean=... std=...
```

测试脚本 `scripts/debug/test_direct3d_s2_infer.py` 复现性流程：
1. Phase A：`reuse_cached_dense_latent_index=False` -> 生成 & 缓存 latent_index。
2. Phase B：`reuse_cached_dense_latent_index=True` -> 复用缓存，比较 log_prob 严格相等。

若 Phase B log_prob 不一致，将打印差异索引与值片段；用于继续定位上游随机性来源。


### 2) `flow_grpo/diffusers_patch/direct3d_s2_sde_with_logprob.py`

用途：提供单步 SDE 采样并返回 step 级 log_prob，便于在稀疏阶段累计。

```python
def sde_step_with_logprob(prev_mean, t_cur, t_prev, rescale_t: float, sigma_min: float, generator):
    """
    单步 SDE 更新：`latents = prev_mean + noise_strength * eps`
    噪声强度：
      t_norm = clamp(t_cur / rescale_t, 0, 1)
      dt_abs = |(t_cur - t_prev)| / rescale_t
      sigma_t = sigma_min + (1 - sigma_min) * t_norm
      noise_strength = sigma_t * sqrt(max(dt_abs, 1e-8))

    修正后的 log_prob 公式（旧版漏掉 -n*log(noise_strength) 项）：
      令 x = prev_mean + s * eps, eps ~ N(0,I), s=noise_strength, n=元素总数
      log p(x|prev_mean,s) = -0.5 * eps^2.sum() - n*log(s) - 0.5*n*log(2π)

    函数额外返回：
      - eps.pow(2).sum() 与 n，用于上层自选 mean_per_dim 或其他归一化。
    """
```


### 3) `scripts/train_direct3d_s2.py`

用途：复用 `scripts/train_trellis.py` 的 GRPO 主循环，切换到 Direct3D‑S2 的 with_logprob 管线。

建议关键函数：

```python
def build_pipeline_and_models(config):
    """
    参考：
    - TRELLIS：`scripts/train_trellis.py`（导入与构建管线的方式，39-47 行，851-855 行 tracker 初始化）
    - Hunyuan3D：`scripts/train_hunyuan3d.py`（181-232 行计算 log_prob 的调用路径参考）
    - 管线：`flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`
    """

def grpo_sampling_step(accelerator, pipeline, batch):
    """
    参考：
    - TRELLIS：`scripts/train_trellis.py`（1149-1153 行附近 GRPO 张量对齐与 advantage/old_lp 使用）
    - SD3：`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`（294-352 行）
    行为：调用 `sample_candidates_with_logprob` 生成候选与 log_probs，并回填到样本字典。
    """
```


### 4) `config/direct3d_s2_grpo_normal-sim.py`

用途：对齐 `config/trellis_stage2_grpo_normal-sim.py`，提供以下关键字段（示例）：

- 采样与指导：`num_inference_steps_dense`、`num_inference_steps_sparse512`、`guidance_scale`、`use_sde`、`sigma_min`、`rescale_t`、`num_candidates`
- 训练：`train.batch_size`、`train.grad_accum`、`train.adv_clip_max`、学习率计划、保存频率
- 数据与奖励：`data_dir`、`camera_normal.cache_dir`、`reward.*`（沿用 `reward_models/rewards_mesh.py`）


### 5) 启动脚本

- `scripts/single_node/main_direct3d_s2_normal-sim.sh`
- `scripts/multi_node/main_direct3d_s2_normal-sim.sh`

对齐 `scripts/single_node/main_trellis_normal-sim.sh` 与 `scripts/multi_node/main_trellis_normal-sim.sh` 的参数与日志目录布局。


## 参考代码映射表（快速跳转）

- Direct3D‑S2：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py`
  - 构造/加载：1-172 行（`Direct3DS2Pipeline`, `from_pretrained`）
  - 设备迁移：54-66 行（`to`）
  - 条件编码/稀疏封装：194-217 行（`encode_image`）
  - 采样主循环（CFG + scheduler + 可选 SDE）：260-314 行（`inference` 主体）
  - 解码与后处理：320-341 行、386-400 行

- 现有管线（with_logprob）
  - SD3：`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`（294-352、341-347）
  - SD3 单步 SDE：`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`（17-80）
  - Hunyuan3D：`flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py`
  - TRELLIS Stage2：`flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py`

- 训练脚本参考
  - TRELLIS：`scripts/train_trellis.py`（采样/统计/GRPO 主循环）
  - Hunyuan3D：`scripts/train_hunyuan3d.py`

- 稀疏工具：`generators/trellis/patches/sparse_tensor_utils.py`
- 奖励模型：`reward_models/rewards_mesh.py`


## 形状与实现要点（必须遵循）

- CFG 融合：`noise = uncond + w * (cond - uncond)  # 形状同 cond`
- ODE 均值：`prev_mean = scheduler.step(noise, t, latents).prev_sample  # 同 latents 形状`
- SDE 注入：`latents = prev_mean + noise_strength * eps  # 同 prev_mean 形状`
- 稠密形状：`(B,C,H,W)`；稀疏形状：特征 `(N_tokens,C)` 与坐标 `(N_tokens,4)`。
- 稀疏阶段 step 级 log_prob 按元素求和，聚合为标量或 `(B,)`。


## 风险与边界

- 稀疏 token 数量波动导致显存与吞吐波动；建议首版仅启用 sparse512 训练，sparse1024 可作为评估或后续开关。
- SDE 参数（`rescale_t/sigma_min/dt`）需与参考实现一致，避免数值不稳。
- 随机数管理：候选之间独立生成器；保证可复现与 log_prob 一致性。
- LoRA 注入范围：仅对 `sparse_dit_512`（或 1024）相关层进行；其余模块 `eval()` 与冻结。


## Direct3D‑S2 非 Flow-Matching：SDE/ODE 与 LogProb 定义

Direct3D‑S2 并非 flow-matching，而是“基于调度器（scheduler）的扩散采样”。在 GRPO 中，我们将“带噪声的离散轨迹”当作策略，并据此定义 `log_prob`。核心做法：

- **ODE（确定性）**：仅执行 `scheduler.step(...)` 得到的 `prev_sample_mean`，不注入噪声，属于 δ 分布，无法产生有效 `log_prob`。因此 ODE 阶段不计 `log_prob`（可用于 Stage1 生成索引）。
- **SDE（建议用于 Stage2 稀疏生成）**：在每个扩散步上使用 Euler–Maruyama 离散化，向 `prev_sample_mean` 注入高斯噪声，并将该步噪声的对数密度累计到 `log_prob`。

### 参考实现位置（逐行对应）
- ODE 均值（调度器）：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py` 287-290 行
- SDE 噪声注入：同文件 291-313 行（`sigma_t/noise_strength/eps` 与 `latents = prev_sample_mean + noise_strength * eps`）
- 单步 `log_prob` 计算风格与数值细节可参考：`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`（17-80 行）

### 噪声与步长定义（标量随时间变）
设调度时间从大到小走：当前步为 \(t\)，前一时刻为 \(t'\)。参考 Direct3D‑S2：

\[ t_{\text{norm}} = \mathrm{clamp}(t/\mathrm{rescale\_t}, 0, 1) \]
\[ \Delta t = \left|\frac{t - t'}{\mathrm{rescale\_t}}\right| \]
\[ \sigma_t = \sigma_{\min} + (1 - \sigma_{\min})\cdot t_{\text{norm}} \]
\[ \text{noise\_strength} = \sigma_t \cdot \sqrt{\max(\Delta t, 1e\!-\!8)} \]

然后执行：

\[ x_{k+1} = \mu_k + \text{noise\_strength}\cdot\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I) \]

其中 \(\mu_k = \mathrm{scheduler.step}(\cdot).\text{prev\_sample}\)。

### `log_prob` 的两种等价写法
- 基于 \(\varepsilon_k\)：
  - 逐步累积 \(\log p(\varepsilon_k)\)，即标准正态密度的对数和。常数项 \(-\tfrac{D}{2}\log(2\pi)\) 与参数无关。
- 基于 \(x_{k+1}\mid\mu_k\)：
  - \(\log p(x_{k+1}\mid\mu_k) = \log \mathcal{N}(x_{k+1}; \mu_k, \text{noise\_strength}^2 I)\)，与上式仅差每步 \(-D\log \text{noise\_strength}\) 常数。因 \(\text{noise\_strength}\) 仅依赖时间步，不依赖可学习参数，对策略梯度无影响。

在实现中，使用 \(\varepsilon_k\) 累积更简洁稳定。

### 稠密与稀疏阶段的最小实现片段

- 稠密（Stage1，建议 ODE，不计 `log_prob`）：

```python
prev_mean = scheduler.step(noise_pred, t, latents).prev_sample  # (B,C,H,W)
latents = prev_mean  # (B,C,H,W)
# log_prob += 0.0
```

- 稀疏（Stage2/Stage3，用于 GRPO，计 `log_prob`）：

```python
prev_mean = scheduler.step(noise, t, feats).prev_sample  # (N_tokens,C)
t_prev = timesteps[i+1] if i+1 < len(timesteps) else t  # 标量
t_cur = torch.as_tensor(float(t), device=prev_mean.device, dtype=torch.float32)  # 标量
t_prev_t = torch.as_tensor(float(t_prev), device=prev_mean.device, dtype=torch.float32)  # 标量
t_norm = torch.clamp(t_cur / rescale_t, 0.0, 1.0)  # 标量
dt_abs = torch.abs((t_cur - t_prev_t) / rescale_t)  # 标量
sigma_t = sigma_min + (1.0 - sigma_min) * t_norm  # 标量
noise_strength = sigma_t * torch.sqrt(torch.clamp(dt_abs, min=1e-8))  # 标量

eps = torch.randn_like(prev_mean)  # (N_tokens,C)
latents_next = prev_mean + noise_strength.to(prev_mean.dtype) * eps  # (N_tokens,C)

step_log_prob = -0.5 * (eps.pow(2).sum() + eps.numel() * math.log(2*math.pi))  # 标量
log_prob = log_prob + step_log_prob  # 标量
feats = latents_next  # (N_tokens,C)
```

或等价从 \(x_{k+1}\) 回推 \(\varepsilon_k\)：

```python
eps_hat = (latents_next - prev_mean) / noise_strength  # (N_tokens,C)
step_log_prob = -0.5 * (eps_hat.pow(2).sum() + eps_hat.numel() * math.log(2*math.pi))  # 标量
```

### 实战建议
- 训练期：Stage1（dense）用 ODE 生成 `latent_index`；Stage2（sparse512）用 SDE 产生候选并累计 `log_prob`；Stage3（1024）可先关闭或仅评估。
- `log_prob` 聚合：
  - 稀疏 `(N_tokens,C)`：对元素求和为标量（或按样本 token 归属聚合为 `(B,)`）。
  - 稠密 `(B,C,H,W)`：若需要，也可按样本聚合为 `(B,)`，但我们的 GRPO 仅在稀疏阶段计 `log_prob`。
- CFG 不改变步级噪声协方差，仅改变均值 \(\mu_k\)，因此 `log_prob` 计算无需对 CFG 做额外修正。


## 里程碑与验收

## 测试与验收加固

### 单元测试（`sde_step_with_logprob`）
- 固定随机种子，验证两种 `log_prob` 写法仅差常数：
  - 基于 `eps` 与基于 `x_{k+1}` 回推 `eps_hat` 的 `log_prob` 差值应等于常数项（与维度相关，与可学习参数无关）。
  - 蒙特卡洛检验：重复采样 N 次，检验 `var(x_{k+1}-\mu)` 与理论 `noise_strength^2` 接近。
  - 形状/设备/精度校验：时间步与噪声强度计算保持 `float32` 且在目标设备上；输出张量形状与输入一致。

### 端到端最小样例
- 单图推理，`num_candidates ∈ {1,4}`：返回候选 `meshes` 与 `log_probs`（形状 `(B,num_candidates)`）。
- 保存中间统计到日志：每步 `noise_strength`、`eps` 范数分布、`log_prob` 的分位数（p1/p50/p99）。
- 关闭 SDE（退化为 ODE）时，不累计 `log_prob`，输出应与参考实现的确定性结果一致。

### 分布式一致性检查（DDP/Accelerate）
- 时间步序列 `timesteps` 在各 rank 上保持一致（广播/固定构造逻辑）。
- 候选层面的随机数生成器相互独立且可复现：
  - 建议派生规则：`seed = seed_base + candidate_id + global_rank * 10_000_000`（同设备 `torch.Generator`）。
- 烟雾测试：启动时各 rank 打印首个候选的 `log_prob` 与 `noise_strength[0]`，用于快速比对一致性。

### 数值监控与告警
- 训练过程中记录 `log_prob` 的 p1/p50/p99、`eps` 范数、以及 `noise_strength` 分布，写入日志/可视化平台。
- 配置项建议：`logprob_drop_const`、`normalize_logprob ∈ {none,batch_mean,batch_zscore}`、`sigma_clamp_min`。当分位数越界时输出告警（不做代码 fallback）。

### 性能与资源回归
- 记录每步稀疏 token 总数分布、显存峰值、吞吐（samples/s）。
- 确认仅 LoRA 参与训练：启动时打印可训练参数计数与白名单模块名。
- 对比 `num_candidates` 变化下的吞吐与奖励变化，评估探索效率。

### 迁移与复现性自检
- 迁移前后对同一输入的 dense/sparse512 推理输出进行数值对比（允许浮点微差）。
- 在 `generators/direct3d_s2/README.md` 记录来源版本与 LICENSE。
- 每次运行保存完整配置快照与 `git commit`/`git diff` 至 `logs/.../meta/`，确保结果可复现。

1) 最小可用版（sparse512）：
- 完成 6 个新增/修改文件（除 1024 可选分支外）。
- 单机单卡脚本可跑通 1-2 epoch，W&B/日志正常。
- 采样 `num_candidates>1` 时，log_prob 与候选 mesh 对齐且可复现。

2) 扩展（sparse1024 与 Refiner 可选）：
- 加入 1024 阶段与 `refiner_1024` 可选分支，验证内腔去除/简化流程。


## 实现提示（代码内注释风格）

- 在上述新增函数/方法的 docstring 中，明确写出本 DEV 中列出的“参考文件路径与函数名/行号”。
- 在每一行张量运算后补充形状注释，例如：

```python
latents = prev_mean + noise_strength * eps  # (B,C,H,W)
feats = prev_mean + noise_strength * eps    # (N_tokens,C)
```

- 不允许使用 try/except 或 fallback 分支。


## 分阶段实现与验证方案（3 个阶段）

### 阶段 1：最小可用采样与 log_prob（仅 sparse512）
- 范围：
  - 实现 `flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`（dense=ODE 仅产索引；sparse512=SDE+log_prob）
  - 实现 `flow_grpo/diffusers_patch/direct3d_s2_sde_with_logprob.py`
  - 保持与参考 `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py` 的时间轴/噪声强度一致
- 交付物（与 TRELLIS 训练接口完全对齐）：
  - 顶层采样函数返回四元组 `(meshes, all_latents, all_log_probs, all_kl)`：
    - `meshes`: List[mesh]，长度 `num_candidates`
    - `all_latents`: List[Tensor]，展平后的整条时序，长度 `num_candidates * (steps+1)`，每项形状 `(N_tokens, C)`
    - `all_log_probs`: List[Tensor]，逐步 log_prob，长度 `num_candidates * steps`，每项形状 `(1,)`（按元素求和的标量）
    - `all_kl`: List[Tensor]，占位零张量，长度 `num_candidates * steps`，每项形状 `(1,)`
  - 所有张量操作行附形状注释；无 try/except
- 验证：
  - 形状与数值单测：
    - `eps` 与 `x_next` 两种 log_prob 写法数值一致（仅差常数）
    - 重复固定种子采样，mesh 与 `log_prob` 可复现
    - SDE 采样方差与理论 `noise_strength^2` 接近（蒙特卡洛检验）
  - 功能对比：
    - 关闭 SDE（退化为 ODE）时，不累计 `log_prob`，输出与参考 `inference(..., mode='sparse512', deterministic=True)` 一致
    - CFG 仅影响均值，不改变噪声分布（检查 `noise_strength` 与步长无关 CFG）

### 阶段 2：接入 GRPO 训练（LoRA 到 sparse_dit_512）
- 范围：
  - 新增 `scripts/train_direct3d_s2.py` 与 `config/direct3d_s2_grpo_normal-sim.py`
  - 在训练采样中：Stage1（dense 生成索引一次）、Stage2（sparse512 重复 num_candidates 次，累计 `log_prob`）
  - LoRA 仅注入 `sparse_dit_512`；冻结其他模块
- 交付物：
  - 单机单卡脚本：`scripts/single_node/main_direct3d_s2_normal-sim.sh` 可跑 1-2 epoch
  - 训练日志含：奖励、KL/熵/`log_prob` 分布、学习率与梯度范数
- 验证：
  - 稳定性：loss/奖励曲线无爆炸，`log_prob` 有限且分布合理
  - 有效性：对比 `num_candidates={1,2,4}`，平均奖励随候选数增加有上升趋势
  - 可复现：固定全局+候选种子，多次运行统计量接近
  - 消融：关闭 SDE（等价不计 `log_prob`）时，训练退化，奖励提升显著减弱

### 阶段 3：迁移 `_reference_codes/Direct3D-S2` 模块到正式代码路径
- 范围：
  - 将 `_reference_codes/Direct3D-S2/direct3d_s2` 下用到的模块（`modules.sparse`、`utils.*`、`Direct3DS2Pipeline` 相关依赖）迁移/复制到工作路径正式包名下（如 `generators/direct3d_s2/` 或 `third_party/direct3d_s2/`）。
  - 修正 import 路径，确保 `from direct3d_s2...` 变为项目内相对/绝对导入。
  - 保持与参考实现行为一致；不引入 1024 阶段与 `refiner_1024` 的任何训练路径。
- 交付物：
  - 迁移后的代码在不联网条件下可直接被 `direct3d_s2_pipeline_with_logprob` 调用。
  - 新增的包含 `__init__.py` 与 README 注明来源与版本。
- 验证：
  - 迁移前后对同一输入的 dense/sparse512 推理输出一致（数值近似，允许浮点微差）。
  - 训练脚本对迁移前后的实现无感；W&B 曲线变化仅在噪声与随机种子统计波动范围内。

#### 迁移路径与文件映射表（Source → Target）
- `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py` → `generators/direct3d_s2/pipeline.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/__init__.py` → `generators/direct3d_s2/utils/__init__.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/util.py` → `generators/direct3d_s2/utils/util.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/image.py` → `generators/direct3d_s2/utils/image.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/rembg.py` → `generators/direct3d_s2/utils/rembg.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/sparse.py` → `generators/direct3d_s2/utils/sparse.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/mesh.py` → `generators/direct3d_s2/utils/mesh.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/utils/fill_hole.py` → `generators/direct3d_s2/utils/fill_hole.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/__init__.py` → `generators/direct3d_s2/modules/sparse/__init__.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/basic.py` → `generators/direct3d_s2/modules/sparse/basic.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/linear.py` → `generators/direct3d_s2/modules/sparse/linear.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/norm.py` → `generators/direct3d_s2/modules/sparse/norm.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/nonlinearity.py` → `generators/direct3d_s2/modules/sparse/nonlinearity.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/spatial.py` → `generators/direct3d_s2/modules/sparse/spatial.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/attention.py` → `generators/direct3d_s2/modules/sparse/attention.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/transformer/__init__.py` → `generators/direct3d_s2/modules/sparse/transformer/__init__.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/transformer/blocks.py` → `generators/direct3d_s2/modules/sparse/transformer/blocks.py`
- `_reference_codes/Direct3D-S2/direct3d_s2/modules/sparse/transformer/modulated.py` → `generators/direct3d_s2/modules/sparse/transformer/modulated.py`

说明：若源目录存在额外依赖（例如 `modules/__init__.py` 或其它被 `pipeline.py` 间接引用的文件），一并迁移到对应子目录，确保 `pipeline.py` 可独立运行。

#### 导入路径替换规则
- 将 `from direct3d_s2.utils import X` 全部替换为 `from generators.direct3d_s2.utils import X`
- 将 `from direct3d_s2.modules import sparse as sp` 替换为 `from generators.direct3d_s2.modules import sparse as sp`
- 若有 `from direct3d_s2.modules.sparse import Y`，替换为 `from generators.direct3d_s2.modules.sparse import Y`
- 训练与管线中引用 `Direct3DS2Pipeline` 的位置，统一改为 `from generators.direct3d_s2.pipeline import Direct3DS2Pipeline`

#### 迁移后的对接点
- `flow_grpo/diffusers_patch/direct3d_s2_pipeline_with_logprob.py`：
  - 使用 `from generators.direct3d_s2.pipeline import Direct3DS2Pipeline`
  - 使用 `from generators.direct3d_s2.modules import sparse as sp`
  - 使用 `from generators.direct3d_s2.utils import preprocess_image, sort_block, extract_tokens_and_coords, normalize_mesh, mesh2index, postprocess_mesh`


