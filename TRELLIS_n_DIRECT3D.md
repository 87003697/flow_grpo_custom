## TRELLIS 与 Direct3D‑S2 对比（v2）

本文件总结两者在 GRPO3D 集成背景下的相同点与差异点。每条均附官方参考实现的代码路径与行号，便于溯源与核对。

### 相同点清单
- **CFG（Classifier‑Free Guidance）合成逻辑相同**：使用 uncond + w * (cond − uncond)
  - TRELLIS: `_reference_codes/TRELLIS/trellis/pipelines/samplers/classifier_free_guidance_mixin.py:9-12`
  - Direct3D‑S2: `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:279-286`
- **两阶段/分阶段生成**：先获取稀疏结构/索引，再在稀疏位置生成结构化潜变量并解码 Mesh
  - TRELLIS（坐标→SLAT→解码）: `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:279-283`
  - Direct3D‑S2（dense 索引→sparse512）: `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:359-369`
- **统一的 from_pretrained 入口**：
  - TRELLIS: `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:45-48`
  - Direct3D‑S2: `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:67-69`
- **条件作为显式输入传入主干计算**：
  - TRELLIS（采样器内 `_inference_model` 使用 cond）: `_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:38-46`
  - Direct3D‑S2（`diffusion_inputs` 包含 cond）: `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:270-276`

### 差异对比表

| 维度 | TRELLIS | Direct3D‑S2 |
|---|---|---|
| **条件图像编码** | 使用 DINOv2 提取 patch tokens，LayerNorm；`neg_cond` 为零张量<br>[trellis_image_to_3d.py (70-81,140-160)](_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py) | 自有 dense/sparse 图像编码器；可返回 token mask，并抽取 class/register/patch tokens 与稀疏坐标（封装为 `SparseTensor`）<br>[pipeline.py (194-217)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py)<br>[utils/sparse.py (21-29)](_reference_codes/Direct3D-S2/direct3d_s2/utils/sparse.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:70-81,140-160` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:194-217`，`_reference_codes/Direct3D-S2/direct3d_s2/utils/sparse.py:21-29` |
| **潜在 tokens / 稀疏定位** | Stage1 采样 occupancy latent，阈值得到 `coords`；Stage2 生成 SLAT（`sp.SparseTensor`）<br>[trellis_image_to_3d.py (179-187,189-193,239-246)](_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py) | dense 阶段生成 `latent_index`（块选择/排序），sparse512 阶段在索引上生成稀疏 latent，后续可 1024 升采<br>[pipeline.py (359-369)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py)<br>[pipeline.py (380-383)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:179-187,189-193,239-246` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:359-369,380-383` |
| **主干网络范式** | Flow‑Matching（Flow 模型 + Euler 采样器）<br>[flow_euler.py (11,38-46,80-129)](_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py) | DIT（Transformer）+ VAE，scheduler 驱动的扩散<br>[pipeline.py (224-229)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:11,38-46,80-129` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:224-229` |
| **条件交互位置** | 采样器将 `cond` 传入 `_inference_model(model, x_t, t, cond)`<br>[flow_euler.py (38-46)](_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py) | `cond` 放入 `diffusion_inputs`，一并传入 DIT<br>[pipeline.py (270-276)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:38-46` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:270-276` |
| **去噪 / 调度器** | Flow‑Matching Euler ODE：t∈[1,0] 放大至 0..1000；确定性步进<br>[flow_euler.py (38-40)](_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py) | DDPM 风格调度器：`set_timesteps` + `step` 得 ODE 均值；可选 SDE 噪声注入（含 `sigma_min`、`rescale_t`、`deterministic`）<br>[pipeline.py (253-255)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py)<br>[pipeline.py (288-314)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:38-40` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:253-255,288-314`（SDE 注入与噪声强度计算） |
| **解码方式（latent→Mesh）** | SLAT 多头解码到 mesh / gaussian / radiance_field<br>[trellis_image_to_3d.py (195-217,281-283,373-375)](_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py) | VAE 解码 + refiner；`inference(..., mode='sparse512')` 直接返回 mesh<br>[pipeline.py (366-369)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:195-217,281-283,373-375` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:366-369` |
| **典型流程** | `preprocess → get_cond → sample_sparse_structure → coords 阈值 → sample_slat → decode_slat`<br>[trellis_image_to_3d.py (258-283)](_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py) | `prepare_image → dense inference 得 latent_index → sparse512 inference → 可选 sparse1024`<br>[pipeline.py (346-351,357-369,380-383)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py) |
| **源码定位** | `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:258-283` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:346-351,357-369,380-383` |
| **重要超参** | 每阶段 `steps`、`cfg_strength`、多图模式（stochastic/multidiffusion）、`seed`<br>[README.md (142-150)](_reference_codes/TRELLIS/README.md)<br>[trellis_image_to_3d.py (285-341,347-375)](_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py) | `num_inference_steps`、`guidance_scale`、`use_sde`、`sigma_min`、`rescale_t`、`deterministic`<br>[pipeline.py (346-351,224-237)](_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py)<br>[demo.py (67-123)](_reference_codes/Direct3D-S2/demo.py)<br>[scripts/demo.sh (13-17,31-35,86-90)](_reference_codes/Direct3D-S2/scripts/demo.sh) |
| **源码定位** | `_reference_codes/TRELLIS/README.md:142-150`，`_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:285-341,347-375` | `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:346-351,224-237`，`_reference_codes/Direct3D-S2/demo.py:67-123`，`_reference_codes/Direct3D-S2/scripts/demo.sh:13-17,31-35,86-90` |

### 额外备注（GRPO3D 集成相关）
- 在 GRPO 训练中：
  - TRELLIS 的 Stage2（SLAT）沿 Flow 采样，通常使用 ODE（确定性）；若需要 log_prob 路径需在采样器中添加 SDE 与对数概率累计（本仓库已在 `flow_grpo/diffusers_patch` 内提供接口）。
  - Direct3D‑S2 的 sparse512 可以启用 SDE，按步累计 `log_prob`；dense 阶段通常 ODE 只用于获得索引，不计 `log_prob`。
- CFG 不改变噪声协方差，仅改变均值，故 `log_prob` 计算不需因 CFG 做额外修正（两侧一致）。

如需进一步核查，请逐条打开上述路径并对照实现细节与实际运行参数。
