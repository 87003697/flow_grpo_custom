
## Trellis 迁移改动方案

1. **Pipeline 入口与稀疏算子**
   - 训练/推理脚本在启动阶段需通过 `sys.path` 注入 `_reference_codes/TRELLIS`，并直接 import 官方 `trellis.modules`, `trellis.pipelines`。
   - `flow_grpo/diffusers_patch/trellis_pipeline_with_logprob.py`、`trellis_sparse_tensor.py` 已整合原有 `sparse_tensor_cat`、`convert_trellis_to_trimesh` 等逻辑；外部模块必须改为引用这些封装。
   - Trellis `pipeline.prepare_image_conditions` 直接返回 `(cond, neg_cond)`；`build_stage1_cond` 和 Stage1/Stage2 调用应统一使用该二元组，不再构造 `{"cond": ..., "neg_cond": ...}` 的 dict。
   - 清理 `generators/trellis` 目录后，要持续通过 `grep trellis` 等方式确认不存在遗留引用（例如 `generators.trellis.` 字样），避免再次引入旧路径。
   - `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py` 的 `from_pretrained` 会实例化 `samplers.FlowEuler*`、填充 `sparse_structure_sampler_params`、`slat_sampler_params`、`slat_normalization` 与 `image_cond_model`，并写入 pipeline；本地 wrapper 需要完整 mirror 这些字段，避免重复解析 `pipeline._pretrained_args`。
   - `sample_sparse_structure`（返回 `coords[int32]`）与 `sample_slat`（返回 `SparseTensor`）的输出会直接喂给 `decode_slat`（默认为 `mesh`/`gaussian`/`radiance_field` 三种键）；扩展评估/可视化时要遵循该结构，必要时增加格式转换层而非修改 pipeline 本身。

2. **模型与 LoRA 配置**
   - `trellis_pipeline_with_logprob` 提供的 `get_trainable_model_stage{1,2}` 需继续返回官方 `trellis.models` 中的 `SparseStructureFlowModel` 与 `SLatFlowModel`；任何包装或 EMA 逻辑都应基于官方模块实例。
   - `apply_lora_if_needed` 的 target modules 需要与官方模块内部命名保持一致（参考 `_reference_codes/TRELLIS/trellis/modules/transformer/modulated.py` 与 `trellis/modules/attention/modules.py`）：
     - Stage1 `ModulatedTransformerCrossBlock`: `blocks.*.self_attn.to_qkv`、`blocks.*.self_attn.to_out`、`blocks.*.cross_attn.to_q`、`blocks.*.cross_attn.to_kv`、`blocks.*.cross_attn.to_out`
     - Stage2 `ModulatedSparseTransformerCrossBlock`：命名同上（`self_attn.*`, `cross_attn.*`, `mlp`）
     - 若需扩展到 `input_layer` / `out_layer` 或 `input_blocks` / `out_blocks` 中的线性层，需显式将模块路径加入 LoRA target 列表
   - Stage1/Stage2 大量使用官方 `trellis.modules.sparse.linear.SparseLinear`、`SparseConv3d` 等稀疏层；`flow_grpo/peft_sparse/sparse_lora.py` 中的 `register_sparse_linear_with_peft()` 需确认能识别这些类型，或新增适配器以支持稀疏层 LoRA。
   - `SparseStructureFlowModel`、`SLatFlowModel` 构造函数分别依赖 `resolution`、`in_channels`、`cond_channels`、`num_blocks`、`num_heads/num_head_channels`、`mlp_ratio`、`patch_size`、`pe_mode`、`use_fp16`、`share_mod`、`qk_rms_norm(_cross)` 等参数，Stage2 还需要 `num_io_res_blocks`、`io_block_channels`、`use_skip_connection`；对应的 config 字段必须全量暴露，否则 `from_config` 与 `from_pretrained` 无法互通。
   - `SLatFlowModel` 的 IO 块依赖 `SparseResBlock3d`（内部为 `SparseConv3d`、`SparseLinear` 与上/下采样器），LoRA/EMA/精度切换需要考虑这些模块名不会被 Transformer 路径覆盖，必要时在 `peft_sparse` 注册额外 target。
   - Structured latent VAE 相关的 `structured_latent_vae/base.py` 通过 `block_attn_config` 组合 `attn_mode`、`window_size`、`shift_sequence`/`shift_window`、`SerializeMode` 等序列化策略；若我们的配置允许切换 `swin`、`shift_window` 等模式，需确保这些参数及其依赖（`SparseTransformerBlock`、`SparseTransformerBase`）在 config/schema 中可控。
   - 稀疏注意力核心层在 `_reference_codes/TRELLIS/trellis/modules/sparse/attention/modules.py`：`SparseMultiHeadAttention` 定义了 `to_qkv`/`to_q`/`to_kv`、`to_out`、`q_rms_norm`、`k_rms_norm` 等命名；Stage2 LoRA、权重拷贝、精度转换都必须使用这些精确路径，尤其在启用 `qk_rms_norm` 或 `use_rope` 时会多出额外参数。

3. **调度器与时间步**
   - 将 `pipeline.ref.sparse_scheduler_512`、`pipeline.ref.dense_scheduler` 等属性替换为 Trellis 暴露的调度器字段，并同步更新 `sparse_timesteps`、`dense_timesteps` 抽取逻辑。
   - 如果 Trellis 的时间归一化（如 `t/1000`）或噪声注入方式不同，需要在 Stage1/Stage2 DiffusionNFT 训练段重新定义 `t_norm`、`xt` 构造方式。
   - `Stage1RuntimeConfig` / `Stage2RuntimeConfig` 由 `flow_grpo/diffusers_patch/trellis_sparse_tensor.py` 提供，需确保训练脚本传入的 `noise_level`、`rescale_t`（经 `SlatSamplerParams`）、`compute_kl`、`kl_reward` 等字段与配置保持一致，避免 `set_trellis_timesteps` 产生越界时间步或 logprob 不稳定。
   - 官方 `FlowEulerSampler`（含 CFG、interval mixin）在 `_inference_model` 内对 `t` 乘以 1000，并依赖 `steps`、`rescale_t`、`sigma_min`、`cfg_strength`、`cfg_interval` 这些命名；训练/评估时若改写 sampler，需要保持参数名一致，否则 sampler kwargs 会被忽略导致时间步错位。

4. **稀疏/稠密样本结构**
   - `Direct3DSample` 与 `Direct3DSampleCollection` 中保存的 `x0_sparse`、`x0_dense`、`cond_patches` 等张量需确认与 Trellis 模型的期望 shape 匹配；若 Trellis 不再需要某些字段（例如 dense 序列），删除相应成员与搬运逻辑。
   - `build_samples_from_generation`、`move_batch_samples` 里调用的稀疏构造函数、MSE 计算函数应替换为 `flow_grpo/diffusers_patch/trellis_sparse_tensor.py` 内实现。
   - Trellis Stage1/Stage2 样本默认只提供稀疏序列（Stage1 可选 dense 轨迹），若后续 DiffusionNFT 不再依赖 `x0_dense`、`latents_seq_dense`，可考虑裁剪存储结构以降内存。
   - `_reference_codes/TRELLIS/trellis/modules/sparse/basic.py` 的 `SparseTensor` 约定 coords 为 `(batch_idx, x, y, z)` 且同一 batch 必须连续存储；我们自己的构造函数如果打乱 layout，会导致 `SparseConv3d`/`SparseLinear` 抛错，需要在数据搬运中保持排序或复用 Trellis 的 `sparse_batch_broadcast`/`sparse_unbind`。
   - Stage1 `sample_sparse_structure` 会将 occupancy decoder (>0) 的稠密体素转换成 int32 coords，Stage2 `sample_slat` 在此基础上创建 `SparseTensor(feats, coords)` 并在采样后用 `slat_normalization` 做 `feats = feats * std + mean`；任何替换 decoder/normalization 的改动都要同步更新样本构造逻辑。
   - `trellis/datasets/structured_latent.py` (`SLat`) 会在 `collate_fn` 中显式拼接 `(batch_idx, xyz)` coords 并缓存 layout；若配置包含 `normalization` 字段，还会对 latent feats 做 `(feats-mean)/std`。自研数据加载/缓存逻辑必须保持这些键名（`latent_model`、`max_num_voxels`、`normalization.mean/std`）和返回结构，以便与官方 Trainer 对接。
   - 数据集基类 `_reference_codes/TRELLIS/trellis/datasets/components.py` 的 `StandardDatasetBase` 会读取各 `root/metadata.csv`、过滤 `sha256` 并记录统计；`TextConditionedMixin`、`ImageConditionedMixin` 分别要求 metadata 包含 `captions`、`cond_rendered` 字段，并在 `get_instance` 里随机抽 caption / 读入 `renders_cond/<sha256>/transforms.json` + RGBA crop。所以任何自定义数据源要么提供同名列，要么新增适配层以保持接口稳定。

5. **训练循环差异**
   - Trellis pipeline 明确保留 Stage1（`sparse_structure_flow_model`）与 Stage2（`slat_flow_model`）两个子模型，因此沿用双阶段 DiffusionNFT 训练；Teacher 输出可直接通过 `pipeline._resolve_structure_flow_module()` / `_resolve_slat_flow_module()` 拿到未挂适配器的参考模型，按现有 `disable_adapter()` 方式复用。
   - 调整 `SlatSamplerParams` 或同类运行参数，确保 `mc_threshold`、`noise_level` 等字段与 Trellis sampler 的配置键一致。
   - 校验 `flow_grpo/diffusers_patch/trellis_sparse_tensor.py` 中的稀疏/稠密损失工具（如 `compute_sparse_weighted_mse`、`compute_dense_weighted_mse`）在 DiffusionNFT 训练链路里的输出形状与 dtype，与 `Direct3DSampleCollection` 生成的张量完全匹配，避免广播或精度问题。
   - 若需单步 logprob 训练（Stage1/Stage2）则使用 `Stage1RuntimeConfig`、`Stage2RuntimeConfig`，确认脚本传入的 `noise_level`、`rescale_t`、`compute_kl` 等参数与 Trellis 默认配置一致，以免调度器在 `trellis_sparse_tensor.py` 中重置时发生越界。
   - `FlowEulerSampler.sample()` 会返回 `samples`、`pred_x_t`、`pred_x_0` 列表；如需在训练中记录 teacher 预测或计算额外奖励，需沿用该返回结构，而非假设 scheduler 只返回最终样本。
   - 官方 `_reference_codes/TRELLIS/train.py` + `trellis/trainers/flow_matching/flow_matching.py` 将 `t_schedule`（默认 `logitNormal`）、`sigma_min`、`diffuse/get_v`、`training_losses`（MSE + `bin_i` 分桶日志）统一封装在 `FlowMatchingTrainer`；我们在 DiffusionNFT 中复用/改写时要确保 `t*1000` 的缩放、`p_uncond`、`sample_t` 等配置项可控，并且日志统计（bin MSE、`dict_reduce` 可视化）保持可用，方便对齐官方训练曲线。
   - 额外视觉指标集中在 `trellis/utils/loss_utils.py`：`psnr`、`ssim`、`lpips`、`normal_angle` 等函数会动态加载 `LPIPS(net='vgg')` 并假设输入范围 `[0,1]`（内部会归一化到 `[-1,1]`），同时依赖 `torchvision`, `lpips`, `pillow-simd`。若训练/评估脚本需要这些指标或感知损失，需在环境中准备相应依赖并确保 GPU 可用。

6. **日志、配置与命名**
   - 更新 `run_name`、W&B 项目名、保存目录等文案，使其体现 Trellis（例如 `flow-grpo-trellis`、`trellis_stage1+2`）。
   - 校验 `config` 中 Direct3D 专属字段（`pretrained.minimal_512_only`、`slat_sampler_params` 等），必要时改为 Trellis 的配置结构或新增字段读取；尤其是 `structured_latent_flow` 依赖的 `patch_size`、`io_block_channels`、`num_io_res_blocks`、`attn_mode`、`window_size`、`pe_mode`、`qk_rms_norm` 等参数需要在配置里显式暴露并传递给 pipeline（cf. `generators/trellis/models/structured_latent_flow.py`、`structured_latent_vae/base.py`）。
   - `run_logger`、W&B 监控指标与 `CheckpointSaver` 中的提示文本应替换掉 Direct3D 专属命名，避免和旧实验混淆（例如日志键从 `epoch/stage2/...` 改为 `epoch/trellis_stage2/...` 或者通用命名）。
   - Trellis 官方 `pipeline._pretrained_args` 会包含 `sparse_structure_sampler`、`slat_sampler`、`slat_normalization`、`image_cond_model` 等字段；Config/Checkpoint saver 若想重新构造 pipeline，必须支持序列化/反序列化这些结构（特别是 sampler `name`/`args`/`params`）以保持可重复性。
   - `_reference_codes/TRELLIS/configs/generation/*.json` 将模型、数据集、trainer 拆成三块：`models.denoiser`（含 `SparseStructureFlowModel`/`ElasticSLatFlowModel` 参数）、`dataset`（名称多为 `ImageConditionedSparseStructureLatent`/`ImageConditionedSLat` 并附 `latent_model`、`normalization.mean/std`、`pretrained_*_dec` 等字段）、`trainer.args`（`p_uncond`、`t_schedule`、`sigma_min`、`elastic`、`grad_clip`、`image_cond_model` 等）。我们自己的配置系统需要支持同名字段，以便直接消费官方 config 或导出兼容的 `config.json`。
   - README 中列出了 HuggingFace 上的预训练权重映射（例如 `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json` 对应 `microsoft/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors`，VAE/decoder 亦如此）；需要在本地 `pretrained_weights/` 中保持一致命名，或在配置里加入映射表，确保 `from_pretrained` 能直接定位这些 safetensors。
   - 当前仓库 `pretrained_weights/TRELLIS-image-large/ckpts/` 已包含 `ss_flow_*`、`slat_flow_*`、`slat_dec_*`、`ss_enc/dec_*` 等 safetensors，但 HuggingFace clone (`pretrained_weights/models--JeffreyXiang--TRELLIS-image-large`) 只有 `refs/main` 而无 `snapshots/<rev>`；若后续要离线调用 `TrellisImageTo3DPipeline.from_pretrained`，需补齐 `snapshots/` 目录或通过环境变量 `HF_HOME` 指向已下载的 ckpt 路径，否则 pipeline 仍会尝试在线拉取。

7. **评估与可视化**
   - `eval_direct3d` 调整为 Trellis 版本：替换 pipeline 调用、稀疏张量构造、mesh 生成流程。
   - 确认 Trellis 输出的 mesh/点云结构能被现有 `save_meshes_for_preview` 处理；若格式不同，修改 `to_mesh_extract` 适配或新增 Trellis 专用可视化路径。
   - Trellis pipeline 默认输出 `KiuiMesh`/`trimesh` 对象，若需继续记录 camera normal / Uni3D 奖励，需要确保 `MeshScorer` 的输入转换（`to_mesh_extract`, `KiuiMeshLike`）已兼容 Trellis Mesh；必要时在 scorer 或可视化路径中添加格式转换。
   - Trellis 官方脚本（`_reference_codes/TRELLIS/train.py`) 使用自带的渲染/导出流程，可参考其 mesh/图像产出路径，决定是否在本工程中复用官方可视化逻辑或继续沿用 Hunyuan3D 风格。
   - `trellis_image_to_3d.TrellisImageTo3DPipeline` 的 `sample_sparse_structure`/`sample_slat` 默认 `verbose=True` 并回传 `decode_slat` 字典，可在我们自己的评估脚本里通过修改 sampler kwargs（如 `steps`、`verbose`）来禁用重复进度条，避免多机日志互相干扰。
   - `_reference_codes/TRELLIS/trellis/utils/render_utils.py` 根据 representation 类型（`Octree`、`Gaussian`、`MeshExtractResult`）自动选择 renderer 并设置 `resolution`、`near/far`、`ssaa`、`kernel_size` 等可选参数；若继续沿用官方渲染器，需要提供同款输入（`get_renderer`、`render_frames`、`render_multiview`）或保持兼容的接口，以便复用 snapshot/video 输出。
   - `trellis/utils/postprocessing_utils.py` 提供 mesh 后处理管线：`postprocess_mesh` 用 `pyvista` 进行 Quadric 简化、再用 `utils3d` + `pymeshfix` 做多视图可见性剪裁与补洞；`parametrize_mesh` 调用 `xatlas` 展 UV；`bake_texture` 依赖 `nvdiffrast`, `opencv-python-headless` 进行纹理回烘。若保留官方 mesh/纹理导出，需要在评估容器中安装这些 CUDA 拓展并遵循函数要求的输入（`np.ndarray` verts/faces、摄像机 extrinsics/intrinsics 等）。
   - 官方示例 `example.py` / `example_text.py` / `README.md` 演示了完整推理 → 渲染 → 导出流程：`TrellisImageTo3DPipeline.run()` 返回 `mesh`/`gaussian`/`radiance_field` 列表，随后用 `render_utils.render_video(...)[key]` 生成 `sample_*.mp4`，再通过 `postprocessing_utils.to_glb(gs, mesh, simplify=?, texture_size=?)` 输出纹理化 GLB，并可调用 `outputs['gaussian'][0].save_ply()` 保存 PLY。我们在自定义可视化脚本中可以直接复用这套流程，并根据需要调整 `SPCONV_ALGO`、`ATTN_BACKEND` 环境变量。
   - `app.py` / `app_text.py` 的 Gradio Demo 在 UI 层公开了 Stage1/Stage2 的 `guidance_strength`、`sampling_steps`、`seed/randomize_seed`、多图模式的 `multiimage_algo`（`stochastic` / `multidiffusion`）、以及 GLB 导出阶段的 `mesh_simplify`（0.9~0.98）/`texture_size`（512~2048）等参数；将来如果要写 CLI/脚本，可以直接映射这些选项到命令行旗标，复用 demo 内的 `image_to_3d` / `text_to_3d` / `extract_glb` / `extract_gaussian` 封装。


9. **环境依赖 / 扩展组件**
   - `_reference_codes/TRELLIS/setup.sh` 定义了官方环境安装流程：Python 3.10 + PyTorch 2.4.x（CUDA 11.8/12.1 或 ROCm 6.1），可选 flag 安装 `xformers`、`flash-attn`、`spconv-cu118/cu120`、`diffoctreerast`、`nvdiffrast`、`mip-splatting`、`vox2seq`、`kaolin`、`gradio` 等模块，并强制安装 `rembg`, `utils3d@git`, `pyvista`, `pymeshfix`, `xatlas`, `igraph`, `onnxruntime` 等依赖。迁移前需确认本地/集群能满足这些 NVIDIA/CUDA 版本要求，并准备 `libjpeg-dev` + `pillow-simd` 等系统依赖，避免运行时缺包。

8. **调试与验证**
   - `scripts/test_trellis_suite.py`、`scripts/debug/test_trellis_suite.py`、`scripts/debug/test_trellis_infer.py`、`scripts/trellis_example.py` 等调试脚本都已切换到新的导入方式；后续若新增调试工具，务必沿用相同的 `_reference_codes/TRELLIS` 注入方案。
   - 由于核心依赖路径发生变更，请在训练/推理脚本上完成一次冒烟验证（如 `test_trellis_suite`、训练阶段的最小 epoch），确保官方模型加载、LoRA 注入、mesh 可视化链路均正常工作。

完成以上步骤后，再根据 Trellis 实际接口做针对性调试，确保训练、评估、日志与可视化链路均在新 pipeline 下正常运行。

