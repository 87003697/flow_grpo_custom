"""TRELLIS.2 蒸馏训练基础配置（按模块拆分，支持 per-stage 独立配置）。

配置层级结构：
    cfg:
        # ===== 全局共享 =====
        seed, logdir, num_epochs, mixed_precision, pipeline_type, ...
        gradient_accumulation_steps          # Accelerator 级全局参数
        data: {train, eval}
        pretrained: {model, dino_local_path}
        freq: {save, eval, profiler}
        reg: {type}

        # ===== 共享渲染基础 =====
        renderer:
            resolution, ssaa, near, far, bg_color, chunk_size

        # ===== Guidance 初始化（全阶段共享，只加载模型） =====
        guidance:
            type, model_path, edit_resolution
            flowedit: {steps, n_max, noise_mode, ...}  # FlowEdit 专属 init 参数

        # ===== Shape 阶段独立 =====
        shape:
            renderer: {type, grad_checkpoint}
            train:    {mode, optimizer, loss}
            guidance: {seed, target_prompt, ..., loss: {...}}

        # ===== Tex 阶段独立 =====
        tex:
            renderer: {envmap_path}
            train:    {mode, optimizer, loss}
            guidance: {seed, target_prompt, ..., loss: {...}}

★ Guidance 配置分两层：
  - cfg.guidance: 初始化配置（model_path 等 + 范式专属子配置如 flowedit.{steps, n_max, ...}），只加载一次模型
  - cfg.{stage}.guidance: 运行时配置（prompt, loss 权重等），每次调用 compute_guidance 传入
    默认使用 _flowedit_runtime_config()；切换到 distillation 时可替换为 _distillation_runtime_config()

每个字段都经过验证，确保在 edit4shape/ 代码中被实际读取。
未使用的字段已清理（详见 git log）。
"""
import ml_collections


# =====================================================================
# 全局共享配置
# =====================================================================

def get_base_config_general():
    """通用配置（seed, epochs, 频率等）。

    这些参数对 Shape / Tex 阶段通用，不需要 per-stage 区分。
    """
    cfg = ml_collections.ConfigDict()
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    cfg.verbose = False
    cfg.pipeline_type = "1024"
    cfg.use_wandb = False  # 是否启用 wandb 日志

    # ★ gradient_accumulation_steps 是 Accelerator 级全局参数，从 train 提升到顶级
    cfg.gradient_accumulation_steps = 4

    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1
    cfg.freq.save.ckpt = 1
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数）
    cfg.freq.eval = 1
    cfg.freq.profiler = 1 # PhaseProfiler 汇总打印频率（每 N 步打印一次平均值）

    # 正则化配置
    # - "none": 不使用正则化
    # - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
    # - "x1": MSE(x0_stu, x0_tea)，不除 t²，小 t 时权重不被放大
    # - "v": MSE(v_stu, v_tea)，梯度仅当前步
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "v"    # none | x0 | x1 | v
    return cfg


def get_base_config_data():
    """数据配置（训练/评估）。"""
    cfg = ml_collections.ConfigDict()

    cfg.train = ml_collections.ConfigDict()
    cfg.train.dir = "dataset/alphaimages_v3/train"
    cfg.train.batch_size = 1
    cfg.train.n_view = 1
    cfg.train.yaw_range = [0.0, 360.0]
    cfg.train.pitch_range = [0.0, 0.0]  # 固定 pitch 角度
    cfg.train.r_range = [2.0, 2.0]
    cfg.train.fov_range = [40.0, 40.0]
    cfg.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.train.adaptive_distance.enabled = True
    cfg.train.adaptive_distance.fill_ratio = 0.9

    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.dir = "dataset/alphaimages_v3/test"
    cfg.eval.batch_size = 1
    cfg.eval.n_view = 6
    cfg.eval.yaw_range = [0.0, 360.0]
    cfg.eval.pitch_range = [0.0, 0.0]
    cfg.eval.r_range = [2.0, 2.0]
    cfg.eval.fov_range = [40.0, 40.0]
    cfg.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.eval.adaptive_distance.enabled = True
    cfg.eval.adaptive_distance.fill_ratio = 0.9
    return cfg


def get_base_config_pretrained():
    """预训练权重路径。"""
    cfg = ml_collections.ConfigDict()
    cfg.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


# =====================================================================
# 共享渲染基础参数
# =====================================================================

def get_base_config_renderer():
    """共享渲染基础参数。

    所有渲染器（MeshRenderer / MeshPeeledRenderer）共用的参数。
    阶段专有参数见 get_base_config_shape_stage().renderer / get_base_config_tex_stage().renderer。
    """
    cfg = ml_collections.ConfigDict()
    cfg.resolution = 1024
    cfg.ssaa = 1
    cfg.bg_color = [1.0, 1.0, 1.0]
    cfg.near = 1.0
    cfg.far = 100.0
    # MeshPeeledRenderer 默认剥离层数（Tex-only 模式 Shape 阶段使用共享 renderer 配置时的 fallback）
    cfg.peel_layers = 8
    return cfg


# =====================================================================
# Guidance 初始化配置（全阶段共享，只加载模型）
# =====================================================================

def get_base_config_guidance():
    """Guidance 初始化配置（模型加载参数，全阶段共享）。

    ★ 仅包含模型加载所需参数。运行时参数（prompt / loss 权重 / 聚合策略）
    在 per-stage 的 cfg.shape.guidance / cfg.tex.guidance 中配置，
    调用 compute_guidance() 时传入。

    当前支持的 Guidance 类型:
    - "flowedit": FlowEdit（编辑图像 → 计算相似度 loss）
    - "distillation": 蒸馏（单步/多步，SDS/CSD 变体）

    结构：
        cfg.type          — 范式选择（共用）
        cfg.model_path    — 模型路径（共用）
        cfg.edit_resolution — 工作分辨率（共用）
        cfg.flowedit      — FlowEdit 专属 init 参数（采样步数 / 噪声 / tracker）
    """
    cfg = ml_collections.ConfigDict()

    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    cfg.type = "flowedit"

    # 模型路径（HuggingFace ID 或本地路径）
    cfg.model_path = "Qwen/Qwen-Image-Edit-2511"
    # 工作分辨率（VAE encode 时使用）
    cfg.edit_resolution = 1024

    # 条件图背景色 float [0,1]，应与 cfg.renderer.bg_color 保持一致
    cfg.bg_color = [0.5, 0.5, 0.5]

    # FlowEdit 专属 init 参数
    _flowedit_init_config(cfg)

    return cfg


def _flowedit_init_config(g: ml_collections.ConfigDict):
    """FlowEdit 专属 init 参数（写入 cfg.guidance.flowedit）。

    这些参数在训练过程中不会变化，仅在构造 Pipeline 时读取一次。
    """
    g.flowedit = ml_collections.ConfigDict()

    # 采样步数
    g.flowedit.steps = 12   # num_inference_steps: 总时间步数
    g.flowedit.n_max = 9   # 实际执行的最后 n_max 步

    # 噪声模式:
    #   - random: 每步随机噪声
    #   - fixed: 固定噪声（所有 step 共用）
    #   - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    #   - delta: 双分支差分补偿 ε -= (v_cfg_tgt - v_cfg_src) * (1 - t)
    g.flowedit.noise_mode = "aligned"

    # MTS 采样: 是否使用均匀分区随机采样
    # - False: 使用 scheduler 的固定时间步序列
    # - True: 在 [0.02, 0.98] 范围内均匀分区随机采样 steps 个时间步
    g.flowedit.use_mts_sampling = True

    # Tracker 记录控制
    # - use_tgt_record: 记录 target 分支的 x0 正负对（默认 True）
    # - use_src_record: 记录 source 分支的 x0 正负对（默认 True）
    g.flowedit.use_tgt_record = True
    g.flowedit.use_src_record = True


# =====================================================================
# Guidance 运行时配置（per-stage，调用 compute_guidance 时传入）
# =====================================================================

def _flowedit_runtime_config():
    """FlowEdit 运行时参数（per-stage 调用时传入 compute_guidance）。

    包含 prompt、CFG scales、loss 权重、聚合策略等，
    不同阶段（Shape / Tex）可使用不同值。

    ★ 采样结构参数（steps / n_max / noise_mode / use_mts_sampling / tracker）
      在 cfg.guidance.flowedit（init 配置）中设置，全阶段共享。

    所有字段均在 edit4shape/guidance/paradigms/flowedit.py 中被读取。
    """
    cfg = ml_collections.ConfigDict()

    # 随机种子（FlowEdit Pipeline 的 generator 种子）
    cfg.seed = 42

    # Target 分支参数
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.negative_prompt_tgt = " "  # target 分支的 negative prompt
    cfg.true_cfg_scale_tgt = 4.0
    # Source 分支参数（full 模式需要；simple 模式下不会读取）
    cfg.true_cfg_scale_src = -1 * cfg.true_cfg_scale_tgt
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # 多步 Loss 配置（分离聚合方式和归一化方式）
    # reduce_mode: 聚合方式
    #   - "final": 只用最后一步
    #   - "mean": 均匀加权
    #   - "weighted": 1/k 加权（前期大）
    #   - "inv_weighted": k/K 加权（后期大）
    cfg.reduce_mode = "mean"
    # ada_normalize: 是否使用自适应归一化
    #   - True: 梯度归一化（稳定训练）
    #   - False: 标准 MSE
    cfg.ada_normalize = True
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    cfg.ada_eps = 1e-1

    # ========== Loss 权重配置 ==========
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.latent_mse = 0.0    # MSE: MSE(src, z_edit)
    cfg.loss.latent_csd = 1.0    # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)

    return cfg


def get_base_config_shape_stage():
    """Shape 阶段独立配置（renderer + train + guidance 运行时）。

    包含：
    - shape.renderer: type, grad_checkpoint
    - shape.train: mode, optimizer, loss
    - shape.guidance: FlowEdit 运行时配置（prompt, loss 权重等）

    注意：
    - chunk_size 在共享 renderer（cfg.renderer.chunk_size）
    - Guidance 初始化配置在 cfg.guidance（model_path 等）
    """
    cfg = ml_collections.ConfigDict()

    # --- Shape 渲染器专有参数 ---
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.type = "hybrid26_peeled"        # "mesh_peeled" | "hybrid26_peeled"
    cfg.renderer.grad_checkpoint = True      # gradient checkpoint（省显存）
    cfg.renderer.bg_color = [0.5, 0.5, 0.5]  # Normal map 背景色（灰色）

    # --- Shape 训练超参 ---
    cfg.train = _base_stage_train()

    # --- Shape Guidance 运行时配置 ---
    cfg.guidance = _flowedit_runtime_config()
    # Shape 阶段默认使用 Normal map prompt
    cfg.guidance.target_prompt = "Move the camera. Convert to normal map."
    cfg.guidance.source_prompt = cfg.guidance.target_prompt

    return cfg


def get_base_config_tex_stage():
    """Tex 阶段独立配置（renderer + train + guidance 运行时）。

    包含：
    - tex.renderer: envmap_path, peel_layers
    - tex.train: mode, optimizer, loss
    - tex.guidance: FlowEdit 运行时配置（prompt, loss 权重等）
    """
    cfg = ml_collections.ConfigDict()

    # --- Tex 渲染器专有参数 ---
    cfg.renderer = ml_collections.ConfigDict()
    # 环境贴图路径（PBR 渲染需要）
    cfg.renderer.envmap_path = "_reference_codes/TRELLIS.2/assets/hdri/forest.exr"
    # DepthPeeler 参数（MeshPeeledRenderer PBR 模式使用）
    cfg.renderer.peel_layers = 8
    cfg.renderer.bg_color = [0.5, 0.5, 0.5]  # PBR 背景色（灰色）

    # --- Tex 训练超参 ---
    cfg.train = _base_stage_train()

    # --- Tex Guidance 运行时配置 ---
    cfg.guidance = _flowedit_runtime_config()
    # Tex 阶段默认使用 RGB prompt
    cfg.guidance.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.guidance.source_prompt = cfg.guidance.target_prompt

    return cfg


def _distillation_runtime_config():
    """Distillation 运行时参数（per-stage 调用时传入 compute_guidance）。

    包含 CFG scale、loss 权重、聚合策略等，
    不同阶段（Shape / Tex）可使用不同值。

    ★ 采样结构参数（min/max_step_percent / num_timesteps / noise_mode）
      在 cfg.guidance.distillation（init 配置）中设置，全阶段共享。

    所有字段均在 edit4shape/guidance/paradigms/distillation.py 中被读取。
    """
    cfg = ml_collections.ConfigDict()

    # 随机种子
    cfg.seed = 42

    # CFG scale
    cfg.true_cfg_scale = 12

    # Loss 权重（控制 MSE/CSD 模式）
    # - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_cfg)
    # - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
    cfg.mse_weight = 0.0
    cfg.csd_weight = 1.0

    # 多步 loss 聚合方式: "final" | "mean" | "weighted" | "inv_weighted"
    cfg.reduce_mode = "mean"

    # 梯度归一化
    cfg.ada_normalize = True
    cfg.ada_eps = 1e-2

    # Prompt
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.negative_prompt = " "

    return cfg


# =====================================================================
# Per-stage 阶段独立配置
# =====================================================================

def _base_stage_train():
    """阶段训练超参的公共默认值。"""
    cfg = ml_collections.ConfigDict()

    # 训练模式: "lora" | "full" | "frozen"
    cfg.mode = "full"

    cfg.optimizer = ml_collections.ConfigDict()
    cfg.optimizer.type = "adan"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.weight_decay = 0
    if cfg.optimizer.type != "sgd":
        cfg.optimizer.eps = 1e-4

    # Loss 总权重（训练循环中乘以 guidance/reg loss）
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.guidance = 1.0  # Guidance loss 总权重
    cfg.loss.reg = 1e-4       # 正则化 loss 总权重
    return cfg

