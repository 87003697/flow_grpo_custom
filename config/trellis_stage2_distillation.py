"""TRELLIS Stage 2 蒸馏训练配置。

对应模块: edit4shape.systems.trellis / edit4shape.systems.trellis_pp

配置结构:
    cfg.guidance       → Guidance 初始化（type, model_path, flowedit.pipeline_type）
    cfg.train.guidance → Guidance 运行时（prompt, loss 权重, 聚合策略等）
    cfg.renderer       → 渲染器配置
    cfg.train          → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）

★ Guidance 配置分两层：
  - cfg.guidance: 初始化配置（model_path 等 + 范式专属子配置），只加载一次模型
  - cfg.train.guidance: 运行时配置（每次 compute_guidance 调用时传入）
"""
import ml_collections


# =====================================================================
# Guidance 初始化子配置（范式专属 init 参数）
# =====================================================================



# =====================================================================
# Guidance 运行时配置（per-call，传入 compute_guidance）
# =====================================================================


def _flowedit_init_config(g: ml_collections.ConfigDict):
    """FlowEdit 专属 init 参数（写入 cfg.guidance.flowedit）。"""
    g.flowedit = ml_collections.ConfigDict()
    # Pipeline 类型: "simple" | "full"
    # - "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
    # - "full": FlowEditFullPipeline，双分支都使用模型推理（效果更好）
    g.flowedit.pipeline_type = "full"


def _flowedit_runtime_config():
    """FlowEdit 运行时参数。

    所有字段均在 edit4shape/guidance/paradigms/flowedit.py
    或 edit4shape/guidance/pipelines/adapters.py 中被读取。
    """
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.steps = 12   # num_inference_steps: 总时间步数
    cfg.n_max = 9    # 实际执行的最后 n_max 步

    # 噪声模式
    # pipeline_type="simple" 支持:
    #   - random / fixed / aligned / delta
    #   - traj_cond / traj_uncond / traj_cfg: DNAEdit 轨迹对齐
    # pipeline_type="full" 支持:
    #   - random: 每步随机噪声
    #   - fixed: 固定噪声（所有 step 共用）
    #   - aligned: DNAEdit 风格累积补偿 ε -= (v_cond - v_uncond) * (1 - t)
    #   - delta: 双分支差分补偿 ε -= (v_cfg_tgt - v_cfg_src) * (1 - t)
    cfg.noise_mode = "aligned"

    # MTS 采样: 是否使用均匀分区随机采样
    # - False: 使用 scheduler 的固定时间步序列
    # - True: 在 [0.02, 0.98] 范围内均匀分区随机采样 steps 个时间步
    cfg.use_mts_sampling = True

    # Target 分支参数
    cfg.true_cfg_scale_tgt = 4
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.negative_prompt_tgt = " "

    # Source 分支参数（full 模式需要；simple 模式下不会读取）
    cfg.true_cfg_scale_src = -1 * cfg.true_cfg_scale_tgt
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # 多步 Loss 配置
    # reduce_mode: 聚合方式
    #   - "final": 只用最后一步
    #   - "mean": 均匀加权
    #   - "weighted": 1/k 加权（前期大）
    #   - "inv_weighted": k/K 加权（后期大）
    cfg.reduce_mode = "mean"
    # ada_normalize: 是否使用自适应归一化
    cfg.ada_normalize = True
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    cfg.ada_eps = 1e-1

    # Loss 权重
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.latent_mse = 0.0   # MSE: MSE(src, z_edit)
    cfg.loss.latent_csd = 1.0   # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)

    return cfg


def _distillation_init_config(g: ml_collections.ConfigDict):
    """Distillation 专属 init 参数（写入 cfg.guidance.distillation）。

    当前 DistillationGuidance.__init__ 不需要额外 init 参数
    （只读 guidance_cfg.model_path），预留占位。
    """
    g.distillation = ml_collections.ConfigDict()


def _distillation_runtime_config():
    """Distillation 运行时参数。

    所有字段均在 edit4shape/guidance/paradigms/distillation.py 中被读取。

    x0 预测定义：
        - x0_pos: 纯 cond 预测 (v_cond)，CSD 正样本
        - x0_neg: 纯 uncond 预测 (v_uncond)，CSD 负样本
        - x0_cfg: CFG 后预测 (v_cfg = v_uncond + scale * (v_cond - v_uncond))

    通过 mse_weight 和 csd_weight 控制 loss 类型：
        - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_cfg)
        - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
        - mse_weight=1, csd_weight=1 → 混合模式
    """
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    cfg.max_step_percent = 0.50   # 最大时间步百分比（0.50 = t=500）

    cfg.true_cfg_scale = 12       # CFG scale

    # Loss 权重（控制 MSE/CSD 模式）
    cfg.mse_weight = 0.0          # MSE loss 权重
    cfg.csd_weight = 1.0          # CSD loss 权重

    # MTS（多时间步采样）
    cfg.num_timesteps = 20
    cfg.reduce_mode = "mean"

    # 梯度归一化
    cfg.ada_normalize = True
    cfg.ada_eps = 1e-2

    # 噪声模式
    # - "random" / "fixed" / "aligned" / "inversion_*" / "traj_*"
    # 注意：delta 模式仅 FlowEdit 双分支可用
    cfg.noise_mode = "fixed"

    # Prompt
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.negative_prompt = " "

    return cfg


# =====================================================================
# 其他辅助配置
# =====================================================================

def _lora_config(cfg: ml_collections.ConfigDict):
    """LoRA 配置（仅在非 full 模式下使用）。"""
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.lora.lora_alpha = 32
    cfg.lora.lora_dropout = 0.0
    cfg.lora.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]


def _sde_rollout_config(cfg: ml_collections.ConfigDict):
    """SDE Rollout 专用配置（仅当 rollout.type == "sde" 时使用）。"""
    cfg.rollout.noise_level = 0.7
    cfg.rollout.sde_type = "cps"


def _adaptive_distance_config(cfg: ml_collections.ConfigDict):
    """统一添加 adaptive_distance 配置。"""
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9

    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9


# =====================================================================
# 主配置
# =====================================================================

def get_config():
    """TRELLIS Stage 2 蒸馏训练配置。

    ★ Guidance 配置分两层：
      - cfg.guidance: 初始化（模型加载 + 范式专属 init 参数）
      - cfg.train.guidance: 运行时（每次 compute_guidance 调用时传入）
    """
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
    cfg.use_wandb = False
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False

    # === 频率控制 ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1
    cfg.freq.save.ckpt = 1
    cfg.freq.save.progress_samples = 4
    cfg.freq.eval = 1

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()

    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v3/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1
    cfg.data.train.yaw_range = [0.0, 360.0]
    cfg.data.train.pitch_range = [0.0, 0.0]
    cfg.data.train.r_range = [2.0, 2.0]
    cfg.data.train.fov_range = [40.0, 40.0]

    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v3/test"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 6
    cfg.data.eval.yaw_range = [0.0, 360.0]
    cfg.data.eval.pitch_range = [0.0, 0.0]
    cfg.data.eval.r_range = [2.0, 2.0]
    cfg.data.eval.fov_range = [40.0, 40.0]
    _adaptive_distance_config(cfg)

    # === 预训练权重 ===
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置 ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024
    cfg.renderer.type = "gs"
    cfg.renderer.ssaa = 1
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]
    if cfg.renderer.type == "mesh":
        cfg.renderer.near, cfg.renderer.far = 1.0, 100.0
    else:
        cfg.renderer.near, cfg.renderer.far = 0.8, 1.6

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()

    # 训练模式: "full" | "lora" | "frozen"
    tr.mode = "full"

    if tr.mode != "full":
        _lora_config(cfg)

    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "sgd"
    tr.optimizer.lr = 1e-3
    tr.optimizer.weight_decay = 0.0

    # === 正则化配置 ===
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "x0"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # =========================================================================
    # Guidance 初始化配置（只加载模型 + 范式专属 init 参数）
    # =========================================================================
    cfg.guidance = g = ml_collections.ConfigDict()

    # ★ 切换 Guidance 类型: "flowedit" | "distillation"
    g.type = "flowedit"

    # 共用 init 参数
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024

    # 范式专属 init 参数（pipeline_type 等）
    if g.type == "flowedit":
        _flowedit_init_config(g)
    elif g.type == "distillation":
        _distillation_init_config(g)
    else:
        raise ValueError(f"Unknown guidance type: {g.type}. Choose from: flowedit, distillation")

    # =========================================================================
    # Guidance 运行时配置（每次 compute_guidance 调用时传入）
    # =========================================================================
    if g.type == "flowedit":
        tr.guidance = _flowedit_runtime_config()
    elif g.type == "distillation":
        tr.guidance = _distillation_runtime_config()

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0
    tr.loss.reg = 0.1

    return cfg
