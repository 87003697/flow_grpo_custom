"""TRELLIS Stage 2 FlowEdit 训练配置。

对应模块: edit4shape.systems.trellis.system

配置结构:
    cfg.guidance       → Guidance 初始化（固定为 flowedit）
    cfg.train.guidance → FlowEdit 运行时参数
    cfg.renderer       → 渲染器配置
    cfg.train          → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）
"""
import ml_collections


# =====================================================================
# FlowEdit Guidance 配置
# =====================================================================



# =====================================================================
# Guidance 运行时配置（per-call，传入 compute_guidance）
# =====================================================================


def _flowedit_init_config(g: ml_collections.ConfigDict):
    """FlowEdit 专属 init 参数（写入 cfg.guidance.flowedit）。

    这些参数在训练过程中不会变化，仅在构造 Pipeline 时读取一次。
    """
    g.flowedit = ml_collections.ConfigDict()

    # 采样步数
    g.flowedit.steps = 12   # num_inference_steps: 总时间步数
    g.flowedit.n_max = 9    # 实际执行的最后 n_max 步

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
    # - use_src_record: 记录 source 分支的 x0 正负对（默认 False）
    g.flowedit.use_tgt_record = True
    g.flowedit.use_src_record = True

    # 条件图背景色 float [0,1]，应与 cfg.renderer.bg_color 保持一致
    g.bg_color = [1.0, 1.0, 1.0]


def _flowedit_runtime_config():
    """FlowEdit 运行时参数。

    所有字段均在 edit4shape/guidance/paradigms/flowedit.py 中被读取。
    """
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0

    # Target 分支参数
    cfg.true_cfg_scale_tgt = 4
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.negative_prompt_tgt = " "

    # Source 分支参数
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
    """TRELLIS Stage 2 FlowEdit 训练配置。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_flowedit"
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
    cfg.data.eval.yaw_range = [90.0, 270.0]
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

    # Per-renderer near/far
    if cfg.renderer.type == "mesh":
        cfg.renderer.mesh = ml_collections.ConfigDict()
        cfg.renderer.mesh.near = 1.0
        cfg.renderer.mesh.far = 100.0
    elif cfg.renderer.type == "gs":
        cfg.renderer.gs = ml_collections.ConfigDict()
        cfg.renderer.gs.near = 0.8
        cfg.renderer.gs.far = 1.6
    else:
        raise ValueError(f"Invalid renderer type: {cfg.renderer.type}")

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()

    # 训练模式: "full" | "lora" | "frozen"
    tr.mode = "full"

    if tr.mode != "full":
        _lora_config(cfg)

    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adan"
    tr.optimizer.lr = 1e-4
    tr.optimizer.weight_decay = 0.0
    if tr.optimizer.type != "sgd":  # 其他优化器需要设置 eps
        tr.optimizer.eps = 1e-5


    # === 正则化配置 ===
    # reg.type: "x0" (MSE/t²) | "x1" (MSE, 不除t²) | "v" (速度场MSE) | "none"
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "x1"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # === Guidance 初始化配置（固定为 FlowEdit） ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "flowedit"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _flowedit_init_config(g)

    # === Guidance 运行时配置（FlowEdit） ===
    tr.guidance = _flowedit_runtime_config()

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0
    tr.loss.reg = 1e-4              # 蒸馏正则化权重（latent space student-teacher matching）

    # === GS 表示正则化（reg_vol / reg_opacity） ===
    # 约束 flow model 输出的 latent 经 GS Decoder 解码后产生合理的 Gaussian：
    #   vol:     惩罚 Gaussian 体积过大（避免巨型 blob），建议 1000~10000
    #   opacity: 鼓励不透明度接近 1（避免半透明模糊），建议 0.001
    # 设为 0 则不启用对应正则化；renderer.type 非 "gs" 时自动跳过
    tr.loss.gs_reg = ml_collections.ConfigDict()
    tr.loss.gs_reg.vol = 0 #10000.0    # 体积正则化权重
    tr.loss.gs_reg.opacity = 0 #0.001  # 不透明度正则化权重

    return cfg
