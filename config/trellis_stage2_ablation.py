"""TRELLIS Stage 2 Distillation ablation 训练配置。

对应模块: edit4shape.systems.trellis.system

配置结构:
    cfg.guidance              → Guidance 初始化（固定为 distillation）
    cfg.guidance.distillation → Distillation 专属 init 配置（采样结构等）
    cfg.train.guidance        → Distillation 运行时参数（prompt / loss 权重等）
    cfg.renderer              → 渲染器配置
    cfg.train                 → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）
"""
import ml_collections


# =====================================================================
# Distillation Guidance 配置
# =====================================================================


def _distillation_init_config(g: ml_collections.ConfigDict):
    """Distillation 专属 init 参数（写入 cfg.guidance.distillation）。

    这些参数在训练过程中不会变化，仅在构造 Pipeline 时读取一次。
    """
    g.distillation = ml_collections.ConfigDict()

    # 噪声模式
    # - "random" / "fixed" / "aligned" / "inversion_*" / "traj_*"
    # 注意：delta 模式仅 FlowEdit 双分支可用
    g.distillation.noise_mode = "fixed"

    # MTS（多时间步采样）
    g.distillation.num_timesteps = 1

    # 时间步范围
    g.distillation.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    g.distillation.max_step_percent = 0.50   # 最大时间步百分比（0.50 = t=500）

    # CSD 正/负样本来源
    # pos: "cond"(纯条件,CFG=1) | "cfg"(原始CFG) | "cfg_rescale"(CFG+L2归一化)
    # neg: "uncond"(纯无条件) | "cond"(纯条件)
    g.distillation.csd_pos_mode = "cond"     # 默认: 纯条件预测
    g.distillation.csd_neg_mode = "uncond"   # 默认: 纯无条件预测

    # 条件图背景色 float [0,1]，应与 cfg.renderer.bg_color 保持一致
    g.bg_color = [1.0, 1.0, 1.0]


def _distillation_runtime_config():
    """Distillation 运行时参数。

    所有字段均在 edit4shape/guidance/paradigms/distillation.py 中被读取。

    x0 预测候选：
        - x0_cond:       纯 cond 预测 (v_cond)
        - x0_uncond:     纯 uncond 预测 (v_uncond)
        - x0_cfg:        原始 CFG 预测 (comb_pred = v_uncond + scale * (v_cond - v_uncond))
        - x0_cfg_rescale: CFG + L2 norm rescale 预测 (v_cfg = comb_pred * (||v_cond|| / ||comb_pred||))

    CSD 正/负样本由 csd_pos_mode / csd_neg_mode 控制（init 配置）：
        - pos="cond",  neg="uncond"      → 原始定义
        - pos="cfg",   neg="uncond"      → 与 Flux 一致（高CFG vs 无条件）
        - pos="cfg_rescale", neg="uncond" → CFG rescale vs 无条件（更稳定）
        - pos="cfg",   neg="cond"        → 仅 CFG 增强部分作为对比

    通过 mse_weight 和 csd_weight 控制 loss 类型：
        - mse_weight=1, csd_weight=0 → 纯 MSE: MSE(src, x0_cfg_rescale)
        - mse_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
        - mse_weight=1, csd_weight=1 → 混合模式
    """
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0

    cfg.true_cfg_scale = 4       # CFG scale

    # Loss 权重（控制 MSE/CSD 模式）
    cfg.mse_weight = 0.0          # MSE loss 权重
    cfg.csd_weight = 1.0          # CSD loss 权重

    # 多步 Loss 配置
    cfg.reduce_mode = "mean"

    # 梯度归一化
    cfg.ada_normalize = True
    cfg.ada_eps = 1e-2

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
    """TRELLIS Stage 2 Distillation ablation 训练配置。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distillation_ablation"
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
    # reg.type: "x0" (MSE/t²) | "x1" (MSE, 不除t²) | "v" (速度场MSE) | "none"
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "x0"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # === Guidance 初始化配置（固定为 Distillation） ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "distillation"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _distillation_init_config(g)

    # === Guidance 运行时配置（Distillation） ===
    tr.guidance = _distillation_runtime_config()

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0
    tr.loss.reg = 0.0001

    return cfg
