"""TRELLIS Stage 2 双层蒸馏（VSD）训练配置。

对应模块: edit4shape.systems.trellis_bilevel

配置结构:
    cfg.guidance                         → Guidance 初始化
    cfg.guidance.bilevel_distillation    → VSD 专属配置（init 时全部绑定）
    cfg.train.guidance                   → 运行时占位（bilevel 实际不使用，接口兼容）
    cfg.renderer                         → 渲染器配置
    cfg.train                            → 训练超参

★ Bilevel 特殊性：
  BilevelDistillationGuidance.__init__ 在初始化时读取所有参数，
  compute_guidance 的 guidance_cfg 参数不被使用。
  cfg.train.guidance 指向 cfg.guidance.bilevel_distillation 以保持接口一致。
"""
import ml_collections


def _bilevel_distillation_config(g: ml_collections.ConfigDict):
    """VSD 双层蒸馏完整配置（init 时全部绑定到实例属性）。

    教师-学生双层优化：
        外层 VSD Loss（优化 3D 模型）：
            x0_pos = x0_teacher, x0_neg = x0_student
            loss = csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))

        内层 Student Loss（优化 LoRA）：
            loss_student = lambda_sup * MSE(v_student, noise - clean_latents)
    """
    g.bilevel_distillation = ml_collections.ConfigDict()

    # 蒸馏基础参数
    g.bilevel_distillation.seed = 0
    g.bilevel_distillation.min_step_percent = 0.02
    g.bilevel_distillation.max_step_percent = 0.50
    g.bilevel_distillation.true_cfg_scale = 4

    # 外层 VSD Loss 权重
    g.bilevel_distillation.mse_weight = 0.0
    g.bilevel_distillation.csd_weight = 1.0

    # MTS（多时间步采样）
    g.bilevel_distillation.num_timesteps = 1
    g.bilevel_distillation.reduce_mode = "mean"

    # 梯度归一化
    g.bilevel_distillation.ada_normalize = True
    g.bilevel_distillation.ada_eps = 1e-2

    # 噪声模式
    g.bilevel_distillation.noise_mode = "random"

    # Prompt
    g.bilevel_distillation.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.bilevel_distillation.negative_prompt = " "

    # VSD 专属参数
    g.bilevel_distillation.lambda_sup = 1.0

    # LoRA 配置
    g.bilevel_distillation.lora_rank = 64
    g.bilevel_distillation.lora_alpha = 64
    g.bilevel_distillation.lora_dropout = 0.1
    g.bilevel_distillation.lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    g.bilevel_distillation.lora_lr = 1e-4


def _lora_config(cfg: ml_collections.ConfigDict):
    """LoRA 配置（仅在非 full 模式下使用）。"""
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.lora.lora_alpha = 32
    cfg.lora.lora_dropout = 0.0
    cfg.lora.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]


def _sde_rollout_config(cfg: ml_collections.ConfigDict):
    """SDE Rollout 专用配置。"""
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


def get_config():
    """TRELLIS Stage 2 双层蒸馏（VSD）训练配置。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_bilevel_distill"
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
    cfg.freq.save.ckpt = 10000
    cfg.freq.save.progress_samples = 4
    cfg.freq.eval = 1

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()

    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v2/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1
    cfg.data.train.yaw_range = [0.0, 360.0]
    cfg.data.train.pitch_range = [0.0, 0.0]
    cfg.data.train.r_range = [2.0, 2.0]
    cfg.data.train.fov_range = [40.0, 40.0]

    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v2/test"
    cfg.data.eval.batch_size = 6
    cfg.data.eval.n_view = 1
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

    tr.mode = "full"
    if tr.mode != "full":
        _lora_config(cfg)

    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "sgd"
    tr.optimizer.lr = 5e-3
    tr.optimizer.weight_decay = 0.0

    # === 正则化配置 ===
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "none"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"
    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # =========================================================================
    # Guidance 初始化配置
    # =========================================================================
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "bilevel_distillation"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024

    # VSD 专属配置（init 时全部绑定）
    _bilevel_distillation_config(g)

    # =========================================================================
    # Guidance 运行时配置
    # ★ Bilevel 在 init 时绑定所有参数，compute_guidance 不使用 guidance_cfg。
    #   这里指向 bilevel_distillation 子配置以保持接口一致。
    # =========================================================================
    tr.guidance = g.bilevel_distillation

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0
    tr.loss.reg = 0.

    return cfg
