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

    # CSD 正/负样本来源
    # pos: "cond" (纯条件,CFG=1) | "cfg" (原始CFG) | "cfg_rescale" (CFG+L2归一化)
    # neg: "uncond" (纯无条件) | "cond" (纯条件)
    g.flowedit.csd_pos_mode = "cfg"     # 默认: 纯条件预测
    g.flowedit.csd_neg_mode = "uncond"   # 默认: 纯无条件预测

    # 是否用 src 分支的 x0_neg 替换 tgt 分支的 x0_neg
    g.flowedit.remove_tgt_neg = True


def _flowedit_runtime_config():
    """FlowEdit 运行时参数。

    所有字段均在 edit4shape/guidance/paradigms/flowedit.py 中被读取。
    """
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.bg_color = [1.0, 1.0, 1.0]  # 条件图背景色 float [0,1]，应与 renderer per-renderer bg_color 保持一致

    # Target 分支参数
    cfg.true_cfg_scale_tgt = 7.5
    cfg.target_prompt = "Rotate the camera. White background."
    cfg.negative_prompt_tgt = " "

    # Source 分支参数
    cfg.true_cfg_scale_src = 0
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # 多步 Loss 配置
    # reduce_mode: 聚合方式
    #   - "final": 只用最后一步
    #   - "mean": 均匀加权
    #   - "weighted": 1/k 加权（前期大）
    #   - "inv_weighted": k/K 加权（后期大）
    cfg.reduce_mode = "final"
    # ada_normalize: 是否使用自适应归一化
    cfg.ada_normalize = False
    # ada_eps: 自适应归一化的 epsilon（防止除零）
    cfg.ada_eps = 1e-4

    # Loss 权重（pixel-only，latent loss 已移除）
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.mse = 1.0          # MSE: 像素空间均方误差
    cfg.loss.ssim = 0.0         # SSIM: 结构相似性（1 - SSIM）
    cfg.loss.lpips = 0.0        # LPIPS: 感知相似性（VGG 特征距离）
    cfg.loss.dino = 0.0         # DINO: DINOv2 特征余弦距离
    cfg.loss.clip = 0.0         # CLIP: CLIP 图像特征余弦距离

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
    cfg.data.eval.n_view = 3
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

    # Per-renderer 配置（near/far + bg_color）
    if cfg.renderer.type == "mesh":
        cfg.renderer.mesh = ml_collections.ConfigDict()
        cfg.renderer.mesh.near = 1.0
        cfg.renderer.mesh.far = 100.0
        cfg.renderer.mesh.bg_color = [1.0, 1.0, 1.0]
    elif cfg.renderer.type == "gs":
        cfg.renderer.gs = ml_collections.ConfigDict()
        cfg.renderer.gs.near = 0.8
        cfg.renderer.gs.far = 1.6
        cfg.renderer.gs.bg_color = [1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Invalid renderer type: {cfg.renderer.type}")

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()

    # 训练模式: "full" | "lora" | "frozen"
    tr.mode = "full"

    if tr.mode != "full":
        _lora_config(cfg)

    tr.dense_optimizer = False
    tr.rollout_mode = "student"
    tr.student_denoise_cfg = False
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adan"
    tr.optimizer.lr = 1e-4
    tr.optimizer.weight_decay = 0.0
    if tr.optimizer.type != "sgd":  # 其他优化器需要设置 eps
        tr.optimizer.eps = 1e-4


    # === 加噪时间步采样 ===
    tr.noise = ml_collections.ConfigDict()
    tr.noise.t_min = 0.02
    tr.noise.t_max = 0.98

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"
    # rollout.reg.type: "x0" (MSE/t²) | "x1" (MSE, 不除t²) | "v" (速度场MSE) | "none"
    cfg.rollout.reg = ml_collections.ConfigDict()
    cfg.rollout.reg.type = "x1"

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
    tr.loss.reg = 1e-0              # 蒸馏正则化权重（latent space student-teacher matching）

    return cfg
