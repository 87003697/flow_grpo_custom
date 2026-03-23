"""TRELLIS Stage 2 Contrastive FlowEdit 训练配置 — Latent 空间对比学习。

对应入口: edit4shape.systems.trellis.entries.contrastive_autograd

训练流程：
  Pretrained Rollout (frozen) → clean z₀
  → 加噪 z₀ → zₜ (随机时间步)
  → Student velocity → ẑ₀
  → Decode/Render Teacher z₀ → src images
  → FlowEdit edit → tgt images
  → DINOv2 encode → c_src, c_tgt
  → Teacher denoise with c_tgt / c_src → positive / negative
  → Contrastive loss → backward → θ.grad

配置结构:
    cfg.guidance              → Guidance 初始化（FlowEdit 模型加载）
    cfg.train.guidance        → FlowEdit 运行时参数（prompt / cfg scale / loss 权重）
    cfg.train.noise           → 加噪时间步采样配置
    cfg.train.loss.contrastive → 对比 loss 参数（ada, eps）
    cfg.renderer              → 渲染器配置（仅 gs）
    cfg.train                 → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）
"""
import ml_collections


# =====================================================================
# 辅助配置函数
# =====================================================================

def _flowedit_init_config(g: ml_collections.ConfigDict):
    """FlowEdit 采样参数（构造 Pipeline 时读取一次）。"""
    g.flowedit = ml_collections.ConfigDict()
    g.flowedit.steps = 12       # 总时间步数
    g.flowedit.n_max = 9        # 实际执行的最后 n_max 步
    g.flowedit.noise_mode = "aligned"  # 加噪模式（aligned / random）
    g.flowedit.csd_pos_mode = "cfg"
    g.flowedit.csd_neg_mode = "uncond"
    g.flowedit.remove_tgt_neg = True


def _flowedit_runtime_config():
    """FlowEdit 运行时参数（per-call，传入 compute_guidance）。"""
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.bg_color = [0.5, 0.5, 0.5]

    # Target 分支
    cfg.true_cfg_scale_tgt = 8
    cfg.target_prompt = "Rotate the camera."
    cfg.negative_prompt_tgt = " "

    # Source 分支
    cfg.true_cfg_scale_src = -1 * cfg.true_cfg_scale_tgt  # 反向引导
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # 多步 Loss 聚合（compute_guidance 内部仍会计算，字段必须存在）
    cfg.reduce_mode = "final"
    cfg.ada_normalize = False
    cfg.ada_eps = 1e-4

    # Guidance 内部 loss 权重（Contrastive 不使用该 loss 反传，全部置 0 跳过计算）
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.latent_mse = 0.0
    cfg.loss.latent_csd = 0.0
    cfg.loss.tgt_branch = 1.0   # ★ 需保持 > 0 以启用 tracker 记录 edited_imgs
    cfg.loss.src_branch = 0.0
    cfg.loss.mse = 0.0
    cfg.loss.ssim = 0.0
    cfg.loss.lpips = 0.0
    cfg.loss.dino = 0.0
    cfg.loss.clip = 0.0

    return cfg


def _adaptive_distance_config(cfg: ml_collections.ConfigDict):
    """为 train/eval 数据添加 adaptive_distance 配置。"""
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
    """TRELLIS Stage 2 Contrastive FlowEdit 训练配置（Latent 空间对比学习）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_contrastive"
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
    cfg.data.eval.n_view = 3
    cfg.data.eval.yaw_range = [90.0, 270.0]
    cfg.data.eval.pitch_range = [0.0, 0.0]
    cfg.data.eval.r_range = [2.0, 2.0]
    cfg.data.eval.fov_range = [40.0, 40.0]
    _adaptive_distance_config(cfg)

    # === 预训练权重 ===
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置（★ 仅 GS Color 渲染） ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024
    cfg.renderer.type = "gs"
    cfg.renderer.ssaa = 1

    cfg.renderer.gs = ml_collections.ConfigDict()
    cfg.renderer.gs.near = 0.8
    cfg.renderer.gs.far = 1.6
    cfg.renderer.gs.bg_color = [1.0, 1.0, 1.0]

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()

    tr.mode = "full"  # Contrastive 使用全参微调

    # Rollout 模式: "pretrained" (off-policy) | "student" (on-policy)
    tr.rollout_mode = "student"

    # 单步去噪是否使用 CFG: True = 保持 pipeline 默认, False = cfg_strength 设为 1（无 CFG）
    tr.denoise_cfg = False

    tr.gradient_accumulation_steps = 1
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adan"
    tr.optimizer.lr = 1e-4
    tr.optimizer.weight_decay = 0.0
    tr.optimizer.eps = 1e-4

    # === 正则化配置 ===
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "none"  # Contrastive 不需要 rollout 正则化

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    # === ★ 加噪时间步采样配置 ===
    tr.noise = ml_collections.ConfigDict()
    tr.noise.t_min = 0.02
    tr.noise.t_max = 0.98

    # === Guidance 初始化配置（FlowEdit 模型加载） ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "flowedit"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _flowedit_init_config(g)

    # === Guidance 运行时配置（FlowEdit） ===
    tr.guidance = _flowedit_runtime_config()
    tr.guidance.bg_color = cfg.renderer.gs.bg_color

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0      # Contrastive loss 权重
    tr.loss.reg = 1.0           # velocity 正则化权重
    tr.loss.reg_type = "x1"     # 正则化类型: "v" | "x0" | "x1"

    # ★ Contrastive loss 配置
    tr.loss.contrastive = ml_collections.ConfigDict()
    tr.loss.contrastive.ada = False
    tr.loss.contrastive.eps = 1e-4

    # GS 表示正则化（可选）
    tr.loss.gs_reg = ml_collections.ConfigDict()
    tr.loss.gs_reg.vol = 0.0
    tr.loss.gs_reg.opacity = 0.0

    return cfg
