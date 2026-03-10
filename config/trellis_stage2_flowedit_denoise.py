"""TRELLIS Stage 2 FlowEdit 训练配置 — Pretrained Rollout + Finetuned 单步去噪。

对应入口: edit4shape.systems.trellis.entries.flowedit_autograd

训练流程：
  Pretrained Rollout (frozen) → clean z₀
  → 加噪 z₀ → zₜ (随机时间步)
  → Finetuned 单步去噪 → ẑ₀
  → Decode + Render → comp_rgb
  → 2D FlowEdit Guidance → loss → autograd backward

配置结构:
    cfg.guidance              → Guidance 初始化（FlowEdit 模型加载）
    cfg.train.guidance        → FlowEdit 运行时参数（prompt / cfg scale / loss 权重）
    cfg.train.noise           → 加噪时间步采样配置
    cfg.renderer              → 渲染器配置（仅 gs）
    cfg.train                 → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）
"""
import ml_collections

from config.trellis_stage2_distillation import (
    _flowedit_init_config,
    _flowedit_runtime_config,
    _lora_config,
    _adaptive_distance_config,
)


# =====================================================================
# 主配置
# =====================================================================

def get_config():
    """TRELLIS Stage 2 FlowEdit 训练配置（Pretrained Rollout + 单步去噪）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_flowedit_denoise"
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
    cfg.renderer.gs.bg_color = [1.0, 1.0, 1.0]  # 白色，与 gs renderer 一致

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()

    # 训练模式: "full" | "lora" | "frozen"
    tr.mode = "full"

    if tr.mode != "full":
        _lora_config(cfg)

    tr.gradient_accumulation_steps = 1
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adan"
    tr.optimizer.lr = 1e-4
    tr.optimizer.weight_decay = 0.0
    if tr.optimizer.type != "sgd":
        tr.optimizer.eps = 1e-4

    # === 正则化配置 ===
    # ★ FlowEdit 单步去噪模式下不需要 rollout 正则化（pretrained rollout 是 frozen 的）
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "none"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    # === ★ 加噪时间步采样配置 ===
    tr.noise = ml_collections.ConfigDict()
    tr.noise.t_min = 0.02   # 时间步采样下界（归一化，[0,1]）
    tr.noise.t_max = 0.98   # 时间步采样上界（归一化，[0,1]）

    # === Guidance 初始化配置（FlowEdit 模型加载） ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "flowedit"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _flowedit_init_config(g)


    # === Guidance 运行时配置（FlowEdit） ===
    tr.guidance = _flowedit_runtime_config()
    tr.guidance.bg_color = cfg.renderer.gs.bg_color
    
    tr.guidance.reduce_mode = "final"
    # ada_normalize: 是否使用自适应归一化
    tr.guidance.ada_normalize = False

    # Loss 权重
    tr.guidance.loss.latent_mse = 1.0   # MSE: MSE(src, z_edit)
    tr.guidance.loss.latent_csd = 0.0   # CSD: MSE(src, x0_pos) - MSE(src, x0_neg)

    # 分支权重（> 0 时启用对应 tracker 并计算 loss）
    tr.guidance.loss.tgt_branch = 1.0   # target 分支权重
    tr.guidance.loss.src_branch = 0.0   # source 分支权重（= 0 不启用）


    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0          # FlowEdit guidance 权重
    tr.loss.reg = 1.0               # velocity 正则化权重
    tr.loss.reg_type = "x1"          # 正则化类型: "v" | "x0" | "x1"

    # === GS 表示正则化（可选） ===
    tr.loss.gs_reg = ml_collections.ConfigDict()
    tr.loss.gs_reg.vol = 0.0
    tr.loss.gs_reg.opacity = 0.0

    return cfg
