"""TRELLIS Stage 2 FlowEdit Hybrid 训练配置。

对应入口: edit4shape.systems.trellis.entries.hybrid_autograd
双路渲染: Mesh Normal + GS Color 同时 guidance

配置结构:
    cfg.guidance              → Guidance 初始化（固定为 flowedit，mesh / gs 共享同一模型）
    cfg.train.guidance_normal → Mesh Normal 路 FlowEdit 运行时参数
    cfg.train.guidance_color  → GS Color 路 FlowEdit 运行时参数
    cfg.renderer              → 渲染器公共配置（near/far 需兼容 mesh + gs）
    cfg.train                 → 训练超参（mode, optimizer, loss, gradient_accumulation_steps）
"""
import ml_collections

from config.trellis_stage2_distillation import (
    _flowedit_init_config,
    _flowedit_runtime_config,
    _lora_config,
    _sde_rollout_config,
    _adaptive_distance_config,
)


# =====================================================================
# Hybrid 专用：为 Normal / Color 各自生成可独立调参的运行时配置
# =====================================================================

def _flowedit_normal_runtime_config():
    """Mesh Normal 路的 FlowEdit 运行时参数。

    默认与 Color 路相同，可按需调整 cfg_scale / prompt / reduce_mode 等。
    """
    cfg = _flowedit_runtime_config()
    # ── Normal 路可独立覆写的参数 ──
    # 例如：Normal guidance 可能需要不同的 prompt
    cfg.target_prompt = "Move the camera. Convert to normal map."
    # 例如：Normal 路可能需要不同的 cfg scale
    # cfg.true_cfg_scale_tgt = 4
    return cfg


def _flowedit_color_runtime_config():
    """GS Color 路的 FlowEdit 运行时参数。

    默认与 Normal 路相同，可按需调整。
    """
    cfg = _flowedit_runtime_config()
    # ── Color 路可独立覆写的参数 ──
    cfg.target_prompt = "Move the camera. High-definition, ultra-detailed."
    return cfg


# =====================================================================
# 主配置
# =====================================================================

def get_config():
    """TRELLIS Stage 2 FlowEdit Hybrid 训练配置。

    双路渲染：Mesh Normal + GS Color 各自独立 guidance，梯度在 proxy 上累加。
    """
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_hybrid_flowedit"
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

    # === Renderer 配置（★ Hybrid: mesh + gs 各自独立 near/far） ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024
    cfg.renderer.type = "hybrid"   # 标记为 hybrid（build_hybrid_system 不读取此字段）
    cfg.renderer.ssaa = 1
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]

    # Per-renderer near/far
    cfg.renderer.mesh = ml_collections.ConfigDict()
    cfg.renderer.mesh.near = 1.0
    cfg.renderer.mesh.far = 100.0

    cfg.renderer.gs = ml_collections.ConfigDict()
    cfg.renderer.gs.near = 0.8
    cfg.renderer.gs.far = 1.6

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
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "x0"

    # === Rollout 配置 ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"

    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # === Guidance 初始化配置（固定为 FlowEdit，mesh / gs 共享同一模型） ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "flowedit"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _flowedit_init_config(g)

    # === Guidance 运行时配置（★ Hybrid: 双路各自独立） ===
    # TrellisHybridOps.get_render_passes() 读取：
    #   cfg.train.guidance_normal  → Mesh Normal 路
    #   cfg.train.guidance_color   → GS Color 路
    tr.guidance_normal = _flowedit_normal_runtime_config()
    tr.guidance_color = _flowedit_color_runtime_config()

    # === Loss 配置（★ Hybrid: 双路各自独立权重） ===
    # TrellisHybridOps.get_render_passes() 读取：
    #   cfg.train.loss.guidance_normal  → Mesh Normal guidance 权重
    #   cfg.train.loss.guidance_color   → GS Color guidance 权重
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance_normal = 1.0   # Mesh Normal guidance 权重
    tr.loss.guidance_color = 1.0    # GS Color guidance 权重
    tr.loss.reg = 1e-4              # 正则化权重（与单路一致）

    return cfg
