import ml_collections


def _bilevel_distillation_config(g: ml_collections.ConfigDict):
    """双层蒸馏配置（VSD - Variational Score Distillation）
    
    教师-学生双层优化：
        外层 VSD Loss（优化 3D 模型）：
            复用 CSD 体系 → x0_pos = x0_teacher, x0_neg = x0_student
            loss = mse_weight * MSE(src, x0_teacher)
                 + csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))
        
        内层 Student Loss（优化 LoRA）：
            loss_student = lambda_sup * MSE(v_student, noise - clean_latents)
    """
    g.bilevel_distillation = ml_collections.ConfigDict()
    
    g.bilevel_distillation.seed = 0
    g.bilevel_distillation.min_step_percent = 0.02    # 最小时间步百分比
    g.bilevel_distillation.max_step_percent = 0.50    # 最大时间步百分比
    
    g.bilevel_distillation.true_cfg_scale = 4        # CFG scale
    
    # 外层 VSD Loss 权重（复用 CSD 体系）
    g.bilevel_distillation.mse_weight = 0.0           # MSE loss: MSE(src, x0_teacher)
    g.bilevel_distillation.csd_weight = 1.0           # CSD loss: MSE(src, x0_teacher) - MSE(src, x0_student)
    
    # MTS（多时间步采样）
    g.bilevel_distillation.num_timesteps = 1         # 采样时间步数量
    g.bilevel_distillation.reduce_mode = "mean"       # 多步 loss 聚合: "final" | "mean" | "weighted" | "inv_weighted"
    
    # 梯度归一化
    g.bilevel_distillation.ada_normalize = True       # 自适应梯度归一化
    g.bilevel_distillation.ada_eps = 1e-2             # 归一化 epsilon
    
    # 噪声模式
    g.bilevel_distillation.noise_mode = "random"      # "random" | "fixed" | "aligned" | "inversion_*" | "traj_*"
    
    # Prompt
    g.bilevel_distillation.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.bilevel_distillation.negative_prompt = " "
    
    # ======== VSD 专属参数 ========
    
    # 内层学生 Loss 权重
    g.bilevel_distillation.lambda_sup = 1.0           # 学生监督 loss 权重
    
    # LoRA 配置（注入到 Transformer 的学生模型）
    g.bilevel_distillation.lora_rank = 64             # LoRA rank
    g.bilevel_distillation.lora_alpha = 64            # LoRA alpha（通常与 rank 相同）
    g.bilevel_distillation.lora_dropout = 0.1         # LoRA dropout
    g.bilevel_distillation.lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]  # 目标模块
    g.bilevel_distillation.lora_lr = 1e-4             # LoRA 学习率


def _lora_config(cfg: ml_collections.ConfigDict):
    """LoRA 配置（仅在非 full 模式下使用）。"""
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.lora.lora_alpha = 32  # LoRA alpha（通常与 rank 相同）
    cfg.lora.lora_dropout = 0.0  # LoRA dropout
    cfg.lora.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]  # 目标模块


def _sde_rollout_config(cfg: ml_collections.ConfigDict):
    """SDE Rollout 专用配置（仅当 rollout.type == "sde" 时使用）。"""
    cfg.rollout.noise_level = 0.7  # 噪声水平 (0~1)
    cfg.rollout.sde_type = "cps"   # SDE 类型: "sde" | "cps"


def _adaptive_distance_config(cfg: ml_collections.ConfigDict):
    """统一添加 adaptive_distance 配置（假设 cfg.data.train / cfg.data.eval 已创建）。"""
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9

    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_bilevel_distill"
    cfg.use_wandb = False  # 是否启用 wandb 日志
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    
    # === 频率控制 ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1  # 训练可视化保存步频
    cfg.freq.save.ckpt = 10000    # ckpt 保存频率（epoch）
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数：4, 9, 16...）
    cfg.freq.eval = 1         # 评估频率（epoch）

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()
    
    # 训练数据配置
    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v2/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1                      # 训练时视角数
    cfg.data.train.yaw_range = [180.0, 180.0]      # yaw 采样范围 (度)
    cfg.data.train.pitch_range = [0.0, 0.0]     # pitch 采样范围 (度)
    cfg.data.train.r_range = [2.0, 2.0]            # 相机距离范围
    cfg.data.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)
    
    # 评估数据配置
    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v2/test"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 1                       # 评估时视角数
    cfg.data.eval.yaw = 180                        # 评估时固定 yaw (度)
    cfg.data.eval.pitch = 0.0                      # 评估时固定 pitch (度)
    cfg.data.eval.r = 2.0                          # 评估时相机距离
    cfg.data.eval.fov = 40.0                       # 评估时视场角 (度)
    _adaptive_distance_config(cfg)

    # === 预训练权重 ===
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置 ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024  # 渲染分辨率，FlowEdit 要求 1024×1024
    cfg.renderer.type = "gs"  # 可选: mesh / gs
    cfg.renderer.ssaa = 1  # 超采样倍数
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]
    if cfg.renderer.type == "mesh":
        cfg.renderer.near, cfg.renderer.far = 1.0, 100.0
    else:
        cfg.renderer.near, cfg.renderer.far = 0.8, 1.6

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    
    # 训练模式: "full" | "lora" | "frozen"
    # - "full": 全参微调，教师模型为冻结副本（放在 guidance 设备）
    # - "lora": LoRA 微调，教师模型为禁用 adapter 后的原始权重
    # - "frozen": 冻结模式（仅推理，不训练）
    tr.mode = "full"

    # LoRA 配置：仅在非 full 模式下添加
    if tr.mode != "full":
        _lora_config(cfg)
    
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "sgd"
    tr.optimizer.lr = 5e-3
    tr.optimizer.weight_decay = 0.0

    # === 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    # - "none": 不使用正则化
    # - "x0": MSE(x0_stu, x0_tea) / t²，梯度可流向历史步
    # - "v": MSE(v_stu, v_tea)，梯度仅当前步
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "none"  # 正则化类型: "none" | "x0" | "v"

    # === Rollout 配置 ===
    # 控制采样方式：ODE（确定性）或 SDE（随机）
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"  # "ode" | "sde"
    
    # SDE 专用配置（仅当 rollout.type == "sde" 时生效）
    if cfg.rollout.type == "sde":
        _sde_rollout_config(cfg)

    # === Guidance 配置（通用）===
    # Guidance 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → Guidance 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # Guidance 类型: bilevel_distillation（VSD 双层蒸馏）
    g.type = "bilevel_distillation"
    
    # 模型路径（HuggingFace ID 或本地路径）
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    
    # 工作分辨率
    g.edit_resolution = 1024

    # 加载 bilevel_distillation 专用配置
    _bilevel_distillation_config(g)

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0     # Guidance loss 权重（统一控制 flowedit/sds/csd/csd_rev）
    tr.loss.reg = 0.           # 正则化 loss 权重（DMD/KL）
    
    return cfg
