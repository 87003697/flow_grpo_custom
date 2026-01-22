"""TRELLIS.2 蒸馏训练基础配置（按模块拆分）。"""
import ml_collections


def get_base_config_general():
    """通用配置（seed, epochs, 频率等）。"""
    cfg = ml_collections.ConfigDict()
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    cfg.verbose = False
    cfg.pipeline_type = "1024"
    cfg.use_wandb = False # 是否启用 wandb 日志
    
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 2
    cfg.freq.save.ckpt = 5
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数）
    cfg.freq.eval = 5
    
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    return cfg


def get_base_config_data():
    """数据配置（训练/评估）。"""
    cfg = ml_collections.ConfigDict()
    
    cfg.train = ml_collections.ConfigDict()
    cfg.train.dir = "dataset/alphaimages_1k/train/images"
    cfg.train.batch_size = 1
    cfg.train.n_view = 1
    cfg.train.yaw_range = [180.0, 180.0]
    cfg.train.pitch_range = [0.0, 0.0]  # 固定 pitch 角度
    cfg.train.r_range = [2.0, 2.0]
    cfg.train.fov_range = [40.0, 40.0]
    
    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.dir = "dataset/alphaimages_1k/test/images"
    cfg.eval.batch_size = 1
    cfg.eval.n_view = 4
    cfg.eval.yaw = 180
    cfg.eval.pitch = 0.0
    cfg.eval.r = 2.0
    cfg.eval.fov = 40.0
    return cfg


def get_base_config_pretrained():
    """预训练权重路径。"""
    cfg = ml_collections.ConfigDict()
    cfg.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


def get_base_config_renderer():
    """渲染器配置。"""
    cfg = ml_collections.ConfigDict()
    cfg.resolution = 1024
    cfg.type = "mesh"
    cfg.ssaa = 1
    cfg.bg_color = [1.0, 1.0, 1.0]
    cfg.near = 1.0
    cfg.far = 100.0
    
    # Normal 渲染模式：
    # - "mesh_pseudo_gt": 伪 GT Mesh 方案（dual_vertices 可微，intersected detach）
    # - "fdg": FDG 可微 Voxel Normal（dual_vertices + intersected_logits 都可微）
    cfg.normal_mode = "fdg"
    return cfg


def get_base_config_train():
    """训练超参（optimizer, loss）。"""
    cfg = ml_collections.ConfigDict()
    
    # 训练模式: "lora" | "full"
    # - "lora": LoRA 微调（默认，显存友好）
    # - "full": 全参微调（需要更多显存，加载独立教师模型）
    cfg.mode = "full"
    
    cfg.gradient_accumulation_steps = 1
    
    cfg.optimizer = ml_collections.ConfigDict()
    cfg.optimizer.type = "adam"
    cfg.optimizer.lr = 3e-5
    cfg.optimizer.beta1 = 0.9
    cfg.optimizer.beta2 = 0.999
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.eps = 1e-4
    
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.ssim = 1.0
    cfg.loss.lpips = 0.0
    cfg.loss.latent_mse = 1.0
    cfg.loss.dino = 0.0
    cfg.loss.reg = 1.0
    cfg.loss.use_neg = False  # 是否启用负样本 loss
    cfg.loss.latent_mse_mode = "weighted"  # "final" | "mean" | "weighted" | "ada" | "ada_position"
    return cfg


def get_base_config_reg():
    """正则化配置。"""
    cfg = ml_collections.ConfigDict()
    cfg.type = "dmd"  # dmd | kl（vsd 已兼容为 dmd）
    cfg.weight_mode = "uniform"  # uniform | t | ada
    cfg.eps = 1e-2  # ada 权重的 epsilon
    return cfg


def get_base_config_guidance():
    """Guidance 配置（FlowEdit + SDS + CSD）。"""
    cfg = ml_collections.ConfigDict()
    
    # ★ 切换 Guidance 类型: "flowedit" | "sds" | "csd"
    cfg.type = "csd"  # 默认使用 CSD
    
    cfg.model_path = "Qwen/Qwen-Image-Edit-2511"
    cfg.edit_resolution = 1024
    
    # 是否使用 autograd 预计算梯度 + SpecifyGradient 注入
    cfg.enable_autograd = True
    
    # === FlowEdit 配置 ===
    cfg.flowedit = ml_collections.ConfigDict()
    cfg.flowedit.pipeline_type = "simple"  # full, simple
    cfg.flowedit.seed = 0
    cfg.flowedit.guidance_scale = 1.0
    cfg.flowedit.steps = 40
    
    # FlowEdit 核心参数
    cfg.flowedit.n_max = 25
    cfg.flowedit.fixed_noise = True
    
    # Target 分支参数
    cfg.flowedit.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.flowedit.true_cfg_scale_tgt = 12.0
    cfg.flowedit.target_prompt_image_indices = [1]
    cfg.flowedit.negative_prompt_tgt = " "
    
    # 多步监督模式: "final" | "mean" | "weighted" | "ada" | "ada_position"
    cfg.flowedit.latent_mse_mode = "weighted"
    
    # === SDS 配置 ===
    cfg.sds = ml_collections.ConfigDict()
    cfg.sds.seed = 0
    cfg.sds.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    cfg.sds.max_step_percent = 0.98   # 最大时间步百分比（0.98 = t=980）
    cfg.sds.weight_type = "uniform"   # 梯度权重类型: "uniform" | "t" | "ada"
    cfg.sds.weight_eps = 1e-2         # ada 权重的 epsilon
    cfg.sds.true_cfg_scale = 1.0      # CFG scale
    cfg.sds.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.sds.negative_prompt = " "
    
    # === CSD 配置 ===
    cfg.csd = ml_collections.ConfigDict()
    cfg.csd.seed = 0
    cfg.csd.min_step_percent = 0.02   # 最小时间步百分比
    cfg.csd.max_step_percent = 0.98   # 最大时间步百分比
    cfg.csd.weight_type = "uniform"   # 梯度权重类型: "uniform" | "t" | "ada"
    cfg.csd.weight_eps = 1e-2         # ada 权重的 epsilon
    cfg.csd.true_cfg_scale = 1.0      # CFG scale
    cfg.csd.target_prompt = "Move the camera. High-definition, ultra-detailed."
    cfg.csd.negative_prompt = " "
    
    return cfg



