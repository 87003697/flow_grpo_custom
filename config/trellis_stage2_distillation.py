import ml_collections


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    
    # === 频率控制 ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 2  # 训练可视化保存步频
    cfg.freq.save.ckpt = 5    # ckpt 保存频率（epoch）
    cfg.freq.eval = 5         # 评估频率（epoch）

    # === LoRA 配置 ===
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # === 数据路径 ===
    cfg.train_data_dir = "dataset/alphaimages_1k/train/images"
    cfg.eval_data_dir = "dataset/alphaimages_1k/test/images"

    # === 预训练权重 ===
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === 数据批次 ===
    cfg.batch_size = 1
    cfg.eval_batch_size = 1

    # === 相机与渲染配置 (TRELLIS 风格: yaw/pitch/r/fov) ===
    cfg.camera = cam = ml_collections.ConfigDict()
    cam.render_resolution = 1024  # FlowEdit 要求 1024×1024

    # 训练时相机参数
    cam.train = ml_collections.ConfigDict()
    cam.train.n_view = 1                      # 训练时视角数
    cam.train.yaw_range = [180.0, 180.0]        # yaw 采样范围 (度)
    cam.train.pitch_range = [-15.0, 45.0]     # pitch 采样范围 (度)
    cam.train.r_range = [2.0, 2.0]            # 相机距离范围
    cam.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)

    # 评估时相机参数
    cam.eval = ml_collections.ConfigDict()
    cam.eval.n_view = 4                       # 评估时视角数
    cam.eval.yaw = 180                        # 评估时固定 yaw (度)
    cam.eval.pitch = 0.0                     # 评估时固定 pitch (度)
    cam.eval.r = 2.0                          # 评估时相机距离
    cam.eval.fov = 40.0                       # 评估时视场角 (度)

    # === Renderer 配置 ===
    cfg.renderer = renderer = ml_collections.ConfigDict()
    renderer.type = "gs"  # 可选: mesh / gs
    renderer.bg_color = "random"
    renderer.near = 0.8  # 近裁剪面
    renderer.far = 1.6  # 远裁剪面
    renderer.ssaa = 1  # 超采样倍数

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adam"
    tr.optimizer.lr = 3e-4
    tr.optimizer.beta1 = 0.5
    tr.optimizer.beta2 = 0.999
    tr.optimizer.weight_decay = 1e-4
    tr.optimizer.eps = 1e-6

    # === Guidance 配置 ===
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # API 服务参数
    g.service = ml_collections.ConfigDict()
    g.service.base_port = 8005
    g.service.timeout = 300.0
    
    # FlowEdit 算法参数
    g.flowedit = ml_collections.ConfigDict()
    g.flowedit.prompt = "Move the camera"
    g.flowedit.seed = 0
    g.flowedit.steps = 40
    g.flowedit.guidance_scale = 1.0
    g.flowedit.true_cfg_scale_tgt = 15.0
    g.flowedit.n_min = 0
    g.flowedit.n_max = 25
    
    # Loss 权重（> 0 时自动开启对应的梯度计算）
    g.flowedit.ssim_weight = 1.0        # SSIM loss 权重
    g.flowedit.lpips_weight = 0.0       # LPIPS loss 权重
    g.flowedit.latent_mse_weight = 0.0  # Latent MSE loss 权重

    return cfg
