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

    # === 数据配置 ===
    cfg.data = ml_collections.ConfigDict()
    
    # 训练数据配置
    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_1k/train/images"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1                      # 训练时视角数
    cfg.data.train.yaw_range = [180.0, 180.0]      # yaw 采样范围 (度)
    cfg.data.train.pitch_range = [-15.0, 45.0]     # pitch 采样范围 (度)
    cfg.data.train.r_range = [2.0, 2.0]            # 相机距离范围
    cfg.data.train.fov_range = [40.0, 40.0]        # 视场角范围 (度)
    
    # 评估数据配置
    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_1k/test/images"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 4                       # 评估时视角数
    cfg.data.eval.yaw = 180                        # 评估时固定 yaw (度)
    cfg.data.eval.pitch = 0.0                      # 评估时固定 pitch (度)
    cfg.data.eval.r = 2.0                          # 评估时相机距离
    cfg.data.eval.fov = 40.0                       # 评估时视场角 (度)

    # === 预训练权重 ===
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置 ===
    cfg.renderer = renderer = ml_collections.ConfigDict()
    renderer.resolution = 1024  # 渲染分辨率，FlowEdit 要求 1024×1024
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
    
    # Loss 权重配置
    tr.loss = ml_collections.ConfigDict()
    tr.loss.ssim = 0.0          # SSIM loss 权重
    tr.loss.lpips = 1.0         # LPIPS loss 权重
    tr.loss.latent_mse = 0.0    # Latent MSE loss 权重
    tr.loss.reg = 1.0           # VSD/KL 正则化 loss 权重

    # === VSD/KL 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    cfg.reg = reg = ml_collections.ConfigDict()
    reg.type = "vsd"  # 正则化类型: "none" | "vsd" | "kl"
                      # - "none": 不使用正则化
                      # - "vsd": 使用 SpecifyGradient 将梯度穿透 rollout
                      # - "kl": 使用 MSE loss 带时间步方差加权
    reg.weight_mode = "uniform"  # 梯度加权模式: "uniform" | "t" | "ada"
                                 # - "uniform": 不加权
                                 # - "t": 按时间步 t 加权
                                 # - "ada": 自适应加权（按参考值归一化）

    # === Guidance 配置 ===
    # FlowEdit 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → FlowEdit 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # FlowEdit 工作分辨率
    g.edit_resolution = 1024
    
    # FlowEdit 算法参数
    g.flowedit = ml_collections.ConfigDict()
    g.flowedit.prompt = "Move the camera"
    g.flowedit.seed = 0
    g.flowedit.steps = 40
    g.flowedit.guidance_scale = 1.0
    g.flowedit.true_cfg_scale_tgt = 15.0
    g.flowedit.n_min = 0
    g.flowedit.n_max = 25

    return cfg
