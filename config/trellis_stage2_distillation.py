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
    renderer.ssaa = 1  # 超采样倍数
    renderer.bg_color = [1.0, 1.0, 1.0]
    renderer.near = 0.8  # 近裁剪面
    renderer.far = 1.6  # 远裁剪面

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adam"
    tr.optimizer.lr = 3e-5
    tr.optimizer.beta1 = 0.9
    tr.optimizer.beta2 = 0.999
    tr.optimizer.weight_decay = 1e-4
    tr.optimizer.eps = 1e-4
    
    # Loss 权重配置
    tr.loss = ml_collections.ConfigDict()
    tr.loss.ssim = 0.0          # SSIM loss 权重
    tr.loss.lpips = 0.0         # LPIPS loss 权重
    tr.loss.latent_mse = 1.0    # Latent MSE loss 权重
    tr.loss.dino = 0.0          # DINO loss 权重
    tr.loss.reg = 1.0           # 正则化 loss 权重（DMD/KL）

    # === 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    cfg.reg = reg = ml_collections.ConfigDict()
    reg.type = "kl"  # 正则化类型: "none" | "dmd" | "kl"
                      # - "none": 不使用正则化
                      # - "dmd": DMD 风格（推荐），grad 在 no_grad 中计算，通过伪 loss 注入（符合 Self-Forcing 原理）
                      # - "kl": KL 风格，直接可导的 MSE loss
    reg.weight_mode = "uniform"  # 梯度加权模式: "uniform" | "t" | "ada"
                                 # - "uniform": 不加权
                                 # - "t": 按时间步 t 加权
                                 # - "ada": 自适应归一化（DMD paper eq.8）

    # === Guidance 配置 ===
    # FlowEdit 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → FlowEdit 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # FlowEdit 模型路径（HuggingFace ID 或本地路径）
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    
    # FlowEdit 工作分辨率
    g.edit_resolution = 1024
    
    # FlowEdit 算法参数
    g.flowedit = ml_collections.ConfigDict()
    g.flowedit.prompt = "Generate a novel view"
    g.flowedit.seed = 0
    g.flowedit.steps = 40
    g.flowedit.guidance_scale = 1.0
    g.flowedit.true_cfg_scale_tgt = 4.0
    g.flowedit.n_min = 0
    g.flowedit.n_max = 15
    g.flowedit.noise_mode = "fixed"  # 噪声模式: "random" | "fixed" | "velocity" | "velocity_fixed"

    return cfg
