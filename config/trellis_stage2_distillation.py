import ml_collections



def _flowedit_config(g: ml_collections.ConfigDict):
    """FlowEdit 专用配置"""
    g.flowedit = ml_collections.ConfigDict()
    
    # Pipeline 类型: "simple" | "full"
    # - "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
    # - "full": FlowEditPipeline，双分支都使用模型推理（效果更好）
    g.flowedit.pipeline_type = "simple"
    
    g.flowedit.seed = 0
    g.flowedit.steps = 40
    g.flowedit.n_max = 25
    g.flowedit.fixed_noise = True  # 是否在所有 step 使用相同噪声
    
    g.flowedit.true_cfg_scale_tgt = 12
    g.flowedit.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.flowedit.negative_prompt_tgt = " "  # target 分支的 negative prompt
    
    # 多步监督模式: "final" | "mean" | "weighted" | "ada" | "ada_position"
    g.flowedit.latent_mse_mode = "weighted"
    
    # # "full" 模式专用参数（仅当 pipeline_type="full" 时生效）
    # g.flowedit.true_cfg_scale_src = 4.0
    # g.flowedit.source_prompt = g.flowedit.negative_prompt_tgt
    # g.flowedit.negative_prompt_src = " "


def _sds_config(g: ml_collections.ConfigDict):
    """SDS 专用配置"""
    g.sds = ml_collections.ConfigDict()
    
    g.sds.seed = 0
    g.sds.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    g.sds.max_step_percent = 0.98   # 最大时间步百分比（0.98 = t=980）
    g.sds.weight_type = "ada"   # 梯度权重类型: "uniform" | "t" | "ada"
                                    # - "uniform": 不加权（w=1）
                                    # - "t": 按时间步加权（w=t/1000）
                                    # - "ada": 自适应权重（根据预测差异归一化）
    g.sds.weight_eps = 1e-2         # ada 权重的 epsilon（防止除零）
    
    g.sds.true_cfg_scale = 1      # CFG scale（条件-无条件混合强度）
    
    # Prompt 配置
    g.sds.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.sds.negative_prompt = " "


def _csd_config(g: ml_collections.ConfigDict):
    """CSD (Classifier Score Distillation) 专用配置
    
    CSD 与 SDS 的区别：
        - SDS: grad = noise_pred - noise（单次推理）
        - CSD: grad = x0_low - x0_high（两次推理，高低 CFG 差分）
    
    CSD 优势：
        - 更稳定：避免了 SDS 中噪声带来的方差问题
        - 更好的信号：利用 CFG 差分捕捉"增强方向"
    """
    g.csd = ml_collections.ConfigDict()
    
    g.csd.seed = 0
    g.csd.min_step_percent = 0.02   # 最小时间步百分比（0.02 = t=20）
    g.csd.max_step_percent = 0.98   # 最大时间步百分比（0.98 = t=980）
    g.csd.weight_type = "ada"   # 梯度权重类型: "uniform" | "t" | "ada"
                                    # - "uniform": 不加权（w=1）
                                    # - "t": 按时间步加权（w=t/1000）
                                    # - "ada": 自适应权重（根据预测差异归一化）
    g.csd.weight_eps = 1e-2         # ada 权重的 epsilon（防止除零）
    
    g.csd.true_cfg_scale = 1      # CFG scale（条件-无条件混合强度）
                                    # 低 CFG 分支固定为 1.0
    
    # Prompt 配置
    g.csd.target_prompt = "Move the camera. High-definition, ultra-detailed."
    g.csd.negative_prompt = " "


def get_config():
    """TRELLIS Stage 2 蒸馏训练配置（精简版，仅保留 trellis.py 实际使用的字段）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_distill"
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
    cfg.freq.save.visual = 2  # 训练可视化保存步频
    cfg.freq.save.ckpt = 5    # ckpt 保存频率（epoch）
    cfg.freq.save.progress_samples = 4  # FlowEdit 中间步采样数（0=不保存，>0 必须是完全平方数：4, 9, 16...）
    cfg.freq.eval = 5         # 评估频率（epoch）

    # === LoRA 配置（仅当 train.mode = "lora" 时生效）===
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.lora.lora_alpha = 32  # LoRA alpha（通常与 rank 相同）
    cfg.lora.lora_dropout = 0.0  # LoRA dropout
    cfg.lora.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]  # 目标模块

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
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer 配置 ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024  # 渲染分辨率，FlowEdit 要求 1024×1024
    cfg.renderer.type = "gs"  # 可选: mesh / gs
    cfg.renderer.ssaa = 1  # 超采样倍数
    cfg.renderer.bg_color = [1.0, 1.0, 1.0]
    cfg.renderer.near = 0.8  # 近裁剪面
    cfg.renderer.far = 1.6  # 远裁剪面

    # === 训练超参 ===
    cfg.train = tr = ml_collections.ConfigDict()
    
    # 训练模式: "full" | "lora" | "frozen"
    # - "full": 全参微调，教师模型为冻结副本（放在 guidance 设备）
    # - "lora": LoRA 微调，教师模型为禁用 adapter 后的原始权重
    # - "frozen": 冻结模式（仅推理，不训练）
    tr.mode = "full"
    
    tr.gradient_accumulation_steps = 4
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adam"
    tr.optimizer.lr = 3e-5
    tr.optimizer.beta1 = 0.9
    tr.optimizer.beta2 = 0.999
    tr.optimizer.weight_decay = 1e-4
    tr.optimizer.eps = 1e-4

    # === 正则化配置 ===
    # 用于 rollout 蒸馏训练，让学生模型对齐教师模型
    cfg.reg = ml_collections.ConfigDict()
    cfg.reg.type = "kl"  # 正则化类型: "none" | "dmd" | "kl"
    cfg.reg.weight_mode = "uniform"  # 梯度加权模式: "uniform" | "t" | "ada"
    cfg.reg.eps = 1e-2  # ada 权重的 epsilon（防止除零）

    # === Guidance 配置（通用）===
    # Guidance 模型自动放在 训练设备+1 的 GPU 上
    # 例如：训练在 cuda:0 → Guidance 在 cuda:1
    cfg.guidance = g = ml_collections.ConfigDict()
    
    # ★ 切换 Guidance 类型: "flowedit" | "sds" | "csd"
    # - "flowedit": FlowEdit 编辑式蒸馏（多步，生成编辑图像）
    # - "sds": Score Distillation Sampling（单步，梯度注入）
    # - "csd": Classifier Score Distillation（两次推理，高低 CFG 差分）
    g.type = "csd"
    
    # 模型路径（HuggingFace ID 或本地路径）
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    
    # 工作分辨率
    g.edit_resolution = 1024
    
    # 是否使用 autograd 预计算梯度 + SpecifyGradient 注入
    # True: 预计算梯度后释放计算图，显存更低
    # False: 正常 autograd，保留完整计算图
    g.enable_autograd = True

    # 加载对应的专用配置
    _flowedit_config(g)
    _sds_config(g)
    _csd_config(g)

    # === Loss 配置 ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.ssim = 0.0          # SSIM loss 权重
    tr.loss.lpips = 0.0         # LPIPS loss 权重
    tr.loss.latent_mse = 1.0    # Latent MSE loss 权重
    tr.loss.dino = 0.0          # DINO loss 权重
    tr.loss.reg = 1.0           # 正则化 loss 权重（DMD/KL）
    
    return cfg
