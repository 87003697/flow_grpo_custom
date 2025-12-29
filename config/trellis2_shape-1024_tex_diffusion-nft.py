import ml_collections as mc


# =============================================================================
# 分辨率配置表（参考 TRELLIS.2 pipeline_type 设计）
# =============================================================================
# 
# pipeline_type 决定了整个生成流程的分辨率配置：
# - ss_resolution: Stage1 稀疏结构采样分辨率
# - lr_resolution: Cascade 低分辨率阶段
# - hr_resolution: Cascade 高分辨率阶段  
# - max_num_tokens: Cascade 时最大 token 数量
# - render_resolution: 法线渲染分辨率
#
# 支持的 pipeline_type:
# - '512': 纯 512 分辨率（不使用 cascade）
# - '1024': 纯 1024 分辨率（不使用 cascade）
# - '1024_cascade': 512→1024 级联（推荐）
# - '1536_cascade': 512→1536 级联（高质量）
# =============================================================================

RESOLUTION_CONFIGS = {
    '1024_cascade': {
        'ss_resolution': 32,       # Stage1 稀疏结构分辨率
        'lr_resolution': 512,      # 低分辨率（cascade 第一阶段）
        'hr_resolution': 1024,     # 高分辨率（cascade 第二阶段）
        'max_num_tokens': 49152,   # 最大 token 数量
        'render_resolution': 512,  # 法线渲染分辨率
    },
    '1536_cascade': {
        'ss_resolution': 32,
        'lr_resolution': 512,
        'hr_resolution': 1536,
        'max_num_tokens': 49152,
        'render_resolution': 512,
    },
}


def get_resolution_config(pipeline_type: str) -> dict:
    """根据 pipeline_type 获取分辨率配置。
    
    Args:
        pipeline_type: 管道类型，如 '1024_cascade'
        
    Returns:
        分辨率配置字典
        
    Raises:
        ValueError: 如果 pipeline_type 无效
    """
    if pipeline_type not in RESOLUTION_CONFIGS:
        valid_types = list(RESOLUTION_CONFIGS.keys())
        raise ValueError(f"Invalid pipeline_type: {pipeline_type}. Valid options: {valid_types}")
    return RESOLUTION_CONFIGS[pipeline_type]


def get_config():
    cfg = mc.ConfigDict()

    # 运行与日志
    cfg.seed = 42
    cfg.logdir = "logs/trellis2_stage2"
    cfg.run_name = "trellis2_stage2_grpo"
    cfg.mixed_precision = "bf16"
    cfg.deterministic = True
    cfg.verbose = True
    cfg.eval_only = False

    # 数据
    cfg.train_data_dir = "dataset/alphaimages_1k/train"
    cfg.eval_data_dir = "dataset/alphaimages_1k/test"

    # CameraNormal
    cfg.camera_normal = mc.ConfigDict()
    cfg.camera_normal_train = mc.ConfigDict(
        dict(
            cache_dir="dataset/alphaimages_1k/train/normals",
            normal_resolution=518,
        )
    )
    cfg.camera_normal_eval = mc.ConfigDict(
        dict(
            cache_dir="dataset/alphaimages_1k/test/normals",
            normal_resolution=518,
        )
    )
    cfg.camera_normal.encoder = "clip"
    cfg.camera_normal.clip_model_id = "openai/clip-vit-large-patch14"
    cfg.camera_normal.clip_processor_id = "openai/clip-vit-large-patch14"
    cfg.camera_normal.vlm_api_source = "1"
    cfg.camera_normal.vlm_prompt_version = "v1"
    cfg.camera_normal.vlm_max_tokens = 8000
    cfg.camera_normal.vlm_enable_thinking = True
    cfg.camera_normal.dino_similarity_type = "dense_all"
    cfg.camera_normal.avg_camera_per_group = False
    cfg.camera_normal.use_RGB_for_comparison = False
    cfg.camera_normal.camera_type = "fixed_v1_max"
    cfg.camera_normal.source_front = "+z"

    # 奖励
    cfg.reward_fn = mc.ConfigDict()
    cfg.reward_fn.camera_normal = 1.0
    cfg.reward_fn.uni3d = 0.0
    cfg.reward_fn.dummy = 0.0

    # 预训练（内部已包含模型/采样器/归一化/steps/CFG）
    cfg.pretrained = mc.ConfigDict()
    cfg.pretrained.pipeline_path = "pretrained_weights/TRELLIS.2-4B"
    cfg.pretrained.subfolder = ""

    # 采样（步数/CFG 由 pipeline.stage2_params 覆盖）
    cfg.sample = mc.ConfigDict()
    cfg.sample.input_batch_size = 1
    cfg.sample.test_batch_size = 1
    cfg.sample.num_steps = 30
    cfg.sample.guidance_scale = 7.0
    cfg.sample.num_meshes_per_image = 2
    cfg.sample.num_batches_per_epoch = 1
    cfg.sample.adv_type = "similarity"
    cfg.sample.adv_from = "average"
    cfg.sample.same_latent = True

    # Slat sampler（rescale_t 会被 pipeline 覆盖）
    cfg.slat_sampler_params = mc.ConfigDict()
    cfg.slat_sampler_params.mc_threshold = 0.95
    cfg.slat_sampler_params.rescale_t = 1.0
    cfg.slat_sampler_params.noise_level = 0.7

    # 训练
    cfg.train = mc.ConfigDict()
    cfg.train.batch_size = 2
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.num_inner_epochs = 1
    cfg.train.timestep_fraction = 1.0
    cfg.train.adv_clip_max = 2.0
    cfg.train.beta = 0.0
    cfg.train.weight_cross_mode = 1.0
    cfg.train.max_grad_norm = 1.0
    cfg.train.ema = False
    cfg.train.ema_decay = 0.9999
    cfg.train.log_freq = 1
    cfg.train.save_freq = 1
    cfg.eval_freq = 1
    cfg.save_visualizations = True
    cfg.save_freq = 1

    # 优化器
    cfg.train.optimizer = mc.ConfigDict()
    cfg.train.optimizer.type = "lion"
    cfg.train.optimizer.lr = 5e-5
    cfg.train.optimizer.beta1 = 0.9
    cfg.train.optimizer.beta2 = 0.99
    cfg.train.optimizer.eps = 1e-8
    cfg.train.optimizer.weight_decay = 0.0

    # LoRA（必须启用）
    cfg.use_lora = True
    cfg.lora = mc.ConfigDict()
    cfg.lora.lora_rank = 64
    cfg.train.lora_path = ""

    # DiffusionNFT
    cfg.nft_beta = 0.5

    # ==========================================================================
    # 分辨率配置（通过 pipeline_type 自动设置）
    # ==========================================================================
    # pipeline_type 选项：
    # - '512': 纯 512 分辨率（快速，低质量）
    # - '1024': 纯 1024 分辨率（中等速度，中等质量）
    # - '1024_cascade': 512→1024 级联（推荐，平衡速度和质量）
    # - '1536_cascade': 512→1536 级联（慢速，高质量）
    cfg.pipeline_type = '1024_cascade'
    
    # 从配置表获取分辨率参数
    _res_cfg = get_resolution_config(cfg.pipeline_type)
    cfg.resolution = mc.ConfigDict()
    cfg.resolution.ss = _res_cfg['ss_resolution']           # Stage1 稀疏结构分辨率
    cfg.resolution.lr = _res_cfg['lr_resolution']           # Cascade 低分辨率
    cfg.resolution.hr = _res_cfg['hr_resolution']           # Cascade 高分辨率
    cfg.resolution.max_tokens = _res_cfg['max_num_tokens']  # 最大 token 数量
    cfg.resolution.render = _res_cfg['render_resolution']   # 法线渲染分辨率

    # 训练轮数（顶层字段，供脚本解析）
    cfg.num_epochs = 10

    # 检查点恢复
    cfg.checkpoint = ""  # 恢复训练时填入检查点路径

    # 梯度检查点（节省显存）
    cfg.gradient_checkpointing = False

    return cfg
