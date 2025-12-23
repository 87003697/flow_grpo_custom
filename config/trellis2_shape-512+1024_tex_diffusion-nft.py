import ml_collections as mc


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
    cfg.sample.top_k = 0
    cfg.sample.top_bottom_k = 0
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
    cfg.train.timestep_keep_ratio = 1.0
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

    # 训练轮数（顶层字段，供脚本解析）
    cfg.num_epochs = 10

    return cfg

