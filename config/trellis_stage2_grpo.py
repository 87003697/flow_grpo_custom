import ml_collections


def get_config():
    """TRELLIS Stage 2 GRPO 训练配置
    - 遵循 `config/hunyuan3d.py` 与 `config/base.py` 的字段命名
    - 仅训练 Stage 2 (`SLatFlowModel`)，Stage 1 冻结
    - 使用 SparseTensor + Flow Matching + SDE + LogProb
    """
    config = ml_collections.ConfigDict()

    # General
    config.run_name = "trellis_stage2_grpo"
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 100
    config.save_freq = 1
    config.eval_freq = 1
    config.num_checkpoint_limit = 20
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True
    config.dataset = "eval3d"
    config.resolution = 768

    # Pretrained / Model Id
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"  # 本地权重路径（建议提前下载）
    pretrained.revision = "main"

    # Sampling (两阶段参数)
    config.sample = sample = ml_collections.ConfigDict()
    # Stage 2 采样步数（Flow Euler）
    sample.num_steps = 20
    sample.eval_num_steps = 20
    # CFG 强度
    sample.guidance_scale = 3.0
    # 批配置（按 GPU）
    sample.train_batch_size = 1
    sample.input_batch_size = 1
    sample.num_image_per_prompt = 2
    sample.num_meshes_per_image = 2
    sample.test_batch_size = 1
    sample.num_batches_per_epoch = 2
    # KL 奖励（与 KL loss 不同；若用 KL loss，参照 train.beta）
    sample.kl_reward = 0.0
    # 是否使用全局 std 计算优势
    sample.global_std = True

    # Training
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = True
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    # 训练期是否使用 CFG（保持与采样一致）
    train.cfg = sample.guidance_scale > 1.0
    train.adv_clip_max = 5.0
    train.clip_range = 0.1
    train.timestep_fraction = 1.0
    # KL loss 比例（与 sample.kl_reward 互补，可设 0 仅用 reward 端）
    train.beta = 0.0
    train.lora_path = None
    train.ema = False

    # Prompt / Reward
    config.prompt_fn = "image_to_3d"
    config.prompt_fn_kwargs = {}
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.uni3d = 1.0
    config.save_dir = "checkpoints/trellis_stage2_grpo"

    # TRELLIS 特有扩展
    config.deterministic = False  # ODE vs SDE 开关
    config.sparse_structure_sampler_params = ml_collections.ConfigDict()
    config.sparse_structure_sampler_params.num_samples = 1
    config.sparse_structure_sampler_params.max_points = 65536

    config.slat_sampler_params = ml_collections.ConfigDict()
    config.slat_sampler_params.sigma_min = 0.002
    config.slat_sampler_params.rescale_t = 1.0

    # 统计
    config.per_image_stat_tracking = True

    # 数据路径
    config.data_dir = "dataset/eval3d"

    return config 