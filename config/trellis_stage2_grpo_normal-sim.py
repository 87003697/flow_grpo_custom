import ml_collections


def get_config():
    """TRELLIS Stage 2 GRPO 训练配置
    - 遵循 `config/hunyuan3d.py` 与 `config/base.py` 的字段命名
    - 仅训练 Stage 2 (`SLatFlowModel`)，Stage 1 冻结
    - 使用 SparseTensor + Flow Matching + SDE + LogProb
    """
    cfg = ml_collections.ConfigDict()

    # General
    cfg.run_name = "trellis_stage2_grpo"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 100
    cfg.save_freq = 2
    cfg.eval_freq = 2
    cfg.num_checkpoint_limit = 999
    cfg.save_visualizations = True
    cfg.mixed_precision = "bf16"  # 可根据硬件改为 "no"/"fp16"
    cfg.allow_tf32 = True
    cfg.resume_from = ""
    cfg.use_lora = True
    cfg.verbose = False
    # 梯度检查点（减少显存占用，增加计算时间）
    cfg.gradient_checkpointing = True
    cfg.deterministic = True  # 控制是否使用 SDE 采样（True->ODE，仅用于调试）

    # LoRA 配置
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    cfg.dataset = "eval3d"
    cfg.resolution = 256

    # Pretrained / Model Id
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"  # 本地权重路径（建议提前下载）
    pretrained.revision = "main"

    # Sampling (两阶段参数)
    cfg.sample = sample = ml_collections.ConfigDict()
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
    sample.num_batches_per_epoch = 2  # 对齐 Hunyuan3D 默认值
    # KL 奖励（与 KL loss 不同；若用 KL loss，参照 train.beta）
    sample.kl_reward = 0.0
    sample.adv_type = "winrate" # "similarity"

    # Training
    cfg.train = tr = ml_collections.ConfigDict()
    tr.batch_size = 1
    # 统一优化器配置
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = 'adam_8bit'
    tr.optimizer.lr = 2e-5
    tr.optimizer.beta1 = 0.9
    tr.optimizer.beta2 = 0.999
    tr.optimizer.eps = 1e-6
    tr.optimizer.weight_decay = 1e-4
    tr.gradient_accumulation_steps = 8  # 增大梯度累积以补偿小批量，保持有效批量大小
    tr.max_grad_norm = 1.0
    tr.num_inner_epochs = 1
    # 训练期是否使用 CFG（保持与采样一致）
    tr.cfg = sample.guidance_scale > 1.0
    tr.adv_clip_max = 2.0
    # 统一为对称 PPO/GRPO 裁剪区间
    tr.clip_range = 0.02
    tr.timestep_fraction = 0.99
    # KL loss 比例（与 sample.kl_reward 互补，可设 0 仅用 reward 端）
    tr.beta = 0.0
    tr.lora_path = None
    tr.ema = False
    # 训练日志频率（按 epoch 记录）
    tr.log_freq = 1

    # Prompt / Reward
    cfg.prompt_fn = "image_to_3d"
    cfg.prompt_fn_kwargs = {}
    cfg.reward_fn = ml_collections.ConfigDict()
    
    # 奖励权重：法线相似度与 Uni3D 按 0.5/0.5 加权
    cfg.reward_fn.uni3d = 0.
    cfg.reward_fn.camera_normal = 1.

    # camera_normal 配置（精简：仅保留必需项）
    cfg.camera_normal = ml_collections.ConfigDict()
    cfg.camera_normal.normal_resolution = 518
    cfg.camera_normal.cache_dir = "dataset/eval3d_hunyuan3d/normals"
    cfg.camera_normal.camera_ckpt = "pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors"  # 目录或 .safetensors
    cfg.camera_normal.save_vis = False
    # Mesh 坐标系对齐（可选项："none"/"zup_to_yup"/"euler_deg"），若为 euler_deg 则提供角度与顺序
    # 指定源 mesh 的前向（与 kiui front_dir 语义一致），仅此一项控制朝向对齐
    cfg.camera_normal.source_front = "-y" # TRELLIS 通常生成的 mesh 朝向是-y
    # 新增：对同一图像组的 K 个候选共享均值相机
    cfg.camera_normal.avg_camera_per_group = True

    # TRELLIS 官方采样器参数
    cfg.sparse_structure_sampler_params = ml_collections.ConfigDict()
    cfg.sparse_structure_sampler_params.num_samples = 1  # 官方参数

    cfg.slat_sampler_params = ml_collections.ConfigDict()
    cfg.slat_sampler_params.sigma_min = 0.002  # 官方参数：FlowEulerSampler
    cfg.slat_sampler_params.rescale_t = 1.0    # 官方参数：FlowEulerSampler

    # 统计（trellis 不使用跨 rank 统计/历史池）

    # 数据路径（严格：拆分训练/评估根目录，目录下需含 images/）
    cfg.train_data_dir = "dataset/eval3d_hunyuan3d"
    cfg.eval_data_dir = "dataset/eval3d_hunyuan3d"

    return cfg 
