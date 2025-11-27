import ml_collections


def get_config():
    """Direct3D‑S2 GRPO 训练配置 (Stage2 sparse512)
    设计参考：`config/trellis_stage2_grpo_normal-sim.py` 与 `DEV.md`
    与 TRELLIS / Hunyuan3D 字段对齐，便于通用训练循环复用。
    仅训练 sparse_dit_512（LoRA），其余模块冻结。
    """
    cfg = ml_collections.ConfigDict()

    # General
    cfg.run_name = "direct3d_s2_grpo"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.save_freq = 5
    cfg.eval_freq = 5
    # 未使用：num_checkpoint_limit
    cfg.save_visualizations = True
    cfg.mixed_precision = "bf16"  # 可根据硬件改为 "no"/"fp16"
    # 未使用：allow_tf32
    # 新增：checkpoint 根目录或具体 checkpoint_*/ 目录，用于 resume
    cfg.checkpoint = ""
    cfg.use_lora = True
    cfg.verbose = False
    cfg.gradient_checkpointing = True
    cfg.deterministic = True  # 控制是否使用 SDE 采样（True->ODE，仅用于调试）
    # 运行模式：是否仅评估（与训练循环中 config.eval_only 对齐）
    cfg.eval_only = False

    # LoRA 配置
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # 数据根目录（严格模式：训练与评估分开配置；目录下需包含 images/ 子目录）
    cfg.train_data_dir = "dataset/alphaimages_1k/train"
    cfg.eval_data_dir = "dataset/alphaimages_1k/test"

    # 预训练权重路径（需指向 Direct3D‑S2 本地解压目录）
    cfg.pretrained = pre = ml_collections.ConfigDict()
    pre.pipeline_path = "./pretrained_weights/direct3d_s2-v-1-1"  # 需包含 config.yaml + model_*.ckpt
    pre.subfolder = "direct3d-s2-v-1-1"  # 若内部再嵌套一层则保持；否则可留空
    pre.minimal_512_only = True  # 仅加载 dense + sparse512
    pre.use_refiner = True  # 是否启用 refiner（需要 model_refiner.ckpt）

    cfg.nft_beta = 1.0  # DiffusionNFT 正负策略混合系数（独立于 KL 系数）

    # 采样参数（dense + sparse512）
    cfg.sample = sm = ml_collections.ConfigDict()
    # 统一使用 num_steps（官方 sparse512 缺省 30）
    sm.num_steps = 30
    # 评估批大小（对齐 TRELLIS：sample.test_batch_size）
    sm.test_batch_size = 8
    # 官方默认 guidance_scale=7.0
    sm.guidance_scale = 7.0
    # 未使用：sample.use_sde（实际从 deterministic 推导 use_sde）
    sm.num_meshes_per_image = 2  # 每张图像生成的候选 mesh 数（GRPO group）
    sm.input_batch_size = 1  # 采样输入（图像）批大小
    sm.num_batches_per_epoch = 1
    # 新增：same latent（按批稳定生成器；K 个候选之间仍为不同噪声但可复现）
    sm.same_latent = True
    # 新增：仅训练奖励极值样本（0 表示关闭）
    sm.top_bottom_k = 0

    # Flow/SDE 采样器参数（对齐 TRELLIS：slat_sampler_params.*）
    cfg.slat_sampler_params = ml_collections.ConfigDict()
    # 与官方一致的解码阈值
    cfg.slat_sampler_params.mc_threshold = 0.2
    # 新增：控制 Flow 步级噪声强度（影响策略采样的随机性）；0.7 与参考实现一致
    cfg.slat_sampler_params.noise_level = 0.7

    # 奖励/优势设置（未使用 kl_reward）
    sm.adv_type = "similarity"  # 可选: "winrate", "winrate_plus"
    # 新增：优势来源（逐子奖励 or 加权总分）
    sm.adv_from = "average"  # 可选: "seperate", "average"|"avg"

    # 训练超参
    cfg.train = tr = ml_collections.ConfigDict()
    tr.batch_size = sm.num_meshes_per_image        # LoRA 小批次
    # 统一的优化器配置
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adam_8bit"  # "adam_8bit" 走 bnb；否则 timm（如 "adamw"/"lion"/"adan"）
    tr.optimizer.lr = 3e-4
    tr.optimizer.beta1 = 0.9
    tr.optimizer.beta2 = 0.999
    tr.optimizer.eps = 1e-6
    tr.optimizer.weight_decay = 1e-4
    tr.gradient_accumulation_steps = 4
    tr.max_grad_norm = 1.0
    tr.num_inner_epochs = 1
    # 未使用：train.cfg
    tr.adv_clip_max = 2.0
    tr.timestep_fraction = 0.99
    tr.timestep_keep_ratio = 1.0
    tr.decay_type = 2  # DiffusionNFT LoRA 老权重融合调度
    tr.beta = 0.0      # KL loss 系数
    tr.lora_path = None
    tr.detach_uncond = False  # 默认不 detach 无条件分支，便于 CLI 覆写
    # 启用 EMA，评估/推理将自动切换至 EMA 权重
    tr.ema = False
    tr.ema_decay = 0.999
    tr.log_freq = 1

    # Prompt / Reward（沿用 mesh 评估）
    # Prompt/Reward（prompt_fn 与 kwargs 未被训练循环使用）
    cfg.reward_fn = rwd = ml_collections.ConfigDict()
    rwd.dummy = 0.0
    rwd.uni3d = 0.0
    rwd.camera_normal = 1.0

    cfg.camera_normal = cn = ml_collections.ConfigDict()
    cn.normal_resolution = 518
    cn.cache_dir = "dataset/alphaimages_1k/normals"
    cn.camera_ckpt = "pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors"
    cn.save_vis = False
    cn.source_front = "+z"
    # 覆盖 reward model 配置：编码器/相似度/性能
    cn.encoder = "dino_v3"
    # 新增：HPSv2 权重路径（当 encoder=hpsv2 时使用）
    cn.hpsv2_ckpt_path = "pretrained_weights/hpsv2/HPS_v2.1_compressed.pt"
    cn.dino_v3_path = "pretrained_weights/dinov3-vith16plus-pretrain-lvd1689m"  # 修改为你的本地路径
    cn.dino_similarity_type = "dense_all"  # 可选: "cls" / "dense" / "dense_all" / "match_gird2pixel" / "match_pixel"
    cn.dense_match_chunk_size = 4096        # 显存吃紧可调小如 8192/4096
    # 相机与渲染/批大小
    cn.camera_param_dim = 9
    cn.img_size = 518
    cn.cam_batch_size = 64
    cn.render_batch_size = 32
    cn.encoding_batch_size = 64
    cn.camera_type = "search"  # 可选: search / fixed_v0 / fixed_v1 / xxx_max
    # 编码器选择与路径（可选： "dino_v2" / "dino_v3" / "pickscore"）
    # - 若选择 "dino_v2" 或 "dino_v3"，需确保对应本地模型目录可用
    # - 若选择 "pickscore"，建议将 use_RGB_for_comparison 设为 True
    cn.encoder = "dino_v3"
    # VLM (Gemini) API 源与 Prompt 版本（可由 CLI 覆写）
    cn.vlm_api_source = "1"
    cn.vlm_prompt_version = "v1"
    cn.vlm_max_tokens = 1000
    cn.vlm_enable_thinking = False
    # 固定视角配置脚本（VGGTObj 参考配置）
    cn.camera_config_py = "_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py"
    cn.use_mesh_support = True
    cn.vis_dir = "logs/dino_vis"
    # 新增：对同一图像组的 K 个候选共享均值相机
    cn.avg_camera_per_group = False
    # 新增：使用 RGB 组进行比较（默认 False，使用法线组）
    cn.use_RGB_for_comparison = False

    # 数据加载专用：训练/评估使用各自的 normals 目录（与严格模式的数据加载断言匹配）
    cfg.camera_normal_train = cnt = ml_collections.ConfigDict()
    cnt.normal_resolution = 518
    cnt.cache_dir = "dataset/alphaimages_1k/train/normals"

    cfg.camera_normal_eval = cne = ml_collections.ConfigDict()
    cne.normal_resolution = 518
    cne.cache_dir = "dataset/alphaimages_1k/test/normals"

    # 统计（direct3d 不再使用按图像 tracking）

    return cfg
