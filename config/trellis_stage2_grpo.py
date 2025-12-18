import ml_collections


def get_config():
    """TRELLIS Stage 2 GRPO 训练配置（与 Direct3D-S2 对齐的数据与采样流程）。"""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_stage2_grpo"
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.save_freq = 5
    cfg.eval_freq = 5
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.use_lora = True
    cfg.verbose = False
    cfg.gradient_checkpointing = True
    cfg.deterministic = True
    cfg.eval_only = False

    # === LoRA 配置 ===
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # === 数据路径（与 Direct3D 共用 Alphaimages + camera normal 缓存）===
    cfg.train_data_dir = "dataset/alphaimages_1k/train"
    cfg.eval_data_dir = "dataset/alphaimages_1k/test"

    # === 预训练权重（沿用 TRELLIS 官方 checkpoint）===
    cfg.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "./pretrained_weights/TRELLIS-image-large"
    pretrained.revision = "main"
    pretrained.pipeline_path = pretrained.model
    pretrained.subfolder = ""

    # === 采样参数（dense + sparse）===
    cfg.sample = sm = ml_collections.ConfigDict()
    sm.num_inference_steps_dense = 50
    sm.num_steps = 30
    sm.test_batch_size = 8
    sm.guidance_scale = 7.0
    sm.num_candidates = 2
    sm.input_batch_size = 1
    sm.num_batches_per_epoch = 1
    sm.num_meshes_per_image = sm.num_candidates
    sm.top_k = 0
    sm.top_bottom_k = 0
    sm.same_latent = True
    sm.adv_type = "similarity"
    sm.adv_from = "average"

    # Flow / SDE sampler 额外参数
    cfg.slat_sampler_params = ml_collections.ConfigDict()
    cfg.slat_sampler_params.mc_threshold = 0.2
    cfg.slat_sampler_params.noise_level = 0.7

    # === 训练超参（与 Direct3D-S2 匹配）===
    cfg.train = tr = ml_collections.ConfigDict()
    tr.batch_size = sm.num_candidates
    tr.use_8bit_adam = True
    tr.learning_rate = 3e-4
    tr.adam_beta1 = 0.9
    tr.adam_beta2 = 0.999
    tr.adam_weight_decay = 1e-4
    tr.adam_epsilon = 1e-8
    tr.gradient_accumulation_steps = 4
    tr.max_grad_norm = 1.0
    tr.num_inner_epochs = 1
    tr.adv_clip_max = 2.0
    tr.clip_range = 0.02
    tr.timestep_fraction = 0.99
    tr.timestep_keep_ratio = 1.0
    tr.beta = 0.0
    tr.lora_path = None
    tr.ema = False
    tr.ema_decay = 0.999
    tr.log_freq = 1
    tr.detach_uncond = False
    # 兼容脚本使用的 train.optimizer.* 覆写方式
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "lion"
    tr.optimizer.lr = 3e-4
    tr.optimizer.beta1 = tr.adam_beta1
    tr.optimizer.beta2 = tr.adam_beta2
    tr.optimizer.weight_decay = tr.adam_weight_decay
    tr.optimizer.eps = tr.adam_epsilon

    # === 奖励/相机配置（复用 camera-normal scorer）===
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
    cn.encoder = "dino_v3"
    cn.dino_v3_path = "pretrained_weights/dinov3-vith16plus-pretrain-lvd1689m"
    cn.dino_similarity_type = "dense_all"
    cn.dense_match_chunk_size = 4096
    cn.camera_type = "fixed_v1_max"
    cn.vlm_api_source = "1"
    cn.vlm_prompt_version = "v1"
    cn.vlm_max_tokens = 8000
    cn.vlm_enable_thinking = False
    cn.camera_param_dim = 9
    cn.img_size = 518
    cn.cam_batch_size = 64
    cn.render_batch_size = 32
    cn.dino_batch_size = 64
    cn.camera_config_py = "_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py"
    cn.use_mesh_support = True
    cn.vis_dir = "logs/dino_vis"
    cn.avg_camera_per_group = False
    cn.use_RGB_for_comparison = False

    cfg.camera_normal_train = cnt = ml_collections.ConfigDict()
    cnt.normal_resolution = 518
    cnt.cache_dir = "dataset/alphaimages_1k/train/normals"

    cfg.camera_normal_eval = cne = ml_collections.ConfigDict()
    cne.normal_resolution = 518
    cne.cache_dir = "dataset/alphaimages_1k/test/normals"

    return cfg
