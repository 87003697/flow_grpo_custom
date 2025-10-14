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
    cfg.num_epochs = 20
    cfg.save_freq = 2
    cfg.eval_freq = 2
    # 未使用：num_checkpoint_limit
    cfg.save_visualizations = True
    cfg.mixed_precision = "bf16"  # 可根据硬件改为 "no"/"fp16"
    # 未使用：allow_tf32
    # 统一 checkpoint：训练/评测均使用该字段（可为 checkpoint_XXXX 或其父目录）
    cfg.checkpoint = ""
    cfg.use_lora = True
    cfg.verbose = False
    cfg.gradient_checkpointing = True
    cfg.deterministic = True  # 控制是否使用 SDE 采样（True->ODE，仅用于调试）

    # LoRA 配置
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32

    # 数据与输入（dataset/resolution 未在训练中使用）
    cfg.data_dir = "dataset/eval3d_hunyuan3d"

    # 预训练权重路径（需指向 Direct3D‑S2 本地解压目录）
    cfg.pretrained = pre = ml_collections.ConfigDict()
    pre.pipeline_path = "./pretrained_weights/direct3d_s2-v-1-1"  # 需包含 config.yaml + model_*.ckpt
    pre.subfolder = "direct3d-s2-v-1-1"  # 若内部再嵌套一层则保持；否则可留空
    pre.minimal_512_only = True  # 仅加载 dense + sparse512

    # 采样参数（dense + sparse512）
    cfg.sample = sm = ml_collections.ConfigDict()
    sm.num_inference_steps_dense = 50
    # 统一使用 num_steps（官方 sparse512 缺省 30）
    sm.num_steps = 30
    # 评估批大小（对齐 TRELLIS：sample.test_batch_size）
    sm.test_batch_size = 1
    # 官方默认 guidance_scale=7.0
    sm.guidance_scale = 7.0
    # 未使用：sample.use_sde（实际从 deterministic 推导 use_sde）
    sm.num_candidates = 2  # 每张图像生成的候选 mesh 数（GRPO group）
    sm.input_batch_size = 1  # 采样输入（图像）批大小
    sm.num_batches_per_epoch = 1
    sm.num_meshes_per_image = sm.num_candidates  # 与其他脚本字段对齐

    # Flow/SDE 采样器参数（对齐 TRELLIS：slat_sampler_params.*）
    cfg.slat_sampler_params = ml_collections.ConfigDict()
    # 与官方一致的解码阈值
    cfg.slat_sampler_params.mc_threshold = 0.2

    # 奖励/优势设置（未使用 kl_reward）
    sm.global_std = True
    sm.adv_type = "winrate"  # 或 similarity

    # 训练超参
    cfg.train = tr = ml_collections.ConfigDict()
    tr.batch_size = sm.num_candidates               # LoRA 小批次
    tr.use_8bit_adam = True
    tr.learning_rate = 2e-5
    tr.adam_beta1 = 0.9
    tr.adam_beta2 = 0.999
    tr.adam_weight_decay = 1e-4
    tr.adam_epsilon = 1e-8
    tr.gradient_accumulation_steps = 4
    tr.max_grad_norm = 1.0
    tr.num_inner_epochs = 1
    # 未使用：train.cfg
    tr.adv_clip_max = 2.0
    tr.clip_range_low = 0.02
    tr.clip_range_high = 1.0
    tr.timestep_fraction = 0.99
    tr.beta = 0.0      # KL loss 系数（与 sm.kl_reward 区分）
    tr.lora_path = None
    # 启用 EMA，评估/推理将自动切换至 EMA 权重
    tr.ema = True
    tr.ema_decay = 0.999
    tr.log_freq = 1

    # Prompt / Reward（沿用 mesh 评估）
    # Prompt/Reward（prompt_fn 与 kwargs 未被训练循环使用）
    cfg.reward_fn = rwd = ml_collections.ConfigDict()
    rwd.uni3d = 0.0
    rwd.camera_normal = 1.0

    cfg.camera_normal = cn = ml_collections.ConfigDict()
    cn.normal_resolution = 518
    cn.cache_dir = "dataset/eval3d_hunyuan3d/normals"
    cn.camera_ckpt = "pretrained_weights/vggt-camera-search/2025.08.20_08.56.06/checkpoints/step_4100/model.safetensors"
    cn.save_vis = False
    cn.source_front = "+z"
    # DINO 相似度类型：cls/dense/match_gird2pixel，可被命令行覆盖
    cn.dino_similarity_type = "cls"

    # 统计
    cfg.per_image_stat_tracking = True

    # eval-only 模式（仅评测，不训练）
    cfg.eval_only = False

    return cfg
