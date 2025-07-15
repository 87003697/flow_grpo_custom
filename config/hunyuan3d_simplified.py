"""简化版Hunyuan3D GRPO训练配置"""

import ml_collections


def get_config():
    """简化版Hunyuan3D GRPO训练配置"""
    config = ml_collections.ConfigDict()

    ###### 基础配置 ######
    config.run_name = "hunyuan3d_grpo_simplified"
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 100
    config.save_freq = 20
    config.eval_freq = 100
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True  # 推荐使用LoRA进行微调
    config.data_dir = "dataset/eval3d"
    config.save_dir = "checkpoints/hunyuan3d_grpo_simplified"
    config.deterministic = True  # 使用确定性模式

    ###### 预训练模型 ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "tencent/Hunyuan3D-2.1"
    pretrained.revision = "main"

    ###### 采样配置 ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 20
    sample.eval_num_steps = 20
    sample.guidance_scale = 5.0
    # 🚀 简化版配置 - 内存友好
    sample.input_batch_size = 1       # 每次处理1张图像
    sample.test_batch_size = 1
    sample.num_batches_per_epoch = 2  # 每个epoch采样2个批次
    sample.kl_reward = 0.1
    sample.global_std = 0.5

    ###### 训练配置 ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1              # 训练batch size
    train.use_8bit_adam = False
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 2
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = False
    train.adv_clip_max = 5.0
    train.clip_range = 0.2
    train.timestep_fraction = 1.0
    train.beta = 0.01
    train.lora_path = None
    train.ema = True
    train.ema_decay = 0.999

    ###### 统计跟踪 ######
    config.per_image_stat_tracking = False  # 🚀 简化：默认关闭
    config.stat_tracking = stat_tracking = ml_collections.ConfigDict()
    stat_tracking.min_count = 16

    ###### 奖励函数 ######
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.geometric_quality = 0.3
    config.reward_fn.uni3d = 0.7

    return config 