"""Hunyuan3D GRPO训练配置 - 优化版本（原simplified版本）"""

import ml_collections


def get_config():
    """Hunyuan3D GRPO训练配置 - 优化版本"""
    config = ml_collections.ConfigDict()

    ###### 基础配置 ######
    config.run_name = "hunyuan3d_grpo"
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 100
    config.save_freq = 20
    config.eval_freq = 100
    config.num_checkpoint_limit = 5
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True  # 推荐使用LoRA进行微调
    config.data_dir = "dataset/eval3d"
    config.save_dir = "checkpoints/hunyuan3d_grpo"
    config.deterministic = True  # 使用确定性模式

    ###### Attention优化配置 ######
    config.attention_optimization = ml_collections.ConfigDict()
    config.attention_optimization.enable_flash_sdp = True        # 启用Flash Attention
    config.attention_optimization.enable_mem_efficient_sdp = True  # 启用Memory Efficient Attention  
    config.attention_optimization.enable_math_sdp = False       # 禁用数学SDPA以优先使用Flash/Memory Efficient
    config.attention_optimization.allow_tf32 = True            # 允许TF32加速

    ###### 预训练模型 ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "tencent/Hunyuan3D-2.1"
    pretrained.revision = "main"

    ###### GRPO采样配置 ######
    config.sample = ml_collections.ConfigDict()
    config.sample.input_batch_size = 1       # 🔧 OOM修复：每次处理1张图像避免显存溢出
    config.sample.num_batches_per_epoch = 2  # 🔧 进一步减少：从4→2批次
    config.sample.num_meshes_per_image = 2   # 🔧 关键修复：从4→2个候选mesh（仍满足GRPO>1要求）
    config.sample.num_steps = 20
    config.sample.guidance_scale = 2.0
    config.sample.kl_reward = 0.02
    
    ###### 训练配置 ######
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 1               # 🔧 OOM修复：保持1避免显存溢出
    config.train.gradient_accumulation_steps = 4  # 🔧 增加：累积4步获得effective batch size=4
    config.train.learning_rate = 1e-5
    config.train.num_inner_epochs = 1
    config.train.clip_range = 0.1
    config.train.adv_clip_max = 5.0
    config.train.beta = 0.01                  # 🔧 修复：添加缺失的beta参数
    config.train.max_grad_norm = 1.0          # 🔧 添加：梯度裁剪最大值
    config.train.ema = False                  # 🚀 内存优化：暂时关闭EMA
    config.train.ema_decay = 0.99
    config.train.use_8bit_adam = True         # 🚀 内存优化：启用8bit Adam

    ###### 统计跟踪 ######
    config.per_image_stat_tracking = False  # 🚀 简化：默认关闭
    config.stat_tracking = stat_tracking = ml_collections.ConfigDict()
    stat_tracking.min_count = 16

    ###### 奖励函数 ######
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.geometric_quality = 1.0  # 🚀 显存优化：只使用几何质量，禁用Uni3D
    config.reward_fn.uni3d = 0.0              # 🚀 显存优化：禁用Uni3D以节省大量显存

    return config 