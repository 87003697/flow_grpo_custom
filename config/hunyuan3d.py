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
    config.allow_tf32 = False
    config.resume_from = ""
    config.use_lora = True  # 推荐使用LoRA进行微调
    config.data_dir = "dataset/eval3d"
    config.save_dir = "checkpoints/hunyuan3d_grpo"
    config.deterministic = True  # 使用确定性模式

    ###### Attention优化配置 ######
    config.attention_optimization = ml_collections.ConfigDict()
    config.attention_optimization.enable_flash_sdp = True        # 启用Flash Attention
    config.attention_optimization.enable_mem_efficient_sdp = False  # 启用Memory Efficient Attention  
    config.attention_optimization.enable_math_sdp = False       # 禁用数学SDPA以优先使用Flash/Memory Efficient
    config.attention_optimization.allow_tf32 = False            # 允许TF32加速

    ###### 预训练模型 ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "tencent/Hunyuan3D-2.1"
    pretrained.revision = "main"

    ###### 采样配置 ######
    config.sample = ml_collections.ConfigDict()
    config.sample.train_batch_size = 1           # 🔧 训练时的batch size
    config.sample.input_batch_size = 1           # 🔧 输入时的batch size  
    config.sample.num_batches_per_epoch = 2      # 🔧 SD3对齐：每个epoch的批次数量
    config.sample.num_steps = 20                 # 扩散步数
    config.sample.guidance_scale = 7.5           # CFG引导尺度
    config.sample.kl_reward = 0.02               # KL奖励系数
    config.sample.num_meshes_per_image = 2       # 🔧 多候选：每个图像生成的mesh数量

    # ✨ 新增：与SD3统计跟踪保持一致
    config.sample.global_std = False             # 🔧 SD3对齐：是否使用全局标准差
    
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
    config.train.timestep_fraction = 1.0      # 🔧 新增：训练时间步比例（类似SD3）
    config.train.ema = False                  # 🚀 内存优化：暂时关闭EMA
    config.train.ema_decay = 0.99
    config.train.use_8bit_adam = True         # 🚀 内存优化：启用8bit Adam
    
    # ✨ 新增：SD3风格的训练控制参数
    config.train.shuffle_timesteps = False    # 🔧 SD3对齐：默认不随机化时间步顺序（与SD3一致）
    config.train.cfg = False                  # 🔧 SD3对齐：是否使用CFG训练
    
    # ✨ 新增：SD3风格的Adam参数设置
    config.train.adam_beta1 = 0.9             # SD3默认值
    config.train.adam_beta2 = 0.999           # SD3默认值  
    config.train.adam_weight_decay = 1e-4     # SD3默认值
    config.train.adam_epsilon = 1e-8          # SD3默认值
    
    ###### ✨ 新增：SD3风格内存管理策略 ######
    # 内存优化级别：'aggressive'(激进), 'moderate'(中等), 'conservative'(保守)
    config.memory_optimization_level = 'aggressive'
    
    # 激进模式：VAE移动到CPU，节省8-12GB显存，适合显存不足的情况
    # 中等模式：VAE保持GPU但使用混合精度，平衡性能和内存
    # 保守模式：SD3默认策略，VAE保持GPU FP32，性能最佳但内存占用最高

    ###### ✨ 新增：FlashVDM优化配置 ######
    config.flashvdm = ml_collections.ConfigDict()
    config.flashvdm.enabled = False                   # 是否启用FlashVDM
    config.flashvdm.adaptive_kv_selection = True     # 自适应K-V选择
    config.flashvdm.topk_mode = 'merge'               # 'mean' 或 'merge'
    config.flashvdm.mc_algo = 'mc'                   # 'mc' 或 'dmc' (dual marching cubes)
    config.flashvdm.replace_vae = False               # 使用turbo VAE版本

    ###### 统计跟踪 ######
    config.per_image_stat_tracking = True   # ✨ 启用：与SD3的PerPromptStatTracker保持一致

    ###### 奖励函数 ######
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.geometric_quality = 1.0  # 🚀 显存优化：只使用几何质量，禁用Uni3D
    config.reward_fn.uni3d = 0.0              # 🚀 显存优化：禁用Uni3D以节省大量显存

    return config 