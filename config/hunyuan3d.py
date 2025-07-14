"""Configuration for Hunyuan3D GRPO training."""

import ml_collections


def get_config():
    """Configuration for Hunyuan3D GRPO training."""
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100
    # number of epochs between saving model checkpoints.
    config.save_freq = 20
    config.eval_freq = 100
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 5
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA.
    config.use_lora = False
    # dataset directory for 3D training data
    config.data_dir = "dataset/eval3d"  # 🔧 更新：使用真实的 eval3d 数据集
    # save directory for checkpoints
    config.save_dir = "checkpoints/hunyuan3d_grpo"
    # whether to use deterministic mode (ODE) instead of stochastic mode (SDE)
    config.deterministic = True # TODO: 修改为False

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. for Hunyuan3D, this is the model identifier
    pretrained.model = "tencent/Hunyuan3D-2.1"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 20  # �� 恢复正常步数
    sample.eval_num_steps = 20  # 🔧 恢复正常步数
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size configuration for 3D generation - 🔧 减少内存占用
    sample.input_batch_size = 1           # 🔧 减少：每次只处理1张图像
    sample.num_meshes_per_image = 2       # 🔧 修改：每张图像生成2个mesh候选
    sample.test_batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # total_meshes * num_gpus`.
    sample.num_batches_per_epoch = 2
    sample.kl_reward = 0.1
    # Whether use all samples in a batch to compute std
    sample.global_std = 0.5

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training - 🔧 减少内存占用
    train.batch_size = 1  # 🔧 减少：训练batch size改为1
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 1e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 2
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = False  # disabled for Hunyuan3D as CFG is handled during sampling
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5.0
    # the PPO clip range.
    train.clip_range = 0.2
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # kl ratio
    train.beta = 0.01
    # pretrained lora path
    train.lora_path = None
    train.ema = True
    train.ema_decay = 0.999

    ###### Prompt Function ######
    # prompt function to use. for Hunyuan3D, this is handled by the dataset class
    config.prompt_fn = "image_to_3d"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Mesh Extraction ######
    config.mesh = mesh = ml_collections.ConfigDict()
    # octree resolution for mesh extraction - controls grid density
    mesh.octree_resolution = 128
    # marching cubes level (iso-value) - controls surface extraction threshold
    mesh.mc_level = 0.0  # common values: 0.0 or -1/512
    # marching cubes algorithm - 'mc' for traditional, 'dmc' for differentiable
    mesh.mc_algo = None  # options: None, 'mc', 'dmc'
    # bounding box volume for mesh extraction
    mesh.box_v = 1.01
    # number of chunks for mesh processing
    mesh.num_chunks = 15000

    ###### Reward Function ######
    # reward function to use. for Hunyuan3D, this uses 3D-specific reward functions
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.geometric_quality = 0.3
    config.reward_fn.uni3d = 0.7

    ###### Evaluation and Metrics ######
    # evaluation setup
    config.eval_freq = 10  # evaluate every N epochs
    config.save_freq = 10  # save every N epochs
    
    # per-image stat tracking - 🔧 修改：image-to-3D 场景使用图像统计跟踪
    config.per_image_stat_tracking = True
    
    return config 