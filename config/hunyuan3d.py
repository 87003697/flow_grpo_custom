"""Hunyuan3D GRPO训练配置 - 基于base.py标准化版本"""

import ml_collections


def get_config():
    """Hunyuan3D GRPO训练配置 - 与base.py保持最大相似性"""
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
    config.mixed_precision = "bf16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA.
    config.use_lora = True
    config.dataset = "eval3d"  # Hunyuan3D specific: dataset name
    config.resolution = 768    # Not directly used in 3D but kept for consistency

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "tencent/Hunyuan3D-2.1"  # Hunyuan3D specific model
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 20      # Hunyuan3D specific: 20 steps for 3D generation
    sample.eval_num_steps = 20
    # eta parameter for the sde sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 7.5  # Hunyuan3D specific: higher guidance for 3D
    # batch size (per GPU!) to use for sampling.
    sample.train_batch_size = 1  # Hunyuan3D specific: reduced for memory constraints
    sample.input_batch_size = 1  # Hunyuan3D specific: input batch size for pipeline
    sample.num_image_per_prompt = 2  # Hunyuan3D specific: equivalent to base.py's num_image_per_prompt
    sample.num_meshes_per_image = 2  # Hunyuan3D specific: kept for script compatibility (same as num_image_per_prompt)
    sample.test_batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 2
    sample.kl_reward = 0.02  # Hunyuan3D specific: adjusted for 3D generation
    # Whether use all samples in a batch to compute std
    sample.global_std = False

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1  # Hunyuan3D specific: reduced for memory constraints
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = True  # Hunyuan3D specific: enabled for memory optimization
    # learning rate.
    train.learning_rate = 1e-5  # Hunyuan3D specific: lower LR for 3D models
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
    train.gradient_accumulation_steps = 4  # Hunyuan3D specific: increased for effective batch size
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = False  # Hunyuan3D specific: CFG handled during sampling
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the PPO clip range.
    train.clip_range = 0.001  # Hunyuan3D specific: adjusted for 3D generation
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # kl ratio
    train.beta = 0.01  # Hunyuan3D specific: adjusted for 3D generation
    # pretrained lora path
    train.lora_path = None
    train.ema = False  # Hunyuan3D specific: disabled for memory optimization

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = "image_to_3d"  # Hunyuan3D specific: image-to-3D generation
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = ml_collections.ConfigDict()
    config.reward_fn.geometric_quality = 0.  # Hunyuan3D specific: geometric quality weight
    config.reward_fn.uni3d = 1.0              # Hunyuan3D specific: disabled for memory optimization
    config.save_dir = 'checkpoints/hunyuan3d_grpo'  # Hunyuan3D specific: save directory

    ###### Per-Image Stat Tracking ######
    # configuration for the per-image stat tracker.
    config.per_image_stat_tracking = per_image_stat_tracking = ml_collections.ConfigDict()
    # a rolling buffer of the last `buffer_size` rewards for each image is kept.
    per_image_stat_tracking.buffer_size = 128
    # the minimum number of rewards to collect for an image before using per-image statistics. if the number of
    # rewards for an image is less than `min_count`, the global statistics will be used instead.
    per_image_stat_tracking.min_count = 4

    ###### Hunyuan3D Specific Extensions ######
    # These are additional configurations specific to Hunyuan3D that don't have equivalents in base SD configs
    
    # Memory optimization level: 'conservative', 'moderate', 'aggressive'
    config.memory_optimization_level = 'aggressive'
    
    # Data directory for 3D training data
    config.data_dir = "dataset/eval3d"

    
    # Whether to save mesh visualizations every 10 epochs (disabled by default for performance)
    config.save_visualizations = True
    
    # Attention optimization configurations
    config.attention_optimization = ml_collections.ConfigDict()
    config.attention_optimization.enable_flash_sdp = True
    config.attention_optimization.enable_mem_efficient_sdp = False
    config.attention_optimization.enable_math_sdp = False
    config.attention_optimization.allow_tf32 = False
    
    # FlashVDM optimization configurations
    config.flashvdm = ml_collections.ConfigDict()
    config.flashvdm.enabled = True
    config.flashvdm.adaptive_kv_selection = True
    config.flashvdm.topk_mode = 'mean'
    config.flashvdm.mc_algo = 'mc'
    config.flashvdm.replace_vae = False

    return config 