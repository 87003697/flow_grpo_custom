"""TRELLIS.2 蒸馏训练基础配置（按模块拆分）。"""
import ml_collections


def get_base_config_general():
    """通用配置（seed, epochs, 频率等）。"""
    cfg = ml_collections.ConfigDict()
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.mixed_precision = "bf16"
    cfg.checkpoint = ""
    cfg.eval_only = False
    cfg.verbose = False
    cfg.pipeline_type = "1024"
    
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 2
    cfg.freq.save.ckpt = 5
    cfg.freq.eval = 5
    
    cfg.lora = ml_collections.ConfigDict()
    cfg.lora.lora_rank = 32
    return cfg


def get_base_config_data():
    """数据配置（训练/评估）。"""
    cfg = ml_collections.ConfigDict()
    
    cfg.train = ml_collections.ConfigDict()
    cfg.train.dir = "dataset/alphaimages_1k/train/images"
    cfg.train.batch_size = 1
    cfg.train.n_view = 1
    cfg.train.yaw_range = [180.0, 180.0]
    cfg.train.pitch_range = [-15.0, 45.0]
    cfg.train.r_range = [2.0, 2.0]
    cfg.train.fov_range = [40.0, 40.0]
    
    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.dir = "dataset/alphaimages_1k/test/images"
    cfg.eval.batch_size = 1
    cfg.eval.n_view = 4
    cfg.eval.yaw = 180
    cfg.eval.pitch = 0.0
    cfg.eval.r = 2.0
    cfg.eval.fov = 40.0
    return cfg


def get_base_config_pretrained():
    """预训练权重路径。"""
    cfg = ml_collections.ConfigDict()
    cfg.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


def get_base_config_renderer():
    """渲染器配置。"""
    cfg = ml_collections.ConfigDict()
    cfg.resolution = 1024
    cfg.type = "mesh"
    cfg.ssaa = 1
    cfg.bg_color = [1.0, 1.0, 1.0]
    cfg.near = 1.0
    cfg.far = 100.0
    return cfg


def get_base_config_train():
    """训练超参（optimizer, loss）。"""
    cfg = ml_collections.ConfigDict()
    cfg.gradient_accumulation_steps = 1
    
    cfg.optimizer = ml_collections.ConfigDict()
    cfg.optimizer.type = "adam"
    cfg.optimizer.lr = 3e-5
    cfg.optimizer.beta1 = 0.9
    cfg.optimizer.beta2 = 0.999
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.eps = 1e-4
    
    cfg.loss = ml_collections.ConfigDict()
    cfg.loss.ssim = 1.0
    cfg.loss.lpips = 0.0
    cfg.loss.latent_mse = 1.0
    cfg.loss.dino = 0.0
    cfg.loss.reg = 1.0
    return cfg


def get_base_config_reg():
    """正则化配置。"""
    cfg = ml_collections.ConfigDict()
    cfg.type = "kl"
    cfg.weight_mode = "ada"
    return cfg


def get_base_config_guidance():
    """Guidance 配置（FlowEdit）。"""
    cfg = ml_collections.ConfigDict()
    cfg.model_path = "Qwen/Qwen-Image-Edit-2511"
    cfg.edit_resolution = 1024
    
    cfg.flowedit = ml_collections.ConfigDict()
    cfg.flowedit.pipeline_type = "full"
    cfg.flowedit.seed = 0
    cfg.flowedit.guidance_scale = 1.0
    cfg.flowedit.n_max = 10
    cfg.flowedit.cfg_normalization = True
    cfg.flowedit.steps = 20
    cfg.flowedit.true_cfg_scale_tgt = 8.0
    cfg.flowedit.prompt = "Move the camera"
    cfg.flowedit.target_prompt_image_indices = [1]
    cfg.flowedit.true_cfg_scale_src = 4.0
    cfg.flowedit.source_prompt = "Reconstruct the image"
    cfg.flowedit.source_prompt_image_indices = [1]
    return cfg

