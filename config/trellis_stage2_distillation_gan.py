"""TRELLIS Stage 2 全轨迹蒸馏 + GAN 配置。"""
import ml_collections
from config.trellis_stage2_distillation import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "trellis_stage2_distillation_gan"
    cfg.guidance.type = "flowedit_gan"
    cfg.guidance.flowedit_gan = cfg.guidance.flowedit

    cfg.train.guidance.loss.gan = 0.01
    cfg.rollout.reg.type = "v"
    cfg.train.guidance.loss.gan_opt = ml_collections.ConfigDict()
    cfg.train.guidance.loss.gan_opt.type = "adam"
    cfg.train.guidance.loss.gan_opt.lr = 2e-5
    cfg.train.guidance.loss.gan_opt.beta1 = 0.0
    cfg.train.guidance.loss.gan_opt.beta2 = 0.99
    cfg.train.guidance.loss.gan_opt.eps = 1e-8
    cfg.train.guidance.loss.gan_opt.weight_decay = 0.0
    cfg.train.guidance.loss.gan_r1_gamma = 0.0
    cfg.train.guidance.loss.gan_model_path = (
        "pretrained_weights/dinov3-vits16-pretrain-lvd1689m/facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    return cfg
