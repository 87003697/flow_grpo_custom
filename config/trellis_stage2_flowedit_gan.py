"""TRELLIS Stage 2 FlowEdit + GAN 配置。"""
from config.trellis_stage2_flowedit_denoise import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "trellis_stage2_flowedit_gan"
    cfg.guidance.type = "flowedit_gan"
    cfg.guidance.flowedit_gan = cfg.guidance.flowedit

    cfg.train.guidance.loss.gan = 0.1
    cfg.train.guidance.loss.gan_lr = 2e-4
    cfg.train.guidance.loss.gan_model_path = (
        "pretrained_weights/dinov3-vits16-pretrain-lvd1689m/facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    return cfg
