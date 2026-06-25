"""TRELLIS Stage 2 FlowEdit + QwenImage Latent GAN。"""
import ml_collections
from config.trellis_stage2_flowedit_denoise import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "trellis_stage2_flowedit_latent_gan"
    cfg.guidance.type = "flowedit_latent_gan"
    cfg.guidance.flowedit_latent_gan = cfg.guidance.flowedit

    # GAN loss
    cfg.train.guidance.loss.gan = 1.0
    cfg.train.guidance.loss.gan_r1_gamma = 0.0
    cfg.train.guidance.loss.gan_hook_block_ids = [14, 29, 44, 59]
    cfg.train.guidance.loss.gan_head_channels = 384
    cfg.train.guidance.loss.gan_t_d_mean = -0.6
    cfg.train.guidance.loss.gan_t_d_std = 1.0

    # D optimizer
    cfg.train.guidance.loss.gan_opt = ml_collections.ConfigDict()
    cfg.train.guidance.loss.gan_opt.type = "adan"
    cfg.train.guidance.loss.gan_opt.lr = 2e-5
    cfg.train.guidance.loss.gan_opt.eps = 1e-8
    cfg.train.guidance.loss.gan_opt.weight_decay = 0.0
    return cfg
