"""TRELLIS Stage 2 FlowEdit + CFG-Diff Latent GAN。"""
import ml_collections
from config.trellis_stage2_flowedit_latent_gan import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "trellis_stage2_flowedit_latent_gan_cfgdiff"
    cfg.guidance.type = "flowedit_latent_gan_cfgdiff"
    cfg.guidance.flowedit_latent_gan_cfgdiff = ml_collections.ConfigDict(
        cfg.guidance.flowedit_latent_gan.to_dict()
    )
    return cfg
