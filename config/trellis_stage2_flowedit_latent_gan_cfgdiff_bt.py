"""TRELLIS Stage 2 FlowEdit + CFG-Diff Latent GAN (Bradley-Terry D)。"""
import ml_collections
from config.trellis_stage2_flowedit_latent_gan_cfgdiff import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "trellis_stage2_flowedit_latent_gan_cfgdiff_bt"
    cfg.guidance.type = "flowedit_latent_gan_cfgdiff_bt"
    cfg.guidance.flowedit_latent_gan_cfgdiff_bt = ml_collections.ConfigDict(
        cfg.guidance.flowedit_latent_gan_cfgdiff.to_dict()
    )
    return cfg
