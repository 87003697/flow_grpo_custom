"""TRELLIS Stage 2 Distillation + CFG-Diff Latent GAN。

Entry: edit4shape.systems.trellis.entries.autograd（VJP through rollout）
Guidance: flowedit_latent_gan_cfgdiff（CFG-Diff Latent GAN，BCE D+G）

与 flowedit denoise 的区别：通过 VJP 训练整个 rollout chain，
而不是冻住 rollout 只训单步去噪。
"""
import ml_collections
from config.trellis_stage2_distillation import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "distill_latent_gan_cfgdiff"

    # === Guidance: 切换到 CFG-Diff Latent GAN ===
    g = cfg.guidance
    g.type = "flowedit_latent_gan_cfgdiff"
    g.flowedit_latent_gan_cfgdiff = ml_collections.ConfigDict(g.flowedit.to_dict())

    # === Prompt ===
    cfg.train.guidance.target_prompt = "Rotate the camera. White background."
    cfg.train.guidance.source_prompt = cfg.train.guidance.target_prompt

    # === Loss: GAN-only (no MSE pixel loss) ===
    cfg.train.guidance.loss.mse = 0.0
    cfg.train.guidance.loss.gan = 0.01
    cfg.train.guidance.loss.gan_r1_gamma = 0.0
    cfg.train.guidance.loss.gan_hook_block_ids = [14, 29, 44, 59]
    cfg.train.guidance.loss.gan_head_channels = 384
    cfg.train.guidance.loss.gan_t_d_mean = -0.6
    cfg.train.guidance.loss.gan_t_d_std = 1.0

    # === D optimizer ===
    cfg.train.guidance.loss.gan_opt = ml_collections.ConfigDict()
    cfg.train.guidance.loss.gan_opt.type = "adan"
    cfg.train.guidance.loss.gan_opt.lr = 2e-5
    cfg.train.guidance.loss.gan_opt.eps = 1e-8
    cfg.train.guidance.loss.gan_opt.weight_decay = 0.0

    # === Reg: velocity MSE (VJP rollout 正则化) ===
    cfg.rollout.reg.type = "v"

    return cfg
