"""TRELLIS Stage 2 FlowEdit Denoise + 三图 DINO GAN (BCE + BT)。

Entry: edit4shape.systems.trellis.entries.flowedit_autograd
Guidance: flowedit_triimage_gan（三图 DINO D：BCE + BT ranking）

训练流程：
  Pretrained Rollout (frozen) → clean z₀
  → 加噪 z₀ → zₜ (随机时间步)
  → Finetuned 单步去噪 → ẑ₀
  → Decode + Render → comp_rgb
  → 2D FlowEdit Guidance (三图 GAN) → loss → autograd backward
"""
import ml_collections
from config.trellis_stage2_flowedit_denoise import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "denoise_triimage_gan"

    # === Guidance: 切换到三图 DINO GAN ===
    g = cfg.guidance
    g.type = "flowedit_triimage_gan"
    g.flowedit_triimage_gan = ml_collections.ConfigDict(g.flowedit.to_dict())

    # === Loss: 添加 GAN 参数 ===
    cfg.train.guidance.loss.gan = 0.01
    cfg.train.guidance.loss.gan_r1_gamma = 0.0
    cfg.train.guidance.loss.gan_bt_weight = 1.0
    cfg.train.guidance.loss.gan_model_path = "facebook/dinov2-small"

    # === D optimizer ===
    cfg.train.guidance.loss.gan_opt = ml_collections.ConfigDict()
    cfg.train.guidance.loss.gan_opt.type = "adan"
    cfg.train.guidance.loss.gan_opt.lr = 2e-5
    cfg.train.guidance.loss.gan_opt.eps = 1e-8
    cfg.train.guidance.loss.gan_opt.weight_decay = 0.0

    return cfg
