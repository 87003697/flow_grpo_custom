"""TRELLIS Stage 2 Distillation + 三图 DINO GAN (BCE + BT)。

Entry: edit4shape.systems.trellis.entries.autograd（VJP through rollout）
Guidance: flowedit_triimage_gan（三图 DINO D：BCE + BT ranking）

D 看三张图：参考(label=1) / 编辑前(label=0) 做 BCE，
编辑后 vs 编辑前做 BT ranking。
"""
import ml_collections
from config.trellis_stage2_distillation import get_config as _base_config


def get_config():
    cfg = _base_config()
    cfg.run_name = "distill_triimage_gan"

    # === Guidance: 切换到三图 DINO GAN ===
    g = cfg.guidance
    g.type = "flowedit_triimage_gan"
    g.flowedit_triimage_gan = ml_collections.ConfigDict(g.flowedit.to_dict())

    # === Prompt ===
    cfg.train.guidance.target_prompt = "Rotate the camera. White background."
    cfg.train.guidance.source_prompt = cfg.train.guidance.target_prompt

    # === Loss ===
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

    # === Reg: velocity MSE (VJP rollout 正则化) ===
    cfg.rollout.reg.type = "v"

    return cfg
