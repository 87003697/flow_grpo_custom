"""FlowEdit + CFG-Diff Latent GAN (Bradley-Terry D) Guidance."""
import logging

from edit4shape.guidance.paradigms.flowedit_latent_gan_cfgdiff import (
    FlowEditLatentGANCFGDiffGuidance,
)
from edit4shape.guidance.latent_discriminator import BTCFGDiffLatentDiscriminatorHelper


class FlowEditLatentGANCFGDiffBTGuidance(FlowEditLatentGANCFGDiffGuidance):
    """FlowEdit + CFG-Diff + Bradley-Terry D loss.

    继承全部 FlowEdit + GAN 逻辑，仅替换 discriminator 为 BT 版本。
    """

    def _ensure_discriminator(self, loss_cfg):
        if self._disc_helper is not None:
            return
        self._loss_cfg = loss_cfg
        self._disc_helper = BTCFGDiffLatentDiscriminatorHelper(loss_cfg, self.pipe, self.device)
        logging.info("[LatentGAN-CFGDiff-BT] D ready (hooks=%s, head_channels=%d)",
                     self._disc_helper._disc.hook_block_ids,
                     loss_cfg.gan_head_channels)
