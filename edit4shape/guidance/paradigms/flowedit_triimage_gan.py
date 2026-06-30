"""FlowEdit + 三图 DINO GAN (BCE + BT) Guidance."""
import logging
from typing import Any, List

import torch
from PIL import Image

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditPipelineOutput
from edit4shape.guidance.paradigms.flowedit_gan import FlowEditGANGuidance
from edit4shape.guidance.discriminator import TriImageDiscriminatorHelper
from edit4shape.systems.utils import composite_alpha


class FlowEditTriImageGANGuidance(FlowEditGANGuidance):
    """三图 DINO D：参考(label=1) / 编辑前(label=0) / 编辑后(BT>编辑前)。"""

    loss_key = "flowedit_triimage_gan"

    def _run_pipeline(self, comp_rgb, condition_images, src_latent, guidance_cfg, B, V):
        output = super()._run_pipeline(comp_rgb, condition_images, src_latent, guidance_cfg, B, V)
        N, C, H, W = comp_rgb.shape
        bg_color = tuple(guidance_cfg.bg_color)
        cond_pils = [composite_alpha(img, bg_color) for img in condition_images for _ in range(V)]
        output.condition_tensor = self.pils_to_tensor(cond_pils, (W, H))
        return output

    def _ensure_discriminator(self, loss_cfg):
        if self._disc_helper is not None:
            return
        self._loss_cfg = loss_cfg
        self._disc_helper = TriImageDiscriminatorHelper(loss_cfg, self.device)
        logging.info("[TriImageGAN] D ready (DINO model=%s)", loss_cfg.gan_model_path)

    def _compute_pixel_loss(self, comp_rgb, pipeline_output, guidance_cfg):
        total_loss, loss_dict = FlowEditGuidance._compute_pixel_loss(
            self, comp_rgb, pipeline_output, guidance_cfg)

        loss_cfg = guidance_cfg.loss
        gan_weight = getattr(loss_cfg, 'gan', 0.0)
        if gan_weight <= 0:
            return total_loss, loss_dict

        self._ensure_discriminator(loss_cfg)
        edited = pipeline_output.edited_tensor.detach()
        condition_tensor = pipeline_output.condition_tensor

        d_loss, r1_val = self._disc_helper.d_step(
            comp_rgb, edited, loss_cfg,
            condition_tensor=condition_tensor,
        )

        g_loss = self._disc_helper.g_step(comp_rgb)

        if not torch.isfinite(g_loss):
            logging.warning("[TriImageGAN] g_loss non-finite, skipping")
            loss_dict["gan_g"] = torch.zeros_like(total_loss)
            loss_dict["gan_d"] = d_loss
            loss_dict["gan_r1"] = r1_val
            return total_loss, loss_dict

        total_loss = total_loss + gan_weight * g_loss
        loss_dict["gan_g"] = (gan_weight * g_loss).detach()
        loss_dict["gan_d"] = d_loss
        loss_dict["gan_r1"] = r1_val
        return total_loss, loss_dict
