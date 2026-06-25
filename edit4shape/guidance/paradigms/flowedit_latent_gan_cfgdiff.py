"""FlowEdit + CFG-Diff Latent GAN Guidance."""
import logging

import torch

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance
from edit4shape.guidance.paradigms.flowedit_latent_gan import FlowEditLatentGANGuidance
from edit4shape.guidance.latent_discriminator import CFGDiffLatentDiscriminatorHelper


class FlowEditLatentGANCFGDiffGuidance(FlowEditLatentGANGuidance):
    """FlowEdit + cond-uncond diff latent-space adversarial loss."""

    loss_key = "flowedit_latent_gan_cfgdiff"

    def _ensure_discriminator(self, loss_cfg):
        if self._disc_helper is not None:
            return
        self._loss_cfg = loss_cfg
        self._disc_helper = CFGDiffLatentDiscriminatorHelper(loss_cfg, self.pipe, self.device)
        logging.info("[LatentGAN-CFGDiff] D ready (hooks=%s, head_channels=%d)",
                     self._disc_helper._disc.hook_block_ids,
                     loss_cfg.gan_head_channels)

    def _compute_pixel_loss(self, comp_rgb, pipeline_output, guidance_cfg):
        total_loss, loss_dict = FlowEditGuidance._compute_pixel_loss(
            self, comp_rgb, pipeline_output, guidance_cfg,
        )

        loss_cfg = guidance_cfg.loss
        gan_weight = getattr(loss_cfg, 'gan', 0.0)
        if gan_weight <= 0:
            return total_loss, loss_dict

        self._ensure_discriminator(loss_cfg)
        edited = pipeline_output.edited_tensor.detach()

        with self._disc_helper.enabled():
            d_loss, r1_val = self._disc_helper.d_step(
                comp_rgb, edited, loss_cfg,
                prompt_embeds=pipeline_output.pos_prompt_embeds,
                prompt_mask=pipeline_output.pos_prompt_embeds_mask,
                negative_prompt_embeds=pipeline_output.neg_prompt_embeds,
                negative_prompt_mask=pipeline_output.neg_prompt_embeds_mask,
            )

            with self._disc_helper.g_enabled():
                g_loss = self._disc_helper.g_step(
                    comp_rgb,
                    prompt_embeds=pipeline_output.pos_prompt_embeds,
                    prompt_mask=pipeline_output.pos_prompt_embeds_mask,
                    negative_prompt_embeds=pipeline_output.neg_prompt_embeds,
                    negative_prompt_mask=pipeline_output.neg_prompt_embeds_mask,
                )

        if not torch.isfinite(g_loss):
            logging.warning("[LatentGAN-CFGDiff] g_loss non-finite, skipping")
            loss_dict["gan_g"] = torch.zeros_like(total_loss)
            loss_dict["gan_d"] = d_loss
            loss_dict["gan_r1"] = r1_val
            return total_loss, loss_dict

        total_loss = total_loss + gan_weight * g_loss
        loss_dict["gan_g"] = (gan_weight * g_loss).detach()
        loss_dict["gan_d"] = d_loss
        loss_dict["gan_r1"] = r1_val
        return total_loss, loss_dict
