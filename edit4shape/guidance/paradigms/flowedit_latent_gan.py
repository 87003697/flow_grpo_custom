"""FlowEdit + QwenImage Latent GAN Guidance."""
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance
from edit4shape.guidance.latent_discriminator import LatentDiscriminatorHelper


class FlowEditLatentGANGuidance(FlowEditGuidance):
    """FlowEdit + QwenImage latent-space adversarial loss."""

    loss_key = "flowedit_latent_gan"

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        super().__init__(guidance_cfg, train_device)
        self._disc_helper: Optional[LatentDiscriminatorHelper] = None
        self._loss_cfg = None

    def _ensure_discriminator(self, loss_cfg):
        if self._disc_helper is not None:
            return
        self._loss_cfg = loss_cfg
        self._disc_helper = LatentDiscriminatorHelper(loss_cfg, self.pipe, self.device)
        logging.info("[LatentGAN] D ready (hooks=%s, head_channels=%d)",
                     self._disc_helper._disc.hook_block_ids,
                     loss_cfg.gan_head_channels)

    def _compute_pixel_loss(self, comp_rgb, pipeline_output, guidance_cfg):
        total_loss, loss_dict = super()._compute_pixel_loss(
            comp_rgb, pipeline_output, guidance_cfg,
        )

        loss_cfg = guidance_cfg.loss
        gan_weight = getattr(loss_cfg, 'gan', 0.0)
        if gan_weight <= 0:
            return total_loss, loss_dict

        self._ensure_discriminator(loss_cfg)
        edited = pipeline_output.edited_tensor.detach()

        prompt_embeds = pipeline_output.prompt_embeds_tgt
        prompt_mask = pipeline_output.prompt_embeds_mask_tgt

        with self._disc_helper.enabled():
            d_loss, r1_val = self._disc_helper.update(
                comp_rgb, edited, loss_cfg,
                prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
            )

            with self._disc_helper.g_enabled():
                g_loss = self._disc_helper.g_loss(
                    comp_rgb,
                    prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                )

        if not torch.isfinite(g_loss):
            logging.warning("[LatentGAN] g_loss is non-finite, skipping GAN loss this step")
            loss_dict["gan_g"] = torch.zeros_like(total_loss)
            loss_dict["gan_d"] = d_loss
            loss_dict["gan_r1"] = r1_val
            return total_loss, loss_dict

        total_loss = total_loss + gan_weight * g_loss
        loss_dict["gan_g"] = (gan_weight * g_loss).detach()
        loss_dict["gan_d"] = d_loss
        loss_dict["gan_r1"] = r1_val
        return total_loss, loss_dict

    # ---- Checkpoint ----

    def save_checkpoint(self, ckpt_dir):
        if self._disc_helper is not None:
            self._disc_helper.save(Path(ckpt_dir) / "guidance_state.pt")

    def load_checkpoint(self, ckpt_dir, *, loss_cfg=None):
        path = Path(ckpt_dir) / "guidance_state.pt"
        if not path.exists():
            return
        if loss_cfg is not None:
            self._loss_cfg = loss_cfg
        if self._disc_helper is None and self._loss_cfg is not None:
            self._ensure_discriminator(self._loss_cfg)
        if self._disc_helper is not None:
            self._disc_helper.load(path)

    def cleanup(self):
        if self._disc_helper is not None:
            self._disc_helper.cleanup()
            self._disc_helper = None
        super().cleanup()
