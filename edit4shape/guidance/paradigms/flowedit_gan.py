"""FlowEdit + DINOv3-S GAN Guidance."""
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditPipelineOutput
from edit4shape.guidance.discriminator import DiscriminatorHelper


class FlowEditGANGuidance(FlowEditGuidance):
    """FlowEdit + DINOv3-S adversarial loss."""

    loss_key = "flowedit_gan"

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        super().__init__(guidance_cfg, train_device)
        self._disc_helper: Optional[DiscriminatorHelper] = None
        self._loss_cfg = None

    def _ensure_discriminator(self, loss_cfg):
        if self._disc_helper is not None:
            return
        self._loss_cfg = loss_cfg
        self._disc_helper = DiscriminatorHelper(loss_cfg, self.device)
        logging.info("[FlowEditGANGuidance] Discriminator ready on %s", self.device)

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

        d_loss, r1_val = self._disc_helper.update(comp_rgb, edited, loss_cfg)
        g_loss = self._disc_helper.g_loss(comp_rgb)

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
            logging.info(
                "[FlowEditGANGuidance] Loaded D checkpoint (step=%d)",
                self._disc_helper.step,
            )

    def cleanup(self):
        if self._disc_helper is not None:
            self._disc_helper.cleanup()
            self._disc_helper = None
        super().cleanup()
