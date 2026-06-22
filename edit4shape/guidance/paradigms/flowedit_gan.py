"""FlowEdit + DINOv3-S GAN Guidance."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance, FlowEditPipelineOutput
from edit4shape.guidance.discriminator import (
    DINOv3sDiscriminator, bce_d_loss, bce_g_loss,
)


class FlowEditGANGuidance(FlowEditGuidance):
    """FlowEdit + DINOv3-S adversarial loss."""

    loss_key = "flowedit_gan"

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        super().__init__(guidance_cfg, train_device)
        self._disc: Optional[DINOv3sDiscriminator] = None
        self._disc_opt = None
        self._disc_step: int = 0
        self._accelerator = None
        self._loss_cfg = None

    def set_accelerator(self, accelerator):
        self._accelerator = accelerator

    def _ensure_discriminator(self, loss_cfg):
        if self._disc is not None:
            return
        self._loss_cfg = loss_cfg
        disc = DINOv3sDiscriminator(model_path=loss_cfg.gan_model_path).to(self.device)
        opt = torch.optim.Adam(disc.trainable_parameters(), lr=loss_cfg.gan_lr, betas=(0.0, 0.99))
        self._disc, self._disc_opt = disc, opt
        logging.info("[FlowEditGANGuidance] Discriminator ready on %s", self.device)

    def _disc_train_mode(self):
        self._disc.train()
        for p in self._disc.trainable_parameters():
            p.requires_grad = True

    def _disc_eval_mode(self):
        self._disc.eval()
        for p in self._disc.trainable_parameters():
            p.requires_grad = False

    def _disc_step_update(self, comp_rgb, edited, loss_cfg):
        self._disc_train_mode()

        real_d = edited.detach()
        d_real = self._disc(real_d)
        d_fake = self._disc(comp_rgb.detach())
        d_loss = bce_d_loss(d_real, d_fake)

        self._disc_opt.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._disc.trainable_parameters(), 1.0)
        self._disc_opt.step()
        self._disc_step += 1
        return d_loss.detach()

    def _gen_step(self, comp_rgb):
        self._disc_eval_mode()
        return bce_g_loss(self._disc(comp_rgb))

    def _compute_pixel_loss(self, comp_rgb, pipeline_output, guidance_cfg):
        total_loss, loss_dict = super()._compute_pixel_loss(comp_rgb, pipeline_output, guidance_cfg)

        loss_cfg = guidance_cfg.loss
        gan_weight = getattr(loss_cfg, 'gan', 0.0)
        if gan_weight <= 0:
            return total_loss, loss_dict

        self._ensure_discriminator(loss_cfg)
        edited = pipeline_output.edited_tensor.detach()

        d_loss = self._disc_step_update(comp_rgb, edited, loss_cfg)
        g_loss = self._gen_step(comp_rgb)

        total_loss = total_loss + gan_weight * g_loss
        loss_dict["gan_g"] = (gan_weight * g_loss).detach()
        loss_dict["gan_d"] = d_loss
        return total_loss, loss_dict

    # ---- Checkpoint hooks ----

    def save_checkpoint(self, ckpt_dir):
        if self._disc is None:
            return
        path = Path(ckpt_dir) / "guidance_state.pt"
        torch.save({
            "disc": self._disc.state_dict(),
            "opt": self._disc_opt.state_dict(),
            "step": self._disc_step,
        }, path)

    def load_checkpoint(self, ckpt_dir, *, loss_cfg=None):
        path = Path(ckpt_dir) / "guidance_state.pt"
        if not path.exists():
            return
        sd = torch.load(path, map_location="cpu")
        if loss_cfg is not None:
            self._loss_cfg = loss_cfg
        if self._disc is None and self._loss_cfg is not None:
            self._ensure_discriminator(self._loss_cfg)
        if self._disc is not None:
            self._disc.load_state_dict(sd["disc"])
            self._disc_opt.load_state_dict(sd["opt"])
            self._disc_step = sd["step"]
            logging.info("[FlowEditGANGuidance] Loaded D checkpoint (step=%d)", self._disc_step)

    def cleanup(self):
        if self._disc is not None:
            del self._disc, self._disc_opt
            self._disc = self._disc_opt = None
        super().cleanup()
