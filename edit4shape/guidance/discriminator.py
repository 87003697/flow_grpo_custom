"""Discriminator helpers（Base + DINOv3-S 实现）。"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import AutoModel

from edit4shape.generators.trellis.training_adpter import _build_optimizer


# =============================================================================
# 共用 loss 函数
# =============================================================================

def bce_d_loss(d_real, d_fake):
    real_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
    fake_loss = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return real_loss + fake_loss


def bce_g_loss(d_fake):
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def r1_gradient_penalty(disc, real_images):
    """R1 gradient penalty (Mescheder et al., 2018).

    Penalizes ||∇_x D(x_real)||² to keep D's decision surface smooth.
    real_images must have requires_grad=True before calling.

    Uses mean (not sum) over spatial dims to keep scale independent of
    input resolution. Uses math-only SDPA backend because flash/efficient
    attention does not support higher-order gradients (create_graph=True).
    """
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        d_real = disc(real_images)
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    return grad_real.pow(2).flatten(1).mean(1).mean()


# =============================================================================
# DINOv3-S Discriminator（网络结构）
# =============================================================================

class SpectralConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.scale = 1.0 / (2 ** 0.5)

    def forward(self, x):
        return (self.fn(x) + x) * self.scale


def _make_head(channels, kernel_size=9):
    return nn.Sequential(
        SpectralConv2d(channels, channels, 1),
        nn.GroupNorm(32, channels),
        nn.LeakyReLU(0.2, inplace=True),
        ResidualBlock(nn.Sequential(
            SpectralConv2d(channels, channels, kernel_size,
                          padding=kernel_size // 2, padding_mode='circular'),
            nn.GroupNorm(32, channels),
            nn.LeakyReLU(0.2, inplace=True),
        )),
        SpectralConv2d(channels, 1, 1),
    )


class DINOv3sDiscriminator(nn.Module):
    """Frozen DINOv3-S + 4 learnable multi-scale heads."""

    def __init__(self, model_path, key_depths=(2, 5, 8, 11), kernel_size=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 384 for ViT-S
        self.num_prefix_tokens = 1 + getattr(self.encoder.config, 'num_register_tokens', 0)
        self.key_depths = key_depths
        self.heads = nn.ModuleList([_make_head(hidden_size, kernel_size) for _ in key_depths])

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def trainable_parameters(self):
        return self.heads.parameters()

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1]."""
        x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.encoder(x, output_hidden_states=True)

        logits = []
        for depth, head in zip(self.key_depths, self.heads):
            feat = out.hidden_states[depth][:, self.num_prefix_tokens:]  # drop CLS + registers
            h = w = int(feat.shape[1] ** 0.5)
            feat = feat.transpose(1, 2).reshape(feat.shape[0], -1, h, w)
            logits.append(head(feat).reshape(feat.shape[0], -1))

        return torch.cat(logits, dim=1)


# =============================================================================
# BaseDiscriminatorHelper — 共用 infra
# =============================================================================

class BaseDiscriminatorHelper:
    """Discriminator 基础设施：network、optimizer、DDP、mode switch、checkpoint。

    公开接口（Base 实现，处理 boilerplate）:
      - d_step(comp_rgb, edited, loss_cfg, **kwargs) → (d_loss, r1)
      - g_step(comp_rgb, **kwargs) → tensor

    子类 override（纯计算）:
      - _compute_d(comp_rgb, edited, loss_cfg, **kwargs) → (d_loss, r1)
      - _compute_g(comp_rgb, **kwargs) → g_loss tensor
    """

    def __init__(self, disc, opt_cfg, device):
        self.device = device
        self._disc = disc
        self._step = 0
        trainable_params = list(disc.trainable_parameters())
        assert trainable_params, "Discriminator has no trainable parameters"
        self._opt = _build_optimizer(trainable_params, opt_cfg)
        self._broadcast_weights()

    @property
    def step(self):
        return self._step

    # ---- DDP ----

    @staticmethod
    def _is_distributed():
        return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    def _broadcast_weights(self):
        if not self._is_distributed():
            return
        for p in self._disc.trainable_parameters():
            torch.distributed.broadcast(p.data, src=0)

    def _sync_gradients(self):
        if not self._is_distributed():
            return
        for p in self._disc.trainable_parameters():
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

    # ---- Mode switch ----

    def _set_train_mode(self):
        self._disc.train()
        for p in self._disc.trainable_parameters():
            p.requires_grad = True

    def _set_eval_mode(self):
        self._disc.eval()
        for p in self._disc.trainable_parameters():
            p.requires_grad = False

    # ---- Optimize step ----

    def _optimize(self, loss):
        self._opt.zero_grad()
        loss.backward()
        self._sync_gradients()
        torch.nn.utils.clip_grad_norm_(self._disc.trainable_parameters(), 1.0)
        self._opt.step()
        self._step += 1

    # ---- 子类 override ----

    def _compute_d(self, comp_rgb, edited, loss_cfg, **kwargs):
        """纯计算：返回 (d_loss, r1)。子类必须实现。"""
        raise NotImplementedError

    def _compute_g(self, comp_rgb, **kwargs):
        """纯计算：返回 g_loss tensor。子类必须实现。"""
        raise NotImplementedError

    # ---- 公开接口 ----

    def d_step(self, comp_rgb, edited, loss_cfg, **kwargs):
        """D step: train mode → _compute_d → optimize → return detached."""
        self._set_train_mode()
        d_loss, r1 = self._compute_d(comp_rgb, edited, loss_cfg, **kwargs)
        r1_gamma = getattr(loss_cfg, 'gan_r1_gamma', 0.0)
        self._optimize(d_loss + (r1_gamma / 2) * r1)
        return d_loss.detach(), r1.detach()

    def g_step(self, comp_rgb, **kwargs):
        """G step: eval mode → _compute_g → return loss."""
        self._set_eval_mode()
        return self._compute_g(comp_rgb, **kwargs)

    # ---- Checkpoint ----

    def save(self, path):
        torch.save({
            "version": 2,
            "disc": {k: v for k, v in self._disc.state_dict().items()
                     if not k.startswith("encoder.") and not k.startswith("transformer.")},
            "opt": self._opt.state_dict(),
            "step": self._step,
        }, path)

    def load(self, path):
        sd = torch.load(path, map_location="cpu")
        missing, unexpected = self._disc.load_state_dict(sd["disc"], strict=False)
        head_keys = {k for k in self._disc.state_dict() if k.startswith("heads.")}
        loaded_keys = set(sd["disc"].keys())
        missed_heads = head_keys - loaded_keys
        if missed_heads:
            logging.warning("D checkpoint missing head keys (random init): %s", missed_heads)
        self._opt.load_state_dict(sd["opt"])
        self._step = sd["step"]

    def cleanup(self):
        del self._disc, self._opt


# =============================================================================
# DINOv3-S DiscriminatorHelper（继承 Base）
# =============================================================================

class DiscriminatorHelper(BaseDiscriminatorHelper):
    """DINOv3-S 判别器。"""

    def __init__(self, loss_cfg, device):
        disc = DINOv3sDiscriminator(model_path=loss_cfg.gan_model_path).to(device)
        super().__init__(disc, opt_cfg=loss_cfg.gan_opt, device=device)

    def _compute_d(self, comp_rgb, edited, loss_cfg, **kwargs):
        d_loss = bce_d_loss(self._disc(edited.detach()), self._disc(comp_rgb.detach()))
        r1 = r1_gradient_penalty(self._disc, edited.detach().requires_grad_(True))
        return d_loss, r1

    def _compute_g(self, comp_rgb, **kwargs):
        return bce_g_loss(self._disc(comp_rgb))
