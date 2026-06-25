"""QwenImage Latent-Space Discriminator (SANA-Sprint LADD 风格)。

核心思想：复用 FlowEdit 已加载的 QwenImage Transformer 作为 frozen D backbone，
通过 register_forward_hook 提取多层中间特征，接轻量 SpectralConv1d heads 产生
per-token logits。输入先 re-noise 到随机时间步再送入 backbone（LADD 核心 trick）。
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm

from edit4shape.guidance.discriminator import BaseDiscriminatorHelper


# =============================================================================
# Head 组件
# =============================================================================

class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._scale = 1.0 / math.sqrt(2)

    def forward(self, x):
        return (self.fn(x) + x) * self._scale


def _make_latent_head(in_channels, head_channels, kernel_size=9):
    assert head_channels % 32 == 0, f"head_channels must be divisible by 32 for GroupNorm, got {head_channels}"
    return nn.Sequential(
        SpectralConv1d(in_channels, head_channels, 1),
        nn.GroupNorm(32, head_channels),
        nn.LeakyReLU(0.2, inplace=True),
        ResidualBlock(nn.Sequential(
            SpectralConv1d(head_channels, head_channels, kernel_size,
                          padding=kernel_size // 2, padding_mode="circular"),
            nn.GroupNorm(32, head_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )),
        SpectralConv1d(head_channels, 1, 1),
    )


# =============================================================================
# QwenImage Discriminator
# =============================================================================

class QwenImageDiscriminator(nn.Module):
    """Frozen QwenImage Transformer backbone + multi-scale DiscHead.

    QwenImageTransformerBlock returns (encoder_hidden_states, hidden_states)
    i.e. (text, image). Hooks capture output[1] for the image stream.
    """
    def __init__(self, transformer, hook_block_ids, head_channels):
        super().__init__()
        self.transformer = transformer
        self.transformer.requires_grad_(False)
        self.hook_block_ids = tuple(sorted(hook_block_ids))
        inner_dim = transformer.inner_dim  # 3072
        self.heads = nn.ModuleList([
            _make_latent_head(inner_dim, head_channels) for _ in hook_block_ids
        ])

    def trainable_parameters(self):
        return self.heads.parameters()

    @contextmanager
    def _hooks(self):
        """Register temporary forward hooks. Exception-safe removal."""
        feat_list = []

        def hook_fn(module, input, output):
            feat_list.append(output[1])  # [B, S, 3072] — image stream

        handles = []
        hook_set = set(self.hook_block_ids)
        for i, block in enumerate(self.transformer.transformer_blocks):
            if i in hook_set:
                handles.append(block.register_forward_hook(hook_fn))
        try:
            yield feat_list
        finally:
            for h in handles:
                h.remove()

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, img_shapes, **kwargs):
        with self._hooks() as feat_list:
            self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                img_shapes=img_shapes,
                return_dict=False,
                **kwargs,
            )

        logits_list = []
        for feat, head in zip(feat_list, self.heads):
            feat_t = feat.float().transpose(1, 2)  # [B, 3072, N] in float32
            out = head(feat_t)             # [B, 1, N]
            logits_list.append(out.reshape(feat.shape[0], -1))  # [B, N]

        return torch.cat(logits_list, dim=1)  # [B, num_hooks * N]


# =============================================================================
# LatentDiscriminatorHelper
# =============================================================================

def _sample_d_timestep(batch_size, mean, std, device):
    """Sample D re-noising timestep: t = sigmoid(Normal(mean, std))."""
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u)


class LatentDiscriminatorHelper(BaseDiscriminatorHelper):
    """QwenImage Latent D lifecycle — inherits BaseDiscriminatorHelper."""

    def __init__(self, loss_cfg, pipe, device):
        self.pipe = pipe
        self._t_d_mean = loss_cfg.gan_t_d_mean
        self._t_d_std = loss_cfg.gan_t_d_std
        self._active = False

        disc = QwenImageDiscriminator(
            pipe.transformer,
            hook_block_ids=loss_cfg.gan_hook_block_ids,
            head_channels=loss_cfg.gan_head_channels,
        ).to(device)
        super().__init__(disc, opt_cfg=loss_cfg.gan_opt, device=device)

    # ---- Context Manager ----

    @contextmanager
    def enabled(self):
        self._active = True
        try:
            yield
        finally:
            self._active = False

    @contextmanager
    def g_enabled(self):
        """G step: enable gradient_checkpointing for 60-layer backprop."""
        transformer = self.pipe.transformer
        was_enabled = transformer.gradient_checkpointing
        if not was_enabled:
            transformer.enable_gradient_checkpointing()
        try:
            yield
        finally:
            if not was_enabled:
                transformer.disable_gradient_checkpointing()

    # ---- Subclass interface ----

    def _d_logits(self, comp_rgb, edited, **kwargs):
        prompt_embeds = kwargs['prompt_embeds']
        prompt_mask = kwargs['prompt_mask']
        B = comp_rgb.shape[0]
        with torch.no_grad():
            z_fake, hw_fake = self._encode_to_latent(comp_rgb.detach())
            z_real, hw_real = self._encode_to_latent(edited.detach())
            t = _sample_d_timestep(B, self._t_d_mean, self._t_d_std, z_fake.device)
            z_t_fake = self._renoise_with_t(z_fake, t)
            z_t_real = self._renoise_with_t(z_real, t)
        d_fake = self._disc_forward(z_t_fake, t, prompt_embeds, prompt_mask, hw_fake)
        d_real = self._disc_forward(z_t_real, t, prompt_embeds, prompt_mask, hw_real)
        return d_real, d_fake

    def _g_logits(self, comp_rgb, **kwargs):
        prompt_embeds = kwargs['prompt_embeds']
        prompt_mask = kwargs['prompt_mask']
        B = comp_rgb.shape[0]
        z_fake, hw = self._encode_to_latent(comp_rgb)
        z_t_fake, t_fake = self._renoise(z_fake, B)
        return self._disc_forward(z_t_fake, t_fake, prompt_embeds, prompt_mask, hw)

    # ---- Internal ----

    def _encode_to_latent(self, images):
        """RGB images → VAE encode → packed latent tokens. Returns (packed, (H_packed, W_packed))."""
        B = images.shape[0]
        edit_res = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        resized = F.interpolate(
            images, size=(edit_res, edit_res),
            mode='bicubic', align_corners=False, antialias=True,
        )
        normalized = (resized * 2 - 1).unsqueeze(2).to(dtype=torch.bfloat16)
        latent_5d = self.pipe._encode_vae_image_differentiable(normalized)
        _, C, _, H, W = latent_5d.shape
        packed = self.pipe._pack_latents(latent_5d, B, C, H, W)
        return packed, (H // 2, W // 2)

    def _renoise(self, z0, batch_size):
        """Re-noise clean latent to random timestep. Returns (z_t, t)."""
        t = _sample_d_timestep(batch_size, self._t_d_mean, self._t_d_std, z0.device)
        z_t = self._renoise_with_t(z0, t)
        return z_t, t

    def _renoise_with_t(self, z0, t):
        """Re-noise with given timestep (D step: shared t for real/fake)."""
        noise = torch.randn_like(z0)
        t_exp = t.view(-1, 1, 1)
        return (1 - t_exp) * z0 + t_exp * noise

    def _disc_forward(self, z_t, timestep, prompt_embeds, prompt_mask, latent_hw):
        """Wrap D forward: expand prompt embedding + build img_shapes."""
        B = z_t.shape[0]
        dtype = self.pipe.transformer.dtype
        embeds = prompt_embeds.to(dtype=dtype)
        mask = prompt_mask
        if embeds.shape[0] != B:
            embeds = embeds.expand(B, -1, -1)
            mask = mask.expand(B, -1) if mask is not None else None
        H_lat, W_lat = latent_hw
        img_shapes = [[(1, H_lat, W_lat)]] * B
        return self._disc(
            hidden_states=z_t.to(dtype=dtype),
            timestep=timestep.to(dtype=dtype),
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=mask,
            img_shapes=img_shapes,
        )
