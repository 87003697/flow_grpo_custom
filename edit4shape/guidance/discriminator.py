"""DINOv3-S Projected Discriminator（参考 FAIL Dinov3sDisc）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import AutoModel


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


def hinge_d_loss(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_g_loss(d_render):
    """G wants D(render) high → loss = -D(render).mean()"""
    return -d_render.mean()
