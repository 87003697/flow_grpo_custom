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


class DiscriminatorHelper:
    """DINOv3-S 判别器生命周期管理：初始化、DDP 同步、更新、checkpoint。"""

    def __init__(self, loss_cfg, device):
        self.device = device
        self._step = 0

        self._disc = DINOv3sDiscriminator(model_path=loss_cfg.gan_model_path).to(device)
        self._opt = torch.optim.Adam(
            self._disc.trainable_parameters(), lr=loss_cfg.gan_lr, betas=(0.0, 0.99),
        )
        self._broadcast_weights()

    @property
    def step(self):
        return self._step

    @staticmethod
    def _is_distributed():
        return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    def _broadcast_weights(self):
        """初始化后从 rank 0 广播 heads 权重，确保各 rank 起点一致。
        encoder 由 from_pretrained 加载，各 rank 已一致，无需广播。"""
        if not self._is_distributed():
            return
        for p in self._disc.trainable_parameters():
            torch.distributed.broadcast(p.data, src=0)

    def _sync_gradients(self):
        """backward 后 all-reduce D 梯度，替代 DDP 包装。"""
        if not self._is_distributed():
            return
        for p in self._disc.trainable_parameters():
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

    def update(self, comp_rgb, edited, loss_cfg):
        """D step: BCE + R1 → backward → sync → clip → step。"""
        self._disc.train()
        for p in self._disc.trainable_parameters():
            p.requires_grad = True

        real_d = edited.detach().requires_grad_(True)
        d_real = self._disc(real_d)
        d_fake = self._disc(comp_rgb.detach())
        d_loss = bce_d_loss(d_real, d_fake)

        r1 = r1_gradient_penalty(self._disc, real_d)
        d_loss = d_loss + (loss_cfg.gan_r1_gamma / 2) * r1

        self._opt.zero_grad()
        d_loss.backward()
        self._sync_gradients()
        torch.nn.utils.clip_grad_norm_(self._disc.trainable_parameters(), 1.0)
        self._opt.step()
        self._step += 1
        return d_loss.detach(), r1.detach()

    def g_loss(self, comp_rgb):
        """G loss: D eval mode，返回 BCE generator loss。"""
        self._disc.eval()
        for p in self._disc.trainable_parameters():
            p.requires_grad = False
        return bce_g_loss(self._disc(comp_rgb))

    def save(self, path):
        torch.save({
            "disc": self._disc.state_dict(),
            "opt": self._opt.state_dict(),
            "step": self._step,
        }, path)

    def load(self, path):
        sd = torch.load(path, map_location="cpu")
        self._disc.load_state_dict(sd["disc"])
        self._opt.load_state_dict(sd["opt"])
        self._step = sd["step"]

    def cleanup(self):
        del self._disc, self._opt
