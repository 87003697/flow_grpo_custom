"""Latent Discriminator 单元测试 — 验证 shape + gradient flow。

避免完整 import chain（trellis 等），直接测试核心组件。
运行: PYTHONPATH=/data/work/run_codes python scripts/test_latent_discriminator.py
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn

# Patch: avoid pulling in trellis through training_adpter.py
# We only need _build_optimizer which uses timm internally
import types
_fake_training_adpter = types.ModuleType("edit4shape.generators.trellis.training_adpter")
def _build_optimizer(params, opt_cfg):
    from timm.optim import create_optimizer_v2
    return create_optimizer_v2(
        params,
        opt=getattr(opt_cfg, 'type', 'adam'),
        lr=getattr(opt_cfg, 'lr', 1e-4),
        betas=(getattr(opt_cfg, 'beta1', 0.9), getattr(opt_cfg, 'beta2', 0.999)),
        eps=getattr(opt_cfg, 'eps', 1e-8),
        weight_decay=getattr(opt_cfg, 'weight_decay', 0.0),
    )
_fake_training_adpter._build_optimizer = _build_optimizer
sys.modules["edit4shape.generators.trellis.training_adpter"] = _fake_training_adpter
sys.modules.setdefault("edit4shape.generators", types.ModuleType("edit4shape.generators"))
sys.modules.setdefault("edit4shape.generators.trellis", types.ModuleType("edit4shape.generators.trellis"))


def test_make_latent_head():
    """验证 head 的 shape 变换: [B, C_in, N] → [B, 1, N]"""
    from edit4shape.guidance.latent_discriminator import _make_latent_head
    head = _make_latent_head(in_channels=3072, head_channels=384)
    x = torch.randn(4, 3072, 64)
    out = head(x)
    assert out.shape == (4, 1, 64), f"Expected (4,1,64), got {out.shape}"
    out.sum().backward()
    print("✓ _make_latent_head: shape + grad OK")


def test_qwen_image_discriminator_with_mock():
    """用 mock transformer 验证 hook + head 逻辑。"""
    from edit4shape.guidance.latent_discriminator import QwenImageDiscriminator

    class MockBlock(nn.Module):
        def forward(self, hidden_states, **kwargs):
            text = torch.randn_like(hidden_states)
            return text, hidden_states  # (text, image) — QwenImage 格式

    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_dim = 3072
            self.transformer_blocks = nn.ModuleList([MockBlock() for _ in range(4)])
            self.img_in = nn.Linear(64, 3072)
            self.gradient_checkpointing = False

        def forward(self, hidden_states, **kwargs):
            hidden_states = self.img_in(hidden_states)
            for block in self.transformer_blocks:
                _, hidden_states = block(hidden_states)
            return hidden_states

    transformer = MockTransformer()
    disc = QwenImageDiscriminator(transformer, hook_block_ids=[0, 2, 3], head_channels=384)

    B, N = 2, 16
    z_t = torch.randn(B, N, 64, requires_grad=True)
    timestep = torch.zeros(B)
    prompt = torch.randn(B, 5, 3072)
    mask = torch.ones(B, 5)
    img_shapes = [[(1, 4, 4)]] * B

    logits = disc(z_t, timestep, prompt, mask, img_shapes)
    assert logits.shape == (B, 3 * N), f"Expected ({B}, {3*N}), got {logits.shape}"

    logits.sum().backward()
    assert z_t.grad is not None and z_t.grad.abs().sum() > 0
    print(f"✓ QwenImageDiscriminator: output {logits.shape}, grad flows to input")


def test_renoise_shape():
    """验证 _sample_d_timestep: shape 不变, t ∈ [0,1]"""
    from edit4shape.guidance.latent_discriminator import _sample_d_timestep
    B = 4
    t = _sample_d_timestep(B, mean=-0.6, std=1.0, device='cpu')
    assert t.shape == (B,)
    assert (t >= 0).all() and (t <= 1).all(), f"t out of range: {t}"

    z0 = torch.randn(B, 64, 64)
    noise = torch.randn_like(z0)
    t_exp = t.view(-1, 1, 1)
    z_t = (1 - t_exp) * z0 + t_exp * noise
    assert z_t.shape == z0.shape
    print("✓ _renoise: shape preserved, t in [0,1]")


def test_base_discriminator_helper_save_load():
    """验证 checkpoint save/load version 兼容性。"""
    from edit4shape.guidance.discriminator import BaseDiscriminatorHelper

    class DummyDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(10, 1)
        def trainable_parameters(self):
            return self.head.parameters()

    class DummyHelper(BaseDiscriminatorHelper):
        def _d_logits(self, comp_rgb, edited, **kwargs):
            return self._disc.head(edited), self._disc.head(comp_rgb)
        def _g_logits(self, comp_rgb, **kwargs):
            return self._disc.head(comp_rgb)

    class OptCfg:
        type = "adam"; lr = 1e-3; beta1 = 0.9; beta2 = 0.999
        eps = 1e-8; weight_decay = 0.0

    helper = DummyHelper(DummyDisc(), OptCfg(), device='cpu')
    path = os.path.join(tempfile.mkdtemp(), "test_ckpt.pt")
    helper.save(path)

    sd = torch.load(path, map_location='cpu')
    assert sd.get("version") == 2, f"Expected version=2, got {sd.get('version')}"
    assert "step" in sd and "disc" in sd and "opt" in sd

    helper2 = DummyHelper(DummyDisc(), OptCfg(), device='cpu')
    helper2.load(path)
    assert helper2.step == helper.step
    print("✓ BaseDiscriminatorHelper: save/load with version=2 OK")


def test_hook_block_ids_sorted():
    """验证 hook_block_ids 存为 sorted tuple。"""
    from edit4shape.guidance.latent_discriminator import QwenImageDiscriminator

    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_dim = 128
            self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(10)])

    disc = QwenImageDiscriminator(MockTransformer(), hook_block_ids=[5, 1, 3], head_channels=64)
    assert disc.hook_block_ids == (1, 3, 5), f"Expected sorted, got {disc.hook_block_ids}"
    print("✓ hook_block_ids sorted")


def test_bce_losses():
    """验证 BCE loss 函数基本行为。"""
    from edit4shape.guidance.discriminator import bce_d_loss, bce_g_loss

    d_real = torch.ones(4, 16) * 5    # strong real
    d_fake = torch.ones(4, 16) * -5   # strong fake
    d_loss = bce_d_loss(d_real, d_fake)
    assert torch.isfinite(d_loss) and d_loss < 0.1, f"D should easily classify, got {d_loss}"

    d_fake_g = torch.zeros(4, 16, requires_grad=True)
    g_loss = bce_g_loss(d_fake_g)
    assert torch.isfinite(g_loss)
    g_loss.backward()
    assert d_fake_g.grad is not None
    print("✓ BCE losses: finite + grad flows")


def test_d_g_step_integration():
    """Integration test: DummyHelper completes full D step + G step."""
    from edit4shape.guidance.discriminator import BaseDiscriminatorHelper

    class LinearDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(8, 1)
        def trainable_parameters(self):
            return self.net.parameters()

    class LinearHelper(BaseDiscriminatorHelper):
        def _d_logits(self, comp_rgb, edited, **kwargs):
            return self._disc.net(edited.detach()), self._disc.net(comp_rgb.detach())
        def _g_logits(self, comp_rgb, **kwargs):
            return self._disc.net(comp_rgb)

    class OptCfg:
        type = "adam"; lr = 1e-3; beta1 = 0.9; beta2 = 0.999
        eps = 1e-8; weight_decay = 0.0

    class LossCfg:
        gan_r1_gamma = 0.0

    helper = LinearHelper(LinearDisc(), OptCfg(), device='cpu')

    comp_rgb = torch.randn(2, 8, requires_grad=True)
    edited = torch.randn(2, 8)

    # D step
    d_loss, r1 = helper.update(comp_rgb, edited, LossCfg())
    assert torch.isfinite(d_loss), f"d_loss not finite: {d_loss}"
    assert helper.step == 1

    # G step
    g_loss = helper.g_loss(comp_rgb)
    assert torch.isfinite(g_loss), f"g_loss not finite: {g_loss}"
    g_loss.backward()
    assert comp_rgb.grad is not None and comp_rgb.grad.abs().sum() > 0
    print(f"✓ D/G step integration: d_loss={d_loss.item():.4f}, g_loss={g_loss.item():.4f}, grad OK")


if __name__ == "__main__":
    test_make_latent_head()
    test_qwen_image_discriminator_with_mock()
    test_renoise_shape()
    test_base_discriminator_helper_save_load()
    test_hook_block_ids_sorted()
    test_bce_losses()
    test_d_g_step_integration()
    print("\n=== All Level 1 unit tests passed ===")
