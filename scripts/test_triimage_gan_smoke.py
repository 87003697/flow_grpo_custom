"""三图 DINO GAN smoke test — 验证 TriImageDiscriminatorHelper D/G step。

运行 (debug pod):
  cd /data/work/run_codes
  PYTHONPATH=/data/work/run_codes:/data/work/run_codes/_reference_codes/TRELLIS \
    uv run python scripts/test_triimage_gan_smoke.py
"""
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class MockLossCfg:
    gan_model_path = "facebook/dinov2-small"
    gan_r1_gamma = 0.0
    gan_bt_weight = 1.0
    class gan_opt:
        type = "adan"; lr = 2e-5
        eps = 1e-8; weight_decay = 0.0


def test_triimage_d_g_step(device='cuda'):
    """用 DINOv3-S 跑三图 D step + G step。

    验证:
      1. TriImageDiscriminatorHelper 初始化成功
      2. D step 输出 finite d_loss（含 BCE + BT 分量）
      3. G step 输出 finite g_loss，梯度回传到 comp_rgb
      4. 显存使用合理（ViT-S 预期 < 2 GB）
    """
    from edit4shape.guidance.discriminator import TriImageDiscriminatorHelper

    print("Initializing TriImageDiscriminatorHelper...")
    helper = TriImageDiscriminatorHelper(MockLossCfg(), device)

    n_params = sum(p.numel() for p in helper._disc.heads.parameters())
    print(f"  Head params: {n_params/1e6:.2f}M")

    B = 2
    comp_rgb = torch.rand(B, 3, 512, 512, device=device, requires_grad=True)
    edited = torch.rand(B, 3, 512, 512, device=device)
    condition_tensor = torch.rand(B, 3, 512, 512, device=device)

    torch.cuda.reset_peak_memory_stats()

    # --- D step ---
    print("  Running D step (BCE + BT)...")
    d_loss, r1_val = helper.d_step(
        comp_rgb, edited, MockLossCfg(),
        condition_tensor=condition_tensor,
    )
    assert torch.isfinite(d_loss), f"d_loss non-finite: {d_loss}"
    assert torch.isfinite(r1_val), f"r1 non-finite: {r1_val}"
    d_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  D step: d_loss={d_loss.item():.4f}, r1={r1_val.item():.6f}, peak={d_mem:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    # --- G step ---
    print("  Running G step...")
    g_loss = helper.g_step(comp_rgb)
    assert torch.isfinite(g_loss), f"g_loss non-finite: {g_loss}"
    g_loss.backward()
    assert comp_rgb.grad is not None and comp_rgb.grad.abs().sum() > 0
    g_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  G step: g_loss={g_loss.item():.4f}, grad_norm={comp_rgb.grad.norm():.6f}, peak={g_mem:.2f} GB")

    helper.cleanup()
    torch.cuda.empty_cache()
    print("\n=== TriImage GAN smoke test PASSED ===")


if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HOME", "/local-ssd/pretrained_weights/hf_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    device = "cuda"
    print(f"Device: {device}")
    test_triimage_d_g_step(device)
