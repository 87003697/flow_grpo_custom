"""Level 2: 单 GPU smoke test — 用真实 QwenImage Transformer 验证完整 D/G step。

运行 (debug pod):
  cd /data/work/run_codes
  PYTHONPATH=/data/work/run_codes:/data/work/run_codes/_reference_codes/TRELLIS \
    /local-ssd/flow_grpo_venv/uv-venv/bin/python scripts/test_latent_gan_smoke.py
"""
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class MockLossCfg:
    gan_t_d_mean = -0.6
    gan_t_d_std = 1.0
    gan_hook_block_ids = [10, 20, 40, 59]
    gan_head_channels = 384
    gan_r1_gamma = 0.0
    class gan_opt:
        type = "adam"; lr = 2e-4; beta1 = 0.0; beta2 = 0.99
        eps = 1e-8; weight_decay = 0.0


def test_full_d_g_step(pipe, device='cuda'):
    """用真实 transformer 跑 D step + G step。

    验证:
      1. D step 输出 finite d_loss
      2. G step 输出 finite g_loss
      3. G step 梯度回传到 comp_rgb
      4. 显存使用合理
    """
    from edit4shape.guidance.latent_discriminator import LatentDiscriminatorHelper

    helper = LatentDiscriminatorHelper(MockLossCfg(), pipe, device)

    # --- 参数量统计 ---
    n_params = sum(p.numel() for p in helper._disc.heads.parameters())
    print(f"  Head params: {n_params/1e6:.1f}M")

    B = 2
    comp_rgb = torch.randn(B, 3, 512, 512, device=device, requires_grad=True)
    edited = torch.randn(B, 3, 512, 512, device=device)
    prompt_embeds = torch.randn(1, 77, 3584, device=device, dtype=torch.bfloat16)
    prompt_mask = torch.ones(1, 77, device=device, dtype=torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()

    # --- D step (no gradient checkpointing) ---
    print("  Running D step...")
    with helper.enabled():
        d_loss, r1_val = helper.update(
            comp_rgb, edited, MockLossCfg(),
            prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
        )
    assert torch.isfinite(d_loss), f"d_loss non-finite: {d_loss}"
    d_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ D step: d_loss={d_loss.item():.4f}, r1={r1_val.item():.4f}, peak={d_mem:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    # --- G step (with gradient checkpointing) ---
    print("  Running G step...")
    with helper.enabled():
        with helper.g_enabled():
            g_loss = helper.g_loss(
                comp_rgb,
                prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
            )
    assert torch.isfinite(g_loss), f"g_loss non-finite: {g_loss}"
    g_loss.backward()
    assert comp_rgb.grad is not None and comp_rgb.grad.abs().sum() > 0
    g_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"✓ G step: g_loss={g_loss.item():.4f}, grad_norm={comp_rgb.grad.norm():.6f}, peak={g_mem:.2f} GB")

    helper.cleanup()
    torch.cuda.empty_cache()
    print("\n=== Level 2 smoke test passed ===")


if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HOME", "/local-ssd/pretrained_weights/hf_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline

    model_path = "Qwen/Qwen-Image-Edit-2511"
    device = "cuda"

    print(f"Loading pipeline from {model_path} (HF_HOME={os.environ['HF_HOME']})...")
    pipe = FlowEditFullPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"Pipeline loaded. Transformer blocks: {len(pipe.transformer.transformer_blocks)}")
    print(f"inner_dim: {pipe.transformer.inner_dim}")

    test_full_d_g_step(pipe, device)
