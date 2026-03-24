#!/usr/bin/env python3
"""
验证 rollout_dense + decode_to_coords 与原生 sample_sparse_structure 输出一致性。

原理：两条路径使用相同的初始噪声和条件编码，应产生完全相同的 dense latent z_s。
  路径 A (原生): 直接调用 sparse_structure_sampler.sample → z_s → decoder → coords
  路径 B (我们): 手动 ODE 循环（同 rollout_dense）→ z_s → decode_to_coords → coords

关键：共享同一个噪声张量，消除 CPU/GPU RNG 差异。

比较内容：
  1. z_s latent 的数值差异
  2. 最终 coords 是否完全一致

示例:
  python scripts/debug/test_dense_rollout_vs_original.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --seed 42
"""

import os
import sys
import argparse

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS")
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

from edit4shape.generators.trellis.pipeline_adapter import (
    TrellisRefAdapter,
    build_pipeline_from_reference,
)
from edit4shape.generators.trellis.rollout.base import (
    prepare_embeddings,
    predict_dense_velocity_with_cfg,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="TRELLIS 预训练模型目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# =====================================================================
# 路径 A: 原生 sampler.sample → z_s → decoder → coords
# =====================================================================

def run_original_sampler(pipe, noise_gpu, cond_dict, device):
    """
    直接调用原生 sampler 和 decoder，使用给定的噪声张量。
    等价于 pipe.sample_sparse_structure()，但噪声从外部传入。

    Returns:
        z_s_original: (1, C, R, R, R) dense latent
        coords_original: (T, 4) int32 coords
    """
    flow_model = pipe.models['sparse_structure_flow_model']
    sampler = pipe.sparse_structure_sampler
    sampler_params = {**pipe.sparse_structure_sampler_params}

    z_s = sampler.sample(
        flow_model,
        noise_gpu,
        **cond_dict,
        **sampler_params,
        verbose=True,
    ).samples  # (1, C, R, R, R)

    decoder = pipe.models['sparse_structure_decoder']
    coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

    return z_s, coords


# =====================================================================
# 路径 B: 手动 ODE 循环（同 rollout_dense），使用给定的噪声张量
# =====================================================================

def run_our_rollout(adapter, noise_gpu, cond_emb, uncond_emb, device):
    """
    手动展开 rollout_dense 核心逻辑，不依赖 system/state。
    噪声从外部传入，不调用 init_latents。

    Returns:
        z_s_ours: (1, C, R, R, R) dense latent
        coords_ours: (T, 4) int32 coords
    """
    ss_steps, ss_guidance, ss_rescale_t, ss_cfg_min, ss_cfg_max = adapter.dense.get_runtime_params()

    x_t = noise_gpu.clone()

    # 时间步序列
    _, t_pairs = adapter.dense.scheduler(ss_steps, ss_rescale_t)

    # ODE 去噪循环
    with torch.no_grad():
        for t, t_prev in t_pairs:
            t_val = float(t)
            velocity = predict_dense_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                ss_guidance, ss_cfg_min, ss_cfg_max, device,
            )  # (1, C, R, R, R)
            delta = t_val - float(t_prev)
            x_t = x_t - delta * velocity

    z_s_ours = x_t  # (1, C, R, R, R)

    # decode_to_coords
    with torch.no_grad():
        coords_ours = adapter.dense.decode_to_coords(z_s_ours, batch_size=1)  # (T, 4)

    return z_s_ours, coords_ours


# =====================================================================
# 比较函数
# =====================================================================

def compare_results(z_s_a, coords_a, z_s_b, coords_b):
    """比较两条路径的结果。"""
    print("\n" + "=" * 60)
    print("  Dense Latent z_s 比较")
    print("=" * 60)
    print(f"  Shape A: {z_s_a.shape}, Shape B: {z_s_b.shape}")

    diff = (z_s_a - z_s_b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (z_s_a.abs() + 1e-8)).mean().item()

    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")
    print(f"  Mean rel diff: {rel_diff:.2e}")

    z_s_match = max_diff < 1e-4
    print(f"  z_s 一致: {'✅ YES' if z_s_match else '❌ NO'}")

    print("\n" + "=" * 60)
    print("  Coords 比较")
    print("=" * 60)
    print(f"  Count A: {coords_a.shape[0]}, Count B: {coords_b.shape[0]}")

    if coords_a.shape == coords_b.shape:
        coords_match = torch.equal(coords_a.cpu(), coords_b.cpu())
        if not coords_match:
            diff_count = (coords_a.cpu() != coords_b.cpu()).any(dim=1).sum().item()
            print(f"  不同坐标数: {diff_count}")
        print(f"  Coords 完全一致: {'✅ YES' if coords_match else '❌ NO'}")
    else:
        print(f"  Coords 数量不同! A={coords_a.shape[0]} vs B={coords_b.shape[0]}")

        # 计算交集
        set_a = set(tuple(c.tolist()) for c in coords_a.cpu())
        set_b = set(tuple(c.tolist()) for c in coords_b.cpu())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        iou = intersection / union if union > 0 else 0
        print(f"  IoU: {iou:.4f} ({intersection}/{union})")
        coords_match = False

    print("\n" + "=" * 60)
    if z_s_match and coords_match:
        print("  🎉 两条路径输出完全一致！")
    elif z_s_match:
        print("  ⚠️ z_s 一致但 coords 不一致（边界 voxel 翻转）")
    else:
        print("  ❌ z_s 数值有差异")
    print("=" * 60)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- 加载模型 ----
    print(f"[INFO] 加载模型: {args.model_path}")
    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = args.model_path
    cfg.verbose = True

    class MockAccelerator:
        pass
    mock_acc = MockAccelerator()
    mock_acc.device = device

    adapter = build_pipeline_from_reference(cfg, mock_acc, device=device)

    # ---- 读取图像 & 条件编码 ----
    print(f"[INFO] 读取图像: {args.image}")
    img = Image.open(args.image).convert("RGB")
    cond_dict = adapter.prepare_image_conditions([img])
    cond_emb = cond_dict["cond"].to(device)      # (1, P, C)
    uncond_emb = cond_dict["neg_cond"].to(device)  # (1, P, C)

    # ---- 生成共享噪声（CPU → GPU，与原生一致）----
    torch.manual_seed(args.seed)
    flow_model = adapter.pipe.models['sparse_structure_flow_model']
    noise_cpu = torch.randn(1, flow_model.in_channels,
                            flow_model.resolution, flow_model.resolution,
                            flow_model.resolution)
    noise_gpu = noise_cpu.to(device)
    print(f"[INFO] 共享噪声: shape={noise_gpu.shape}, seed={args.seed}")

    # ---- 路径 A: 原生 sampler ----
    print("\n[INFO] 路径 A: 原生 sampler.sample + decoder")
    with torch.no_grad():
        z_s_a, coords_a = run_original_sampler(adapter.pipe, noise_gpu, cond_dict, device)
    print(f"  z_s shape: {z_s_a.shape}, coords count: {coords_a.shape[0]}")

    # ---- 路径 B: 我们的 rollout_dense + decode_to_coords ----
    print("\n[INFO] 路径 B: rollout_dense + decode_to_coords")
    with torch.no_grad():
        z_s_b, coords_b = run_our_rollout(adapter, noise_gpu, cond_emb, uncond_emb, device)
    print(f"  z_s shape: {z_s_b.shape}, coords count: {coords_b.shape[0]}")

    # ---- 比较 ----
    compare_results(z_s_a, coords_a, z_s_b, coords_b)


if __name__ == "__main__":
    main()
