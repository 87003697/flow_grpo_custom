#!/usr/bin/env python3
"""调试 Direct3D-S2 SDE 步长与 log_prob 数值稳定性。"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
    Direct3DS2PipelineWithLogProb,
    SlatSamplerParams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Direct3D-S2 SDE 步长差值")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="用于调试的图像路径（默认取 eval3d_direct3d/images 第一张）",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=None,
        help="Direct3D-S2 预训练管线所在目录（默认 utils 推断）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Stage2 采样步数",
    )
    return parser.parse_args()


def pick_default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    if args.pipeline is not None:
        pipeline_path = args.pipeline
    else:
        pipeline_path = repo_root / "pretrained_weights" / "direct3d_s2-v-1-1"
    if args.image is not None:
        image_path = args.image
    else:
        image_dir = repo_root / "dataset" / "eval3d_direct3d" / "images"
        candidates = sorted(image_dir.glob("*"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"未在 {image_dir} 找到图像")
        image_path = candidates[0]
    return pipeline_path, image_path


def load_pipeline(pipeline_path: Path) -> Direct3DS2PipelineWithLogProb:
    pipe = Direct3DS2PipelineWithLogProb.from_pretrained(
        str(pipeline_path),
        subfolder="direct3d-s2-v-1-1",
        dtype=torch.float16,
        minimal_512_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device=str(device))
    return pipe


def prepare_stage1_entries(
    pipeline: Direct3DS2PipelineWithLogProb,
    image_pil: Image.Image,
    num_inference_steps: int,
    guidance_scale: float,
) -> dict:
    coords = pipeline.forward_stage1(
        image=image_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=None,
    ).int()  # 形状: (N, 4)
    cond_batch, neg_batch = pipeline.prepare_image_conditions([image_pil])
    entry = {
        "cond": cond_batch[0:1],  # 形状: (1, P, C)
        "neg_cond": (neg_batch[0:1] if neg_batch is not None else None),  # 形状: (1, P, C) 或 None
        "coords": coords,
        "image_path": "debug-image",
    }
    return entry


def analyze_sigma_differences(
    pipeline: Direct3DS2PipelineWithLogProb,
    latents_seq: list,
    t_seq: np.ndarray,
    sigma_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_t_list = []  # 形状: (steps,)
    sigma_prev_list = []  # 形状: (steps,)
    step_std_list = []  # 形状: (steps,)
    for j in range(len(t_seq) - 1):
        t_cur = float(t_seq[j])  # 标量
        t_prev = float(t_seq[j + 1])  # 标量
        t_norm = np.clip(t_cur / 1000.0, 0.0, 1.0)  # 标量
        t_prev_norm = np.clip(t_prev / 1000.0, 0.0, 1.0)  # 标量
        sigma_t = sigma_min + (1.0 - sigma_min) * t_norm  # 标量
        sigma_prev = sigma_min + (1.0 - sigma_min) * t_prev_norm  # 标量
        dt_sigma = sigma_prev - sigma_t  # 标量 (≤0)
        one_minus_sigma = max(1.0 - sigma_t, 1e-8)  # 标量
        std_dev_t = np.sqrt(sigma_t / one_minus_sigma) * 0.7  # 标量
        step_std = std_dev_t * np.sqrt(max(-dt_sigma, 1e-12))  # 标量
        sigma_t_list.append(sigma_t)
        sigma_prev_list.append(sigma_prev)
        step_std_list.append(step_std)
    return (
        np.asarray(sigma_t_list, dtype=np.float64),
        np.asarray(sigma_prev_list, dtype=np.float64),
        np.asarray(step_std_list, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    pipeline_path, image_path = pick_default_paths(args)

    print(f"✅ 使用模型目录: {pipeline_path}")
    print(f"✅ 使用图像: {image_path}")

    pipeline = load_pipeline(pipeline_path)
    pipeline_device = pipeline.device

    image_pil = Image.open(image_path).convert("RGB")
    stage1_entry = prepare_stage1_entries(
        pipeline,
        image_pil,
        num_inference_steps=50,
        guidance_scale=0.0,
    )

    sampler_params = SlatSamplerParams(
        sigma_min=0.002,
        rescale_t=0.5,
        mc_threshold=0.2,
        use_sde=True,
    )

    meshes, all_latents, all_log_probs, _ = pipeline.stage2_with_logprob(
        num_inference_steps=int(args.num_steps),
        guidance_scale=0.0,
        generator=None,
        deterministic=False,
        slat_sampler_params=sampler_params,
        stage1_cond_dict=[stage1_entry],
        num_candidates=1,
        verbose=False,
    )

    steps = int(args.num_steps)
    latents_seq = all_latents[: steps + 1]  # 长度: steps + 1
    log_probs = torch.stack(all_log_probs[:steps])  # 形状: (steps,)
    t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000  # 形状: (steps+1,)
    t_seq = float(sampler_params.rescale_t) * t_seq / (
        1 + (float(sampler_params.rescale_t) - 1) * t_seq / 1000
    )  # 形状: (steps+1,)

    sigma_t, sigma_prev, step_std = analyze_sigma_differences(
        pipeline,
        latents_seq,
        t_seq,
        sampler_params.sigma_min,
    )

    print("===== 步长统计 =====")
    print(f"sigma_t 最小值: {sigma_t.min():.6e}, 最大值: {sigma_t.max():.6e}")
    print(f"sigma_prev 与 sigma_t 平均差值: {np.mean(sigma_prev - sigma_t):.6e}")
    print(f"step_std 最小值: {step_std.min():.6e}, 最大值: {step_std.max():.6e}")
    print(f"log_prob 是否有限: {torch.isfinite(log_probs).all().item()}")
    non_finite_idx = (~torch.isfinite(log_probs)).nonzero(as_tuple=False).flatten()
    if non_finite_idx.numel() > 0:
        idx = int(non_finite_idx[0].item())
        print(f"⚠️ 第 {idx} 步出现非有限 log_prob, step_std={step_std[idx]:.6e}, Δsigma={(sigma_prev[idx]-sigma_t[idx]):.6e}")
    else:
        print("✅ 所有 log_prob 均为有限值")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

