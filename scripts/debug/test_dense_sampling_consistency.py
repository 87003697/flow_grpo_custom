#!/usr/bin/env python3
"""
Dense 采样一致性测试脚本。

验证 edit4shape/generators/trellis2 的 dense (structure) 采样接口
与 _reference_codes/TRELLIS.2 参考实现产生完全相同的结果。

测试项目：
  1. Scheduler 时间步一致性（FlowEulerScheduler.ss_scheduler vs 参考实现）
  2. 单步 velocity 预测一致性（ss_sampling_step vs 参考模型直接调用）
  3. CFG 一致性（trellis2_cfg_dense vs 参考实现的 CFG mixin）
  4. 完整 dense rollout 一致性（手动 Euler 循环 vs pipe.sample_sparse_structure）
  5. Binarize structure 一致性（binarize_structure vs 参考 decode 逻辑）

用法:
  python scripts/debug/test_dense_sampling_consistency.py \
    --model_path ./pretrained_weights/TRELLIS.2-4B \
    --dino_local_path ./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# ============================================================
# Path Setup
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
TRELLIS2_REF = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS.2")
for p in [PROJECT_ROOT, TRELLIS2_REF]:
    if p not in sys.path:
        sys.path.insert(0, p)

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers.flow_euler import (
    FlowEulerSampler,
    FlowEulerGuidanceIntervalSampler,
)
from edit4shape.generators.trellis2.pipeline_adapter import Trellis2RefAdapter
from edit4shape.generators.trellis2.scheduler import FlowEulerScheduler
from edit4shape.generators.trellis2.rollout.base import (
    _dense_pred_to_xstart,
    _dense_xstart_to_pred,
    trellis2_cfg_dense,
)
from PIL import Image


# ============================================================
# 工具函数
# ============================================================

def seed_everything(seed: int):
    """设置所有随机种子。"""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_close(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """检查两个 tensor 是否一致，打印结果。"""
    if a.shape != b.shape:
        print(f"  ❌ {name}: 形状不匹配 {a.shape} vs {b.shape}")
        return False
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    is_close = torch.allclose(a, b, rtol=rtol, atol=atol)
    status = "✅" if is_close else "❌"
    print(f"  {status} {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
          f"shape={list(a.shape)}")
    return is_close


# ============================================================
# 测试 1: Scheduler 时间步一致性
# ============================================================

def test_scheduler_timesteps(adapter: Trellis2RefAdapter, device: torch.device):
    """对比我们的 ss_scheduler 和参考实现的时间步序列。"""
    print("\n" + "=" * 60)
    print("[Test 1] Scheduler 时间步一致性")
    print("=" * 60)

    ss_params = adapter.get_ss_params()
    steps = int(ss_params["steps"])
    rescale_t = float(ss_params["rescale_t"])

    # 参考实现的时间步计算
    t_seq_ref = np.linspace(1, 0, steps + 1)
    t_seq_ref = rescale_t * t_seq_ref / (1 + (rescale_t - 1) * t_seq_ref)

    # 我们的 scheduler
    sched = adapter.ss_scheduler()
    sched.set_timesteps(steps, device)

    t_seq_ours = sched._timesteps_np

    max_diff = np.max(np.abs(t_seq_ref - t_seq_ours))
    print(f"  Steps: {steps}, rescale_t: {rescale_t}")
    print(f"  参考: {t_seq_ref[:5]}...")
    print(f"  我们: {t_seq_ours[:5]}...")
    is_close = np.allclose(t_seq_ref, t_seq_ours, atol=1e-15)
    status = "✅" if is_close else "❌"
    print(f"  {status} 时间步 max_diff={max_diff:.2e}")
    return is_close


# ============================================================
# 测试 2: 单步 Velocity 预测一致性
# ============================================================

def test_single_step_velocity(
    adapter: Trellis2RefAdapter,
    pipe: Trellis2ImageTo3DPipeline,
    cond: torch.Tensor,
    device: torch.device,
):
    """对比 ss_sampling_step 与直接调用模型的结果。"""
    print("\n" + "=" * 60)
    print("[Test 2] 单步 Velocity 预测一致性")
    print("=" * 60)

    flow_model = pipe.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels

    # 随机输入
    seed_everything(123)
    x_t = torch.randn(1, in_channels, reso, reso, reso, device=device)  # (1, C, R, R, R)
    t_val = 0.75

    # 方式 1: 直接调用模型（参考实现方式）
    t_scaled_ref = torch.tensor(
        [1000 * t_val], device=device, dtype=torch.float32
    )  # (1,)
    with torch.no_grad():
        v_ref = flow_model(x_t, t_scaled_ref, cond)  # (1, C, R, R, R)

    # 方式 2: 通过 adapter.ss_sampling_step
    with torch.no_grad():
        v_ours = adapter.ss_sampling_step(x_t, t_val, cond)  # (1, C, R, R, R)

    return check_close("单步 velocity", v_ref, v_ours, atol=1e-6)


# ============================================================
# 测试 3: Dense CFG 一致性
# ============================================================

def test_dense_cfg(
    adapter: Trellis2RefAdapter,
    pipe: Trellis2ImageTo3DPipeline,
    device: torch.device,
):
    """对比 trellis2_cfg_dense 与参考实现的 CFG 逻辑。"""
    print("\n" + "=" * 60)
    print("[Test 3] Dense CFG 一致性")
    print("=" * 60)

    flow_model = pipe.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels
    sigma_min = adapter.get_ss_sigma_min()
    guidance_rescale = adapter.get_ss_guidance_rescale()
    ss_params = adapter.get_ss_params()
    guidance_strength = float(ss_params["guidance_strength"])

    print(f"  sigma_min: {sigma_min}")
    print(f"  guidance_strength: {guidance_strength}")
    print(f"  guidance_rescale: {guidance_rescale}")

    # 随机输入
    seed_everything(456)
    x_t = torch.randn(1, in_channels, reso, reso, reso, device=device)  # (1, C, R, R, R)
    pred_pos = torch.randn_like(x_t)  # (1, C, R, R, R)
    pred_neg = torch.randn_like(x_t)  # (1, C, R, R, R)
    t_val = 0.8

    # ---- 参考实现 CFG ----
    # 复制自 ClassifierFreeGuidanceSamplerMixin._inference_model
    ref_sampler = pipe.sparse_structure_sampler
    pred_cfg_ref = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg  # (1, C, R, R, R)
    if guidance_rescale > 0:
        x_0_pos = ref_sampler._pred_to_xstart(x_t, t_val, pred_pos)  # (1, C, R, R, R)
        x_0_cfg = ref_sampler._pred_to_xstart(x_t, t_val, pred_cfg_ref)  # (1, C, R, R, R)
        std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)  # (1, 1, 1, 1, 1)
        std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)  # (1, 1, 1, 1, 1)
        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)  # (1, C, R, R, R)
        x_0_ref = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg  # (1, C, R, R, R)
        pred_cfg_ref = ref_sampler._xstart_to_pred(x_t, t_val, x_0_ref)  # (1, C, R, R, R)

    # ---- 我们的 CFG ----
    pred_cfg_ours = trellis2_cfg_dense(
        cond_pred=pred_pos,
        uncond_pred=pred_neg,
        guidance_strength=guidance_strength,
        guidance_rescale=guidance_rescale,
        x_t=x_t,
        t=t_val,
        sigma_min=sigma_min,
    )  # (1, C, R, R, R)

    ok1 = check_close("CFG 结果", pred_cfg_ref, pred_cfg_ours, atol=1e-6)

    # ---- 测试 pred_to_xstart / xstart_to_pred 对称性 ----
    x0_ours = _dense_pred_to_xstart(x_t, t_val, pred_pos, sigma_min)  # (1, C, R, R, R)
    x0_ref = ref_sampler._pred_to_xstart(x_t, t_val, pred_pos)  # (1, C, R, R, R)
    ok2 = check_close("pred_to_xstart", x0_ref, x0_ours, atol=1e-7)

    pred_back_ours = _dense_xstart_to_pred(x_t, t_val, x0_ours, sigma_min)  # (1, C, R, R, R)
    pred_back_ref = ref_sampler._xstart_to_pred(x_t, t_val, x0_ref)  # (1, C, R, R, R)
    ok3 = check_close("xstart_to_pred 往返", pred_back_ref, pred_back_ours, atol=1e-7)

    return ok1 and ok2 and ok3


# ============================================================
# 测试 4: 完整 Dense Rollout 一致性
# ============================================================

def test_full_dense_rollout(
    adapter: Trellis2RefAdapter,
    pipe: Trellis2ImageTo3DPipeline,
    cond_dict: dict,
    ss_resolution: int,
    seed: int,
    device: torch.device,
):
    """
    对比：
      A) 参考实现 pipe.sample_sparse_structure（直接调用内部 sampler）
      B) 我们用 adapter 的 ss_scheduler + ss_sampling_step 手动循环

    验证得到的 z_s 完全一致。
    """
    print("\n" + "=" * 60)
    print("[Test 4] 完整 Dense Rollout 一致性")
    print("=" * 60)

    flow_model = pipe.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels
    ss_params = adapter.get_ss_params()
    steps = int(ss_params["steps"])
    guidance_strength = float(ss_params["guidance_strength"])
    guidance_rescale = adapter.get_ss_guidance_rescale()
    cfg_interval = adapter.get_ss_cfg_interval()
    sigma_min = adapter.get_ss_sigma_min()

    print(f"  Model resolution: {reso}")
    print(f"  In channels: {in_channels}")
    print(f"  Steps: {steps}")
    print(f"  Guidance: {guidance_strength}, rescale: {guidance_rescale}")
    print(f"  CFG interval: {cfg_interval}")
    print(f"  SS resolution: {ss_resolution}")
    print(f"  Sigma min: {sigma_min}")

    # ---- A) 参考实现 ----
    print("\n  [A] 运行参考实现 sample_sparse_structure...")
    torch.manual_seed(seed)
    noise_ref = torch.randn(1, in_channels, reso, reso, reso).to(device)  # (1, C, R, R, R)
    sampler_params = {**ss_params}

    if pipe.low_vram:
        flow_model.to(device)
    z_s_ref = pipe.sparse_structure_sampler.sample(
        flow_model, noise_ref, **cond_dict, **sampler_params,
        verbose=True, tqdm_desc="[REF] Sampling structure",
    ).samples  # (1, C, R, R, R)
    if pipe.low_vram:
        flow_model.cpu()
    print(f"  参考 z_s: shape={list(z_s_ref.shape)}, "
          f"mean={z_s_ref.mean().item():.6f}, std={z_s_ref.std().item():.6f}")

    # ---- B) 我们的手动 Euler 循环 ----
    print("\n  [B] 运行我们的 ss_scheduler + ss_sampling_step 循环...")
    torch.manual_seed(seed)
    noise_ours = torch.randn(1, in_channels, reso, reso, reso).to(device)  # (1, C, R, R, R)

    # 验证噪声一致
    assert torch.equal(noise_ref, noise_ours), "噪声不一致！"
    print(f"  ✅ 初始噪声一致")

    cond = cond_dict["cond"]  # (B, S, C)
    neg_cond = cond_dict["neg_cond"]  # (B, S, C)

    sched = adapter.ss_scheduler()
    sched.set_timesteps(steps, device)

    sample = noise_ours.clone()  # (1, C, R, R, R)

    if pipe.low_vram:
        flow_model.to(device)

    step_indices = sched.get_timesteps_for_loop()
    for idx in step_indices:
        t_val = sched.get_precise_t(idx)
        t_prev_val = sched.get_precise_t(idx + 1)

        # 判断是否使用 CFG（对齐 GuidanceIntervalSamplerMixin）
        use_cfg = cfg_interval[0] <= t_val <= cfg_interval[1]

        if use_cfg and guidance_strength != 1.0:
            # 分别计算条件和无条件 velocity
            with torch.no_grad():
                v_cond = adapter.ss_sampling_step(sample, t_val, cond)  # (1, C, R, R, R)
                v_uncond = adapter.ss_sampling_step(sample, t_val, neg_cond)  # (1, C, R, R, R)

            # CFG + rescale
            v_pred = trellis2_cfg_dense(
                cond_pred=v_cond,
                uncond_pred=v_uncond,
                guidance_strength=guidance_strength,
                guidance_rescale=guidance_rescale,
                x_t=sample,
                t=t_val,
                sigma_min=sigma_min,
            )  # (1, C, R, R, R)
        else:
            # 不使用 CFG，仅条件预测
            with torch.no_grad():
                v_pred = adapter.ss_sampling_step(sample, t_val, cond)  # (1, C, R, R, R)

        # Euler 步进
        out = sched.step_dense_by_index(v_pred, idx, sample)
        sample = out.prev_sample  # (1, C, R, R, R)

    if pipe.low_vram:
        flow_model.cpu()

    z_s_ours = sample  # (1, C, R, R, R)
    print(f"  我们 z_s: shape={list(z_s_ours.shape)}, "
          f"mean={z_s_ours.mean().item():.6f}, std={z_s_ours.std().item():.6f}")

    ok = check_close("z_s (structure latent)", z_s_ref, z_s_ours, atol=1e-5)
    return ok


# ============================================================
# 测试 5: Decode Structure 一致性
# ============================================================

def test_binarize_structure(
    adapter: Trellis2RefAdapter,
    pipe: Trellis2ImageTo3DPipeline,
    z_s: torch.Tensor,
    ss_resolution: int,
    device: torch.device,
):
    """对比 adapter.binarize_structure 与参考实现的解码逻辑。"""
    print("\n" + "=" * 60)
    print("[Test 5] Binarize Structure 一致性")
    print("=" * 60)

    # ---- 参考实现 ----
    decoder = pipe.models['sparse_structure_decoder']
    if pipe.low_vram:
        decoder.to(device)
    decoded_ref = decoder(z_s) > 0  # (B, 1, D, D, D)
    if pipe.low_vram:
        decoder.cpu()
    if ss_resolution != decoded_ref.shape[2]:
        ratio = decoded_ref.shape[2] // ss_resolution  # ()
        decoded_ref = torch.nn.functional.max_pool3d(
            decoded_ref.float(), ratio, ratio, 0
        ) > 0.5  # (B, 1, R, R, R)
    coords_ref = torch.argwhere(decoded_ref)[:, [0, 2, 3, 4]].int()  # (T, 4)

    # ---- 我们的实现 ----
    coords_ours = adapter.binarize_structure(z_s, ss_resolution)  # (T, 4)

    print(f"  参考 coords: shape={list(coords_ref.shape)}")
    print(f"  我们 coords: shape={list(coords_ours.shape)}")

    if coords_ref.shape != coords_ours.shape:
        print(f"  ❌ 形状不匹配")
        return False

    ok = torch.equal(coords_ref, coords_ours)
    status = "✅" if ok else "❌"
    print(f"  {status} Coords 完全一致: {ok}")
    return ok


# ============================================================
# 测试 6: dense_sampling_with_latent 端到端一致性
# ============================================================

def test_dense_sampling_with_latent(
    adapter: Trellis2RefAdapter,
    pipe: Trellis2ImageTo3DPipeline,
    cond_dict: dict,
    ss_resolution: int,
    seed: int,
    device: torch.device,
):
    """
    对比 adapter.dense_sampling_with_latent 与参考实现。
    
    注意：dense_sampling_with_latent 内部调用 torch.manual_seed(seed)，
    所以需要用相同的 seed 调用参考实现来对比。
    """
    print("\n" + "=" * 60)
    print("[Test 6] dense_sampling_with_latent 端到端一致性")
    print("=" * 60)

    # ---- 参考实现 ----
    flow_model = pipe.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels

    torch.manual_seed(seed)
    noise = torch.randn(1, in_channels, reso, reso, reso).to(device)  # (1, C, R, R, R)
    sampler_params = {**pipe.sparse_structure_sampler_params}

    if pipe.low_vram:
        flow_model.to(device)
    z_s_ref = pipe.sparse_structure_sampler.sample(
        flow_model, noise, **cond_dict, **sampler_params,
        verbose=True, tqdm_desc="[REF] sample_structure",
    ).samples  # (1, C, R, R, R)
    if pipe.low_vram:
        flow_model.cpu()

    # 参考 decode
    decoder = pipe.models['sparse_structure_decoder']
    if pipe.low_vram:
        decoder.to(device)
    decoded = decoder(z_s_ref) > 0
    if pipe.low_vram:
        decoder.cpu()
    if ss_resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // ss_resolution
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords_ref = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # (T, 4)

    # ---- 我们的实现 ----
    z_s_ours, coords_ours = adapter.dense_sampling_with_latent(
        cond_dict, ss_resolution, seed,
    )

    print(f"  参考 z_s: mean={z_s_ref.mean().item():.6f}, std={z_s_ref.std().item():.6f}")
    print(f"  我们 z_s: mean={z_s_ours.mean().item():.6f}, std={z_s_ours.std().item():.6f}")

    ok1 = check_close("z_s (latent)", z_s_ref, z_s_ours, atol=1e-6)

    if coords_ref.shape == coords_ours.shape:
        ok2 = torch.equal(coords_ref, coords_ours)
        status = "✅" if ok2 else "❌"
        print(f"  {status} Coords 完全一致: {ok2}")
    else:
        print(f"  ❌ Coords 形状不匹配: {coords_ref.shape} vs {coords_ours.shape}")
        ok2 = False

    return ok1 and ok2


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense 采样一致性测试"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="TRELLIS.2 预训练模型路径",
    )
    parser.add_argument(
        "--dino_local_path", type=str, default=None,
        help="DINOv3 本地模型路径（可选）",
    )
    parser.add_argument(
        "--pipeline_type", type=str, default="1024",
        choices=["512", "1024", "1024_cascade"],
        help="Pipeline 类型 (default: 1024)",
    )
    parser.add_argument(
        "--input_image", type=str, default=None,
        help="输入图像路径（可选，未指定则使用随机条件）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low_vram", action="store_true",
        help="低显存模式",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Dense 采样一致性测试")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Pipeline: {args.pipeline_type}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # 1. 加载 pipeline
    print("\n加载 Pipeline...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        args.model_path, dino_local_path=args.dino_local_path,
    )
    pipe.low_vram = args.low_vram
    pipe.to(device)
    adapter = Trellis2RefAdapter(pipe, pipeline_type=args.pipeline_type)
    print(f"  加载完成: {time.time() - t0:.1f}s")

    # 2. 获取条件编码
    print("\n准备条件编码...")
    ss_resolution_map = {"512": 32, "1024": 64, "1024_cascade": 32}
    ss_resolution = ss_resolution_map[args.pipeline_type]

    if args.input_image is not None:
        image = Image.open(args.input_image).convert("RGB")
        image = pipe.preprocess_image(image)
        cond_dict = pipe.get_cond([image], 512)
    else:
        # 使用随机条件
        print("  未指定输入图像，使用随机条件编码")
        seed_everything(args.seed + 1000)
        cond_dict = {
            "cond": torch.randn(1, 257, 1024, device=device),
            "neg_cond": torch.zeros(1, 257, 1024, device=device),
        }

    cond = cond_dict["cond"].to(device)  # (B, S, C)
    neg_cond = cond_dict["neg_cond"].to(device)  # (B, S, C)
    print(f"  cond: {cond.shape}, neg_cond: {neg_cond.shape}")

    # 3. 运行各项测试
    results = {}

    results["scheduler"] = test_scheduler_timesteps(adapter, device)

    results["single_step"] = test_single_step_velocity(
        adapter, pipe, cond, device
    )

    results["cfg"] = test_dense_cfg(adapter, pipe, device)

    results["full_rollout"] = test_full_dense_rollout(
        adapter, pipe, cond_dict, ss_resolution, args.seed, device,
    )

    # 使用 full rollout 得到的 z_s 来测试 decode
    # 先获取一个 z_s
    torch.manual_seed(args.seed)
    flow_model = pipe.models['sparse_structure_flow_model']
    noise = torch.randn(
        1, flow_model.in_channels, flow_model.resolution,
        flow_model.resolution, flow_model.resolution
    ).to(device)  # (1, C, R, R, R)
    ss_params = {**pipe.sparse_structure_sampler_params}
    if pipe.low_vram:
        flow_model.to(device)
    z_s_for_decode = pipe.sparse_structure_sampler.sample(
        flow_model, noise, **cond_dict, **ss_params,
        verbose=False, tqdm_desc="Getting z_s for decode test",
    ).samples  # (1, C, R, R, R)
    if pipe.low_vram:
        flow_model.cpu()

    results["binarize"] = test_binarize_structure(
        adapter, pipe, z_s_for_decode, ss_resolution, device,
    )

    results["e2e"] = test_dense_sampling_with_latent(
        adapter, pipe, cond_dict, ss_resolution, args.seed, device,
    )

    # 4. 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("🎉 所有测试通过！Dense 采样与参考实现完全一致。")
    else:
        print("⚠️  部分测试未通过，请检查上方详情。")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
