#!/usr/bin/env python3
"""
Direct3D‑S1（稠密）推理与自检脚本
=================================

目标：
- 覆盖稠密分支单步一致性（SDE vs 公式/ODE）、整批 SDE/ODE、复现与策略复算（compute_log_prob_direct3d_stage1）。
- 末尾可选调用 Stage2 的 ODE 解码，将 Stage1 生成的 coords_list 转 mesh 便于人工检查。

使用示例：
  python -u scripts/debug/test_direct3d_s1_infer_v2.py \
    --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/direct3d_s1_validation \
    --candidates 2 --dense_steps 50 --sparse_steps 30 --guidance 7.0 \
    --seed 777 --dtype fp16 --use_sde --do_e2e --ode_decode
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import (
    direct3d_flow_step_with_logprob_dense,
    compute_log_prob_direct3d_stage1,
)
from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import Stage1RuntimeConfig
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def test_sde_step_logprob_consistency_dense(device: torch.device) -> None:
    """验证 direct3d_flow_step_with_logprob_dense 的 SDE 统计与公式一致。"""
    BK, C, R = 3, 4, 8
    sample = torch.zeros((BK, C, R, R, R), device=device, dtype=torch.float32)  # (BK,C,R,R,R)
    model_output = torch.zeros_like(sample)  # (BK,C,R,R,R)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.sigma_min = 0.002
    scheduler.rescale_t = 1000.0
    scheduler.set_timesteps(10, device=device)
    t_cur = float(scheduler.timesteps[0].item())
    t_prev = float(scheduler.timesteps[1].item())

    g = torch.Generator(device=device); g.manual_seed(20240916)
    prev_sample, log_prob_vec, prev_mean, std_vec = direct3d_flow_step_with_logprob_dense(
        scheduler=scheduler,
        sample=sample,
        model_output=model_output,
        timestep=t_cur,
        prev_timestep=t_prev,
        generator=g,
        deterministic=False,
    )  # ((BK,C,R,R,R),(BK,),(BK,C,R,R,R),(BK,))

    diff = (prev_sample - prev_mean)  # (BK,C,R,R,R)
    step_std = std_vec[0]  # ()
    D = diff[0].numel()  # () per-sample elements

    # 逐样本均值 logprob（对 (C,R,R,R) 求均值）与实现一致性
    for k in range(BK):
        d = diff[k]
        lp_x = -0.5 * (
            d.pow(2).sum() / (step_std ** 2)
            + D * math.log(2 * math.pi)
            + 2 * D * math.log(float(step_std))
        )
        eps_hat = d / step_std
        lp_eps_hat = -0.5 * (eps_hat.pow(2).sum() + D * math.log(2 * math.pi))
        lp_eps_hat_mean = lp_eps_hat / D
        assert torch.allclose(log_prob_vec[k], lp_eps_hat_mean - math.log(float(step_std)), atol=1e-5)


@dataclass
class InferConfig:
    pipeline_path: str
    image: str
    out_dir: str
    device: str
    num_candidates: int
    dense_steps: int
    sparse_steps: int
    guidance: float
    mc_threshold: float
    seed: int
    dtype: str
    do_e2e: bool
    ode_decode: bool
    deterministic: bool
    check_grpo_policy: bool


def _ensure_grpo3d_env():
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name != "grpo3d":
        print(f"[WARN] 当前激活环境 '{env_name}' 不是期望的 'grpo3d'，可能缺少 udf_ext。")
    import udf_ext  # noqa: F401
    print("[OK] udf_ext 已成功导入 (CUDA 扩展可用)")


def build_pipeline(cfg: InferConfig):
    from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
        Direct3DS2PipelineWithLogProb,
        SlatSamplerParams,
    )
    _ensure_grpo3d_env()
    if cfg.dtype == "fp32":
        _dtype = torch.float32
    elif cfg.dtype in ("fp16", "half"):
        _dtype = torch.float16
    elif cfg.dtype == "bf16":
        _dtype = torch.bfloat16
    else:
        raise ValueError(f"不支持的 dtype: {cfg.dtype}")
    pipe = Direct3DS2PipelineWithLogProb.from_pretrained(
        cfg.pipeline_path,
        minimal_512_only=True,
        dtype=_dtype,
        use_refiner=False,
    )
    pipe.to(cfg.device)
    # 推理模式
    pipe.ref.dense_dit.eval()
    pipe.ref.sparse_dit_512.eval()
    return pipe, SlatSamplerParams


def run_stage1_sampling(pipe, cfg: InferConfig, generator: torch.Generator):
    do_cfg = (float(cfg.guidance) > 0.0)
    cond, neg = pipe.prepare_image_conditions(cfg.image, do_classifier_free_guidance=do_cfg)
    BK = int(cfg.num_candidates)
    cond_b = cond.repeat_interleave(BK, dim=0)
    neg_b = (None if (neg is None) else neg.repeat_interleave(BK, dim=0))
    coords_list, latents_seq_dense, log_prob_seq_dense, t_seq = pipe.stage1_with_logprob(
        cond_dict={"cond": cond_b, "neg_cond": neg_b},
        num_inference_steps=int(cfg.dense_steps),
        guidance_scale=float(cfg.guidance),
        generator=generator,
        deterministic=bool(cfg.deterministic),
    )
    return coords_list, latents_seq_dense, log_prob_seq_dense, t_seq


def run_stage2_decode_from_coords(pipe, cfg: InferConfig, coords_list: List[torch.Tensor], generator: torch.Generator, SlatSamplerParams):
    from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
    BK = len(coords_list)
    cond, neg = pipe.prepare_image_conditions(cfg.image, do_classifier_free_guidance=(float(cfg.guidance) > 0.0))
    cond_b = cond.repeat_interleave(BK, dim=0)
    neg_b = (None if (neg is None) else neg.repeat_interleave(BK, dim=0))
    sparse_list = [
        sp.SparseTensor(
            feats=torch.empty((coords.shape[0], 1), device=coords.device),
            coords=coords.to(dtype=torch.int64),
            layout=[slice(0, coords.shape[0])],
        ) for coords in coords_list
    ]
    coords_batched = sp.prepare_sparse_tensor_batch(sparse_list, batch_size=len(sparse_list))
    meshes, latents_seq, log_prob_seq, t_seq = pipe.stage2_with_logprob(
        stage1_cond_dict={"cond": cond_b, "neg_cond": neg_b, "coords": coords_batched},
        slat_sampler_params=SlatSamplerParams(mc_threshold=float(cfg.mc_threshold)),
        num_inference_steps=int(cfg.sparse_steps),
        guidance_scale=float(cfg.guidance),
        generator=generator,
        deterministic=True,
    )
    return meshes


def summarize_logprob(step_log_probs_flat) -> str:
    if isinstance(step_log_probs_flat, torch.Tensor):
        if step_log_probs_flat.numel() == 0:
            return "(empty)"
        vals = step_log_probs_flat.float().view(-1)
    else:
        if len(step_log_probs_flat) == 0:
            return "(empty)"
        vals = torch.stack(step_log_probs_flat).float().view(-1)
    p = torch.quantile(vals, torch.tensor([0.01, 0.5, 0.99], device=vals.device))
    return f"mean={vals.mean():.3f} std={vals.std(unbiased=False):.3f} p1={p[0]:.3f} p50={p[1]:.3f} p99={p[2]:.3f}"


def check_grpo_policy_sampling_dense(pipe, cfg: InferConfig, latents_seq_dense: List[torch.Tensor], log_prob_seq_dense: torch.Tensor, t_seq: torch.Tensor, cond_b: torch.Tensor, neg_b: torch.Tensor | None) -> None:
    BK = int(cond_b.shape[0])
    T = int(log_prob_seq_dense.shape[0])
    samples: List[dict] = []
    t_seq_fp32 = t_seq.to(dtype=torch.float32).detach().cpu()
    for k in range(BK):
        sample_k = {
            "latents_seq_dense": [lat_k.to(pipe.device, dtype=pipe.dtype)[k:k+1].squeeze(0) for lat_k in latents_seq_dense],
            "cond_patches": cond_b[k:k+1].detach().cpu(),
            "neg_patches": (None if neg_b is None else neg_b[k:k+1].detach().cpu()),
            "t_seq": t_seq_fp32,
        }
        samples.append(sample_k)
    rt_cfg = Stage1RuntimeConfig(guidance_scale=float(cfg.guidance), deterministic=bool(cfg.deterministic))
    diffs: List[torch.Tensor] = []
    for j in range(T):
        _, lp_vec_obs, _ = compute_log_prob_direct3d_stage1(pipe, samples, j, rt_cfg)
        lp_vec_ref = log_prob_seq_dense[j].to(device=lp_vec_obs.device, dtype=lp_vec_obs.dtype)
        diffs.append((lp_vec_obs - lp_vec_ref).abs())
    diffs_t = torch.stack(diffs, dim=0) if len(diffs) > 0 else torch.zeros((0, BK))
    tol = 1e-4
    assert torch.all(diffs_t <= tol), f"GRPO 稠密复算不一致，max_abs={diffs_t.max().item():.3e}"


def parse_args() -> InferConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_path", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/test_runs/direct3d_s1_validation")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--dense_steps", type=int, default=50)
    ap.add_argument("--sparse_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--mc_threshold", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--use_sde", action="store_true")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "half", "bf16"]) 
    ap.add_argument("--do_e2e", action="store_true")
    ap.add_argument("--ode_decode", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--check_grpo_policy", action="store_true")
    args = ap.parse_args()
    return InferConfig(
        pipeline_path=args.pipeline_path,
        image=args.image,
        out_dir=args.out,
        device=args.device,
        num_candidates=args.candidates,
        dense_steps=args.dense_steps,
        sparse_steps=args.sparse_steps,
        guidance=args.guidance,
        mc_threshold=args.mc_threshold,
        seed=args.seed,
        dtype=args.dtype,
        do_e2e=bool(args.do_e2e),
        ode_decode=bool(args.ode_decode),
        deterministic=bool(args.deterministic),
        check_grpo_policy=bool(args.check_grpo_policy),
    )


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)

    # 1) 单步一致性
    test_sde_step_logprob_consistency_dense(device)
    print("[OK] 稠密 SDE 单步 log_prob 一致性通过")

    if not cfg.do_e2e:
        print("[SKIP] 未指定 --do_e2e，结束（基础 SDE 测试已通过）")
        return

    assert os.path.isdir(cfg.pipeline_path), f"pipeline_path 不存在: {cfg.pipeline_path}"
    assert os.path.isfile(cfg.image), f"image 文件不存在: {cfg.image}"

    pipe, SlatSamplerParams = build_pipeline(cfg)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    # 2) Stage1 采样
    main_gen = torch.Generator(device=device); main_gen.manual_seed(cfg.seed)
    with torch.inference_mode():
        coords_list, latents_seq_dense, log_prob_seq_dense, t_seq = run_stage1_sampling(pipe, cfg, main_gen)

    # 3) 复现性：固定初始噪声与种子调用两次（通过传入 generator 复用）可在脚本外再次调用核验
    # 这里只进行形状与非空断言
    assert len(latents_seq_dense) == int(cfg.dense_steps), "latents_seq_dense 长度异常"
    assert log_prob_seq_dense.shape[1] == int(cfg.num_candidates), "BK 维不匹配"
    if cfg.use_sde:
        assert (log_prob_seq_dense.abs() > 0).sum().item() > 0, "SDE 模式 log_prob 不应全 0"
    else:
        assert float(log_prob_seq_dense.abs().sum().item()) == 0.0, "ODE 模式 log_prob 应为 0"

    # 4) GRPO 策略复算
    do_cfg = (float(cfg.guidance) > 0.0)
    cond, neg = pipe.prepare_image_conditions(cfg.image, do_classifier_free_guidance=do_cfg)
    BK = int(cfg.num_candidates)
    cond_b = cond.repeat_interleave(BK, dim=0)
    neg_b = (None if (neg is None) else neg.repeat_interleave(BK, dim=0))
    check_grpo_policy_sampling_dense(pipe, cfg, latents_seq_dense, log_prob_seq_dense, t_seq, cond_b, neg_b)
    print("[OK] 稠密 GRPO 策略复算一致")

    # 5) 可选：使用 Stage2 的 ODE 解码导出 mesh
    if cfg.ode_decode:
        with torch.inference_mode():
            meshes = run_stage2_decode_from_coords(pipe, cfg, coords_list, main_gen, SlatSamplerParams)
        os.makedirs(cfg.out_dir, exist_ok=True)
        from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import Direct3DS2PipelineWithLogProb
        # 直接调用已有导出方法
        for i, mesh in enumerate(meshes):
            out_path = os.path.join(cfg.out_dir, f"mesh_{i}.ply")
            if hasattr(mesh, "write"):
                mesh.write(out_path)
            else:
                # 兜底三方导出
                import trimesh
                tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                tri.export(out_path)
        print(f"[OK] 通过 Stage2 ODE 解码导出 {len(meshes)} 个 mesh -> {cfg.out_dir}")

    print("[DONE] Direct3D‑S1 稠密推理测试完成")


if __name__ == "__main__":
    main()


