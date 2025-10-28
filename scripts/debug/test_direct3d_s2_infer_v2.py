#!/usr/bin/env python3
"""
Direct3D‑S2 推理与自检脚本（基于仓库内最小集成实现）
=================================================

目标：
 - 完全使用当前工作区代码（不依赖外部示例脚本）验证 Direct3D‑S2 集成是否健康。
 - 覆盖：SDE 单步一致性、可复现性、端到端采样（dense->sparse1024）、log_prob / 轨迹长度 / 形状校验、网格导出。
 - 支持 ODE 与 SDE 两种模式（通过 --no_sde 切换）。

与旧脚本 (`test_direct3d_s2_stage1_minimal.py`) 的改进：
 - 种子复现测试针对完整候选采样（含多步 & 多候选）。
 - 验证 `all_latents` / `all_log_probs` 数量与期望匹配。
 - 提供更清晰的统计输出（log_prob 分位数、noise_strength 范围，如可获得）。
 - Mesh 导出统一到指定目录，并打印顶点 / 面数量。
 - 使用严格断言；若任何一步失败退出非零码。

限制：
 - 仅测试 sparse1024（1024-only）。
 - 不修改 pipeline 内部逻辑（例如 candidate 内部 seed 派生方式），只在外层控制主生成器种子。

使用示例：
  python scripts/debug/test_direct3d_s2_infer.py \
    --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/direct3d_s2_validation \
    --candidates 2 --dense_steps 50 --sparse_steps 30 --guidance 7.0 \
    --sigma_min 0.002 --rescale_t 1000.0 --seed 777 --dtype fp16 --do_e2e

"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Any

import torch


# 仅使用仓库内实现（注意：pipeline 含 direct3d_s2.* 依赖，会触发 udf_ext 需求；延迟到需要时再导入）
from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import (
    direct3d_flow_step_with_logprob,
    SparseTensor,
    compute_log_prob_direct3d_stage2,
    Stage2RuntimeConfig,
    extract_sparse_tensor_from_batch,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import SlatSamplerParams


# ------------------------------
# 单步 SDE 数学一致性测试
# ------------------------------
def test_sde_step_logprob_consistency(device: torch.device) -> None:
    """验证 direct3d_flow_step_with_logprob 返回的 SDE 统计与公式一致。"""
    coords = torch.zeros((13, 4), device=device, dtype=torch.int32)  # (N=13,4)
    layout = [slice(0, 13)]

    prev_mean_feats = torch.zeros((13, 6), device=device, dtype=torch.float32)
    sample = SparseTensor(coords=coords, feats=prev_mean_feats, layout=layout)
    model_output = SparseTensor(coords=coords, feats=torch.zeros_like(prev_mean_feats), layout=layout)

    rescale_t = 1000.0
    sigma_min = 0.002
    g = torch.Generator(device=device)
    g.manual_seed(20240916)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.sigma_min = sigma_min  # 形状: 标量
    scheduler.rescale_t = rescale_t  # 形状: 标量
    scheduler.set_timesteps(30, device=device)  # 形状: (T=30,)
    t_cur = float(scheduler.timesteps[0].item())   # 形状: 标量
    t_prev = float(scheduler.timesteps[1].item())  # 形状: 标量
    prev_sample, log_prob_vec, prev_mean, std_vec = direct3d_flow_step_with_logprob(
        scheduler=scheduler,
        sample=sample,
        model_output=model_output,
        timestep=t_cur,
        prev_timestep=t_prev,
        generator=g,
        deterministic=False,
    )

    diff = prev_sample.feats - prev_mean.feats
    step_std = std_vec[0]
    D = diff.numel()

    lp_x = -0.5 * (
        diff.pow(2).sum() / (step_std ** 2)
        + D * math.log(2 * math.pi)
        + 2 * D * math.log(float(step_std))
    )
    eps_hat = diff / step_std
    lp_eps_hat = -0.5 * (eps_hat.pow(2).sum() + D * math.log(2 * math.pi))

    lp_eps_hat_mean = lp_eps_hat / D
    lp_x_mean = lp_x / D

    assert torch.allclose(log_prob_vec[0], lp_eps_hat_mean - math.log(float(step_std)), atol=1e-5)
    const_expected = math.log(float(step_std))
    diff_val = (log_prob_vec[0] - (lp_eps_hat_mean - const_expected)).item()
    assert abs(diff_val) < 1e-4


# ------------------------------
# 端到端采样与验证
# ------------------------------
@dataclass
class InferConfig:
    """参考：无官方对应（脚本局部配置结构）。"""
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
    sigma_min: float
    rescale_t: float
    use_sde: bool
    minimal_512: bool
    use_refiner: bool
    dtype: str
    do_e2e: bool
    deterministic: bool
    check_grpo_policy: bool


def _ensure_grpo3d_env():
    """参考：无官方对应（环境检查与 CUDA 扩展加载）。"""
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name != "grpo3d":
        print(f"[WARN] 当前激活环境 '{env_name}' 不是期望的 'grpo3d'，可能缺少 udf_ext。")
    import udf_ext  # noqa: F401
    print("[OK] udf_ext 已成功导入 (CUDA 扩展可用)")


def build_pipeline(cfg: InferConfig):
    """参考：
    - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:68-172`（from_pretrained）
    - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:54-66`（to）
    """
    # 延迟导入，确保已激活正确的 conda 环境并可加载 udf_ext
    from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
        Direct3DS2PipelineWithLogProb,  # noqa
    )
    _ensure_grpo3d_env()
    if cfg.minimal_512:
        os.environ["DIRECT3D_S2_MINIMAL_512"] = "1"
    # 解析 dtype
    if cfg.dtype == "fp32":
        _dtype = torch.float32
    elif cfg.dtype in ("fp16", "half"):
        _dtype = torch.float16
    elif cfg.dtype == "bf16":
        _dtype = torch.bfloat16
    elif cfg.dtype == "fp8":
        raise NotImplementedError("fp8 暂不支持：算子与库兼容性不足")
    else:
        raise ValueError(f"不支持的 dtype: {cfg.dtype}")

    pipe = Direct3DS2PipelineWithLogProb.from_pretrained(
        cfg.pipeline_path,
        minimal_512_only=cfg.minimal_512,
        dtype=_dtype,
        use_refiner=bool(cfg.use_refiner),
    )
    pipe.to(cfg.device)
    return pipe


def run_sampling(
    pipe, cfg: InferConfig, generator: torch.Generator, coords_override: torch.Tensor | None = None
) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[dict]]:
    """参考：
    - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:253-291`（采样循环）
    - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:320-341`（解码与后处理）
    - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:359-363`（__call__ 中 sort_block 用法）
    """
    # 将现有配置映射为 trellis 风格调用
    slat_sampler_params = SlatSamplerParams(
        mc_threshold=float(cfg.mc_threshold), # 标量
        use_sde=bool(cfg.use_sde),            # 标量
    )
    # 构造 Stage1 条件与 K 候选合批
    if coords_override is None:
        stage1_cond_dict = _build_stage1_for_image(
            pipe=pipe,
            image=cfg.image,
            dense_steps=int(cfg.dense_steps),
            guidance=float(cfg.guidance),
            generator=generator,
            k=int(cfg.num_candidates),
        )
    else:
        # 复用与 _build_stage1_for_image 相同的合批逻辑
        do_cfg = (float(cfg.guidance) > 0.0)
        cond, neg = pipe.prepare_image_conditions(cfg.image, do_classifier_free_guidance=do_cfg)
        latent_coords = coords_override  # 形状 (N,4)
        from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
        k = int(cfg.num_candidates)
        sparse_list = [
            sp.SparseTensor(
                feats=torch.empty((latent_coords.shape[0], 1), device=latent_coords.device),
                coords=latent_coords.to(dtype=torch.int64),
                layout=[slice(0, latent_coords.shape[0])],
            )
            for _ in range(k)
        ]
        coords_batched = sp.prepare_sparse_tensor_batch(sparse_list, batch_size=k)
        cond_b = cond.repeat_interleave(k, dim=0)
        neg_b = (None if (neg is None) else neg.repeat_interleave(k, dim=0))
        stage1_cond_dict = {
            "cond": cond_b,
            "neg_cond": neg_b,
            "coords": coords_batched,
        }

    meshes, latents_seq_flat, step_log_probs_flat, step_t_seq = pipe.stage2_with_logprob(
        stage1_cond_dict=stage1_cond_dict,
        slat_sampler_params=SlatSamplerParams(mc_threshold=float(cfg.mc_threshold), use_sde=bool(cfg.use_sde)),
        num_inference_steps=int(cfg.sparse_steps),
        guidance_scale=float(cfg.guidance),
        generator=generator,
        deterministic=bool(cfg.deterministic),
    )
    # 释放不必要的 GPU 张量占用（仅保留必要返回的标量log_prob与mesh）
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meshes, latents_seq_flat, step_log_probs_flat, step_t_seq, [stage1_cond_dict]


# ------------------------------
# 单步 SDE vs ODE 差异分析辅助
# ------------------------------
def compare_single_step(pipe, cfg: InferConfig, stage1_entry: dict, t_index: int | None = None) -> None:
    """比较 direct3d_flow_step_with_logprob 与 scheduler 的稠密实现。"""
    from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
    from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import direct3d_flow_step_with_logprob
    from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import sparse_tensor_cfg_guidance

    sched = pipe.ref.sparse_scheduler_512
    sparse_dit_module = pipe._resolve_sparse_dit_module()

    # 适配 batched 稀疏：提取第0个候选，并将 cond/neg 裁成 (1,P,C)
    cond_full = stage1_entry["cond"].to(device=pipe.device, dtype=pipe.dtype)
    cond = cond_full[0:1]
    uncond_full = stage1_entry.get("neg_cond")
    uncond = None if (uncond_full is None) else uncond_full.to(device=pipe.device, dtype=pipe.dtype)[0:1]

    coords_any = stage1_entry["coords"]
    if hasattr(coords_any, "coords") and hasattr(coords_any, "layout"):
        # SparseTensor 批：提取第0个候选
        from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import extract_sparse_tensor_from_batch
        single_sp = extract_sparse_tensor_from_batch(coords_any.to(pipe.device), 0)
        coords_int = single_sp.coords.int()
    else:
        coords_int = coords_any.to(pipe.device).int()

    sched.set_timesteps(int(cfg.sparse_steps), device=pipe.device)
    timesteps = sched.timesteps  # (T,)
    num_pairs = int(len(timesteps) - 1)
    if num_pairs <= 0:
        print("[compare] timesteps 长度不足，跳过比较")
        return

    indices = range(num_pairs) if t_index is None else [t_index]
    rand_gen = torch.Generator(device=pipe.device)
    rand_gen.manual_seed(int(cfg.seed))

    for idx in indices:
        if idx < 0 or idx >= num_pairs:
            print(f"[compare] t_index {idx} 超出范围 {num_pairs-1}")
            continue

        t = timesteps[idx].item()
        t_prev = timesteps[idx + 1].item()

        latent_shape = (int(coords_int.shape[0]), int(sparse_dit_module.out_channels))
        latents = torch.randn(latent_shape, dtype=pipe.dtype, device=pipe.device, generator=rand_gen)

        x_sp = sp.SparseTensor(latents, coords_int)
        t_tensor = latents.new_tensor([t])
        noise_cond = sparse_dit_module(x_sp, t_tensor, cond)
        if uncond is not None:
            noise_uncond = sparse_dit_module(x_sp, t_tensor, uncond)
            model_output_sparse = sparse_tensor_cfg_guidance(
                positive_sparse=noise_cond,
                negative_sparse=noise_uncond,
                guidance_scale=float(cfg.guidance),
            )
        else:
            model_output_sparse = noise_cond

        # 复用与 t/t_prev 同源的调度器 sched，确保 index_for_timestep 精确匹配
        sde_prev, sde_log_prob, sde_mean, sde_std = direct3d_flow_step_with_logprob(
            scheduler=sched,
            sample=x_sp,
            model_output=model_output_sparse,
            timestep=float(t),
            prev_timestep=float(t_prev),
            generator=rand_gen,
            deterministic=False,
        )

        ode_prev, ode_log_prob, ode_mean, ode_std = direct3d_flow_step_with_logprob(
            scheduler=sched,
            sample=x_sp,
            model_output=model_output_sparse,
            timestep=float(t),
            prev_timestep=float(t_prev),
            generator=None,
            deterministic=True,
        )

        print(f"[compare] step index: {idx}/{num_pairs-1}")
        print("  t -> t_prev:", t, "->", t_prev)
        print("  SDE prev_mean vs ODE prev_mean (max abs):", (sde_mean.feats - ode_mean.feats).abs().max().item())
        print("  SDE prev_sample vs ODE prev_sample (max abs):", (sde_prev.feats - ode_prev.feats).abs().max().item())
        print("  SDE std vs ODE std (max abs):", (sde_std - ode_std).abs().max().item())
        print("  SDE log_prob vs ODE log_prob (abs diff):", (sde_log_prob - ode_log_prob).abs().max().item())


# （已移除）：脚本内 direct3d_s2_stage2_with_logprob，统一直接调用 pipe.stage2_with_logprob


def _as_tensor_2d(lp):
    return lp if isinstance(lp, torch.Tensor) else torch.stack(lp)


def _build_stage1_for_image(pipe, image, dense_steps, guidance, generator, k):
    do_cfg = (float(guidance) > 0.0)
    cond, neg = pipe.prepare_image_conditions(image, do_classifier_free_guidance=do_cfg)
    coords_list = pipe.forward_stage1(images=[image], num_inference_steps=int(dense_steps), guidance_scale=float(guidance), generator=generator)
    coords = coords_list[0]
    from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
    sparse_list = [
        sp.SparseTensor(
            feats=torch.empty((coords.shape[0], 1), device=coords.device),  # 形状: (N,1)
            coords=coords,  # 形状: (N,4)（保持 forward_stage1 返回 dtype/顺序）
        )
        for _ in range(int(k))
    ]
    coords_batched = sp.prepare_sparse_tensor_batch(sparse_list, batch_size=len(sparse_list))  # 形状: 稀疏(批)，候选级 layout
    cond_b = cond.repeat_interleave(int(k), dim=0)
    neg_b = (None if (neg is None) else neg.repeat_interleave(int(k), dim=0))
    return {"cond": cond_b, "neg_cond": neg_b, "coords": coords_batched}


def validate_sampling_outputs(
    meshes: List[Any],
    latents_seq_flat: List[torch.Tensor],
    step_log_probs_flat,
    cfg: InferConfig,
) -> None:
    """参考：无官方对应（脚本校验逻辑）。"""
    # 期望 log_prob 条目数 = candidates * sparse_steps
    # 允许传入 list[Tensor] 或 Tensor(T, BK)
    lp_tensor = _as_tensor_2d(step_log_probs_flat)

    expected_steps = max(0, int(cfg.sparse_steps) - 1)
    expected_bk = int(cfg.num_candidates)
    assert lp_tensor.shape[0] == expected_steps, f"log_prob 步数 {lp_tensor.shape[0]} != 期望 {expected_steps}"
    actual_bk = 1 if lp_tensor.ndim == 1 else int(lp_tensor.shape[1])
    assert actual_bk == expected_bk, f"log_prob BK 维 {actual_bk} != 期望 {expected_bk}"

    if cfg.use_sde:
        non_zero = (lp_tensor.abs() > 0).sum().item()
        assert non_zero > 0, "SDE 模式 log_prob 全为 0"
    else:
        assert float(lp_tensor.abs().sum().item()) == 0.0, "ODE 模式不应产生 log_prob"

    # latent 序列长度：当前返回的是按时间步聚合的 batched 稀疏（长度=steps）
    if len(latents_seq_flat) > 0:
        expected_len = int(cfg.sparse_steps)
        assert len(latents_seq_flat) == expected_len, f"latents 序列长度 {len(latents_seq_flat)} != 期望 {expected_len}"

    # Mesh 非空校验（仅检查前三个）
    for i, m in enumerate(meshes[:3]):
        v = getattr(m, "vertices", None)
        f = getattr(m, "faces", None)
        if v is not None and f is not None:
            if torch.is_tensor(v):
                assert v.numel() > 0, f"mesh[{i}] 顶点为空"
            else:
                assert len(v) > 0, f"mesh[{i}] 顶点为空"


def export_meshes(meshes: List[Any], out_dir: str, pipeline_obj: Any = None, mc_threshold: float = 0.2) -> None:
    """参考：无官方对应（导出与可选解码辅助）。"""
    os.makedirs(out_dir, exist_ok=True)
    for i, mesh in enumerate(meshes):
        print(f"[DEBUG] Export candidate {i}: type={type(mesh)} attrs={dir(mesh)[:15]}")
        # 如果是 SparseTensor（尚未 decode）并且提供了 pipeline，可尝试 decode
        if type(mesh).__name__ == 'SparseTensor' and pipeline_obj is not None:
            feats = getattr(mesh, 'feats', None)
            coords = getattr(mesh, 'indices', None) or getattr(mesh, 'coords', None)
            if feats is not None and coords is not None:
                decoded = pipeline_obj._decode_sparse_mesh(feats, coords, mc_threshold=mc_threshold, remove_interior=False)
                mesh = decoded
                print("[DEBUG] SparseTensor 通过 pipeline._decode_sparse_mesh 解码，转换为 KiuiMesh")
        # 如果 mesh 是 dict
        if isinstance(mesh, dict):
            if 'mesh' in mesh:
                mesh = mesh['mesh']
            else:
                print(f"[WARN] dict mesh 缺少 'mesh' 键，跳过 {i}")
                continue
        # 直接支持 KiuiMesh.write 接口
        if hasattr(mesh, "write"):
            out_path = os.path.join(out_dir, f"mesh_{i}.ply")
            mesh.write(out_path)
            print(f"[DEBUG] Saved via KiuiMesh.write -> {out_path}")
            continue
        v = getattr(mesh, "vertices", None)
        f = getattr(mesh, "faces", None)
        if v is None or f is None:
            # 尝试 export 接口
            if hasattr(mesh, "export"):
                out_path = os.path.join(out_dir, f"mesh_{i}.ply")
                mesh.export(out_path)
                print(f"[DEBUG] Saved via mesh.export -> {out_path}")
            continue
        if torch.is_tensor(v):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = v
        if torch.is_tensor(f):
            f_np = f.detach().cpu().numpy()
        else:
            f_np = f
        import trimesh  # 延迟导入

        tri = trimesh.Trimesh(vertices=v_np, faces=f_np)
        out_path = os.path.join(out_dir, f"mesh_{i}.ply")
        tri.export(out_path)
        print(f"[DEBUG] Saved via trimesh.Trimesh -> {out_path}")


def reproducibility_check(pipe: Any, cfg: InferConfig) -> None:
    """基于固定 coords 的严格复现性：两次运行使用同一 coords，应得到相同的 log_prob 序列。"""
    # 先生成一次 coords
    gen_a = torch.Generator(device=torch.device(cfg.device)); gen_a.manual_seed(cfg.seed)
    coords_list = pipe.forward_stage1(
        images=[cfg.image],
        num_inference_steps=int(cfg.dense_steps),
        guidance_scale=float(cfg.guidance),
        generator=gen_a,
    )  # 列表长度1
    coords = coords_list[0]  # (N,4)

    # Phase A：使用 coords 采样
    g1 = torch.Generator(device=torch.device(cfg.device)); g1.manual_seed(cfg.seed)
    run1 = run_sampling(pipe, cfg, g1, coords_override=coords)

    # Phase B：相同 coords 再采样
    g2 = torch.Generator(device=torch.device(cfg.device)); g2.manual_seed(cfg.seed)
    run2 = run_sampling(pipe, cfg, g2, coords_override=coords)

    lp1 = _as_tensor_2d(run1[2]).float().view(-1).cpu()
    lp2 = _as_tensor_2d(run2[2]).float().view(-1).cpu()
    if lp1.shape != lp2.shape:
        raise AssertionError(f"log_prob shape mismatch {lp1.shape} vs {lp2.shape}")
    max_abs = (lp1 - lp2).abs().max().item()
    if not torch.allclose(lp1, lp2, atol=1e-6, rtol=1e-6):
        diff_idx = (lp1 != lp2).nonzero(as_tuple=False).view(-1)[:10]
        print(f"[REPRO][ERR] log_prob mismatch max_abs={max_abs:.6g} indices_sample={diff_idx.tolist()}")
        print("lp1 sample:", lp1[diff_idx].tolist())
        print("lp2 sample:", lp2[diff_idx].tolist())
        raise AssertionError("同 coords 下 log_prob 不一致")
    if cfg.use_sde:
        print(f"[REPRO] log_prob match (max_abs={max_abs:.2e}) under fixed coords reuse")


def summarize_logprob(step_log_probs_flat) -> str:
    """参考：无官方对应（统计摘要）。支持 Tensor 或 list[Tensor]。"""
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


def check_grpo_policy_sampling(
    pipe: Any,
    cfg: InferConfig,
    latents_seq_flat: List[SparseTensor],
    step_log_probs_flat: Any,
    step_t_seq: torch.Tensor,
    stage1_entry: dict,
) -> None:
    """对齐 GRPO 训练的策略采样校验：
    - 使用 observed_prev_sample 逐步复算 log_prob，并与采样时记录的 log_prob 对比。
    """
    # 统一 log_prob 张量为 (steps, BK)
    lp_tensor = _as_tensor_2d(step_log_probs_flat)  # 形状: (T, BK)

    # 读取 BK、时间步数、条件张量
    cond_full = stage1_entry["cond"].to(device=pipe.device, dtype=pipe.dtype)  # 形状: (BK, P, C)
    neg_full = stage1_entry.get("neg_cond")
    neg_full = None if (neg_full is None) else neg_full.to(device=pipe.device, dtype=pipe.dtype)  # 形状: (BK, P, C) 或 None
    BK = int(cond_full.shape[0])  # 形状: 标量
    T = int(lp_tensor.shape[0])  # 形状: 标量

    # 构造 samples 列表（长度 BK），每个含 per-candidate 的稀疏时间序列
    samples: List[dict] = []  # 形状: 长度 BK
    t_seq_fp32 = step_t_seq.to(dtype=torch.float32).detach().cpu()  # 形状: (T+1,)

    for k in range(BK):
        lat_seq_k: List[SparseTensor] = []  # 形状: 长度 T+1
        for j in range(T + 1):
            sp_batched = latents_seq_flat[j].to(pipe.device)  # 形状: 稀疏(批)
            sp_k = extract_sparse_tensor_from_batch(sp_batched, k)  # 形状: 稀疏(单)
            lat_seq_k.append(sp_k)  # 形状: 追加 1 个稀疏
        sample_k = {
            "latents_seq": lat_seq_k,  # 形状: [T+1] 稀疏
            "cond_patches": cond_full[k : k + 1],  # 形状: (1, P, C)
            "neg_patches": (None if neg_full is None else neg_full[k : k + 1]),  # 形状: (1, P, C) 或 None
            "t_seq": t_seq_fp32,  # 形状: (T+1,)
        }
        samples.append(sample_k)  # 形状: 追加 1 个样本

    # 逐步复算 log_prob（与训练一致：observed_prev_sample）
    rt_cfg = Stage2RuntimeConfig(guidance_scale=float(cfg.guidance), deterministic=bool(cfg.deterministic))  # 形状: 标量配置
    all_diff: List[torch.Tensor] = []  # 形状: 长度 T，元素 (BK,)

    ref_rows: List[torch.Tensor] = []  # 形状: 长度 T，元素 (BK,)
    obs_rows: List[torch.Tensor] = []  # 形状: 长度 T，元素 (BK,)

    for j in range(T):
        _, lp_vec_obs, _ = compute_log_prob_direct3d_stage2(
            pipeline=pipe,
            samples=samples,
            j=int(j),
            config=rt_cfg,
        )  # 形状: (BK,)
        lp_vec_ref = lp_tensor[j].to(device=lp_vec_obs.device, dtype=lp_vec_obs.dtype)  # 形状: (BK,)
        diff = (lp_vec_obs - lp_vec_ref).abs()  # 形状: (BK,)
        all_diff.append(diff)  # 形状: 追加 (BK,)
        ref_rows.append(lp_vec_ref.detach().cpu())
        obs_rows.append(lp_vec_obs.detach().cpu())

    diffs = torch.stack(all_diff, dim=0) if len(all_diff) > 0 else torch.zeros((0, BK))  # 形状: (T, BK)
    max_abs = (diffs.max().item() if diffs.numel() > 0 else 0.0)  # 形状: 标量
    mean_abs = (diffs.mean().item() if diffs.numel() > 0 else 0.0)  # 形状: 标量

    # 断言一致性：SDE/ODE 两种模式都应匹配（ODE 为全 0）
    tol = 1e-4  # 形状: 标量
    if not torch.all(diffs <= tol):  # 形状: 标量布尔
        # 输出前若干不一致样本以定位
        bad_idx = (diffs > tol).nonzero(as_tuple=False)  # 形状: (M, 2)
        print(f"[GRPO][ERR] 复算与采样 log_prob 不一致：max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} count={bad_idx.shape[0]}")
        for rr in bad_idx[:10]:
            t_i = int(rr[0].item()); k_i = int(rr[1].item())  # 形状: 标量
            obs_val = ref_rows[t_i][k_i].item()  # 形状: 标量
            rec_val = obs_rows[t_i][k_i].item()  # 形状: 标量
            print(f"  step={t_i} k={k_i} | sample_lp={obs_val:.6f} recompute_lp={rec_val:.6f} | abs_diff={abs(rec_val-obs_val):.3e}")
        raise AssertionError("GRPO 策略采样 log_prob 复算校验失败")

    print(f"[GRPO][OK] 策略采样 log_prob 复算一致：max_abs={max_abs:.2e} mean_abs={mean_abs:.2e}")


def parse_args() -> InferConfig:
    """参考：无官方对应（参数解析）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_path", type=str, required=True, help="包含 config.yaml 与权重的目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径 (文件) 或原路径传递让 pipeline 内部处理")
    ap.add_argument("--out", type=str, default="outputs/test_runs/direct3d_s2_validation")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--dense_steps", type=int, default=50)
    ap.add_argument("--sparse_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--mc_threshold", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--sigma_min", type=float, default=0.)
    ap.add_argument("--rescale_t", type=float, default=1000.0)
    # 默认 ODE；--use_sde 显式开启
    ap.add_argument("--no_sde", dest="no_sde", action="store_true", help="关闭 SDE (默认)")
    ap.add_argument("--use_sde", dest="no_sde", action="store_false", help="启用 SDE")
    ap.set_defaults(no_sde=True)
    ap.add_argument("--minimal_512", action="store_true", help="仅加载 dense + sparse512")
    ap.add_argument("--use_refiner", action="store_true", help="启用 refiner")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "half", "bf16", "fp8"], help="主 dtype")
    ap.add_argument("--do_e2e", action="store_true", help="执行端到端采样")
    ap.add_argument("--deterministic", action="store_true", help="启用后端确定性 (cuDNN)")
    ap.add_argument("--check_grpo_policy", action="store_true", help="复算 GRPO 策略采样 log_prob 并对比采样记录")
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
        sigma_min=args.sigma_min,
        rescale_t=args.rescale_t,
        use_sde=not args.no_sde,
        minimal_512=args.minimal_512,
        use_refiner=args.use_refiner,
        dtype=args.dtype,
        do_e2e=bool(args.do_e2e),
        deterministic=bool(args.deterministic),
        check_grpo_policy=bool(args.check_grpo_policy),
    )


def main() -> None:
    """参考：无官方对应（脚本主入口）。"""
    cfg = parse_args()
    device = torch.device(cfg.device)

    # 1) 单步 SDE 数学一致性
    test_sde_step_logprob_consistency(device)
    print("[OK] SDE 单步 log_prob 一致性通过")

    # 2) 构建管线（若未执行 e2e，可提前结束）
    if not cfg.do_e2e:
        print("[SKIP] 未指定 --do_e2e，结束（基础 SDE 测试已通过）")
        return

    assert os.path.isdir(cfg.pipeline_path), f"pipeline_path 不存在: {cfg.pipeline_path}"
    assert os.path.isfile(cfg.image), f"image 文件不存在: {cfg.image}"

    pipe = build_pipeline(cfg)
    if cfg.dense_steps <= 0 or cfg.sparse_steps <= 0:
        raise ValueError("dense_steps 与 sparse_steps 必须为正整数")
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        print("[INFO] 启用确定性后端设置")

    # 3) 端到端首次采样
    main_gen = torch.Generator(device=device)
    main_gen.manual_seed(cfg.seed)
    meshes, latents_seq_flat, step_log_probs_flat, t_seq_out, stage1_entries = run_sampling(
        pipe, cfg, main_gen
    )

    validate_sampling_outputs(meshes, latents_seq_flat, step_log_probs_flat, cfg)

    # 4) 可复现性（基于 log_prob 序列严格比较）
    reproducibility_check(pipe, cfg)
    print("[OK] 可复现性（同种子）验证通过")

    # 4.5) 单步差异对比（可选）
    compare_single_step(pipe, cfg, stage1_entries[0], t_index=None)

    # 4.6) GRPO 策略采样复算校验（可选）
    if bool(cfg.check_grpo_policy):
        # 统一 step_log_probs_flat / t_seq 形状并执行校验
        stage1_entry = stage1_entries[0]  # 形状: 字典（BK 批）
        check_grpo_policy_sampling(
            pipe=pipe,
            cfg=cfg,
            latents_seq_flat=latents_seq_flat,  # 形状: 长度 T+1
            step_log_probs_flat=step_log_probs_flat,  # 形状: (T, BK) 或 List[(BK,)]
            step_t_seq=t_seq_out,  # 形状: (T+1,)
            stage1_entry=stage1_entry,
        )

    # 5) 统计输出
    stats = summarize_logprob(step_log_probs_flat)
    print(f"[Stats] log_prob: {stats}")

    # 6) 导出网格
    export_meshes(meshes, cfg.out_dir, pipeline_obj=pipe, mc_threshold=cfg.mc_threshold)
    print(f"[OK] 导出 {len(meshes)} 个 mesh -> {cfg.out_dir}")

    print("[DONE] Direct3D‑S2 集成推理测试全部通过")


if __name__ == "__main__":
    main()
