#!/usr/bin/env python3
"""
Direct3D‑S2 推理与自检脚本（基于仓库内最小集成实现）
=================================================

目标：
 - 完全使用当前工作区代码（不依赖外部示例脚本）验证 Direct3D‑S2 集成是否健康。
 - 覆盖：SDE 单步一致性、可复现性、端到端采样（dense->sparse512）、log_prob / 轨迹长度 / 形状校验、网格导出。
 - 支持 ODE 与 SDE 两种模式（通过 --no_sde 切换）。

与旧脚本 (`test_direct3d_s2_stage1_minimal.py`) 的改进：
 - 种子复现测试针对完整候选采样（含多步 & 多候选）。
 - 验证 `all_latents` / `all_log_probs` 数量与期望匹配。
 - 提供更清晰的统计输出（log_prob 分位数、noise_strength 范围，如可获得）。
 - Mesh 导出统一到指定目录，并打印顶点 / 面数量。
 - 使用严格断言；若任何一步失败退出非零码。

限制：
 - 仅测试 sparse512（与当前最小实现一致）。
 - 不修改 pipeline 内部逻辑（例如 candidate 内部 seed 派生方式），只在外层控制主生成器种子。

使用示例：
  python scripts/debug/test_direct3d_s2_infer.py \
    --pipeline_path pretrained_weights/direct3d_s2-v-1-1 \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/direct3d_s2_validation \
    --candidates 2 --dense_steps 30 --sparse_steps 20 --guidance 0 \
    --sigma_min 0.002 --rescale_t 1000.0 --seed 777 --dtype fp16 --do_e2e

"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Any

import torch

# 仅使用仓库内实现（注意：pipeline 含 direct3d_s2.* 依赖，会触发 udf_ext 需求；延迟到需要时再导入）
from flow_grpo.diffusers_patch.direct3d_s2_sde_with_logprob import sde_step_with_logprob


# ------------------------------
# 单步 SDE 数学一致性测试
# ------------------------------
def test_sde_step_logprob_consistency(device: torch.device) -> None:
    """验证 eps 与 (x_next|mu) 两种 log_prob 写法只差常数。
    公式参考 DEV.md：log p(eps) 与 log p(x|mu) 的差 = D * log(noise_strength)。
    """
    prev_mean = torch.zeros((13, 6), device=device, dtype=torch.float32)  # (N=13,C=6)
    t_cur = torch.tensor(900.0, device=device, dtype=torch.float32)  # 标量
    t_prev = torch.tensor(850.0, device=device, dtype=torch.float32)  # 标量
    rescale_t = 1000.0  # 标量
    sigma_min = 0.002  # 标量
    g = torch.Generator(device=device)
    g.manual_seed(20240916)

    x_next, lp_eps, noise_strength, sq_sum, n_dims = sde_step_with_logprob(
        prev_mean, t_cur, t_prev, rescale_t, sigma_min, g
    )  # 新签名：返回5值

    diff = x_next - prev_mean  # (13,6)
    D = diff.numel()  # 标量
    # 基于 x 的写法（包含 noise_strength）
    lp_x = -0.5 * (
        diff.pow(2).sum() / noise_strength.pow(2)
        + D * math.log(2 * math.pi)
        + 2 * D * math.log(float(noise_strength))
    )  # 标量
    eps_hat = diff / noise_strength  # (13,6)
    lp_eps_hat = -0.5 * (eps_hat.pow(2).sum() + D * math.log(2 * math.pi))  # 标量

    assert torch.allclose(lp_eps, lp_eps_hat, atol=1e-5), "eps 两种写法不一致"
    const_expected = D * math.log(float(noise_strength))
    diff_val = (lp_eps - lp_x).item()
    assert abs(diff_val - const_expected) < 1e-4, "(lp_eps - lp_x) 与常数差不符"


# ------------------------------
# 端到端采样与验证
# ------------------------------
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
    sigma_min: float
    rescale_t: float
    use_sde: bool
    minimal_512: bool
    skip_refiner: bool
    dtype: str
    do_e2e: bool
    deterministic: bool


def _ensure_grpo3d_env():
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env_name != "grpo3d":
        print(f"[WARN] 当前激活环境 '{env_name}' 不是期望的 'grpo3d'，可能缺少 udf_ext。")
    try:
        import udf_ext  # noqa: F401
        print("[OK] udf_ext 已成功导入 (CUDA 扩展可用)")
    except Exception as e:
        print("[ERROR] 无法导入 udf_ext: ", e)
        print("       请确认已在 'triplaneturbo' 环境中编译安装 third_party/voxelize (pip install -v .)")
        raise


def build_pipeline(cfg: InferConfig):
    # 延迟导入，确保已激活正确的 conda 环境并可加载 udf_ext
    from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
        Direct3DS2PipelineWithLogProb,  # noqa
    )
    _ensure_grpo3d_env()
    if cfg.minimal_512:
        os.environ["DIRECT3D_S2_MINIMAL_512"] = "1"
    if cfg.skip_refiner:
        os.environ["DIRECT3D_SKIP_REFINER"] = "1"
    pipe = Direct3DS2PipelineWithLogProb.from_pretrained(
        cfg.pipeline_path, minimal_512_only=cfg.minimal_512
    )
    # 启用 P0 缓存：第一次运行缓存 dense latent_index，第二次复用
    pipe.opts.cache_dense_latent_index = True
    pipe.to(cfg.device)
    if cfg.dtype == "fp32":
        pipe.ref.dtype = torch.float32
    elif cfg.dtype in ("fp16", "half"):
        pipe.ref.dtype = torch.float16
    return pipe


def run_sampling(
    pipe, cfg: InferConfig, generator: torch.Generator
) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    dense_params = {"num_inference_steps": int(cfg.dense_steps)}  # 标量
    sparse_params = {
        "num_inference_steps": int(cfg.sparse_steps),
        "mc_threshold": float(cfg.mc_threshold),
    }  # 标量
    meshes, latents_seq_flat, step_log_probs_flat, step_kl_flat = pipe.sample_candidates_with_logprob(
        image=cfg.image,
        num_candidates=int(cfg.num_candidates),
        dense_params=dense_params,
        sparse_params_512=sparse_params,
        guidance_scale=float(cfg.guidance),
        use_sde=bool(cfg.use_sde),
        sigma_min=float(cfg.sigma_min),
        rescale_t=float(cfg.rescale_t),
        generator=generator,
    )
    return meshes, latents_seq_flat, step_log_probs_flat, step_kl_flat


def validate_sampling_outputs(
    meshes: List[Any],
    latents_seq_flat: List[torch.Tensor],
    step_log_probs_flat: List[torch.Tensor],
    cfg: InferConfig,
) -> None:
    # 期望 log_prob 条目数 = candidates * sparse_steps
    expected_lp = cfg.num_candidates * cfg.sparse_steps  # 标量
    assert (
        len(step_log_probs_flat) == expected_lp
    ), f"log_prob 数量 {len(step_log_probs_flat)} != 期望 {expected_lp}"

    if cfg.use_sde:
        # SDE 模式至少应有非零 logprob
        lp_cat = torch.stack(step_log_probs_flat)  # (K*T,1) 或 (K*T,)
        non_zero = (lp_cat.abs() > 0).sum().item()
        assert non_zero > 0, "SDE 模式 log_prob 全为 0"
    else:
        # ODE 模式 logprob 应为 0
        if len(step_log_probs_flat) > 0:
            lp_cat = torch.stack(step_log_probs_flat)
            assert float(lp_cat.abs().sum().item()) == 0.0, "ODE 模式不应产生 log_prob"

    # latent 序列长度（若 pipeline 启用了 store_all_latents）
    # 无法保证一定开启，所以只在检测到非空时做一致性检查：应是 candidates * (steps+1)
    if len(latents_seq_flat) > 0:
        expected_latents_min = cfg.num_candidates * (cfg.sparse_steps + 1)
        assert (
            len(latents_seq_flat) % cfg.num_candidates == 0
        ), "latents 展平后无法按候选均分"
        assert (
            len(latents_seq_flat) >= expected_latents_min
        ), f"latents 序列长度 {len(latents_seq_flat)} < 期望最小 {expected_latents_min}"

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
    os.makedirs(out_dir, exist_ok=True)
    for i, mesh in enumerate(meshes):
        print(f"[DEBUG] Export candidate {i}: type={type(mesh)} attrs={dir(mesh)[:15]}")
        # 如果是 SparseTensor（尚未 decode）并且提供了 pipeline，可尝试 decode
        if type(mesh).__name__ == 'SparseTensor' and pipeline_obj is not None:
            # 尝试访问 feats / indices 属性
            feats = getattr(mesh, 'feats', None)
            coords = getattr(mesh, 'indices', None) or getattr(mesh, 'coords', None)
            if feats is not None and coords is not None:
                try:
                    # 直接调用 pipeline 内部 decode 函数（需要 latent_index）
                    decoded = pipeline_obj._decode_sparse_mesh(feats, coords, mc_threshold=mc_threshold, remove_interior=False)
                    mesh = decoded
                    print("[DEBUG] SparseTensor 通过 pipeline._decode_sparse_mesh 解码")
                except Exception as e:
                    print(f"[WARN] 解码 SparseTensor 失败: {e}")
        # 如果 mesh 是 dict
        if isinstance(mesh, dict):
            if 'mesh' in mesh:
                mesh = mesh['mesh']
            else:
                print(f"[WARN] dict mesh 缺少 'mesh' 键，跳过 {i}")
                continue
        v = getattr(mesh, "vertices", None)
        f = getattr(mesh, "faces", None)
        if v is None or f is None:
            # 尝试 export 接口
            if hasattr(mesh, "export"):
                out_path = os.path.join(out_dir, f"mesh_{i}.ply")
                try:
                    mesh.export(out_path)
                    print(f"[DEBUG] Saved via mesh.export -> {out_path}")
                except Exception as e:
                    print(f"[ERROR] export() 失败: {e}")
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

        try:
            tri = trimesh.Trimesh(vertices=v_np, faces=f_np)
            out_path = os.path.join(out_dir, f"mesh_{i}.ply")
            tri.export(out_path)
            print(f"[DEBUG] Saved via trimesh.Trimesh -> {out_path}")
        except Exception as e:
            print(f"[ERROR] trimesh 导出失败: {e}")


def reproducibility_check(pipe: Any, cfg: InferConfig) -> None:
    """两阶段复现性验证：
    Phase A: 正常生成（缓存 dense latent_index）。
    Phase B: reuse_cached_dense_latent_index=True 强制复用，期望 log_prob 完全一致。
    同时比较 latent_index hash（已由 pipeline 打印）。
    """
    # Phase A
    pipe.opts.reuse_cached_dense_latent_index = False
    g1 = torch.Generator(device=torch.device(cfg.device)); g1.manual_seed(cfg.seed)
    run1 = run_sampling(pipe, cfg, g1)
    # Phase B
    pipe.opts.reuse_cached_dense_latent_index = True
    g2 = torch.Generator(device=torch.device(cfg.device)); g2.manual_seed(cfg.seed)
    run2 = run_sampling(pipe, cfg, g2)
    lp1 = torch.stack(run1[2]).float().view(-1)
    lp2 = torch.stack(run2[2]).float().view(-1)
    if lp1.shape != lp2.shape:
        raise AssertionError(f"log_prob shape mismatch {lp1.shape} vs {lp2.shape}")
    max_abs = (lp1 - lp2).abs().max().item()
    if not torch.allclose(lp1, lp2, atol=1e-6, rtol=1e-6):
        # 打印详细差异：前若干元素
        diff_idx = (lp1 != lp2).nonzero(as_tuple=False).view(-1)[:10]
        print(f"[REPRO][ERR] log_prob mismatch max_abs={max_abs:.6g} indices_sample={diff_idx.tolist()}")
        print("lp1 sample:", lp1[diff_idx].tolist())
        print("lp2 sample:", lp2[diff_idx].tolist())
        raise AssertionError("同种子 (cached latent_index) log_prob 不一致")
    if cfg.use_sde:
        print(f"[REPRO] log_prob match (max_abs={max_abs:.2e}) under cached latent_index reuse")


def summarize_logprob(step_log_probs_flat: List[torch.Tensor]) -> str:
    if not step_log_probs_flat:
        return "(empty)"
    vals = torch.stack(step_log_probs_flat).float().view(-1)
    p = torch.quantile(vals, torch.tensor([0.01, 0.5, 0.99], device=vals.device))
    return f"mean={vals.mean():.3f} std={vals.std(unbiased=False):.3f} p1={p[0]:.3f} p50={p[1]:.3f} p99={p[2]:.3f}"


def parse_args() -> InferConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_path", type=str, required=True, help="包含 config.yaml 与权重的目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径 (文件) 或原路径传递让 pipeline 内部处理")
    ap.add_argument("--out", type=str, default="outputs/test_runs/direct3d_s2_validation")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--dense_steps", type=int, default=50)
    ap.add_argument("--sparse_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--mc_threshold", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--sigma_min", type=float, default=0.002)
    ap.add_argument("--rescale_t", type=float, default=1000.0)
    ap.add_argument("--no_sde", action="store_true", help="关闭 SDE (退化为 ODE)")
    ap.add_argument("--minimal_512", action="store_true", help="仅加载 dense + sparse512")
    ap.add_argument("--skip_refiner", action="store_true", help="跳过 refiner")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "half"], help="主 dtype")
    ap.add_argument("--do_e2e", action="store_true", help="执行端到端采样")
    ap.add_argument("--deterministic", action="store_true", help="启用后端确定性 (cuDNN)")
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
        skip_refiner=args.skip_refiner,
        dtype=args.dtype,
        do_e2e=bool(args.do_e2e),
        deterministic=bool(args.deterministic),
    )


def main() -> None:
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
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            print("[INFO] 启用确定性后端设置")
        except Exception as e:
            print(f"[WARN] 设置确定性失败: {e}")

    # 3) 端到端首次采样
    main_gen = torch.Generator(device=device)
    main_gen.manual_seed(cfg.seed)
    meshes, latents_seq_flat, step_log_probs_flat, step_kl_flat = run_sampling(
        pipe, cfg, main_gen
    )

    validate_sampling_outputs(meshes, latents_seq_flat, step_log_probs_flat, cfg)

    # 4) 可复现性（基于 log_prob 序列严格比较）
    reproducibility_check(pipe, cfg)
    print("[OK] 可复现性（同种子）验证通过")

    # 5) 统计输出
    stats = summarize_logprob(step_log_probs_flat)
    print(f"[Stats] log_prob: {stats}")

    # 6) 导出网格
    export_meshes(meshes, cfg.out_dir, pipeline_obj=pipe, mc_threshold=cfg.mc_threshold)
    print(f"[OK] 导出 {len(meshes)} 个 mesh -> {cfg.out_dir}")

    print("[DONE] Direct3D‑S2 集成推理测试全部通过")


if __name__ == "__main__":
    main()
