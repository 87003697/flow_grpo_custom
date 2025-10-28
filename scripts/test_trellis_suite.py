#!/usr/bin/env python3
"""
整合 TRELLIS 测试为统一入口（无 try/except、支持并行输出校验）。

包含：
- 稀疏张量与 Flow 单步 logprob 基础测试
- 管线加载/图像条件/Stage1 推理
- Stage2 并行采样 (B×K) 行为与返回长度校验
- 可选：SDE vs ODE 渲染对比（关闭 CFG），保存到 outputs/trellis_sde_vs_ode/

使用：
  python scripts/test_trellis_suite.py --quick true --steps 10 --num-candidates 2
"""

import sys
import os
from pathlib import Path
import argparse

import torch

# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 固定环境（避免网络/加速器差异）
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def run_basic_sparse_and_flow_tests() -> None:
    """基础：SparseTensor 拼接、Flow 单步 logprob、CFG 合并。
    自包含实现，避免外部依赖。
    """
    # 1) SparseTensor 拼接
    from generators.trellis import sparse as sp  # type: ignore
    from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat

    coords = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4], [1, 1, 2, 3]], dtype=torch.int32)  # (3,4)
    feats = torch.randn(3, 64)  # (3,64)
    st1 = sp.SparseTensor(coords=coords, feats=feats)
    coords2 = torch.tensor([[0, 4, 5, 6], [1, 5, 6, 7]], dtype=torch.int32)  # (2,4)
    feats2 = torch.randn(2, 64)  # (2,64)
    st2 = sp.SparseTensor(coords=coords2, feats=feats2)
    combined = sparse_tensor_cat([st1, st2])
    assert combined.coords.shape[0] == st1.coords.shape[0] + st2.coords.shape[0]
    assert combined.feats.shape[1] == st1.feats.shape[1]

    # 2) Flow 单步 logprob（期望 batch 维输出 (1,)）
    from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_step_with_logprob
    coords = torch.tensor([[0, 10, 20, 30], [0, 15, 25, 35]], dtype=torch.int32)  # (N=2,4)
    sample = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))  # feats (2,32)
    model_out = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))  # feats (2,32)
    prev_sample, log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
        sample=sample, model_output=model_out, t=500.0, t_prev=450.0, sigma_min=0.002, generator=None, deterministic=False
    )
    assert prev_sample.feats.shape == sample.feats.shape
    assert tuple(log_prob.shape) == (1,)
    assert tuple(std_dev.shape) == (1,)

    # 3) CFG 合并
    from flow_grpo.diffusers_patch.trellis_sparse_tensor import sparse_tensor_cfg_guidance
    pos = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))
    neg = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))
    cfg = sparse_tensor_cfg_guidance(pos, neg, guidance_scale=3.0)
    assert cfg.feats.shape == pos.feats.shape


def run_batched_logprob_quick_test() -> None:
    """快速单元测试：验证 compute_log_prob_trellis_stage2_batched 输出形状与数值有效性。

    构造 B=3 的子批，每个样本包含 2 个时间点的 SparseTensor（steps=1，对第 j=0 步重算）。
    使用 DummyPipeline + DummyModel，模型输出恒为 0，便于稳定验证。
    """
    import numpy as np
    from generators.trellis import sparse as sp  # type: ignore
    import torch
    from flow_grpo.diffusers_patch.trellis_sparse_tensor import compute_log_prob_trellis_stage2_batched
    import ml_collections

    class DummyModel(torch.nn.Module):
        def forward(self, sample: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs):
            feats = torch.zeros_like(sample.feats)  # (N, C)
            return sp.SparseTensor(coords=sample.coords, feats=feats)  # (N,C)

    class DummyPipeline:
        def __init__(self):
            self._m = DummyModel()
        def get_trainable_model(self):
            return self._m

    # 构造 B=3 的样本
    B = 3  # 标量
    steps = 1  # 标量（仅 j=0）
    t_seq = np.array([1000.0, 0.0], dtype=float)  # (steps+1,)
    samples = []
    image_conds_list = []
    for b in range(B):
        # coords: 两个点，保证 batch=0 单样本（拼接时会重写 batch 维）
        coords = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=torch.int32)  # (N=2,4)
        feats = torch.randn(2, 16)  # (N=2,C=16)
        st_cur = sp.SparseTensor(coords=coords, feats=feats)  # 当前 x_t (N,C)
        # 观测到的上一时刻（令其等于 mean=x_t，避免随机噪声影响）
        st_prev = sp.SparseTensor(coords=coords.clone(), feats=feats.clone())  # (N,C)
        # 注意：为避免批次维被推断为 >1，强制将每个样本的 coords[:,0] 设为 0
        st_cur = sp.SparseTensor(coords=st_cur.coords.clone().index_fill(1, torch.tensor([0], dtype=torch.long), 0), feats=st_cur.feats)
        st_prev = sp.SparseTensor(coords=st_prev.coords.clone().index_fill(1, torch.tensor([0], dtype=torch.long), 0), feats=st_prev.feats)

        samples.append({
            "latents_seq": [st_cur, st_prev],  # 长度 steps+1
            "t_seq": t_seq,  # (2,)
        })
        # 条件（patch 级）：(1,P,C)
        cond = torch.randn(1, 4, 8)  # (1,P=4,C=8)
        image_conds_list.append({"cond": cond, "neg_cond": None})

    cfg = ml_collections.FrozenConfigDict({
        "guidance_scale": 1.0,
        "num_inference_steps": steps,
        "sigma_min": 0.002,
        "rescale_t": 1.0,
        "deterministic": False,
        "kl_reward": 0.0,
    })

    pipeline = DummyPipeline()
    log_prob_vec, kl_vec = compute_log_prob_trellis_stage2_batched(
        pipeline=pipeline,
        samples=samples,
        j=0,
        image_conds_list=image_conds_list,
        config=cfg,
    )

    # 断言形状与数值
    assert tuple(log_prob_vec.shape) == (B,), f"log_prob_vec 形状错误: {tuple(log_prob_vec.shape)}"
    assert tuple(kl_vec.shape) == (B,), f"kl_vec 形状错误: {tuple(kl_vec.shape)}"
    assert not torch.isnan(log_prob_vec).any(), "log_prob_vec 存在 NaN"
    assert not torch.isnan(kl_vec).any(), "kl_vec 存在 NaN"
    print(f"✅ compute_log_prob_trellis_stage2_batched 通过：形状 {tuple(log_prob_vec.shape)}，无 NaN")

def run_pipeline_stage1_tests():
    """加载 Pipeline、编码条件、运行 Stage1，返回 (pipeline, coords, image_conds)。"""
    from generators.trellis.pipeline import TrellisStage2Pipeline
    from PIL import Image
    import numpy as np

    pipeline = TrellisStage2Pipeline(verbose=False)
    if torch.cuda.is_available():
        pipeline.cuda()

    # 随机图像（512x512）
    img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    image_conds = pipeline.prepare_image_conditions([img])  # {'cond': (B,P,C), 'neg_cond': (B,P,C)}
    assert 'cond' in image_conds and 'neg_cond' in image_conds
    assert image_conds['cond'].ndim == 3

    coords = pipeline.forward_stage1(image_conds)  # (N,4)
    assert coords.shape[1] == 4
    return pipeline, coords, image_conds


def run_stage2_parallel_validation(pipeline, coords, image_conds, steps: int, num_candidates: int) -> None:
    """校验 Stage2 B×K 并行输出长度与形状。

    - image_conds: 字典，包含 'cond'/'neg_cond'；'cond' 形状 (B, P, C)
    - coords: (N, 4)
    - 最终断言 all_latents 与 all_log_probs 条目数符合 (B*K)*(steps+1)/(B*K)*steps
    """
    from flow_grpo.diffusers_patch.trellis_stage2_with_logprob import trellis_stage2_with_logprob

    B = int(image_conds['cond'].shape[0])  # 标量
    # 构造参数
    meshes, all_latents, all_log_probs, _ = trellis_stage2_with_logprob(
        pipeline=pipeline,
        stage1_cond_dict=image_conds,  # {'cond': (B,P,C), 'neg_cond': (B,P,C)}
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        kl_reward=0.0,
        deterministic=True,
        sparse_structure_sampler_params=dict(max_points=int(coords.shape[0])),
        slat_sampler_params=dict(sigma_min=0.002, rescale_t=1.0),
        num_candidates=int(num_candidates),
        output_type="latent",
    )

    # 期望长度
    expected_latents = B * int(num_candidates) * (int(steps) + 1)  # 标量
    expected_logs = B * int(num_candidates) * int(steps)           # 标量

    # 长度断言
    assert len(all_latents) == expected_latents, f"latents 条目数不匹配: {len(all_latents)} vs {expected_latents}"
    assert len(all_log_probs) == expected_logs, f"log_probs 条目数不匹配: {len(all_log_probs)} vs {expected_logs}"

    # 形状抽查：每个 log_prob 应为 (1,)
    if expected_logs > 0:
        assert all(tuple(t.shape) == (1,) for t in all_log_probs), "每步 log_prob 形状应为 (1,)"


def run_parallel_visualization(
    pipeline,
    image_conds: dict,
    steps: int,
    num_candidates: int,
    out_dir: Path,
    preset: str = "turntable",
    max_meshes: int = 8,
):
    """并行 (B×K) 结果可视化：渲染并保存网格预览。

    - image_conds: 官方 get_cond 输出 {'cond': (B,P,C), 'neg_cond': (B,P,C)}
    - 输出: out_dir 下保存若干 PNG，文件名包含样本与候选索引
    """
    from flow_grpo.diffusers_patch.trellis_stage2_with_logprob import trellis_stage2_with_logprob
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import (
        render_mesh_multiple_views,
    )

    B = int(image_conds['cond'].shape[0])  # 标量
    meshes, _, _, _ = trellis_stage2_with_logprob(
        pipeline=pipeline,
        stage1_cond_dict=image_conds,
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        kl_reward=0.0,
        deterministic=True,
        sparse_structure_sampler_params={},
        slat_sampler_params=dict(sigma_min=0.002, rescale_t=1.0),
        num_candidates=int(num_candidates),
        output_type="trimesh",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(meshes)  # 标量，期望 B*K
    limit = min(int(max_meshes), total)
    for idx in range(limit):
        mesh = meshes[idx]
        b = idx // int(num_candidates)  # 标量
        k = idx % int(num_candidates)   # 标量
        save_path = out_dir / f"viz_b{b}_k{k}.png"
        render_mesh_multiple_views(
            mesh_trimesh=mesh,
            save_path=str(save_path),
            preset=preset,
            device=pipeline.device.type,
        )
    print(f"已保存并行可视化 {limit}/{total} 张到: {str(out_dir)}")


def run_sde_vs_ode_render(pipeline, coords, image_conds, steps: int, out_dir: Path) -> None:
    """运行 SDE 与 ODE 的对比渲染（关闭 CFG），保存两张对比图。"""
    # 本地实现：build_initial_noise / run_stage2 / decode_and_render
    from generators.trellis import sparse as sp  # type: ignore
    from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_multiple_views

    device = pipeline.device
    slat_flow_model = pipeline.get_trainable_model()
    in_channels = int(getattr(slat_flow_model, "in_channels"))  # 标量

    # 固定 coords 到设备
    coords = coords.to(device=device)  # (N, 4)

    # 初始噪声
    noise_feats = torch.randn(coords.shape[0], in_channels, device=device)  # (N,C)
    initial_noise = sp.SparseTensor(coords=coords, feats=noise_feats)

    # 采样封装
    def _run_stage2(deterministic: bool, seed: int):
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        final_slat, all_latents, all_log_probs, _ = trellis_flow_euler_sampler_with_logprob(
            model=slat_flow_model,
            noise=initial_noise,
            cond=image_conds['cond'],
            steps=int(steps),
            sigma_min=0.002,
            rescale_t=1.0,
            generator=rng if not deterministic else None,
            deterministic=deterministic,
            guidance_scale=1.0,
            neg_cond=None,
            verbose=True,
        )
        return final_slat

    sde_slat = _run_stage2(deterministic=False, seed=42)
    ode_slat = _run_stage2(deterministic=True, seed=42)

    out_dir.mkdir(parents=True, exist_ok=True)
    sde_img = render_mesh_multiple_views(mesh_trimesh=pipeline.decode_slat_to_mesh(sde_slat)[0], save_path=str(out_dir / "sde_turntable.png"), preset="turntable", device=pipeline.device.type)
    ode_img = render_mesh_multiple_views(mesh_trimesh=pipeline.decode_slat_to_mesh(ode_slat)[0], save_path=str(out_dir / "ode_turntable.png"), preset="turntable", device=pipeline.device.type)
    print(f"SDE 渲染: {sde_img}")
    print(f"ODE 渲染: {sde_img}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", type=str, default="true", help="true/false，是否跳过渲染对比")
    parser.add_argument("--steps", type=int, default=10, help="Stage2 采样步数")
    parser.add_argument("--num-candidates", type=int, default=2, help="每图候选数 K")
    parser.add_argument("--viz", type=str, default="true", help="true/false，是否保存并行可视化")
    parser.add_argument("--viz-max", type=int, default=8, help="最多保存多少张并行可视化")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "outputs/trellis_sde_vs_ode"))
    args = parser.parse_args()

    quick = args.quick.lower() in ["1", "true", "yes"]
    steps = int(args.steps)
    num_candidates = int(args["num_candidates"]) if isinstance(args, dict) and "num_candidates" in args else int(getattr(args, "num_candidates"))
    do_viz = args.viz.lower() in ["1", "true", "yes"]
    viz_max = int(args.viz_max)

    print("== 基础稀疏张量/Flow 单步测试 ==")
    run_basic_sparse_and_flow_tests()

    print("\n== Batched 单步重算 测试 ==")
    run_batched_logprob_quick_test()

    print("\n== 管线加载/图像条件/Stage1 推理 ==")
    pipeline, coords, image_conds = run_pipeline_stage1_tests()

    print("\n== Stage2 并行 (B×K) 输出校验 ==")
    run_stage2_parallel_validation(pipeline, coords, image_conds, steps=steps, num_candidates=num_candidates)

    if do_viz:
        print("\n== 并行 (B×K) 结果可视化 ==")
        run_parallel_visualization(
            pipeline=pipeline,
            image_conds=image_conds,
            steps=steps,
            num_candidates=num_candidates,
            out_dir=Path(args.out_dir) / "parallel",
            preset="turntable",
            max_meshes=viz_max,
        )

    if not quick:
        print("\n== SDE vs ODE 渲染对比 ==")
        run_sde_vs_ode_render(pipeline, coords, image_conds, steps=max(steps, 10), out_dir=Path(args.out_dir))

    print("\n🎉 测试整合完成：所有子测均通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())


