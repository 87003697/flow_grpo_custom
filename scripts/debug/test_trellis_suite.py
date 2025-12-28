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

# 项目根路径（scripts/debug -> scripts -> root）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 注入 TRELLIS 官方代码路径
_TRELLIS_ROOT = str(PROJECT_ROOT / "_reference_codes" / "TRELLIS")
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

# 注入 VGGTObj 参考渲染器路径
_VGGT_ROOT = PROJECT_ROOT / "_reference_codes" / "VGGTObj"
if str(_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_ROOT))

import torch
from trellis.modules import sparse as sp  # type: ignore
from flow_grpo.diffusers_patch.trellis_pipeline_with_logprob import (
    TrellisPipelineWithLogProb,
    convert_trellis_to_trimesh,
)
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from reward_models.camera_normal_scorer.render.adapter import to_mesh_extract, KiuiMeshLike


def _mesh_to_trimesh(mesh_obj):
    import trimesh
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    vertices = getattr(mesh_obj, "vertices", None)
    faces = getattr(mesh_obj, "faces", None)
    if vertices is not None and faces is not None:
        return convert_trellis_to_trimesh([mesh_obj])[0]
    v_attr = getattr(mesh_obj, "v", None)
    f_attr = getattr(mesh_obj, "f", None)
    if v_attr is not None and f_attr is not None:
        v_np = v_attr.detach().cpu().numpy() if torch.is_tensor(v_attr) else np.asarray(v_attr)
        f_np = f_attr.detach().cpu().numpy() if torch.is_tensor(f_attr) else np.asarray(f_attr)
        return trimesh.Trimesh(vertices=v_np, faces=f_np)
    return convert_trellis_to_trimesh([mesh_obj])[0]

def save_meshes_for_preview(
    meshes,
    repeated_image_paths,
    rewards,
    epoch: int,
    save_dir: str,
    device_str: str = "cuda",
    repeated_image_pils=None,
    write_mesh: bool = False,
):
    """使用参考渲染器渲染三视角法线并保存 2×2 预览 PNG；可选导出 OBJ。"""
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device_str)
    renderer = RefMeshRenderer(img_size=512, device=device_str)

    preview_files = []
    mesh_files = []

    elevation = 15.0
    distance = 3.0
    fovy = 50.0
    azimuths = [0.0, 120.0, 240.0]
    predefined_poses = [
        {"distance": distance, "fovy": fovy, "elevation": elevation, "azimuth": a} for a in azimuths
    ]

    for idx, (mesh, img_path) in enumerate(zip(meshes, repeated_image_paths)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        safe_base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).rstrip()
        case_dir = os.path.join(save_dir, f"{safe_base}_epoch{epoch}")
        os.makedirs(case_dir, exist_ok=True)
        preview_path = os.path.join(case_dir, f"preview_{idx}.png")

        mesh_ex = to_mesh_extract(mesh, device)

        if write_mesh:
            import trimesh
            v_np = mesh_ex.vertices.detach().cpu().numpy()
            f_np = mesh_ex.faces.detach().cpu().numpy().astype(np.int32)
            tri = trimesh.Trimesh(vertices=v_np, faces=f_np)
            mesh_path = os.path.join(case_dir, f"mesh_{idx}.obj")
            tri.export(mesh_path)
            mesh_files.append(mesh_path)

        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)
        cams = renderer.sample_camera_poses(num_random_views=0, predefined_poses=predefined_poses)
        out = renderer.render_mesh(
            mesh_kiui,
            cams,
            return_depth=False,
            return_normals=True,
            return_positions=False,
            return_masks=True,
        )

        images_t = out["images"]
        R = images_t.shape[-1]

        def to_pil(img_chw: torch.Tensor) -> Image.Image:
            img01 = img_chw.clamp(0, 1)
            img255 = (img01 * 255.0).round().to(torch.uint8)
            img_hwc = img255.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(img_hwc)

        pil_renders = [to_pil(images_t[i]) for i in range(3)]

        if repeated_image_pils is not None and idx < len(repeated_image_pils):
            rgb_in = repeated_image_pils[idx]
        else:
            rgb_in = Image.new("RGB", (512, 512), (255, 255, 255))
        w, h = rgb_in.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        rgb_sq = rgb_in.crop((left, top, left + side, top + side)).resize((R, R), Image.BICUBIC)

        panel = Image.new("RGB", (R * 2, R * 2))
        panel.paste(rgb_sq, (0, 0))
        panel.paste(pil_renders[0], (R, 0))
        panel.paste(pil_renders[1], (0, R))
        panel.paste(pil_renders[2], (R, R))
        panel.save(preview_path)

        preview_files.append(preview_path)

    return mesh_files, preview_files
from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_multiple_views
from flow_grpo.diffusers_patch.trellis_sparse_tensor import (
    trellis_flow_step_with_logprob,
    create_trellis_scheduler,
    sparse_tensor_cfg_guidance,
    compute_log_prob_trellis_stage2,
    sparse_tensor_cat,
)
from PIL import Image
import numpy as np
import ml_collections

# 兼容别名
TrellisStage2Pipeline = TrellisPipelineWithLogProb

# 固定环境（避免网络/加速器差异）
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def run_basic_sparse_and_flow_tests() -> None:
    """基础：SparseTensor 拼接、Flow 单步 logprob、CFG 合并。
    自包含实现，避免外部依赖。
    """
    # 1) SparseTensor 拼接

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
    coords = torch.tensor([[0, 10, 20, 30], [0, 15, 25, 35]], dtype=torch.int32)  # (N=2,4)
    sample = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))  # feats (2,32)
    model_out = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))  # feats (2,32)
    scheduler = create_trellis_scheduler(steps=1, device=sample.coords.device)
    # 使用调度器时间对 (t=1000 -> t_prev=0)
    prev_sample, log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
        scheduler=scheduler,
        sample=sample,
        model_output=model_out,
        timestep=1000.0,
        prev_timestep=0.0,
        generator=None,
        deterministic=False,
    )
    assert prev_sample.feats.shape == sample.feats.shape
    assert tuple(log_prob.shape) == (1,)
    assert tuple(std_dev.shape) == (1,)

    # 3) CFG 合并
    pos = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))
    neg = sp.SparseTensor(coords=coords, feats=torch.randn(2, 32))
    cfg = sparse_tensor_cfg_guidance(pos, neg, guidance_scale=3.0)
    assert cfg.feats.shape == pos.feats.shape


def run_batched_logprob_quick_test() -> None:
    """快速单元测试：验证 compute_log_prob_trellis_stage2_batched 输出形状与数值有效性。

    构造 B=3 的子批，每个样本包含 2 个时间点的 SparseTensor（steps=1，对第 j=0 步重算）。
    使用 DummyPipeline + DummyModel，模型输出恒为 0，便于稳定验证。
    """
    

    class DummyModel(torch.nn.Module):
        def forward(self, sample: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, **kwargs):
            feats = torch.zeros_like(sample.feats)  # (N, C)
            return sp.SparseTensor(coords=sample.coords, feats=feats)  # (N,C)

    class DummyPipeline:
        def __init__(self):
            self._m = DummyModel()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.stage2_scheduler = create_trellis_scheduler(steps=steps, device=self.device)
        def get_flow_module(self, kind: str, unwrap_ddp: bool = True):
            if kind != "shape_slat":
                raise ValueError(f"unsupported kind: {kind}")
            return self._m

    # 构造 B=3 的样本
    B = 3  # 标量
    steps = 1  # 标量（仅 j=0）
    t_seq = np.array([1000.0, 0.0], dtype=float)  # (steps+1,)
    samples = []
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

        # 条件（patch 级）：(1,P,C)
        cond = torch.randn(1, 4, 8)  # (1,P=4,C=8)

        samples.append({
            "latents_seq": [st_cur, st_prev],  # 长度 steps+1
            "t_seq": t_seq,  # (2,)
            "cond_patches": cond,  # (1,P,C)
            "neg_patches": None,   # guidance_scale=1.0 时可以为 None
        })

    cfg = ml_collections.FrozenConfigDict({
        "guidance_scale": 1.0,
        "num_inference_steps": steps,
        "sigma_min": 0.002,
        "rescale_t": 1.0,
        "deterministic": False,
        "kl_reward": 0.0,
        "noise_level": 0.7,
    })

    pipeline = DummyPipeline()
    _, log_prob_vec, kl_vec = compute_log_prob_trellis_stage2(
        pipeline=pipeline,
        samples=samples,
        j=0,
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
    
    model_path = os.environ.get(
        "TRELLIS_MODEL_PATH",
        str(PROJECT_ROOT / "pretrained_weights" / "TRELLIS-image-large"),
    )
    pipeline = TrellisStage2Pipeline.from_pretrained(model_path)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # 默认使用固定测试图像，若不存在则回退到随机图像
    default_image = PROJECT_ROOT / "dataset" / "alphaimages_1k" / "test" / "images" / "00098.png"
    if default_image.is_file():
        img = Image.open(default_image).convert("RGB")
        print(f"[Stage1] 使用测试图像: {default_image}")
    else:
        print(f"[Stage1] 警告: 未找到 {default_image}，使用随机图像代替。")
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    cond, neg_cond = pipeline.prepare_image_conditions([img])  # (cond, neg_cond)
    image_conds = {'cond': cond, 'neg_cond': neg_cond}
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
    
    B = int(image_conds['cond'].shape[0])  # 标量
    # 构造参数
    with torch.inference_mode():
        coords_list_eval, _, _, _ = pipeline.stage1_with_logprob(
            cond_dict=image_conds,
            num_inference_steps=int(steps),
            guidance_scale=1.0,
        )
    # 打包 BK 的 stage1 条目
    st_list = []
    for i in range(B):
        for _ in range(int(num_candidates)):
            c = coords_list_eval[i]
            st_list.append(sp.SparseTensor(coords=c, feats=torch.empty((c.shape[0], 0), device=c.device)))
    coords_batched = sparse_tensor_cat(st_list)
    cond_bk = torch.cat([image_conds['cond'][i:i+1].repeat(int(num_candidates), 1, 1) for i in range(B)], dim=0)
    neg_bk = torch.cat([image_conds['neg_cond'][i:i+1].repeat(int(num_candidates), 1, 1) for i in range(B)], dim=0) if (image_conds['neg_cond'] is not None) else None
    stage1_packed = { 'cond': cond_bk, 'neg_cond': neg_bk, 'coords': coords_batched }
    meshes, all_latents, all_log_probs, _ = pipeline.stage2_with_logprob(
        stage1_cond_dict=stage1_packed,
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        slat_sampler_params=dict(sigma_min=0.002, rescale_t=1.0),
        deterministic=True,
    )

    # 期望长度
    expected_latents = int(steps) + 1  # stage2 返回的是每个时间步的 batched 稀疏张量
    expected_logs = int(steps)

    # 长度断言
    assert len(all_latents) == expected_latents, f"latents 条目数不匹配: {len(all_latents)} vs {expected_latents}"
    assert len(all_log_probs) == expected_logs, f"log_probs 条目数不匹配: {len(all_log_probs)} vs {expected_logs}"

    # 形状抽查：每个 log_prob 应为 (1,)
    if expected_logs > 0:
        expected_log_shape = (B * int(num_candidates),)
        assert all(tuple(t.shape) == expected_log_shape for t in all_log_probs), f"每步 log_prob 形状应为 {expected_log_shape}"


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
    

    B = int(image_conds['cond'].shape[0])  # 标量
    with torch.inference_mode():
        coords_list_eval, _, _, _ = pipeline.stage1_with_logprob(
            cond_dict=image_conds,
            num_inference_steps=int(steps),
            guidance_scale=1.0,
        )
    st_list = []
    for i in range(B):
        for _ in range(int(num_candidates)):
            c = coords_list_eval[i]
            st_list.append(sp.SparseTensor(coords=c, feats=torch.empty((c.shape[0], 0), device=c.device)))
    coords_batched = sparse_tensor_cat(st_list)
    cond_bk = torch.cat([image_conds['cond'][i:i+1].repeat(int(num_candidates), 1, 1) for i in range(B)], dim=0)
    neg_bk = torch.cat([image_conds['neg_cond'][i:i+1].repeat(int(num_candidates), 1, 1) for i in range(B)], dim=0) if (image_conds['neg_cond'] is not None) else None
    stage1_packed = { 'cond': cond_bk, 'neg_cond': neg_bk, 'coords': coords_batched }
    meshes, _, _, _ = pipeline.stage2_with_logprob(
        stage1_cond_dict=stage1_packed,
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        slat_sampler_params=dict(sigma_min=0.002, rescale_t=1.0),
        deterministic=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(meshes)
    limit = min(int(max_meshes), total)
    if limit <= 0:
        print("⚠️ 无可用 mesh 进行可视化")
        return

    dummy_paths = [f"parallel_b{idx // int(num_candidates)}_k{idx % int(num_candidates)}.png" for idx in range(limit)]
    dummy_pils = [Image.new("RGB", (512, 512), (255, 255, 255)) for _ in range(limit)]
    rewards = np.zeros(limit, dtype=np.float32)

    save_meshes_for_preview(
        meshes=meshes[:limit],
        repeated_image_paths=dummy_paths,
        rewards=rewards,
        epoch=0,
        save_dir=str(out_dir),
        device_str=pipeline.device.type,
        repeated_image_pils=dummy_pils,
        write_mesh=False,
    )
    print(f"已保存并行可视化 {limit}/{total} 条到: {str(out_dir)}")


def run_sde_vs_ode_render(pipeline, coords, image_conds, steps: int, out_dir: Path) -> None:
    """运行 SDE 与 ODE 的对比渲染（关闭 CFG），保存两张对比图。"""
    # 本地实现：build_initial_noise / run_stage2 / decode_and_render
    

    device = pipeline.device
    slat_flow_model = pipeline.get_trainable_model_stage2()
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
        scheduler = create_trellis_scheduler(steps=int(steps), device=device, rescale_t=1.0)
        t_seq = scheduler.timesteps.cpu().numpy().astype(np.float32)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(int(steps))]
        sample = initial_noise
        for t, t_prev in t_pairs:
            B = int(sample.shape[0])
            t_tensor = torch.tensor([t] * B, device=device, dtype=torch.float32)
            with torch.no_grad():
                model_out = slat_flow_model(sample, t_tensor, image_conds['cond'])
            sample, _, _, _ = trellis_flow_step_with_logprob(
                scheduler=scheduler,
                sample=sample,
                model_output=model_out,
                timestep=float(t),
                prev_timestep=float(t_prev),
                generator=(None if deterministic else rng),
                deterministic=deterministic,
            )
        return sample

    sde_slat = _run_stage2(deterministic=False, seed=42)
    ode_slat = _run_stage2(deterministic=True, seed=42)

    out_dir.mkdir(parents=True, exist_ok=True)
    sde_mesh = pipeline.decode_slat_to_mesh(sde_slat)[0]
    ode_mesh = pipeline.decode_slat_to_mesh(ode_slat)[0]
    dummy_paths = ["sde.png", "ode.png"]
    dummy_pils = [Image.new("RGB", (512, 512), (255, 255, 255)) for _ in range(2)]
    rewards = np.zeros(2, dtype=np.float32)
    save_meshes_for_preview(
        meshes=[sde_mesh, ode_mesh],
        repeated_image_paths=dummy_paths,
        rewards=rewards,
        epoch=0,
        save_dir=str(out_dir),
        device_str=pipeline.device.type,
        repeated_image_pils=dummy_pils,
        write_mesh=False,
    )
    print(f"SDE/ODE 渲染输出目录: {str(out_dir)}")


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

    if quick:
        print("\n⚡ 快速模式：跳过 Stage2 并行验证与可视化")
    else:
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


