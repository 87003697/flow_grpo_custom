#!/usr/bin/env python3
"""
TRELLIS ODE vs FlowEdit 对比脚本
=================================================
对标 test_trellis_SDE-vs-ODE.py 风格

功能：
- 共享 Stage1 坐标生成
- ODE: 标准 Euler 采样（无 FlowEdit）
- FlowEdit: ODE 产出 x_src → 差分双分支编辑
- 渲染统一视角的多视图 normal/color 图像
- 生成对比拼图并保存

示例：
  python scripts/debug/test_trellis_ODE-vs-FlowEdit.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/ode_vs_flowedit \
    --steps 50 --guidance 3.0 --seed 777 \
    --fe_steps 50 --fe_n_max 40 --fe_cfg_tgt 7.5 --fe_cfg_src -7.5
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
import math
import numpy as np

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS")
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

from edit4shape.generators.trellis.pipeline_adapter import TrellisRefAdapter, build_pipeline_from_reference
from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler
from edit4shape.generators.trellis.rollout.base import (
    _predict_sparse_cond_velocity,
    _predict_dense_cond_velocity,
    _expand_cond_to_batch,
)

from trellis.modules.sparse import SparseTensor
from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="TRELLIS 预训练模型目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径")
    ap.add_argument("--out", type=str, default="outputs/test_runs/ode_vs_flowedit", help="输出目录")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=50, help="Stage2 ODE 采样步数")
    ap.add_argument("--guidance", type=float, default=3.0, help="ODE CFG 引导系数")
    ap.add_argument("--rescale_t", type=float, default=1.0, help="时间重标")
    ap.add_argument("--seed", type=int, default=777, help="随机种子")
    ap.add_argument("--render_resolution", type=int, default=512, help="渲染分辨率")
    ap.add_argument("--num_views", type=int, default=4, help="统一视角数量")
    ap.add_argument("--camera_r", type=float, default=2.0, help="相机距离")
    ap.add_argument("--camera_fov", type=float, default=40.0, help="视场角 (度)")
    ap.add_argument("--camera_pitch", type=float, default=0.3, help="相机俯仰角 (弧度)")
    # FlowEdit 参数
    ap.add_argument("--fe_steps", type=int, default=None, help="FlowEdit 总步数（默认与 --steps 相同）")
    ap.add_argument("--fe_n_max", type=int, default=40, help="FlowEdit 实际执行步数")
    ap.add_argument("--fe_cfg_tgt", type=float, default=7.5, help="FlowEdit 正向 CFG scale")
    ap.add_argument("--fe_cfg_src", type=float, default=-7.5, help="FlowEdit 反向 CFG scale（负值）")
    return ap.parse_args()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_uniform_cameras(
    num_views: int,
    r: float = 2.0,
    fov: float = 40.0,
    pitch: float = 0.3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    yaws = [2 * math.pi * i / num_views for i in range(num_views)]
    pitchs = [pitch] * num_views
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    return extrinsics, intrinsics


def predict_velocity_with_cfg(
    adapter: TrellisRefAdapter,
    x_t: SparseTensor,
    t_val: float,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """预测速度场，支持 CFG 引导。使用 Trellis 原生 CFG 公式。"""
    model = adapter.pipe.models["slat_flow_model"]
    batch_size = cond_emb.shape[0]
    t_scaled = torch.full((batch_size,), t_val * 1000, device=device, dtype=torch.float32)

    uncond_v = model(x_t, t_scaled, uncond_emb)
    cond_v = model(x_t, t_scaled, cond_emb)

    t_norm = t_val
    if cfg_min <= t_norm <= cfg_max:
        cfg_weight = guidance
    else:
        cfg_weight = 0.0

    # Trellis CFG: (1 + scale) * cond - scale * uncond
    guided_feats = (1 + cfg_weight) * cond_v.feats - cfg_weight * uncond_v.feats
    return SparseTensor(coords=x_t.coords, feats=guided_feats)


def rollout_ode(
    adapter: TrellisRefAdapter,
    x_t: SparseTensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    scheduler: TrellisFlowScheduler,
    guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """ODE 采样循环（同 SDE-vs-ODE 脚本）"""
    steps = list(scheduler.timesteps)[:-1]

    for t in tqdm(steps, desc="ODE Rollout", leave=False):
        t_val = float(t.item())

        with torch.no_grad():
            velocity = predict_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                guidance, cfg_min, cfg_max, device,
            )

        x_t = scheduler.step(velocity, t_val, x_t).prev_sample
    return x_t


def rollout_flowedit(
    adapter: TrellisRefAdapter,
    x_src: SparseTensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    scheduler: TrellisFlowScheduler,
    cfg_scale_tgt: float,
    cfg_scale_src: float,
    n_max: int,
    device: torch.device,
    generator: torch.Generator,
) -> SparseTensor:
    """
    FlowEdit 差分采样循环。

    x_src 为 normalized 的 teacher clean z₀（ODE rollout 产出，经 re-normalize）。
    """
    z_edit_feats = x_src.feats.clone()
    noise = torch.randn(
        x_src.feats.shape, generator=generator,
        device=device, dtype=x_src.feats.dtype,
    )

    steps = list(scheduler.timesteps)[:-1]
    num_steps = len(steps)
    B = cond_emb.shape[0]

    model = adapter.pipe.models["slat_flow_model"]

    for i, t in enumerate(tqdm(steps, desc="FlowEdit Rollout", leave=False)):
        if num_steps - i > n_max:
            continue

        t_val = float(t.item())
        t_prev_val = float(scheduler.timesteps[i + 1].item())
        dt = t_prev_val - t_val
        t_scaled = torch.full((B,), t_val * 1000, device=device, dtype=torch.float32)

        with torch.no_grad():
            # ---- Source Branch ----
            latents_src_feats = (1 - t_val) * x_src.feats + t_val * noise
            latents_src = SparseTensor(coords=x_src.coords, feats=latents_src_feats)

            v_cond_src = model(latents_src, t_scaled, cond_emb)
            v_uncond_src = model(latents_src, t_scaled, uncond_emb)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src_feats = (
                (1 + cfg_scale_src) * v_cond_src.feats - cfg_scale_src * v_uncond_src.feats
            )

            # ---- Target Branch ----
            latents_tgt_feats = z_edit_feats + (latents_src_feats - x_src.feats)
            latents_tgt = SparseTensor(coords=x_src.coords, feats=latents_tgt_feats)

            v_cond_tgt = model(latents_tgt, t_scaled, cond_emb)
            v_uncond_tgt = model(latents_tgt, t_scaled, uncond_emb)

            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt_feats = (
                (1 + cfg_scale_tgt) * v_cond_tgt.feats - cfg_scale_tgt * v_uncond_tgt.feats
            )

            # ---- 差分 Euler 步 ----
            v_delta = v_cfg_tgt_feats - v_cfg_src_feats
            z_edit_feats = z_edit_feats + dt * v_delta

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt.feats - v_uncond_tgt.feats) * (1.0 - t_val)

    return SparseTensor(coords=x_src.coords, feats=z_edit_feats)


def render_mesh_multiview(
    mesh: Any,
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    resolution: int = 512,
) -> Dict[str, List[np.ndarray]]:
    render_out = render_frames(
        mesh, extrinsics, intrinsics,
        options={'resolution': resolution, 'bg_color': (0, 0, 0)},
        verbose=False,
    )
    return render_out


def concat_images_horizontally(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def concat_images_vertically(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)


def add_label_to_image(image: np.ndarray, label: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((10, 10), label, font=font)
    padding = 5
    draw.rectangle(
        [text_bbox[0] - padding, text_bbox[1] - padding,
         text_bbox[2] + padding, text_bbox[3] + padding],
        fill=(0, 0, 0)
    )
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.fe_steps is None:
        args.fe_steps = args.steps

    assert os.path.isdir(args.model_path), f"模型目录不存在: {args.model_path}"
    assert os.path.isfile(args.image), f"图像文件不存在: {args.image}"

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)

    # ========== 加载模型 ==========
    print(f"[INFO] 加载模型: {args.model_path}")

    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = args.model_path
    cfg.verbose = True

    class MockAccelerator:
        pass
    mock_accelerator = MockAccelerator()
    mock_accelerator.device = device

    adapter = build_pipeline_from_reference(cfg, mock_accelerator, device=device)
    pipe = adapter.pipe

    # ========== 读取输入图像 ==========
    print(f"[INFO] 读取输入图像: {args.image}")
    img = load_image(args.image)
    img.save(os.path.join(args.out, "input.png"))

    # ========== 条件编码 ==========
    print(f"[INFO] 条件编码...")
    cond_dict = adapter.prepare_image_conditions([img])
    cond_emb = cond_dict['cond'].to(device)
    uncond_emb = cond_dict.get('neg_cond', torch.zeros_like(cond_emb)).to(device)

    # ========== Stage1: 稀疏结构生成（使用 pipe 原生 API）==========
    print(f"[INFO] Stage1: 生成稀疏结构...")
    coords = pipe.sample_sparse_structure(cond_dict, num_samples=1)
    print(f"  生成坐标数量: {coords.shape[0]}")

    # ========== 获取 Stage2 参数 ==========
    slat_steps = args.steps
    slat_guidance = args.guidance
    slat_rescale_t = args.rescale_t
    slat_params = pipe.slat_sampler_params
    cfg_interval = slat_params.get("cfg_interval", (0.0, 1.0))
    cfg_min, cfg_max = cfg_interval

    in_channels = pipe.models['slat_flow_model'].in_channels

    # ========== 生成统一的相机视角 ==========
    print(f"[INFO] 生成 {args.num_views} 个统一相机视角...")
    extrinsics, intrinsics = generate_uniform_cameras(
        num_views=args.num_views,
        r=args.camera_r,
        fov=args.camera_fov,
        pitch=args.camera_pitch,
    )

    # ========== ODE 采样 ==========
    print(f"[INFO] Stage2 ODE 采样 (deterministic)...")
    g_ode = torch.Generator(device=device)
    g_ode.manual_seed(args.seed)

    x_t_ode = adapter.sparse.init_latents(
        coords=coords,
        in_channels=in_channels,
        generator=g_ode,
    )

    scheduler_ode = adapter.sparse.scheduler()
    scheduler_ode.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)

    x_0_ode = rollout_ode(
        adapter, x_t_ode, cond_emb, uncond_emb,
        scheduler_ode, slat_guidance, cfg_min, cfg_max, device,
    )

    # 反归一化
    norm = pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)
    mean = torch.tensor(norm['mean'])[None].to(device)
    denorm_feats_ode = x_0_ode.feats * std + mean
    x_0_ode_denorm = SparseTensor(coords=x_0_ode.coords, feats=denorm_feats_ode)

    decoded_ode = adapter.sparse.decode(x_0_ode_denorm, formats=['gaussian', 'mesh'])
    gaussians_ode = decoded_ode['gaussian']
    meshes_ode = decoded_ode['mesh']
    print(f"  ODE: 生成 {len(gaussians_ode)} 个 gaussian, {len(meshes_ode)} 个 mesh")

    # ========== FlowEdit 采样 ==========
    print(f"[INFO] Stage2 FlowEdit 采样 (fe_steps={args.fe_steps}, n_max={args.fe_n_max}, "
          f"cfg_tgt={args.fe_cfg_tgt}, cfg_src={args.fe_cfg_src})...")

    # x_src = ODE 产出的 normalized latent
    x_src = x_0_ode  # 已经是 normalized 的

    g_fe = torch.Generator(device=device)
    g_fe.manual_seed(args.seed + 1)

    scheduler_fe = adapter.sparse.scheduler()
    scheduler_fe.set_timesteps(args.fe_steps, device=device, rescale_t=slat_rescale_t)

    x_0_fe = rollout_flowedit(
        adapter, x_src, cond_emb, uncond_emb,
        scheduler_fe,
        cfg_scale_tgt=args.fe_cfg_tgt,
        cfg_scale_src=args.fe_cfg_src,
        n_max=args.fe_n_max,
        device=device,
        generator=g_fe,
    )

    # 反归一化
    denorm_feats_fe = x_0_fe.feats * std + mean
    x_0_fe_denorm = SparseTensor(coords=x_0_fe.coords, feats=denorm_feats_fe)

    decoded_fe = adapter.sparse.decode(x_0_fe_denorm, formats=['gaussian', 'mesh'])
    gaussians_fe = decoded_fe['gaussian']
    meshes_fe = decoded_fe['mesh']
    print(f"  FlowEdit: 生成 {len(gaussians_fe)} 个 gaussian, {len(meshes_fe)} 个 mesh")

    # ========== 渲染对比图 ==========
    print(f"[INFO] 渲染统一视角对比图...")

    for idx in range(len(gaussians_ode)):
        gs_ode = gaussians_ode[idx]
        gs_fe = gaussians_fe[idx]

        # 3DGS color 渲染
        render_gs_ode = render_mesh_multiview(
            gs_ode, extrinsics, intrinsics, resolution=args.render_resolution
        )
        render_gs_fe = render_mesh_multiview(
            gs_fe, extrinsics, intrinsics, resolution=args.render_resolution
        )

        # Mesh normal 渲染
        render_mesh_ode = render_mesh_multiview(
            meshes_ode[idx], extrinsics, intrinsics, resolution=args.render_resolution
        )
        render_mesh_fe = render_mesh_multiview(
            meshes_fe[idx], extrinsics, intrinsics, resolution=args.render_resolution
        )

        # ---- 3DGS Color 对比图（ODE vs FlowEdit）----
        gs_views = []
        for v_idx in range(args.num_views):
            img_ode = render_gs_ode['color'][v_idx]
            img_fe = render_gs_fe['color'][v_idx]

            img_ode_labeled = add_label_to_image(img_ode, f"ODE View-{v_idx}")
            img_fe_labeled = add_label_to_image(img_fe, f"FlowEdit View-{v_idx}")

            view_comparison = concat_images_vertically([img_ode_labeled, img_fe_labeled])
            gs_views.append(view_comparison)

        gs_comparison = concat_images_horizontally(gs_views)
        gs_path = os.path.join(args.out, f"comparison_gs_color_{idx}.png")
        Image.fromarray(gs_comparison).save(gs_path)
        print(f"  保存 3DGS Color 对比图: {gs_path}")

        # ---- Mesh Normal 对比图（ODE vs FlowEdit）----
        mesh_views = []
        for v_idx in range(args.num_views):
            img_ode = render_mesh_ode['normal'][v_idx]
            img_fe = render_mesh_fe['normal'][v_idx]

            img_ode_labeled = add_label_to_image(img_ode, f"ODE View-{v_idx}")
            img_fe_labeled = add_label_to_image(img_fe, f"FlowEdit View-{v_idx}")

            view_comparison = concat_images_vertically([img_ode_labeled, img_fe_labeled])
            mesh_views.append(view_comparison)

        mesh_comparison = concat_images_horizontally(mesh_views)
        mesh_path = os.path.join(args.out, f"comparison_mesh_normal_{idx}.png")
        Image.fromarray(mesh_comparison).save(mesh_path)
        print(f"  保存 Mesh Normal 对比图: {mesh_path}")

        # ---- 单独保存 3DGS 多视角拼图 ----
        ode_gs_row = concat_images_horizontally(render_gs_ode['color'])
        fe_gs_row = concat_images_horizontally(render_gs_fe['color'])

        ode_gs_labeled = add_label_to_image(ode_gs_row, "ODE 3DGS Color")
        fe_gs_labeled = add_label_to_image(
            fe_gs_row,
            f"FlowEdit 3DGS (n_max={args.fe_n_max}, tgt={args.fe_cfg_tgt}, src={args.fe_cfg_src})"
        )

        Image.fromarray(ode_gs_labeled).save(os.path.join(args.out, f"ode_gs_{idx}_all_views.png"))
        Image.fromarray(fe_gs_labeled).save(os.path.join(args.out, f"flowedit_gs_{idx}_all_views.png"))

        # ---- 导出 mesh ----
        adapter.export_mesh_obj(meshes_ode[idx], os.path.join(args.out, f"ode_mesh_{idx}.obj"))
        adapter.export_mesh_obj(meshes_fe[idx], os.path.join(args.out, f"flowedit_mesh_{idx}.obj"))

    # ========== 汇总统计 ==========
    print(f"\n{'='*60}")
    print(f"[完成] ODE vs FlowEdit 对比测试")
    print(f"{'='*60}")
    print(f"  输入图像: {args.image}")
    print(f"  输出目录: {args.out}")
    print(f"  ODE 采样步数: {args.steps}")
    print(f"  ODE 引导系数: {args.guidance}")
    print(f"  FlowEdit 总步数: {args.fe_steps}")
    print(f"  FlowEdit 执行步数 (n_max): {args.fe_n_max}")
    print(f"  FlowEdit cfg_tgt: {args.fe_cfg_tgt}")
    print(f"  FlowEdit cfg_src: {args.fe_cfg_src}")
    print(f"  随机种子: {args.seed}")
    print(f"  视角数量: {args.num_views}")
    print(f"  ODE mesh 数量: {len(meshes_ode)}")
    print(f"  FlowEdit mesh 数量: {len(meshes_fe)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
