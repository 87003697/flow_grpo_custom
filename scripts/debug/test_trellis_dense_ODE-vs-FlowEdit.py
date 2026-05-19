#!/usr/bin/env python3
"""
TRELLIS Dense ODE vs FlowEdit 对比脚本
=================================================
对标 test_trellis_ODE-vs-FlowEdit.py，但对比点在 Stage 1（Dense）：

- Stage 1 ODE:      标准 Euler 采样 → z_s_ode → decode → coords_ode
- Stage 1 FlowEdit: 以 ODE 产出为 x_src 差分采样 → z_s_fe  → decode → coords_fe
- Stage 2:          两路分别用各自坐标跑相同的 Sparse ODE
- 渲染统一视角的多视图 normal/color 图像并生成对比拼图

示例：
  python scripts/debug/test_trellis_dense_ODE-vs-FlowEdit.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/dense_ode_vs_flowedit \
    --ss_steps 12 --slat_steps 12 --guidance 3.0 --seed 777 \
    --fe_steps 12 --fe_n_max 8 --fe_cfg_tgt 3.0 --fe_cfg_src -3.0
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
from edit4shape.generators.trellis.rollout.base import (
    predict_dense_velocity_with_cfg,
    _predict_dense_cond_velocity,
    _expand_cond_to_batch,
    predict_sparse_velocity_with_cfg,
)
from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler

from trellis.modules.sparse import SparseTensor
from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="TRELLIS 预训练模型目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径")
    ap.add_argument("--out", type=str, default="outputs/test_runs/dense_ode_vs_flowedit", help="输出目录")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--ss_steps", type=int, default=12, help="Stage1 Dense ODE 采样步数")
    ap.add_argument("--slat_steps", type=int, default=12, help="Stage2 Sparse ODE 采样步数")
    ap.add_argument("--guidance", type=float, default=3.0, help="ODE CFG 引导系数（Stage1 & Stage2 共用）")
    ap.add_argument("--rescale_t", type=float, default=1.0, help="时间重标")
    ap.add_argument("--seed", type=int, default=777, help="随机种子")
    ap.add_argument("--render_resolution", type=int, default=512, help="渲染分辨率")
    ap.add_argument("--num_views", type=int, default=4, help="统一视角数量")
    ap.add_argument("--camera_r", type=float, default=2.0, help="相机距离")
    ap.add_argument("--camera_fov", type=float, default=40.0, help="视场角 (度)")
    ap.add_argument("--camera_pitch", type=float, default=0.3, help="相机俯仰角 (弧度)")
    # Dense FlowEdit 参数
    ap.add_argument("--fe_steps", type=int, default=None, help="Dense FlowEdit 总步数（默认与 --ss_steps 相同）")
    ap.add_argument("--fe_n_max", type=int, default=8, help="Dense FlowEdit 实际执行步数")
    ap.add_argument("--fe_cfg_tgt", type=float, default=3.0, help="Dense FlowEdit 正向 CFG scale")
    ap.add_argument("--fe_cfg_src", type=float, default=-3.0, help="Dense FlowEdit 反向 CFG scale（负值）")
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


# =====================================================================
# Dense ODE rollout
# =====================================================================

def rollout_dense_ode(
    adapter: TrellisRefAdapter,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    ss_steps: int,
    guidance: float,
    rescale_t: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Dense ODE 采样，返回 z_s (B, C, R, R, R)。Stage 1 无 normalization。"""
    _, ss_guidance, _, cfg_min, cfg_max = adapter.dense.get_runtime_params()

    # init_latents 内部在 CPU 上生成噪声，需要 CPU generator
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)
    x_t = adapter.dense.init_latents(batch_size=1, generator=cpu_gen)  # (1, C, R, R, R)
    _, t_pairs = adapter.dense.scheduler(ss_steps, rescale_t)

    with torch.no_grad():
        for t, t_prev in tqdm(t_pairs, desc="Dense ODE", leave=False):
            t_val = float(t)
            velocity = predict_dense_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                guidance, cfg_min, cfg_max, device,
            )
            delta = t_val - float(t_prev)
            x_t = x_t - delta * velocity

    return x_t  # (1, C, R, R, R)


# =====================================================================
# Dense FlowEdit rollout
# =====================================================================

def rollout_dense_flowedit(
    adapter: TrellisRefAdapter,
    x_src: torch.Tensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    fe_steps: int,
    fe_n_max: int,
    cfg_scale_tgt: float,
    cfg_scale_src: float,
    rescale_t: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """
    Dense FlowEdit 差分采样，返回 z_edit (B, C, R, R, R)。

    x_src 为 ODE 产出的 dense latent（Stage 1 无 normalization，直接使用）。
    """
    B = x_src.shape[0]
    z_edit = x_src.clone()
    # noise 在 CUDA 上生成，使用 CUDA generator
    cuda_gen = torch.Generator(device=device)
    cuda_gen.manual_seed(seed)
    noise = torch.randn(
        x_src.shape, generator=cuda_gen,
        device=device, dtype=x_src.dtype,
    )

    _, t_pairs = adapter.dense.scheduler(fe_steps, rescale_t)
    num_steps = len(t_pairs)

    cond_input = _expand_cond_to_batch(cond_emb, B)
    uncond_input = _expand_cond_to_batch(uncond_emb, B)

    with torch.no_grad():
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Dense FlowEdit", leave=False)):
            if num_steps - i > fe_n_max:
                continue

            t_val = float(t)
            dt = float(t_prev) - t_val  # < 0

            # ---- Source Branch（反向 CFG）----
            latents_src = (1 - t_val) * x_src + t_val * noise

            v_cond_src = _predict_dense_cond_velocity(adapter, latents_src, t_val, cond_input)
            v_uncond_src = _predict_dense_cond_velocity(adapter, latents_src, t_val, uncond_input)
            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_src = (1 + cfg_scale_src) * v_cond_src - cfg_scale_src * v_uncond_src

            # ---- Target Branch（正向 CFG）----
            latents_tgt = z_edit + (latents_src - x_src)

            v_cond_tgt = _predict_dense_cond_velocity(adapter, latents_tgt, t_val, cond_input)
            v_uncond_tgt = _predict_dense_cond_velocity(adapter, latents_tgt, t_val, uncond_input)
            # Trellis CFG: (1 + scale) * cond - scale * uncond
            v_cfg_tgt = (1 + cfg_scale_tgt) * v_cond_tgt - cfg_scale_tgt * v_uncond_tgt

            # ---- 差分 Euler 步 ----
            z_edit = z_edit + dt * (v_cfg_tgt - v_cfg_src)

            # ---- Aligned noise update ----
            noise = noise - (v_cond_tgt - v_uncond_tgt) * (1.0 - t_val)

    return z_edit  # (1, C, R, R, R)


# =====================================================================
# Sparse ODE rollout（Stage 2）
# =====================================================================

def rollout_sparse_ode(
    adapter: TrellisRefAdapter,
    coords: torch.Tensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    slat_steps: int,
    guidance: float,
    rescale_t: float,
    device: torch.device,
    generator: torch.Generator,
) -> SparseTensor:
    """Sparse ODE 采样，返回反归一化后的 SparseTensor。"""
    pipe = adapter.pipe
    _, _, _, cfg_min, cfg_max, _ = adapter.sparse.get_runtime_params()

    in_channels = pipe.models['slat_flow_model'].in_channels
    x_t = adapter.sparse.init_latents(
        coords=coords,
        in_channels=in_channels,
        generator=generator,
    )

    scheduler = adapter.sparse.scheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=rescale_t)

    steps = list(scheduler.timesteps)[:-1]
    B = cond_emb.shape[0]

    with torch.no_grad():
        for t in tqdm(steps, desc="Sparse ODE", leave=False):
            t_val = float(t.item())
            velocity = predict_sparse_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                guidance, cfg_min, cfg_max, device,
            )
            x_t = scheduler.step(velocity, t, x_t).prev_sample

    # 反归一化
    norm = pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)
    mean = torch.tensor(norm['mean'])[None].to(device)
    denorm_feats = x_t.feats * std + mean
    return x_t.replace(denorm_feats)


# =====================================================================
# 渲染与图像处理工具
# =====================================================================

def render_mesh_multiview(
    mesh: Any,
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    resolution: int = 512,
) -> Dict[str, List[np.ndarray]]:
    return render_frames(
        mesh, extrinsics, intrinsics,
        options={'resolution': resolution, 'bg_color': (0, 0, 0)},
        verbose=False,
    )


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


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.fe_steps is None:
        args.fe_steps = args.ss_steps

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

    # ========== 读取输入图像 ==========
    print(f"[INFO] 读取输入图像: {args.image}")
    img = load_image(args.image)
    img.save(os.path.join(args.out, "input.png"))

    # ========== 条件编码 ==========
    print("[INFO] 条件编码...")
    cond_dict = adapter.prepare_image_conditions([img])
    cond_emb = cond_dict['cond'].to(device)
    uncond_emb = cond_dict.get('neg_cond', torch.zeros_like(cond_emb)).to(device)

    # ========== 生成统一相机视角 ==========
    print(f"[INFO] 生成 {args.num_views} 个统一相机视角...")
    extrinsics, intrinsics = generate_uniform_cameras(
        num_views=args.num_views,
        r=args.camera_r,
        fov=args.camera_fov,
        pitch=args.camera_pitch,
    )

    # ========== Stage 1 ODE ==========
    print("[INFO] Stage1 Dense ODE 采样...")

    z_s_ode = rollout_dense_ode(
        adapter, cond_emb, uncond_emb,
        ss_steps=args.ss_steps,
        guidance=args.guidance,
        rescale_t=args.rescale_t,
        device=device,
        seed=args.seed,
    )

    # Stage 1 ODE → 坐标
    coords_ode = adapter.dense.decode_to_coords(z_s_ode)
    print(f"  ODE 坐标数量: {coords_ode.shape[0]}")

    # ========== Stage 1 FlowEdit ==========
    print(f"[INFO] Stage1 Dense FlowEdit 采样 "
          f"(fe_steps={args.fe_steps}, n_max={args.fe_n_max}, "
          f"cfg_tgt={args.fe_cfg_tgt}, cfg_src={args.fe_cfg_src})...")

    z_s_fe = rollout_dense_flowedit(
        adapter, z_s_ode, cond_emb, uncond_emb,
        fe_steps=args.fe_steps,
        fe_n_max=args.fe_n_max,
        cfg_scale_tgt=args.fe_cfg_tgt,
        cfg_scale_src=args.fe_cfg_src,
        rescale_t=args.rescale_t,
        device=device,
        seed=args.seed + 1,
    )

    # Stage 1 FlowEdit → 坐标
    coords_fe = adapter.dense.decode_to_coords(z_s_fe)
    print(f"  FlowEdit 坐标数量: {coords_fe.shape[0]}")

    # ========== Stage 2 ODE（两路各自独立）==========
    slat_rescale_t = args.rescale_t

    print("[INFO] Stage2 Sparse ODE (ODE coords)...")
    g_slat_ode = torch.Generator(device=device)
    g_slat_ode.manual_seed(args.seed + 10)

    x_0_ode = rollout_sparse_ode(
        adapter, coords_ode, cond_emb, uncond_emb,
        slat_steps=args.slat_steps,
        guidance=args.guidance,
        rescale_t=slat_rescale_t,
        device=device,
        generator=g_slat_ode,
    )

    print("[INFO] Stage2 Sparse ODE (FlowEdit coords)...")
    g_slat_fe = torch.Generator(device=device)
    g_slat_fe.manual_seed(args.seed + 10)  # 相同种子，控制变量：仅结构不同

    x_0_fe = rollout_sparse_ode(
        adapter, coords_fe, cond_emb, uncond_emb,
        slat_steps=args.slat_steps,
        guidance=args.guidance,
        rescale_t=slat_rescale_t,
        device=device,
        generator=g_slat_fe,
    )

    # ========== 解码 ==========
    print("[INFO] 解码 Gaussian & Mesh...")
    decoded_ode = adapter.sparse.decode(x_0_ode, formats=['gaussian', 'mesh'])
    decoded_fe = adapter.sparse.decode(x_0_fe, formats=['gaussian', 'mesh'])

    gaussians_ode = decoded_ode['gaussian']
    meshes_ode = decoded_ode['mesh']
    gaussians_fe = decoded_fe['gaussian']
    meshes_fe = decoded_fe['mesh']

    print(f"  ODE:      {len(gaussians_ode)} gaussian, {len(meshes_ode)} mesh")
    print(f"  FlowEdit: {len(gaussians_fe)} gaussian, {len(meshes_fe)} mesh")

    # ========== 渲染对比图 ==========
    print("[INFO] 渲染统一视角对比图...")

    for idx in range(len(gaussians_ode)):
        gs_ode = gaussians_ode[idx]
        gs_fe = gaussians_fe[idx]
        mesh_ode = meshes_ode[idx]
        mesh_fe = meshes_fe[idx]

        render_gs_ode = render_mesh_multiview(gs_ode, extrinsics, intrinsics, args.render_resolution)
        render_gs_fe = render_mesh_multiview(gs_fe, extrinsics, intrinsics, args.render_resolution)
        render_mesh_ode_out = render_mesh_multiview(mesh_ode, extrinsics, intrinsics, args.render_resolution)
        render_mesh_fe_out = render_mesh_multiview(mesh_fe, extrinsics, intrinsics, args.render_resolution)

        # ---- 3DGS Color 对比图 ----
        gs_views = []
        for v_idx in range(args.num_views):
            ode_labeled = add_label_to_image(render_gs_ode['color'][v_idx], f"ODE View-{v_idx}")
            fe_labeled = add_label_to_image(render_gs_fe['color'][v_idx], f"FlowEdit View-{v_idx}")
            gs_views.append(concat_images_vertically([ode_labeled, fe_labeled]))

        gs_comparison = concat_images_horizontally(gs_views)
        gs_path = os.path.join(args.out, f"comparison_gs_color_{idx}.png")
        Image.fromarray(gs_comparison).save(gs_path)
        print(f"  保存 3DGS Color 对比图: {gs_path}")

        # ---- Mesh Normal 对比图 ----
        mesh_views = []
        for v_idx in range(args.num_views):
            ode_labeled = add_label_to_image(render_mesh_ode_out['normal'][v_idx], f"ODE View-{v_idx}")
            fe_labeled = add_label_to_image(render_mesh_fe_out['normal'][v_idx], f"FlowEdit View-{v_idx}")
            mesh_views.append(concat_images_vertically([ode_labeled, fe_labeled]))

        mesh_comparison = concat_images_horizontally(mesh_views)
        mesh_path = os.path.join(args.out, f"comparison_mesh_normal_{idx}.png")
        Image.fromarray(mesh_comparison).save(mesh_path)
        print(f"  保存 Mesh Normal 对比图: {mesh_path}")

        # ---- 单独保存各路多视角拼图 ----
        ode_gs_row = add_label_to_image(
            concat_images_horizontally(render_gs_ode['color']), "ODE Dense → 3DGS Color"
        )
        fe_gs_row = add_label_to_image(
            concat_images_horizontally(render_gs_fe['color']),
            f"Dense FlowEdit (n_max={args.fe_n_max}, tgt={args.fe_cfg_tgt}, src={args.fe_cfg_src}) → 3DGS"
        )
        Image.fromarray(ode_gs_row).save(os.path.join(args.out, f"ode_gs_{idx}_all_views.png"))
        Image.fromarray(fe_gs_row).save(os.path.join(args.out, f"flowedit_gs_{idx}_all_views.png"))

        # ---- 导出 mesh ----
        adapter.export_mesh_obj(mesh_ode, os.path.join(args.out, f"ode_mesh_{idx}.obj"))
        adapter.export_mesh_obj(mesh_fe, os.path.join(args.out, f"flowedit_mesh_{idx}.obj"))

    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print("[完成] Dense ODE vs FlowEdit 对比测试")
    print(f"{'='*60}")
    print(f"  输入图像       : {args.image}")
    print(f"  输出目录       : {args.out}")
    print(f"  Stage1 ODE 步数: {args.ss_steps}")
    print(f"  Stage2 ODE 步数: {args.slat_steps}")
    print(f"  引导系数       : {args.guidance}")
    print(f"  FlowEdit 总步数: {args.fe_steps}")
    print(f"  FlowEdit n_max : {args.fe_n_max}")
    print(f"  FlowEdit cfg_tgt: {args.fe_cfg_tgt}")
    print(f"  FlowEdit cfg_src: {args.fe_cfg_src}")
    print(f"  随机种子       : {args.seed}")
    print(f"  视角数量       : {args.num_views}")
    print(f"  ODE 坐标数     : {coords_ode.shape[0]}")
    print(f"  FlowEdit 坐标数: {coords_fe.shape[0]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
