"""
TRELLIS.2 Hybrid26NormalRenderer vs MeshRenderer 对比测试

对比 Hybrid26（重心采样版）渲染结果与 MeshRenderer 渲染结果。
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 设置路径
repo_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def compare_tensors(name: str, ref: torch.Tensor, our: torch.Tensor, atol=1e-5, rtol=1e-4):
    """对比两个 tensor，打印差异统计"""
    if ref.shape != our.shape:
        print(f"[{name}] ❌ 形状不匹配: ref={ref.shape}, our={our.shape}")
        return False
    diff = (ref.float() - our.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(ref.float(), our.float(), atol=atol, rtol=rtol)
    status = "✓" if is_close else "❌"
    print(f"  [{name}] {status} max={max_diff:.4f}, mean={mean_diff:.4f}")
    return is_close


def save_normal_image(normal_hw3, mask_hw, path):
    """保存 normal 图像 (HWC [0,1])，背景白色"""
    n = normal_hw3.detach().cpu().float()
    m = mask_hw.detach().cpu().float()
    if m.dim() == 2:
        m = m.unsqueeze(-1)  # (H,W,1)
    vis = n * m + (1 - m)
    img = (vis.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/hybrid26_vs_mesh")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--render_res", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # =========================================================================
    # 1. 加载 Pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. 加载 Pipeline")
    print("=" * 60)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/"
                        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    pipe.low_vram = False
    pipe.to(device)

    # =========================================================================
    # 2. 生成 Shape SLat
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. 生成 Shape SLat")
    print("=" * 60)

    image = Image.open(args.image)
    image_proc = pipe.preprocess_image(image)

    torch.manual_seed(args.seed)
    cond_512 = pipe.get_cond([image_proc], resolution=512)
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)

    coords = pipe.sample_sparse_structure(cond_512, 64, num_samples=1)
    print(f"  coords: {coords.shape}")

    torch.manual_seed(args.seed + 1000)
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(
        cond_1024, shape_flow_model, coords,
        sampler_params=pipe.shape_slat_sampler_params,
    )
    print(f"  shape_slat: feats={shape_slat.feats.shape}")

    # =========================================================================
    # 3. 解码 → h + subs → 手动构建 mesh
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. 解码 Shape SLat")
    print("=" * 60)

    resolution = args.resolution
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    decoder.eval()

    # 调用基类 forward 获取 raw h 和 subs
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

    with torch.no_grad():
        h, subs = SparseUnetVaeDecoder.forward(
            decoder, shape_slat, return_subs=True)

    voxel_margin = decoder.voxel_margin
    print(f"  h.feats: {h.feats.shape}, voxel_margin={voxel_margin}")
    print(f"  subs: {len(subs)} layers")
    for i, sub in enumerate(subs):
        print(f"    sub[{i}]: coords={sub.coords.shape}, feats={sub.feats.shape}")

    # 构建 mesh
    from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
    from trellis2.representations.mesh import Mesh

    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * torch.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    intersected = h.replace((h.feats[..., 3:6] > 0).detach())
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))

    meshes = []
    for v, i_sp, q in zip(vertices_sp, intersected, quad_lerp):
        verts, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:], v.feats, i_sp.feats, q.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=False,
        )
        meshes.append(Mesh(verts, faces))

    mesh = meshes[0]
    print(f"  mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")

    # 准备 Hybrid26 所需的额外数据（取 batch 0）
    h_0 = h[0]
    subs_0 = [sub[0] for sub in subs]
    coords_0 = h_0.coords[:, 1:]          # (N, 3) voxel 整数坐标
    intersect_logits_0 = h_0.feats[..., 3:6]  # (N, 3)
    print(f"  coords_0: {coords_0.shape}, intersect_logits_0: {intersect_logits_0.shape}")

    # =========================================================================
    # 4. 创建渲染器
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. 创建渲染器")
    print("=" * 60)

    from trellis2.renderers import MeshRenderer
    from edit4shape.renderers.hybrid_peeled_trellis2 import Hybrid26NormalRenderer

    render_res = args.render_res
    mesh_renderer = MeshRenderer(
        rendering_options={
            "resolution": render_res, "ssaa": 1,
            "near": 1.0, "far": 100.0,
        },
        device=device,
    )
    hybrid_renderer = Hybrid26NormalRenderer(
        rendering_options={
            "resolution": render_res, "ssaa": 1,
            "near": 1.0, "far": 100.0,
            "grad_checkpoint": False,
        },
        device=device,
    )
    print(f"  render_res={render_res}")

    # =========================================================================
    # 5. 单视角渲染对比
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. 单视角渲染对比")
    print("=" * 60)

    from trellis2.utils import render_utils

    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov)
    if isinstance(extr, list):
        extr = torch.stack(extr, dim=0)
        intr = torch.stack(intr, dim=0)
    extr = extr.to(device)
    intr = intr.to(device)
    print(f"  Camera: yaw={yaw}, pitch={pitch}, r={r}, fov={fov}")

    # ---- MeshRenderer ----
    with torch.no_grad():
        mesh_out = mesh_renderer.render(
            mesh, extr[0], intr[0], return_types=["normal", "mask"])
    # MeshRenderer 输出: normal (3, H, W) [0,1], mask (H, W)
    mesh_normal_chw = mesh_out["normal"]   # (3, H, W)
    mesh_mask_hw = mesh_out["mask"]        # (H, W)
    mesh_normal_hwc = mesh_normal_chw.permute(1, 2, 0)  # (H, W, 3)
    print(f"  MeshRenderer: normal={mesh_normal_chw.shape}, mask_sum={mesh_mask_hw.sum():.0f}")

    # ---- Hybrid26NormalRenderer ----
    torch.cuda.empty_cache()
    with torch.no_grad():
        hybrid_out = hybrid_renderer.render(
            mesh=mesh,
            subs=subs_0,
            coords=coords_0,
            intersect_logits=intersect_logits_0,
            extrinsics=extr[0],
            intrinsics=intr[0],
            voxel_resolution=resolution,
            return_types=["normal", "mask"],
        )
    # Hybrid26 输出: normal (H, W, 3) [0,1], mask (H, W)
    hybrid_normal_hwc = hybrid_out["normal"]  # (H, W, 3)
    hybrid_mask_hw = hybrid_out["mask"]       # (H, W)
    print(f"  Hybrid26: normal={hybrid_normal_hwc.shape}, mask_sum={hybrid_mask_hw.sum():.0f}")

    # ---- 对比 ----
    print("\n  [单视角对比]")
    compare_tensors("normal", mesh_normal_hwc, hybrid_normal_hwc, atol=0.15)
    compare_tensors("mask", mesh_mask_hw, hybrid_mask_hw, atol=0.1)

    # ---- 保存 ----
    save_normal_image(mesh_normal_hwc, mesh_mask_hw, f"{args.save_dir}/mesh_normal.png")
    save_normal_image(hybrid_normal_hwc, hybrid_mask_hw, f"{args.save_dir}/hybrid_normal.png")
    print(f"  保存: {args.save_dir}/mesh_normal.png, hybrid_normal.png")

    # 差异图
    diff = (mesh_normal_hwc - hybrid_normal_hwc).abs()
    diff_gray = diff.mean(dim=-1).detach().cpu()
    diff_max = diff_gray.max().item()
    if diff_max > 0:
        diff_gray = diff_gray / diff_max
    diff_img = (diff_gray.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(diff_img).save(f"{args.save_dir}/diff_normal.png")
    print(f"  保存: {args.save_dir}/diff_normal.png")

    # =========================================================================
    # 6. 多视角渲染对比
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. 多视角渲染对比")
    print("=" * 60)

    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    mesh_views = []
    hybrid_views = []

    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0)
        if isinstance(extr_i, list):
            extr_i = torch.stack(extr_i, dim=0).to(device)
            intr_i = torch.stack(intr_i, dim=0).to(device)

        # MeshRenderer
        with torch.no_grad():
            m_out = mesh_renderer.render(
                mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])
        m_n = m_out["normal"].permute(1, 2, 0)  # (H,W,3)
        m_m = m_out["mask"]                      # (H,W)

        # Hybrid26
        torch.cuda.empty_cache()
        with torch.no_grad():
            h_out = hybrid_renderer.render(
                mesh=mesh, subs=subs_0, coords=coords_0,
                intersect_logits=intersect_logits_0,
                extrinsics=extr_i[0], intrinsics=intr_i[0],
                voxel_resolution=resolution,
                return_types=["normal", "mask"])
        h_n = h_out["normal"]  # (H,W,3)
        h_m = h_out["mask"]    # (H,W)

        compare_tensors(f"yaw{yaw_i}", m_n, h_n, atol=0.15)

        # 转图像
        def to_img(n_hwc, m_hw, size=256):
            n = n_hwc.detach().cpu().float()
            m = m_hw.detach().cpu().float().unsqueeze(-1)
            vis = n * m + (1 - m)
            arr = (vis.numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr).resize((size, size))

        mesh_views.append(to_img(m_n, m_m))
        hybrid_views.append(to_img(h_n, h_m))

    # 拼接：上 Mesh，下 Hybrid26
    n_views = len(yaw_angles)
    grid = Image.new('RGB', (256 * n_views, 256 * 2), (255, 255, 255))
    for i, (mv, hv) in enumerate(zip(mesh_views, hybrid_views)):
        grid.paste(mv, (i * 256, 0))
        grid.paste(hv, (i * 256, 256))
    grid.save(f"{args.save_dir}/multiview_comparison.png")
    print(f"  保存: {args.save_dir}/multiview_comparison.png")

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  结果目录: {args.save_dir}/")
    print(f"  - mesh_normal.png       : MeshRenderer 单视角")
    print(f"  - hybrid_normal.png     : Hybrid26 单视角")
    print(f"  - diff_normal.png       : 差异图")
    print(f"  - multiview_comparison.png : 多视角（上 Mesh，下 Hybrid26）")


if __name__ == "__main__":
    main()
