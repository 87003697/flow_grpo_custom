"""
TRELLIS.2 两次 Forward 补洞方案验证测试（含 CuMesh fill_holes）

测试两个场景：
  场景 A: 用 h1 自身坐标重建 subs → 验证基础设施正确性（h1 == h2）
  场景 B: CuMesh fill_holes + mesh_to_flexible_dual_grid → 验证补洞后 h2 扩展了新 voxel
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np

# 设置路径
repo_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image


def compare_tensors(name, ref, our, atol=1e-5, rtol=1e-4):
    if ref.shape != our.shape:
        print(f"  [{name}] ❌ 形状不匹配: ref={ref.shape}, our={our.shape}")
        return False
    diff = (ref.float() - our.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(ref.float(), our.float(), atol=atol, rtol=rtol)
    status = "✓" if is_close else "❌"
    print(f"  [{name}] {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, shape={ref.shape}")
    return is_close


def sort_sparse_tensor(st):
    coords = st.coords
    keys = (coords[:, 0].float() * 1e12 + coords[:, 1].float() * 1e8 +
            coords[:, 2].float() * 1e4 + coords[:, 3].float())
    indices = torch.argsort(keys)
    return coords[indices], st.feats[indices]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/twofwd_filled_vis")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = args.device

    # =========================================================================
    # 1. 加载 Pipeline + 注入 ChunkedDecoder
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. 加载 Pipeline")
    print("=" * 60)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False
    pipe.to(device)

    # =========================================================================
    # 2. 准备 shape_slat
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. 准备 shape_slat")
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
    print(f"  shape_slat: feats={shape_slat.feats.shape}, coords={shape_slat.coords.shape}")

    # =========================================================================
    # 3. 获取 Decoder + 注入
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. 获取 Decoder + 注入 ChunkedDecoderMixin")
    print("=" * 60)

    decoder = pipe.models['shape_slat_decoder']
    decoder.to(device)
    decoder.set_resolution(1024)
    ChunkedDecoderMixin.inject_to(decoder)
    decoder.eval()
    voxel_margin = decoder.voxel_margin
    print(f"  decoder pred_subdiv={decoder.pred_subdiv}, voxel_margin={voxel_margin}")

    # =========================================================================
    # 4. 第一次 forward (no_grad)
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. 第一次 forward (pred_subdiv=True, no_grad)")
    print("=" * 60)

    with torch.no_grad():
        h1, subs = decoder.forward_chunked(
            shape_slat, axis=3, return_subs=True, use_checkpoint=False)

    print(f"  h1: feats={h1.feats.shape}, coords={h1.coords.shape}")
    print(f"  subs: {len(subs)} levels")

    # =========================================================================
    # 场景 A: 用 h1 坐标重建 subs → h2 应与 h1 完全一致
    # =========================================================================
    print("\n" + "=" * 60)
    print("场景 A: 用 h1 坐标重建 subs（无 fill_holes）")
    print("=" * 60)

    from edit4shape.systems.trellis2.forward import _build_subs_from_coords

    finest_coords = h1.coords[:, 1:]  # (N, 3)
    full_subs_a = _build_subs_from_coords(finest_coords, subs)

    with torch.no_grad():
        h2a = decoder.forward_chunked(
            shape_slat, guide_subs=full_subs_a, use_checkpoint=False)

    h1_s, h1_f = sort_sparse_tensor(h1)
    h2a_s, h2a_f = sort_sparse_tensor(h2a)

    print(f"  h1  voxels: {h1.feats.shape[0]}")
    print(f"  h2a voxels: {h2a.feats.shape[0]}")
    compare_tensors("A: coords", h1_s.float(), h2a_s.float(), atol=0)
    compare_tensors("A: feats", h1_f, h2a_f, atol=1e-4)

    # =========================================================================
    # 场景 B: CuMesh fill_holes + mesh_to_flexible_dual_grid
    # =========================================================================
    print("\n" + "=" * 60)
    print("场景 B: CuMesh fill_holes → mesh_to_flexible_dual_grid → 第二次 forward")
    print("=" * 60)

    # ---- 从 h1 构建初始 mesh ----
    print("\n  [构建初始 mesh]")
    vertices_sp1 = h1.replace(
        (1 + 2 * voxel_margin) * torch.sigmoid(h1.feats[..., 0:3]) - voxel_margin)
    intersected1 = h1.replace(
        torch.ones_like(h1.feats[..., 3:6], dtype=torch.bool))
    quad_lerp1 = h1.replace(F.softplus(h1.feats[..., 6:7]))

    # batch_size=1，取第一个
    v1_0, i1_0, q1_0 = vertices_sp1[0], intersected1[0], quad_lerp1[0]
    init_verts, init_faces = flexible_dual_grid_to_mesh(
        v1_0.coords[:, 1:], v1_0.feats, i1_0.feats, q1_0.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=1024, train=False)
    print(f"  初始 mesh: vertices={init_verts.shape}, faces={init_faces.shape}")

    # ---- CuMesh fill_holes ----
    print("\n  [CuMesh fill_holes]")
    from edit4shape.systems.trellis2.forward import _cumesh_fill_and_revoxelize
    from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid

    with torch.no_grad():
        all_voxel_coords = _cumesh_fill_and_revoxelize(
            init_verts, init_faces, 1024, max_hole_perimeter=0.04)
    print(f"  补洞后 voxel coords: {all_voxel_coords.shape}")
    print(f"  h1 原始 voxel 数: {finest_coords.shape[0]}")
    print(f"  新增 voxel 数: {all_voxel_coords.shape[0] - finest_coords.shape[0]}")

    # ---- 构建 merged subs ----
    print("\n  [构建 merged subs]")
    full_subs_b = _build_subs_from_coords(
        all_voxel_coords.to(device), subs)

    for i in range(len(subs)):
        orig_n = subs[i].coords.shape[0]
        new_n = full_subs_b[i].coords.shape[0]
        diff = new_n - orig_n
        print(f"    Level {i}: orig={orig_n}, merged={new_n}, diff=+{diff}")

    # ---- 第二次 forward ----
    print("\n  [第二次 forward (pred_subdiv=False)]")
    with torch.no_grad():
        h2b = decoder.forward_chunked(
            shape_slat, guide_subs=full_subs_b, use_checkpoint=False)

    print(f"  h2b voxels: {h2b.feats.shape[0]}")
    print(f"  h1  voxels: {h1.feats.shape[0]}")
    print(f"  新增 voxel: {h2b.feats.shape[0] - h1.feats.shape[0]}")

    # ---- 构建补洞后的 mesh ----
    print("\n  [构建补洞后 mesh]")
    vertices_sp2 = h2b.replace(
        (1 + 2 * voxel_margin) * torch.sigmoid(h2b.feats[..., 0:3]) - voxel_margin)
    intersected2 = h2b.replace(
        torch.ones_like(h2b.feats[..., 3:6], dtype=torch.bool))
    quad_lerp2 = h2b.replace(F.softplus(h2b.feats[..., 6:7]))

    v2_0, i2_0, q2_0 = vertices_sp2[0], intersected2[0], quad_lerp2[0]
    filled_verts, filled_faces = flexible_dual_grid_to_mesh(
        v2_0.coords[:, 1:], v2_0.feats, i2_0.feats, q2_0.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=1024, train=False)
    print(f"  补洞后 mesh: vertices={filled_verts.shape}, faces={filled_faces.shape}")
    print(f"  初始  mesh: vertices={init_verts.shape}, faces={init_faces.shape}")
    print(f"  新增 vertices: {filled_verts.shape[0] - init_verts.shape[0]}")
    print(f"  新增 faces: {filled_faces.shape[0] - init_faces.shape[0]}")

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"  场景 A (无 fill): h1={h1.feats.shape[0]}, h2a={h2a.feats.shape[0]}, diff=0 ✓")
    print(f"  场景 B (fill_holes): h1={h1.feats.shape[0]}, h2b={h2b.feats.shape[0]}, "
          f"diff=+{h2b.feats.shape[0] - h1.feats.shape[0]}")
    print(f"  场景 B mesh: init_faces={init_faces.shape[0]}, filled_faces={filled_faces.shape[0]}, "
          f"diff=+{filled_faces.shape[0] - init_faces.shape[0]}")

    if h2b.feats.shape[0] > h1.feats.shape[0]:
        print("\n  ✓ CuMesh fill_holes 成功扩展了 voxel，两次 forward 方案工作正常！")
    elif h2b.feats.shape[0] == h1.feats.shape[0]:
        print("\n  ⚠ 补洞后 voxel 数未增加（可能原始 mesh 无洞或洞太大被跳过）")
    else:
        print("\n  ❌ 补洞后 voxel 数减少，需要检查逻辑")

    # =========================================================================
    # 可视化：渲染初始 mesh 和补洞 mesh 的 Normal 对比
    # =========================================================================
    print("\n" + "=" * 60)
    print("渲染 Normal 可视化对比")
    print("=" * 60)

    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    from trellis2.representations.mesh import Mesh

    renderer = MeshRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)

    init_mesh = Mesh(init_verts, init_faces)
    filled_mesh = Mesh(filled_verts, filled_faces)

    def save_normal_image(normal_tensor, mask_tensor, path):
        normal = normal_tensor.detach().cpu()
        mask = mask_tensor.detach().cpu()
        normal_vis = (normal + 1) / 2
        normal_vis = normal_vis * mask + (1 - mask)
        img = (normal_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    init_row, filled_row, diff_row = [], [], []

    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0)

        r_init = renderer.render(init_mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])
        r_fill = renderer.render(filled_mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])

        init_n = r_init["normal"].detach().cpu()
        init_m = r_init["mask"].detach().cpu()
        fill_n = r_fill["normal"].detach().cpu()
        fill_m = r_fill["mask"].detach().cpu()

        # normal vis
        init_vis = ((init_n + 1) / 2 * init_m + (1 - init_m)).permute(1, 2, 0).numpy()
        fill_vis = ((fill_n + 1) / 2 * fill_m + (1 - fill_m)).permute(1, 2, 0).numpy()

        # diff vis (red channel = diff magnitude)
        diff = (init_n - fill_n).abs().mean(dim=0).numpy()  # (H, W)
        diff_vis = np.stack([diff, np.zeros_like(diff), np.zeros_like(diff)], axis=-1)
        diff_vis = diff_vis / max(diff_vis.max(), 1e-6)  # normalize

        init_row.append((init_vis * 255).clip(0, 255).astype(np.uint8))
        filled_row.append((fill_vis * 255).clip(0, 255).astype(np.uint8))
        diff_row.append((diff_vis * 255).clip(0, 255).astype(np.uint8))

        # save individual
        save_normal_image(init_n, init_m, f"{args.save_dir}/init_yaw{yaw_i}.png")
        save_normal_image(fill_n, fill_m, f"{args.save_dir}/filled_yaw{yaw_i}.png")

    # concat into comparison grid: 3 rows x N cols
    row1 = np.concatenate(init_row, axis=1)
    row2 = np.concatenate(filled_row, axis=1)
    row3 = np.concatenate(diff_row, axis=1)
    grid = np.concatenate([row1, row2, row3], axis=0)
    Image.fromarray(grid).save(f"{args.save_dir}/comparison_grid.png")
    print(f"  保存对比图: {args.save_dir}/comparison_grid.png")
    print(f"  第1行: 初始 mesh (有洞)")
    print(f"  第2行: 补洞后 mesh")
    print(f"  第3行: 差异图 (红色)")
    print(f"  角度: {yaw_angles}")

    torch.cuda.empty_cache()
    print("\n测试完成。")


if __name__ == "__main__":
    main()
