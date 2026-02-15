"""
TRELLIS.2 HybridPeeled26NormalRenderer vs MeshRenderer 对比测试

对比 HybridPeeled26NormalRenderer（DepthPeeler + 26-neighbor）与 MeshRenderer 的渲染结果。
renderer 内部自动计算 face_alpha（gather+sigmoid）和 active_voxel_ids（VoxelRenderer+dilate）。
当 intersect_logits > 0 时 sigmoid ≈ 1，peeled renderer 行为应接近原版 Hybrid26。
"""

import os
import sys
import torch
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
    print(f"[{name}] {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    return is_close


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/hybrid26_peeled_comparison")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = args.device
    
    # =========================================================================
    # 加载 Pipeline
    # =========================================================================
    print("\n" + "="*60)
    print("加载 Pipeline")
    print("="*60)
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False  # 关闭低内存模式，保持模型在 GPU 上
    pipe.to(device)
    
    # =========================================================================
    # 准备数据
    # =========================================================================
    print("\n" + "="*60)
    print("准备数据")
    print("="*60)
    
    image = Image.open(args.image)
    image_proc = pipe.preprocess_image(image)
    
    # 设置种子
    torch.manual_seed(args.seed)
    
    # 获取条件编码
    cond_512 = pipe.get_cond([image_proc], resolution=512)
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)
    print(f"cond_512: {cond_512['cond'].shape}")
    print(f"cond_1024: {cond_1024['cond'].shape}")
    
    # Dense Sampling
    coords = pipe.sample_sparse_structure(cond_512, 64, num_samples=1)
    print(f"coords: {coords.shape}")
    
    # =========================================================================
    # 生成 Shape SLat
    # =========================================================================
    print("\n" + "="*60)
    print("生成 Shape SLat")
    print("="*60)
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    torch.manual_seed(args.seed + 1000)
    
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(
        cond_1024,
        shape_flow_model,
        coords,
        sampler_params=pipe.shape_slat_sampler_params,
    )
    print(f"shape_slat: feats={shape_slat.feats.shape}")
    
    # =========================================================================
    # 解码 Mesh（使用 Chunked Forward 减少显存）
    # =========================================================================
    print("\n" + "="*60)
    print("解码 Mesh（Chunked Forward）")
    print("="*60)
    
    # 先把不需要的模型移到 CPU 释放显存
    print("  释放不再需要的模型...")
    pipe.image_cond_model.cpu()
    for name, model in pipe.models.items():
        if name != 'shape_slat_decoder':
            model.cpu()
    torch.cuda.empty_cache()
    
    # 注入 ChunkedDecoderMixin 到 decoder
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(1024)
    decoder.to(device)
    ChunkedDecoderMixin.inject_to(decoder)
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    # 使用 chunked forward 解码（减小 chunk_size 以降低内存峰值）
    h, subs = decoder.forward_chunked(shape_slat, axis=3, return_subs=True)
    
    # 手动提取 mesh（复制 fdg_vae.py 的推理逻辑）
    import torch.nn.functional as F_nn
    from o_voxel.convert import flexible_dual_grid_to_mesh
    from trellis2.representations import Mesh
    
    voxel_margin = decoder.voxel_margin
    resolution = decoder.resolution
    
    vertices_sp = h.replace((1 + 2 * voxel_margin) * F_nn.sigmoid(h.feats[..., 0:3]) - voxel_margin)
    intersected = h.replace(h.feats[..., 3:6] > 0)
    quad_lerp = h.replace(F_nn.softplus(h.feats[..., 6:7]))
    
    meshes = [Mesh(*flexible_dual_grid_to_mesh(
        v.coords[:, 1:], v.feats, i.feats, q.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=resolution,
        train=False
    )) for v, i, q in zip(vertices_sp, intersected, quad_lerp)]
    
    mesh = meshes[0]
    print(f"mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    from edit4shape.renderers.hybrid_peeled_trellis2 import _MAX_FACES_PER_CHUNK
    num_faces = mesh.faces.shape[0]
    print(f"  faces={num_faces}, _MAX_FACES_PER_CHUNK={_MAX_FACES_PER_CHUNK}, "
          f"需要分 chunk: {num_faces > _MAX_FACES_PER_CHUNK} "
          f"(K={((num_faces + _MAX_FACES_PER_CHUNK - 1) // _MAX_FACES_PER_CHUNK)})")
    print(f"h.coords: {h.coords.shape}")
    print(f"subs: {len(subs)} layers")
    for i, sub in enumerate(subs):
        print(f"  sub[{i}]: coords={sub.coords.shape}, feats={sub.feats.shape}")
    
    # 获取 voxel 坐标（与 mesh.vertices 一一对应）
    voxel_coords = h.coords[:, 1:]  # (N, 3) 去掉 batch 维度
    print(f"voxel_coords: {voxel_coords.shape}")
    
    decoder.cpu()
    
    # =========================================================================
    # 构造 peeled renderer 的参数
    # =========================================================================
    print("\n" + "="*60)
    print("构造 intersect_logits（renderer 内部自动计算 face_alpha + active_voxel_ids）")
    print("="*60)
    
    # intersect_logits: (N, 3) 原始 logits，传给 renderer 内部按需 gather + sigmoid
    intersect_logits = h.feats[..., 3:6]  # (N, 3)
    print(f"intersect_logits: {intersect_logits.shape}, "
          f"min={intersect_logits.min().item():.4f}, max={intersect_logits.max().item():.4f}")
    
    # =========================================================================
    # 渲染设置
    # =========================================================================
    print("\n" + "="*60)
    print("渲染设置")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    from edit4shape.renderers.hybrid_peeled_trellis2 import HybridPeeled26NormalRenderer
    
    # Mesh 渲染器（参考基线）
    mesh_renderer = MeshRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    # HybridPeeled26 渲染器
    peeled_renderer = HybridPeeled26NormalRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0,
        "peel_layers": 8, "grad_checkpoint": False,  # 测试时不需要 checkpoint
    }, device=device)
    
    voxel_resolution = 1024
    
    # 相机参数
    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    if isinstance(extr, list):
        extr = torch.stack(extr, dim=0)
    if isinstance(intr, list):
        intr = torch.stack(intr, dim=0)
    extr = extr.to(device)
    intr = intr.to(device)
    print(f"Camera: yaw={yaw}, pitch={pitch}, r={r}, fov={fov}")
    
    # =========================================================================
    # 渲染 Mesh Normal（参考基线）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Mesh Normal（参考基线）")
    print("="*60)
    
    mesh_render = mesh_renderer.render(mesh, extr[0], intr[0], return_types=["normal", "mask"])
    mesh_normal = mesh_render["normal"]  # (3, H, W)
    mesh_mask = mesh_render["mask"]  # (H, W)
    print(f"mesh_normal: {mesh_normal.shape}, mask sum: {mesh_mask.sum().item():.0f}")
    
    # =========================================================================
    # 渲染 HybridPeeled26 Normal
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 HybridPeeled26 Normal")
    print("="*60)
    
    with torch.no_grad():
        peeled_render = peeled_renderer.render(
            mesh=mesh,
            subs=subs,
            coords=voxel_coords,
            intersect_logits=intersect_logits,
            extrinsics=extr[0],
            intrinsics=intr[0],
            voxel_resolution=voxel_resolution,
            return_types=["normal", "mask"],
        )
    
    peeled_normal = peeled_render["normal"]  # (H, W, 3)
    peeled_mask = peeled_render["mask"]      # (H, W) 或 (H, W, 1)
    if peeled_mask.dim() == 3:
        peeled_mask = peeled_mask.squeeze(-1)  # (H, W)
    print(f"peeled_normal: {peeled_normal.shape}, mask sum: {peeled_mask.sum().item():.0f}")
    
    # 转换为 (3, H, W) 格式以便对比
    peeled_normal_chw = peeled_normal.permute(2, 0, 1)  # (3, H, W)
    
    # =========================================================================
    # 对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print("\n[Mesh vs HybridPeeled26]（范围均为 [0, 1]）")
    compare_tensors("normal_mesh_vs_peeled", mesh_normal, peeled_normal_chw, atol=0.1)
    compare_tensors("mask_mesh_vs_peeled", mesh_mask, peeled_mask, atol=1e-3)
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    def save_normal_image(normal_tensor, mask_tensor, path, is_01_range=True):
        """保存 normal 图像，背景设为白色"""
        normal = normal_tensor.detach().cpu()  # (3, H, W) or (H, W, 3)
        if normal.dim() == 3 and normal.shape[-1] == 3:
            normal = normal.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        
        mask = mask_tensor.detach().cpu()  # (H, W) or (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
        mask = mask.expand_as(normal)  # (3, H, W)
        
        if is_01_range:
            normal_vis = normal
        else:
            normal_vis = (normal + 1) / 2
        
        normal_vis = normal_vis * mask + (1 - mask)
        
        img = (normal_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"  保存: {path}")
    
    # 保存 Mesh 渲染结果
    save_normal_image(mesh_normal, mesh_mask, f"{args.save_dir}/normal_mesh.png", is_01_range=True)
    
    # 保存 HybridPeeled26 渲染结果
    save_normal_image(peeled_normal_chw, peeled_mask, f"{args.save_dir}/normal_peeled.png", is_01_range=True)
    
    # 保存差异图
    def save_diff_image(ref, our, path):
        """保存差异图"""
        if ref.dim() == 3 and ref.shape[-1] == 3:
            ref = ref.permute(2, 0, 1)
        if our.dim() == 3 and our.shape[-1] == 3:
            our = our.permute(2, 0, 1)
        
        diff = (ref - our).abs()
        diff_max = diff.max().item()
        if diff_max > 0:
            diff_normalized = diff / diff_max
        else:
            diff_normalized = diff
        diff_gray = diff_normalized.mean(dim=0).detach().cpu()
        diff_img = (diff_gray.numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(diff_img).save(path)
        print(f"  保存: {path}")
    
    save_diff_image(mesh_normal, peeled_normal_chw, f"{args.save_dir}/diff_mesh_vs_peeled.png")
    
    # =========================================================================
    # 多视角渲染对比
    # =========================================================================
    print("\n" + "="*60)
    print("多视角渲染对比")
    print("="*60)
    
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0
        )
        if isinstance(extr_i, list):
            extr_i = torch.stack(extr_i, dim=0).to(device)
            intr_i = torch.stack(intr_i, dim=0).to(device)
        
        # Mesh 渲染
        mesh_out = mesh_renderer.render(mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])
        mesh_n = mesh_out["normal"]  # (3, H, W)
        
        # HybridPeeled26 渲染
        with torch.no_grad():
            peeled_out = peeled_renderer.render(
                mesh=mesh,
                subs=subs,
                coords=voxel_coords,
                intersect_logits=intersect_logits,
                extrinsics=extr_i[0],
                intrinsics=intr_i[0],
                voxel_resolution=voxel_resolution,
                return_types=["normal"],
            )
        peeled_n = peeled_out["normal"].permute(2, 0, 1)  # (3, H, W)
        
        compare_tensors(f"yaw{yaw_i}_mesh_vs_peeled", mesh_n, peeled_n, atol=0.15)
    
    # =========================================================================
    # 保存多视角拼接图
    # =========================================================================
    print("\n" + "="*60)
    print("保存多视角拼接图")
    print("="*60)
    
    # 收集多视角图像
    mesh_views = []
    peeled_views = []
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0
        )
        if isinstance(extr_i, list):
            extr_i = torch.stack(extr_i, dim=0).to(device)
            intr_i = torch.stack(intr_i, dim=0).to(device)
        
        # Mesh
        mesh_out = mesh_renderer.render(mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])
        mesh_n = mesh_out["normal"].detach().cpu()  # (3, H, W)
        mesh_m = mesh_out["mask"].detach().cpu()  # (H, W)
        if mesh_m.dim() == 2:
            mesh_m = mesh_m.unsqueeze(0)
        mesh_m = mesh_m.expand_as(mesh_n)
        mesh_vis = mesh_n * mesh_m + (1 - mesh_m)
        mesh_img = (mesh_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        mesh_views.append(Image.fromarray(mesh_img).resize((256, 256)))
        
        # HybridPeeled26
        with torch.no_grad():
            peeled_out = peeled_renderer.render(
                mesh=mesh,
                subs=subs,
                coords=voxel_coords,
                intersect_logits=intersect_logits,
                extrinsics=extr_i[0],
                intrinsics=intr_i[0],
                voxel_resolution=voxel_resolution,
                return_types=["normal", "mask"],
            )
        peeled_n = peeled_out["normal"].permute(2, 0, 1).detach().cpu()  # (3, H, W)
        peeled_m = peeled_out["mask"].detach().cpu()  # (H, W) 或 (H, W, 1)
        if peeled_m.dim() == 3:
            peeled_m = peeled_m.squeeze(-1)
        if peeled_m.dim() == 2:
            peeled_m = peeled_m.unsqueeze(0)
        peeled_m = peeled_m.expand_as(peeled_n)
        peeled_vis = peeled_n * peeled_m + (1 - peeled_m)
        peeled_img = (peeled_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        peeled_views.append(Image.fromarray(peeled_img).resize((256, 256)))
    
    # 拼接：上面一行 mesh，下面一行 peeled
    grid_width = 256 * len(yaw_angles)
    grid_height = 256 * 2
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, (mv, pv) in enumerate(zip(mesh_views, peeled_views)):
        grid_img.paste(mv, (i * 256, 0))
        grid_img.paste(pv, (i * 256, 256))
    
    grid_img.save(f"{args.save_dir}/multiview_comparison.png")
    print(f"  保存: {args.save_dir}/multiview_comparison.png")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"可视化结果已保存到: {args.save_dir}/")
    print("  - normal_mesh.png: MeshRenderer 渲染结果")
    print("  - normal_peeled.png: HybridPeeled26NormalRenderer 渲染结果")
    print("  - diff_mesh_vs_peeled.png: 差异图")
    print("  - multiview_comparison.png: 多视角对比（上 Mesh，下 HybridPeeled26）")


if __name__ == "__main__":
    main()
