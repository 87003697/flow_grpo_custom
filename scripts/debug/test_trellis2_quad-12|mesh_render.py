"""
TRELLIS.2 12-Quad Normal vs Mesh Normal 对比测试

对比基于 12-Quad 的可微 voxel 法向量渲染与 mesh 渲染结果。
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
    parser.add_argument("--save_dir", type=str, default="./outputs/quad12_comparison")
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
    import torch.nn.functional as F
    from o_voxel.convert.flexible_dual_grid import flexible_dual_grid_to_mesh
    from trellis2.representations import Mesh
    
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False
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
    # 解码获取 h 和 subs（直接调用 decoder）
    # =========================================================================
    print("\n" + "="*60)
    print("解码获取 h 和 subs")
    print("="*60)
    
    resolution = 1024
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # 获取 h 和 subs（调用父类的 forward 来获取原始 h，而不是 mesh）
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    h, subs = SparseUnetVaeDecoder.forward(decoder, shape_slat, return_subs=True)
    print(f"h: coords={h.coords.shape}, feats={h.feats.shape}")
    print(f"subs: {len(subs)} layers")
    for i, sub in enumerate(subs):
        print(f"  sub[{i}]: coords={sub.coords.shape}, feats={sub.feats.shape}")
    
    # 获取 voxel_margin
    voxel_margin = getattr(decoder, 'voxel_margin', 0.0)
    print(f"voxel_margin: {voxel_margin}")
    
    # 从 h 生成 mesh（用于对比）
    vertices = h.replace((1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin)
    intersected = h.replace(h.feats[..., 3:6] > 0)
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
    
    mesh = Mesh(*flexible_dual_grid_to_mesh(
        vertices.coords[:, 1:], vertices.feats, intersected.feats, quad_lerp.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=resolution,
        train=False
    ))
    print(f"mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    # =========================================================================
    # 渲染设置
    # =========================================================================
    print("\n" + "="*60)
    print("渲染设置")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    from edit4shape.renderers.diff_voxel_normal_quad12 import (
        Quad12RenderConfig as RenderConfig, render_normal_12quad
    )
    
    # Mesh 渲染器
    mesh_renderer = MeshRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
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
    # 渲染 Mesh Normal
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Mesh Normal")
    print("="*60)
    
    mesh_render = mesh_renderer.render(mesh, extr[0], intr[0], return_types=["normal", "mask"])
    mesh_normal = mesh_render["normal"]  # (3, H, W)
    mesh_mask = mesh_render["mask"]  # (1, H, W)
    print(f"mesh_normal: {mesh_normal.shape}, mask sum: {mesh_mask.sum().item():.0f}")
    
    # =========================================================================
    # 渲染 12-Quad Voxel Normal
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 12-Quad Voxel Normal")
    print("="*60)
    
    config = RenderConfig(
        extrinsic=extr[0],
        intrinsic=intr[0],
        resolution=resolution,
    )
    
    # 释放部分显存
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        voxel_normal, voxel_mask = render_normal_12quad(
            h, subs, config,
            voxel_margin=voxel_margin,
            use_checkpoint=False,
            temperature=2.0,
        )
    voxel_normal_chw = voxel_normal.permute(2, 0, 1)  # (3, H, W)
    voxel_mask_chw = voxel_mask.unsqueeze(0).float()  # (1, H, W)
    print(f"voxel_normal: {voxel_normal_chw.shape}, mask sum: {voxel_mask_chw.sum().item():.0f}")
    
    # =========================================================================
    # 对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    # 转换 Mesh Normal 从 [0, 1] 到 [-1, 1] 以便对比
    mesh_normal_neg11 = mesh_normal * 2 - 1  # [0, 1] -> [-1, 1]
    
    print("\n[Mesh vs 12-Quad Voxel]（范围已统一到 [-1, 1]）")
    compare_tensors("normal_mesh_vs_voxel", mesh_normal_neg11, voxel_normal_chw, atol=0.1)
    compare_tensors("mask_mesh_vs_voxel", mesh_mask.squeeze(0), voxel_mask_chw.squeeze(0), atol=1e-3)
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    def save_normal_image(normal_tensor, mask_tensor, path, is_normalized=True):
        """保存 normal 图像，背景设为白色"""
        normal = normal_tensor.detach().cpu()  # (3, H, W)
        mask = mask_tensor.detach().cpu()  # (1, H, W)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.expand_as(normal)
        
        if is_normalized:
            normal_vis = normal
        else:
            normal_vis = (normal + 1) / 2
        
        normal_vis = normal_vis * mask + (1 - mask)
        
        img = (normal_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"  保存: {path}")
    
    # 保存 Mesh 渲染结果
    save_normal_image(mesh_normal, mesh_mask, f"{args.save_dir}/normal_mesh.png", is_normalized=True)
    
    # 保存 12-Quad Voxel 渲染结果
    save_normal_image(voxel_normal_chw, voxel_mask_chw, f"{args.save_dir}/normal_voxel_quad12.png", is_normalized=False)
    
    # 保存差异图
    def save_diff_image(ref, our, path):
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
    
    save_diff_image(mesh_normal_neg11, voxel_normal_chw, f"{args.save_dir}/diff_mesh_vs_voxel.png")
    
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
        
        # Voxel 渲染
        config_i = RenderConfig(
            extrinsic=extr_i[0],
            intrinsic=intr_i[0],
            resolution=resolution,
        )
        with torch.no_grad():
            voxel_n, _ = render_normal_12quad(
                h, subs, config_i,
                voxel_margin=voxel_margin,
                use_checkpoint=False,
                temperature=2.0,
            )
        voxel_n_chw = voxel_n.permute(2, 0, 1)  # (3, H, W)
        
        # 转换范围
        mesh_n_neg11 = mesh_n * 2 - 1
        
        compare_tensors(f"yaw{yaw_i}_mesh_vs_voxel", mesh_n_neg11, voxel_n_chw, atol=0.15)
    
    # =========================================================================
    # 保存多视角拼接图
    # =========================================================================
    print("\n" + "="*60)
    print("保存多视角拼接图")
    print("="*60)
    
    # 收集多视角图像
    mesh_views = []
    voxel_views = []
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0
        )
        if isinstance(extr_i, list):
            extr_i = torch.stack(extr_i, dim=0).to(device)
            intr_i = torch.stack(intr_i, dim=0).to(device)
        
        # Mesh
        mesh_out = mesh_renderer.render(mesh, extr_i[0], intr_i[0], return_types=["normal", "mask"])
        mesh_n = mesh_out["normal"].detach().cpu()
        mesh_m = mesh_out["mask"].detach().cpu().expand_as(mesh_n)
        mesh_vis = mesh_n * mesh_m + (1 - mesh_m)
        mesh_img = (mesh_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        mesh_views.append(Image.fromarray(mesh_img).resize((256, 256)))
        
        # Voxel
        config_i = RenderConfig(
            extrinsic=extr_i[0],
            intrinsic=intr_i[0],
            resolution=resolution,
        )
        with torch.no_grad():
            voxel_n, voxel_m = render_normal_12quad(
                h, subs, config_i,
                voxel_margin=voxel_margin,
                use_checkpoint=False,
                temperature=2.0,
            )
        voxel_n_chw = voxel_n.permute(2, 0, 1).detach().cpu()
        voxel_m_chw = voxel_m.unsqueeze(0).float().detach().cpu().expand_as(voxel_n_chw)
        voxel_vis = (voxel_n_chw + 1) / 2 * voxel_m_chw + (1 - voxel_m_chw)
        voxel_img = (voxel_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        voxel_views.append(Image.fromarray(voxel_img).resize((256, 256)))
    
    # 拼接：上面一行 mesh，下面一行 voxel
    grid_width = 256 * len(yaw_angles)
    grid_height = 256 * 2
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, (mv, vv) in enumerate(zip(mesh_views, voxel_views)):
        grid_img.paste(mv, (i * 256, 0))
        grid_img.paste(vv, (i * 256, 256))
    
    grid_img.save(f"{args.save_dir}/multiview_comparison.png")
    print(f"  保存: {args.save_dir}/multiview_comparison.png")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"可视化结果已保存到: {args.save_dir}/")
    print("  - normal_mesh.png: Mesh 渲染结果")
    print("  - normal_voxel_quad12.png: 12-Quad Voxel 渲染结果")
    print("  - diff_mesh_vs_voxel.png: Mesh vs Voxel 差异图")
    print("  - multiview_comparison.png: 多视角对比（上 Mesh，下 Voxel）")


if __name__ == "__main__":
    main()
