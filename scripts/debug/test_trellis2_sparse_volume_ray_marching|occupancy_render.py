"""
对比测试：MultiResolutionSparseVolume (Ray Marching) vs soft_voxel_render

流程：
1. 从 Decoder 获取 subs（各层 subdivision 预测）
2. 用两种渲染器分别渲染 alpha 图
3. 对比两种渲染器的结果是否一致
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 路径设置
ROOT = Path(__file__).resolve().parents[2]
ref_root = ROOT / "_reference_codes" / "TRELLIS.2"
sys.path.insert(0, str(ref_root))
sys.path.insert(0, str(ROOT))


def save_image(tensor, path):
    """保存图像"""
    img = tensor.detach().cpu()
    if img.dim() == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    img = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"  保存: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/renderer_comparison")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    resolution = 512
    
    # =========================================================================
    # 加载 Pipeline
    # =========================================================================
    print("\n" + "="*60)
    print("加载 Pipeline")
    print("="*60)
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import MeshRenderer
    
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False
    pipe.to(device)
    
    # =========================================================================
    # Shape Rollout
    # =========================================================================
    print("\n" + "="*60)
    print("Shape Rollout")
    print("="*60)
    
    image = Image.open(args.image)
    image_proc = pipe.preprocess_image(image)
    
    torch.manual_seed(args.seed)
    cond_512 = pipe.get_cond([image_proc], resolution=512)
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)
    
    coords = pipe.sample_sparse_structure(cond_512, 64, num_samples=1)
    print(f"coords: {coords.shape}")
    
    torch.manual_seed(args.seed + 1000)
    flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(cond_1024, flow_model, coords)
    print(f"shape_slat: coords={shape_slat.coords.shape}, feats={shape_slat.feats.shape}")
    
    # =========================================================================
    # 获取 Decoder 输出（h.feats 和 subs）
    # =========================================================================
    print("\n" + "="*60)
    print("获取 Decoder 输出")
    print("="*60)
    
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # 调用父类 forward 获取 h 和 subs
    parent_class = decoder.__class__.__bases__[0]  # SparseUnetVaeDecoder
    h, subs = parent_class.forward(decoder, shape_slat, return_subs=True)
    
    print(f"h.feats: {h.feats.shape}, h.coords: {h.coords.shape}")
    print(f"subs 层数: {len(subs)}")
    for i, sub in enumerate(subs):
        print(f"  subs[{i}]: coords={sub.coords.shape}, feats={sub.feats.shape}")
    
    # =========================================================================
    # 相机参数
    # =========================================================================
    num_frames = 60
    yaws = -torch.linspace(0, 2 * 3.1415, num_frames) + np.pi/2
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    
    extrinsics_list, intrinsics_list = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws.tolist(), pitch.tolist(), rs=2, fovs=40
    )
    
    # 使用第一个视角进行对比
    extr = extrinsics_list[0].to(device)
    intr = intrinsics_list[0].to(device)
    
    # =========================================================================
    # Mesh 渲染（作为 GT）
    # =========================================================================
    print("\n" + "="*60)
    print("Mesh 渲染（GT）")
    print("="*60)
    
    # 解码 Mesh
    meshes, _ = pipe.decode_shape_slat(shape_slat, resolution=resolution)
    mesh = meshes[0]
    print(f"Mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    # 渲染
    mesh_renderer = MeshRenderer(rendering_options={
        "resolution": resolution, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    mesh_out = mesh_renderer.render(mesh, extr, intr, return_types=["mask"])
    mesh_mask = mesh_out["mask"].squeeze(0)  # (H, W)
    print(f"[Mesh] mask sum: {mesh_mask.sum().item():.0f}")
    
    # =========================================================================
    # 渲染器 1: soft_voxel_render（基于投影的软体素渲染）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染器 1: soft_voxel_render")
    print("="*60)
    
    from edit4shape.renderers.soft_voxel_renderer_trellis2 import (
        expand_subdivision_to_voxels, soft_voxel_render
    )
    
    base_resolution = 64  # 第 0 层父分辨率
    render_size = 256
    
    soft_voxel_alphas = []
    
    for i, sub in enumerate(subs):
        parent_res = base_resolution * (2 ** i)
        
        # 展开 subdivision
        coords_i = sub.coords[:, 1:] if sub.coords.shape[1] == 4 else sub.coords
        positions, occupancies = expand_subdivision_to_voxels(coords_i, sub.feats, parent_res)
        
        print(f"[subs[{i}]] parent_res={parent_res}, positions={positions.shape}, "
              f"occupancies range=[{occupancies.min():.3f}, {occupancies.max():.3f}]")
        
        # 渲染
        out = soft_voxel_render(positions, occupancies, extr, intr, render_size, render_size, temperature=50.0)
        alpha_i = out['alpha']  # (render_size, render_size)
        
        soft_voxel_alphas.append(alpha_i)
        print(f"  渲染: {render_size}x{render_size}, alpha sum={alpha_i.sum().item():.0f}, max={alpha_i.max().item():.3f}")
    
    # =========================================================================
    # 渲染器 2: MultiResolutionSparseVolume（基于 Ray Marching）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染器 2: MultiResolutionSparseVolume (Ray Marching)")
    print("="*60)
    
    from edit4shape.renderers.sparse_ray_marching_trellis2 import MultiResolutionSparseVolume
    
    # 构建多分辨率稀疏体积
    sparse_volume = MultiResolutionSparseVolume(aabb=(-0.5, 0.5), device=device)
    sparse_volume.build_from_subs(subs, base_resolution=base_resolution)
    
    ray_march_alphas = []
    
    # 每层单独渲染
    for level_idx in range(len(subs)):
        level = sparse_volume.levels[level_idx]
        
        # 生成射线
        rays_o, rays_d = sparse_volume._generate_rays(extr, intr, render_size, render_size)
        
        # 采样
        num_samples = 64
        near, far = 0.5, 2.5
        t_vals = torch.linspace(near, far, num_samples, device=device)  # (S,)
        # (H, W, 1, 3) + (1,) * (H, W, 1, 1) -> (H, W, S, 3)
        pts = rays_o.unsqueeze(2) + t_vals.reshape(1, 1, -1, 1) * rays_d.unsqueeze(2)  # (H, W, S, 3)
        
        # 查询密度
        sigma = sparse_volume.query_density_at_level(level_idx, pts)  # (H, W, S)
        
        # 体渲染积分
        delta = (far - near) / num_samples
        alpha = 1 - torch.exp(-F.relu(sigma) * delta * 10)  # (H, W, S)
        T = torch.cumprod(1 - alpha + 1e-10, dim=-1)  # (H, W, S)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)  # (H, W, S)
        weights = T * alpha  # (H, W, S)
        
        alpha_map = weights.sum(dim=-1)  # (H, W)
        
        ray_march_alphas.append(alpha_map)
        print(f"[Level {level_idx}] res={level.resolution}, alpha sum={alpha_map.sum().item():.0f}, max={alpha_map.max().item():.3f}")
    
    # =========================================================================
    # 对比两种渲染器
    # =========================================================================
    print("\n" + "="*60)
    print("对比两种渲染器")
    print("="*60)
    
    for i in range(len(subs)):
        soft_alpha = soft_voxel_alphas[i]
        ray_alpha = ray_march_alphas[i]
        
        # MSE
        mse = F.mse_loss(soft_alpha.clamp(0, 1), ray_alpha.clamp(0, 1)).item()
        
        # 相关系数
        soft_flat = soft_alpha.flatten()
        ray_flat = ray_alpha.flatten()
        corr = torch.corrcoef(torch.stack([soft_flat, ray_flat]))[0, 1].item()
        
        # 与 Mesh GT 的 MSE
        mesh_mask_down = F.interpolate(
            mesh_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(render_size, render_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        mse_soft_mesh = F.mse_loss(soft_alpha.clamp(0, 1), mesh_mask_down).item()
        mse_ray_mesh = F.mse_loss(ray_alpha.clamp(0, 1), mesh_mask_down).item()
        
        print(f"[Layer {i}] soft vs ray MSE={mse:.6f}, corr={corr:.4f}")
        print(f"          soft vs mesh MSE={mse_soft_mesh:.6f}")
        print(f"          ray  vs mesh MSE={mse_ray_mesh:.6f}")
    
    # =========================================================================
    # 层次化渲染测试
    # =========================================================================
    print("\n" + "="*60)
    print("层次化渲染测试 (Coarse-to-Fine)")
    print("="*60)
    
    hierarchical_out = sparse_volume.render_hierarchical(
        extr, intr, H=render_size, W=render_size,
        coarse_samples=32, fine_samples=32, near=0.5, far=2.5
    )
    
    hierarchical_alpha = hierarchical_out['alpha']  # (H, W)
    hierarchical_depth = hierarchical_out['depth']  # (H, W)
    
    print(f"[Hierarchical] alpha sum={hierarchical_alpha.sum().item():.0f}, max={hierarchical_alpha.max().item():.3f}")
    print(f"[Hierarchical] depth range=[{hierarchical_depth.min().item():.3f}, {hierarchical_depth.max().item():.3f}]")
    
    # 与 Mesh 对比
    mesh_mask_down = F.interpolate(
        mesh_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(render_size, render_size),
        mode='bilinear', align_corners=False
    ).squeeze()
    
    mse_hierarchical = F.mse_loss(hierarchical_alpha.clamp(0, 1), mesh_mask_down).item()
    print(f"[Hierarchical] vs mesh MSE={mse_hierarchical:.6f}")
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    # Mesh mask
    save_image(mesh_mask, f"{args.save_dir}/mesh_mask.png")
    
    # 各层对比
    for i in range(len(subs)):
        soft_alpha = soft_voxel_alphas[i]
        ray_alpha = ray_march_alphas[i]
        
        # 上采样到统一分辨率
        soft_up = F.interpolate(
            soft_alpha.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        ray_up = F.interpolate(
            ray_alpha.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        # 差异图
        diff = (soft_up - ray_up).abs()
        diff_normalized = diff / (diff.max() + 1e-8)
        
        save_image(soft_up.clamp(0, 1), f"{args.save_dir}/layer{i}_soft_voxel.png")
        save_image(ray_up.clamp(0, 1), f"{args.save_dir}/layer{i}_ray_march.png")
        save_image(diff_normalized, f"{args.save_dir}/layer{i}_diff.png")
    
    # 层次化渲染结果
    hierarchical_up = F.interpolate(
        hierarchical_alpha.unsqueeze(0).unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear', align_corners=False
    ).squeeze()
    save_image(hierarchical_up.clamp(0, 1), f"{args.save_dir}/hierarchical_alpha.png")
    
    # 深度图（归一化）
    depth_normalized = (hierarchical_depth - hierarchical_depth.min()) / (hierarchical_depth.max() - hierarchical_depth.min() + 1e-8)
    depth_up = F.interpolate(
        depth_normalized.unsqueeze(0).unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear', align_corners=False
    ).squeeze()
    save_image(depth_up, f"{args.save_dir}/hierarchical_depth.png")
    
    # 拼接对比图：Mesh | soft_voxel (最后一层) | ray_march (最后一层) | hierarchical
    mesh_down = F.interpolate(
        mesh_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(resolution, resolution),
        mode='bilinear', align_corners=False
    ).squeeze()
    
    last_soft = F.interpolate(
        soft_voxel_alphas[-1].unsqueeze(0).unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear', align_corners=False
    ).squeeze().clamp(0, 1)
    
    last_ray = F.interpolate(
        ray_march_alphas[-1].unsqueeze(0).unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear', align_corners=False
    ).squeeze().clamp(0, 1)
    
    all_imgs = [
        mesh_down.unsqueeze(-1).repeat(1, 1, 3),
        last_soft.unsqueeze(-1).repeat(1, 1, 3),
        last_ray.unsqueeze(-1).repeat(1, 1, 3),
        hierarchical_up.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3),
    ]
    
    # 横向拼接
    concat = torch.cat(all_imgs, dim=1)  # (H, W*4, 3)
    save_image(concat, f"{args.save_dir}/comparison_all.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("对比图从左到右: Mesh GT | soft_voxel | ray_march | hierarchical")


if __name__ == "__main__":
    main()
