"""
多分辨率 Occupancy 渲染 vs Mesh 渲染对比测试

流程：
1. 从 Decoder 获取 subs（各层 subdivision 预测）
2. 对每层 subs 渲染 alpha 图（使用 expand_subdivision_to_voxels + soft_voxel_render）
3. 与 Mesh 渲染的 mask 对比（验证 subs 结构预测是否与最终 Mesh 一致）
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
    parser.add_argument("--save_dir", type=str, default="./outputs/occupancy_comparison")
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
    # 相机参数（与 test_trellis2_renderer_comparison.py 一致）
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
    # 多分辨率 Occupancy 渲染
    # =========================================================================
    print("\n" + "="*60)
    print("多分辨率 Occupancy 渲染")
    print("="*60)
    
    from edit4shape.renderers.soft_voxel_renderer import (
        expand_subdivision_to_voxels, soft_voxel_render
    )
    
    base_resolution = 64  # 第 0 层父分辨率
    max_render_size = 256
    
    subs_alphas = []
    
    for i, sub in enumerate(subs):
        parent_res = base_resolution * (2 ** i)
        render_size = min(parent_res * 2, max_render_size)
        
        # 展开 subdivision
        coords_i = sub.coords[:, 1:] if sub.coords.shape[1] == 4 else sub.coords
        positions, occupancies = expand_subdivision_to_voxels(coords_i, sub.feats, parent_res)
        
        print(f"[subs[{i}]] parent_res={parent_res}, positions={positions.shape}, occupancies range=[{occupancies.min():.3f}, {occupancies.max():.3f}]")
        
        # 渲染
        out = soft_voxel_render(positions, occupancies, extr, intr, render_size, render_size, temperature=50.0)
        alpha_i = out['alpha']  # (render_size, render_size)
        
        subs_alphas.append({
            'layer': i,
            'parent_res': parent_res,
            'render_size': render_size,
            'alpha': alpha_i,
        })
        
        print(f"  渲染: {render_size}x{render_size}, alpha sum={alpha_i.sum().item():.0f}, max={alpha_i.max().item():.3f}")
    
    # =========================================================================
    # 对比渲染结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比渲染结果")
    print("="*60)
    
    for item in subs_alphas:
        i = item['layer']
        render_size = item['render_size']
        alpha_i = item['alpha']
        
        # 下采样 Mesh mask 到该层分辨率
        mesh_mask_i = F.interpolate(
            mesh_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(render_size, render_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        # 计算 IoU
        alpha_bin = (alpha_i > 0.5).float()
        mesh_bin = (mesh_mask_i > 0.5).float()
        intersection = (alpha_bin * mesh_bin).sum()
        union = ((alpha_bin + mesh_bin) > 0).float().sum()
        iou = (intersection / (union + 1e-8)).item()
        
        # MSE
        mse = F.mse_loss(alpha_i.clamp(0, 1), mesh_mask_i).item()
        
        print(f"[subs[{i}]] IoU={iou:.4f}, MSE={mse:.6f}")
    
    # =========================================================================
    # 测试梯度
    # =========================================================================
    print("\n" + "="*60)
    print("测试梯度")
    print("="*60)
    
    # 选择最后一层 subs 测试梯度
    sub_test = subs[-1]
    sub_feats_grad = sub_test.feats.detach().clone().requires_grad_(True)
    
    parent_res = base_resolution * (2 ** (len(subs) - 1))
    coords_test = sub_test.coords[:, 1:] if sub_test.coords.shape[1] == 4 else sub_test.coords
    positions, occupancies = expand_subdivision_to_voxels(coords_test, sub_feats_grad, parent_res)
    
    out_grad = soft_voxel_render(positions, occupancies, extr, intr, 128, 128, temperature=50.0)
    loss = out_grad['alpha'].sum()
    loss.backward()
    
    if sub_feats_grad.grad is not None:
        grad_norm = sub_feats_grad.grad.norm().item()
        grad_nonzero = (sub_feats_grad.grad.abs() > 1e-10).sum().item()
        print(f"✅ subs[{len(subs)-1}].feats 梯度正常: norm={grad_norm:.6f}, nonzero={grad_nonzero}")
    else:
        print(f"❌ subs[{len(subs)-1}].feats 梯度为 None")
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    # Mesh mask
    save_image(mesh_mask, f"{args.save_dir}/mesh_mask.png")
    
    # 各层 alpha
    for item in subs_alphas:
        i = item['layer']
        alpha_i = item['alpha']
        # 上采样到统一分辨率便于对比
        alpha_up = F.interpolate(
            alpha_i.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear', align_corners=False
        ).squeeze()
        save_image(alpha_up.clamp(0, 1), f"{args.save_dir}/subs_{i}_alpha.png")
    
    # 拼接对比图
    all_imgs = [mesh_mask.unsqueeze(-1).repeat(1, 1, 3)]
    for item in subs_alphas:
        alpha_up = F.interpolate(
            item['alpha'].unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear', align_corners=False
        ).squeeze().clamp(0, 1)
        all_imgs.append(alpha_up.unsqueeze(-1).repeat(1, 1, 3))
    
    # 横向拼接
    concat = torch.cat(all_imgs, dim=1)  # (H, W*n, 3)
    save_image(concat, f"{args.save_dir}/comparison_all.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")


if __name__ == "__main__":
    main()
