"""
TRELLIS.2 渲染器对比测试：OVoxelRenderer vs DiffVoxelNormal

对比两种渲染流程的 Normal 输出：
1. OVoxelRenderer：shape_slat → VoxelProxy → o_voxel CUDA kernel → depth_to_normal（现有方法）
2. DiffVoxelNormal：shape_slat → h.feats → FDG 模式 → 可微 normal（新实现）
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    print(f"[{name}] {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    return is_close


def save_image(tensor, path, permute=False):
    """保存图像"""
    img = tensor.detach().cpu()
    if permute and img.dim() == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)  # (3, H, W) → (H, W, 3)
    if img.dim() == 2:
        img = img.unsqueeze(-1).repeat(1, 1, 3)  # (H, W) → (H, W, 3)
    img = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"  保存: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/diff_voxel_comparison")
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
    print(f"coords: {coords.shape}")  # (N, 4)
    
    # Shape 采样
    torch.manual_seed(args.seed + 1000)
    flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(cond_1024, flow_model, coords)
    print(f"shape_slat: coords={shape_slat.coords.shape}, feats={shape_slat.feats.shape}")
    
    # =========================================================================
    # 获取 Decoder 输出（用于两种渲染方式）
    # =========================================================================
    print("\n" + "="*60)
    print("获取 Decoder 输出 h.feats")
    print("="*60)
    
    from edit4shape.renderers.ovoxel_trellis2 import VoxelProxy
    
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    voxel_margin = decoder.voxel_margin
    
    # 调用父类的 forward 获取 h.feats
    parent_class = decoder.__class__.__bases__[0]  # SparseUnetVaeDecoder
    h, _ = parent_class.forward(decoder, shape_slat, return_subs=True)
    print(f"h.feats: {h.feats.shape}, h.coords: {h.coords.shape}")  # (N, 7), (N, 4)
    
    # 提取 FDG 参数（用于 DiffVoxelNormal）
    # 参考 fdg_vae.py: vertices = (1 + 2 * voxel_margin) * sigmoid(h.feats[..., 0:3]) - voxel_margin
    raw_vertices = h.feats[..., 0:3]  # (N, 3)
    dual_vertices = (1 + 2 * voxel_margin) * F.sigmoid(raw_vertices) - voxel_margin  # (N, 3)
    intersected_logits = h.feats[..., 3:6]  # (N, 3)
    voxel_coords = h.coords[:, 1:]  # (N, 3) 去掉 batch_idx
    
    print(f"dual_vertices: {dual_vertices.shape}, range: [{dual_vertices.min():.4f}, {dual_vertices.max():.4f}]")
    print(f"intersected_logits: {intersected_logits.shape}, range: [{intersected_logits.min():.4f}, {intersected_logits.max():.4f}]")
    print(f"voxel_coords: {voxel_coords.shape}, range: [{voxel_coords.min()}, {voxel_coords.max()}]")
    
    # 构建 VoxelProxy（用于 OVoxelRenderer）
    voxel_proxy = VoxelProxy.from_fdg_decoder(
        h.feats, h.coords, resolution, voxel_margin
    )
    proxy_b0 = voxel_proxy.filter_by_batch(0)
    print(f"VoxelProxy batch 0: {proxy_b0.position.shape[0]} voxels")
    
    # 相机参数（与 test_trellis2_shape+render.py 保持一致）
    # yaw=180°, pitch=0°, r=2.0, fov=40°
    from trellis2.utils import render_utils
    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    extr = extr[0].to(device)  # (4, 4)
    intr = intr[0].to(device)  # (3, 3)
    print(f"Camera: yaw={yaw}, pitch={pitch}, r={r}, fov={fov}")
    
    # =========================================================================
    # 方法 A：OVoxelRenderer（现有方法：o_voxel + depth_to_normal）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 A：OVoxelRenderer（o_voxel + depth_to_normal）")
    print("="*60)
    
    from edit4shape.renderers.ovoxel_trellis2 import DiffVoxelRenderer
    from edit4shape.renderers.base_renderer import RenderConfig as BaseRenderConfig
    
    ovoxel_config = BaseRenderConfig(resolution=resolution, ssaa=1, near=1.0, far=100.0)
    ovoxel_renderer = DiffVoxelRenderer(ovoxel_config, device=device)
    
    # 使用新的 render 接口
    ovoxel_out = ovoxel_renderer.render(proxy_b0, extr, intr, return_types=['normal', 'mask', 'depth'])
    ovoxel_normal = ovoxel_out.normal  # (H, W, 3)
    ovoxel_mask = ovoxel_out.mask  # (H, W)
    ovoxel_depth = ovoxel_out.depth  # (H, W)
    print(f"[OVoxel] normal: {ovoxel_normal.shape}, mask sum: {ovoxel_mask.sum().item():.0f}")
    if ovoxel_mask.sum() > 0:
        print(f"[OVoxel] depth range: [{ovoxel_depth[ovoxel_mask > 0.5].min():.4f}, {ovoxel_depth[ovoxel_mask > 0.5].max():.4f}]")
    
    # =========================================================================
    # 方法 B：DiffVoxelNormal（新实现：FDG 模式可微 normal）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 B：DiffVoxelNormal（FDG 模式可微 normal）")
    print("="*60)
    
    from edit4shape.renderers.diff_voxel_normal import (
        RenderConfig, render_normal_fdg, normal_to_rgb
    )
    
    # 只取 batch 0
    batch_mask = h.coords[:, 0] == 0
    coords_b0 = voxel_coords[batch_mask]  # (N_b0, 3)
    dual_vertices_b0 = dual_vertices[batch_mask]  # (N_b0, 3)
    intersected_logits_b0 = intersected_logits[batch_mask]  # (N_b0, 3)
    
    print(f"batch 0: {coords_b0.shape[0]} voxels")
    
    # 构建 RenderConfig（简化版：voxel_size/origin/grid_size 自动计算）
    config = RenderConfig(
        extrinsic=extr,
        intrinsic=intr,
        resolution=resolution,
    )
    
    # 渲染
    diff_normal, diff_mask = render_normal_fdg(
        coords_b0.int(), 
        dual_vertices_b0, 
        intersected_logits_b0, 
        config
    )
    print(f"[DiffVoxel] normal: {diff_normal.shape}, mask sum: {diff_mask.sum().item():.0f}")
    
    # 转换为可视化格式
    diff_normal_vis = normal_to_rgb(diff_normal, diff_mask)
    
    # =========================================================================
    # 对比渲染结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比渲染结果")
    print("="*60)
    
    print("\n--- OVoxel vs DiffVoxel ---")
    compare_tensors("OVoxel vs DiffVoxel normal", ovoxel_normal, diff_normal_vis, atol=0.1)
    compare_tensors("OVoxel vs DiffVoxel mask", ovoxel_mask, diff_mask.float(), atol=0.1)
    
    # =========================================================================
    # 测试 DiffVoxelNormal 梯度
    # =========================================================================
    print("\n" + "="*60)
    print("测试 DiffVoxelNormal 梯度")
    print("="*60)
    
    # 重新提取参数（带梯度）
    h_feats_grad = h.feats.detach().clone()
    raw_vertices_grad = h_feats_grad[..., 0:3].requires_grad_(True)
    intersected_logits_grad = h_feats_grad[..., 3:6].requires_grad_(True)
    
    dual_vertices_grad = (1 + 2 * voxel_margin) * F.sigmoid(raw_vertices_grad) - voxel_margin
    
    # 只取 batch 0
    batch_mask = h.coords[:, 0] == 0
    coords_b0 = voxel_coords[batch_mask].int()
    dual_vertices_b0_grad = dual_vertices_grad[batch_mask]
    intersected_logits_b0_grad = intersected_logits_grad[batch_mask]
    
    # 渲染（带梯度）
    diff_normal_grad, diff_mask_grad = render_normal_fdg(
        coords_b0, 
        dual_vertices_b0_grad, 
        intersected_logits_b0_grad, 
        config
    )
    
    # 计算 loss
    loss = diff_normal_grad[diff_mask_grad].sum()
    print(f"loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    if raw_vertices_grad.grad is not None:
        grad_norm = raw_vertices_grad.grad.norm().item()
        grad_nonzero = (raw_vertices_grad.grad.abs() > 1e-10).sum().item()
        print(f"✅ raw_vertices 梯度正常: norm={grad_norm:.6f}, nonzero={grad_nonzero}")
    else:
        print("❌ raw_vertices 梯度为 None")
    
    if intersected_logits_grad.grad is not None:
        grad_norm = intersected_logits_grad.grad.norm().item()
        grad_nonzero = (intersected_logits_grad.grad.abs() > 1e-10).sum().item()
        print(f"✅ intersected_logits 梯度正常: norm={grad_norm:.6f}, nonzero={grad_nonzero}")
    else:
        print("❌ intersected_logits 梯度为 None")
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)
    
    # OVoxel normal（已经是可视化格式）
    save_image(ovoxel_normal, f"{args.save_dir}/normal_ovoxel.png")
    
    # DiffVoxel normal
    save_image(diff_normal_vis, f"{args.save_dir}/normal_diff_voxel.png")
    
    # 差异图
    diff_map = (ovoxel_normal - diff_normal_vis).abs().mean(dim=-1)
    max_diff = diff_map.max() + 1e-8
    save_image(diff_map / max_diff, f"{args.save_dir}/diff_ovoxel_vs_diff.png")
    
    # Mask 对比
    save_image(ovoxel_mask, f"{args.save_dir}/mask_ovoxel.png")
    save_image(diff_mask.float(), f"{args.save_dir}/mask_diff_voxel.png")
    
    # 深度图
    if ovoxel_mask.sum() > 0:
        ovoxel_depth_vis = ovoxel_depth / (ovoxel_depth.max() + 1e-8) * ovoxel_mask
        save_image(ovoxel_depth_vis, f"{args.save_dir}/depth_ovoxel.png")
    
    # =========================================================================
    # 多视角渲染对比（与 test_trellis2_shape+render.py 保持一致）
    # =========================================================================
    print("\n" + "="*60)
    print("多视角渲染对比")
    print("="*60)
    
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0  # 多视角用 pitch=20°
        )
        extr_i = extr_i[0].to(device)
        intr_i = intr_i[0].to(device)
        
        # OVoxel 渲染（使用新的 render 接口）
        ovoxel_out_i = ovoxel_renderer.render(proxy_b0, extr_i, intr_i, return_types=['normal'])
        ovoxel_n_i = ovoxel_out_i.normal
        
        # DiffVoxel 渲染
        config_i = RenderConfig(
            extrinsic=extr_i,
            intrinsic=intr_i,
            resolution=resolution,
        )
        diff_n_i, diff_m_i = render_normal_fdg(
            coords_b0, 
            dual_vertices_b0, 
            intersected_logits_b0, 
            config_i
        )
        diff_n_vis_i = normal_to_rgb(diff_n_i, diff_m_i)
        
        print(f"\n--- yaw={yaw_i}° ---")
        compare_tensors(f"OVoxel vs DiffVoxel (yaw={yaw_i})", ovoxel_n_i, diff_n_vis_i, atol=0.1)
        
        # 保存多视角结果
        save_image(ovoxel_n_i, f"{args.save_dir}/normal_ovoxel_yaw{yaw_i}.png")
        save_image(diff_n_vis_i, f"{args.save_dir}/normal_diff_voxel_yaw{yaw_i}.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("  - normal_ovoxel.png: OVoxelRenderer 渲染结果（depth_to_normal）")
    print("  - normal_diff_voxel.png: DiffVoxelNormal 渲染结果（FDG 模式）")
    print("  - diff_*.png: 两种方法的差异图")


if __name__ == "__main__":
    main()
