"""
TRELLIS.2 渲染器对比测试：MeshRenderer vs OVoxelRenderer vs SoftVoxelRenderer

对比三种渲染流程的 Normal 输出：
1. MeshRenderer：shape_slat → Mesh → nvdiffrast
2. OVoxelRenderer (o_voxel)：shape_slat → VoxelProxy → o_voxel CUDA kernel → depth_to_normal
3. SoftVoxelRenderer (纯 PyTorch)：shape_slat → VoxelProxy → soft_voxel_render → depth_to_normal
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument("--image", type=str, default="_reference_codes/TRELLIS.2/assets/example_image/image_01.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/renderer_comparison")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    resolution = 512  # 降低分辨率以节省显存
    skip_mesh = True  # 跳过 Mesh 渲染以节省显存（主要对比两种 Voxel 渲染器）
    
    # =========================================================================
    # 加载 Pipeline
    # =========================================================================
    print("\n" + "="*60)
    print("加载 Pipeline")
    print("="*60)
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    
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
    # 获取 Decoder 输出（用于 Voxel 渲染）
    # =========================================================================
    print("\n" + "="*60)
    print("获取 Decoder 输出 h.feats")
    print("="*60)
    
    from edit4shape.renderers.voxel_proxy import VoxelProxy
    
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # 调用父类的 forward 获取 h.feats
    parent_class = decoder.__class__.__bases__[0]  # SparseUnetVaeDecoder
    h, _ = parent_class.forward(decoder, shape_slat, return_subs=True)
    print(f"h.feats: {h.feats.shape}, h.coords: {h.coords.shape}")  # (N, 7), (N, 4)
    
    # 构建 VoxelProxy
    voxel_proxy = VoxelProxy.from_fdg_decoder(
        h.feats, h.coords, resolution, decoder.voxel_margin
    )
    print(f"VoxelProxy: position={voxel_proxy.position.shape}, opacities={voxel_proxy.opacities.shape}")
    print(f"  opacities range: [{voxel_proxy.opacities.min():.4f}, {voxel_proxy.opacities.max():.4f}]")
    print(f"  opacities > 0.5: {(voxel_proxy.opacities > 0.5).sum().item()}")
    
    # 获取 batch 0
    proxy_b0 = voxel_proxy.filter_by_batch(0)
    print(f"  batch 0: {proxy_b0.position.shape[0]} voxels")
    
    # 相机参数
    from trellis2.utils import render_utils
    yaw, pitch, r, fov = 180.0, 20.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    extr = extr[0].to(device)  # (4, 4)
    intr = intr[0].to(device)  # (3, 3)
    
    # =========================================================================
    # 方法 A：MeshRenderer（原流程）- 可选
    # =========================================================================
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)
    mesh_normal_vis = None
    mesh_mask_hw = None
    mesh_renderer = None
    mesh = None
    
    if not skip_mesh:
        print("\n" + "="*60)
        print("方法 A：MeshRenderer（Mesh 渲染）")
        print("="*60)
        
        from trellis2.renderers import MeshRenderer
        
        # 释放之前的 shape_slat 内存，重新生成
        del shape_slat
        torch.cuda.empty_cache()
        
        # 重新采样
        torch.manual_seed(args.seed + 1000)
        shape_slat = pipe.sample_shape_slat(cond_1024, flow_model, coords)
        
        # 解码 Mesh
        meshes, subs = pipe.decode_shape_slat(shape_slat, resolution=resolution)
        mesh = meshes[0]
        print(f"Mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
        
        # 创建 MeshRenderer
        mesh_renderer = MeshRenderer(rendering_options={
            "resolution": resolution, "ssaa": 1, "near": 1.0, "far": 100.0
        }, device=device)
        
        # 渲染
        mesh_out = mesh_renderer.render(mesh, extr, intr, return_types=["normal", "mask"])
        mesh_normal = mesh_out["normal"]  # (3, H, W)
        mesh_mask = mesh_out["mask"]  # (1, H, W)
        print(f"[Mesh] normal: {mesh_normal.shape}, mask sum: {mesh_mask.sum().item():.0f}")
        
        # 转换为可视化格式
        mesh_normal_hwc = mesh_normal.permute(1, 2, 0)  # (H, W, 3)
        mesh_mask_hw = mesh_mask.squeeze(0)  # (H, W)
        mesh_normal_vis = (-mesh_normal_hwc * 0.5 + 0.5) * mesh_mask_hw[..., None] + \
                          bg_color * (1 - mesh_mask_hw[..., None])
    else:
        print("\n" + "="*60)
        print("跳过方法 A：MeshRenderer（节省显存）")
        print("="*60)
    
    # =========================================================================
    # 方法 B：OVoxelRenderer（o_voxel CUDA kernel）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 B：OVoxelRenderer（o_voxel CUDA 渲染）")
    print("="*60)
    
    from edit4shape.renderers.ovoxel_trellis2 import DiffVoxelRenderer
    
    ovoxel_renderer = DiffVoxelRenderer(rendering_options={
        "resolution": resolution, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    ovoxel_out = ovoxel_renderer._render_single(proxy_b0, extr, intr)
    ovoxel_normal = ovoxel_out.normal  # (H, W, 3)
    ovoxel_mask = ovoxel_out.mask  # (H, W)
    ovoxel_depth = ovoxel_out.depth  # (H, W)
    print(f"[OVoxel] normal: {ovoxel_normal.shape}, mask sum: {ovoxel_mask.sum().item():.0f}")
    print(f"[OVoxel] depth range: [{ovoxel_depth[ovoxel_mask > 0.5].min():.4f}, {ovoxel_depth[ovoxel_mask > 0.5].max():.4f}]")
    
    # =========================================================================
    # 方法 C：SoftVoxelRenderer（纯 PyTorch）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 C：SoftVoxelRenderer（纯 PyTorch 渲染）")
    print("="*60)
    
    from edit4shape.renderers.soft_voxel_renderer import SoftVoxelRenderer
    
    soft_renderer = SoftVoxelRenderer(resolution=resolution, temperature=50.0)
    
    soft_out = soft_renderer.render(proxy_b0, extr, intr)
    soft_normal = soft_out.normal  # (H, W, 3)
    soft_mask = soft_out.mask  # (H, W)
    soft_depth = soft_out.depth  # (H, W)
    soft_alpha = soft_out.alpha  # (H, W)
    print(f"[Soft] normal: {soft_normal.shape}, mask sum: {soft_mask.sum().item():.0f}")
    print(f"[Soft] alpha range: [{soft_alpha.min():.4f}, {soft_alpha.max():.4f}]")
    print(f"[Soft] alpha > 0 pixels: {(soft_alpha > 0).sum().item()}")
    if soft_mask.sum() > 0:
        valid_depth = soft_depth[soft_mask > 0.5]
        if valid_depth.numel() > 0:
            print(f"[Soft] depth range: [{valid_depth.min():.4f}, {valid_depth.max():.4f}]")
    
    # =========================================================================
    # 对比渲染结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比渲染结果")
    print("="*60)
    
    if mesh_normal_vis is not None:
        print("\n--- Mesh vs OVoxel ---")
        compare_tensors("Mesh vs OVoxel normal", mesh_normal_vis, ovoxel_normal, atol=0.1)
        compare_tensors("Mesh vs OVoxel mask", mesh_mask_hw, ovoxel_mask, atol=0.1)
        
        print("\n--- Mesh vs Soft ---")
        compare_tensors("Mesh vs Soft normal", mesh_normal_vis, soft_normal, atol=0.1)
        compare_tensors("Mesh vs Soft mask", mesh_mask_hw, soft_mask, atol=0.1)
    
    print("\n--- OVoxel vs Soft ---")
    compare_tensors("OVoxel vs Soft normal", ovoxel_normal, soft_normal, atol=0.1)
    compare_tensors("OVoxel vs Soft mask", ovoxel_mask, soft_mask, atol=0.1)
    
    # =========================================================================
    # 测试 SoftVoxelRenderer 梯度
    # =========================================================================
    print("\n" + "="*60)
    print("测试 SoftVoxelRenderer 梯度")
    print("="*60)
    
    # 重新构建 VoxelProxy（带梯度）
    h_feats_grad = h.feats.detach().clone().requires_grad_(True)
    voxel_proxy_grad = VoxelProxy.from_fdg_decoder(
        h_feats_grad, h.coords.detach(), resolution, decoder.voxel_margin
    )
    proxy_b0_grad = voxel_proxy_grad.filter_by_batch(0)
    
    # 渲染
    soft_out_grad = soft_renderer.render(proxy_b0_grad, extr, intr)
    
    # 计算 loss
    loss = soft_out_grad.normal.sum() + soft_out_grad.depth.sum()
    print(f"loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    if h_feats_grad.grad is not None:
        grad_norm = h_feats_grad.grad.norm().item()
        grad_nonzero = (h_feats_grad.grad.abs() > 1e-10).sum().item()
        print(f"✅ h_feats 梯度正常: norm={grad_norm:.6f}, nonzero={grad_nonzero}")
    else:
        print("❌ h_feats 梯度为 None")
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
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
    
    if mesh_normal_vis is not None:
        save_image(mesh_normal_vis, f"{args.save_dir}/normal_mesh.png")
    save_image(ovoxel_normal, f"{args.save_dir}/normal_ovoxel.png")
    save_image(soft_normal, f"{args.save_dir}/normal_soft.png")
    
    # 差异图
    diff_ovoxel_soft = (ovoxel_normal - soft_normal).abs().mean(dim=-1)
    
    if mesh_normal_vis is not None:
        diff_mesh_ovoxel = (mesh_normal_vis - ovoxel_normal).abs().mean(dim=-1)
        diff_mesh_soft = (mesh_normal_vis - soft_normal).abs().mean(dim=-1)
        max_diff = max(diff_mesh_ovoxel.max(), diff_mesh_soft.max(), diff_ovoxel_soft.max()) + 1e-8
        save_image(diff_mesh_ovoxel / max_diff, f"{args.save_dir}/diff_mesh_vs_ovoxel.png")
        save_image(diff_mesh_soft / max_diff, f"{args.save_dir}/diff_mesh_vs_soft.png")
    else:
        max_diff = diff_ovoxel_soft.max() + 1e-8
    
    save_image(diff_ovoxel_soft / max_diff, f"{args.save_dir}/diff_ovoxel_vs_soft.png")
    
    # 深度图对比
    if ovoxel_mask.sum() > 0:
        ovoxel_depth_vis = ovoxel_depth / (ovoxel_depth.max() + 1e-8) * ovoxel_mask
        save_image(ovoxel_depth_vis, f"{args.save_dir}/depth_ovoxel.png")
    
    if soft_mask.sum() > 0:
        soft_depth_vis = soft_depth / (soft_depth.max() + 1e-8) * soft_mask
        save_image(soft_depth_vis, f"{args.save_dir}/depth_soft.png")
    
    # alpha 图对比
    save_image(ovoxel_mask, f"{args.save_dir}/mask_ovoxel.png")
    save_image(soft_mask, f"{args.save_dir}/mask_soft.png")
    save_image(soft_alpha.clamp(0, 1), f"{args.save_dir}/alpha_soft.png")
    
    # =========================================================================
    # 多视角渲染对比
    # =========================================================================
    print("\n" + "="*60)
    print("多视角渲染对比")
    print("="*60)
    
    yaw_angles = [0, 90, 180, 270]
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0
        )
        extr_i = extr_i[0].to(device)
        intr_i = intr_i[0].to(device)
        
        # Mesh 渲染（如果可用）
        mesh_n_vis_i = None
        if mesh_renderer is not None and mesh is not None:
            mesh_out_i = mesh_renderer.render(mesh, extr_i, intr_i, return_types=["normal", "mask"])
            mesh_n_i = mesh_out_i["normal"].permute(1, 2, 0)
            mesh_m_i = mesh_out_i["mask"].squeeze(0)
            mesh_n_vis_i = (-mesh_n_i * 0.5 + 0.5) * mesh_m_i[..., None] + bg_color * (1 - mesh_m_i[..., None])
        
        # OVoxel 渲染
        ovoxel_out_i = ovoxel_renderer._render_single(proxy_b0, extr_i, intr_i)
        ovoxel_n_i = ovoxel_out_i.normal
        
        # Soft 渲染
        soft_out_i = soft_renderer.render(proxy_b0, extr_i, intr_i)
        soft_n_i = soft_out_i.normal
        
        print(f"\n--- yaw={yaw_i}° ---")
        if mesh_n_vis_i is not None:
            compare_tensors(f"Mesh vs OVoxel (yaw={yaw_i})", mesh_n_vis_i, ovoxel_n_i, atol=0.1)
            compare_tensors(f"Mesh vs Soft (yaw={yaw_i})", mesh_n_vis_i, soft_n_i, atol=0.1)
            save_image(mesh_n_vis_i, f"{args.save_dir}/normal_mesh_yaw{yaw_i}.png")
        compare_tensors(f"OVoxel vs Soft (yaw={yaw_i})", ovoxel_n_i, soft_n_i, atol=0.1)
        
        # 保存多视角结果
        save_image(ovoxel_n_i, f"{args.save_dir}/normal_ovoxel_yaw{yaw_i}.png")
        save_image(soft_n_i, f"{args.save_dir}/normal_soft_yaw{yaw_i}.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("  - normal_mesh.png: MeshRenderer 渲染结果")
    print("  - normal_ovoxel.png: OVoxelRenderer 渲染结果")
    print("  - normal_soft.png: SoftVoxelRenderer 渲染结果")
    print("  - diff_*.png: 各方法之间的差异图")


if __name__ == "__main__":
    main()
