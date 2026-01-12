"""
TRELLIS.2 MeshRenderer vs DiffVoxelRenderer 渲染对比测试

对比两种渲染流程的 Normal 输出：
- MeshRenderer：shape_slat → Mesh → nvdiffrast
- DiffVoxelRenderer：shape_slat → VoxelProxy → depth_to_normal
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
    parser.add_argument("--image", type=str, default="assets/example_image/image_01.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/mesh_vs_voxel_render")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    resolution = 1024
    
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
    # 方法 A：MeshRenderer（原流程）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 A：MeshRenderer（Mesh 渲染）")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    
    # 解码 Mesh
    meshes, subs = pipe.decode_shape_slat(shape_slat, resolution=resolution)
    mesh = meshes[0]
    print(f"Mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    # 创建 MeshRenderer
    mesh_renderer = MeshRenderer(rendering_options={
        "resolution": resolution, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    # 相机参数
    yaw, pitch, r, fov = 180.0, 20.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    extr = extr[0].to(device)  # (4, 4)
    intr = intr[0].to(device)  # (3, 3)
    
    # 渲染
    mesh_out = mesh_renderer.render(mesh, extr, intr, return_types=["normal", "mask"])
    mesh_normal = mesh_out["normal"]  # (3, H, W)
    mesh_mask = mesh_out["mask"]  # (1, H, W)
    print(f"[Mesh] normal: {mesh_normal.shape}, mask sum: {mesh_mask.sum().item():.0f}")
    
    # =========================================================================
    # 方法 B：DiffVoxelRenderer（Voxel 渲染）
    # =========================================================================
    print("\n" + "="*60)
    print("方法 B：DiffVoxelRenderer（Voxel 渲染）")
    print("="*60)
    
    from edit4shape.renderers.ovoxel_trellis2 import DiffVoxelRenderer
    from edit4shape.renderers.voxel_proxy import VoxelProxy
    
    # 获取 Decoder 的原始输出 h.feats（绕过 Mesh 提取）
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
    
    # 创建 DiffVoxelRenderer
    voxel_renderer = DiffVoxelRenderer(rendering_options={
        "resolution": resolution, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    # 渲染（filter_by_batch 获取 batch 0）
    proxy_b0 = voxel_proxy.filter_by_batch(0)
    voxel_out = voxel_renderer._render_single(proxy_b0, extr, intr)
    voxel_normal = voxel_out.normal  # (H, W, 3)
    voxel_mask = voxel_out.mask  # (H, W)
    print(f"[Voxel] normal: {voxel_normal.shape}, mask sum: {voxel_mask.sum().item():.0f}")
    
    # =========================================================================
    # 对比渲染结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比渲染结果")
    print("="*60)
    
    # 统一格式：(H, W, 3)
    mesh_normal_hwc = mesh_normal.permute(1, 2, 0)  # (3, H, W) → (H, W, 3)
    mesh_mask_hw = mesh_mask.squeeze(0)  # (1, H, W) → (H, W)
    
    # 注意：MeshRenderer 的 normal 是 [-1, 1]，DiffVoxelRenderer 已经做了可视化转换 [0, 1]
    # 需要将 mesh_normal 也转换为可视化格式
    mesh_normal_vis = (-mesh_normal_hwc * 0.5 + 0.5) * mesh_mask_hw[..., None] + \
                      torch.tensor([0.5, 0.5, 1.0], device=device) * (1 - mesh_mask_hw[..., None])
    
    compare_tensors("normal", mesh_normal_vis, voxel_normal, atol=0.1)
    compare_tensors("mask", mesh_mask_hw, voxel_mask, atol=0.1)
    
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
    
    save_image(mesh_normal_vis, f"{args.save_dir}/normal_mesh.png")
    save_image(voxel_normal, f"{args.save_dir}/normal_voxel.png")
    
    # 差异图
    diff = (mesh_normal_vis - voxel_normal).abs().mean(dim=-1)  # (H, W)
    diff_normalized = diff / (diff.max() + 1e-8)
    save_image(diff_normalized, f"{args.save_dir}/normal_diff.png")
    
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
        extr_i = extr_i[0].to(device)
        intr_i = intr_i[0].to(device)
        
        # Mesh 渲染
        mesh_out_i = mesh_renderer.render(mesh, extr_i, intr_i, return_types=["normal", "mask"])
        mesh_n_i = mesh_out_i["normal"].permute(1, 2, 0)  # (H, W, 3)
        mesh_m_i = mesh_out_i["mask"].squeeze(0)  # (H, W)
        mesh_n_vis_i = (-mesh_n_i * 0.5 + 0.5) * mesh_m_i[..., None] + \
                       torch.tensor([0.5, 0.5, 1.0], device=device) * (1 - mesh_m_i[..., None])
        
        # Voxel 渲染
        voxel_out_i = voxel_renderer._render_single(proxy_b0, extr_i, intr_i)
        voxel_n_i = voxel_out_i.normal  # (H, W, 3)
        
        compare_tensors(f"normal_yaw{yaw_i}", mesh_n_vis_i, voxel_n_i, atol=0.1)
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("  - normal_mesh.png: MeshRenderer 渲染结果")
    print("  - normal_voxel.png: DiffVoxelRenderer 渲染结果")
    print("  - normal_diff.png: 差异图")


if __name__ == "__main__":
    main()
