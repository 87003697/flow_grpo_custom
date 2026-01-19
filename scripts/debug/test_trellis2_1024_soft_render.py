"""
TRELLIS.2 1024 层可微法向量渲染测试

对比：
1. Mesh Normal（Ground Truth）
2. 26-Neighbor 不可导（对照）
3. 1024 Soft 可微版本（新算法）
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


def compare_tensors(name: str, ref: torch.Tensor, our: torch.Tensor, atol=0.1):
    """对比两个 tensor，打印差异统计"""
    if ref.shape != our.shape:
        print(f"[{name}] ❌ 形状不匹配: ref={ref.shape}, our={our.shape}")
        return False
    
    diff = (ref.float() - our.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = max_diff < atol
    status = "✓" if is_close else "❌"
    print(f"[{name}] {status} max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    return is_close


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/1024_soft_test")
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
    
    torch.manual_seed(args.seed)
    
    cond_512 = pipe.get_cond([image_proc], resolution=512)
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)
    print(f"cond_512: {cond_512['cond'].shape}")
    print(f"cond_1024: {cond_1024['cond'].shape}")
    
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
    # 解码 h 和 Subs（调用父类方法，不转成 mesh）
    # =========================================================================
    print("\n" + "="*60)
    print("解码 h 和 Subs")
    print("="*60)
    
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(1024)
    
    # 调用父类方法，获取原始的 (h, subs)，不做 mesh 转换
    h, subs = SparseUnetVaeDecoder.forward(decoder, shape_slat, return_subs=True)
    print(f"h (1024): coords={h.coords.shape}, feats={h.feats.shape}")
    print(f"subs: {len(subs)} layers")
    for i, sub in enumerate(subs):
        res = 2 ** (i + 6)
        print(f"  sub[{i}] ({res}): coords={sub.coords.shape}, feats={sub.feats.shape}")
    
    # =========================================================================
    # 渲染设置
    # =========================================================================
    print("\n" + "="*60)
    print("渲染设置")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.representations import Mesh
    from o_voxel.convert import flexible_dual_grid_to_mesh
    from edit4shape.renderers.diff_voxel_normal_neighbor26 import (
        RenderConfig, 
        render_normal_26neighbor,
        render_sub_normal_soft,
    )
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    import torch.nn.functional as F
    
    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    extr = extr.to(device)
    intr = intr.to(device)
    print(f"Camera: yaw={yaw}, pitch={pitch}, r={r}, fov={fov}")
    
    # =========================================================================
    # 从 h 生成 Mesh（用于 Ground Truth）
    # =========================================================================
    print("\n" + "="*60)
    print("从 h 生成 Mesh")
    print("="*60)
    
    voxel_margin = decoder.voxel_margin  # 通常是 0.5
    vertices = h.replace((1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin)
    intersected = h.replace(h.feats[..., 3:6] > 0)
    quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
    
    mesh = Mesh(*flexible_dual_grid_to_mesh(
        vertices.coords[:, 1:], vertices.feats, intersected.feats, quad_lerp.feats,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=1024,
        train=False
    ))
    print(f"mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    # =========================================================================
    # 渲染 Mesh Normal（Ground Truth）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Mesh Normal（Ground Truth）")
    print("="*60)
    
    mesh_renderer = MeshRenderer(rendering_options={"resolution": 1024, "near": 1.0, "far": 100.0, "ssaa": 1})
    mesh_out = mesh_renderer.render(mesh, extr, intr, return_types=["normal", "mask"])
    mesh_normal = mesh_out["normal"].squeeze(0)  # (3, H, W) 范围 [0, 1]
    mesh_mask = mesh_out["mask"].squeeze(0)  # (1, H, W)
    print(f"mesh_normal: {mesh_normal.shape}, mask sum: {mesh_mask.sum().item():.0f}")
    
    # 转换到 [-1, 1]
    mesh_normal_neg11 = mesh_normal * 2 - 1  # [0, 1] -> [-1, 1]
    
    # =========================================================================
    # 获取 1024 层坐标（直接从 h 获取）
    # =========================================================================
    print("\n" + "="*60)
    print("获取 1024 层坐标")
    print("="*60)
    
    coords_1024 = h.coords[:, 1:]  # (M, 3)，去掉 batch 维度
    print(f"1024 层 voxel: {coords_1024.shape[0]}")
    
    # =========================================================================
    # 释放显存
    # =========================================================================
    del mesh, pipe
    torch.cuda.empty_cache()
    print(f"释放 mesh 和 pipeline 后的显存")
    
    # =========================================================================
    # 渲染 26-Neighbor 不可导版本（对照）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 26-Neighbor 不可导版本（1024 层）")
    print("="*60)
    
    config_1024 = RenderConfig(
        extrinsic=extr,
        intrinsic=intr,
        resolution=1024,
        ssaa=1,
        near=1.0,
        far=100.0,
    )
    
    with torch.no_grad():
        hard_normal, hard_mask = render_normal_26neighbor(
            coords_1024, config_1024, target_size=None
        )
    hard_normal_chw = hard_normal.permute(2, 0, 1)  # (3, H, W)
    print(f"hard_normal: {hard_normal_chw.shape}, mask sum: {hard_mask.sum().item():.0f}")
    
    # =========================================================================
    # 渲染 1024 Soft 可微版本（新算法）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 1024 Soft 可微版本")
    print("="*60)
    
    soft_normal, soft_mask = render_sub_normal_soft(
        subs, config_1024, h=h, voxel_resolution=1024, target_size=None
    )
    soft_normal_chw = soft_normal.permute(2, 0, 1)  # (3, H, W)
    print(f"soft_normal: {soft_normal_chw.shape}, mask sum: {soft_mask.sum().item():.0f}")
    
    # =========================================================================
    # 对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print("\n[Mesh vs 26-Neighbor Hard]（范围 [-1, 1]）")
    compare_tensors("normal_mesh_vs_hard", mesh_normal_neg11, hard_normal_chw)
    
    print("\n[Mesh vs 1024 Soft]（范围 [-1, 1]）")
    compare_tensors("normal_mesh_vs_soft", mesh_normal_neg11, soft_normal_chw)
    
    print("\n[Hard vs Soft]")
    compare_tensors("normal_hard_vs_soft", hard_normal_chw, soft_normal_chw)
    
    # =========================================================================
    # 保存可视化
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化")
    print("="*60)
    
    def save_normal_image(normal_chw, mask, path):
        """保存法向量图（[-1,1] -> [0,255]）"""
        normal_01 = (normal_chw + 1) / 2  # [-1, 1] -> [0, 1]
        normal_np = normal_01.detach().cpu().permute(1, 2, 0).numpy()
        normal_np = (normal_np * 255).clip(0, 255).astype(np.uint8)
        # 应用 mask
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            normal_np = np.where(mask_np[..., None], normal_np, 255)
        Image.fromarray(normal_np).save(path)
        print(f"  保存: {path}")
    
    # mesh normal 已经是 [0, 1]
    mesh_normal_np = mesh_normal.detach().cpu().permute(1, 2, 0).numpy()
    mesh_normal_np = (mesh_normal_np * 255).clip(0, 255).astype(np.uint8)
    mesh_mask_np = mesh_mask.squeeze(0).detach().cpu().numpy() > 0.5
    mesh_normal_np = np.where(mesh_mask_np[..., None], mesh_normal_np, 255)
    Image.fromarray(mesh_normal_np).save(f"{args.save_dir}/normal_mesh.png")
    print(f"  保存: {args.save_dir}/normal_mesh.png")
    
    save_normal_image(hard_normal_chw, hard_mask, f"{args.save_dir}/normal_hard_26neighbor.png")
    save_normal_image(soft_normal_chw, soft_mask, f"{args.save_dir}/normal_soft_1024.png")
    
    # 差异图
    def save_diff_image(ref, our, path):
        diff = (ref - our).abs()
        diff_gray = diff.mean(dim=0).detach().cpu()
        diff_img = (diff_gray.numpy() * 255 * 5).clip(0, 255).astype(np.uint8)  # 放大 5 倍
        Image.fromarray(diff_img).save(path)
        print(f"  保存: {path}")
    
    save_diff_image(mesh_normal_neg11, hard_normal_chw, f"{args.save_dir}/diff_mesh_vs_hard.png")
    save_diff_image(mesh_normal_neg11, soft_normal_chw, f"{args.save_dir}/diff_mesh_vs_soft.png")
    save_diff_image(hard_normal_chw, soft_normal_chw, f"{args.save_dir}/diff_hard_vs_soft.png")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
