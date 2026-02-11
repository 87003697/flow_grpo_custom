"""
TRELLIS.2 Chunked Decoder 对比测试

对比原始 VAE Decoder forward 与 Chunked forward 的输出一致性，
包括数值对比和渲染可视化对比。
"""

import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
import argparse

# 设置路径
repo_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
ovoxel_root = os.path.join(trellis2_ref_root, "o-voxel")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)
if ovoxel_root not in sys.path:
    sys.path.insert(0, ovoxel_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


def sort_sparse_tensor(st):
    """按坐标排序 SparseTensor，返回排序后的 coords 和 feats"""
    coords = st.coords  # [N, 4]
    # 转换成可排序的 key（batch_idx * 1e12 + x * 1e8 + y * 1e4 + z）
    keys = coords[:, 0].float() * 1e12 + coords[:, 1].float() * 1e8 + \
           coords[:, 2].float() * 1e4 + coords[:, 3].float()
    indices = torch.argsort(keys)
    return coords[indices], st.feats[indices]


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


def save_diff_image(ref, our, path):
    """保存差异热力图"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=16, help="强制指定 chunk_size（用于测试分块功能）")
    parser.add_argument("--save_dir", type=str, default="./outputs/chunked_decoder_comparison")
    args = parser.parse_args()
    
    # 开启 chunked_mixin 的 INFO 日志，以便观察每层分块情况
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
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
    print(f"shape_slat: feats={shape_slat.feats.shape}, coords={shape_slat.coords.shape}")
    
    # =========================================================================
    # 获取 Decoder 并注入 Chunked 方法
    # =========================================================================
    print("\n" + "="*60)
    print("获取 Decoder 并注入 Chunked 方法")
    print("="*60)
    
    decoder = pipe.models['shape_slat_decoder']
    decoder.to(device)
    decoder.set_resolution(1024)
    
    # 注入 chunked 方法
    ChunkedDecoderMixin.inject_to(decoder)
    print(f"已注入 ChunkedDecoderMixin")
    print(f"  - forward_chunked 方法已添加: {hasattr(decoder, 'forward_chunked')}")
    
    # =========================================================================
    # 运行两种 Forward（推理模式）
    # =========================================================================
    print("\n" + "="*60)
    print("运行两种 Forward（推理模式）")
    print("="*60)
    
    decoder.eval()
    
    print(f"\nDecoder 类型: {type(decoder)}")
    print(f"Decoder 类名: {decoder.__class__.__name__}")
    
    # FlexiDualGridVaeDecoder 的 forward 会自动转 mesh
    # 我们需要直接调用父类 SparseUnetVaeDecoder.forward 来获取 SparseTensor
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    
    with torch.no_grad():
        # 原始 forward（调用父类方法，绕过 mesh 生成）
        print("\n[原始 Forward (SparseUnetVaeDecoder)]")
        original_out, original_subs = SparseUnetVaeDecoder.forward(decoder, shape_slat, return_subs=True)
        print(f"  output: feats={original_out.feats.shape}, coords={original_out.coords.shape}")
        print(f"  subs: {len(original_subs)} layers")
        
        # Chunked forward
        print(f"\n[Chunked Forward] chunk_size_override={args.chunk_size}")
        chunked_out = decoder.forward_chunked(shape_slat, chunk_size_override=args.chunk_size)
        print(f"  output: feats={chunked_out.feats.shape}, coords={chunked_out.coords.shape}")
    
    # =========================================================================
    # 数值对比
    # =========================================================================
    print("\n" + "="*60)
    print("数值对比")
    print("="*60)
    
    # 检查形状
    print("\n[形状检查]")
    shape_match = original_out.coords.shape == chunked_out.coords.shape
    print(f"  coords 形状: original={original_out.coords.shape}, chunked={chunked_out.coords.shape}, match={shape_match}")
    print(f"  feats 形状: original={original_out.feats.shape}, chunked={chunked_out.feats.shape}")
    
    # 按坐标排序后对比（因为 chunked 合并后顺序可能不同）
    print("\n[排序后对比]")
    orig_coords_sorted, orig_feats_sorted = sort_sparse_tensor(original_out)
    chunk_coords_sorted, chunk_feats_sorted = sort_sparse_tensor(chunked_out)
    
    coords_match = compare_tensors("coords（排序后）", orig_coords_sorted.float(), chunk_coords_sorted.float(), atol=0)
    feats_match = compare_tensors("feats（排序后）", orig_feats_sorted, chunk_feats_sorted, atol=1e-4)
    
    # 直接对比（不排序，检查顺序是否一致）
    print("\n[直接对比（不排序）]")
    compare_tensors("coords（原序）", original_out.coords.float(), chunked_out.coords.float(), atol=0)
    compare_tensors("feats（原序）", original_out.feats, chunked_out.feats, atol=1e-4)
    
    # =========================================================================
    # 解码 Mesh（从 SparseTensor 输出转换为 Mesh）
    # =========================================================================
    print("\n" + "="*60)
    print("解码 Mesh")
    print("="*60)
    
    from trellis2.representations import Mesh
    from o_voxel.convert import flexible_dual_grid_to_mesh
    import torch.nn.functional as F
    
    def sparse_tensor_to_mesh(h, decoder, resolution=1024):
        """从 SparseTensor 输出转换为 Mesh（复制 FlexiDualGridVaeDecoder 的逻辑）"""
        voxel_margin = decoder.voxel_margin
        vertices = h.replace((1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin)
        intersected = h.replace(h.feats[..., 3:6] > 0)
        quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
        
        mesh = Mesh(*flexible_dual_grid_to_mesh(
            vertices.coords[:, 1:], vertices.feats, intersected.feats, quad_lerp.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=False
        ))
        return mesh
    
    # 使用原始 forward 结果解码 mesh
    print("\n[原始 Forward 结果解码]")
    orig_mesh = sparse_tensor_to_mesh(original_out, decoder, resolution=1024)
    print(f"  mesh: vertices={orig_mesh.vertices.shape}, faces={orig_mesh.faces.shape}")
    
    # 使用 chunked forward 结果解码 mesh
    print("\n[Chunked Forward 结果解码]")
    chunked_mesh = sparse_tensor_to_mesh(chunked_out, decoder, resolution=1024)
    print(f"  mesh: vertices={chunked_mesh.vertices.shape}, faces={chunked_mesh.faces.shape}")
    
    # 对比 mesh
    print("\n[Mesh 对比]")
    compare_tensors("vertices", orig_mesh.vertices, chunked_mesh.vertices, atol=1e-4)
    compare_tensors("faces", orig_mesh.faces.float(), chunked_mesh.faces.float(), atol=0)
    
    # =========================================================================
    # 渲染 Normal
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Normal")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    
    # 创建渲染器
    mesh_renderer = MeshRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    # 相机参数（参考 test_trellis2_neighbor-26|mesh_render.py）
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
    
    # 渲染原始 forward 的 mesh
    print("\n[渲染原始 Forward Mesh Normal]")
    orig_render = mesh_renderer.render(orig_mesh, extr[0], intr[0], return_types=["normal", "mask"])
    orig_normal = orig_render["normal"]  # (3, H, W)
    orig_mask = orig_render["mask"]  # (1, H, W)
    print(f"  normal: {orig_normal.shape}, mask sum: {orig_mask.sum().item():.0f}")
    
    # 渲染 chunked forward 的 mesh
    print("\n[渲染 Chunked Forward Mesh Normal]")
    chunked_render = mesh_renderer.render(chunked_mesh, extr[0], intr[0], return_types=["normal", "mask"])
    chunked_normal = chunked_render["normal"]  # (3, H, W)
    chunked_mask = chunked_render["mask"]  # (1, H, W)
    print(f"  normal: {chunked_normal.shape}, mask sum: {chunked_mask.sum().item():.0f}")
    
    # 对比渲染结果
    print("\n[渲染结果对比]")
    render_match = compare_tensors("normal", orig_normal, chunked_normal, atol=1e-3)
    compare_tensors("mask", orig_mask, chunked_mask, atol=1e-5)
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    # 保存原始 forward 渲染结果
    save_normal_image(orig_normal, orig_mask, f"{args.save_dir}/normal_original.png", is_normalized=True)
    
    # 保存 chunked forward 渲染结果
    save_normal_image(chunked_normal, chunked_mask, f"{args.save_dir}/normal_chunked.png", is_normalized=True)
    
    # 保存差异图
    save_diff_image(orig_normal, chunked_normal, f"{args.save_dir}/normal_diff.png")
    
    # 保存 feats 统计信息
    print("\n[Feats 统计]")
    print(f"  Original: mean={original_out.feats.mean():.6f}, std={original_out.feats.std():.6f}")
    print(f"  Chunked:  mean={chunked_out.feats.mean():.6f}, std={chunked_out.feats.std():.6f}")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    all_passed = coords_match and feats_match and render_match
    if all_passed:
        print("✓ 所有测试通过！Chunked Forward 与原始 Forward 输出一致。")
    else:
        print("❌ 存在差异，请检查上述对比结果。")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("  - normal_original.png: 原始 Forward 渲染结果")
    print("  - normal_chunked.png: Chunked Forward 渲染结果")
    print("  - normal_diff.png: 差异图")


if __name__ == "__main__":
    main()
