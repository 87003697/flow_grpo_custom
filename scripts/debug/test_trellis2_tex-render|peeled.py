"""
PBR Renderer 对比测试：原生 vs 改进后分 chunk Renderer

对比链路：
  1. 加载参考实现 pipeline
  2. 生成 shape + tex → decode → MeshWithVoxel
  3. 释放 pipeline 模型以回收显存
  4. 用原生 PbrMeshRenderer（trellis2.renderers）渲染
  5. 用改进后 PbrMeshRenderer（edit4shape.renderers.pbr_peeled_trellis2）渲染
  6. 对比输出 + 保存可视化
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import time
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
    parser = argparse.ArgumentParser(description="对比原生 vs 改进后 PBR Renderer")
    parser.add_argument("--image", type=str, default="dataset/alphaimages_v2/test/test_01.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="渲染分辨率")
    parser.add_argument("--save_dir", type=str, default="./outputs/comparison_pbr_renderer")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    import gc
    import cv2

    # =================================================================
    # 1. 加载参考实现 pipeline
    # =================================================================
    print("\n" + "=" * 60)
    print("1. 加载参考实现 pipeline")
    print("=" * 60)
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False
    pipe.to(device)

    # =================================================================
    # 2. 生成 shape + tex → MeshWithVoxel
    # =================================================================
    print("\n" + "=" * 60)
    print("2. 生成 shape + tex → MeshWithVoxel")
    print("=" * 60)
    
    image = Image.open(args.image)
    image_proc = pipe.preprocess_image(image)
    
    torch.manual_seed(args.seed)
    
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)
    print(f"cond_1024: {cond_1024['cond'].shape}")
    
    coords = pipe.sample_sparse_structure(cond_1024, 64, num_samples=1)
    print(f"coords: {coords.shape}")
    
    # 采样 shape_slat
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(cond_1024, shape_flow_model, coords)
    print(f"shape_slat: feats={shape_slat.feats.shape}")

    # 采样 tex_slat
    tex_flow_model = pipe.models['tex_slat_flow_model_1024']
    tex_slat = pipe.sample_tex_slat(cond_1024, tex_flow_model, shape_slat)
    print(f"tex_slat: feats={tex_slat.feats.shape}")

    # 解码 shape
    meshes, subs = pipe.decode_shape_slat(shape_slat, resolution=1024)
    mesh = meshes[0]
    print(f"mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")

    # 解码 tex → MeshWithVoxel
    tex_voxels = pipe.decode_tex_slat(tex_slat, subs)
    print(f"tex_voxels: feats={tex_voxels[0].feats.shape}")
    
    from trellis2.representations import MeshWithVoxel
    mesh_voxel = MeshWithVoxel(
        mesh.vertices, mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 1024,
        coords=tex_voxels[0].coords[:, 1:],
        attrs=tex_voxels[0].feats,
        voxel_shape=torch.Size([*tex_voxels[0].shape, *tex_voxels[0].spatial_shape]),
        layout=pipe.pbr_attr_layout
    )
    print(f"MeshWithVoxel attrs: {mesh_voxel.attrs.shape}, "
          f"range=[{mesh_voxel.attrs.min():.4f}, {mesh_voxel.attrs.max():.4f}]")

    # =================================================================
    # 3. 释放 pipeline 模型以回收显存
    # =================================================================
    print("\n" + "=" * 60)
    print("3. 释放 pipeline 模型以回收显存")
    print("=" * 60)

    del shape_slat, tex_slat, tex_voxels, subs, meshes, coords
    del cond_1024, shape_flow_model, tex_flow_model, image_proc
    for k in list(pipe.models.keys()):
            pipe.models[k].cpu()
    pipe.image_cond_model.cpu()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    free_mem = torch.cuda.mem_get_info(device)[0] / 1024**3
    print(f"释放后 GPU 可用显存: {free_mem:.2f} GiB")

    # =================================================================
    # 4. 准备渲染参数
    # =================================================================
    print("\n" + "=" * 60)
    print("4. 准备渲染参数")
    print("=" * 60)

    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    
    # 加载环境贴图
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(
            cv2.imread(
                os.path.join(repo_root, '_reference_codes/TRELLIS.2/assets/hdri/forest.exr'),
                cv2.IMREAD_UNCHANGED
            ),
            cv2.COLOR_BGR2RGB
        ),
        dtype=torch.float32, device=device
    ))
    
    # 多角度渲染以充分对比
    view_configs = [
        {"yaw": 180.0, "pitch": 0.0,   "label": "front"},
        {"yaw": 270.0, "pitch": 0.0,   "label": "side"},
        {"yaw": 180.0, "pitch": 30.0,  "label": "top"},
    ]
    r, fov = 2.0, 40.0

    mesh_voxel_dev = mesh_voxel.to(device)
    print(f"渲染分辨率: {args.resolution}, 视角数: {len(view_configs)}")

    # =================================================================
    # 5. 原生 PbrMeshRenderer 渲染
    # =================================================================
    print("\n" + "=" * 60)
    print("5. 原生 PbrMeshRenderer 渲染")
    print("=" * 60)

    from trellis2.renderers import PbrMeshRenderer as RefPbrMeshRenderer

    ref_renderer = RefPbrMeshRenderer(rendering_options={
        "resolution": args.resolution,
        "ssaa": 1,
        "near": 1.0,
        "far": 100.0,
    }, device=device)

    ref_results = {}
    for vc in view_configs:
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [vc["yaw"]], [vc["pitch"]], r, fov
        )
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = ref_renderer.render(mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
        torch.cuda.synchronize()
        dt = time.time() - t0
        ref_results[vc["label"]] = result
        print(f"  [{vc['label']}] 完成, 耗时 {dt:.3f}s, "
              f"shaded={result['shaded'].shape}, alpha range=[{result['alpha'].min():.4f}, {result['alpha'].max():.4f}]")

    # 释放原生 renderer
    del ref_renderer
    torch.cuda.empty_cache()

    # =================================================================
    # 6. 改进后 PbrMeshRenderer 渲染
    # =================================================================
    print("\n" + "=" * 60)
    print("6. 改进后 PbrMeshRenderer（chunk + depth peeling）渲染")
    print("=" * 60)

    from edit4shape.renderers.pbr_peeled_trellis2 import PbrMeshRenderer as OurPbrMeshRenderer

    our_renderer = OurPbrMeshRenderer(rendering_options={
        "resolution": args.resolution,
        "ssaa": 1,
        "near": 1.0,
        "far": 100.0,
    }, device=device)

    our_results = {}
    for vc in view_configs:
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [vc["yaw"]], [vc["pitch"]], r, fov
        )
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = our_renderer.render(mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
        torch.cuda.synchronize()
        dt = time.time() - t0
        our_results[vc["label"]] = result
        print(f"  [{vc['label']}] 完成, 耗时 {dt:.3f}s, "
              f"shaded={result['shaded'].shape}, alpha range=[{result['alpha'].min():.4f}, {result['alpha'].max():.4f}]")

    del our_renderer
    torch.cuda.empty_cache()

    # =================================================================
    # 7. 数值对比
    # =================================================================
    print("\n" + "=" * 60)
    print("7. 数值对比（原生 vs 改进后）")
    print("=" * 60)

    compare_keys = ["shaded", "normal", "base_color", "metallic", "roughness", "alpha", "clay"]

    for view_label in ref_results:
        print(f"\n--- 视角: {view_label} ---")
        ref_r = ref_results[view_label]
        our_r = our_results[view_label]
        for k in compare_keys:
            if k in ref_r and k in our_r:
                # PBR 渲染中存在 depth peeling 排序差异，容差适当放宽
                atol = 5e-2 if k in ("shaded", "clay") else 1e-2
                compare_tensors(f"{view_label}/{k}", ref_r[k], our_r[k], atol=atol)
            elif k in ref_r:
                print(f"  [{view_label}/{k}] ⚠ 改进后缺少此输出")
            elif k in our_r:
                print(f"  [{view_label}/{k}] ⚠ 原生缺少此输出")

    # =================================================================
    # 8. VRAM 峰值对比（简单测量）
    # =================================================================
    print("\n" + "=" * 60)
    print("8. VRAM 峰值测量")
    print("=" * 60)

    # 重新创建两个 renderer 分别测量 VRAM
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [180.0], [0.0], r, fov
    )

    # 原生
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ref_renderer_2 = RefPbrMeshRenderer(rendering_options={
        "resolution": args.resolution, "ssaa": 1, "near": 1.0, "far": 100.0,
    }, device=device)
    with torch.no_grad():
        _ = ref_renderer_2.render(mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
    ref_peak = torch.cuda.max_memory_allocated(device) / 1024**3
    del ref_renderer_2, _
    torch.cuda.empty_cache()
    print(f"  [原生]   VRAM peak: {ref_peak:.2f} GiB")

    # 改进后
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    our_renderer_2 = OurPbrMeshRenderer(rendering_options={
        "resolution": args.resolution, "ssaa": 1, "near": 1.0, "far": 100.0,
    }, device=device)
    with torch.no_grad():
        _ = our_renderer_2.render(mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
    our_peak = torch.cuda.max_memory_allocated(device) / 1024**3
    del our_renderer_2, _
    torch.cuda.empty_cache()
    print(f"  [改进后] VRAM peak: {our_peak:.2f} GiB")
    print(f"  差值: {ref_peak - our_peak:+.2f} GiB")

    # =================================================================
    # 9. 保存可视化
    # =================================================================
    print("\n" + "=" * 60)
    print("9. 保存可视化")
    print("=" * 60)

    def save_image(tensor, path):
        """保存 (C, H, W) 或 (H, W) tensor 为 PNG"""
        t = tensor.detach().cpu().float()
        if t.dim() == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        img = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    for view_label in ref_results:
        ref_r = ref_results[view_label]
        our_r = our_results[view_label]

        # 白色背景 shaded
        ref_shaded = ref_r['shaded'] + (1 - ref_r['alpha'].unsqueeze(0)) * 1.0  # (3, H, W)
        our_shaded = our_r['shaded'] + (1 - our_r['alpha'].unsqueeze(0)) * 1.0  # (3, H, W)

        save_image(ref_shaded, f"{args.save_dir}/{view_label}_ref_shaded.png")
        save_image(our_shaded, f"{args.save_dir}/{view_label}_our_shaded.png")

        # 差异热图（放大 10 倍方便观察）
        diff = (ref_shaded - our_shaded).abs().mean(dim=0)  # (H, W)
        diff_max = diff.max().item()
        if diff_max > 0:
            diff = diff / diff_max
        save_image(diff, f"{args.save_dir}/{view_label}_diff.png")

        # normal
        ref_n = ref_r['normal'] * 0.5 + 0.5  # (3, H, W) [-1,1] → [0,1]
        our_n = our_r['normal'] * 0.5 + 0.5  # (3, H, W)
        save_image(ref_n, f"{args.save_dir}/{view_label}_ref_normal.png")
        save_image(our_n, f"{args.save_dir}/{view_label}_our_normal.png")

        print(f"  [{view_label}] 已保存 ref_shaded / our_shaded / diff / ref_normal / our_normal")

    # 保存输入图片
    image = Image.open(args.image)
    image.save(f"{args.save_dir}/input.png")
    print(f"  已保存 input.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("完成！")


if __name__ == "__main__":
    main()
