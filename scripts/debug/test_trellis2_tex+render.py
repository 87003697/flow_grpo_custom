"""
TRELLIS.2 Tex Rollout + PBR Render 单进程对比测试

确保两个实现使用完全相同的 shape_slat 和初始噪声来对比 Tex 采样过程，
并对比 decode 和 PBR 渲染结果。

对比链路：
  共享 shape_slat (denormalized)
  → Tex 初始化（shape_cond 归一化 + 噪声生成）
  → Tex Rollout（逐步对比 velocity / CFG / Euler step）
  → Tex 反归一化
  → decode_tex_slat（对比 tex_voxels）
  → MeshWithVoxel 构建
  → PBR 渲染对比
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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
    parser.add_argument("--save_dir", type=str, default="./outputs/comparison_tex_output")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # =================================================================
    # 1. 加载参考实现
    # =================================================================
    print("\n" + "="*60)
    print("加载参考实现")
    print("="*60)
    
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "./pretrained_weights/TRELLIS.2-4B",
        dino_local_path="./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    pipe.low_vram = False
    pipe.to(device)

    # =================================================================
    # 2. 加载我们的实现
    # =================================================================
    print("\n" + "="*60)
    print("加载我们的实现")
    print("="*60)
    
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference, FlowEulerScheduler
    from edit4shape.generators.trellis2.rollout.base import trellis2_cfg_sparse
    import ml_collections
    
    cfg = ml_collections.ConfigDict()
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS.2-4B"
    cfg.pretrained.dino_local_path = "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
    cfg.pipeline_type = "1024"
    cfg.verbose = False
    
    class MockAccelerator:
        pass
    accelerator = MockAccelerator()
    accelerator.device = torch.device(device)
    
    our_pipeline = build_pipeline_from_reference(cfg, accelerator)

    # =================================================================
    # 3. 共享 Shape 采样（使用参考实现生成 shape_slat）
    # =================================================================
    print("\n" + "="*60)
    print("共享 Shape 采样")
    print("="*60)
    
    image = Image.open(args.image)
    image_proc = pipe.preprocess_image(image)
    
    torch.manual_seed(args.seed)
    
    cond_1024 = pipe.get_cond([image_proc], resolution=1024)
    print(f"cond_1024: {cond_1024['cond'].shape}")
    
    coords = pipe.sample_sparse_structure(cond_1024, 64, num_samples=1)
    print(f"coords: {coords.shape}")
    
    # 使用参考实现完整采样 shape_slat（denormalized）
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    shape_slat = pipe.sample_shape_slat(
        cond_1024, shape_flow_model, coords
    )
    print(f"shape_slat (denormalized): feats={shape_slat.feats.shape}, "
          f"mean={shape_slat.feats.mean():.4f}")

    # =================================================================
    # 4. ★ 对比 shape_cond 归一化
    # =================================================================
    print("\n" + "="*60)
    print("对比 shape_cond 归一化")
    print("="*60)
    
    # 参考实现：手动归一化 shape_slat（复制参考 sample_tex_slat 的逻辑）
    shape_norm = pipe.shape_slat_normalization
    std_shape = torch.tensor(shape_norm['std'])[None].to(device)  # (1, C)
    mean_shape = torch.tensor(shape_norm['mean'])[None].to(device)  # (1, C)
    ref_shape_cond = (shape_slat - mean_shape) / std_shape  # SparseTensor
    print(f"[Ref] shape_cond (normalized): feats[:3,:3]=\n{ref_shape_cond.feats[:3,:3]}")
    
    # 我们的实现：模拟 rollout_shape 的输出
    # shape_slat_norm 应 = normalize(shape_slat, "shape")
    our_shape_cond = our_pipeline.normalize(shape_slat, "shape")
    print(f"[Our] shape_cond (normalized): feats[:3,:3]=\n{our_shape_cond.feats[:3,:3]}")
    
    compare_tensors("shape_cond_normalized", ref_shape_cond.feats, our_shape_cond.feats, atol=1e-6)

    # =================================================================
    # 5. 生成共享初始噪声（Tex）
    # =================================================================
    print("\n" + "="*60)
    print("生成共享初始噪声（Tex）")
    print("="*60)
    
    torch.manual_seed(args.seed + 2000)
    
    tex_flow_model = pipe.models['tex_slat_flow_model_1024']
    tex_in_channels = tex_flow_model.in_channels
    shape_channels = shape_slat.feats.shape[1]
    noise_channels = tex_in_channels - shape_channels
    
    our_noise_channels = our_pipeline.get_in_channels("tex", 1024)
    
    print(f"tex_in_channels: {tex_in_channels}")
    print(f"shape_channels: {shape_channels}")
    print(f"[Ref] noise_channels: {noise_channels}")
    print(f"[Our] noise_channels: {our_noise_channels}")
    assert noise_channels == our_noise_channels, \
        f"噪声通道数不匹配: ref={noise_channels}, our={our_noise_channels}"
    
    # 生成初始噪声（ref: 使用 shape_slat.replace）
    initial_noise = torch.randn(coords.shape[0], noise_channels, device=device)  # (N, C_noise)
    ref_initial = ref_shape_cond.replace(feats=initial_noise.clone())  # SparseTensor, same coords
    our_initial = SparseTensor(coords=coords, feats=initial_noise.clone())  # SparseTensor
    
    print(f"initial_noise: {initial_noise.shape}")
    compare_tensors("initial_latent_feats", ref_initial.feats, our_initial.feats, atol=1e-7)
    compare_tensors("initial_latent_coords", ref_initial.coords, our_initial.coords, atol=0)

    # =================================================================
    # 6. 手动展开参考实现的 Tex 采样
    # =================================================================
    print("\n" + "="*60)
    print("参考实现：手动展开 Tex 采样")
    print("="*60)
    
    tex_sampler = pipe.tex_slat_sampler
    tex_params = pipe.tex_slat_sampler_params
    print(f"tex_sampler_params: {tex_params}")
    
    steps = tex_params['steps']
    rescale_t = tex_params['rescale_t']
    guidance_strength = tex_params['guidance_strength']
    guidance_rescale = tex_params['guidance_rescale']
    guidance_interval = tex_params['guidance_interval']
    
    t_seq = np.linspace(1, 0, steps + 1)  # float64
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
    print(f"t_seq: {t_seq[:4]}...{t_seq[-2:]}")
    
    ref_sample = ref_initial
    ref_intermediates = []
    
    with torch.no_grad():
        for i, (t, t_prev) in enumerate(t_pairs):
            pred_x_0, pred_eps, pred_v = tex_sampler._get_model_prediction(
                tex_flow_model, ref_sample, float(t),
                cond_1024['cond'],
                neg_cond=cond_1024['neg_cond'],
                guidance_strength=guidance_strength,
                guidance_rescale=guidance_rescale,
                guidance_interval=tuple(guidance_interval),
                concat_cond=ref_shape_cond,
            )
            
            delta_t = float(t - t_prev)
            ref_sample = ref_sample - delta_t * pred_v
            
            if i == 0 or i == steps - 1:
                print(f"[Ref] step {i}: t={t:.6f}, sample.feats[:3,:3]=\n{ref_sample.feats[:3,:3]}")
                ref_intermediates.append(ref_sample.feats.clone())
    
    # Denormalize
    tex_norm = pipe.tex_slat_normalization
    std_tex = torch.tensor(tex_norm['std'])[None].to(device)  # (1, C)
    mean_tex = torch.tensor(tex_norm['mean'])[None].to(device)  # (1, C)
    ref_tex_slat = ref_sample * std_tex + mean_tex
    print(f"[Ref] final tex_slat: feats[:3,:3]=\n{ref_tex_slat.feats[:3,:3]}")

    # =================================================================
    # 7. 手动展开我们的 Tex 采样
    # =================================================================
    print("\n" + "="*60)
    print("我们的实现：手动展开 Tex 采样")
    print("="*60)
    
    our_sampler_params = our_pipeline.get_sampler_params("tex")
    print(f"our tex_sampler_params: {our_sampler_params}")
    
    our_steps = int(our_sampler_params["steps"])
    our_cfg_strength = float(our_sampler_params["guidance_strength"])
    our_cfg_rescale = float(our_sampler_params["guidance_rescale"])
    our_cfg_min, our_cfg_max = our_pipeline.get_cfg_interval("tex")
    our_sigma_min = pipe.tex_slat_sampler.sigma_min
    
    scheduler = our_pipeline.scheduler("tex")
    scheduler.set_timesteps(our_steps, device=device)
    
    cond_emb = cond_1024["cond"].to(device)  # (B, S, C)
    uncond_emb = cond_1024["neg_cond"].to(device)  # (B, S, C)
    
    our_sample = SparseTensor(coords=coords, feats=initial_noise.clone())
    our_shape_cond = our_shape_cond  # 已在前面准备好
    our_intermediates = []
    
    with torch.no_grad():
        for idx in range(len(scheduler._timesteps_np) - 1):
            t_val = scheduler.get_precise_t(idx)
            t_norm = t_val
            use_cfg = our_cfg_min <= t_norm <= our_cfg_max
            
            cond_pred = our_pipeline.sampling_step(
                our_sample, t_val, cond_emb, "tex", 1024,
                shape_cond=our_shape_cond
            )
            
            if use_cfg:
                uncond_pred = our_pipeline.sampling_step(
                    our_sample, t_val, uncond_emb, "tex", 1024,
                    shape_cond=our_shape_cond
                )
                velocity = trellis2_cfg_sparse(
                    cond_pred, uncond_pred, our_cfg_strength,
                    guidance_rescale=our_cfg_rescale,
                    x_t=our_sample, t=t_val,
                    sigma_min=our_sigma_min
                )
            else:
                velocity = cond_pred
            
            our_sample = scheduler.step_by_index(velocity, idx, our_sample).prev_sample
            
            if idx == 0 or idx == our_steps - 1:
                print(f"[Our] step {idx}: t={t_val:.6f}, sample.feats[:3,:3]=\n{our_sample.feats[:3,:3]}")
                our_intermediates.append(our_sample.feats.clone())
    
    our_tex_slat = our_pipeline.denormalize(our_sample, "tex")
    print(f"[Our] final tex_slat: feats[:3,:3]=\n{our_tex_slat.feats[:3,:3]}")

    # =================================================================
    # 8. 对比 Tex Rollout 结果
    # =================================================================
    print("\n" + "="*60)
    print("对比 Tex Rollout 结果")
    print("="*60)
    
    print("\n[中间结果对比]")
    for i, (ref_int, our_int) in enumerate(zip(ref_intermediates, our_intermediates)):
        compare_tensors(f"tex_intermediate_{i}", ref_int, our_int, atol=1e-4)
    
    print("\n[最终 tex_slat 对比]")
    compare_tensors("tex_slat_feats", ref_tex_slat.feats, our_tex_slat.feats, atol=1e-3)

    # =================================================================
    # 9. 释放 our_pipeline 以回收显存（rollout 已验证一致）
    # =================================================================
    print("\n" + "="*60)
    print("释放 our_pipeline 以回收显存")
    print("="*60)
    
    # our_pipeline 加载了第二份完整模型，rollout 对比已完成，不再需要
    del our_pipeline, our_shape_cond
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    
    free_mem = torch.cuda.mem_get_info(device)[0] / 1024**3
    print(f"释放后 GPU 可用显存: {free_mem:.2f} GiB")

    # =================================================================
    # 10. 解码 Tex（需先 decode_shape 获取 meshes 和 subs）
    # =================================================================
    print("\n" + "="*60)
    print("解码 Tex")
    print("="*60)
    
    # 共享 shape 解码
    ref_meshes, ref_subs = pipe.decode_shape_slat(shape_slat, resolution=1024)
    ref_mesh = ref_meshes[0]
    print(f"shared mesh: vertices={ref_mesh.vertices.shape}, faces={ref_mesh.faces.shape}")
    
    # 参考实现 decode tex（标准 forward，无分块）
    ref_tex_voxels = pipe.decode_tex_slat(ref_tex_slat, ref_subs)
    print(f"[Ref] tex_voxels: feats={ref_tex_voxels[0].feats.shape}")
    
    # ★ 我们的实现 decode tex（使用 forward_chunked + ChunkedDecoderMixin，逐层自适应分块）
    from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
    tex_decoder = pipe.models['tex_slat_decoder']
    ChunkedDecoderMixin.inject_to(tex_decoder)
    print("[Our] 已注入 ChunkedDecoderMixin，使用 forward_chunked 解码")
    
    our_tex_voxels_raw = tex_decoder.forward_chunked(
        our_tex_slat, guide_subs=ref_subs, use_checkpoint=False  # 推理无需 checkpoint
    ) * 0.5 + 0.5  # SparseTensor feats: (N, C_out)，归一化到 [0, 1]
    print(f"[Our] tex_voxels (chunked): feats={our_tex_voxels_raw[0].feats.shape}")
    
    # ★ chunked forward 的 merge 操作会对坐标做规范排序，导致点顺序不同于标准 forward。
    # 必须按坐标对齐后再对比 feats，否则 index-wise 对比是无意义的。
    def sort_sparse_by_coords(st):
        """按坐标规范序排列 SparseTensor 的点"""
        c = st.coords  # (N, 4)
        D = c[:, 1:].max().item() + 1
        key = c[:, 0] * (D**3) + c[:, 1] * (D**2) + c[:, 2] * D + c[:, 3]  # (N,)
        idx = key.argsort()  # (N,)
        return c[idx], st.feats[idx]
    
    ref_c, ref_f = sort_sparse_by_coords(ref_tex_voxels[0])
    our_c, our_f = sort_sparse_by_coords(our_tex_voxels_raw[0])
    
    print("\n[Tex Voxels 对比（Ref=标准forward vs Our=forward_chunked，坐标对齐后）]")
    coords_match = torch.equal(ref_c, our_c)
    print(f"[coords_match] {'✓' if coords_match else '❌'} 坐标集合一致={coords_match}")
    compare_tensors("tex_voxels_feats (aligned)", ref_f, our_f, atol=5e-3)

    # =================================================================
    # 11. 构建 MeshWithVoxel 并对比（模拟 our decode_tex 的 safe_clamp）
    # =================================================================
    print("\n" + "="*60)
    print("构建 MeshWithVoxel 对比")
    print("="*60)
    
    from trellis2.representations import MeshWithVoxel
    
    # 参考实现的构建方式（与 decode_latent 中一致，attrs 直接用 feats）
    ref_mesh_voxel = MeshWithVoxel(
        ref_mesh.vertices, ref_mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 1024,
        coords=ref_tex_voxels[0].coords[:, 1:],
        attrs=ref_tex_voxels[0].feats,
        voxel_shape=torch.Size([*ref_tex_voxels[0].shape, *ref_tex_voxels[0].spatial_shape]),
        layout=pipe.pbr_attr_layout
    )
    
    # 模拟我们的实现 decode_tex 中的 safe_clamp 逻辑
    EPS = 1e-4
    our_v = our_tex_voxels_raw[0]
    clamped_rgb = torch.clamp(our_v.feats[:, :3], EPS, 1.0 - EPS)  # (N, 3) base_color
    our_attrs = torch.cat([clamped_rgb, our_v.feats[:, 3:]], dim=1)  # (N, 6)
    our_mesh_voxel = MeshWithVoxel(
        ref_mesh.vertices, ref_mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 1024,
        coords=our_v.coords[:, 1:],
        attrs=our_attrs,
        voxel_shape=torch.Size([*our_v.shape, *our_v.spatial_shape]),
        layout=pipe.pbr_attr_layout
    )
    
    print(f"[Ref] attrs range: [{ref_mesh_voxel.attrs.min():.4f}, {ref_mesh_voxel.attrs.max():.4f}]")
    print(f"[Our] attrs range: [{our_mesh_voxel.attrs.min():.4f}, {our_mesh_voxel.attrs.max():.4f}]")
    
    # ★ MeshWithVoxel.attrs 保持 voxel 的原始顺序（ref=标准forward序，our=chunked序），
    #   不能直接 index-wise 对比。使用已排序的 tex_voxels feats 来对比。
    # 先对比 raw feats（无 safe_clamp）以隔离 chunked forward 的影响
    print("\n[坐标对齐后的 attrs 对比]")
    compare_tensors("attrs_raw (aligned, 无safe_clamp)", ref_f, our_f, atol=5e-3)
    
    # 再模拟 safe_clamp 后对比
    ref_clamped = torch.cat([torch.clamp(ref_f[:, :3], EPS, 1.0 - EPS), ref_f[:, 3:]], dim=1)  # (N, 6)
    our_clamped = torch.cat([torch.clamp(our_f[:, :3], EPS, 1.0 - EPS), our_f[:, 3:]], dim=1)  # (N, 6)
    compare_tensors("attrs_with_safe_clamp (aligned)", ref_clamped, our_clamped, atol=5e-3)
    
    # safe_clamp vs 无 safe_clamp（衡量 safe_clamp 本身的影响范围）
    diff_clamp = (ref_f - ref_clamped).abs()
    print(f"[safe_clamp 影响] max={diff_clamp.max().item():.4f}, "
          f"affected_points={((diff_clamp > 1e-6).any(dim=1)).sum().item()}/{ref_f.shape[0]}")

    # =================================================================
    # 12. PBR 渲染对比
    # =================================================================
    print("\n" + "="*60)
    print("PBR 渲染对比")
    print("="*60)
    
    # 释放不再需要的大张量以腾出显存
    del shape_slat, ref_tex_slat, our_tex_slat, ref_tex_voxels, our_tex_voxels_raw
    del ref_subs, ref_meshes, coords, initial_noise
    del cond_emb, uncond_emb, cond_1024, ref_shape_cond
    # 释放 pipe 中不再需要的模型
    for k in list(pipe.models.keys()):
        if k != 'tex_slat_decoder':  # 保留 tex decoder 以防后续需要
            pipe.models[k].cpu()
    pipe.image_cond_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    free_mem = torch.cuda.mem_get_info(device)[0] / 1024**3
    print(f"PBR 渲染前 GPU 可用显存: {free_mem:.2f} GiB")
    
    # ★ 参考实现：使用官方 trellis2 PBR 渲染器（ACES tonemapping，无 gamma/sRGB）
    from trellis2.renderers import PbrMeshRenderer as RefPbrMeshRenderer, EnvMap
    # ★ 本地实现：使用带 sRGB transfer 的本地 PBR 渲染器
    from edit4shape.renderers.pbr_peeled_trellis2 import PbrMeshRenderer as OurPbrMeshRenderer
    from trellis2.utils import render_utils
    import cv2
    
    render_opts = {"resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0, "peel_layers": 8}
    
    # 创建两个 PBR 渲染器
    ref_pbr_renderer = RefPbrMeshRenderer(rendering_options=dict(render_opts), device=device)
    our_pbr_renderer = OurPbrMeshRenderer(rendering_options=dict(render_opts), device=device)
    print(f"[Ref] 渲染器: trellis2.renderers.PbrMeshRenderer (ACES tonemapping)")
    print(f"[Our] 渲染器: edit4shape.renderers.pbr_peeled_trellis2.PbrMeshRenderer (ACES + sRGB transfer)")
    
    # 加载环境贴图
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(os.path.join(repo_root, '_reference_codes/TRELLIS.2/assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device=device
    ))
    
    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    
    # ★ 渲染参考实现（官方渲染器）
    ref_mesh_voxel_dev = ref_mesh_voxel.to(device)
    ref_pbr = ref_pbr_renderer.render(ref_mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
    ref_alpha = ref_pbr['alpha']  # (H, W)
    ref_shaded = ref_pbr['shaded'] + (1 - ref_alpha.unsqueeze(0)) * 1.0  # (3, H, W), 白色背景
    print(f"[Ref] shaded: {ref_shaded.shape}, range=[{ref_shaded.min():.4f}, {ref_shaded.max():.4f}]")
    
    # ★ 渲染本地实现（带 sRGB transfer 的渲染器）
    our_mesh_voxel_dev = our_mesh_voxel.to(device)
    our_pbr = our_pbr_renderer.render(our_mesh_voxel_dev, extr[0], intr[0], envmap=envmap)
    our_alpha = our_pbr['alpha']  # (H, W)
    our_shaded = our_pbr['shaded'] + (1 - our_alpha.unsqueeze(0)) * 1.0  # (3, H, W), 白色背景
    print(f"[Our] shaded: {our_shaded.shape}, range=[{our_shaded.min():.4f}, {our_shaded.max():.4f}]")
    
    # sRGB transfer 会让整体更亮，容差设置较宽
    compare_tensors("pbr_shaded", ref_shaded, our_shaded, atol=1e-1)

    # =================================================================
    # 13. 保存可视化
    # =================================================================
    print("\n" + "="*60)
    print("保存可视化")
    print("="*60)
    
    def save_shaded_image(tensor, path):
        img = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"  保存: {path}")
    
    save_shaded_image(ref_shaded, f"{args.save_dir}/pbr_ref.png")
    save_shaded_image(our_shaded, f"{args.save_dir}/pbr_our.png")
    
    diff = (ref_shaded - our_shaded).abs()
    diff_gray = diff.mean(dim=0)
    diff_max = diff_gray.max().item()
    if diff_max > 0:
        diff_gray = diff_gray / diff_max
    diff_img = (diff_gray.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(diff_img).save(f"{args.save_dir}/pbr_diff.png")
    print(f"  保存: {args.save_dir}/pbr_diff.png")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")


if __name__ == "__main__":
    main()