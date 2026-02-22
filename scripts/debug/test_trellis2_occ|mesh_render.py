"""
TRELLIS.2 多分辨率 Sub 模式 vs Mesh 渲染对比测试

测试目标：
- 验证多层 Sub 渲染的 normal 与 Mesh 渲染的 normal 对比
- Sub 模式是粗糙近似，主要验证形状轮廓是否一致
- 计算 mask IoU 评估形状覆盖率
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


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """计算两个 mask 的 IoU"""
    m1 = mask1.bool()  # (H, W)
    m2 = mask2.bool()  # (H, W)
    intersection = (m1 & m2).sum().float()  # scalar
    union = (m1 | m2).sum().float()  # scalar
    iou = (intersection / union.clamp(min=1)).item()  # scalar
    return iou


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./outputs/comparison_output_sub")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = args.device
    
    # =========================================================================
    # 加载参考实现
    # =========================================================================
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
    
    # =========================================================================
    # 加载我们的实现
    # =========================================================================
    print("\n" + "="*60)
    print("加载我们的实现")
    print("="*60)
    
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference, FlowEulerScheduler
    from edit4shape.systems.trellis2.shape import trellis2_cfg_sparse
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
    
    # =========================================================================
    # 准备数据（共享随机状态）
    # =========================================================================
    print("\n" + "="*60)
    print("准备共享数据")
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
    # 生成共享的初始噪声
    # =========================================================================
    print("\n" + "="*60)
    print("生成共享初始噪声")
    print("="*60)
    
    # 重置种子确保一致
    torch.manual_seed(args.seed + 1000)  # 用不同的种子来生成初始噪声
    
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    in_channels = shape_flow_model.in_channels
    print(f"in_channels: {in_channels}")
    
    # 生成初始噪声
    initial_noise = torch.randn(coords.shape[0], in_channels, device=device)  # (N, C)
    initial_slat = SparseTensor(coords=coords, feats=initial_noise)
    print(f"initial_noise: {initial_noise.shape}, mean={initial_noise.mean():.4f}, std={initial_noise.std():.4f}")
    
    # =========================================================================
    # 手动展开参考实现的采样过程
    # =========================================================================
    print("\n" + "="*60)
    print("参考实现：手动展开采样")
    print("="*60)
    
    sampler = pipe.shape_slat_sampler
    params = pipe.shape_slat_sampler_params
    print(f"sampler_params: {params}")
    
    steps = params['steps']
    rescale_t = params['rescale_t']
    guidance_strength = params['guidance_strength']
    guidance_rescale = params['guidance_rescale']
    guidance_interval = params['guidance_interval']
    
    # 计算 timesteps（与参考实现完全一致）
    t_seq = np.linspace(1, 0, steps + 1)  # float64
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
    print(f"t_seq: {t_seq[:4]}...{t_seq[-2:]}")
    
    # 采样
    ref_sample = initial_slat
    ref_intermediates = []
    
    with torch.no_grad():
        for i, (t, t_prev) in enumerate(t_pairs):
            # 获取模型预测（参数需要展开，不能用 dict）
            pred_x_0, pred_eps, pred_v = sampler._get_model_prediction(
                shape_flow_model, ref_sample, float(t),
                cond_1024['cond'],  # cond 作为位置参数
                neg_cond=cond_1024['neg_cond'],
                guidance_strength=guidance_strength,
                guidance_rescale=guidance_rescale,
                guidance_interval=tuple(guidance_interval),
            )
            
            # Euler step
            delta_t = float(t - t_prev)
            ref_sample = ref_sample - delta_t * pred_v
            
            if i == 0 or i == steps - 1:
                print(f"[Ref] step {i}: t={t:.6f}, sample.feats[:3,:3]=\n{ref_sample.feats[:3,:3]}")
                ref_intermediates.append(ref_sample.feats.clone())
    
    # Denormalize
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(device)
    ref_shape_slat = ref_sample * std + mean
    print(f"[Ref] final shape_slat: feats[:3,:3]=\n{ref_shape_slat.feats[:3,:3]}")
    
    # =========================================================================
    # 手动展开我们的采样过程
    # =========================================================================
    print("\n" + "="*60)
    print("我们的实现：手动展开采样")
    print("="*60)
    
    stage_config = our_pipeline.get_stage_config("shape")
    resolution = stage_config["flow_resolution"]
    
    sampler_params = our_pipeline.get_sampler_params("shape")
    print(f"our sampler_params: {sampler_params}")
    
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["guidance_strength"])
    cfg_rescale_val = float(sampler_params["guidance_rescale"])
    cfg_min, cfg_max = our_pipeline.get_cfg_interval("shape")
    sigma_min = pipe.shape_slat_sampler.sigma_min
    
    scheduler = our_pipeline.scheduler("shape")
    scheduler.set_timesteps(steps, device=device)
    
    cond_emb = cond_1024["cond"].to(device)
    uncond_emb = cond_1024["neg_cond"].to(device)
    
    # 使用相同的初始噪声
    our_sample = SparseTensor(coords=coords, feats=initial_noise.clone())
    our_intermediates = []
    
    with torch.no_grad():
        for idx in range(len(scheduler._timesteps_np) - 1):
            t_val = scheduler.get_precise_t(idx)
            t_norm = t_val
            use_cfg = cfg_min <= t_norm <= cfg_max
            
            cond_pred = our_pipeline.sampling_step(our_sample, t_val, cond_emb, "shape", resolution)
            
            if use_cfg:
                uncond_pred = our_pipeline.sampling_step(our_sample, t_val, uncond_emb, "shape", resolution)
                velocity = trellis2_cfg_sparse(
                    cond_pred, uncond_pred, cfg_strength,
                    guidance_rescale=cfg_rescale_val, x_t=our_sample, t=t_val, sigma_min=sigma_min
                )
            else:
                velocity = cond_pred
            
            our_sample = scheduler.step_by_index(velocity, idx, our_sample).prev_sample
            
            if idx == 0 or idx == steps - 1:
                print(f"[Our] step {idx}: t={t_val:.6f}, sample.feats[:3,:3]=\n{our_sample.feats[:3,:3]}")
                our_intermediates.append(our_sample.feats.clone())
    
    # Denormalize
    our_shape_slat = our_pipeline.denormalize(our_sample, "shape")
    print(f"[Our] final shape_slat: feats[:3,:3]=\n{our_shape_slat.feats[:3,:3]}")
    
    # =========================================================================
    # 对比 Shape Rollout 结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比 Shape Rollout 结果")
    print("="*60)
    
    print("\n[中间结果对比]")
    for i, (ref_int, our_int) in enumerate(zip(ref_intermediates, our_intermediates)):
        compare_tensors(f"intermediate_{i}", ref_int, our_int, atol=1e-4)
    
    print("\n[最终 shape_slat 对比]")
    compare_tensors("shape_slat_feats", ref_shape_slat.feats, our_shape_slat.feats, atol=1e-3)
    
    # =========================================================================
    # 解码 Mesh
    # =========================================================================
    print("\n" + "="*60)
    print("解码 Mesh")
    print("="*60)
    
    # 参考实现解码
    ref_meshes, ref_subs = pipe.decode_shape_slat(ref_shape_slat, resolution=1024)
    ref_mesh = ref_meshes[0]
    print(f"[Ref] mesh: vertices={ref_mesh.vertices.shape}, faces={ref_mesh.faces.shape}")
    
    # 我们的实现解码
    our_decode_result = our_pipeline.decode_shape(our_shape_slat, resolution=1024)
    our_mesh = our_decode_result["meshes"][0]
    print(f"[Our] mesh: vertices={our_mesh.vertices.shape}, faces={our_mesh.faces.shape}")
    
    # 对比 mesh
    print("\n[Mesh 对比]")
    compare_tensors("mesh_vertices", ref_mesh.vertices, our_mesh.vertices, atol=1e-4)
    compare_tensors("mesh_faces", ref_mesh.faces, our_mesh.faces)
    
    # =========================================================================
    # 渲染 Normal
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Normal")
    print("="*60)
    
    from trellis2.renderers import MeshRenderer
    from trellis2.utils import render_utils
    from edit4shape.renderers.diff_voxel_normal import RenderConfig, render_normal_sub, render_normal_sub_pyramid
    
    # 创建渲染器（与参考实现 render_utils.py 的默认值一致：near=1, far=100）
    renderer = MeshRenderer(rendering_options={
        "resolution": 1024, "ssaa": 1, "near": 1.0, "far": 100.0
    }, device=device)
    
    # 相机参数（与 config/trellis2_shape_distillation.py eval 配置一致）
    # yaw=180°, pitch=0°, r=2.0, fov=40°
    yaw, pitch, r, fov = 180.0, 0.0, 2.0, 40.0
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], r, fov
    )
    if isinstance(extr, list):
        extr = torch.stack(extr, dim=0)  # (1, 4, 4)
    if isinstance(intr, list):
        intr = torch.stack(intr, dim=0)  # (1, 3, 3)
    extr = extr.to(device)  # (1, 4, 4)
    intr = intr.to(device)  # (1, 3, 3)
    print(f"Camera: yaw={yaw}, pitch={pitch}, r={r}, fov={fov}")
    
    # 渲染参考实现的 mesh
    ref_render = renderer.render(ref_mesh, extr[0], intr[0], return_types=["normal", "mask"])
    ref_normal = ref_render["normal"]  # (3, H, W)
    ref_mask = ref_render["mask"]  # (1, H, W)
    print(f"[Ref] normal: {ref_normal.shape}, mask sum: {ref_mask.sum().item():.0f}")
    
    # 渲染我们实现的 mesh
    our_render = renderer.render(our_mesh, extr[0], intr[0], return_types=["normal", "mask"])
    our_normal = our_render["normal"]  # (3, H, W)
    our_mask = our_render["mask"]  # (1, H, W)
    print(f"[Our] normal: {our_normal.shape}, mask sum: {our_mask.sum().item():.0f}")

    # =========================================================================
    # Diff Voxel Normal 渲染（多分辨率 Sub 模式）
    # =========================================================================
    print("\n" + "="*60)
    print("渲染 Diff Voxel Normal（多分辨率 Sub）")
    print("="*60)

    decoder = our_pipeline.pipe.models["shape_slat_decoder"]
    decoder.set_resolution(1024)
    parent_class = decoder.__class__.__bases__[0]
    h, subs = parent_class.forward(decoder, our_shape_slat, return_subs=True)  # h.feats: (N, 7), subs: List[SparseTensor]
    
    print(f"获取到 {len(subs)} 层 sub_logits:")
    for i, sub in enumerate(subs):
        print(f"  layer {i}: coords={sub.coords.shape}, feats={sub.feats.shape}")

    extr_0 = extr[0]  # (4, 4)
    intr_0 = intr[0]  # (3, 3)
    base_resolution = 1024
    num_layers = len(subs)
    target_size = (1024, 1024)  # 统一 resize 到最终分辨率
    
    # 参考 mesh 的 mask（用于计算 IoU）
    mesh_mask = our_mask.squeeze(0).bool()  # (H, W)
    
    # 存储每层的渲染结果
    sub_results = []  # List[(normal, mask, layer_resolution)]
    iou_results = []  # List[(layer_idx, iou)]
    
    print("\n[多层渲染]")
    for i, sub in enumerate(subs):
        # 计算该层的分辨率
        # subs[0] 是最低分辨率，subs[-1] 是最高分辨率
        # 每层 2x 上采样
        layer_resolution = base_resolution // (2 ** (num_layers - i))
        
        config_i = RenderConfig(
            extrinsic=extr_0,
            intrinsic=intr_0,
            resolution=layer_resolution,
        )
        
        # 渲染并 resize 到目标分辨率
        sub_normal, sub_mask = render_normal_sub(sub, config_i, target_size)  # (H, W, 3), (H, W)
        sub_normal_chw = sub_normal.permute(2, 0, 1)  # (3, H, W)
        
        # 计算与 mesh 的 mask IoU
        iou = compute_mask_iou(sub_mask, mesh_mask)
        iou_results.append((i, layer_resolution, iou))
        
        sub_results.append((sub_normal_chw, sub_mask, layer_resolution))
        print(f"  layer {i} (res={layer_resolution}): mask_sum={sub_mask.sum().item():.0f}, IoU={iou:.4f}")
    
    # 对比渲染结果
    print("\n[渲染结果对比]")
    compare_tensors("normal_ref_vs_our", ref_normal, our_normal, atol=1e-3)
    compare_tensors("mask_ref_vs_our", ref_mask, our_mask, atol=1e-5)
    
    print("\n[各层与 Mesh 对比]")
    for i, (sub_normal_chw, sub_mask, layer_res) in enumerate(sub_results):
        sub_mask_2d = sub_mask.float()  # (H, W)
        our_mask_2d = our_mask.squeeze(0).float()  # (H, W)
        # Sub 模式是粗糙近似，使用宽松阈值
        compare_tensors(f"mask_layer{i}(res={layer_res})", our_mask_2d, sub_mask_2d, atol=0.1)
    
    print("\n[IoU 统计]")
    for layer_idx, layer_res, iou in iou_results:
        status = "✓" if iou > 0.8 else ("⚠" if iou > 0.5 else "❌")
        print(f"  layer {layer_idx} (res={layer_res}): IoU = {iou:.4f} {status}")
    
    # =========================================================================
    # 金字塔融合渲染
    # =========================================================================
    print("\n" + "="*60)
    print("金字塔融合渲染")
    print("="*60)
    
    # 构建每层的 config
    pyramid_configs = []
    for i in range(num_layers):
        layer_resolution = base_resolution // (2 ** (num_layers - i))
        config_i = RenderConfig(
            extrinsic=extr_0,
            intrinsic=intr_0,
            resolution=layer_resolution,
        )
        pyramid_configs.append(config_i)
    
    # 金字塔融合（默认权重）
    pyramid_normal, pyramid_mask = render_normal_sub_pyramid(
        subs, pyramid_configs, target_size
    )  # (H, W, 3), (H, W)
    pyramid_normal_chw = pyramid_normal.permute(2, 0, 1)  # (3, H, W)
    
    # 计算金字塔融合与 Mesh 的 IoU
    pyramid_iou = compute_mask_iou(pyramid_mask, mesh_mask)
    print(f"金字塔融合 IoU: {pyramid_iou:.4f}")
    
    # 对比金字塔融合与最高分辨率层
    best_layer_iou = iou_results[-1][2]  # 最高分辨率层的 IoU
    improvement = pyramid_iou - best_layer_iou
    print(f"相比最高分辨率层 (IoU={best_layer_iou:.4f})，提升: {improvement:+.4f}")
    
    # =========================================================================
    # 保存可视化结果
    # =========================================================================
    print("\n" + "="*60)
    print("保存可视化结果")
    print("="*60)
    
    def save_normal_image(normal_tensor, mask_tensor, path, is_normalized=True):
        """保存 normal 图像，背景设为白色
        
        Args:
            normal_tensor: (3, H, W) or (H, W, 3)
            mask_tensor: (1, H, W) or (H, W)
            path: 保存路径
            is_normalized: 如果 True，输入范围是 [0, 1]（MeshRenderer 输出）
                          如果 False，输入范围是 [-1, 1]（原始法线）
        """
        normal = normal_tensor.detach().cpu()  # (3, H, W)
        mask = mask_tensor.detach().cpu()  # (1, H, W)
        
        # 确保 mask 是 (1, H, W) 或 (3, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1:
            mask = mask.expand_as(normal)
        
        if is_normalized:
            # MeshRenderer 输出已经是 [0, 1]，直接使用
            normal_vis = normal  # (3, H, W)
        else:
            # 原始法线范围是 [-1, 1]，转换到 [0, 1]
            normal_vis = (normal + 1) / 2  # (3, H, W)
        
        # 应用 mask，背景设为白色
        normal_vis = normal_vis * mask + (1 - mask)  # (3, H, W)
        
        # 转换为 uint8
        img = (normal_vis.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"  保存: {path}")
    
    def save_diff_image(tensor1, tensor2, path):
        """保存差异图"""
        diff = (tensor1.float() - tensor2.float()).abs()  # (3, H, W) or (H, W)
        if diff.dim() == 3:
            diff = diff.mean(dim=0)  # (H, W)
        diff_max = diff.max().item()
        if diff_max > 0:
            diff = diff / diff_max
        diff_img = (diff.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(diff_img).save(path)
        print(f"  保存: {path}")
    
    # MeshRenderer 输出已经是 [0, 1] 范围（is_normalized=True）
    save_normal_image(ref_normal, ref_mask, f"{args.save_dir}/normal_ref.png", is_normalized=True)
    save_normal_image(our_normal, our_mask, f"{args.save_dir}/normal_mesh.png", is_normalized=True)
    
    # 保存每层的 Sub 渲染结果（[-1, 1] 范围）
    for i, (sub_normal_chw, sub_mask, layer_res) in enumerate(sub_results):
        sub_mask_chw = sub_mask.unsqueeze(0).float()  # (1, H, W)
        save_normal_image(sub_normal_chw, sub_mask_chw, 
                         f"{args.save_dir}/normal_sub_layer{i}_res{layer_res}.png", is_normalized=False)
        # 保存与 mesh 的差异图
        save_diff_image(our_normal, sub_normal_chw, 
                       f"{args.save_dir}/diff_layer{i}_res{layer_res}.png")
    
    # 保存金字塔融合结果
    pyramid_mask_chw = pyramid_mask.unsqueeze(0).float()  # (1, H, W)
    save_normal_image(pyramid_normal_chw, pyramid_mask_chw,
                     f"{args.save_dir}/normal_pyramid.png", is_normalized=False)
    save_diff_image(our_normal, pyramid_normal_chw,
                   f"{args.save_dir}/diff_pyramid.png")
    
    # 保存 IoU 统计
    with open(f"{args.save_dir}/iou_summary.txt", "w") as f:
        f.write("Layer\tResolution\tIoU\n")
        for layer_idx, layer_res, iou in iou_results:
            f.write(f"{layer_idx}\t{layer_res}\t{iou:.4f}\n")
        f.write(f"pyramid\tfused\t{pyramid_iou:.4f}\n")
    print(f"  保存: {args.save_dir}/iou_summary.txt")
    
    # =========================================================================
    # 多视角渲染对比（使用最高分辨率的 Sub 层）
    # =========================================================================
    print("\n" + "="*60)
    print("多视角渲染对比（最高分辨率 Sub 层）")
    print("="*60)
    
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # 使用最高分辨率的 sub 层
    best_sub = subs[-1]
    best_layer_res = base_resolution // 2  # 最高层分辨率
    
    multi_view_ious = []
    
    for yaw_i in yaw_angles:
        extr_i, intr_i = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [float(yaw_i)], [20.0], 2.0, 40.0
        )
        extr_i = extr_i[0].to(device)  # (4, 4)
        intr_i = intr_i[0].to(device)  # (3, 3)
        
        # Mesh 渲染
        mesh_out = renderer.render(our_mesh, extr_i, intr_i, return_types=["normal", "mask"])
        mesh_mask_i = mesh_out["mask"].squeeze(0).bool()  # (H, W)
        
        # Sub 渲染
        config_i = RenderConfig(
            extrinsic=extr_i,
            intrinsic=intr_i,
            resolution=best_layer_res,
        )
        sub_normal_i, sub_mask_i = render_normal_sub(best_sub, config_i, target_size)  # (H, W, 3), (H, W)
        
        # 计算 IoU
        iou_i = compute_mask_iou(sub_mask_i, mesh_mask_i)
        multi_view_ious.append((yaw_i, iou_i))
        
        status = "✓" if iou_i > 0.8 else ("⚠" if iou_i > 0.5 else "❌")
        print(f"  yaw={yaw_i:3d}°: IoU = {iou_i:.4f} {status}")
    
    avg_iou = sum(iou for _, iou in multi_view_ious) / len(multi_view_ious)
    print(f"\n  平均 IoU: {avg_iou:.4f}")
    
    print(f"\n可视化结果已保存到: {args.save_dir}/")
    print("  - normal_ref.png: 参考实现渲染结果")
    print("  - normal_mesh.png: Mesh 渲染结果")
    print("  - normal_sub_layer*.png: 各层 Sub 渲染结果")
    print("  - normal_pyramid.png: 金字塔融合渲染结果")
    print("  - diff_layer*.png: 各层与 Mesh 差异图")
    print("  - diff_pyramid.png: 金字塔融合与 Mesh 差异图")
    print("  - iou_summary.txt: IoU 统计")


if __name__ == "__main__":
    main()

