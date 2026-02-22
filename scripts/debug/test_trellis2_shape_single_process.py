"""
TRELLIS.2 Shape Rollout 单进程对比测试

确保两个实现使用完全相同的初始噪声来对比采样过程。
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
    args = parser.parse_args()
    
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
    # 对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    print("\n[中间结果对比]")
    for i, (ref_int, our_int) in enumerate(zip(ref_intermediates, our_intermediates)):
        compare_tensors(f"intermediate_{i}", ref_int, our_int, atol=1e-4)
    
    print("\n[最终结果对比]")
    compare_tensors("shape_slat_feats", ref_shape_slat.feats, our_shape_slat.feats, atol=1e-3)


if __name__ == "__main__":
    main()

