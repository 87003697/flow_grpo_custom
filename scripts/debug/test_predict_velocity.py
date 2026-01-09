#!/usr/bin/env python3
"""
对比 _predict_velocity 和直接调用 flow_model 的差异
"""

import os
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
sys.path.insert(0, trellis2_ref_root)

import torch
import numpy as np
from PIL import Image


def main():
    device = torch.device("cuda")
    seed = 42
    
    print("=" * 70)
    print("  对比 _predict_velocity 和直接调用 flow_model")
    print("=" * 70)
    
    # 加载 pipeline
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    
    ref_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        dino_local_path=os.path.join(
            repo_root, 
            "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
    )
    ref_pipeline.low_vram = False
    ref_pipeline.cuda()
    ref_pipeline.image_cond_model.to(device)
    
    from edit4shape.generators.trellis2.pipeline_adapter import Trellis2RefAdapter
    your_pipeline = Trellis2RefAdapter(ref_pipeline, pipeline_type="1024")
    
    # 加载图像并生成条件
    test_image_path = os.path.join(trellis2_ref_root, "assets/example_image/image_01.png")
    image = Image.open(test_image_path)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ref_image_proc = ref_pipeline.preprocess_image(image)
    ref_cond_512 = ref_pipeline.get_cond([ref_image_proc], 512)
    ref_cond_1024 = ref_pipeline.get_cond([ref_image_proc], 1024)
    
    # 生成 coords
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    coords = ref_pipeline.sample_sparse_structure(
        ref_cond_512, 
        resolution=64,
        num_samples=1,
    )
    
    flow_model = ref_pipeline.models['shape_slat_flow_model_1024']
    
    # 生成初始噪声
    torch.manual_seed(seed)
    in_channels = flow_model.in_channels
    feats = torch.randn(coords.shape[0], in_channels, device=device)
    
    cond = ref_cond_1024["cond"].to(device)
    
    t = 1.0  # 第一个时间步
    
    print(f"\n测试参数:")
    print(f"  coords shape: {coords.shape}")
    print(f"  feats shape: {feats.shape}")
    print(f"  cond shape: {cond.shape}")
    print(f"  t = {t}")
    
    # ========== 直接调用 flow_model ==========
    print("\n" + "=" * 70)
    print("  方式 1: 直接调用 flow_model")
    print("=" * 70)
    
    x_t = SparseTensor(coords=coords.to(device), feats=feats)
    t_scaled = torch.tensor([1000 * t] * 1, device=device, dtype=torch.float32)
    
    print(f"  x_t.coords shape: {x_t.coords.shape}")
    print(f"  x_t.feats shape: {x_t.feats.shape}")
    print(f"  t_scaled: {t_scaled}")
    
    with torch.no_grad():
        direct_velocity = flow_model(x_t, t_scaled, cond)
    
    print(f"  direct_velocity.feats shape: {direct_velocity.feats.shape}")
    print(f"  direct_velocity.feats[:5, :5]:")
    print(direct_velocity.feats[:5, :5])
    
    # ========== 通过 _predict_velocity ==========
    print("\n" + "=" * 70)
    print("  方式 2: 通过 _predict_velocity")
    print("=" * 70)
    
    from edit4shape.systems.trellis2 import _predict_velocity
    
    with torch.no_grad():
        via_predict = _predict_velocity(
            your_pipeline, coords.to(device), feats,
            t, cond, "shape", 1024, None
        )
    
    print(f"  via_predict shape: {via_predict.shape}")
    print(f"  via_predict[:5, :5]:")
    print(via_predict[:5, :5])
    
    # ========== 对比 ==========
    print("\n" + "=" * 70)
    print("  对比结果")
    print("=" * 70)
    
    diff = (direct_velocity.feats - via_predict).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(direct_velocity.feats, via_predict, rtol=1e-4, atol=1e-6)
    
    status = "✅" if is_close else "❌"
    print(f"\n{status} velocity: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    
    if not is_close:
        # 详细分析差异
        print(f"\n差异分析:")
        print(f"  direct_velocity.feats.mean(): {direct_velocity.feats.mean().item():.6f}")
        print(f"  via_predict.mean(): {via_predict.mean().item():.6f}")
        print(f"  direct_velocity.feats.std(): {direct_velocity.feats.std().item():.6f}")
        print(f"  via_predict.std(): {via_predict.std().item():.6f}")


if __name__ == "__main__":
    main()





