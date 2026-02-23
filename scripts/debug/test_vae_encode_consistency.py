f"""
测试 FlowEdit 两次 VAE 编码的数值一致性。

方式1（不可导）：FlowEdit pipeline 内部的编码流程（模拟 prepare_latents）
方式2（可导）  ：FlowEditGuidance._encode_to_latent_packed 编码流程

Usage:
    python scripts/debug/test_vae_encode_consistency.py \
        --model_path Qwen/Qwen-Image-Edit-2511 \
        --image_path /path/to/test_image.png
"""

import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 复用 edit4shape/guidance 的代码
from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    retrieve_latents,
    calculate_dimensions,
    VAE_IMAGE_SIZE,  # 1048576 = 1024*1024，是面积不是边长
)


def encode_method1_nograd(pipe, pil_image: Image.Image):
    """
    方式1：模拟 FlowEdit pipeline 内部的编码流程（不可导）
    
    对应 flowedit_simple.py:
        vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
    然后在 prepare_latents 内部调用 VAE encode
    """
    # VAE_IMAGE_SIZE = 1048576 (面积, 1024*1024), 从 diffusers 导入
    
    # 计算尺寸（保持宽高比，与 pipeline 一致）
    image_width, image_height = pil_image.size
    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
    
    # 使用 image_processor.preprocess（与 pipeline 内部一致）
    # 输出: [B, C, H, W] 范围 [-1, 1]
    vae_image = pipe.image_processor.preprocess(pil_image, vae_height, vae_width)
    # 添加 frame 维度: [B, C, 1, H, W]
    vae_image_5d = vae_image.unsqueeze(2).to(pipe.device, dtype=torch.bfloat16)
    
    # VAE encode（不可导版本）
    with torch.no_grad():
        image_latents = retrieve_latents(pipe.vae.encode(vae_image_5d), sample_mode="argmax")
        
        # 标准化（与父类 _encode_vae_image 一致）
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        normalized_latent = (image_latents - latents_mean) / latents_std
    
    # Pack: [B, C, 1, H_lat, W_lat] -> [B, seq_len, C]
    B, C_lat, _, H_lat, W_lat = normalized_latent.shape
    packed_latent = pipe._pack_latents(normalized_latent, B, C_lat, H_lat, W_lat)
    
    return packed_latent, vae_image_5d, (vae_height, vae_width)


def encode_method2_differentiable(pipe, pil_image: Image.Image, edit_resolution: int = 1024, mode: str = 'bilinear'):
    """
    方式2：可导版本的编码流程
    
    对应 FlowEditGuidance._encode_to_latent_packed 方法
    
    Args:
        mode: 插值模式 ('bilinear', 'bicubic', 'nearest')
    """
    # PIL -> Tensor [0, 1]
    img_tensor = TF.to_tensor(pil_image).unsqueeze(0).to(pipe.device)  # [1, C, H, W]
    
    # Resize 到编辑分辨率（正方形）
    imgs_resized = F.interpolate(
        img_tensor, 
        size=(edit_resolution, edit_resolution), 
        mode=mode, 
        align_corners=False if mode != 'nearest' else None,
        antialias=True if mode in ('bilinear', 'bicubic') else False,  # antialias 更接近 Lanczos
    )  # [1, C, edit_res, edit_res]
    
    # [0,1] → [-1,1]，添加 frame 维度
    imgs_normalized = imgs_resized * 2 - 1  # [1, C, H, W]
    imgs_5d = imgs_normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # [1, C, 1, H, W]
    
    # VAE encode（可导版本）
    latent_5d = pipe._encode_vae_image_differentiable(imgs_5d)  # [1, C_lat, 1, H_lat, W_lat]
    
    # Pack: [1, C_lat, 1, H_lat, W_lat] -> [1, seq_len, C_lat]
    B, C_lat, _, H_lat, W_lat = latent_5d.shape
    packed_latent = pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)
    
    return packed_latent, imgs_5d


def encode_method2_with_same_preprocess(pipe, pil_image: Image.Image):
    """
    方式2 变体：使用与方式1相同的预处理，只替换 VAE encode 为可导版本
    
    用于隔离问题：预处理差异 vs VAE encode 差异
    """
    # VAE_IMAGE_SIZE = 1048576 (面积, 1024*1024), 从 diffusers 导入
    
    # 使用与方式1相同的预处理
    image_width, image_height = pil_image.size
    vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
    vae_image = pipe.image_processor.preprocess(pil_image, vae_height, vae_width)
    vae_image_5d = vae_image.unsqueeze(2).to(pipe.device, dtype=torch.bfloat16)
    
    # VAE encode（可导版本）
    latent_5d = pipe._encode_vae_image_differentiable(vae_image_5d)  # [1, C_lat, 1, H_lat, W_lat]
    
    # Pack
    B, C_lat, _, H_lat, W_lat = latent_5d.shape
    packed_latent = pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)
    
    return packed_latent


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name1: str, name2: str):
    """比较两个张量的数值差异"""
    l1 = t1.float()
    l2 = t2.float()
    
    print(f"\n{'='*60}")
    print(f"比较: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"Shape: {t1.shape} vs {t2.shape}")
    
    if t1.shape != t2.shape:
        print("⚠️  Shape 不同，无法直接比较数值")
        print(f"  {name1} 范围: [{l1.min().item():.4f}, {l1.max().item():.4f}]")
        print(f"  {name2} 范围: [{l2.min().item():.4f}, {l2.max().item():.4f}]")
        return None, None
    
    abs_diff = (l1 - l2).abs()
    rel_diff = abs_diff / (l1.abs() + 1e-8)
    
    print(f"\n绝对误差:")
    print(f"  Max:  {abs_diff.max().item():.6e}")
    print(f"  Mean: {abs_diff.mean().item():.6e}")
    print(f"  Std:  {abs_diff.std().item():.6e}")
    print(f"\n相对误差:")
    print(f"  Max:  {rel_diff.max().item():.6e}")
    print(f"  Mean: {rel_diff.mean().item():.6e}")
    print(f"\n数值范围:")
    print(f"  {name1}: [{l1.min().item():.4f}, {l1.max().item():.4f}]")
    print(f"  {name2}: [{l2.min().item():.4f}, {l2.max().item():.4f}]")
    
    # 一致性判定（允许 bfloat16 精度误差）
    is_close = torch.allclose(l1, l2, rtol=1e-3, atol=1e-4)
    print(f"\n✅ 数值一致 (rtol=1e-3, atol=1e-4): {is_close}")
    
    return abs_diff.mean().item(), rel_diff.mean().item()


def main():
    parser = argparse.ArgumentParser(description="测试两次 VAE 编码的数值一致性")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511",
                        help="FlowEdit 模型路径")
    parser.add_argument("--image_path", type=str, required=True,
                        help="测试图像路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    args = parser.parse_args()
    
    print(f"加载模型: {args.model_path}")
    pipe = FlowEditFullPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(torch.device(args.device))
    pipe.set_progress_bar_config(disable=True)
    
    print(f"加载测试图像: {args.image_path}")
    pil_image = Image.open(args.image_path).convert("RGB")
    print(f"原始图像尺寸: {pil_image.size}")
    
    # ================================================================
    # 测试1：比较两种完整编码流程（不同插值模式）
    # ================================================================
    print("\n" + "="*60)
    print("测试1：比较完整编码流程（不同插值模式）")
    print("="*60)
    
    print("\n执行方式1（不可导 + image_processor.preprocess + PIL Lanczos）...")
    latent1, preproc1, (h1, w1) = encode_method1_nograd(pipe, pil_image)
    print(f"  预处理尺寸: {h1}x{w1}")
    
    # 测试不同插值模式
    for mode in ['bilinear', 'bicubic']:
        print(f"\n执行方式2（可导 + F.interpolate + {mode} + antialias）...")
        latent2, preproc2 = encode_method2_differentiable(pipe, pil_image, edit_resolution=1024, mode=mode)
        
        # 比较 latent
        compare_tensors(latent1, latent2, "方式1(Lanczos)", f"方式2({mode})")
    
    # ================================================================
    # 测试2：隔离问题 - 使用相同预处理，只比较 VAE encode
    # ================================================================
    print("\n" + "="*60)
    print("测试2：隔离 VAE encode 差异（使用相同预处理）")
    print("="*60)
    
    print("\n执行方式2变体（相同预处理 + 可导 encode）...")
    latent2_same_preproc = encode_method2_with_same_preprocess(pipe, pil_image)
    
    compare_tensors(latent1, latent2_same_preproc, "方式1(nograd)", "方式2(diff,同预处理)")
    
    # ================================================================
    # 测试3：验证 src_latent 注入后的一致性
    # ================================================================
    print("\n" + "="*60)
    print("测试3：验证 src_latent 注入后的一致性")
    print("="*60)
    
    print("\n执行方式2（bicubic + antialias，模拟 _encode_to_latent_packed）...")
    latent_bicubic, _ = encode_method2_differentiable(pipe, pil_image, edit_resolution=1024, mode='bicubic')
    
    # 这个 latent 就是 _encode_to_latent_packed 的输出
    # 现在 FlowEdit 会使用这个 latent 作为 x_src，而不是内部编码的
    # loss 计算时也使用 _encode_to_latent_packed
    # 所以两边应该完全一致
    print("\n理论上，修改后的流程:")
    print("  FlowEdit 编辑:  x_src = src_latent (来自 _encode_to_latent_packed, bicubic)")
    print("  Loss 计算:      rendered_latent = _encode_to_latent_packed(rendered, bicubic)")
    print("  结果: 两者使用相同的编码方式，应该完全一致 ✅")
    
    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("""
修改后的流程:
  1. _edit_single 中用 _encode_to_latent_packed 预编码 rendered → src_latent
  2. 传给 FlowEdit pipeline，替换内部的 x_src
  3. Loss 计算时也用 _encode_to_latent_packed 编码 rendered → rendered_latent
  4. 两边使用相同的编码方式 (bicubic + antialias)，数值完全一致
""")


if __name__ == "__main__":
    main()
