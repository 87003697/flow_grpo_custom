#!/usr/bin/env python3
"""
QwenImageDistillationPipeline 迭代优化测试脚本。

类似 Flux GuidancePipeline 的测试方式：
- 使用优化器迭代更新 latent
- 通过 CSD/MSE Loss 引导生成
- 可视化迭代优化过程

流程：
1. 加载 Pipeline（VAE, Transformer, Text Encoder）
2. 准备条件图并编码为 latent
3. 初始化可优化的 latent（随机噪声）
4. 迭代优化：
   - 调用 Pipeline 获取 x0 预测
   - 计算 CSD Loss
   - 优化器更新 latent
5. 保存最终图像和优化过程视频
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int):
    """设置随机种子以确保可重复性"""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_video(frames: list, output_path: str, fps: int = 10):
    """保存视频或 GIF"""
    try:
        import imageio.v2 as imageio
        
        # 尝试使用 ffmpeg 保存为 mp4
        try:
            writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', codec='libx264')
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"[INFO] 视频已保存到: {output_path}")
        except Exception:
            # 备选方案：保存为 GIF
            gif_path = output_path.replace('.mp4', '.gif')
            imageio.mimsave(gif_path, frames, duration=1.0/fps)
            print(f"[INFO] GIF 已保存到: {gif_path}")
            
    except ImportError:
        print("[WARNING] imageio 未安装，无法保存视频")
    except Exception as e:
        print(f"[WARNING] 保存视频失败: {e}")


def decode_latent_to_image(
    pipe, 
    latent: torch.Tensor, 
    height: int, 
    width: int
) -> Image.Image:
    """
    将 packed latent 解码为 PIL Image。
    
    Args:
        pipe: Pipeline 实例
        latent: [B, seq, C*4] packed latent
        height: 图像高度
        width: 图像宽度
    
    Returns:
        PIL Image
    """
    # Unpack latent
    unpacked = pipe._unpack_latents(
        latent.detach(), 
        height, 
        width, 
        pipe.vae_scale_factor
    )  # [B, C, 1, H_lat, W_lat]
    
    # 反标准化
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.latent_channels, 1, 1, 1)
        .to(unpacked.device, unpacked.dtype)
    )  # [1, C_lat, 1, 1, 1]
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.latent_channels, 1, 1, 1)
        .to(unpacked.device, unpacked.dtype)
    )  # [1, C_lat, 1, 1, 1]
    
    denormalized = unpacked * latents_std + latents_mean  # [B, C, 1, H_lat, W_lat]
    
    # VAE decode
    with torch.no_grad():
        decoded = pipe.vae.decode(denormalized, return_dict=False)[0]  # [B, 3, T, H, W] 或 [B, 3, H, W]
    
    # 处理可能的多余维度
    while decoded.dim() > 4:
        decoded = decoded.squeeze(2)  # 去除时间维度 T
    
    # 转换为 PIL Image: [B, C, H, W] -> [H, W, C]
    decoded = decoded[0].float().clamp(-1, 1)  # [C, H, W]
    decoded = (decoded + 1) / 2  # [0, 1]
    decoded = decoded.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    decoded = (decoded * 255).astype(np.uint8)  # uint8
    
    return Image.fromarray(decoded)


def encode_image_to_latent(
    pipe,
    image: Image.Image,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    将 PIL Image 编码为 packed latent。
    
    Args:
        pipe: Pipeline 实例
        image: PIL Image
        height: 目标高度
        width: 目标宽度
        dtype: 数据类型
        device: 设备
    
    Returns:
        [B, seq, C*4] packed latent
    """
    # 预处理图像
    image = image.resize((width, height))
    image_tensor = pipe.image_processor.preprocess(image, height, width)  # [B, C, H, W]
    image_tensor = image_tensor.unsqueeze(2).to(device=device, dtype=dtype)  # [B, C, 1, H, W]
    
    # VAE encode
    with torch.no_grad():
        latent = pipe._encode_vae_image(image_tensor, generator=None)  # [B, C_lat, 1, H_lat, W_lat]
    
    # Pack latent
    B, C, _, H_lat, W_lat = latent.shape
    packed = pipe._pack_latents(latent, B, C, H_lat, W_lat)  # [B, seq, C*4]
    
    return packed


def parse_args():
    parser = argparse.ArgumentParser(description="QwenImageDistillationPipeline 迭代优化测试")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511",
                        help="Qwen-Image-Edit 模型路径")
    
    # 输入输出
    parser.add_argument("--condition_image", type=str, required=True,
                        help="条件图像路径")
    parser.add_argument("--prompt", type=str, 
                        default="A detailed 3D model with realistic textures and lighting",
                        help="目标文本描述")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, low quality, distorted",
                        help="负面描述")
    parser.add_argument("--output_dir", type=str, 
                        default="outputs/qwen_distillation_test",
                        help="输出目录")
    
    # 图像参数
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    
    # 优化参数
    parser.add_argument("--num_optimization_steps", type=int, default=100,
                        help="优化迭代步数")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="学习率")
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                        choices=["Adam", "AdamW", "SGD"],
                        help="优化器类型")
    
    # Loss 参数
    parser.add_argument("--loss_type", type=str, default="csd",
                        choices=["csd", "mse", "mixed"],
                        help="Loss 类型")
    parser.add_argument("--mse_weight", type=float, default=0.0,
                        help="MSE Loss 权重（mixed 模式）")
    parser.add_argument("--csd_weight", type=float, default=1.0,
                        help="CSD Loss 权重")
    parser.add_argument("--ada", action="store_true",
                        help="使用自适应归一化")
    parser.add_argument("--ada_eps", type=float, default=1e-4,
                        help="自适应归一化 epsilon")
    
    # CFG 参数
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="CFG 强度")
    
    # 时间步参数
    parser.add_argument("--min_step_percent", type=float, default=0.02,
                        help="最小时间步比例")
    parser.add_argument("--max_step_percent", type=float, default=0.98,
                        help="最大时间步比例")
    parser.add_argument("--num_timesteps", type=int, default=1,
                        help="每次优化采样的时间步数量（MTS）")
    parser.add_argument("--noise_mode", type=str, default="fixed",
                        choices=["random", "fixed", "aligned", 
                                 "inversion_cond", "inversion_uncond", "inversion_cfg"],
                        help="噪声模式")
    
    # CSD 正/负样本模式
    parser.add_argument("--csd_pos_mode", type=str, default="cond",
                        choices=["cond", "cfg", "cfg_rescale"],
                        help="CSD 正样本来源: cond=纯条件(CFG=1), cfg=原始CFG, cfg_rescale=CFG+L2归一化")
    parser.add_argument("--csd_neg_mode", type=str, default="uncond",
                        choices=["uncond", "cond"],
                        help="CSD 负样本来源: uncond=纯无条件, cond=纯条件")
    
    # 初始化方式
    parser.add_argument("--init_mode", type=str, default="random",
                        choices=["random", "condition"],
                        help="latent 初始化方式：random=随机噪声，condition=从条件图编码")
    parser.add_argument("--init_noise_scale", type=float, default=1.0,
                        help="初始噪声缩放（init_mode=random 时使用）")
    
    # 调试参数
    parser.add_argument("--save_debug_images", action="store_true",
                        help="保存调试图像")
    parser.add_argument("--debug_save_interval", type=int, default=10,
                        help="调试图像保存间隔")
    parser.add_argument("--generate_video", action="store_true",
                        help="生成优化过程视频")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="计算精度")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置设备和精度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 精度: {args.dtype}")
    
    # 设置随机种子
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    print(f"[INFO] 开始 Qwen-Image 蒸馏测试...")
    print(f"[INFO] 提示: {args.prompt}")
    print(f"[INFO] 负面提示: {args.negative_prompt}")
    print(f"[INFO] Loss 类型: {args.loss_type}")
    print(f"[INFO] 优化步数: {args.num_optimization_steps}")
    print(f"[INFO] 学习率: {args.learning_rate}")
    print(f"[INFO] CFG 强度: {args.cfg_scale}")
    print(f"[INFO] CSD 正样本: {args.csd_pos_mode}")
    print(f"[INFO] CSD 负样本: {args.csd_neg_mode}")
    print(f"[INFO] 图像尺寸: {args.width}x{args.height}")
    print("-" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    debug_images_path = None
    if args.save_debug_images:
        debug_images_path = os.path.join(args.output_dir, "debug_images")
        os.makedirs(debug_images_path, exist_ok=True)
    
    # =========================================================================
    # 1. 加载 Pipeline
    # =========================================================================
    print("[INFO] 加载 QwenImageDistillationPipeline...")
    start_time = time.time()
    
    from edit4shape.guidance.pipelines.qwen_image_edit.distillation import (
        QwenImageDistillationPipeline
    )
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
    from diffusers.models import QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
    
    # 加载各组件
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=dtype
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=dtype
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.model_path, subfolder="tokenizer"
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path, subfolder="processor"
    )
    
    pipe = QwenImageDistillationPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    pipe.to(device)
    
    # 启用内存优化
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    if hasattr(pipe, 'enable_vae_tiling'):
        pipe.enable_vae_tiling()
    
    torch.cuda.empty_cache()
    
    load_time = time.time() - start_time
    print(f"[INFO] Pipeline 加载完成，用时: {load_time:.2f}秒")
    
    # =========================================================================
    # 2. 准备条件图
    # =========================================================================
    print(f"[INFO] 加载条件图像: {args.condition_image}")
    condition_image = Image.open(args.condition_image).convert("RGB")
    
    # 调整尺寸
    if condition_image.size != (args.width, args.height):
        print(f"[INFO] 调整图像尺寸从 {condition_image.size} 到 {args.width}x{args.height}")
        condition_image = condition_image.resize((args.width, args.height))
    
    # 保存条件图
    condition_image.save(os.path.join(args.output_dir, "condition_image.png"))
    
    # =========================================================================
    # 3. 初始化可优化的 latent
    # =========================================================================
    print(f"[INFO] 初始化 latent (模式: {args.init_mode})...")
    
    # 计算 latent 尺寸
    latent_height = 2 * (args.height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (args.width // (pipe.vae_scale_factor * 2))
    num_channels = pipe.latent_channels
    seq_len = (latent_height // 2) * (latent_width // 2)
    
    if args.init_mode == "random":
        # 随机初始化
        latent = torch.randn(
            1, seq_len, num_channels * 4,
            device=device, dtype=dtype, generator=generator
        ) * args.init_noise_scale  # [1, seq, C*4]
    else:
        # 从条件图编码
        latent = encode_image_to_latent(
            pipe, condition_image, args.height, args.width, dtype, device
        )  # [1, seq, C*4]
    
    # 设置 requires_grad
    latent = latent.clone().detach().requires_grad_(True)
    
    print(f"[INFO] Latent shape: {latent.shape}")
    
    # =========================================================================
    # 4. 设置优化器
    # =========================================================================
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW([latent], lr=args.learning_rate)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD([latent], lr=args.learning_rate)
    
    print(f"[INFO] 优化器: {args.optimizer_type}, 学习率: {args.learning_rate}")
    
    # =========================================================================
    # 5. 迭代优化
    # =========================================================================
    print(f"[INFO] 开始迭代优化...")
    generation_start_time = time.time()
    
    debug_frames = []
    loss_history = []
    grad_norm = 0.0  # 初始化梯度范数
    
    pbar = tqdm(range(args.num_optimization_steps), desc="优化中")
    
    for step in pbar:
        optimizer.zero_grad()
        
        # 解码当前 latent 为图像（用于 Pipeline 输入）
        with torch.no_grad():
            current_image = decode_latent_to_image(pipe, latent, args.height, args.width)
        
        # 调用 Pipeline
        result = pipe(
            image=[current_image, condition_image],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            src_latent=latent,
            min_step_percent=args.min_step_percent,
            max_step_percent=args.max_step_percent,
            num_timesteps=args.num_timesteps,
            noise_mode=args.noise_mode,
            csd_pos_mode=args.csd_pos_mode,
            csd_neg_mode=args.csd_neg_mode,
            generator=generator,
        )
        
        tracker = result.tracker
        
        # 调试：检查 x0_pos 和 x0_neg 的差异
        if step % 10 == 0 and len(tracker.x0_pos) > 0:
            x0_pos = tracker.x0_pos[-1]
            x0_neg = tracker.x0_neg[-1]
            diff = (x0_pos - x0_neg).norm().item()
            print(f"[Step {step}] x0_pos - x0_neg norm: {diff:.6f}")
        
        # 计算 Loss
        if args.loss_type == "csd":
            loss = tracker.loss(
                src=latent,
                mse_weight=0.0,
                csd_weight=args.csd_weight,
                ada=args.ada,
                eps=args.ada_eps,
            )
        elif args.loss_type == "mse":
            loss = tracker.loss(
                src=latent,
                mse_weight=1.0,
                csd_weight=0.0,
                ada=args.ada,
                eps=args.ada_eps,
            )
        else:  # mixed
            loss = tracker.loss(
                src=latent,
                mse_weight=args.mse_weight,
                csd_weight=args.csd_weight,
                ada=args.ada,
                eps=args.ada_eps,
            )
        
        # 反向传播
        loss.backward()
        
        # 调试：检查梯度
        if step % 10 == 0:
            if latent.grad is not None:
                grad_norm = latent.grad.norm().item()
                grad_max = latent.grad.abs().max().item()
                print(f"[Step {step}] Gradient norm: {grad_norm:.6f}, max: {grad_max:.6f}")
            else:
                print(f"[Step {step}] WARNING: No gradient!")
        
        # 记录优化前的 latent 范数（用于对比）
        latent_before = latent.data.clone()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)
        
        # 优化器更新
        optimizer.step()
        
        # 调试：检查 latent 变化量
        if step % 10 == 0:
            latent_change = (latent.data - latent_before).norm().item()
            print(f"[Step {step}] Latent change: {latent_change:.6f}")
        
        # 记录 loss
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        # 更新进度条
        pbar.set_postfix({"loss": f"{loss_val:.6f}", "grad": f"{grad_norm:.4f}" if latent.grad is not None else "N/A"})
        
        # 保存调试图像
        if args.save_debug_images and (step % args.debug_save_interval == 0 or step == args.num_optimization_steps - 1):
            with torch.no_grad():
                debug_image = decode_latent_to_image(pipe, latent, args.height, args.width)
                debug_image.save(os.path.join(debug_images_path, f"step_{step:04d}.png"))
                debug_frames.append(np.array(debug_image))
        
        # 打印详细信息
        if step % 10 == 0:
            print(f"[Step {step}] Loss: {loss_val:.6f}")
    
    generation_time = time.time() - generation_start_time
    print(f"[INFO] 优化完成，用时: {generation_time:.2f}秒")
    
    # =========================================================================
    # 6. 保存最终结果
    # =========================================================================
    print("[INFO] 保存结果...")
    
    # 保存最终图像
    with torch.no_grad():
        final_image = decode_latent_to_image(pipe, latent, args.height, args.width)
        final_image.save(os.path.join(args.output_dir, "final_image.png"))
    print(f"[INFO] 最终图像已保存")
    
    # 生成视频
    if args.generate_video and debug_frames:
        video_path = os.path.join(args.output_dir, "optimization_process.mp4")
        save_video(debug_frames, video_path, fps=10)
    
    # 保存 loss 曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Optimization Loss Curve")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
        plt.close()
        print("[INFO] Loss 曲线已保存")
    except ImportError:
        print("[WARNING] matplotlib 未安装，无法保存 loss 曲线")
    
    # 保存参数
    params_file = os.path.join(args.output_dir, "parameters.txt")
    with open(params_file, "w") as f:
        f.write("QwenImageDistillationPipeline 测试参数:\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nPipeline 加载时间: {load_time:.2f}秒\n")
        f.write(f"优化时间: {generation_time:.2f}秒\n")
        f.write(f"总用时: {load_time + generation_time:.2f}秒\n")
        f.write(f"最终 Loss: {loss_history[-1]:.6f}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"[SUCCESS] 测试完成！结果保存到: {args.output_dir}")
    print(f"[INFO] 总用时: {load_time + generation_time:.2f}秒")
    print(f"[INFO] 最终 Loss: {loss_history[-1]:.6f}")
    print("✅ QwenImageDistillationPipeline 测试完成!")


if __name__ == "__main__":
    main()
