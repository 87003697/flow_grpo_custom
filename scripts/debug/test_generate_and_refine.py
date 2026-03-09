#!/usr/bin/env python3
"""
Generate → FlowEdit Refine 测试脚本。

两阶段流程：
  Stage 1: 用 QwenImageEditPlusPipeline 标准推理生成编辑结果
  Stage 2: 用 FlowEditPipeline 对生成结果做差分 Refine

Refine 配置约定：
  - target_prompt = source_prompt = prompt（与生成相同）
  - negative_prompt_tgt = negative_prompt_src = negative_prompt（与生成相同）
  - true_cfg_scale_tgt = +cfg_scale
  - true_cfg_scale_src = -cfg_scale（负 CFG，核心创新点）

复用现有 pipeline，不创建新的 pipeline 类。
"""

import argparse
import os
import time

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_latent_to_image(pipe, latent, height, width):
    """将 packed latent [B, seq, C] 解码为 PIL Image。"""
    unpacked = pipe._unpack_latents(
        latent.detach(), height, width, pipe.vae_scale_factor
    )  # [B, C, 1, H_lat, W_lat]

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(unpacked.device, unpacked.dtype)
    )  # [1, C_lat, 1, 1, 1]
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(unpacked.device, unpacked.dtype)  # [1, C_lat, 1, 1, 1]

    denormalized = unpacked / latents_std + latents_mean  # [B, C, 1, H_lat, W_lat]

    with torch.no_grad():
        decoded = pipe.vae.decode(denormalized, return_dict=False)[0]  # [B, 3, T, H, W]

    while decoded.dim() > 4:
        decoded = decoded.squeeze(2)

    decoded = decoded[0].float().clamp(-1, 1)  # [C, H, W]
    decoded = (decoded + 1) / 2  # [0, 1]
    decoded = decoded.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    decoded = (decoded * 255).astype(np.uint8)

    return Image.fromarray(decoded)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate → FlowEdit Refine 测试"
    )

    # 模型
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")

    # 输入输出
    parser.add_argument("--input_image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="编辑指令")
    parser.add_argument("--negative_prompt", type=str, default="", help="负提示词")
    parser.add_argument("--output_dir", type=str, default="outputs/generate_and_refine")

    # 图像尺寸（None 时自动从输入图像推算）
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    # Stage 1: 生成参数
    parser.add_argument("--gen_steps", type=int, default=50, help="生成阶段推理步数")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG 强度")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="guidance-distilled 模型的 guidance scale")

    # Stage 2: Refine 参数
    parser.add_argument("--refine_steps", type=int, default=20, help="Refine 阶段推理步数")
    parser.add_argument("--refine_n_max", type=int, default=20, help="FlowEdit 生效步数范围")
    parser.add_argument("--refine_cfg_scale", type=float, default=None,
                        help="Refine 阶段的 CFG 强度（默认同 cfg_scale）")
    parser.add_argument("--noise_mode", type=str, default="aligned",
                        choices=["random", "fixed", "aligned"])
    parser.add_argument("--num_refine_rounds", type=int, default=1,
                        help="Refine 轮数（>1 时迭代 refine）")

    # 调试
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    refine_cfg = args.refine_cfg_scale if args.refine_cfg_scale is not None else args.cfg_scale

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generate → FlowEdit Refine 测试")
    print("=" * 60)
    print(f"  Prompt:          {args.prompt}")
    print(f"  Negative:        {args.negative_prompt}")
    print(f"  CFG:             {args.cfg_scale}")
    print(f"  Guidance Scale:  {args.guidance_scale}")
    print(f"  Gen Steps:       {args.gen_steps}")
    print(f"  Refine Steps:    {args.refine_steps}")
    print(f"  Refine n_max:    {args.refine_n_max}")
    print(f"  Refine CFG:      tgt=+{refine_cfg}, src=-{refine_cfg}")
    print(f"  Refine Rounds:   {args.num_refine_rounds}")
    print(f"  Noise Mode:      {args.noise_mode}")
    print("=" * 60)

    # =================================================================
    # 1. 加载输入图像
    # =================================================================
    input_image = Image.open(args.input_image).convert("RGB")
    input_image.save(os.path.join(args.output_dir, "input.png"))
    print(f"[INFO] 输入图像: {args.input_image} ({input_image.size[0]}x{input_image.size[1]})")

    # =================================================================
    # 2. 加载 Pipeline（共享模型组件）
    # =================================================================
    print("[INFO] 加载 QwenImageEditPlusPipeline...")
    t0 = time.time()

    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        QwenImageEditPlusPipeline,
    )
    from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline

    pipe_edit = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)

    # 用相同组件构建 FlowEdit pipeline（零开销，只是引用同一套模型）
    pipe_flow = FlowEditFullPipeline(
        scheduler=pipe_edit.scheduler,
        vae=pipe_edit.vae,
        text_encoder=pipe_edit.text_encoder,
        tokenizer=pipe_edit.tokenizer,
        processor=pipe_edit.processor,
        transformer=pipe_edit.transformer,
    )

    print(f"[INFO] Pipeline 加载完成，用时: {time.time() - t0:.1f}s")

    # =================================================================
    # 3. Stage 1: 标准生成
    # =================================================================
    print("\n[Stage 1] 标准 QwenImageEditPlus 生成...")
    t1 = time.time()

    # 注意：negative_prompt 即使是空字符串也要传入（不能转 None），否则 CFG 会被禁用
    neg_prompt = args.negative_prompt if args.negative_prompt else " "  # 空字符串→空格，确保 CFG 生效

    gen_result = pipe_edit(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=neg_prompt,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.gen_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        output_type="latent",  # 拿 packed latent [B, seq_len, C]
    )

    generated_latent = gen_result.images  # output_type="latent" 时 images 就是 packed latent

    # 获取实际使用的 height/width（可能由 pipeline 自动计算）
    # 从 latent shape 反推: latent shape = [B, (H/16)*(W/16), C*4]
    # seq_len = (H/16) * (W/16), 需要知道 aspect ratio
    # 最简单的方式：使用 pipeline 内部计算的尺寸
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
    image_size = input_image.size  # (W, H)
    calc_w, calc_h = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    actual_h = args.height or calc_h
    actual_w = args.width or calc_w
    multiple_of = pipe_edit.vae_scale_factor * 2
    actual_w = actual_w // multiple_of * multiple_of
    actual_h = actual_h // multiple_of * multiple_of

    gen_time = time.time() - t1
    print(f"[Stage 1] 生成完成，用时: {gen_time:.1f}s")
    print(f"[Stage 1] Latent shape: {generated_latent.shape}")
    print(f"[Stage 1] 图像尺寸: {actual_w}x{actual_h}")

    # 解码并保存生成结果
    gen_image = decode_latent_to_image(pipe_edit, generated_latent, actual_h, actual_w)
    gen_image.save(os.path.join(args.output_dir, "stage1_generated.png"))
    print("[Stage 1] 生成图像已保存")

    # =================================================================
    # 4. Stage 2: FlowEdit Refine（可多轮迭代）
    # =================================================================
    current_latent = generated_latent

    for round_idx in range(args.num_refine_rounds):
        round_label = f"Round {round_idx + 1}/{args.num_refine_rounds}" if args.num_refine_rounds > 1 else ""
        print(f"\n[Stage 2] FlowEdit Refine {round_label}...")
        t2 = time.time()

        # 解码当前 latent 为 PIL Image（FlowEdit 需要 PIL 输入做图像预处理）
        with torch.no_grad():
            current_image = decode_latent_to_image(
                pipe_flow, current_latent, actual_h, actual_w
            )

        # 调用 FlowEdit Refine
        # image=[current_image, input_image]:
        #   - image[0] = current_image → 占位（被 src_latent 覆盖）
        #   - image[1] = input_image → 条件图（VLM 编码 + latent 拼接）
        refine_result = pipe_flow(
            image=[current_image, input_image],
            target_prompt=args.prompt,
            source_prompt=args.prompt,
            negative_prompt_tgt=neg_prompt,
            negative_prompt_src=neg_prompt,
            true_cfg_scale_tgt=refine_cfg,
            true_cfg_scale_src=-refine_cfg,       # 负 CFG（核心）
            src_latent=current_latent,             # Stage 1 输出（或上轮 refine 输出）
            num_inference_steps=args.refine_steps,
            n_max=args.refine_n_max,
            noise_mode=args.noise_mode,
            guidance_scale=args.guidance_scale,
            height=actual_h,
            width=actual_w,
            generator=torch.Generator(device=device).manual_seed(args.seed + round_idx),
            output_type="pil",
            use_tgt_record=False,   # 纯推理，不需要 CSD 记录
            use_src_record=False,
        )

        refine_time = time.time() - t2
        print(f"[Stage 2] Refine {round_label} 完成，用时: {refine_time:.1f}s")

        # 保存 refine 结果
        refine_image = refine_result.images[0]
        if args.num_refine_rounds > 1:
            refine_image.save(
                os.path.join(args.output_dir, f"stage2_refined_round{round_idx + 1}.png")
            )
        else:
            refine_image.save(os.path.join(args.output_dir, "stage2_refined.png"))
        print(f"[Stage 2] Refine 图像已保存")

        # 如果还有下一轮 refine，将当前 refine 输出作为下一轮的 src_latent
        if round_idx < args.num_refine_rounds - 1:
            current_latent = refine_result.latents

    # =================================================================
    # 5. 保存对比图
    # =================================================================
    print("\n[INFO] 生成对比图...")

    images_to_compare = [input_image, gen_image, refine_image]
    labels = ["Input", "Stage1: Generated", "Stage2: Refined"]

    if args.num_refine_rounds > 1:
        # 多轮 refine 时，收集所有中间结果
        labels[-1] = f"Refined (Round {args.num_refine_rounds})"

    # 拼接对比图
    widths = [img.size[0] for img in images_to_compare]
    heights = [img.size[1] for img in images_to_compare]
    total_w = sum(widths) + 20 * (len(images_to_compare) - 1)  # 20px 间距
    max_h = max(heights)

    comparison = Image.new("RGB", (total_w, max_h + 40), color=(255, 255, 255))
    x_offset = 0
    for img, label in zip(images_to_compare, labels):
        comparison.paste(img, (x_offset, 40))
        x_offset += img.size[0] + 20

    comparison.save(os.path.join(args.output_dir, "comparison.png"))

    # =================================================================
    # 6. 保存参数
    # =================================================================
    with open(os.path.join(args.output_dir, "parameters.txt"), "w") as f:
        f.write("Generate → FlowEdit Refine 测试参数:\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n实际图像尺寸: {actual_w}x{actual_h}\n")
        f.write(f"Stage 1 用时: {gen_time:.1f}s\n")
        f.write(f"Stage 2 用时: {refine_time:.1f}s\n")

    print(f"\n[SUCCESS] 所有结果保存到: {args.output_dir}")
    total_time = gen_time + refine_time
    print(f"[INFO] 总用时: {total_time:.1f}s (生成: {gen_time:.1f}s + Refine: {refine_time:.1f}s)")


if __name__ == "__main__":
    main()
