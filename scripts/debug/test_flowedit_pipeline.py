#!/usr/bin/env python3
"""
FlowEditPipeline 迭代优化测试脚本。

验证 FlowEdit 双分支差分编辑 + CSD Loss 迭代优化是否有效。

配置约定（与训练一致）：
- target_prompt = source_prompt = prompt
- negative_prompt_tgt = negative_prompt_src = negative_prompt
- true_cfg_scale_tgt = cfg_scale, true_cfg_scale_src = -cfg_scale

流程：
1. 加载 FlowEditPipeline
2. 单次 FlowEdit 编辑，验证编辑本身有效
3. 初始化可优化 latent，迭代优化验证 CSD 梯度有效
   - 外层：运行 FlowEdit 推理（昂贵）
   - 内层：复用 tracker 多次优化 latent（便宜）
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
    """将 packed latent 解码为 PIL Image。"""
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


def run_flowedit(pipe, latent, current_image, condition_image, args, device, step_seed):
    """
    调用 FlowEdit Pipeline 一次。

    配置约定：
    - target_prompt = source_prompt = args.prompt
    - negative_prompt_tgt = negative_prompt_src = args.negative_prompt
    - true_cfg_scale_tgt = +args.cfg_scale
    - true_cfg_scale_src = -args.cfg_scale
    """
    return pipe(
        image=[current_image, condition_image],
        target_prompt=args.prompt,
        source_prompt=args.prompt,
        negative_prompt_src=args.negative_prompt,
        negative_prompt_tgt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale_src=-args.cfg_scale,
        true_cfg_scale_tgt=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        n_max=args.n_max,
        noise_mode=args.noise_mode,
        use_tgt_record=args.tgt_weight > 0,
        use_src_record=args.src_weight > 0,
        csd_pos_mode=args.csd_pos_mode,
        csd_neg_mode=args.csd_neg_mode,
        remove_tgt_neg=args.remove_tgt_neg,
        src_latent=latent,
        generator=torch.Generator(device=device).manual_seed(step_seed),
        height=args.height,
        width=args.width,
    )


def compute_loss(latent, result, args):
    """计算分支 loss（tgt + src 加权）。"""
    total = torch.tensor(0.0, device=latent.device, dtype=torch.float32)  # []

    if args.tgt_weight > 0 and result.tracker_tgt is not None:
        loss_tgt = result.tracker_tgt.loss(
            src=latent, mse_weight=args.mse_weight, csd_weight=args.csd_weight,
            reduce=args.reduce_mode, ada=args.ada, eps=args.ada_eps,
        )  # []
        total = total + args.tgt_weight * loss_tgt  # []

    if args.src_weight > 0 and result.tracker_src is not None:
        loss_src = result.tracker_src.loss(
            src=latent, mse_weight=args.mse_weight, csd_weight=args.csd_weight,
            reduce=args.reduce_mode, ada=args.ada, eps=args.ada_eps,
        )  # []
        total = total + args.src_weight * loss_src  # []

    return total  # []


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEditPipeline 迭代优化测试")

    # 模型
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")

    # 输入输出
    parser.add_argument("--condition_image", type=str, required=True, help="条件图像路径")
    parser.add_argument("--rendered_image", type=str, default=None, help="渲染图像路径（默认同条件图）")
    parser.add_argument("--prompt", type=str, default="A detailed 3D model with realistic textures")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/flowedit_test")

    # FlowEdit 参数
    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG 强度（tgt=+cfg, src=-cfg）")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="guidance-distilled 模型的 guidance scale")
    parser.add_argument("--n_max", type=int, default=9)
    parser.add_argument("--reduce_mode", type=str, default="final",
                        choices=["final", "mean", "weighted", "inv_weighted"],
                        help="多步 loss 聚合方式")

    # 优化参数
    parser.add_argument("--num_optimization_steps", type=int, default=50,
                        help="外层步数（FlowEdit 推理次数）")
    parser.add_argument("--inner_steps", type=int, default=5,
                        help="每次 FlowEdit 推理后的内层优化步数")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["Adam", "AdamW", "SGD"])

    # 调试
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # ---- 硬编码参数 ----
    args.height = 1024
    args.width = 1024
    args.noise_mode = "aligned"
    args.csd_pos_mode = "cond"
    args.csd_neg_mode = "uncond"
    args.remove_tgt_neg = False
    args.tgt_weight = 1.0
    args.src_weight = 0.0
    args.mse_weight = 1.0
    args.csd_weight = 0.0
    args.ada = False
    args.ada_eps = 1e-4
    args.init_noise_scale = 1.0

    seed_everything(args.seed)

    print(f"[INFO] Prompt: {args.prompt}")
    print(f"[INFO] Negative: {args.negative_prompt}")
    print(f"[INFO] CFG: tgt=+{args.cfg_scale}, src=-{args.cfg_scale}")
    print(f"[INFO] Steps: {args.num_inference_steps}, n_max: {args.n_max}, noise: {args.noise_mode}")
    print(f"[INFO] Loss: tgt_w={args.tgt_weight}, src_w={args.src_weight}, "
          f"mse={args.mse_weight}, csd={args.csd_weight}")
    print(f"[INFO] Image: {args.height}x{args.width}")
    print(f"[INFO] Optim: {args.optimizer_type}, lr={args.learning_rate}, "
          f"outer={args.num_optimization_steps}, inner={args.inner_steps}")
    print("-" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)

    # =====================================================================
    # 1. 加载 Pipeline
    # =====================================================================
    print("[INFO] 加载 FlowEditPipeline...")
    t0 = time.time()

    from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
    pipe = FlowEditFullPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    print(f"[INFO] Pipeline 加载完成，用时: {time.time() - t0:.1f}s")

    # =====================================================================
    # 2. 准备图像
    # =====================================================================
    condition_image = Image.open(args.condition_image).convert("RGB")
    rendered_image = (
        Image.open(args.rendered_image).convert("RGB")
        if args.rendered_image
        else condition_image.copy()
    )

    condition_image.save(os.path.join(args.output_dir, "condition.png"))
    rendered_image.save(os.path.join(args.output_dir, "rendered_init.png"))

    # =====================================================================
    # 3. 单次 FlowEdit 编辑（验证编辑本身有效）
    # =====================================================================
    print("[INFO] ===== 单次 FlowEdit 编辑测试 =====")
    with torch.no_grad():
        result = run_flowedit(
            pipe, None, rendered_image, condition_image, args, device, args.seed
        )

    if result.images and not isinstance(result.images, torch.Tensor):
        result.images[0].save(os.path.join(args.output_dir, "flowedit_single_edit.png"))
        print("[INFO] 编辑图已保存")

    for name, tracker in [("tgt", result.tracker_tgt), ("src", result.tracker_src)]:
        if tracker is None:
            continue
        print(f"[INFO] Tracker {name}: {len(tracker)} 步")
        for i, (pos, neg, t) in enumerate(zip(tracker.x0_pos, tracker.x0_neg, tracker.ts)):
            diff = (pos - neg).norm().item()
            print(f"  t={t:.4f}: ||x0_pos - x0_neg|| = {diff:.4f}")

    print("-" * 60)

    # =====================================================================
    # 4. 初始化可优化 latent（随机噪声）
    # =====================================================================
    generator = torch.Generator(device=device).manual_seed(args.seed)
    num_channels = pipe.transformer.config.in_channels // 4
    latent_h = 2 * (args.height // (pipe.vae_scale_factor * 2))
    latent_w = 2 * (args.width // (pipe.vae_scale_factor * 2))
    seq_len = (latent_h // 2) * (latent_w // 2)

    latent = torch.randn(
        1, seq_len, num_channels * 4,
        device=device, dtype=dtype, generator=generator,
    ) * args.init_noise_scale  # [1, seq, C*4]
    latent = latent.requires_grad_(True)
    print(f"[INFO] Latent shape: {latent.shape}")

    # =====================================================================
    # 5. 设置优化器
    # =====================================================================
    optimizer = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[args.optimizer_type]([latent], lr=args.learning_rate)

    # =====================================================================
    # 6. 迭代优化（外层：FlowEdit 推理，内层：复用 tracker 多次优化）
    # =====================================================================
    print("[INFO] ===== 开始迭代优化 =====")
    print(f"[INFO] 外层步数: {args.num_optimization_steps}, 内层步数: {args.inner_steps}")
    print(f"[INFO] 总优化步数: {args.num_optimization_steps * args.inner_steps}")
    t0 = time.time()
    loss_history = []
    global_opt_step = 0

    pbar = tqdm(range(args.num_optimization_steps), desc="FlowEdit 推理")
    for outer_step in pbar:

        # ---- 外层：运行一次 FlowEdit（昂贵） ----
        with torch.no_grad():
            current_image = decode_latent_to_image(pipe, latent, args.height, args.width)

        result = run_flowedit(
            pipe, latent, current_image, condition_image,
            args, device, args.seed + outer_step,
        )

        # ---- 内层：复用 tracker 多次优化（便宜） ----
        for inner_step in range(args.inner_steps):
            optimizer.zero_grad()
            loss = compute_loss(latent, result, args)  # []
            loss.backward()

            torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)
            global_opt_step += 1

        # ---- 调试输出 ----
        if outer_step % args.save_interval == 0:
            grad_norm = latent.grad.norm().item() if latent.grad is not None else 0.0
            print(f"\n[Outer {outer_step}] loss={loss_val:.6f}, "
                  f"grad_norm={grad_norm:.6f}, total_opt_steps={global_opt_step}")

            with torch.no_grad():
                decode_latent_to_image(pipe, latent, args.height, args.width).save(
                    os.path.join(debug_dir, f"latent_{outer_step:04d}.png")
                )
                if result.images and not isinstance(result.images, torch.Tensor):
                    result.images[0].save(
                        os.path.join(debug_dir, f"edited_{outer_step:04d}.png")
                    )

        pbar.set_postfix(loss=f"{loss_val:.6f}", opt=global_opt_step)

    opt_time = time.time() - t0
    print(f"\n[INFO] 优化完成，用时: {opt_time:.1f}s")

    # =====================================================================
    # 7. 保存最终结果
    # =====================================================================
    with torch.no_grad():
        final_image = decode_latent_to_image(pipe, latent, args.height, args.width)
        final_image.save(os.path.join(args.output_dir, "final_latent.png"))

        final_result = run_flowedit(
            pipe, None, final_image, condition_image, args, device, args.seed
        )
        if final_result.images and not isinstance(final_result.images, torch.Tensor):
            final_result.images[0].save(os.path.join(args.output_dir, "final_edited.png"))

    # Loss 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title(f"FlowEdit Optimization (outer={args.num_optimization_steps}, inner={args.inner_steps})")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 保存参数
    with open(os.path.join(args.output_dir, "parameters.txt"), "w") as f:
        f.write("FlowEditPipeline 迭代优化测试参数:\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n优化时间: {opt_time:.1f}s\n")
        f.write(f"总优化步数: {global_opt_step}\n")
        f.write(f"最终 Loss: {loss_history[-1]:.6f}\n")

    print(f"[SUCCESS] 结果保存到: {args.output_dir}")
    print(f"[INFO] 最终 Loss: {loss_history[-1]:.6f}")


if __name__ == "__main__":
    main()
