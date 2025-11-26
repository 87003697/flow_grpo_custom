#!/usr/bin/env python3
"""测试 OpenAI 格式的 Gemini VLM 编码器（支持高并发）"""

import argparse
import torch
from PIL import Image
from reward_models.camera_normal_scorer.encoders.vlm_encoder import (
    GeminiOpenAIEncoder,
    GeminiOpenAIGroupEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(description="测试 GeminiOpenAI 编码器（base & group）")
    parser.add_argument(
        "--encoder-type",
        choices=["base", "group"],
        default="base",
        help="选择测试的编码器类型",
    )
    parser.add_argument("--api-source", default="3", help="API key / base_url 映射编号")
    parser.add_argument("--ref-path", default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--cand-path", default="dataset/alphaimages_1k/test/normals/R518/00098.png", help="单张候选图路径")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--debug-response", action="store_true")
    parser.add_argument("--batch", type=int, default=10, help="批量测试的 pair/组 数量")
    return parser.parse_args()


def main():
    args = parse_args()
    # 形状: 标量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 形状: PIL(H,W,3), PIL(R,R,3)
    ref_img = Image.open(args.ref_path).convert("RGB")
    cand_imgs = [Image.open(args.cand_path).convert("RGB")]  # 形状: 列表(1)

    is_group = args.encoder_type == "group"
    prompt_version = args.prompt_version

    encoder_cls = GeminiOpenAIGroupEncoder if is_group else GeminiOpenAIEncoder
    encoder = encoder_cls(
        device=device,
        api_source=args.api_source,
        model="gemini-2.5-flash",
        max_concurrent=args.max_concurrent,  # 形状: 标量，最大并发数
        prompt_version=prompt_version,  # 形状: 字符串
        max_tokens=args.max_tokens,  # 形状: 标量
        thinking_enabled=args.thinking,  # 形状: 布尔
        debug_raw_response=args.debug_response,  # 形状: 布尔
    )

    mode_label = "group" if is_group else "base"
    print(f"测试 {mode_label} 模式下的单次调用...")
    if is_group:
        single_scores = encoder.score_pairs(  # 形状: (num_cand,)
            group_pils=[ref_img],
            mesh_pils=cand_imgs,
            mesh_group_indices=[0] * len(cand_imgs),
        )
        print(f"单组 {len(cand_imgs)} 张候选的分数: {single_scores.tolist()}")
    else:
        scores = encoder.score_pairs(  # 形状: (1,)
            group_pils=[ref_img],
            mesh_pils=[cand_imgs[0]],
            mesh_group_indices=[0],
        )
        print(f"Gemini 相似度分数: {float(scores.item()):.4f}")

    # 测试批量并发
    batch_units = max(1, args.batch)
    mode_desc = f"{batch_units} 组，每组 {len(cand_imgs)} 张候选" if is_group else f"{batch_units} 对图像"
    print(f"\n测试批量并发（{mode_desc}）...")
    import time
    start = time.time()
    scores_batch = encoder.score_pairs(  # 形状: (batch_units,)
        group_pils=[ref_img],
        mesh_pils=[cand_imgs[0]] * batch_units,
        mesh_group_indices=[0] * batch_units,
    )
    elapsed = time.time() - start
    print(f"批量分数: {scores_batch.tolist()}")
    throughput = (batch_units / elapsed) if elapsed > 0 else float("inf")
    unit_label = "组/秒" if is_group else "对/秒"
    print(f"耗时: {elapsed:.2f}s ({throughput:.1f} {unit_label})")


if __name__ == "__main__":
    main()

