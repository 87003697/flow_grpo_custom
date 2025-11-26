#!/usr/bin/env python3
"""测试 OpenAI 格式的 Gemini VLM 编码器（支持高并发）"""

import argparse
import torch
from PIL import Image
from reward_models.camera_normal_scorer.encoders.vlm_encoder import GeminiOpenAIEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="测试 GeminiOpenAIEncoder")
    parser.add_argument("--api-source", default="3", help="API key / base_url 映射编号")
    parser.add_argument("--ref-path", default="dataset/alphaimages_1k/test/images/00098.png")
    parser.add_argument("--cand-path", default="dataset/alphaimages_1k/test/normals/R518/00098.png")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--debug-response", action="store_true")
    parser.add_argument("--batch", type=int, default=10, help="批量测试的 pair 数")
    return parser.parse_args()


def main():
    args = parse_args()
    # 形状: 标量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 形状: PIL(H,W,3), PIL(R,R,3)
    ref_img = Image.open(args.ref_path).convert("RGB")
    cand_img = Image.open(args.cand_path).convert("RGB")
    
    encoder = GeminiOpenAIEncoder(
        device=device,
        api_source=args.api_source,
        model="gemini-2.5-flash",
        max_concurrent=args.max_concurrent,  # 形状: 标量，最大并发数
        prompt_version=args.prompt_version,  # 形状: 字符串
        max_tokens=args.max_tokens,  # 形状: 标量
        thinking_enabled=args.thinking,  # 形状: 布尔
        debug_raw_response=args.debug_response,  # 形状: 布尔
    )
    
    print("测试单对图像...")
    scores = encoder.score_pairs(  # 形状: (1,)
        group_pils=[ref_img],
        mesh_pils=[cand_img],
        mesh_group_indices=[0],
    )
    print(f"Gemini 相似度分数（OpenAI 格式）: {float(scores.item()):.4f}")
    
    # 测试批量并发
    print("\n测试批量并发（10 对图像）...")
    import time
    start = time.time()
    batch_pairs = max(1, args.batch)
    scores_batch = encoder.score_pairs(  # 形状: (batch_pairs,)
        group_pils=[ref_img],
        mesh_pils=[cand_img] * batch_pairs,
        mesh_group_indices=[0] * batch_pairs,
    )
    elapsed = time.time() - start
    print(f"批量分数: {scores_batch.tolist()}")
    print(f"耗时: {elapsed:.2f}s ({batch_pairs/elapsed:.1f} 对/秒)")


if __name__ == "__main__":
    main()

