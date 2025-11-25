#!/usr/bin/env python3
"""测试 OpenAI 格式的 Gemini VLM 编码器（支持高并发）"""

import torch
from PIL import Image
from reward_models.camera_normal_scorer.encoders.vlm_encoder import GeminiOpenAIEncoder

# 配置
API_SOURCE = "3"  # 形状: 字符串，映射到 vlm_encoder.API_KEYS / BASE_URLS
REF_PATH = "/home/zhiyuan_ma/code2/flow_grpo_custom_2nd/dataset/alphaimages_1k/test/images/00098.png"
CAND_PATH = "/home/zhiyuan_ma/code2/flow_grpo_custom_2nd/dataset/alphaimages_1k/test/normals/R518/00098.png"


def main():
    # 形状: 标量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 形状: PIL(H,W,3), PIL(R,R,3)
    ref_img = Image.open(REF_PATH).convert("RGB")
    cand_img = Image.open(CAND_PATH).convert("RGB")
    
    encoder = GeminiOpenAIEncoder(
        device=device,
        api_source=API_SOURCE,
        model="gemini-2.5-flash",
        max_concurrent=50,  # 形状: 标量，最大并发数
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
    scores_batch = encoder.score_pairs(  # 形状: (10,)
        group_pils=[ref_img],
        mesh_pils=[cand_img] * 10,
        mesh_group_indices=[0] * 10,
    )
    elapsed = time.time() - start
    print(f"批量分数: {scores_batch.tolist()}")
    print(f"耗时: {elapsed:.2f}s ({10/elapsed:.1f} 对/秒)")


if __name__ == "__main__":
    main()

