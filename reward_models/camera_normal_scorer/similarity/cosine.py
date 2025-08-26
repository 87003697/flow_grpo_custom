import torch


@torch.no_grad()
def cosine_rewards_single_to_batch(f_img: torch.Tensor, f_mesh: torch.Tensor) -> torch.Tensor:
    """将单个图像特征与一批 mesh 特征计算余弦相似度，并映射到 [0,1]。

    输入:
        f_img: (1,D)
        f_mesh: (B,D)
    输出:
        (B,) in [0,1]
    参考: 无（标准余弦相似度）。
    """
    cos = (f_mesh @ f_img.t()).squeeze(-1)  # 形状: (B,)
    return ((cos + 1.0) * 0.5).float()  # 形状: (B,)


