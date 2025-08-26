import torch


@torch.no_grad()
def cosine_rewards_single_to_batch(f_img: torch.Tensor, f_mesh: torch.Tensor) -> torch.Tensor:
    cos = (f_mesh @ f_img.t()).squeeze(-1)  # 形状: (B,)
    return ((cos + 1.0) * 0.5).float()  # 形状: (B,)


