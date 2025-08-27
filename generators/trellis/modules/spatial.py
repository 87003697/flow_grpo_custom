import torch


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape  # 形状 (B, C, H, W, D)
    C_ = C // scale_factor**3  # 形状 标量，对应输出通道数
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)  # 形状 (B, C_, s, s, s, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # 形状 (B, C_, H, s, W, s, D, s)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)  # 形状 (B, C_, sH, sW, sD)
    return x


def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []))  # (B, C, R/ps, ps, R/ps, ps, R/ps, ps)
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]))  # (B, C, ps, ps, ps, R/ps, R/ps, R/ps)
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))  # (B, C*ps^DIM, R/ps, R/ps, R/ps)
    return x


def unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    assert x.shape[1] % (patch_size ** DIM) == 0, f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size ** DIM), *([patch_size] * DIM), *(x.shape[-DIM:]))  # (B, C/ps^DIM, ps, ps, ps, R/ps, R/ps, R/ps)
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))  # (B, C/ps^DIM, R/ps, ps, R/ps, ps, R/ps, ps)
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)])  # (B, C/ps^DIM, R, R, R)
    return x
