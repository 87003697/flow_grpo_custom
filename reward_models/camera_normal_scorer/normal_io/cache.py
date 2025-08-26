from typing import Tuple
import os
import numpy as np
import torch
from PIL import Image


def _cache_path_from_image(image_path_or_name: str, cache_dir: str, resolution: int) -> str:
    """根据图像名和分辨率生成法线缓存 PNG 路径。

    输入:
        image_path_or_name: 原图路径或文件名。
        cache_dir: 缓存根目录。
        resolution: R（正方形）。
    输出:
        缓存文件路径。
    """
    stem = os.path.splitext(os.path.basename(image_path_or_name))[0]  # 形状: 标量
    dir_r = os.path.join(cache_dir, f"R{int(resolution)}")  # 形状: 标量
    return os.path.join(dir_r, f"{stem}.png")  # 形状: 标量


def load_normal_from_cache(image_path: str, cache_dir: str, resolution: int) -> torch.Tensor:
    """从缓存读取法线 PNG 并还原为 [-1,1] 的法线张量。

    输入:
        image_path: 原图路径（用于构造缓存文件名）。
        cache_dir: 缓存目录。
        resolution: R。
    输出:
        normal: (3,R,R) 张量，值域 [-1,1]。
    """
    path = _cache_path_from_image(image_path, cache_dir, resolution)  # 形状: 标量
    img = Image.open(path).convert("RGB")  # 形状: (R,R,3)
    arr = torch.from_numpy(np.array(img)).to(torch.float32)  # 形状: (R,R,3)
    x01 = (arr / 255.0).permute(2, 0, 1)  # 形状: (3,R,R)
    normal = (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)  # 形状: (3,R,R)
    return normal  # 形状: (3,R,R)


def save_normal_cache_png(normal: torch.Tensor, cache_path: str) -> None:
    """将法线张量保存为 PNG 缓存（值域[-1,1]→[0,255]）。

    输入:
        normal: (3,R,R) 或 (1,3,R,R)
        cache_path: 输出文件路径。
    输出:
        None
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if normal.ndim == 4:
        normal = normal[0]  # 形状: (3,R,R)
    x01 = ((normal.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)  # 形状: (3,R,R)
    img = Image.fromarray(x01.permute(1, 2, 0).cpu().numpy())  # 形状: (R,R,3)
    img.save(cache_path)


def _predict_normals_sobel(images_tensor: torch.Tensor, resolution: int) -> torch.Tensor:
    """Sobel 近似的法线预测（用于快速 baseline 或调试）。

    输入:
        images_tensor: (S,3,H,W) [0,1]
        resolution: 目标 R。
    输出:
        (S,3,R,R) in [-1,1]
    参考: 无（Sobel 边缘算子）。
    """
    S, _, H, W = images_tensor.shape  # 形状: 标量, 标量, 标量, 标量
    device = images_tensor.device  # 形状: 设备
    weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=images_tensor.dtype).view(1, 3, 1, 1)  # 形状: (1,3,1,1)
    gray = (images_tensor * weights).sum(dim=1, keepdim=True)  # 形状: (S,1,H,W)
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=images_tensor.dtype).view(1, 1, 3, 3)  # 形状: (1,1,3,3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=images_tensor.dtype).view(1, 1, 3, 3)  # 形状: (1,1,3,3)
    gx = torch.nn.functional.conv2d(torch.nn.functional.pad(gray, (1,1,1,1), mode="replicate"), kx)  # 形状: (S,1,H,W)
    gy = torch.nn.functional.conv2d(torch.nn.functional.pad(gray, (1,1,1,1), mode="replicate"), ky)  # 形状: (S,1,H,W)
    nz = torch.ones_like(gx)  # 形状: (S,1,H,W)
    nx = -gx  # 形状: (S,1,H,W)
    ny = -gy  # 形状: (S,1,H,W)
    n = torch.cat([nx, ny, nz], dim=1)  # 形状: (S,3,H,W)
    n = torch.nn.functional.normalize(n, dim=1)  # 形状: (S,3,H,W)
    n_res = torch.nn.functional.interpolate(n, size=(resolution, resolution), mode="bilinear", align_corners=False)  # 形状: (S,3,R,R)
    return n_res.clamp(-1.0, 1.0)  # 形状: (S,3,R,R)


def _load_images_batch(img_paths, resize_518: bool = True) -> torch.Tensor:
    """加载一批图像为 [0,1] 张量（可选 resize 到 518）。

    输入:
        img_paths: 路径序列。
        resize_518: 是否统一到 518×518。
    输出:
        (S,3,H,W)
    """
    arrs = []  # 长度 S
    for p in img_paths:
        img = Image.open(p).convert("RGB")  # 形状: (h,w,3)
        if resize_518:
            img = img.resize((518, 518), Image.BILINEAR)
        a = torch.from_numpy(np.array(img)).float() / 255.0  # 形状: (H,W,3)
        a = a.permute(2, 0, 1)  # 形状: (3,H,W)
        arrs.append(a)
    return torch.stack(arrs, dim=0)  # 形状: (S,3,H,W)


