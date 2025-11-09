# -*- coding: utf-8 -*-
from typing import List
import torch
from PIL import Image
import torchvision.transforms as T

@torch.no_grad()
def to_tensor_from_normal_pil(normal_pil: Image.Image, R: int, device: torch.device) -> torch.Tensor:
    # 复用批量 API，单张输入后取索引 0
    if normal_pil.mode != "RGB":
        normal_pil = normal_pil.convert("RGB")  # 形状: PIL(H,W,3)
    x = pils_to_tensor([normal_pil], (int(R), int(R)), device)  # 形状: (1,3,R,R)
    return x[0]  # 形状: (3,R,R)

@torch.no_grad()
def to_tensor_from_rgb_pil(rgb_pil: Image.Image, R: int, device: torch.device) -> torch.Tensor:
    # 复用批量 API，单张输入后取索引 0；若为 RGBA，先做白底合成
    if rgb_pil.mode == "RGBA":
        bg = Image.new('RGBA', rgb_pil.size, (255, 255, 255, 255))  # 形状: PIL(H,W,4)
        rgb_pil = Image.alpha_composite(bg, rgb_pil).convert("RGB")  # 形状: PIL(H,W,3)
    elif rgb_pil.mode != "RGB":
        rgb_pil = rgb_pil.convert("RGB")  # 形状: PIL(H,W,3)
    x = pils_to_tensor([rgb_pil], (int(R), int(R)), device)  # 形状: (1,3,R,R)
    return x[0]  # 形状: (3,R,R)

@torch.no_grad()
def normal_tensor_to_pil(n: torch.Tensor) -> Image.Image:
    # n: (3,R,R) in [-1,1]
    n01 = (n + 1.0) * 0.5  # 形状: (3,R,R)
    n255 = (n01.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # 形状: (3,R,R)
    arr = n255.permute(1, 2, 0).detach().cpu().numpy()  # 形状: (R,R,3)
    pil = Image.fromarray(arr, mode="RGB")  # 形状: PIL(R,R,3)
    return pil  # 形状: PIL(R,R,3)

@torch.no_grad()
def map_to_01_from_m11(normals: torch.Tensor) -> torch.Tensor:
    # normals: (B,3,R,R) in [-1,1]
    return ((normals + 1.0) * 0.5).clamp(0.0, 1.0)  # 形状: (B,3,R,R)

@torch.no_grad()
def pils_to_tensor(pils: List[Image.Image], size_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    将 List[PIL] 统一 resize 到 size_hw 后转换为法线张量:
      - 输入: List[PIL], size_hw=(H,W)
      - 输出: (B,3,H,W) in [-1,1]
    """
    Ht, Wt = int(size_hw[0]), int(size_hw[1])  # 形状: 标量, 标量
    B = len(pils)  # 形状: 标量
    if B == 0:
        return torch.zeros(0, 3, Ht, Wt, device=device)  # 形状: (0,3,H,W)
    tfm = T.Compose([
        T.Resize((Ht, Wt), interpolation=T.InterpolationMode.BICUBIC),  # 形状: -> PIL(H,W)
        T.ToTensor(),  # 形状: -> (3,H,W) in [0,1]
    ])
    xs = []
    for img in pils:
        x01 = tfm(img)  # 形状: (3,H,W)
        xs.append(x01)  # 形状: 追加
    x01_b = torch.stack(xs, dim=0).to(device)  # 形状: (B,3,H,W)
    x11_b = (x01_b * 2.0) - 1.0  # 形状: (B,3,H,W)
    return x11_b  # 形状: (B,3,H,W)


