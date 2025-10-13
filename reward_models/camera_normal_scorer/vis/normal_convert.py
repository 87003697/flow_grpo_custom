from typing import Any
import torch
from PIL import Image
import torchvision.transforms as T


def tensor_from_normal_pil(normal_pil: Image.Image, R: int, device: torch.device) -> torch.Tensor:
    """将 normal PIL 变换为 (3,R,R) 且值域在 [-1,1]。

    输入:
        normal_pil: PIL 图像
        R: 目标分辨率
        device: 设备
    输出:
        (3,R,R) [-1,1]
    """
    transform = T.Compose([
        T.Resize((int(R), int(R)), interpolation=T.InterpolationMode.BICUBIC),  # 形状: -> PIL(R,R)
        T.ToTensor(),  # 形状: -> (3,R,R) in [0,1]
    ])
    x01 = transform(normal_pil).to(device)  # 形状: (3,R,R)
    x11 = (x01 * 2.0) - 1.0  # 形状: (3,R,R)
    return x11  # 形状: (3,R,R)


def normal_tensor_to_pil(n: torch.Tensor) -> Image.Image:
    """将法线张量 [-1,1] 的 (3,R,R) 转为 RGB PIL。

    输入:
        n: (3,R,R) in [-1,1]
    输出:
        PIL(R,R,3)
    """
    n01 = (n + 1.0) * 0.5  # 形状: (3,R,R)
    n255 = (n01.clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # 形状: (3,R,R)
    arr = n255.permute(1, 2, 0).detach().cpu().numpy()  # 形状: (R,R,3)
    pil = Image.fromarray(arr, mode="RGB")  # 形状: PIL(R,R,3)
    return pil  # 形状: PIL(R,R,3)


