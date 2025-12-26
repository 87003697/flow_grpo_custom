"""
Guidance 工具函数。

提供图像格式转换等通用工具。
"""

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    将 (C,H,W) Tensor 转换为 Base64 字符串 (PNG 格式)。
    
    Args:
        tensor: 图像张量 (C,H,W)，float32 [0,1]
    
    Returns:
        str: Base64 编码的 PNG 图像
    """
    img_np = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
    if img_np.shape[-1] == 1:
        img_np = img_np[..., 0]  # (H,W)
    img = Image.fromarray(img_np)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str: str, device: torch.device) -> torch.Tensor:
    """
    将 Base64 字符串转换为 (C,H,W) Tensor (float32 [0,1])。
    
    Args:
        b64_str: Base64 编码的图像
        device: 目标设备
    
    Returns:
        torch.Tensor: 图像张量 (C,H,W)
    """
    img_data = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # (H,W,C)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)  # (C,H,W)
    return tensor

