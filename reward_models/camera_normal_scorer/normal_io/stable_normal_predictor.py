import os
from typing import List
import torch
import numpy as np


class StableNormalPredictor(torch.nn.Module):
    def __init__(self, hub_repo: str, hub_entry: str, yoso_weight_path: str, device: torch.device) -> None:
        super().__init__()
        import torch as _torch  # 形状: 模块
        local_cache_dir = os.path.dirname(os.path.abspath(yoso_weight_path))  # 形状: 标量
        yoso_version = os.path.basename(os.path.abspath(yoso_weight_path))  # 形状: 标量
        device_str = "cuda:0" if device.type == "cuda" else "cpu"  # 形状: 标量
        self.predictor = _torch.hub.load(
            hub_repo,
            hub_entry,
            trust_repo=True,
            local_cache_dir=local_cache_dir,
            device=device_str,
            yoso_version=yoso_version,
        )  # 形状: 预测器
        self.device = device  # 形状: 设备

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """将图像批次 (S,3,H,W) 映射为法线 (S,3,H,W)，值域 ∈[-1,1]。
        输入 images 预期为 [0,1]，与 vggt 的预处理保持一致。
        """
        from PIL import Image  # 形状: 模块
        S, _, H, W = images.shape  # 形状: 标量, 标量, 标量, 标量
        imgs_uint8 = (images.clamp(0, 1) * 255.0).round().to(torch.uint8)  # 形状: (S,3,H,W)
        normals_list: List[torch.Tensor] = []  # 长度 S
        for i in range(S):  # 形状: 循环
            arr = imgs_uint8[i].permute(1, 2, 0).cpu().numpy()  # 形状: (H,W,3)
            pil = Image.fromarray(arr)  # 形状: PIL.Image
            out_img = self.predictor(pil, data_type="object")  # 形状: PIL.Image(法线可视图，背景按 object 蒙版置白)
            out_arr = np.array(out_img)  # 形状: (H,W,3)
            out_tensor = torch.from_numpy(out_arr).to(torch.float32).permute(2, 0, 1)  # 形状: (3,H,W)
            n = (out_tensor / 255.0) * 2.0 - 1.0  # 形状: (3,H,W)
            normals_list.append(n)
        normals = torch.stack(normals_list, dim=0)  # 形状: (S,3,H,W)
        return normals.clamp(-1.0, 1.0)  # 形状: (S,3,H,W)


def create_predictor(model_dir: str, device: torch.device) -> torch.nn.Module:
    """创建 Stable Normal 预测器（torch.hub 方式）。
    - 使用 repo: "hugoycj/StableNormal"，entry: "StableNormal_turbo"。
    - 通过 yoso_weight_path 指向本地 diffusers 权重目录，脱网加载。
    """
    hub_repo = "hugoycj/StableNormal"  # 形状: 标量
    hub_entry = "StableNormal_turbo"  # 形状: 标量
    yoso_weight_path = os.path.abspath(model_dir)  # 形状: 标量
    return StableNormalPredictor(hub_repo, hub_entry, yoso_weight_path, device)


