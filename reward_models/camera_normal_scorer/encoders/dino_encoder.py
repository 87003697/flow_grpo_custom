from typing import Any, List
import torch
import torch.nn.functional as F
from PIL import Image


class DinoNormalEncoder:
    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)  # 形状: 处理器
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()  # 形状: 模型

    @torch.no_grad()
    def features_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """将一批法线 (B,3,R,R) 转为 L2 归一化特征 (B,D)。"""
        pils: List[Image.Image] = []  # 长度 B
        for i in range(normals.shape[0]):
            x = ((normals[i].clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8)  # 形状: (3,R,R)
            arr = x.permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
            pils.append(Image.fromarray(arr))  # 形状: PIL
        batch = self.processor(images=pils, return_tensors="pt")  # 形状: dict, pixel_values (B,3,h,w)
        device = next(self.model.parameters()).device  # 形状: 设备
        batch = {k: v.to(device) for k, v in batch.items()}  # 形状: 字典，各张量 (B,3,h,w)
        out = self.model(**batch)  # 形状: last_hidden_state (B,N,D)
        feats = out.last_hidden_state.mean(dim=1)  # 形状: (B,D)
        feats = F.normalize(feats, dim=-1)  # 形状: (B,D)
        return feats  # 形状: (B,D)

    @torch.no_grad()
    def feature_from_normal(self, normal: torch.Tensor) -> torch.Tensor:
        """单张法线 (3,R,R) -> 特征 (1,D)。"""
        return self.features_from_normals(normal.unsqueeze(0))  # 形状: (1,D)


