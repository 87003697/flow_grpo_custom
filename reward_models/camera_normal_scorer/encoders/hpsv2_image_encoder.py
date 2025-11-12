# -*- coding: utf-8 -*-
import torch
from PIL import Image
from typing import List

from open_clip import create_model_and_transforms


class HPSV2ImageEncoder:
    def __init__(
        self,
        device: torch.device,
        ckpt_path: str,
        dtype: torch.dtype = torch.bfloat16,
        similarity_type: str = "cls",
        **kwargs,
    ) -> None:
        self.device = device  # 形状: 标量
        self.dtype = dtype  # 形状: 标量
        # 仅支持全局 CLS 特征余弦
        self._sim_type = "cls"
        # 初始化 HPSv2 主干与预处理
        self.model, _, self.preprocess_val = create_model_and_transforms(
            model_name='ViT-H-14',
            pretrained=None,
            device=device,
        )  # 形状: (model, _, preprocess_val)
        # 加载权重
        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model = self.model.to(device).to(dtype).eval()

    @torch.no_grad()
    def cosine_score_cls_from_feats(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        # 输入: feats_a (B,D), feats_b (B,D)
        na = feats_a.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)  # 形状: (B,1)
        nb = feats_b.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)  # 形状: (B,1)
        fa = feats_a / na  # 形状: (B,D)
        fb = feats_b / nb  # 形状: (B,D)
        cos = (fa * fb).sum(dim=-1)  # 形状: (B,)
        return (cos + 1.0) * 0.5  # 形状: (B,)

    @torch.no_grad()
    def _encode_cls_from_pils(self, pils: List[Image.Image], bs: int) -> torch.Tensor:
        # 输出: 单位化特征 (B,D)
        B = int(len(pils))  # 形状: 标量
        bs = max(1, int(bs))  # 形状: 标量
        outputs = []  # 形状: 列表[(b,D)]
        for s in range(0, B, bs):
            e = min(B, s + bs)  # 形状: 标量
            batch_pils = pils[s:e]  # 形状: 长度 b
            imgs = [self.preprocess_val(p).unsqueeze(0) for p in batch_pils]  # 形状: 列表[(1,3,H,W)]
            imgs = torch.cat(imgs, dim=0).to(self.device)  # 形状: (b,3,H,W)
            imgs = imgs.to(dtype=self.dtype)  # 形状: (b,3,H,W) 与权重 dtype 对齐
            feats = self.model.encode_image(imgs)  # 形状: (b,D)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)  # 形状: (b,D)
            outputs.append(feats)
        return torch.cat(outputs, dim=0) if len(outputs) > 0 else torch.empty(0, device=self.device)  # 形状: (B,D)

    @torch.no_grad()
    def score_pairs(
        self,
        group_pils: List[Image.Image],  # 形状: 长度 G
        mesh_pils: List[Image.Image],   # 形状: 长度 M
        mesh_group_indices: List[int] | torch.Tensor,  # 形状: (M,)
        mask_mesh_px: torch.Tensor | None,  # 未使用
        encoding_batch_size: int,  # 形状: 标量
    ) -> torch.Tensor:
        # 统一 CLS 相似度路径
        G = int(len(group_pils))  # 形状: 标量
        M = int(len(mesh_pils))  # 形状: 标量
        group_idx = torch.as_tensor(mesh_group_indices, device=self.device, dtype=torch.long)  # 形状: (M,)
        assert group_idx.dim() == 1 and int(group_idx.shape[0]) == M

        if self._sim_type == "cls":
            fG = self._encode_cls_from_pils(group_pils, encoding_batch_size)  # 形状: (G,D)
            fM = self._encode_cls_from_pils(mesh_pils, encoding_batch_size)  # 形状: (M,D)
            fG_sel = fG.index_select(0, group_idx.to(fG.device))  # 形状: (M,D)
            scores = self.cosine_score_cls_from_feats(fG_sel, fM)  # 形状: (M,)
            return scores
        else:
            raise ValueError(f"不支持的相似度类型: {self._sim_type}")


