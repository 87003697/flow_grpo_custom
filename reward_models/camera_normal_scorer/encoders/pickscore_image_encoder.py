# -*- coding: utf-8 -*-
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image  # 形状: 类型引用


class PickScoreImageEncoder:
    def __init__(
        self,
        device: torch.device,  # 形状: 标量
        model_id: str = "yuvalkirstain/PickScore_v1",  # 形状: 字符串
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # 形状: 字符串
        dtype: torch.dtype = torch.bfloat16,  # 形状: 标量
        similarity_type: str = "cls",  # 形状: 字符串（仅支持 cls）
        **kwargs,
    ) -> None:
        self.device = device  # 形状: 标量
        self.dtype = dtype  # 形状: 标量
        # 固定对比方式为 'cls'（logit comparison 使用全局特征余弦）
        self.processor = AutoProcessor.from_pretrained(processor_id)  # 形状: 处理器
        self.model = AutoModel.from_pretrained(model_id).to(device).to(dtype=dtype).eval()  # 形状: 模型
        self._sim_type = "cls"  # 形状: 字符串

    @torch.no_grad()
    def cosine_score_cls_from_feats(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        """对已编码的全局特征计算余弦相似度，并线性映射到 [0,1]。
        
        输入:
            feats_a: (B,D)
            feats_b: (B,D)
        输出:
            scores: (B,) in [0,1]
        """
        assert feats_a.shape == feats_b.shape, "特征形状需一致"  # 形状: 条件
        na = feats_a.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)  # 形状: (B,1)
        nb = feats_b.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)  # 形状: (B,1)
        fa = feats_a / na  # 形状: (B,D)
        fb = feats_b / nb  # 形状: (B,D)
        scores = (fa * fb).sum(dim=-1)  # 形状: (B,)
        scores = (scores + 1.0) * 0.5  # 形状: (B,)
        return scores  # 形状: (B,)
    
    @torch.no_grad()
    def _encode_cls_from_pils(self, pils: list[Image.Image], bs: int) -> torch.Tensor:
        """按批量分块对 List[PIL] 编码，返回单位化全局特征。
        
        输入:
            pils: 长度 B 的 PIL 列表
            bs: 分块批大小（>0）
        输出:
            feats: (B,D) 单位化特征
        """
        B = int(len(pils))  # 形状: 标量
        bs = max(1, int(bs))  # 形状: 标量
        out_parts: list[torch.Tensor] = []  # 形状: 列表
        for s in range(0, B, bs):
            e = min(B, s + bs)  # 形状: 标量
            batch_pil = pils[s:e]  # 形状: 长度 b
            inputs = self.processor(images=batch_pil, return_tensors="pt", padding=True)  # 形状: 字典
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 形状: 同 keys
            f = self.model.get_image_features(**inputs)  # 形状: (b,D)
            f = f / f.norm(p=2, dim=-1, keepdim=True)  # 形状: (b,D)
            out_parts.append(f)  # 形状: 追加
        return torch.cat(out_parts, dim=0) if len(out_parts) > 0 else torch.empty(0, device=self.device)  # 形状: (B,D)


    @torch.no_grad()
    def score_pairs(
        self,
        group_pils: list[Image.Image],  # 形状: 长度 G
        mesh_pils: list[Image.Image],   # 形状: 长度 M
        mesh_group_indices: list[int] | torch.Tensor,  # 形状: (M,)
        batch_size: int = 32,  # 形状: 标量
        **kwargs,
    ) -> torch.Tensor:
        G = int(len(group_pils))  # 形状: 标量
        M = int(len(mesh_pils))  # 形状: 标量
        device = self.device  # 形状: 设备
        group_idx_t = torch.as_tensor(mesh_group_indices, device=device, dtype=torch.long)  # 形状: (M,)
        assert group_idx_t.dim() == 1 and int(group_idx_t.shape[0]) == M  # 形状: 条件

        if self._sim_type == "cls":
            fG = self._encode_cls_from_pils(group_pils, batch_size)  # 形状: (G,D)
            fM = self._encode_cls_from_pils(mesh_pils, batch_size)  # 形状: (M,D)
            fG_sel = fG.index_select(0, group_idx_t.to(fG.device))  # 形状: (M,D)
            scores = self.cosine_score_cls_from_feats(fG_sel, fM)  # 形状: (M,)
            return scores  # 形状: (M,)
        else:
            raise ValueError(f"不支持的相似度类型: {self._sim_type}")
        
        
