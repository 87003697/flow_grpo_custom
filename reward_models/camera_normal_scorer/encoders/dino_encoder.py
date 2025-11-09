import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig  # 形状: HF 加载器
from transformers.utils import is_flash_attn_2_available  # 形状: 检测 FA2
from PIL import Image  # 形状: 类型引用
from reward_models.camera_normal_scorer.utils.transforms import pils_to_tensor  # 形状: 工具导入


"""DINO 法线编码器封装（方式 A 导入参考实现）。

- 将项目根加入 sys.path，以 `_reference_codes.VGGTObj...` 路径导入参考仓库的实现。
- 保持对外构造签名不变：`DinoNormalEncoder(model_id, device)`。
- 兼容旧接口 `features_from_normals`，并暴露增强的 dense/match 相似度 API。
"""

# 将项目根加入 sys.path，允许使用 `_reference_codes.VGGTObj...` 绝对导入
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# 从参考仓库导入 DINO 编码器与加载函数（含 bfloat16 + FA2/SDPA 自动选择）
from _reference_codes.VGGTObj.vggt_camera_search.pipelines.normal_inference import (  # type: ignore
    DinoNormalEncoder as VGGT_DinoNormalEncoder,
    _normals_to_foreground_mask as vggt_normals_to_mask,
)


class DinoNormalEncoder:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        similarity_type: str = "match_pixel",  # 形状: 字符串
        dense_match_chunk_size: int = 16384,   # 形状: 标量
    ) -> None:
        """构造封装编码器。

        输入:
            model_id: DINO 模型 ID 或本地目录。
            device: 设备。
            similarity_type: 相似度类型（cls/dense/dense_all/match_gird2pixel/match_pixel）。
            dense_match_chunk_size: 像素级匹配分块大小。
        实现:
            - 复用参考实现的加载函数，自动设置 torch_dtype=bfloat16 与注意力实现。
        """
        # 本地通过 Transformers 加载 DINO，再交给参考实现的编码器
        mid = str(model_id)  # 形状: 字符串
        processor = AutoImageProcessor.from_pretrained(mid)  # 形状: 处理器
        cfg = AutoConfig.from_pretrained(mid)  # 形状: 配置
        model_type = str(getattr(cfg, "model_type", "")).lower()  # 形状: 字符串
        if model_type == "dinov2":
            attn_impl = "sdpa"  # 形状: 字符串
        else:
            attn_impl = "flash_attention_2" if bool(is_flash_attn_2_available()) else "sdpa"  # 形状: 字符串
        model = AutoModel.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(device).eval()  # 形状: 模型
        self._enc = VGGT_DinoNormalEncoder(processor, model)  # 形状: 编码器对象
        self._sim_type = str(similarity_type)  # 形状: 字符串
        self._chunk = int(dense_match_chunk_size)  # 形状: 标量
        self.device = device  # 形状: 设备

    # 删除 encode_*/cosine_* 直通方法，统一通过 score_pairs 接口

    # -------------------- 预编码与从令牌打分 API（可选） --------------------

    # 预编码与从令牌打分 API（外部仅需 score_pairs；以下方法为内部/扩展使用）

    @torch.no_grad()
    def cosine_score_cls_from_feats(self, feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
        """已编码 CLS 特征的分数。"""
        return self._enc.cosine_score_cls_from_feats(feats_a, feats_b)  # 形状: (B,)

    @torch.no_grad()
    def cosine_score_dense_from_tokens(self, tok_a: torch.Tensor, tok_b: torch.Tensor, mask_a=None, mask_b=None, batch_chunk: int = 32) -> torch.Tensor:
        """已编码稠密令牌（无像素搜索）的分数，始终按 batch_chunk 分块执行。"""
        out_parts: list[torch.Tensor] = []  # 形状: 列表
        B = tok_a.shape[0]  # 形状: 标量
        bs = int(batch_chunk)  # 形状: 标量
        for s in range(0, B, bs):
            e = min(B, s + bs)  # 形状: 标量
            ma_s = (None if mask_a is None else mask_a[s:e])  # 形状: (b,L) 或 None
            mb_s = (None if mask_b is None else mask_b[s:e])  # 形状: (b,L) 或 None
            out_parts.append(self._enc.cosine_score_dense_from_tokens(tok_a[s:e], tok_b[s:e], mask_a=ma_s, mask_b=mb_s))  # 形状: 追加 (b,)
        return torch.cat(out_parts, dim=0)  # 形状: (B,)

    @torch.no_grad()
    def cosine_score_dense_match_from_tokens(
        self,
        tok_a: torch.Tensor,
        tok_b: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        image_hw: tuple[int, int],
        mask_a: torch.Tensor | None = None,  # 形状: (B,H,W) 或 None
        mask_b: torch.Tensor | None = None,  # 形状: (B,H,W) 或 None
        chunk_size: int | None = None,
        batch_chunk: int,
        return_details: bool = False,
    ) -> torch.Tensor:
        """按 batch_chunk 分块，直接转调参考实现的像素级最近邻（tokens→pixels 在对方实现）。"""
        ch = int(self._chunk if (chunk_size is None) else chunk_size)
        B = tok_a.shape[0]
        bs = int(batch_chunk)
        outs: list[torch.Tensor] = []
        for s in range(0, B, bs):
            e = min(B, s + bs)
            ma_s = None if mask_a is None else mask_a[s:e]
            mb_s = None if mask_b is None else mask_b[s:e]
            out_s = self._enc.cosine_score_dense_match_from_tokens(
                tok_a[s:e], tok_b[s:e], grid_hw=grid_hw, image_hw=image_hw, mask_a=ma_s, mask_b=mb_s, chunk_size=ch, return_details=False
            )
            outs.append(out_s)
        return torch.cat(outs, dim=0)

    # -------------------- 内部通用分块编码 --------------------

    @torch.no_grad()
    def _encode_cls_in_chunks(self, normals: torch.Tensor, bs: int) -> torch.Tensor:
        B = normals.shape[0]  # 形状: 标量
        feats: list[torch.Tensor] = []  # 形状: 列表
        for s in range(0, B, int(bs)):
            e = min(B, s + int(bs))  # 形状: 标量
            feats.append(self._enc.encode_normals_cls(normals[s:e]))  # 形状: 追加 (b,D)
        return torch.cat(feats, dim=0)  # 形状: (B,D)

    @torch.no_grad()
    def _encode_tokens_in_chunks(self, normals: torch.Tensor, bs: int) -> tuple[torch.Tensor, tuple[int, int]]:
        B = normals.shape[0]  # 形状: 标量
        toks: list[torch.Tensor] = []  # 形状: 列表
        hw: tuple[int, int] | None = None  # 形状: 可选
        for s in range(0, B, int(bs)):
            e = min(B, s + int(bs))  # 形状: 标量
            t, hw_s = self._enc.encode_normals_dense_tokens(normals[s:e])  # 形状: (b,L,D), (H',W')
            toks.append(t)  # 形状: 追加
            if hw is None:
                hw = (int(hw_s[0]), int(hw_s[1]))  # 形状: (2,)
        assert hw is not None, "空输入"  # 形状: 条件
        return torch.cat(toks, dim=0), hw  # 形状: (B,L,D), (H',W')

    @torch.no_grad()
    def _encode_all_layer_tokens_in_chunks(self, normals: torch.Tensor, bs: int):
        B = normals.shape[0]  # 形状: 标量
        toks: list[torch.Tensor] = []  # 形状: 列表
        hw: tuple[int, int] | None = None  # 形状: 可选
        nlay: int | None = None  # 形状: 可选
        for s in range(0, B, int(bs)):
            e = min(B, s + int(bs))  # 形状: 标量
            t_s, hw_s, nl_s = self._enc.encode_normals_dense_tokens_all_layers(normals[s:e])  # 形状: (b,Ltotal,D), (2,), 标量
            toks.append(t_s)  # 形状: 追加
            if hw is None:
                hw = (int(hw_s[0]), int(hw_s[1]))  # 形状: (2,)
            if nlay is None:
                nlay = int(nl_s)  # 形状: 标量
        assert hw is not None and nlay is not None, "空输入"  # 形状: 条件
        return torch.cat(toks, dim=0), hw, nlay  # 形状: (B,Ltotal,D), (2,), 标量

    @torch.no_grad()
    def _encode_two_token_sets_in_chunks(
        self,
        normals_a: torch.Tensor,  # 形状: (A,3,R,R)
        normals_b: torch.Tensor,  # 形状: (B,3,R,R)
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """并行编码两组法线，单次循环处理拼接后的 batch 并在末尾切分。"""
        A = normals_a.shape[0]  # 形状: 标量
        B = normals_b.shape[0]  # 形状: 标量
        both = torch.cat([normals_a, normals_b], dim=0)  # 形状: (A+B,3,R,R)
        toks_all, hw = self._encode_tokens_in_chunks(both, bs)  # 形状: (A+B,L,D), (H',W')
        tok_a = toks_all[:A]  # 形状: (A,L,D)
        tok_b = toks_all[A:]  # 形状: (B,L,D)
        return tok_a, tok_b, hw  # 形状: (A,L,D), (B,L,D), (H',W')


    @torch.no_grad()
    def _encode_two_all_layer_token_sets_in_chunks(self, normals_a: torch.Tensor, normals_b: torch.Tensor, bs: int):
        A = normals_a.shape[0]  # 形状: 标量
        B = normals_b.shape[0]  # 形状: 标量
        both = torch.cat([normals_a, normals_b], dim=0)  # 形状: (A+B,3,R,R)
        toks_all, hw, _nl = self._encode_all_layer_tokens_in_chunks(both, bs)  # 形状: (A+B,Ltotal,D), (2,), 标量
        tok_a = toks_all[:A]  # 形状: (A,Ltotal,D)
        tok_b = toks_all[A:]  # 形状: (B,Ltotal,D)
        return tok_a, tok_b, hw  # 形状: (A,Ltotal,D), (B,Ltotal,D), (2,)

    # -------------------- 统一 pairs 打分（内部依据 sim_type 决策） --------------------

    # 已不再需要令牌级掩码下采样；像素掩码直接传递给参考实现

    @torch.no_grad()
    def score_pairs(
        self,
        group_pils: list[Image.Image],   # 形状: 长度 G
        mesh_pils: list[Image.Image],    # 形状: 长度 M
        mesh_group_indices: list[int] | torch.Tensor,  # 形状: 长度 M
        mask_mesh_px: torch.Tensor | None,  # 形状: (M,R,R) 或 None
        dino_batch_size: int,  # 形状: 标量
    ) -> torch.Tensor:
        """按 sim_type 计算每个 mesh 与其所属组图像法线的分数，返回 (M,)。"""
        assert len(mesh_pils) > 0, "空输入：mesh_pils"  # 形状: 条件
        Wm, Hm = mesh_pils[0].size  # 形状: 标量, 标量 (PIL.size=(W,H))
        normals_G = pils_to_tensor(group_pils, size_hw=(Hm, Wm), device=self.device)  # 形状: (G,3,H,W)
        normals_M = pils_to_tensor(mesh_pils,  size_hw=(Hm, Wm), device=self.device)  # 形状: (M,3,H,W)
        G = int(normals_G.shape[0])  # 形状: 标量
        M = int(normals_M.shape[0])  # 形状: 标量
        bs = int(max(1, dino_batch_size))  # 形状: 标量
        device = normals_G.device  # 形状: 设备
        # 统一转张量并强校验 mesh_group_indices
        group_idx_t = torch.as_tensor(mesh_group_indices, device=device, dtype=torch.long)  # 形状: (M,)
        assert group_idx_t.dim() == 1, "mesh_group_indices 必须为 1D 索引向量"  # 形状: 条件
        assert int(group_idx_t.shape[0]) == M, "mesh_group_indices 长度必须等于 mesh 数量"  # 形状: 条件
        assert G > 0 and M > 0, "空输入：组数与样本数必须 > 0"  # 形状: 条件
        mn = int(group_idx_t.min().item())  # 形状: 标量
        mx = int(group_idx_t.max().item())  # 形状: 标量
        assert mn >= 0 and mx < G, "mesh_group_indices 的取值需在 [0, G-1]"  # 形状: 条件

        if self._sim_type == "cls":
            fG = self._encode_cls_in_chunks(normals_G, bs)  # 形状: (G,D)
            fM = self._encode_cls_in_chunks(normals_M, bs)  # 形状: (M,D)
            fg = fG.index_select(0, group_idx_t)  # 形状: (M,D)
            return self.cosine_score_cls_from_feats(fg, fM)  # 形状: (M,)

        if self._sim_type == "dense":
            tokG, tokM, _hw = self._encode_two_token_sets_in_chunks(normals_G, normals_M, bs)  # 形状: (G,L,D), (M,L,D), (H',W')
            ta = tokG.index_select(0, group_idx_t)  # 形状: (M,L,D)
            return self.cosine_score_dense_from_tokens(ta, tokM, batch_chunk=bs)  # 形状: (M,)

        if self._sim_type == "dense_all":
            tokG, tokM, _hw = self._encode_two_all_layer_token_sets_in_chunks(normals_G, normals_M, bs)  # 形状: (G,Ltot,D), (M,Ltot,D), (H',W')
            ta = tokG.index_select(0, group_idx_t)  # 形状: (M,Ltot,D)
            return self.cosine_score_dense_from_tokens(ta, tokM, batch_chunk=bs)  # 形状: (M,)

        if self._sim_type == "match_pixel":
            tokG, tokM, hw = self._encode_two_token_sets_in_chunks(normals_G, normals_M, bs)  # 形状: (G,L,D), (M,L,D), (H',W')
            H = int(normals_M.shape[-2]); W = int(normals_M.shape[-1])  # 形状: 标量, 标量
            ta = tokG.index_select(0, group_idx_t)  # 形状: (M,L,D)
            # 像素级掩码：A 侧直接调用参考函数生成前景掩码；B 侧使用渲染掩码
            mb_px = mask_mesh_px if (mask_mesh_px is not None and mask_mesh_px.numel() > 0) else None  # 形状: (M,R,R) 或 None
            mask_group_px = vggt_normals_to_mask(normals_G)  # 形状: (G,R,R)
            ma_px = mask_group_px.index_select(0, group_idx_t)  # 形状: (M,R,R)
            return self.cosine_score_dense_match_from_tokens(
                ta, tokM,
                grid_hw=hw,
                image_hw=(H, W),
                mask_a=ma_px,
                mask_b=mb_px,
                chunk_size=self._chunk,
                batch_chunk=bs,
                return_details=False,
            )

        raise ValueError(f"不支持的相似度类型: {self._sim_type}")


