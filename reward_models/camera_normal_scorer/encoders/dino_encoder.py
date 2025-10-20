import os
import sys
import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig  # 形状: HF 加载器
from transformers.utils import is_flash_attn_2_available  # 形状: 检测 FA2


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
            similarity_type: 相似度类型（cls/dense/match_gird2pixel/match_pixel）。
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

    @torch.no_grad()
    def features_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """兼容旧接口：返回 CLS 全局特征 (B,D)。"""
        return self._enc.encode_normals_cls(normals)  # 形状: (B,D)

    # 删除 encode_*/cosine_* 直通方法，统一通过 score 接口

    def set_similarity_type(self, similarity_type: str) -> None:
        self._sim_type = str(similarity_type)  # 形状: 字符串

    @torch.no_grad()
    def score(self, normals_a: torch.Tensor, normals_b: torch.Tensor, mask_a=None, mask_b=None) -> torch.Tensor:
        """统一对外的打分接口，返回 (B,) 分数 in [0,1]。"""
        if self._sim_type == "cls":
            return self._enc.cosine_score_cls(normals_a, normals_b)  # 形状: (B,)
        if self._sim_type == "dense":
            return self._enc.cosine_score_dense(normals_a, normals_b)  # 形状: (B,)
        if self._sim_type == "match_pixel":
            return self._enc.cosine_score_dense_match(normals_a, normals_b, chunk_size=self._chunk, return_details=False, mask_a=mask_a, mask_b=mask_b)  # 形状: (B,)
        if self._sim_type == "match_gird2pixel":
            s, _ = self._enc.cosine_score_match_grid2pix(normals_a, normals_b)  # 形状: (B,), 列表
            return s  # 形状: (B,)
        # 兜底：退回 cls
        return self._enc.cosine_score_cls(normals_a, normals_b)  # 形状: (B,)


