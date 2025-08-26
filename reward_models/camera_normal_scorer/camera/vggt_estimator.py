from typing import Tuple
import os
import sys
import torch

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)

from .model_factory import create_vggt_camera_search_model
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # 形状: 可调用


class VGGTSearchEstimator:
    def __init__(self, device: torch.device, camera_param_dim: int = 9, img_size: int = 518, ckpt: str | None = None, embed_dim: int = 1024) -> None:
        """VGGT Camera-Search 相机估计器封装。

        功能:
            - 创建对齐训练配置的 VGGT 模型，仅启用 camera head，关闭 depth/point 分支以节省显存。
            - 加载 LoRA/非 LoRA 的权重，宽松匹配键名。

        输入:
            device: 目标设备。
            camera_param_dim: 姿态编码维度（默认 9）。
            img_size: 模型期望输入尺寸（默认 518）。
            ckpt: checkpoint 路径或目录（必须提供）。
            embed_dim: 基础 ViT embed 维度。
        参考:
            - 模型工厂: `_reference_codes/VGGTObj/training/models/model_factory.py`
            - 姿态反解: 本文件 `estimate`
        """
        self.model = create_vggt_camera_search_model(
            device=device,
            camera_param_dim=int(camera_param_dim),
            img_size=int(img_size),
            ckpt=ckpt,
            embed_dim=int(embed_dim),
        )
        self.device = device

    @torch.no_grad()
    def estimate(self, images_batched: torch.Tensor, support: torch.Tensor, image_hw: Tuple[int, int]):
        """对 (K,S,3,H,W) 与 (K,S-1,D) 进行相机估计，输出 (K,4,4),(K,3,3)。

        参考: 本文件 `estimate`
        """
        def extrinsics34_to44(extri_3x4: torch.Tensor) -> torch.Tensor:
            B = extri_3x4.shape[0]  # 形状: 标量
            bottom = torch.tensor([0, 0, 0, 1], dtype=extri_3x4.dtype, device=extri_3x4.device).view(1, 1, 4)  # 形状: (1,1,4)
            extri_4x4 = torch.cat([extri_3x4, bottom.expand(B, -1, -1)], dim=-2)  # 形状: (B,4,4)
            return extri_4x4  # 形状: (B,4,4)

        H, W = image_hw  # 形状: 标量, 标量
        preds = self.model(images_batched, support)  # 形状: dict，含 'pose_enc'
        pose_enc_q = preds["pose_enc"][:, -1:, :]  # 形状: (B,1,D)
        extri_b1, intr_b1 = pose_encoding_to_extri_intri(pose_enc_q, (H, W))  # 形状: (B,1,3,4),(B,1,3,3)
        extri_3x4 = extri_b1[:, 0]  # 形状: (B,3,4)
        intr_3x3 = intr_b1[:, 0]   # 形状: (B,3,3)
        extri_4x4 = extrinsics34_to44(extri_3x4)  # 形状: (B,4,4)
        return extri_4x4, intr_3x3  # 形状: (B,4,4),(B,3,3)


