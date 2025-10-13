from typing import Tuple, Optional
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
    def __init__(self, device: torch.device, camera_param_dim: int = 9, img_size: int = 518, ckpt: Optional[str] = None, embed_dim: int = 1024, model: Optional[torch.nn.Module] = None) -> None:
        """VGGT Camera-Search 相机估计器封装（支持外部注入模型以最小化本地 glue）。

        功能:
            - 若未提供 `model`，则通过本地工厂创建仅启用 camera head 的 VGGT 模型并加载权重。
            - 若提供 `model`，直接复用（需兼容 forward(images_batched, support) 接口）。

        输入:
            device: 目标设备。
            camera_param_dim: 姿态编码维度（默认 9）。
            img_size: 模型期望输入尺寸（默认 518）。
            ckpt: checkpoint 路径或目录（用于本地工厂创建）。
            embed_dim: 基础 ViT embed 维度。
            model: 预先构建好的 VGGT 相机搜索模型（可选）。
        """
        self.model = model if model is not None else create_vggt_camera_search_model(
            device=device,  # 形状: 设备
            camera_param_dim=int(camera_param_dim),  # 形状: 标量
            img_size=int(img_size),  # 形状: 标量
            ckpt=ckpt,  # 形状: 路径或空
            embed_dim=int(embed_dim),  # 形状: 标量
        )  # 形状: 模型
        self.device = device  # 形状: 设备

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


