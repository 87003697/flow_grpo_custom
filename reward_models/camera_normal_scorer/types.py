from typing import Protocol, Tuple, Dict, Any
import torch


class NormalEncoderProtocol(Protocol):
    def features_from_normals(self, normals: torch.Tensor) -> torch.Tensor: ...  # 形状: (B,3,R,R) -> (B,D)
    def feature_from_normal(self, normal: torch.Tensor) -> torch.Tensor: ...  # 形状: (3,R,R) -> (1,D)


class CameraEstimatorProtocol(Protocol):
    def preprocess_image(self, image_path: str) -> torch.Tensor: ...  # 形状: () -> (1,3,H,W)
    def estimate(self, images_batched: torch.Tensor, support: torch.Tensor, image_hw: Tuple[int, int]): ...  # 形状: (K,1,3,H,W),(K,0,12),(H,W) -> (K,4,4),(K,3,3)


class RendererProtocol(Protocol):
    def render(self, mesh_ex, extri_4x4: torch.Tensor, intr_3x3: torch.Tensor, return_types): ...  # 返回包含 normal/mask 的 dict


