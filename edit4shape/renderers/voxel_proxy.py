"""
VoxelProxy: 从 FDG VAE Decoder 输出构建的伪体素对象。

用于可微 VoxelRenderer，让 dual_vertices 和 intersected 特征参与梯度优化。

梯度流: Loss → Normal → Depth → VoxelProxy → h.feats → Decoder → Flow Model

优化的特征:
    - dual_vertices [0:3]: 控制体素位置偏移
    - intersected [3:6]: 控制体素不透明度
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VoxelProxy:
    """
    从 FDG Decoder 输出构建的伪体素对象。
    
    Attributes:
        position: (N, 3) 体素位置，可微
        opacities: (N,) 体素不透明度，可微
        voxel_size: 体素大小
        batch_indices: (N,) batch 索引
    """
    position: torch.Tensor      # (N, 3)
    opacities: torch.Tensor     # (N,)
    voxel_size: float
    batch_indices: torch.Tensor  # (N,)
    
    @classmethod
    def from_fdg_decoder(
        cls,
        h_feats: torch.Tensor,    # (N, 7)
        coords: torch.Tensor,      # (N, 4) [batch_idx, x, y, z]
        resolution: int,
        voxel_margin: float = 0.5,
    ) -> "VoxelProxy":
        """
        从 FDG Decoder 输出构建 VoxelProxy。
        
        Args:
            h_feats: (N, 7) decoder 输出，[0:3] dual_vertices, [3:6] intersected
            coords: (N, 4) 稀疏坐标
            resolution: 网格分辨率
            voxel_margin: 顶点偏移范围
        
        Returns:
            VoxelProxy 对象
        """
        device = h_feats.device
        origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
        voxel_size = 1.0 / resolution
        
        # 位置: base_position + dual_vertices 偏移 (可微)
        dual_vertices = (1 + 2 * voxel_margin) * F.sigmoid(h_feats[..., 0:3]) - voxel_margin  # (N, 3)
        base_position = (coords[:, 1:4].float() + 0.5) * voxel_size + origin  # (N, 3)
        position = base_position + (dual_vertices - 0.5) * voxel_size  # (N, 3)
        
        # 不透明度: sigmoid(max(intersected_logits)) (可微)
        intersected_logits = h_feats[..., 3:6]  # (N, 3)
        max_logit = intersected_logits.max(dim=-1).values  # (N,)
        opacities = torch.sigmoid(max_logit * 10.0)  # (N,)
        
        return cls(position, opacities, voxel_size, coords[:, 0])
    
    def filter_by_batch(self, batch_idx: int) -> "VoxelProxy":
        """
        过滤指定 batch 的体素。
        
        Args:
            batch_idx: batch 索引
        
        Returns:
            只包含该 batch 的 VoxelProxy
        """
        mask = self.batch_indices == batch_idx
        return VoxelProxy(
            self.position[mask],
            self.opacities[mask],
            self.voxel_size,
            self.batch_indices[mask],
        )
