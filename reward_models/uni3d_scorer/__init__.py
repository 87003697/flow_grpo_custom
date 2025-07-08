"""
Uni3D Scorer Module

基于 Uni3D 预训练模型的 3D mesh 语义质量评分器
"""

# 导入 Uni3D 评分器（现在使用纯 PyTorch 实现，无需 pointnet2_ops）
from .uni3d_scorer import Uni3DScorer

__all__ = ['Uni3DScorer'] 