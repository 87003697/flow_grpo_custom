"""
Trellis2 StageLatent — 各阶段的 latent 状态容器。

按阶段分组管理 latent、中间产物和正则化 loss，
与 Trellis v1 的 StageLatent 设计对齐。

层次：
    Trellis2State
    ├── dense  (DenseStageLatent)   — Dense Sampling 产物
    ├── shape  (ShapeStageLatent)   — Shape 阶段 latent + decode 中间产物
    └── tex    (TexStageLatent)     — Tex 阶段 latent
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DenseStageLatent:
    """Dense Sampling 阶段状态容器。

    Attributes:
        coords: (N, 4) 稀疏坐标。第 0 列为 batch 索引，后 3 列为 (x, y, z)。
    """
    coords: Any = None


@dataclass
class ShapeStageLatent:
    """Shape 阶段状态容器。

    Attributes:
        z0:         SparseTensor — rollout 产出的反归一化 latent (denormalized)
        z0_norm:    SparseTensor — 归一化 latent (normalized, detached)，
                    用作 tex 阶段的 concat_cond
        subs:       List[SparseTensor] — Shape 解码中间结果（用于 tex decode）
        meshes:     List[Mesh] — Shape 解码输出的 mesh（用于 tex 渲染）
        reg_loss:   正则化 loss（用于反向传播）
        reg_metric: 正则化 metric（用于日志记录）
    """
    z0: Any = None
    z0_norm: Any = None
    subs: Any = None
    meshes: Any = None
    reg_loss: Any = None
    reg_metric: Any = None


@dataclass
class TexStageLatent:
    """Tex 阶段状态容器。

    Attributes:
        z0:         SparseTensor — rollout 产出的反归一化 latent (denormalized)
        z0_norm:    SparseTensor — 归一化 latent (normalized, detached)
        reg_loss:   正则化 loss（用于反向传播）
        reg_metric: 正则化 metric（用于日志记录）
    """
    z0: Any = None
    z0_norm: Any = None
    reg_loss: Any = None
    reg_metric: Any = None
