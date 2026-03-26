"""
StageLatent — 单阶段 latent 状态容器

统一管理一个流匹配阶段的所有 latent 状态。
两个 State 类（TrellisState, TrellisContrastiveState）共享此定义。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StageLatent:
    """单阶段的 latent 状态容器。

    Stage 1 (structure): z0 为密集 Tensor (B, 8, 16, 16, 16)
    Stage 2 (SLAT):      z0 为 SparseTensor (N, C)

    Attributes:
        z0:          干净 latent — rollout 产出（反归一化后）
        zt:          加噪 latent — student 训练输入
        z0_pred:     student 预测的 ẑ₀
        z0_teacher:  contrastive teacher 参考（z₀ 副本）
        coords:      Stage 1 独有 — decoder(z₀) 后提取的稀疏坐标 (N, 4)
        reg_loss:    rollout 产出的 v-reg loss
    """
    z0: Any = None
    zt: Any = None
    z0_pred: Any = None
    z0_teacher: Any = None
    coords: Any = None       # Stage 1 only: argwhere(decoder(z₀) > 0)
    reg_loss: Any = None     # rollout 产出的正则化 loss
