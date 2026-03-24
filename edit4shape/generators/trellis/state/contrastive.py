"""
TrellisContrastiveState - Contrastive FlowEdit 训练的状态容器

存储对比学习流程中的所有中间状态，支持多阶段 (Stage 1: structure, Stage 2: SLAT):
- StageLatent: 每阶段的 latent 状态 (z0 / zt / z0_pred / z0_teacher / reg_loss)
- 稀疏结构坐标 (BaseState.coords)
- 相机参数 (cameras)
- 条件信息 (views_conditioned): 输入图像 + DINOv2/CLIP 嵌入（继承自 BaseState）
- 生成结果 (views_generated): Teacher z₀ 渲染 + DINOv2 condition
- 编辑结果 (views_edited): FlowEdit 编辑 + DINOv2 condition
- Trackers (velocity + guidance)
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from edit4shape.systems.base import BaseState
from .stage_latent import StageLatent


@dataclass
class TrellisContrastiveState(BaseState):
    """
    Contrastive FlowEdit 训练的状态容器。

    独立于 TrellisState，不继承 TrellisState（避免字段冲突）。
    继承 BaseState 获取 cameras / views_conditioned / extract_embeddings。

    使用 StageLatent 统一管理两个阶段的 latent 状态：
        stage1: structure flow model latent — Tensor (B, 8, 16, 16, 16)
        stage2: SLAT flow model latent — SparseTensor (N, C)

    训练流程挂载约定:
        P0  → views_conditioned (attach_batch), cameras (attach_batch)
        P1  → stage2.z0 = teacher z₀, stage2.z0_teacher = 同引用
              BaseState.coords = 稀疏结构坐标 (由 dense sampling 产出)
        P3  → z0_hat_norm 保留为局部变量
        P4a → views_generated.image_tensor (渲染 teacher z₀)
        P4b → views_edited.image_tensor (FlowEdit 编辑结果)
              trackers.guidance = FlowEdit 中间步 trackers
        P4c → views_generated.image_condition / views_edited.image_condition (DINOv2)
        P5  → trackers.velocity = VelocityTracker
    """

    @dataclass
    class ViewsGenerated:
        """生成视角缓存（覆盖基类，增加 DINOv2 编码）。

        - image_tensor:    (B, V, H, W, C) 渲染结果
        - image_condition: (B, S, C) DINOv2 patch token 编码
        """
        image_tensor: Any = None
        image_condition: Any = None

    @dataclass
    class ViewsEdited:
        """编辑结果缓存（覆盖基类，增加 DINOv2 编码）。

        - image_tensor:    (B, V, H, W, C) 编辑结果
        - image_condition: (B, S, C) DINOv2 patch token 编码
        """
        image_tensor: Any = None
        image_condition: Any = None

    @dataclass
    class Trackers:
        """Tracker 容器。"""
        velocity: Any = None   # VelocityTracker
        guidance: Any = None   # List[StateTracker]（FlowEdit 中间步）

    # ============== batch key 映射 ==============
    _CAMERA_KEYS: ClassVar[List[str]] = [
        "c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions",
    ]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]

    # ============== 子状态容器 ==============
    stage1: StageLatent = field(default_factory=StageLatent)
    stage2: StageLatent = field(default_factory=StageLatent)
    trackers: Trackers = field(default_factory=Trackers)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)
    dense_fallback: bool = False

    # ============================================================
    # attach_batch
    # ============================================================

    def attach_batch(
        self, batch: Dict[str, Any], pipeline: Any = None,
    ) -> "TrellisContrastiveState":
        """从数据批次中提取并挂载条件、相机等信息。"""
        # 1. views_conditioned
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])

        if "image_pils" in batch and pipeline is not None:
            cond = pipeline.prepare_image_conditions(batch["image_pils"])
            self.views_conditioned.cond_embed = cond["cond"]
            self.views_conditioned.uncond_embed = cond.get(
                "neg_cond", torch.zeros_like(cond["cond"]),
            )

        # 2. Guidance 数据
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]

        # 3. 相机参数
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])

        return self

    # ============================================================
    # visual_packs — 供 VisualIO.save_contrastive_train 使用
    # ============================================================

    def visual_packs(self) -> List[Dict[str, Any]]:
        """
        返回可视化数据包列表。

        每个 pack: {"suffix", "render_tensor", "edit_tensor", "trackers"}
        """
        packs = []
        if self.views_generated.image_tensor is not None:
            packs.append({
                "suffix": "_contrastive",
                "render_tensor": self.views_generated.image_tensor,   # (B,V,H,W,C)
                "edit_tensor": self.views_edited.image_tensor,       # (B,V,C,H,W)
                "trackers": self.trackers.guidance,
            })
        return packs
