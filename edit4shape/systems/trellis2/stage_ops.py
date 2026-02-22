"""
Trellis2 StageOps 具体实现 — Shape / Tex 阶段的计算操作。

ABC (StageOps) 和异常 (StageSkipError) 定义在 edit4shape.systems.utils.stage_ops，
本文件提供 Trellis2 特定的实现。

实现：
  ShapeOps        — Shape 阶段（含 dense_sampling）
  TexOps          — Tex-only 阶段（含 shape_frozen_prepare）
  TexOpsFromShape — Tex-from-Shape 阶段（跳过 shape_frozen_prepare）

使用方式：
  # 同步模板
  three_phase_step(ShapeOps(), state, system, ...)

  # 异步模板（后续扩展）
  PendingMicroBatch(ops=ShapeOps(), ...)

设计原则：
  - 同一个 ShapeOps 在 shape-only / shape+tex / cascade 中一行不改
  - 清理策略由 Slot（编排层）决定，不由 Ops 决定
  - Ops 只回答 "这个阶段自身的计算是什么？"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from edit4shape.generators.trellis2.rollout import RolloutTracker

# ABC 和异常从 utils 导入（模型无关的抽象层）
from edit4shape.systems.utils.stage_ops import StageOps, StageSkipError  # noqa: F401 — re-export


# =====================================================================
# 具体实现 — Shape
# =====================================================================

class ShapeOps(StageOps):
    """
    Shape 阶段 Ops — 包装 shape_autograd.py 中的现有函数。

    计算链：
      dense_sampling → rollout_shape → decode_and_render_normal → guidance → VJP

    使用 lazy import 避免循环导入（shape_autograd.py 可能反向导入本模块）。
    """

    def get_model(self, system):
        return system.shape.model

    def get_stage_name(self) -> str:
        return "shape"

    def get_seed_offset(self) -> int:
        return 0

    def get_reg_weight(self, system) -> float:
        return system.cfg.shape.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.shape.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.shape.guidance

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Dense Sampling → 填充 state.coords。"""
        from edit4shape.systems.trellis2.forward import dense_sampling_no_grad
        dense_sampling_no_grad(state, system)

    def rollout(self, state, system, seed) -> RolloutTracker:
        """Phase 1: Shape rollout → proxy chain + tracker。"""
        from edit4shape.systems.trellis2.phases import shape_phase1_rollout
        return shape_phase1_rollout(state, system, seed)

    def decode_render(self, state, system) -> torch.Tensor:
        """Phase 2a: decode + render Normal → comp_rgb。"""
        from edit4shape.systems.trellis2.phases import shape_phase2a_decode_render
        return shape_phase2a_decode_render(state, system)

    def vjp_loop(self, state, system, tracker) -> Dict[str, Any]:
        """Phase 3: Shape VJP loop → θ_shape.grad 累积。"""
        from edit4shape.systems.trellis2.phases import shape_phase3_rollout_grad_backward
        return shape_phase3_rollout_grad_backward(state, system, tracker)


# =====================================================================
# 具体实现 — Tex (standalone)
# =====================================================================

class TexOps(StageOps):
    """
    Tex-only 阶段 Ops — 含 Phase 0 (shape_frozen_prepare)。

    用于 tex-only 训练模式，Phase 0 执行冻结的 Shape forward + 全量 detach，
    然后 Tex rollout 使用 shape 产物作为条件。

    计算链：
      shape_frozen_prepare → rollout_tex → decode_and_render_pbr → guidance → VJP
    """

    def get_model(self, system):
        return system.tex.model

    def get_stage_name(self) -> str:
        return "tex"

    def get_seed_offset(self) -> int:
        return 1000

    def get_reg_weight(self, system) -> float:
        return system.cfg.tex.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.tex.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.tex.guidance

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Shape 冻结前置（no_grad shape forward + detach）。"""
        from edit4shape.systems.trellis2.phases import shape_frozen_prepare_no_grad
        shape_frozen_prepare_no_grad(state, system, global_step)

    def rollout(self, state, system, seed) -> RolloutTracker:
        """Phase 1: Tex rollout → proxy chain + tracker。"""
        from edit4shape.systems.trellis2.phases import tex_phase1_rollout
        return tex_phase1_rollout(state, system, seed)

    def decode_render(self, state, system) -> torch.Tensor:
        """Phase 2a: decode + render PBR → comp_rgb。"""
        from edit4shape.systems.trellis2.phases import tex_phase2a_decode_render
        return tex_phase2a_decode_render(state, system)

    def vjp_loop(self, state, system, tracker) -> Dict[str, Any]:
        """Phase 3: Tex VJP loop → θ_tex.grad 累积。"""
        from edit4shape.systems.trellis2.phases import tex_phase3_rollout_grad_backward
        return tex_phase3_rollout_grad_backward(state, system, tracker)


# =====================================================================
# 具体实现 — Tex (from Shape)
# =====================================================================

class TexOpsFromShape(TexOps):
    """
    Tex-from-Shape 阶段 Ops — 用于双阶段（shape+tex）训练模式。

    前置条件：
      state 中已有 detach 过的 Shape 产物
      (coords, shape_slat, shape_slat_norm, subs, meshes)。

    与 TexOps 的差异：
    - pre_rollout 为 no-op（Shape 产物由上游 Shape 阶段 + detach 提供）
    - decode_render 增加 meshes 可用性检查（Shape P2 OOM 时 meshes 为 None）
    """

    def pre_rollout(self, state, system, global_step) -> None:
        """No-op: Shape 产物由上游 Shape 阶段 + detach 转接提供。"""
        pass

    def decode_render(self, state, system) -> torch.Tensor:
        """
        Tex decode+render，带 meshes 可用性检查。

        如果上游 Shape P2a OOM 导致 meshes 为 None，
        抛出 StageSkipError 使模板跳过 P2/P3。
        """
        if state.features.meshes is None:
            raise StageSkipError(
                "meshes 不可用（Shape P2a OOM），跳过 Tex P2a"
            )
        return super().decode_render(state, system)
