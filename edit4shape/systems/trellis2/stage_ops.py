"""
Trellis2 StageOps 具体实现 — Shape / Tex 阶段的计算操作。

继承 Trellis2StageOps（generators/trellis2/stage_ops_base.py），获得：
  - vjp_loop:            通用 VJP 循环默认实现
  - normalize_slat / denormalize_slat / get_flow_resolution / get_sigma_min
  - pretrained_rollout / add_noise / sample_timestep / predict_cfg_velocity 等 onestep 公共方法

本文件仅实现 Trellis2 具体阶段的差异部分。

实现：
  ShapeOps        — Shape 阶段（含 dense_sampling）
  TexOps          — Tex-only 阶段（含 shape_frozen_prepare）
  TexOpsFromShape — Tex-from-Shape 阶段（跳过 shape_frozen_prepare）

使用方式：
  # 同步模板（VJP）
  three_phase_step(ShapeOps(), state, system, ...)

  # 同步模板（Onestep）
  onestep_step(ShapeOps(), state, system, ...)

  # 异步模板
  StageContext(ops=ShapeOps(), tracker=tracker)

设计原则：
  - 同一个 ShapeOps 在 shape-only / shape+tex / cascade / onestep 中一行不改
  - vjp_loop 继承自 Trellis2StageOps，子类无需重复
  - 清理策略由编排层决定，不由 Ops 决定
  - Ops 只回答 "这个阶段自身的计算是什么？"
"""

from __future__ import annotations

from typing import Any, Dict

import torch

# 中间基类（Trellis2 公共逻辑：vjp_loop / normalize / denormalize / onestep 方法）
from edit4shape.generators.trellis2.stage_ops_base import Trellis2StageOps

# 异常从 utils 导入（模型无关的抽象层）
from edit4shape.systems.utils.stage_ops import StageSkipError  # noqa: F401 — re-export

# 渲染 & Phase 辅助
from edit4shape.systems.trellis2.forward import (
    decode_and_render_normal,
    decode_and_render_normal_filled,
    decode_and_render_normal_hybrid26,
    decode_and_render_pbr,
    dense_sampling_no_grad,
    trellis2_shape_forward,
    _detach_shape_outputs,
)
from edit4shape.generators.trellis2.rollout import (
    rollout_shape, rollout_tex, RolloutTracker,
)


# =====================================================================
# 具体实现 — Shape
# =====================================================================

class ShapeOps(Trellis2StageOps):
    """
    Shape 阶段 Ops — 包装 shape_autograd.py 中的现有函数。

    计算链（VJP 模式）：
      dense_sampling → rollout_shape → decode_and_render_normal → guidance → VJP

    计算链（Onestep 模式）：
      dense_sampling → pretrained_rollout → add_noise → predict_cfg_velocity
      → decode_render → guidance → relay

    根据 system.cfg.shape.renderer.type 自动选择渲染路径：
      - "mesh_peeled":     decode_and_render_normal（face normal 路径）
      - "hybrid26_peeled": decode_and_render_normal_hybrid26（26-neighbor voxel normal 路径）

    VJP loop 继承自 Trellis2StageOps（shape/tex 逻辑完全相同）。
    """

    def get_model(self, system):
        return system.shape.model

    def get_stage_name(self) -> str:
        return "shape"

    def get_seed_offset(self) -> int:
        return 0

    def get_reg_weight(self, system) -> float:
        return system.cfg.shape.train.loss.reg

    def get_reg_type(self, system) -> str:
        return str(system.cfg.shape.train.loss.reg_type)

    def get_guidance_weight(self, system) -> float:
        return system.cfg.shape.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.shape.guidance

    def get_guidance_grad_max_norm(self, system) -> float:
        return system.cfg.shape.train.loss.guidance_grad_max_norm

    # ── Async 友好查询 ──

    def get_slat(self, state):
        return state.features.shape_slat

    # get_shape_cond → 继承默认 None

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """decode+render Normal → 原始字典（不含 vis 挂载）。

        根据 system.cfg.shape.renderer.type 选择渲染路径：
          - "mesh_peeled":     face normal 路径（MeshPeeledRenderer）
          - "hybrid26_peeled": 26-neighbor voxel normal 路径（Hybrid26NormalRenderer）
        """
        renderer_type = system.cfg.shape.renderer.type
        if renderer_type == "hybrid26_peeled":
            decode_fn = decode_and_render_normal_hybrid26
        elif renderer_type == "mesh_peeled":
            decode_fn = decode_and_render_normal
        elif renderer_type == "mesh_filled":
            decode_fn = decode_and_render_normal_filled
        else:
            raise ValueError(f"Unknown shape renderer type: {renderer_type}")

        return decode_fn(
            state.features.shape_slat,
            state.cameras,
            system.pipeline,
            system.shape.renderer,
            system.accelerator.device,
            resolution=system.pipeline.target_resolution,
            bg_color=tuple(system.cfg.shape.renderer.bg_color),
            grad_shrink_scale=system.cfg.shape.renderer.grad_shrink_scale,
        )

    # ── Phase 函数 ──

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Dense Sampling → 填充 state.coords。"""
        dense_sampling_no_grad(state, system)

    def rollout(self, state, system, seed) -> RolloutTracker:
        """Phase 1: Shape rollout → proxy chain + tracker。"""
        stage_config = system.pipeline.get_stage_config("shape")
        device = system.accelerator.device
        tracker = RolloutTracker()
        gen = torch.Generator(device=device).manual_seed(seed)
        rollout_shape(
            state, system.cfg, system, device,
            resolution=stage_config["flow_resolution"],
            generator=gen,
            is_training=False,   # 模型推理 no_grad
            tracker=tracker,     # ★ 记录 proxy 轨迹
        )
        return tracker

    def decode_render(self, state, system) -> torch.Tensor:
        """Phase 2a: decode + render Normal → comp_rgb（含 vis 挂载）。"""
        render_out = self.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, 3)
        # 挂载 vis 和中间产物
        state.views_generated.shape_tensor = comp_rgb.detach()
        state.features.subs = render_out["subs"]
        state.features.meshes = render_out["meshes"]
        return comp_rgb

    # vjp_loop → 继承自 Trellis2StageOps（通用实现）

    # ── Onestep 专用：_pretrained_rollout_impl ──

    def _pretrained_rollout_impl(self, state, system, seed) -> None:
        """
        Shape pretrained rollout（在 teacher_context + no_grad 上下文内调用）。

        直接调用 rollout_shape（tracker=None），不记录 proxy chain。
        """
        cfg = system.cfg
        device = system.accelerator.device
        stage_config = system.pipeline.get_stage_config("shape")
        gen = torch.Generator(device=device).manual_seed(seed)

        rollout_shape(
            state, cfg, system, device,
            resolution=stage_config["flow_resolution"],
            generator=gen,
            is_training=False,
            tracker=None,  # 不需要 proxy chain
        )


# =====================================================================
# 具体实现 — Tex (standalone)
# =====================================================================

class TexOps(Trellis2StageOps):
    """
    Tex-only 阶段 Ops — 含 Phase 0 (shape_frozen_prepare)。

    用于 tex-only 训练模式，Phase 0 执行冻结的 Shape forward + 全量 detach，
    然后 Tex rollout 使用 shape 产物作为条件。

    计算链（VJP 模式）：
      shape_frozen_prepare → rollout_tex → decode_and_render_pbr → guidance → VJP

    计算链（Onestep 模式）：
      shape_frozen_prepare → pretrained_rollout → add_noise → predict_cfg_velocity
      → decode_render → guidance → relay

    VJP loop 继承自 Trellis2StageOps（shape/tex 逻辑完全相同）。
    """

    def get_model(self, system):
        return system.tex.model

    def get_stage_name(self) -> str:
        return "tex"

    def get_seed_offset(self) -> int:
        return 1000

    def get_reg_weight(self, system) -> float:
        return system.cfg.tex.train.loss.reg

    def get_reg_type(self, system) -> str:
        return str(system.cfg.tex.train.loss.reg_type)

    def get_guidance_weight(self, system) -> float:
        return system.cfg.tex.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.tex.guidance

    def get_guidance_grad_max_norm(self, system) -> float:
        return system.cfg.tex.train.loss.guidance_grad_max_norm

    # ── Async 友好查询 ──

    def get_slat(self, state):
        return state.features.tex_slat

    def get_shape_cond(self, state):
        """Tex VJP 需要 shape_slat_norm 作为 concat_cond。"""
        return state.features.shape_slat_norm

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """decode+render PBR → 原始字典（不含 vis 挂载）。"""
        return decode_and_render_pbr(
            state.features.meshes,
            state.features.tex_slat,
            state.features.subs,
            state.cameras,
            system.pipeline,
            system.tex.renderer,
            system.accelerator.device,
            resolution=system.pipeline.target_resolution,
            bg_color=tuple(system.cfg.tex.renderer.bg_color),
            grad_shrink_scale=system.cfg.tex.renderer.grad_shrink_scale,
        )

    # ── Phase 函数 ──

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Shape 冻结前置（no_grad shape forward + detach）。"""
        with torch.no_grad():
            trellis2_shape_forward(
                system, state, global_step,
                is_training=False,
                render_normal=False,
            )
        _detach_shape_outputs(state)

    def rollout(self, state, system, seed) -> RolloutTracker:
        """Phase 1: Tex rollout → proxy chain + tracker。"""
        stage_config = system.pipeline.get_stage_config("tex")
        device = system.accelerator.device
        tracker = RolloutTracker()
        gen = torch.Generator(device=device).manual_seed(seed)
        rollout_tex(
            state, system.cfg, system, device,
            resolution=stage_config["flow_resolution"],
            generator=gen,
            is_training=False,   # 模型推理 no_grad
            tracker=tracker,     # ★ 记录 proxy 轨迹
        )
        return tracker

    def decode_render(self, state, system) -> torch.Tensor:
        """Phase 2a: decode + render PBR → comp_rgb（含 vis 挂载）。"""
        render_out = self.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, 3)
        state.views_generated.pbr_tensor = comp_rgb.detach()
        return comp_rgb

    # vjp_loop → 继承自 Trellis2StageOps（通用实现）

    # ── Onestep 专用：_pretrained_rollout_impl ──

    def _pretrained_rollout_impl(self, state, system, seed) -> None:
        """
        Tex pretrained rollout（在 teacher_context + no_grad 上下文内调用）。

        直接调用 rollout_tex（tracker=None），不记录 proxy chain。
        前置条件：shape 产物已就绪（由 pre_rollout / shape_frozen_prepare 提供）。
        """
        cfg = system.cfg
        device = system.accelerator.device
        stage_config = system.pipeline.get_stage_config("tex")
        gen = torch.Generator(device=device).manual_seed(seed)

        rollout_tex(
            state, cfg, system, device,
            resolution=stage_config["flow_resolution"],
            generator=gen,
            is_training=False,
            tracker=None,  # 不需要 proxy chain
        )


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
    - decode_render / decode_render_dict 增加 meshes 可用性检查（Shape P2 OOM 时 meshes 为 None）
    """

    def pre_rollout(self, state, system, global_step) -> None:
        """No-op: Shape 产物由上游 Shape 阶段 + detach 转接提供。"""
        pass

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        Tex decode+render dict，带 meshes 可用性检查。

        如果上游 Shape P2a OOM 导致 meshes 为 None，
        抛出 StageSkipError 使模板跳过 P2/P3。
        """
        if state.features.meshes is None:
            raise StageSkipError(
                "meshes 不可用（Shape P2a OOM），跳过 Tex P2a"
            )
        return super().decode_render_dict(state, system)
