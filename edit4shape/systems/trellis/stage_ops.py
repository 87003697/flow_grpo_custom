"""
Trellis v1 StageOps 具体实现 — 单阶段（SLAT Flow Model）。

ABC (StageOps) 定义在 edit4shape.systems.utils.stage_ops，
本文件提供 Trellis v1 特定的实现，使其可接入 autograd_template 编排模板。

实现：
  TrellisOps      — 通用单阶段，根据 cfg.renderer.type 分发 mesh/gs 渲染
  TrellisMeshOps  — 强制 Mesh Normal 渲染（覆写 decode_render_dict）
  TrellisGsOps    — 强制 GS Color 渲染（覆写 decode_render_dict）

使用方式：
  from edit4shape.systems.trellis.stage_ops import TrellisOps
  from edit4shape.systems.trellis.autograd_template import trellis_three_phase_step

  trellis_three_phase_step(TrellisOps(), state, system, ...)

设计原则：
  - 同一个 TrellisOps 在 standard / autograd / bilevel 入口中一行不改
  - 子类仅需覆写 decode_render_dict 即可切换渲染策略
  - 清理策略由编排层的 clean_for_vjp 回调注入
  - vjp_loop 手动合并 reg_grads（适配 3-sub-step Phase 2）
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

# ABC 从 utils 导入（模型无关的抽象层）
from edit4shape.systems.utils.stage_ops import StageOps  # noqa: F401 — re-export

# Phase 函数 & 渲染
from edit4shape.systems.trellis.forward import decode_and_render_mesh, decode_and_render_gs
from edit4shape.systems.trellis.phases import dense_sampling_no_grad, phase3_rollout_grad_backward
from edit4shape.generators.trellis.rollout import rollout_sparse, RolloutTracker


# =====================================================================
# 通用实现 — 根据 cfg.renderer.type 分发
# =====================================================================

class TrellisOps(StageOps):
    """
    Trellis v1 单阶段 Ops — SLAT Flow Model。

    计算链：
      dense_sampling → rollout_sparse → decode_and_render_{mesh|gs} → guidance → VJP

    Phase 2 保持 3-sub-step 显存优化：
      P2a: no_grad decode → detached comp_rgb
      P2b: guidance-only backward → rgb_grad
      P2c: with-grad decode → backward(rgb_grad) → cond_proxy.grad（仅 guidance）

    Phase 3 手动合并 reg_grads：
      v_grad = guidance_grad + reg_weight * reg_grad → VJP
    """

    # ═══════════════════════════════════════════════════════
    # 配置查询
    # ═══════════════════════════════════════════════════════

    def get_model(self, system):
        """返回 DDP 包装的 slat_flow_model。"""
        return system.pipeline.pipe.models['slat_flow_model']

    def get_stage_name(self) -> str:
        return "slat"

    def get_seed_offset(self) -> int:
        return 0

    def get_reg_weight(self, system) -> float:
        return system.cfg.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.train.guidance

    def get_gs_reg_config(self, system) -> Dict[str, float]:
        """
        返回 GS 表示正则化权重（reg_vol / reg_opacity）。

        从 cfg.train.loss.gs_reg 读取；未配置时返回 0（不启用）。
        """
        cfg = system.cfg
        gs_reg = cfg.train.loss.get("gs_reg", {})
        return {
            "lambda_vol": float(gs_reg.get("vol", 0.0)),
            "lambda_opacity": float(gs_reg.get("opacity", 0.0)),
        }

    # ═══════════════════════════════════════════════════════
    # Async 友好查询
    # ═══════════════════════════════════════════════════════

    def get_slat(self, state):
        return state.features.slat

    # get_shape_cond → 继承默认 None（单模型无 shape cond）

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        根据 cfg.renderer.type 分发到 mesh 或 gs 渲染。

        子类可覆写此方法以实现自定义渲染策略
        （如 TrellisMeshOps / TrellisGsOps / 自定义混合渲染）。
        """
        latents = state.features.slat
        device = system.accelerator.device

        renderer_type = system.cfg.renderer.type
        renderer = system.renderers[renderer_type]  # 从 renderers dict 查找
        if renderer_type == "gs":
            return decode_and_render_gs(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
            render_out["color"] = render_out["normal"]
            return render_out

    # ═══════════════════════════════════════════════════════
    # Phase 函数
    # ═══════════════════════════════════════════════════════

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Dense Sampling → 填充 state.coords。"""
        dense_sampling_no_grad(state, system, system.accelerator.device)

    def rollout(self, state, system, seed) -> RolloutTracker:
        """
        Phase 1: rollout_sparse → proxy chain + tracker。

        对齐 StageOps 签名：接收 seed（由编排函数计算），
        内部创建 Generator 并调用 rollout_sparse。
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)
        tracker = RolloutTracker()

        rollout_sparse(
            state, cfg, system, device,
            generator=generator,
            is_training=False,   # 模型推理 no_grad
            tracker=tracker,     # ★ 记录 proxy 轨迹
        )
        torch.cuda.empty_cache()
        return tracker

    def decode_render(self, state, system) -> torch.Tensor:
        """
        Phase 2a: decode + render → comp_rgb（含 autograd 图，连接到 proxy chain）。

        调用 decode_render_dict + 挂载可视化数据。
        """
        render_out = self.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, C)
        state.views_generated.image_tensor = comp_rgb.detach()
        return comp_rgb

    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: VJP → θ.grad 累积（委托 phases.phase3_rollout_grad_backward）。

        ★ 适配 3-sub-step Phase 2：cond_proxy.grad 仅含 guidance 梯度，
        phase3_rollout_grad_backward 内部手动合并 reg_weight * reg_grads[i]。
        """
        log = tracker.collect_log()  # loss/reg + grad_norm/*（VJP 前收集）

        phase3_rollout_grad_backward(
            state, system, system.cfg, system.accelerator.device, tracker,
        )

        return log


# =====================================================================
# 显式渲染策略子类
# =====================================================================

class TrellisMeshOps(TrellisOps):
    """强制使用 Mesh Normal 渲染，不受 cfg.renderer.type 控制。"""

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        render_out = decode_and_render_mesh(
            state.features.slat, state.cameras,
            system.pipeline, system.renderers["mesh"], system.accelerator.device,
        )
        render_out["color"] = render_out["normal"]
        return render_out


class TrellisGsOps(TrellisOps):
    """强制使用 GS Color 渲染，不受 cfg.renderer.type 控制。"""

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        return decode_and_render_gs(
            state.features.slat, state.cameras,
            system.pipeline, system.renderers["gs"], system.accelerator.device,
        )


# =====================================================================
# 双路渲染 — Mesh Normal + GS Color 同时提供 guidance
# =====================================================================

class TrellisHybridOps(TrellisOps):
    """
    双路渲染 Ops：Mesh Normal + GS Color，各自独立提供 guidance。

    设计：
      - decode_render_dict(renderer_key=...) 按指定渲染器分发
      - get_render_passes() 返回多路渲染配置，供 hybrid 编排循环使用
      - 继承 TrellisOps 的 rollout / pre_rollout / vjp_loop（共享 SLAT Flow Model）

    使用方式：
      from edit4shape.systems.trellis.stage_ops import TrellisHybridOps
      from edit4shape.systems.trellis.autograd_template import trellis_hybrid_three_phase_step

      trellis_hybrid_three_phase_step(TrellisHybridOps(), state, system, ...)

    配置要求（cfg.train 下）：
      guidance_normal:      Mesh Normal guidance 配置
      guidance_color:       GS Color guidance 配置
      loss.guidance_normal: Mesh Normal guidance 权重
      loss.guidance_color:  GS Color guidance 权重
    """

    def decode_render_dict(self, state, system, renderer_key: str = "gs") -> Dict[str, Any]:
        """
        按 renderer_key 分发到指定渲染器。

        Args:
            state: TrellisState
            system: TrellisSystem（需 renderers 包含 "mesh" 和 "gs"）
            renderer_key: "mesh" 或 "gs"

        Returns:
            渲染输出字典，"color" key 统一为各渲染器的主要输出：
              mesh → normal,  gs → color
        """
        latents = state.features.slat
        device = system.accelerator.device
        renderer = system.renderers[renderer_key]

        if renderer_key == "gs":
            return decode_and_render_gs(
                latents, state.cameras, system.pipeline, renderer, device,
            )
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, renderer, device,
            )
            render_out["color"] = render_out["normal"]
            return render_out

    def get_render_passes(self, system) -> List[Tuple[str, Any, float]]:
        """
        返回多路渲染配置列表。

        Returns:
            [(renderer_key, guidance_cfg, guidance_weight), ...] 的列表。
            编排函数会循环处理每一路：P2a → P2b → P2c。
        """
        cfg = system.cfg
        return [
            ("mesh", cfg.train.guidance_normal, cfg.train.loss.guidance_normal),
            ("gs", cfg.train.guidance_color, cfg.train.loss.guidance_color),
        ]
