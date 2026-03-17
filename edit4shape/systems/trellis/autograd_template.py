"""
Trellis v1 三阶段 Autograd 模板 — 3-sub-step Phase 2 显存优化。

提供两种编排：
  ● trellis_three_phase_step        — 单路渲染
  ● trellis_hybrid_three_phase_step — 双路渲染（Mesh Normal + GS Color）

单路编排：
  P0 → P1 (rollout) → P2a/P2b/P2c (decode-guidance-backward)
  → clean_for_vjp → P3 (VJP) → 日志

双路编排：
  P0 → P1 (rollout) → 梯度中转 (detach slat → leaf)
  → {Mesh P2a/P2b/P2c} → 路间清理 → {GS P2a/P2b/P2c}
  → 梯度中继 (leaf.grad → proxy chain) → clean_for_vjp → P3 (VJP) → 日志

通过 StageOps 参数化渲染策略：
  TrellisOps      → cfg.renderer.type 自动分发（单路）
  TrellisMeshOps  → 强制 Mesh Normal（单路）
  TrellisGsOps    → 强制 GS Color（单路）
  TrellisHybridOps → Mesh Normal + GS Color（双路）

使用方式::

    from edit4shape.systems.trellis.autograd_template import (
        trellis_three_phase_step,
        trellis_hybrid_three_phase_step,
    )
    from edit4shape.systems.trellis.stage_ops import TrellisOps, TrellisHybridOps
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import VelocityTracker
from edit4shape.systems.trellis.system import TrellisSystem
from edit4shape.systems.trellis.forward import compute_gs_regularization
from edit4shape.systems.utils.profiler import PhaseProfiler
from edit4shape.systems.utils.logging import build_autograd_step_log


# =====================================================================
# Phase 3.5: Velocity Regularization Backward（FlowEdit 专用）
# =====================================================================

def _phase3_5_velocity_reg(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    tracker: "VelocityTracker",
    zt_feats: torch.Tensor,
    t_val: float,
    reg_weight: float,
    reg_type: str = "v",
) -> None:
    """
    Phase 3.5: teacher velocity 预测 + 正则化 backward → tracker.reg_grad。

    支持三种正则化类型：
      - "v":  MSE(v_proxy, v_teacher)
      - "x0": MSE(x0_stu, x0_tea) / (t² + ε)，单步下与 v reg 数学等价
      - "x1": MSE(x0_stu, x0_tea)，不除以 t²，小 t 时正则化更弱

    完成后 tracker 中写入：
      - reg_grad:     (N, C) detached 正则化梯度
      - reg_loss_val: float 标量值（日志用）
      - v_proxy.grad: 被清零（留给 P4c guidance 梯度）

    Args:
        ops: TrellisFlowEditOps 实例
        state: TrellisState
        system: 系统组件
        tracker: VelocityTracker（已 setup_proxy）
        zt_feats: (N, C) 加噪后的特征，detached
        t_val: float, 归一化时间步 [0, 1]
        reg_weight: 正则化权重（> 0）
        reg_type: 正则化类型，"v" | "x0" | "x1"
    """
    from edit4shape.generators.trellis.rollout.ode import (
        _compute_x0_regularization,
        _compute_x1_regularization,
        _compute_v_regularization,
    )

    v_teacher = ops.predict_velocity_teacher(
        state, system, zt_feats, t_val,
    )  # (N, C), detached

    if reg_type == "x0":
        x0_stu = zt_feats - t_val * tracker.v_proxy  # (N, C), 依赖 v_proxy
        x0_tea = zt_feats - t_val * v_teacher         # (N, C), detached
        raw_reg = _compute_x0_regularization(x0_stu, x0_tea, t_val)  # scalar
    elif reg_type == "x1":
        x0_stu = zt_feats - t_val * tracker.v_proxy  # (N, C), 依赖 v_proxy
        x0_tea = zt_feats - t_val * v_teacher         # (N, C), detached
        raw_reg = _compute_x1_regularization(x0_stu, x0_tea)  # scalar
    elif reg_type == "v":
        raw_reg = _compute_v_regularization(tracker.v_proxy, v_teacher)  # scalar
    else:
        raise ValueError(
            f"Unknown reg_type: {reg_type!r}, expected 'v', 'x0', or 'x1'"
        )

    reg_loss = reg_weight * raw_reg  # scalar
    reg_loss.backward()  # → v_proxy.grad = reg_grad
    tracker.reg_grad = tracker.v_proxy.grad.detach().clone()  # (N, C)
    tracker.reg_loss_val = reg_loss.item()
    tracker.v_proxy.grad = None  # ★ 清零，给 P4c 的 guidance 梯度腾位

    del v_teacher, reg_loss
    torch.cuda.empty_cache()


# =====================================================================
# Phase 2b: Guidance-Only Backward → rgb_grad
# =====================================================================

def _phase2_guidance_only_backward(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    comp_rgb_detached: torch.Tensor,
    guidance_cfg: Any = None,
    guidance_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    3-sub-step Phase 2b: guidance-only backward → rgb_grad。

    ★ 不合并 reg_loss（reg 梯度已在 Phase 1 预计算到 tracker.reg_grads）。

    参数来源（优先级从高到低）：
      1. 显式传入 guidance_cfg / guidance_weight（hybrid 多路编排使用）
      2. 通过 ops 查询 get_guidance_cfg / get_guidance_weight（单路编排使用）

    Args:
        ops: StageOps 实例（提供 guidance 配置查询的 fallback）
        state: TrellisState
        system: 系统组件（含 cfg, accelerator）
        comp_rgb_detached: (B, V, H, W, C) Phase 2a 产出的无梯度图像
        guidance_cfg: 显式 guidance 配置（可选，hybrid 用）
        guidance_weight: 显式 guidance 权重（可选，hybrid 用）

    Returns:
        rgb_grad: (B, V, H, W, C) guidance 对渲染图像的梯度
        guidance_log: 日志字典
    """
    accelerator = system.accelerator
    device = accelerator.device
    if guidance_cfg is None:
        guidance_cfg = ops.get_guidance_cfg(system)
    if guidance_weight is None:
        guidance_weight = ops.get_guidance_weight(system)

    # ---- 创建 proxy 叶节点（guidance 梯度的终点）----
    comp_rgb_proxy = comp_rgb_detached.detach().requires_grad_(True)  # (B,V,H,W,C) leaf

    # ---- guidance forward ----
    guidance_result = system.guidance.compute_guidance(
        comp_rgb_proxy,
        state.views_conditioned.image_pils,
        guidance_cfg=guidance_cfg,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)

    # ---- guidance-only backward → rgb_grad ----
    guidance_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    accelerator.backward(guidance_loss)

    rgb_grad = comp_rgb_proxy.grad.detach().clone()  # (B, V, H, W, C)

    # ---- 构建日志 ----
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = guidance_loss.item()

    # ---- 释放 guidance 计算图 ----
    del comp_rgb_proxy, guidance_loss, guidance_result
    torch.cuda.empty_cache()

    return rgb_grad, guidance_log


# =====================================================================
# 单路渲染三阶段训练步
# =====================================================================

def trellis_three_phase_step(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    global_step: int,
    profiler: PhaseProfiler,
    clean_for_vjp: Callable,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Trellis v1 三阶段训练步 — 保留 3-sub-step Phase 2 显存优化 + StageOps 多态。

    编排：
      P0 (ops.pre_rollout) → P1 (ops.rollout + tracker)
      → P2a (ops.decode_render_dict no_grad) → P2b (guidance-only backward → rgb_grad)
      → P2c (ops.decode_render_dict with grad + backward(rgb_grad))
      → clean_for_vjp → P3 (ops.vjp_loop with reg_grads merge) → 返回日志

    显存峰值 = max(guidance, decode_render)，保持 3-sub-step 显存优化。

    通过 StageOps 参数化渲染策略：
      TrellisOps     → cfg.renderer.type 自动分发
      TrellisMeshOps → 强制 Mesh Normal
      TrellisGsOps   → 强制 GS Color
      自定义子类     → 覆写 decode_render_dict 实现任意渲染

    Args:
        ops: StageOps 实例（参数化渲染/rollout/VJP 计算）
        state: 已 attach_batch 的 TrellisState
        system: 系统组件（含 cfg, accelerator）
        global_step: 全局步数
        profiler: PhaseProfiler（tick / collect 接口）
        clean_for_vjp: P2→P3 过渡清理函数（由调用方注入）
        prefix: profiler tick 和日志 key 的前缀

    Returns:
        合并的日志字典（不含 profiler 计时——由调用方决定是否收集）
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()

    # ── Phase 0: 准备（dense_sampling）──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout → proxy chain + tracker ──
    profiler.tick(f"{prefix}P1_rollout")
    tracker = ops.rollout(state, system, seed)

    # ── ★ Rollout 结束后：清理 spatial cache（neighbor maps / window partition indices）──
    # rollout 过程中 SparseTensor 会累积大量 spatial cache，decode 阶段不再需要
    state.features.slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── Phase 2a: no_grad decode/render ──
    profiler.tick(f"{prefix}P2a_decode_no_grad")
    with torch.no_grad():
        render_out = ops.decode_render_dict(state, system)
    comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)
    state.views_generated.image_tensor = comp_rgb_detached
    del render_out

    # ── Phase 2b: guidance-only backward → rgb_grad ──
    profiler.tick(f"{prefix}P2b_guidance_backward")
    rgb_grad, guidance_log = _phase2_guidance_only_backward(
        ops, state, system, comp_rgb_detached,
    )
    del comp_rgb_detached

    # ── Phase 2c: with-grad decode/render + backward(rgb_grad + GS reg) ──
    profiler.tick(f"{prefix}P2c_decode_grad")
    render_out = ops.decode_render_dict(state, system)
    comp_rgb = render_out["color"]  # (B, V, H, W, C), has autograd graph
    gaussians = render_out.get("gaussians")  # list[B] or None（mesh 路无此 key）

    # ---- 计算 GS 正则化（仅 GS 渲染时生效）----
    gs_reg_log: Dict[str, Any] = {}
    gs_reg_cfg = ops.get_gs_reg_config(system)
    has_gs_reg = gaussians is not None and (
        gs_reg_cfg["lambda_vol"] > 0 or gs_reg_cfg["lambda_opacity"] > 0
    )
    if has_gs_reg:
        gs_reg_loss, gs_reg_log = compute_gs_regularization(gaussians, **gs_reg_cfg)
        # 合并 guidance + GS reg 的 backward，单次 pass 释放整个 decode/render 图
        total_loss = (comp_rgb * rgb_grad).sum() + gs_reg_loss  # scalar
        total_loss.backward()
    else:
        comp_rgb.backward(rgb_grad)  # → cond_proxy.grad（仅 guidance 梯度）

    del comp_rgb, render_out, rgb_grad, gaussians
    state.regularization.reg_loss = None
    torch.cuda.empty_cache()

    # ── P2→P3 过渡：调用方注入的清理策略 ──
    clean_for_vjp(state)

    # ── Phase 3: VJP (ops 内部合并 reg_grads) → θ.grad 累积 ──
    profiler.tick(f"{prefix}P3_vjp")
    phase3_log = ops.vjp_loop(state, system, tracker)

    profiler.tick(f"{prefix}end")

    log = build_autograd_step_log(
        guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )
    log.update({f"{prefix}{k}": v for k, v in gs_reg_log.items()})
    return log


# =====================================================================
# 双路 Guidance Backward 辅助函数
# =====================================================================

def _phase2_color_guidance_backward(
    ops, state, system, comp_rgb_detached, guidance_cfg, guidance_weight,
):
    """P2b Color 路：guidance backward → rgb_grad，结果存入 color_tensor / color_trackers。"""
    rgb_grad, log = _phase2_guidance_only_backward(
        ops, state, system, comp_rgb_detached, guidance_cfg, guidance_weight,
    )
    # attach_guidance_result 默认已写入 color_tensor / color_trackers，无需搬运
    return rgb_grad, log


def _phase2_normal_guidance_backward(
    ops, state, system, comp_rgb_detached, guidance_cfg, guidance_weight,
):
    """P2b Normal 路：guidance backward → rgb_grad，结果搬入 normal_tensor / normal_trackers。"""
    rgb_grad, log = _phase2_guidance_only_backward(
        ops, state, system, comp_rgb_detached, guidance_cfg, guidance_weight,
    )
    # 从默认的 color_* 缓冲区搬到 normal_* 字段
    state.views_edited.normal_tensor = state.views_edited.color_tensor
    state.views_edited.normal_trackers = state.views_edited.color_trackers
    state.views_edited.color_tensor = None
    state.views_edited.color_trackers = None
    return rgb_grad, log


# =====================================================================
# 双路渲染三阶段训练步
# =====================================================================

def trellis_hybrid_three_phase_step(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    global_step: int,
    profiler: PhaseProfiler,
    clean_for_vjp: Callable,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    双路渲染三阶段训练步 — P2 循环处理多路渲染，梯度在 proxy.grad 上累加。

    编排：
      P0 (ops.pre_rollout) → P1 (ops.rollout + tracker)
      → 梯度中转：detach slat → slat_feats_leaf（新 leaf）
      → 循环每路渲染 {
            P2a (ops.decode_render_dict(renderer_key=key) no_grad)
            → P2b (guidance-only backward → rgb_grad，使用各路独立 cfg/weight)
            → P2c (ops.decode_render_dict(renderer_key=key) with grad + backward → slat_feats_leaf.grad)
        }
      → 梯度中继：original_slat.feats.backward(slat_feats_leaf.grad) → proxy.grad
      → clean_for_vjp → P3 (ops.vjp_loop with reg_grads merge) → 返回日志

    ★ 梯度中转设计：
      - 在 slat 层切断 proxy chain，建立 leaf 中转点
      - 每路 P2c backward 独立终止在 slat_feats_leaf（不需要 retain_graph）
      - 每路结束后 decode/render 图立即释放
      - 循环结束后一次性把累积梯度回传到 proxy chain

    显存峰值 = max(guidance, decode_render)（与单路相同，每路 P2c 结束即释放）。
    额外开销仅为 slat_feats_leaf.grad 一个向量（与 slat 特征同维度）。

    Args:
        ops: TrellisHybridOps 实例（需提供 get_render_passes / decode_render_dict(renderer_key=)）
        state: 已 attach_batch 的 TrellisState
        system: 系统组件（含 cfg, accelerator，renderers 需包含所有渲染器）
        global_step: 全局步数
        profiler: PhaseProfiler
        clean_for_vjp: P2→P3 过渡清理函数
        prefix: profiler tick 和日志 key 的前缀

    Returns:
        合并的日志字典（不含 profiler 计时——由调用方决定是否收集）
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()

    # ── Phase 0: 准备（dense_sampling）──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout → proxy chain + tracker ──
    profiler.tick(f"{prefix}P1_rollout")
    tracker = ops.rollout(state, system, seed)

    # ── ★ Rollout 结束后：清理 spatial cache（neighbor maps / window partition indices）──
    # rollout 过程中 SparseTensor 会累积大量 spatial cache，decode 阶段不再需要
    state.features.slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── ★ 梯度中转：在 slat 层切断 proxy chain，建立 leaf 中转点 ──
    # original_slat.feats 依赖 proxy chain（scheduler → proxies）
    # slat_feats_leaf 是 detached leaf，decode/render 的 backward 终止于此
    # 多路梯度在 slat_feats_leaf.grad 上累加，循环结束后一次性回传到 proxy chain
    original_slat = state.features.slat
    slat_feats_leaf = original_slat.feats.detach().requires_grad_(True)
    # replace() 保留 spatial_cache / indice_dict / layout，仅换 feats
    state.features.slat = original_slat.replace(slat_feats_leaf)

    # ── Phase 2: 双路渲染（Mesh Normal + GS Color）──
    # ★ 每路独立 try/except OOM 保护 + 路间 _spatial_cache.clear()。
    #   OOM 时跳过该路梯度，不影响另一路和后续 VJP。
    #   P2 不经过模型参数（梯度终止在 slat_feats_leaf），不触发 DDP hooks。
    all_guidance_log: Dict[str, Any] = {}
    cfg = system.cfg

    # ────────────────────────────────────────────────────
    # Mesh Normal 路（★ OOM 保护）
    # ────────────────────────────────────────────────────
    try:
        # P2a: no_grad decode/render
        profiler.tick(f"{prefix}P2a_mesh")
        with torch.no_grad():
            render_out = ops.decode_render_dict(state, system, renderer_key="mesh")
        comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)
        del render_out

        # P2b: guidance backward → rgb_grad
        profiler.tick(f"{prefix}P2b_mesh")
        state.views_generated.normal_tensor = comp_rgb_detached  # (B, V, H, W, C) Mesh Normal
        rgb_grad, guidance_log = _phase2_normal_guidance_backward(
            ops, state, system, comp_rgb_detached,
            cfg.train.guidance_normal, cfg.train.loss.guidance_normal,
        )
        all_guidance_log.update({f"mesh/{k}": v for k, v in guidance_log.items()})
        del comp_rgb_detached

        # P2c: with-grad decode/render + backward(rgb_grad)
        # ★ 无需 retain_graph：backward 终止在 slat_feats_leaf（leaf），
        #   每路 decode/render 图独立创建、独立释放
        profiler.tick(f"{prefix}P2c_mesh")
        render_out = ops.decode_render_dict(state, system, renderer_key="mesh")
        comp_rgb = render_out["color"]  # (B, V, H, W, C), has autograd graph
        comp_rgb.backward(rgb_grad)  # slat_feats_leaf.grad += mesh 路梯度
        del comp_rgb, render_out, rgb_grad
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        e.__traceback__ = None
        logging.warning(
            f"[Step {global_step}] {prefix}P2_mesh OOM: {e} → 跳过 mesh 梯度，继续 GS"
        )
        # ★ 释放 try 块中可能残留的 GPU 张量（尤其是 P2c 的 autograd 图）
        render_out = comp_rgb = comp_rgb_detached = rgb_grad = None
        gc.collect()
        torch.cuda.empty_cache()

    # ── ★ 路间清理：释放 Mesh decode 累积的 spatial cache，为 GS 路腾出显存 ──
    state.features.slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────
    # GS Color 路（★ OOM 保护）
    # ────────────────────────────────────────────────────
    try:
        # P2a: no_grad decode/render
        profiler.tick(f"{prefix}P2a_gs")
        with torch.no_grad():
            render_out = ops.decode_render_dict(state, system, renderer_key="gs")
        comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)
        del render_out

        # P2b: guidance backward → rgb_grad
        profiler.tick(f"{prefix}P2b_gs")
        state.views_generated.image_tensor = comp_rgb_detached  # (B, V, H, W, C) GS Color
        rgb_grad, guidance_log = _phase2_color_guidance_backward(
            ops, state, system, comp_rgb_detached,
            cfg.train.guidance_color, cfg.train.loss.guidance_color,
        )
        all_guidance_log.update({f"gs/{k}": v for k, v in guidance_log.items()})
        del comp_rgb_detached

        # P2c: with-grad decode/render + backward(rgb_grad + GS reg)
        profiler.tick(f"{prefix}P2c_gs")
        render_out = ops.decode_render_dict(state, system, renderer_key="gs")
        comp_rgb = render_out["color"]  # (B, V, H, W, C), has autograd graph
        gaussians = render_out.get("gaussians")  # list[B] of Gaussian

        # ---- GS 正则化（reg_vol / reg_opacity）----
        gs_reg_cfg = ops.get_gs_reg_config(system)
        has_gs_reg = gaussians is not None and (
            gs_reg_cfg["lambda_vol"] > 0 or gs_reg_cfg["lambda_opacity"] > 0
        )
        if has_gs_reg:
            gs_reg_loss, gs_reg_log = compute_gs_regularization(gaussians, **gs_reg_cfg)
            all_guidance_log.update({f"gs/{k}": v for k, v in gs_reg_log.items()})
            # 合并 guidance + GS reg 的 backward，单次 pass 释放整个 decode/render 图
            total_loss = (comp_rgb * rgb_grad).sum() + gs_reg_loss  # scalar
            total_loss.backward()  # slat_feats_leaf.grad += gs guidance + reg 梯度
        else:
            comp_rgb.backward(rgb_grad)  # slat_feats_leaf.grad += gs 路梯度

        del comp_rgb, render_out, rgb_grad, gaussians
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        e.__traceback__ = None
        logging.warning(
            f"[Step {global_step}] {prefix}P2_gs OOM: {e} → 跳过 gs 梯度"
        )
        # ★ 释放 try 块中可能残留的 GPU 张量
        render_out = comp_rgb = comp_rgb_detached = rgb_grad = None
        gc.collect()
        torch.cuda.empty_cache()

    # ── ★ 梯度中继：把累积的多路梯度一次性回传到 proxy chain → proxy.grad ──
    # mesh OOM 时 slat_feats_leaf.grad 仅含 GS 梯度（仍然有效）
    profiler.tick(f"{prefix}P2_proxy_relay")
    try:
        if slat_feats_leaf.grad is not None:
            original_slat.feats.backward(slat_feats_leaf.grad)
        else:
            logging.warning(
                f"[Step {global_step}] slat_feats_leaf.grad is None → 跳过 proxy relay"
            )
    except torch.cuda.OutOfMemoryError as e:
        e.__traceback__ = None
        logging.warning(
            f"[Step {global_step}] {prefix}P2_proxy_relay OOM: {e} → 跳过梯度中继"
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ── P2→P3 过渡 ──
    state.features.slat = original_slat  # 恢复（Phase 3 需要 coords / layout）
    state.regularization.reg_loss = None
    clean_for_vjp(state)

    # ── Phase 3: VJP → θ.grad 累积（内部已有逐步 OOM 保护）──
    profiler.tick(f"{prefix}P3_vjp")
    phase3_log = ops.vjp_loop(state, system, tracker)

    profiler.tick(f"{prefix}end")

    return build_autograd_step_log(
        all_guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )


# =====================================================================
# FlowEdit 训练步：Pretrained Rollout + Finetuned 单步去噪
# =====================================================================

def trellis_flowedit_step(
    ops,
    state: TrellisState,
    system: TrellisSystem,
    global_step: int,
    profiler: PhaseProfiler,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    FlowEdit 训练步 — Rollout + Finetuned 单步去噪 + 2D FlowEdit Guidance。

    编排：
      P0 (ops.pre_rollout)
      → P1 (ops.rollout, pretrained 或 student, no_grad) → clean z₀
      → P2 (ops.add_noise) → zₜ
      → P3 (ops.predict_velocity_student) → v_student (有图)
           setup velocity proxy → v_proxy (leaf)
           ẑ₀ = zₜ - t·v_proxy → denormalize → update slat
      → P3.5 (可选, reg_weight > 0, reg_type ∈ {"v", "x0", "x1"})
           ops.predict_velocity_teacher → v_teacher (detached)
           reg_loss = reg_fn(v_proxy, v_teacher, t) → backward → reg_grad
      → P4a (decode/render, no_grad → detached comp_rgb)
      → P4b (guidance forward + backward → rgb_grad)
      → P4c (decode/render, 有梯度 + backward(rgb_grad) → v_proxy.grad)
      → P5 (relay: v_student.backward(v_proxy.grad + reg_grad) → θ.grad)

    ★ VelocityTracker 在 velocity 空间追踪 guidance 和 reg 梯度：
      - grad_norm/guidance: P4c backward 填充的 v_proxy.grad
      - grad_norm/reg:     P3.5 backward 填充的 reg_grad
      - loss/reg:          velocity reg loss (v/x0/x1)
      日志 key 与 RolloutTracker.collect_log 一致。

    ★ 保留 3-sub-step decode 显存优化：
      P4a no_grad decode → P4b guidance backward → P4c with-grad decode + backward
      显存峰值 = max(guidance, decode_render)

    Args:
        ops: TrellisFlowEditOps 实例
        state: 已 attach_batch 的 TrellisState
        system: 系统组件
        global_step: 全局步数
        profiler: PhaseProfiler
        prefix: profiler tick 和日志 key 的前缀

    Returns:
        日志字典
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()
    cfg = system.cfg
    device = system.accelerator.device

    # ── Phase 0: Dense Sampling ──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout (frozen, no_grad) → clean z₀ ──
    profiler.tick(f"{prefix}P1_rollout")
    ops.rollout(state, system, seed)
    # state.features.slat 现在是反归一化后的 clean z₀

    # ── Phase 2: 加噪 z₀ → zₜ ──
    profiler.tick(f"{prefix}P2_add_noise")
    z0_norm = ops.normalize_slat(state, system)  # (N, C), detached
    t_val = ops.sample_timestep(system)  # float, [0, 1]

    zt_feats = ops.add_noise(z0_norm.detach(), t_val)  # (N, C), detached

    # ── Phase 3: predict velocity + setup proxy ──
    profiler.tick(f"{prefix}P3_velocity")
    v_student = ops.predict_velocity_student(state, system, zt_feats, t_val)  # (N,C), 有图

    tracker = VelocityTracker()
    tracker.setup_proxy(v_student)  # v_proxy = v_student.detach().requires_grad_(True)

    # ẑ₀ = zₜ - t·v_proxy（梯度终止在 v_proxy leaf，P5 中继到 θ）
    z0_hat_norm = zt_feats - t_val * tracker.v_proxy  # (N, C)
    z0_hat_denorm = ops.denormalize_feats(z0_hat_norm, system)  # (N, C)
    state.features.slat = state.features.slat.replace(z0_hat_denorm)

    # ★ 清理 rollout 阶段累积的 spatial cache，为 decode 腾出显存
    state.features.slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── Phase 3.5: teacher velocity + reg backward（可选） ──
    reg_weight = float(cfg.train.loss.reg)
    reg_type = str(cfg.train.loss.reg_type)
    if reg_weight > 0:
        profiler.tick(f"{prefix}P3.5_reg")
        _phase3_5_velocity_reg(
            ops, state, system, tracker,
            zt_feats, t_val,
            reg_weight=reg_weight,
            reg_type=reg_type,
        )

    # ── Phase 4a: no_grad decode/render → detached comp_rgb ──
    profiler.tick(f"{prefix}P4a_decode_no_grad")
    with torch.no_grad():
        render_out = ops.decode_render_dict(state, system)
    comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)
    state.views_generated.image_tensor = comp_rgb_detached
    del render_out

    # ── Phase 4b: guidance-only backward → rgb_grad ──
    profiler.tick(f"{prefix}P4b_guidance_backward")
    guidance_cfg = ops.get_guidance_cfg(system)
    guidance_weight = ops.get_guidance_weight(system)

    rgb_grad, guidance_log = _phase2_guidance_only_backward(
        ops, state, system, comp_rgb_detached,
        guidance_cfg=guidance_cfg,
        guidance_weight=guidance_weight,
    )
    del comp_rgb_detached

    # ★ 清理 P4a decode 的 spatial cache，为 P4c 有梯度 decode 腾出显存
    state.features.slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── Phase 4c: with-grad decode/render + backward(rgb_grad) → v_proxy.grad ──
    profiler.tick(f"{prefix}P4c_decode_grad")
    render_out = ops.decode_render_dict(state, system)
    comp_rgb = render_out["color"]  # (B, V, H, W, C), autograd 图 → z0_hat → v_proxy
    gaussians = render_out.get("gaussians")

    # GS 正则化（可选）
    gs_reg_log: Dict[str, Any] = {}
    gs_reg_cfg = ops.get_gs_reg_config(system)
    has_gs_reg = gaussians is not None and (
        gs_reg_cfg["lambda_vol"] > 0 or gs_reg_cfg["lambda_opacity"] > 0
    )
    if has_gs_reg:
        gs_reg_loss, gs_reg_log = compute_gs_regularization(gaussians, **gs_reg_cfg)
        total_loss = (comp_rgb * rgb_grad).sum() + gs_reg_loss
        total_loss.backward()
    else:
        comp_rgb.backward(rgb_grad)
    # ★ v_proxy.grad 现在包含 guidance (+ gs_reg) 梯度

    del comp_rgb, render_out, rgb_grad, gaussians
    torch.cuda.empty_cache()

    # ── Phase 5: relay → θ.grad ──
    profiler.tick(f"{prefix}P5_relay")
    tracker.relay_and_backward()  # v_student.backward(v_proxy.grad + reg_grad) → θ.grad

    profiler.tick(f"{prefix}end")

    # ── 构建日志 ──
    log: Dict[str, Any] = {}
    log.update({f"{prefix}{k}": v for k, v in guidance_log.items()})
    log.update({f"{prefix}{k}": v for k, v in gs_reg_log.items()})
    # VelocityTracker 日志（grad_norm/guidance, grad_norm/reg, loss/reg, grad_norm/ratio）
    log.update({f"{prefix}{k}": v for k, v in tracker.collect_log(reg_weight=reg_weight).items()})
    log[f"{prefix}noise/t"] = t_val

    del tracker
    torch.cuda.empty_cache()
    return log
