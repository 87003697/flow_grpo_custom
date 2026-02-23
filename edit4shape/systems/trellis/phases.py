"""
Trellis 三阶段 Autograd 函数（纯计算步 + StageOps 编排）。

三阶段架构（按执行顺序）：
  Phase 0  dense_sampling_no_grad
           结构生成，无梯度
  Phase 1  phase1_rollout
           rollout_sparse(tracker=) no_grad 推理 + proxy chain + reg 梯度预计算
  Phase 2  phase2_decode_render_no_grad → phase2_guidance_backward → phase2_decode_render_grad
           decode/render 跑两遍：
             (1) no_grad decode/render → 生成图像（不保存 activations）
             (2) guidance forward + backward → rgb_grad → 释放 guidance 图
             (3) with-grad decode/render → backward(rgb_grad) → 填充 cond_proxy.grad → 释放 decode 图
           ★ 显存峰值从 decode_render + guidance（求和）降为 max(guidance, decode_render)（取最大值）
  Phase 3  phase3_rollout_grad_backward
           逐步重算 cond f_θ → VJP backward → θ.grad 累积（显存 O(1)）

编排函数：
  three_phase_step          — 旧版（直接传 cfg/device/accelerator）
  trellis_three_phase_step  — 新版（StageOps 参数化，对齐 Trellis2 架构）

设计参考：trellis2_distill 分支的 trellis2_shape_autograd_async.py
"""

from __future__ import annotations

import torch
import ml_collections
from typing import Any, Callable, Dict, Optional, Tuple

from accelerate import Accelerator

from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import (
    rollout_sparse,
    RolloutTracker,
    _predict_cond_velocity,
)
from edit4shape.systems.trellis.system import TrellisSystem
from edit4shape.systems.utils.profiler import PhaseProfiler
from edit4shape.systems.trellis.forward import (
    decode_and_render_mesh,
    decode_and_render_gs,
)
from edit4shape.systems.utils.logging import build_autograd_step_log

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Phase 0: Dense Sampling
# =====================================================================

def dense_sampling_no_grad(
    state: TrellisState,
    system: TrellisSystem,
    device: torch.device,
) -> None:
    """
    Phase 0: Dense Sampling（结构生成），无梯度。

    Side Effects:
        - state.coords: 挂载稀疏坐标 (N, 4)
    """
    pipeline = system.pipeline
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_embed,
            "neg_cond": state.views_conditioned.uncond_embed,
        }
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N, 4)
    state.coords = coords  # (N, 4)


# =====================================================================
# Phase 1: Rollout
# =====================================================================

def phase1_rollout(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
) -> RolloutTracker:
    """
    Phase 1: Rollout（no_grad 推理 + proxy chain + reg 梯度预计算）。

    - 模型推理在 no_grad 下执行（is_training=False）
    - cond_pred 上插入 proxy 节点构建轻量计算图
    - reg 梯度通过 autograd.grad 提前算好存入 tracker

    Args:
        state: TrellisState，需要 state.coords 已挂载
        system: 系统组件
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数（用于随机种子）

    Returns:
        tracker: 填充完成的 RolloutTracker

    Side Effects:
        - state.features.slat: 挂载反归一化后的 SparseTensor
        - state.regularization.reg_loss: 挂载 reg_loss（含 proxy chain 图）
    """
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    tracker = RolloutTracker()

    # ⚠️ 不可包裹 torch.no_grad()：rollout_sparse 内部 is_training=False
    #    已用 no_grad 做模型推理，但 tracker 的 proxy 需要 autograd 图
    #    （scheduler 用 proxy 推进 → slat 依赖 proxy chain）
    rollout_sparse(
        state, cfg, system, device,
        generator=generator,
        is_training=False,   # 模型推理 no_grad
        tracker=tracker,     # ★ 记录 proxy 轨迹
    )

    torch.cuda.empty_cache()
    return tracker


# =====================================================================
# Phase 2: Decode / Render / Guidance
# =====================================================================

def phase2_decode_render_no_grad(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
) -> torch.Tensor:
    """
    Phase 2 step 1: no_grad decode/render → 生成图像。

    ★ 显存优化：decode/render 在 no_grad 下执行，不保存 activations。
    输出是 detached 的渲染图像，仅用于后续 guidance 和可视化。

    Args:
        state: TrellisState，需要 state.features.slat 已挂载
        system: 系统组件
        cfg: 配置对象
        device: 运行设备

    Returns:
        comp_rgb_detached: (B, V, H, W, C) 渲染图像（无梯度）

    Side Effects:
        - state.views_generated.image_tensor: 挂载渲染图像（detached，仅可视化）
    """
    with torch.no_grad():
        latents = state.features.slat  # SparseTensor（依赖 proxy chain）
        renderer_type = cfg.renderer.type
        renderer = system.renderers[renderer_type]  # 从 renderers dict 查找
        if renderer_type == "gs":
            render_out = decode_and_render_gs(
                latents, state.cameras, system.pipeline, renderer, device
            )  # dict with "color": (B, V, H, W, C)
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, renderer, device
            )  # dict with "color"/"normal": (B, V, H, W, C)
            render_out["color"] = render_out["normal"]

    comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)
    state.views_generated.image_tensor = comp_rgb_detached  # 挂载可视化用
    del render_out

    return comp_rgb_detached


def phase2_guidance_backward(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    comp_rgb_detached: torch.Tensor,
    device: torch.device,
    accelerator: Accelerator,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Phase 2 step 2: guidance forward + backward → rgb_grad。

    在 detached 图像上创建叶节点 proxy，guidance backward 只产生 guidance 图，
    不含 decode/render activations。backward 完成后释放整个 guidance 图。

    ★ 只 backward guidance loss，不混入 reg_loss。
      reg 梯度已在 Phase 1 通过 autograd.grad 预计算存入 tracker.reg_grads，
      Phase 3 合并 guidance + reg 梯度后做 VJP。

    显存峰值 ≈ guidance_activations（decode/render 贡献 = 0）。

    Args:
        state: TrellisState
        system: 系统组件
        cfg: 配置对象
        comp_rgb_detached: (B, V, H, W, C) Phase 2 step 1 产出的无梯度图像
        device: 运行设备
        accelerator: Accelerate 加速器

    Returns:
        rgb_grad: (B, V, H, W, C) guidance 对渲染图像的梯度
        logs: 训练日志字典

    Side Effects:
        - state.guidance.*: 挂载 guidance 结果
    """
    # ---- 创建 proxy 叶节点（guidance 梯度的终点）----
    comp_rgb_proxy = comp_rgb_detached.detach().requires_grad_(True)  # (B, V, H, W, C), leaf

    # ---- guidance forward ----
    guidance_result = system.guidance.compute_guidance(
        comp_rgb_proxy,
        state.views_conditioned.image_pils,
        guidance_cfg=cfg.train.guidance,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)

    # ---- guidance backward → rgb_grad ----
    guidance_loss = state.guidance.loss.to(device) * cfg.train.loss.guidance  # ()
    accelerator.backward(guidance_loss)

    rgb_grad = comp_rgb_proxy.grad.detach().clone()  # (B, V, H, W, C)

    # ---- 构建日志 ----
    logs: Dict[str, Any] = {}
    if state.guidance.loss_dict:
        logs.update({
            f"loss/{k}": v.item()
            for k, v in state.guidance.loss_dict.items()
            if v is not None
        })
    logs["loss/guidance"] = guidance_loss.item()

    # ---- 释放 guidance 计算图 ----
    del comp_rgb_proxy, guidance_loss, guidance_result
    torch.cuda.empty_cache()

    return rgb_grad, logs


def phase2_decode_render_grad(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    rgb_grad: torch.Tensor,
) -> None:
    """
    Phase 2 step 3: with-grad decode/render → backward(rgb_grad) → cond_proxy.grad。

    ★ 显存优化：此时 guidance 计算图已释放，
    只有 decode/render 的 activations 驻留在显存中。
    backward 完成后立即释放 decode/render 图。

    显存峰值 ≈ decode_render_activations（guidance 贡献 = 0）。

    Args:
        state: TrellisState，需要 state.features.slat 已挂载
        system: 系统组件
        cfg: 配置对象
        device: 运行设备
        rgb_grad: (B, V, H, W, C) Phase 2 step 2 产出的梯度

    Side Effects:
        - tracker.output_trajectory[i].grad: 被 proxy chain 反传填充（纯 guidance 梯度）
    """
    latents = state.features.slat  # SparseTensor（依赖 proxy chain）

    renderer_type = cfg.renderer.type
    renderer = system.renderers[renderer_type]  # 从 renderers dict 查找
    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color": (B, V, H, W, C)
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, renderer, device
        )  # dict with "color"/"normal": (B, V, H, W, C)
        render_out["color"] = render_out["normal"]

    comp_rgb = render_out["color"]  # (B, V, H, W, C)

    # ---- backward: rgb_grad → render → decode → slat → proxy chain → cond_proxy.grad ----
    comp_rgb.backward(rgb_grad)  # cond_proxy.grad 仅含 guidance 分量

    # ---- 释放 decode/render 计算图 ----
    del comp_rgb, render_out, rgb_grad
    state.regularization.reg_loss = None  # 不再需要 reg_loss tensor
    torch.cuda.empty_cache()


# =====================================================================
# Phase 3: VJP Backward
# =====================================================================

def phase3_rollout_grad_backward(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    tracker: RolloutTracker,
) -> None:
    """
    Phase 3: 逐步重算 cond f_θ → VJP backward → θ.grad 累积。

    显存 O(1)：每步只保留单次 f_θ 的计算图，VJP 后立即释放。

    - v_grad 来自 tracker.output_trajectory[i].grad（Phase 2 backward 填充，仅含 guidance 梯度）
    - reg_grads 来自 tracker.reg_grads（Phase 1 autograd.grad 预计算）
    - Phase 3 合并两者: v_grad = guidance_grad + reg_weight * reg_grad
    - 仅重算条件预测 f_θ(x_t, t)，无需 uncond / CFG / reg
    - VJP: (v_grad * cond_pred.feats).sum().backward()

    Args:
        state: TrellisState
        system: 系统组件
        cfg: 配置对象
        device: 运行设备
        tracker: Phase 1 填充的 RolloutTracker
    """
    pipeline = system.pipeline
    cond_emb, _ = state.extract_embeddings()
    cond_emb = cond_emb.to(device)  # (B, S, C)
    B = cond_emb.shape[0]  # ()

    T = len(tracker.timesteps)
    has_reg_grads = len(tracker.reg_grads) == T
    reg_weight = cfg.train.loss.reg if has_reg_grads else 0.0

    for i in range(T):
        t_val = tracker.timesteps[i]
        x_t_feats = tracker.input_trajectory[i]  # (N, C)，纯数据

        # 用 state.features.slat 的 coords 重建 SparseTensor（x_t_feats 无梯度）
        x_t = SparseTensor(
            feats=x_t_feats,
            coords=state.features.slat.coords,
        )  # SparseTensor(feats: (N, C), coords: (N, 4))

        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)

        # ★ 重算条件预测（有 θ 梯度，x_t 在 _predict_cond_velocity 内部 detach）
        cond_pred = _predict_cond_velocity(
            pipeline, x_t, t_batch, cond_emb
        )  # SparseTensor(feats: (N, C))

        # ---- 合并 guidance + reg 梯度 ----
        v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
        if v_grad is None:
            continue

        if has_reg_grads:
            v_grad = v_grad + reg_weight * tracker.reg_grads[i]  # (N, C)

        # ---- VJP：(v_grad * cond_pred.feats).sum().backward() ----
        # 图仅包含本次 f_θ 调用，backward 后立即释放
        (v_grad * cond_pred.feats).sum().backward()  # θ.grad += ...

    torch.cuda.empty_cache()


# =====================================================================
# 编排函数 - 三阶段训练步
# =====================================================================

def three_phase_step(
    state: TrellisState,
    system: TrellisSystem,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    accelerator: Accelerator,
    profiler: Optional[PhaseProfiler] = None,
) -> Dict[str, Any]:
    """
    三阶段训练步编排：Phase 0 → 1 → 2(no_grad → guidance → grad) → 3。

    与原始端到端 backward 数学等价，但显存峰值大幅降低。
    Phase 2 分三步：先 no_grad decode/render，再 guidance backward → rgb_grad，
    最后 with-grad decode/render backward(rgb_grad) → cond_proxy.grad。

    显存峰值从 decode_render + guidance（求和）降为 max(guidance, decode_render)。

    Args:
        state: TrellisState（已挂载 batch 数据）
        system: 系统组件
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        accelerator: Accelerate 加速器
        profiler: 可选的 PhaseProfiler

    Returns:
        logs: 训练日志字典
    """
    if profiler:
        profiler.tick("P0_dense_sampling")

    # ---- Phase 0: Dense Sampling (no_grad) ----
    dense_sampling_no_grad(state, system, device)

    if profiler:
        profiler.tick("P1_rollout")

    # ---- Phase 1: Rollout (no_grad + proxy chain) ----
    tracker = phase1_rollout(state, system, cfg, device, global_step)

    if profiler:
        profiler.tick("P2_decode_no_grad")

    # ---- Phase 2 step 1: Decode/Render (no_grad) ----
    comp_rgb_detached = phase2_decode_render_no_grad(state, system, cfg, device)

    if profiler:
        profiler.tick("P2_guidance_bw")

    # ---- Phase 2 step 2: Guidance backward → rgb_grad ----
    rgb_grad, logs = phase2_guidance_backward(
        state, system, cfg, comp_rgb_detached, device, accelerator,
    )
    del comp_rgb_detached

    if profiler:
        profiler.tick("P2_decode_grad")

    # ---- Phase 2 step 3: Decode/Render (with-grad) backward(rgb_grad) ----
    phase2_decode_render_grad(state, system, cfg, device, rgb_grad)

    if profiler:
        profiler.tick("P3_rollout_grad_bw")

    # ---- Phase 3: VJP Backward ----
    phase3_rollout_grad_backward(state, system, cfg, device, tracker)

    if profiler:
        profiler.tick("end")

    # ---- 补充 reg 日志 ----
    if tracker.reg_loss_val is not None:
        logs["loss/reg"] = tracker.reg_loss_val

    return logs


# =====================================================================
# StageOps 版编排 — 3-sub-step Phase 2 + 多态渲染
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
        guidance_cfg=ops.get_guidance_cfg(system),
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

    # ── Phase 2c: with-grad decode/render + backward(rgb_grad) ──
    profiler.tick(f"{prefix}P2c_decode_grad")
    render_out = ops.decode_render_dict(state, system)
    comp_rgb = render_out["color"]  # (B, V, H, W, C), has autograd graph
    comp_rgb.backward(rgb_grad)  # → cond_proxy.grad（仅 guidance 梯度）

    del comp_rgb, render_out, rgb_grad
    state.regularization.reg_loss = None
    torch.cuda.empty_cache()

    # ── P2→P3 过渡：调用方注入的清理策略 ──
    clean_for_vjp(state)

    # ── Phase 3: VJP (ops 内部合并 reg_grads) → θ.grad 累积 ──
    profiler.tick(f"{prefix}P3_vjp")
    phase3_log = ops.vjp_loop(state, system, tracker)

    profiler.tick(f"{prefix}end")

    return build_autograd_step_log(
        guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )


# =====================================================================
# Hybrid 双路渲染编排 — P2 循环 mesh + gs
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
      → 循环每路渲染 {
            P2a (ops.decode_render_dict(renderer_key=key) no_grad)
            → P2b (guidance-only backward → rgb_grad，使用各路独立 cfg/weight)
            → P2c (ops.decode_render_dict(renderer_key=key) with grad + backward(rgb_grad))
        }
      → clean_for_vjp → P3 (ops.vjp_loop with reg_grads merge) → 返回日志

    显存峰值 = max(guidance, decode_render)（与单路相同，每路 P2c 结束即释放）。
    多路梯度通过 proxy.grad += 自动累加。

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

    # ── Phase 2: 循环处理每路渲染 ──
    all_guidance_log: Dict[str, Any] = {}
    render_passes = ops.get_render_passes(system)

    for pass_idx, (key, guid_cfg, guid_weight) in enumerate(render_passes):
        # ── P2a: no_grad decode/render ──
        profiler.tick(f"{prefix}P2a_{key}")
        with torch.no_grad():
            render_out = ops.decode_render_dict(state, system, renderer_key=key)
        comp_rgb_detached = render_out["color"].detach()  # (B, V, H, W, C)

        # 按渲染器类型分别挂载可视化
        if key == "mesh":
            state.views_generated.normal_tensor = comp_rgb_detached  # (B, V, H, W, C) Mesh Normal
        else:
            state.views_generated.image_tensor = comp_rgb_detached   # (B, V, H, W, C) GS Color
        del render_out

        # ── P2b: guidance-only backward → rgb_grad（按渲染器类型分发）──
        profiler.tick(f"{prefix}P2b_{key}")
        if key == "mesh":
            rgb_grad, guidance_log = _phase2_normal_guidance_backward(
                ops, state, system, comp_rgb_detached, guid_cfg, guid_weight,
            )
        else:
            rgb_grad, guidance_log = _phase2_color_guidance_backward(
                ops, state, system, comp_rgb_detached, guid_cfg, guid_weight,
            )
        # 为各路日志添加前缀
        all_guidance_log.update({
            f"{key}/{k}": v for k, v in guidance_log.items()
        })
        del comp_rgb_detached

        # ── P2c: with-grad decode/render + backward(rgb_grad) ──
        profiler.tick(f"{prefix}P2c_{key}")
        render_out = ops.decode_render_dict(state, system, renderer_key=key)
        comp_rgb = render_out["color"]  # (B, V, H, W, C), has autograd graph
        comp_rgb.backward(rgb_grad)  # proxy.grad += 本路梯度（多路自动累加）

        del comp_rgb, render_out, rgb_grad
        torch.cuda.empty_cache()

    # ── P2→P3 过渡 ──
    state.regularization.reg_loss = None
    clean_for_vjp(state)

    # ── Phase 3: VJP (ops 内部合并 reg_grads) → θ.grad 累积 ──
    profiler.tick(f"{prefix}P3_vjp")
    phase3_log = ops.vjp_loop(state, system, tracker)

    profiler.tick(f"{prefix}end")

    return build_autograd_step_log(
        all_guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )
