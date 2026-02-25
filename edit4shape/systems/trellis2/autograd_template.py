"""
三阶段 Autograd 通用模板 — 参数化的同步训练步。

将三阶段训练步（P0 + P1 + P2a/P2 + P3）的公共骨架提取为通用函数，
通过 StageOps 参数化阶段特有的计算，通过 clean_for_vjp 回调参数化清理策略。

本模块不依赖任何特定模型后端，仅通过 StageOps 协议与 state/system 的
duck-typed 接口通信。任何实现了 StageOps 的模型后端均可使用。

使用方式::

    from edit4shape.systems.trellis2.autograd_template import three_phase_step
    from my_model.stage_ops import MyShapeOps

    three_phase_step(MyShapeOps(), state, system, ...,
        clean_for_vjp=lambda s: s.prepare_for_vjp())

state / system 隐含协议：
    state.views_conditioned.image_pils  — 条件图像列表
    state.attach_guidance_result(result) — 挂载 guidance 结果
    state.regularization.reg_loss       — reg loss tensor (Optional)
    system.accelerator                  — Accelerate 加速器
    system.guidance.compute_guidance()  — 同步 guidance 前向
    system.cfg.seed                     — 全局种子
"""

import logging
from typing import Any, Callable, Dict

import torch

from edit4shape.systems.utils.stage_ops import StageOps, StageSkipError
from edit4shape.systems.utils.logging import build_autograd_step_log


# =====================================================================
# Phase 2: 通用 Guidance + Backward
# =====================================================================

def _phase2_guidance_and_backward(
    ops: StageOps,
    state,
    system,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    通用 Phase 2（同步版）: guidance + reg 合并 backward → 填充 tracker 梯度 → 释放图。

    ★ 通过 ops 获取 guidance_weight / reg_weight / guidance_cfg，
      消除 shape/tex 两套 phase2_guidance_and_backward 的重复。

    数学：
      total_loss = guidance_loss * w_guid + reg_loss * w_reg
      accelerator.backward(total_loss)
      → output_trajectory[t].grad = ∂total_loss/∂cond_proxy_t

    Args:
        ops: 阶段 Ops（提供 guidance 配置）
        state: 训练状态（含 views_conditioned, regularization 等）
        system: 训练系统（含 guidance, accelerator）
        comp_rgb: Phase 2a 产出的渲染图 (B, V, H, W, 3)（有 autograd 图）

    Returns:
        guidance 日志字典（含 loss/guidance, loss/reg 等）
    """
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = ops.get_guidance_weight(system)
    reg_weight = ops.get_reg_weight(system)

    # 1. Guidance 前向（同步阻塞）
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=ops.get_guidance_cfg(system),
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)

    # 2. 合并 loss: guidance + reg
    # comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    # reg_loss ← MSE/velocity ← CFG ← cond_proxy
    # → 两路梯度汇聚到 cond_proxy.grad
    total_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    reg_loss = state.regularization.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()

    # 3. Backward（一路反传到 output_trajectory[t].grad）
    accelerator.backward(total_loss)

    # 4. 构建日志
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = (
        guidance_result.loss.to(device) * guidance_weight
    ).item()
    if reg_loss is not None:
        guidance_log["loss/reg"] = reg_loss.item()

    # 5. 释放所有计算图引用
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None
    torch.cuda.empty_cache()

    return guidance_log


# =====================================================================
# 通用三阶段训练步
# =====================================================================

def three_phase_step(
    ops: StageOps,
    state,
    system,
    global_step: int,
    profiler,
    clean_for_vjp: Callable,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    通用三阶段 Autograd 训练步。

    编排：
      P0 (pre_rollout) → P1 (rollout + tracker)
      → P2a (decode + render) → P2 (guidance + backward)
      → clean_for_vjp → P3 (VJP) → 返回日志

    Args:
        ops: 阶段特有的计算操作（任何 StageOps 实现）
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler（tick / collect 接口）
        clean_for_vjp: P2→P3 过渡清理函数（由调用方注入，编码清理策略）
        prefix: profiler tick 和日志 key 的前缀（如 "shape/" 或 "tex/"）

    Returns:
        合并的日志字典（不含 profiler 计时——由调用方决定是否收集）
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()

    # ── Phase 0: 准备（dense_sampling / shape_frozen_prepare / no-op）──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout → proxy chain + tracker ──
    profiler.tick(f"{prefix}P1_rollout")
    tracker = ops.rollout(state, system, seed)

    # ── Phase 2a + Phase 2: Decode/Render + Guidance Backward ──
    profiler.tick(f"{prefix}P2a_decode_render")
    comp_rgb = None
    skip_phase3 = False
    guidance_log: Dict[str, Any] = {}
    try:
        comp_rgb = ops.decode_render(state, system)

        profiler.tick(f"{prefix}P2_guidance_backward")
        guidance_log = _phase2_guidance_and_backward(ops, state, system, comp_rgb)
    except (torch.cuda.OutOfMemoryError, StageSkipError) as e:
        # P2a OOM / decode 前置条件不满足 → 跳过 P3
        # ★ 不做 reg-only VJP：超大样本的 VJP 可能耗时过长，导致 NCCL timeout。
        # 安全性：P2a/P2 不经过模型参数，不触发 DDP hooks，不会导致分布式死锁。
        logging.warning(
            f"[Step {global_step}] {prefix}P2a/P2 failed: {e} → 跳过 P3"
        )
        skip_phase3 = True
        del comp_rgb
        torch.cuda.empty_cache()

    # ── P2→P3 过渡：调用方注入的清理策略 ──
    clean_for_vjp(state)

    # ── Phase 3: VJP → θ.grad 累积 ──
    if not skip_phase3:
        profiler.tick(f"{prefix}P3_grad_backward")
        phase3_log = ops.vjp_loop(state, system, tracker)
    else:
        profiler.tick(f"{prefix}P3_skip")
        # 仅清理 tracker 数据，不执行 VJP
        del tracker.input_trajectory[:], tracker.output_trajectory[:]
        del tracker.timesteps[:]
        torch.cuda.empty_cache()
        phase3_log = {}

    profiler.tick(f"{prefix}end")

    # 合并日志（不含 profiler 计时——由调用方按需添加）
    return build_autograd_step_log(
        guidance_log, ops.get_reg_weight(system), phase3_log, prefix=prefix,
    )
