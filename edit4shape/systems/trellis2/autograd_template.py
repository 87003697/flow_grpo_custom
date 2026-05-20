"""
Trellis2 Autograd 通用模板 — 参数化的同步训练步。

提供两种编排模板：
  ● three_phase_step  — VJP 三阶段训练步（P0+P1+P2+P3）
  ● onestep_step      — Onestep 单步去噪训练步（Pretrained Rollout + CFG Velocity + 3-sub-step Decode）

通过 StageOps (Trellis2StageOps) 参数化阶段特有的计算，
通过回调参数化清理策略。

使用方式::

    from edit4shape.systems.trellis2.autograd_template import three_phase_step, onestep_step
    from edit4shape.systems.trellis2.stage_ops import Trellis2ShapeOps

    # VJP 模式
    three_phase_step(Trellis2ShapeOps(), state, system, ...,
        clean_for_vjp=lambda s: s.prepare_for_vjp())

    # Onestep 模式
    onestep_step(Trellis2ShapeOps(), state, system, ...)

state / system 隐含协议：
    state.views_conditioned.image_pils  — 条件图像列表
    state.attach_guidance_result(result) — 挂载 guidance 结果
    state.{shape|tex}.reg_loss           — reg loss tensor (Optional)
    system.accelerator                  — Accelerate 加速器
    system.guidance.compute_guidance()  — 同步 guidance 前向
    system.cfg.seed                     — 全局种子
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from edit4shape.systems.utils.stage_ops import StageOps, StageSkipError
from edit4shape.systems.utils.logging import build_autograd_step_log
from edit4shape.generators.trellis2.rollout import VelocityTracker


# =====================================================================
# Phase 3.5: Velocity Regularization Backward（Onestep 专用）
# =====================================================================

def _phase3_5_velocity_reg(
    ops,
    state,
    system,
    tracker: VelocityTracker,
    zt_feats: torch.Tensor,
    t_val: float,
    reg_weight: float,
    reg_type: str = "v",
) -> None:
    """
    Phase 3.5: teacher velocity 预测 + 正则化 backward → tracker.reg_grad。

    支持三种正则化类型（与 trellis v1 对齐）：
      - "v":  MSE(v_proxy, v_teacher)
      - "x0": MSE(x0_stu, x0_tea) / (t² + ε)，单步下与 v reg 数学等价
      - "x1": MSE(x0_stu, x0_tea)，不除以 t²，小 t 时正则化更弱

    完成后 tracker 中写入：
      - reg_grad:     (N, C) detached 正则化梯度
      - reg_loss_val: float 标量值（日志用）
      - v_proxy.grad: 被清零（留给 P4c guidance 梯度）

    Args:
        ops: Trellis2StageOps 实例
        state: 已 attach_batch 的状态
        system: 训练系统
        tracker: VelocityTracker（已 setup_proxy）
        zt_feats: (N, C) 加噪后的特征，detached
        t_val: float, 归一化时间步 [0, 1]
        reg_weight: 正则化权重（> 0）
        reg_type: 正则化类型，"v" | "x0" | "x1"
    """
    from edit4shape.generators.trellis2.rollout.base import (
        _compute_x0_regularization,
        _compute_x1_regularization,
        _compute_v_regularization,
    )

    v_teacher = ops.predict_cfg_velocity_teacher(
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
      total_loss.backward()
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
    stage_latent = getattr(state, ops.get_stage_name())
    reg_loss = stage_latent.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()

    # 3. Backward（一路反传到 output_trajectory[t].grad）
    # ★ 直接 backward，不经过 accelerator.backward（不做梯度预缩放），
    #   由 sync_grads_and_step(n_accumulated) 统一做 /n 平均。
    total_loss.backward()

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
    stage_latent.reg_loss = None
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

    # ── Phase 3: VJP → θ.grad 累积（vjp_loop 内部 model.no_sync()，不触发 DDP all-reduce）──
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


# =====================================================================
# Phase 4b: Guidance-Only Backward → rgb_grad（3-sub-step 用）
# =====================================================================

def _phase_guidance_only_backward(
    ops: StageOps,
    state,
    system,
    comp_rgb_detached: torch.Tensor,
    guidance_cfg=None,
    guidance_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    3-sub-step Phase 4b: guidance-only backward → rgb_grad。

    ★ 不合并 reg_loss（reg 梯度在 velocity 空间独立处理）。

    参数来源（优先级从高到低）：
      1. 显式传入 guidance_cfg / guidance_weight
      2. 通过 ops 查询 get_guidance_cfg / get_guidance_weight

    Args:
        ops: StageOps 实例
        state: 训练状态
        system: 训练系统
        comp_rgb_detached: (B, V, H, W, C) 无梯度图像
        guidance_cfg: 显式 guidance 配置（可选）
        guidance_weight: 显式 guidance 权重（可选）

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
    # ★ 直接 backward，不经过 accelerator.backward（不做梯度预缩放）。
    guidance_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    guidance_loss.backward()

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
# 公共工具：手动 all-reduce + NaN guard + grad clip + optimizer step
# =====================================================================

def sync_grads_and_step(
    model: torch.nn.Module,
    optimizer,
    grad_clipper,
    n_accumulated: int = 1,
) -> None:
    """
    手动 all-reduce 梯度 → 除以累积数 → NaN 拦截 → grad clip → step → zero_grad。

    VJP / Onestep 训练步内的 backward 都在 model.no_sync() 下执行，
    不触发 DDP 自动 all-reduce。因此需要在 optimizer.step() 前
    手动做一次跨 rank 梯度同步，并除以实际累积的 micro-batch 数。

    ★ 模板内的 backward 均使用直接 backward()（不经过 accelerator.backward），
      不做梯度预缩放。对齐异步版本的 _sync_grads_and_step(n_accumulated) 语义。

    Args:
        model: DDP 包装的模型
        optimizer: 对应的 optimizer
        grad_clipper: AdaptiveGradClipper（或任何接受 parameters() 的 callable）
        n_accumulated: 本次 step 实际累积的 micro-batch 数（尾部可能 < accum_steps）
    """
    is_distributed = dist.is_initialized()
    has_nan = False
    for p in model.parameters():
        if p.grad is None:
            continue
        if is_distributed:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if n_accumulated > 1:
            p.grad.div_(n_accumulated)
        if not has_nan and not torch.isfinite(p.grad).all():
            has_nan = True
    if has_nan:
        logging.warning("[NaN Guard] 检测到 NaN/Inf 梯度，跳过本次 optimizer step")
        optimizer.zero_grad()
        return
    grad_clipper(model.parameters())
    optimizer.step()
    optimizer.zero_grad()


# =====================================================================
# Onestep 训练步：Rollout + 单步去噪 + 3-sub-step Decode
# =====================================================================

def onestep_step(
    ops,
    state,
    system,
    global_step: int,
    profiler,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Onestep 训练步 — Rollout + Finetuned 单步去噪 + 2D FlowEdit Guidance。

    编排：
      P0 (ops.pre_rollout)
      → P1 (ops.pretrained_rollout, pretrained 或 student, no_grad) → clean z₀
      → P2 (ops.add_noise) → zₜ
      → P3 (ops.predict_cfg_velocity) → v_student (有图到 θ)
           setup VelocityTracker proxy → v_proxy (leaf)
           ẑ₀ = zₜ - t·v_proxy → denormalize → update slat
      → P3.5 (可选, reg_weight > 0, reg_type ∈ {"v", "x0", "x1"})
           ops.predict_cfg_velocity_teacher → v_teacher (detached)
           reg_loss = reg_fn(v_proxy, v_teacher, t) → backward → reg_grad
      → P4a (ops.decode_render_dict, no_grad → detached comp_rgb)
      → P4b (guidance forward + backward → rgb_grad)
      → P4c (ops.decode_render_dict, 有梯度 + backward(rgb_grad) → v_proxy.grad)
      → P5 (relay: v_student.backward(v_proxy.grad + reg_grad) → θ.grad)

    ★ 可配置 Rollout 模式（cfg.{stage}.train.rollout_mode）：
      - "pretrained"：teacher_context()，使用 pretrained 权重（off-policy）
      - "student"：直接使用当前 finetuned 权重（on-policy）

    ★ Student Denoise CFG 开关（cfg.{stage}.train.student_denoise_cfg）：
      - True：P3/P3.5 保持 cond+uncond 双 forward（CFG 增强）
      - False：临时置空 uncond_embed，只做 cond forward（省约 50% P3/P3.5 计算量）

    ★ VelocityTracker 在 velocity 空间追踪 guidance 和 reg 梯度：
      - grad_norm/guidance: P4c backward 填充的 v_proxy.grad
      - grad_norm/reg:     P3.5 backward 填充的 reg_grad
      - loss/reg:          velocity reg loss (v/x0/x1)

    ★ 保留 3-sub-step decode 显存优化：
      P4a no_grad decode → P4b guidance backward → P4c with-grad decode + backward
      显存峰值 = max(guidance, decode_render)

    DDP 安全：
      P5 relay backward 在 model.no_sync() 下执行，不触发 DDP all-reduce。
      P4c OOM 时安全降级（跳过 P5），不会导致分布式死锁。
      梯度同步由 entry 层的 sync_grads_and_step() 在 optimizer.step 前手动完成。

    Args:
        ops: Trellis2StageOps 实例（Trellis2ShapeOps / Trellis2TexOps）
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler（tick / collect 接口）
        prefix: profiler tick 和日志 key 的前缀

    Returns:
        日志字典（不含 profiler 计时——由调用方决定是否收集）
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()
    device = system.accelerator.device
    stage_name = ops.get_stage_name()
    model = ops.get_model(system)

    # ── Phase 0: 准备（dense_sampling / shape_frozen_prepare / no-op）──
    profiler.tick(f"{prefix}P0_pre_rollout")
    ops.pre_rollout(state, system, global_step)

    # ── Phase 1: Rollout (pretrained 或 student, no_grad) → clean z₀ ──
    profiler.tick(f"{prefix}P1_rollout")
    ops.pretrained_rollout(state, system, seed)
    # state.shape.z0 / state.tex.z0 现在是反归一化后的 clean z₀

    # ── Phase 2: 加噪 z₀ → zₜ ──
    profiler.tick(f"{prefix}P2_add_noise")
    z0_norm = ops.normalize_slat(ops.get_latent(state), system)  # SparseTensor → (N, C) normalized
    z0_feats = z0_norm.feats.detach()  # (N, C), detached
    t_val = ops.sample_timestep(system)  # float, [0, 1]
    zt_feats = ops.add_noise(z0_feats, t_val)  # (N, C), detached

    # ── Phase 3: predict CFG velocity (student, with grad) + setup proxy ──
    profiler.tick(f"{prefix}P3_velocity")
    student_denoise_cfg = ops.get_student_denoise_cfg(system)  # True = 用 CFG, False = 跳过 uncond forward
    with state.disable_uncond_embeddings(not student_denoise_cfg):
        v_student = ops.predict_cfg_velocity(state, system, zt_feats, t_val)  # (N,C), 有图

    tracker = VelocityTracker()
    tracker.setup_proxy(v_student)  # v_proxy = v_student.detach().requires_grad_(True)

    # ẑ₀ = zₜ - t·v_proxy（梯度终止在 v_proxy leaf，P5 中继到 θ）
    z0_hat_norm = zt_feats - t_val * tracker.v_proxy  # (N, C)
    z0_hat_denorm = ops.denormalize_slat(
        ops.get_latent(state).replace(z0_hat_norm), system
    )  # SparseTensor in denormalized domain

    # 更新 state 中的 slat
    slat = ops.get_latent(state)
    new_slat = slat.replace(z0_hat_denorm.feats)
    if stage_name == "shape":
        state.shape.z0 = new_slat
    else:
        state.tex.z0 = new_slat

    # ★ 清理 rollout 阶段累积的 spatial cache，为 decode 腾出显存
    new_slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── Phase 3.5: teacher velocity + reg backward（可选） ──
    reg_weight = ops.get_reg_weight(system)
    reg_type = ops.get_reg_type(system)
    if reg_weight > 0:
        profiler.tick(f"{prefix}P3.5_reg")
        with state.disable_uncond_embeddings(not student_denoise_cfg):
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
    # 挂载 vis + 保存 decode 产物
    if stage_name == "shape":
        state.views_generated.shape_tensor = comp_rgb_detached
        # ★ 保存 subs/meshes（Shape decode 产物，Tex P4a/P4c decode 需要）
        # 对齐 VJP 版本的 ops.decode_render() 行为
        state.shape.subs = render_out.get("subs")
        state.shape.meshes = render_out.get("meshes")
    else:
        state.views_generated.pbr_tensor = comp_rgb_detached
    del render_out

    # ── Phase 4b: guidance-only backward → rgb_grad ──
    profiler.tick(f"{prefix}P4b_guidance_backward")
    rgb_grad, guidance_log = _phase_guidance_only_backward(
        ops, state, system, comp_rgb_detached,
    )
    del comp_rgb_detached

    # ★ 清理 P4a decode 的 spatial cache，为 P4c 有梯度 decode 腾出显存
    new_slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── Phase 4c: with-grad decode/render + backward(rgb_grad) → v_proxy.grad ──
    # ★ OOM 保护：P4c 不经过 flow model 参数（图到 v_proxy leaf），不触发 DDP hooks，
    #   OOM 时安全跳过 P5 relay，不会导致分布式死锁。
    profiler.tick(f"{prefix}P4c_decode_grad")
    skip_relay = False
    try:
        render_out = ops.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, C), autograd 图 → z0_hat → v_proxy
        comp_rgb.backward(rgb_grad)  # → v_proxy.grad = guidance_grad
        del comp_rgb, render_out, rgb_grad
    except torch.cuda.OutOfMemoryError as e:
        logging.warning(
            f"[Step {global_step}] {prefix}P4c OOM: {e} → 跳过 P5 relay"
        )
        skip_relay = True
        del rgb_grad
    torch.cuda.empty_cache()

    # ── Phase 5: relay → θ.grad（no_sync，不触发 DDP all-reduce）──
    if not skip_relay:
        profiler.tick(f"{prefix}P5_relay")
        with model.no_sync():
            tracker.relay_and_backward()  # v_student.backward(v_proxy.grad + reg_grad) → θ.grad
    else:
        profiler.tick(f"{prefix}P5_skip")

    profiler.tick(f"{prefix}end")

    # ── 构建日志 ──
    log: Dict[str, Any] = {}
    log.update({f"{prefix}{k}": v for k, v in guidance_log.items()})
    # VelocityTracker 日志（grad_norm/guidance, grad_norm/reg, loss/reg, grad_norm/ratio）
    log.update({
        f"{prefix}{k}": v
        for k, v in tracker.collect_log(reg_weight=reg_weight).items()
    })
    log[f"{prefix}noise/t"] = t_val

    del tracker
    torch.cuda.empty_cache()
    return log
