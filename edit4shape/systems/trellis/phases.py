"""
Trellis 三阶段 Autograd 纯计算函数（Phase 级别构建块）。

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

编排模板（已移至独立文件）：
  autograd_template.py              — 单路 / 双路 3-sub-step 编排（StageOps 参数化）

旧版编排函数（仍保留于本文件）：
  three_phase_step  — 直接传 cfg/device/accelerator 的早期版本

设计参考：trellis2_distill 分支的 trellis2_shape_autograd_async.py
"""

from __future__ import annotations

import logging

import torch
import ml_collections
from typing import Any, Dict, Optional, Tuple

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

        try:
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
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                f"P3 VJP step {i}/{T} OOM → partial grad"
            )
            break

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
