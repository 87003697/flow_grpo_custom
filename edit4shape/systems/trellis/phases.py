"""
Trellis 三阶段 Autograd — Phase 3 VJP 实现。

Phase 3: phase3_rollout_grad_backward
  逐步重算 cond f_θ → VJP backward → θ.grad 累积（显存 O(1)）

其他 Phase (0/1/2) 已迁移到：
  - stage_ops.py        — pre_rollout (inline dense sampling), rollout, decode_render
  - autograd_template.py — 编排 + guidance backward + decode-grad

设计参考：trellis2_distill 分支的 trellis2_shape_autograd_async.py
"""

from __future__ import annotations

import logging

import torch

from edit4shape.generators.trellis.state import TrellisState
from edit4shape.generators.trellis.rollout import (
    RolloutTracker,
    _predict_sparse_cond_velocity,
)
from edit4shape.systems.trellis.system import TrellisSystem

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
from trellis.modules.sparse import SparseTensor


# =====================================================================
# Phase 3: VJP Backward
# =====================================================================

def phase3_rollout_grad_backward(
    state: TrellisState,
    system: TrellisSystem,
    cfg,
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

        # 用 state.stage2.z0 的 coords 重建 SparseTensor（x_t_feats 无梯度）
        x_t = SparseTensor(
            feats=x_t_feats,
            coords=state.stage2.z0.coords,
        )  # SparseTensor(feats: (N, C), coords: (N, 4))

        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)

        try:
            # ★ 重算条件预测（有 θ 梯度，x_t 在 _predict_sparse_cond_velocity 内部 detach）
            cond_pred = _predict_sparse_cond_velocity(
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
