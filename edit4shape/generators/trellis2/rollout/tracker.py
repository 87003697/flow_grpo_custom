"""
RolloutTracker: 三阶段 Autograd 架构的 cond-level proxy 记录器。

Phase 1 写入轨迹，Phase 2 backward 自动填充 .grad，Phase 3 消费 .grad。

proxy 建在 cond_pred 上（CFG 之前），CFG 混合在 proxy 之后进行。
Phase 2 中 (guidance_loss + reg_loss).backward() 一路反传穿过
renderer → decoder → slat → scheduler → CFG → cond_proxy，
cond_proxy.grad 自动包含 CFG 缩放因子和 reg 梯度 → Phase 3 只需重算 cond forward + VJP。

reg_loss 在 Phase 1 的 rollout 中计算（连着 cond_proxy 的计算图），
存储在 state.regularization.reg_loss 中，Phase 2 合并到 guidance_loss 一起 backward。
Phase 3 完全不感知 reg，只做纯粹的 VJP。

数据流:
  Phase 1  → 写入 input_trajectory / output_trajectory / timesteps
             + 计算 reg_loss → state.regularization.reg_loss（保留图，连 cond_proxy）
  Phase 2a → slat(含 proxy chain) → decoder → renderer → comp_rgb
  Phase 2  → (guidance_loss + reg_loss).backward()
             → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）→ 释放所有图
  Phase 3  → 读取 timesteps[t] + input_trajectory[t] + output_trajectory[t].grad
             → 仅重算 cond f_θ 并即时 VJP backward（无需 uncond/CFG/reg）
"""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class RolloutTracker:
    """
    Rollout 过程中的 proxy 记录器 — Phase 间的自包含数据传递载体。

    无 slat_proxy 中间层：decode/render 直接连接带 proxy chain 的 slat，
    loss.backward() 一路反传穿过 renderer → decoder → slat → scheduler → CFG → cond_proxy.grad。

    Attributes:
        input_trajectory:  T × (N, C)
            每步 x_t.feats.detach().clone()
            Phase 3 重算 f_θ 时的输入快照。
        output_trajectory: T × (N, C)
            每步 cond_pred.feats.detach().clone().requires_grad_(True)
            ★ 条件 velocity proxy（不是 CFG 后的 velocity）。
            proxy 建在 cond_pred 上，CFG 混合在 proxy 之后进行，
            因此 Phase 2 backward 沿 renderer → decoder → slat → scheduler → CFG → cond_proxy chain 反传后，
            .grad 自动包含 CFG 缩放因子 + reg 梯度。
            Phase 3 只需重算 cond_pred，无需 uncond / CFG 混合 / reg 计算。
        timesteps: T × float
            每步的精确 t_val（float64 精度）。
            Phase 3 直接读取，无需重建 scheduler。
    """

    # Phase 1 写入：rollout 每步的输入/输出快照 + 时间步
    input_trajectory: List[torch.Tensor] = field(default_factory=list)
    output_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   ★ 存 cond_pred proxy（CFG 之前），.grad 由 Phase 2 沿 CFG chain 自动填充
    timesteps: List[float] = field(default_factory=list)
