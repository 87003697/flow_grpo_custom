"""
RolloutTracker: 三阶段 Autograd 架构的 cond-level proxy 记录器。

Phase 1 写入轨迹 + 提前计算 reg 梯度，Phase 2 backward 填充 guidance .grad，Phase 3 合并消费。

proxy 建在 cond_pred 上（CFG 之前），CFG 混合在 proxy 之后进行。

reg 梯度在 Phase 1 通过 torch.autograd.grad 提前算好，存入 reg_grads（纯数据，无图），
与 Phase 2 的 guidance backward 完全解耦。即使 Phase 2 OOM，reg 梯度仍可用。

数据流:
  Phase 1  → 写入 input_trajectory / output_trajectory / timesteps
             + 计算 reg_loss → autograd.grad → reg_grads（纯数据，存 tracker）
             + reg_loss_val（标量，日志用）
  Phase 2a → slat(含 proxy chain) → decoder → renderer → comp_rgb
  Phase 2  → comp_rgb.backward(rgb_grad)
             → 填充 output_trajectory[t].grad（仅 guidance 梯度，含 CFG 因子）
  Phase 3  → v_grad = guidance_grad + reg_weight * reg_grad
             → 仅重算 cond f_θ 并即时 VJP backward（无需 uncond/CFG/reg）
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class RolloutTracker:
    """
    Rollout 过程中的 proxy 记录器 — Phase 间的自包含数据传递载体。

    无 slat_proxy 中间层：decode/render 直接连接带 proxy chain 的 slat，
    guidance backward 沿 renderer → decoder → slat → scheduler → CFG → cond_proxy.grad。

    Attributes:
        input_trajectory:  T × (N, C)
            每步 x_t.feats.detach().clone()
            Phase 3 重算 f_θ 时的输入快照。
        output_trajectory: T × (N, C)
            每步 cond_pred.feats.detach().clone().requires_grad_(True)
            ★ 条件 velocity proxy（不是 CFG 后的 velocity）。
            proxy 建在 cond_pred 上，CFG 混合在 proxy 之后进行，
            Phase 2 guidance backward 沿 CFG chain 反传 → .grad 含 CFG 缩放因子。
            Phase 3 合并 reg_grads 后做 VJP，无需 uncond / CFG 混合 / reg 计算。
        timesteps: T × float
            每步的精确 t_val（0~1 范围，float64 精度）。
            Phase 3 直接读取，无需重建 scheduler。
        reg_grads: T × (N, C) 或空列表
            Phase 1 通过 autograd.grad(reg_loss, proxies) 提前计算的 reg 梯度。
            纯数据（detach），不依赖任何计算图，Phase 2 OOM 也不受影响。
            Phase 3 合并: v_grad = guidance_grad + reg_weight * reg_grad。
        reg_loss_val: float
            reg loss 标量值，用于日志记录。
    """

    # Phase 1 写入：rollout 每步的输入/输出快照 + 时间步
    input_trajectory: List[torch.Tensor] = field(default_factory=list)
    output_trajectory: List[torch.Tensor] = field(default_factory=list)
    #   ★ 存 cond_pred proxy（CFG 之前），.grad 由 Phase 2 guidance backward 填充
    timesteps: List[float] = field(default_factory=list)

    # Phase 1 写入：reg 梯度（提前计算，与 Phase 2 解耦）
    reg_grads: List[torch.Tensor] = field(default_factory=list)
    #   ★ 纯数据（detach），Phase 2 OOM 也不丢失
    reg_loss_val: Optional[float] = None
    #   ★ reg loss 标量值，日志用
