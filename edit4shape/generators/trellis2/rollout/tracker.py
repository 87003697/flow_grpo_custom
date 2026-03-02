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
from typing import Dict, List, Optional

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
            每步的精确 t_val（float64 精度）。
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

    def collect_log(self, reg_weight: float = 1.0) -> Dict[str, float]:
        """
        收集 tracker 中所有可日志化的指标。

        在 VJP 循环前调用（Phase 2c backward 已填充 .grad，Phase 1 已填充 reg_grads）。

        Args:
            reg_weight: reg 梯度权重（用于计算有效梯度 norm，对齐 guidance 梯度的尺度）。
                        guidance 梯度已含 guidance_weight（Phase 2b backward 内嵌），
                        reg 梯度是原始值，需乘 reg_weight 才与 guidance 同尺度。
                        默认 1.0（即报告原始 reg 梯度 norm）。

        Returns:
            日志字典，可能包含：
            - loss/reg:           reg loss 标量值
            - grad_norm/guidance: guidance 梯度平均 L2 范数（proxy 级，含 guidance_weight）
            - grad_norm/reg:     reg 梯度平均 L2 范数（proxy 级，乘 reg_weight 后）
            - grad_norm/ratio:   guidance / reg 有效梯度范数比
        """
        log: Dict[str, float] = {}

        # ---- loss/reg ----
        if self.reg_loss_val is not None:
            log["loss/reg"] = self.reg_loss_val

        # ---- 梯度 norm ----
        T = len(self.timesteps)
        has_reg = len(self.reg_grads) == T

        guid_norms: List[float] = []
        reg_norms: List[float] = []

        for i in range(T):
            g = self.output_trajectory[i].grad  # (N, C) or None
            if g is None:
                continue
            guid_norms.append(g.norm().item())
            if has_reg and reg_weight != 0:
                # ★ 乘 reg_weight，对齐 guidance 梯度的尺度（含 guidance_weight）
                reg_norms.append((reg_weight * self.reg_grads[i]).norm().item())

        if guid_norms:
            avg_guid = sum(guid_norms) / len(guid_norms)
            log["grad_norm/guidance"] = avg_guid
            if reg_norms:
                avg_reg = sum(reg_norms) / len(reg_norms)
                log["grad_norm/reg"] = avg_reg
                log["grad_norm/ratio"] = avg_guid / max(avg_reg, 1e-8)

        return log

    def clip_guidance_grads(self, max_norm: float) -> Dict[str, float]:
        """
        裁剪 per-timestep guidance 梯度的 L2 范数。

        在 collect_log() 之前调用，使日志记录裁剪后的 grad norm。
        VJP 循环消费的也是裁剪后的梯度。

        Args:
            max_norm: 每步梯度的最大 L2 范数（≤0 则不裁剪）

        Returns:
            日志字典，包含 grad_clip/clipped_ratio
        """
        T = len(self.timesteps)
        if max_norm <= 0 or T == 0:
            return {}
        n_clipped = 0
        for i in range(len(self.output_trajectory)):
            g = self.output_trajectory[i].grad  # (N, C) or None
            if g is None:
                continue
            norm = g.norm()  # ()
            if norm > max_norm:
                self.output_trajectory[i].grad = g * (max_norm / (norm + 1e-8))  # (N, C)
                n_clipped += 1
        return {"grad_clip/clipped_ratio": n_clipped / T}
