"""
StageOps 抽象基类 — 训练阶段的计算操作协议。

定义每个训练阶段 "做什么" 的接口，不包含任何清理/编排/生命周期知识。
同一个 Ops 实例可在同步/异步/单阶段/多阶段模板中复用。

此文件仅包含模型无关的 ABC 和异常类；
具体实现（ShapeOps / TexOps 等）由各模型后端自行提供。

使用方式::

    from edit4shape.systems.utils.stage_ops import StageOps

    class MyOps(StageOps):
        ...

设计原则：
  - 清理策略由 Slot（编排层）决定，不由 Ops 决定
  - Ops 只回答 "这个阶段自身的计算是什么？"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from edit4shape.generators.trellis2.rollout import RolloutTracker


# =====================================================================
# 异常
# =====================================================================

class StageSkipError(Exception):
    """阶段降级异常 — 前置条件不满足（如 meshes 为 None），跳过后续计算。"""
    pass


# =====================================================================
# 抽象基类
# =====================================================================

class StageOps(ABC):
    """
    单个训练阶段的计算操作接口。

    封装阶段特有的计算逻辑（rollout / decode+render / VJP），
    不包含任何清理策略、生命周期管理或编排知识。

    子类只需关注 "我这个阶段怎么计算"，不需要知道：
    - 我是不是最后一个阶段
    - 我的 subs/meshes 是否被后续阶段需要
    - 我是同步执行还是异步流水线
    """

    # ═══════════════════════════════════════════════════════
    # 配置查询
    # ═══════════════════════════════════════════════════════

    @abstractmethod
    def get_model(self, system):
        """返回该阶段的 model（DDP 包装后）。"""
        ...

    @abstractmethod
    def get_stage_name(self) -> str:
        """返回阶段名称（"shape" / "tex"），用于 pipeline.get_stage_config() 等。"""
        ...

    @abstractmethod
    def get_seed_offset(self) -> int:
        """返回 seed 偏移量，避免与其他阶段的 seed 冲突。"""
        ...

    @abstractmethod
    def get_reg_weight(self, system) -> float:
        """返回 reg loss 权重（cfg.{stage}.train.loss.reg）。"""
        ...

    @abstractmethod
    def get_reg_type(self, system) -> str:
        """返回 reg 类型（cfg.{stage}.train.loss.reg_type），'v' | 'x0' | 'x1'。"""
        ...

    @abstractmethod
    def get_guidance_weight(self, system) -> float:
        """返回 guidance loss 权重（cfg.{stage}.train.loss.guidance）。"""
        ...

    @abstractmethod
    def get_guidance_cfg(self, system):
        """返回 guidance 配置（cfg.{stage}.guidance）。"""
        ...

    # ═══════════════════════════════════════════════════════
    # Async 友好查询（VJP loop / P2-grad 复用）
    # ═══════════════════════════════════════════════════════

    @abstractmethod
    def get_slat(self, state):
        """返回该阶段的 slat（shape_slat / tex_slat），VJP 通过 .replace() 构建 x_t。"""
        ...

    def get_shape_cond(self, state):
        """返回 VJP 所需的 shape_cond。Shape→None, Tex→shape_slat_norm。默认 None。"""
        return None

    @abstractmethod
    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        decode+render → 原始字典（含 'color' key）。

        与 decode_render 的区别：不做 vis 挂载，不做 subs/meshes 赋值。
        用于异步 P2-grad 重跑（只需 comp_rgb.backward）和 P2-no-grad。

        子类 decode_render 应调用此方法 + 附加 vis 挂载逻辑。
        """
        ...

    # ═══════════════════════════════════════════════════════
    # Phase 函数
    # ═══════════════════════════════════════════════════════

    def pre_rollout(self, state, system, global_step) -> None:
        """
        Phase 0: Rollout 前的准备（可选）。

        - Shape: dense_sampling → 填充 state.coords
        - Tex (standalone): shape_frozen_prepare → 填充 shape 产物 + detach
        - Tex (from shape): no-op（shape 产物由上游提供）

        默认 no-op，子类按需覆写。
        """
        pass

    @abstractmethod
    def rollout(self, state, system, seed) -> RolloutTracker:
        """
        Phase 1: rollout → proxy chain + tracker。

        填充 state.features.{shape,tex}_slat (proxy chain)。
        返回 RolloutTracker（含 input/output trajectory, timesteps, reg_grads）。
        """
        ...

    @abstractmethod
    def decode_render(self, state, system) -> torch.Tensor:
        """
        Phase 2a: decode + render → comp_rgb。

        返回 comp_rgb (B, V, H, W, 3)。
        调用时是否有 autograd 图取决于调用方上下文（同步模板带梯度，异步模板 no_grad）。

        Side effects:
        - 挂载 vis 数据到 state（e.g., views_generated.shape_tensor）
        - 挂载中间产物到 state（e.g., features.subs, features.meshes）

        Raises:
            StageSkipError: 前置条件不满足（如 meshes 为 None）
        """
        ...

    @abstractmethod
    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: VJP → θ.grad 累积。

        逐步/批量重算 f_θ，用 tracker 中的梯度做 VJP → 模型参数梯度累积。
        内部自行清理 tracker 数据。

        Returns:
            日志字典（通常含 loss/reg 或空 dict）
        """
        ...
