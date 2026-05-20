"""
PendingJob + StageContext — 异步 Guidance 流水线的公共抽象。

本文件包含两套并行的 building block 基类：

1. VJP 范式（三阶段 Autograd）：
   StageContext         — per-stage 异步状态（RolloutTracker + flags + log）
   ctx_* 自由函数       — VJP 原子操作（drain / VJP / unpack）
   PendingJob           — VJP micro-batch 基类

2. Onestep 范式（单步去噪 + 标准 Autograd）：
   OnestepContext       — per-stage 异步状态（VelocityTracker + t_val + submitted）
   OnestepPendingJob    — Onestep micro-batch 基类

3. Contrastive 范式（对比学习 + FlowEdit 编辑）：
   ContrastiveStageContext — per-stage 运行时状态（ops + VelocityTracker + t_val）
   ContrastivePendingJob  — Contrastive micro-batch 基类（stage-list 驱动）

关注点分离：
  StageOps (策略)      — 数据无关的阶段策略，始终由调用方显式传入
  *Context (状态)      — per-sample 异步生命周期，不绑定任何策略
  *PendingJob (基类)   — 不持有 ctx，提供参数化 building block 方法

使用方式::

  # ── VJP 范式 ──
  # 单阶段子类
  class ShapePendingJob(PendingJob):
      ctx: StageContext
      def drain_guidance(self, ops, system, profiler):
          self._drain_stage_guidance(ops, self.ctx, system, profiler,
              clean_decode=self._clean_p2_decode,
              clean_for_vjp=self._clean_for_vjp)

  # 多阶段子类
  class DualPendingJob(PendingJob):
      shape_ctx: StageContext; tex_ctx: StageContext
      def drain_shape_guidance(self, ops, system, profiler):
          self._drain_stage_guidance(ops, self.shape_ctx, ..., prefix="S_", ...)

  # ── Onestep 范式 ──
  # 单阶段子类
  class ShapeOnestepJob(OnestepPendingJob):
      ctx: OnestepContext
      def drain(self, ops, system, profiler):
          return self._drain_onestep_stage(ops, self.ctx, system, profiler,
              clean_decode=..., clean_for_relay=...)

  # 多阶段子类
  class DualOnestepJob(OnestepPendingJob):
      shape_ctx: OnestepContext; tex_ctx: OnestepContext
      def drain_shape(self, ops, system, profiler):
          return self._drain_onestep_stage(ops, self.shape_ctx, ..., prefix="S_", ...)

  # ── Contrastive 范式 ──
  class MyContrastiveJob(ContrastivePendingJob):
      @classmethod
      def create(cls, batch, ...):
          entries = [ContrastiveStageContext(ops=ShapeOps()), ...]
          inst = cls(state=state, stage_entries=entries)
          for i in range(len(entries)):
              reg_log = inst._create_stage(i, system, profiler)
          return inst
      def drain(self, system, profiler, tgt_embeds, src_embeds):
          for i in range(len(self.stage_entries)):
              self._drain_contrastive_stage(i, system, profiler, tgt_embeds, src_embeds)

导出清单（供子类导入）：
  # VJP
  StageContext, PendingJob,
  ctx_p2_wait, ctx_p2_grad, ctx_vjp_loop,
  ctx_unpack_result, ctx_invalidate, ctx_build_vjp_log, ctx_clean_tracker,
  # Onestep
  OnestepContext, OnestepPendingJob,
  # Contrastive
  ContrastiveStageContext, ContrastivePendingJob,
  # 通用
  _log_mem, _reclaim
"""

# =====================================================================
# 标准库导入
# =====================================================================
import gc
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# =====================================================================
# 第三方库导入
# =====================================================================
import torch
import torch.distributed as dist
import torch.nn.functional as F

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.guidance.pipeline_parallel import AsyncGuidanceResult
from edit4shape.generators.trellis2.state import Trellis2State as Trellis2StateBase
from edit4shape.generators.trellis2.rollout import RolloutTracker, VelocityTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity, _vjp_loader
from edit4shape.systems.base import TrainModeGuard
from edit4shape.systems.utils import AsyncPhaseProfiler
from edit4shape.systems.utils.logging import build_autograd_step_log
from edit4shape.systems.utils.stage_ops import StageOps, StageSkipError
from edit4shape.guidance.pipelines.utils.loss_functions import contrastive_loss_step


# =====================================================================
# StageContext — 单个阶段的异步状态（不含 ops）
# =====================================================================

@dataclass
class StageContext:
    """
    单个 stage 在一个 micro-batch 中的异步生命周期状态。

    纯 per-sample 数据容器，不绑定任何 StageOps 策略对象。
    策略（ops）始终由调用方显式传入 ctx_* 自由函数或 PendingJob.drain_* 方法。

    字段：
      tracker        — RolloutTracker（含 input/output trajectory, timesteps, reg_grads）
      submitted      — P2-no-grad + submit 是否成功
      skip_vjp       — 是否跳过 VJP（P2 OOM / guidance 不可用时置 True）
      guidance_log   — guidance 阶段的日志字典
    """
    tracker: RolloutTracker
    submitted: bool = False
    skip_vjp: bool = False
    guidance_log: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 通用辅助（module-level）
# =====================================================================

def _log_mem(global_step: int, tag: str, *, warn: bool = False) -> None:
    """记录 GPU 显存状态。"""
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv = torch.cuda.memory_reserved() / 1024**3
    msg = (
        f"[Step {global_step}] {tag} | "
        f"allocated={alloc:.2f} GiB, reserved={resv:.2f} GiB"
    )
    (logging.warning if warn else logging.info)(msg)


def _reclaim() -> None:
    """gc.collect + empty_cache 二连。"""
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# ctx_* 自由函数 — 原子操作
# =====================================================================

def ctx_unpack_result(
    ctx: StageContext,
    state: Trellis2StateBase,
    result: AsyncGuidanceResult,
    profiler: AsyncPhaseProfiler,
) -> torch.Tensor:
    """
    解包 AsyncGuidanceResult → rgb_grad + 挂载 vis + 填充 ctx.guidance_log + profiler 计时。

    Returns:
        rgb_grad: detached 梯度张量 (B, V, H, W, 3)
    """
    rgb_grad = result.rgb_grad.detach()  # (B, V, H, W, 3)

    # 挂载可视化数据
    state.views_edited.image_tensor = result.edited_imgs
    state.views_edited.trackers = result.trackers

    # 构建日志
    log: Dict[str, Any] = {}
    if result.loss_dict:
        log.update({f"loss/{k}": v for k, v in result.loss_dict.items()})
    log["loss/guidance"] = result.loss_scalar

    # guidance GPU 计时
    if result.guid_wall_start is not None:
        profiler.set_guid_timing(result.guid_wall_start, result.guid_wall_end)

    ctx.guidance_log = log
    return rgb_grad


def ctx_invalidate(ctx: StageContext) -> None:
    """OOM 降级：清空 output_trajectory 梯度 + guidance 日志。"""
    for out_t in ctx.tracker.output_trajectory:
        out_t.grad = None
    ctx.guidance_log = {}


def ctx_build_vjp_log(ctx: StageContext, reg_weight: float = 1.0) -> Dict[str, Any]:
    """构建 VJP 阶段日志（loss/reg + grad_norm/*）。"""
    return ctx.tracker.collect_log(reg_weight=reg_weight)


def ctx_clean_tracker(ctx: StageContext) -> None:
    """释放 tracker 全部轨迹数据。"""
    del ctx.tracker.input_trajectory[:], ctx.tracker.output_trajectory[:]
    del ctx.tracker.timesteps[:], ctx.tracker.reg_grads[:]


# =====================================================================
# ctx_* 自由函数 — P2 阶段
# =====================================================================

def ctx_p2_wait(
    ctx: StageContext,
    state: Trellis2StateBase,
    global_step: int,
    system: Any,
    profiler: AsyncPhaseProfiler,
    prefix: str = "",
) -> Optional[torch.Tensor]:
    """
    P2-wait: 阻塞等待 guidance GPU → rgb_grad。返回 None 表示不可用。

    OOM: 断开 traceback 引用链 → 显存日志 → gc 回收 → None。
    其他异常: 日志告警 → None。
    """
    if not ctx.submitted:
        return None

    profiler.tick(f"{prefix}P2_wait")
    try:
        result = system.guidance.wait_and_get(
            target_device=system.accelerator.device,
        )
        return ctx_unpack_result(ctx, state, result, profiler)
    except torch.cuda.OutOfMemoryError as e:
        e.__traceback__ = None  # 断开 traceback → frame locals 引用链
        _log_mem(global_step, f"{prefix}P2-wait OOM → skip VJP", warn=True)
        _reclaim()
        return None
    except Exception as e:
        logging.warning(
            f"[Step {global_step}] {prefix}P2-wait failed: {e} → skip VJP"
        )
        return None


def ctx_p2_grad(
    ops: StageOps,
    ctx: StageContext,
    state: Trellis2StateBase,
    global_step: int,
    system: Any,
    profiler: AsyncPhaseProfiler,
    rgb_grad: torch.Tensor,
    prefix: str = "",
    pre_grad: Optional[Callable] = None,
    clean_decode: Optional[Callable] = None,
) -> None:
    """
    P2-grad: 重跑 decode+render（带梯度）→ backward。

    通过 ops.decode_render_dict 获取 comp_rgb，
    用 rgb_grad 做 backward → 填充 output_trajectory[t].grad。

    OOM / StageSkipError: ctx_invalidate + skip_vjp=True → 显存日志。
    Finally: del 大张量 → clean_decode → gc + empty_cache → 显存日志。

    Args:
        ops:          StageOps 策略对象（提供 decode_render_dict）
        pre_grad:     在 decode 前执行的回调（如 reload_decode_cache_to_gpu）
        clean_decode: decode 后的清理回调（如 release_spatial_cache）。
                      不需要调用 _reclaim — 本函数末尾统一调用。
    """
    profiler.tick(f"{prefix}P2_grad")
    comp_rgb = None
    if pre_grad is not None:
        pre_grad()
    try:
        comp_rgb = ops.decode_render_dict(state, system)["color"]
        comp_rgb.backward(rgb_grad)
    except (torch.cuda.OutOfMemoryError, StageSkipError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
        ctx_invalidate(ctx)
        # ★ 不再无条件 skip_vjp：reg 梯度在 P1 已预计算，P2-grad OOM 不影响
        if not ctx.tracker.reg_grads:
            ctx.skip_vjp = True
        _log_mem(global_step, f"{prefix}P2-grad failed: {e} → skip VJP", warn=True)
    finally:
        del comp_rgb, rgb_grad
        if clean_decode is not None:
            clean_decode()
        _reclaim()
        _log_mem(global_step, f"{prefix}P2-grad cleanup done")


# =====================================================================
# ctx_* 自由函数 — VJP 阶段
# =====================================================================

def ctx_vjp_loop(
    ops: StageOps,
    ctx: StageContext,
    state: Trellis2StateBase,
    global_step: int,
    system: Any,
    chunk_size: int = 4,
) -> Dict[str, Any]:
    """
    通用 VJP loop — 逐步/批量重算 f_θ，合并 guidance + reg 梯度 → θ.grad 累积。
    显存 O(1)，不随步数增长。

    通过 ops 获取：
      - get_stage_name() → "shape" / "tex"
      - get_latent(state) → shape_slat / tex_slat
      - get_shape_cond(state) → None / shape_slat_norm
      - get_model(system) → model（用于 no_sync）
      - get_reg_weight(system) → reg weight

    梯度来源：
    - guidance: output_trajectory[t].grad（P2 backward 填充，含 CFG 因子）
    - reg:     tracker.reg_grads[t]（P1 autograd.grad 预计算）

    DDP 安全：整个 VJP 循环在 model.no_sync() 下执行，
    backward 只做本地累积，不触发 DDP all-reduce。
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_name = ops.get_stage_name()
    stage_config = pipeline.get_stage_config(stage_name)
    flow_res = stage_config["flow_resolution"]
    reg_weight = ops.get_reg_weight(system)

    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)

    slat = ops.get_latent(state)
    shape_cond = ops.get_shape_cond(state)
    model = ops.get_model(system)

    # ★ 先裁剪 guidance 梯度，再收集日志（日志记录的是裁剪后的 grad norm）
    log = ctx.tracker.clip_guidance_grads(ops.get_guidance_grad_max_norm(system))
    log.update(ctx_build_vjp_log(ctx, reg_weight=reg_weight))  # ★ 记录裁剪后的 grad_norm/guidance

    _no_sync = model.no_sync() if dist.is_initialized() else nullcontext()
    with _no_sync:
        for x_t, t_batch, cond_k, v_grad, sc_k in _vjp_loader(
            ctx.tracker, slat, cond_emb, shape_cond,
            reg_weight, device, chunk_size,
        ):
            try:
                cond_pred = _predict_velocity(
                    pipeline, x_t, t_batch, cond_k,
                    stage_name, flow_res, sc_k,
                )  # SparseTensor
                cond_pred.feats.backward(v_grad)
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    f"[Step {global_step}] {stage_name} P1-grad OOM → partial grad"
                )
                break

    return log


# =====================================================================
# PendingJob — 异步流水线 micro-batch 基类（支持单/多阶段）
# =====================================================================

@dataclass
class PendingJob:
    """
    异步流水线 micro-batch 基类 — 支持单阶段和多阶段继承。

    不持有 StageOps（ops 始终由调用方显式传入）。
    不持有 StageContext — 子类按需定义（单阶段：ctx，双阶段：shape_ctx + tex_ctx）。

    提供两个参数化的 building block 方法：
      _drain_stage_guidance(ops, ctx, ...)  — P2 全流程 (wait → grad → clean)
      _drain_stage_vjp(ops, ctx, ...)       — VJP 循环 → θ.grad → clean → 合并日志

    子类职责：
      - 定义自己的 ctx 字段
      - 定义公开 drain 方法，调用 building block 并传入正确的 ctx / callbacks
      - 实现清理回调

    使用方式::

      # 单阶段子类 — 定义 ctx + 清理方法
      class ShapePendingJob(PendingJob):
          ctx: StageContext
          def drain_guidance(self, ops, system, profiler):
              self._drain_stage_guidance(ops, self.ctx, system, profiler,
                  clean_decode=self._clean_p2_decode,
                  clean_for_vjp=self._clean_for_vjp)

      # 多阶段子类 — 定义 shape_ctx + tex_ctx + 各自清理方法
      class DualPendingJob(PendingJob):
          shape_ctx: StageContext; tex_ctx: StageContext
          def drain_shape_guidance(self, ops, system, profiler):
              self._drain_stage_guidance(ops, self.shape_ctx, ..., prefix="S_", ...)
    """

    state: Trellis2StateBase
    global_step: int = 0
    batch_size: int = 0

    # ════════════════════════════════════════════════════════
    # Building Blocks — 子类通过传入不同参数来定制行为
    # ════════════════════════════════════════════════════════

    def _drain_stage_guidance(
        self,
        ops: StageOps,
        ctx: StageContext,
        system: Any,
        profiler: AsyncPhaseProfiler,
        *,
        prefix: str = "",
        pre_grad: Optional[Callable] = None,
        clean_decode: Optional[Callable] = None,
        clean_for_vjp: Callable,
    ) -> None:
        """
        通用 P2 全流程 building block: wait → grad → clean_for_vjp。

        子类通过传入不同的 ctx / prefix / callbacks 来定制行为。

        Args:
            ops:           StageOps 策略对象
            ctx:           目标 StageContext（shape_ctx / tex_ctx / ctx）
            prefix:        profiler tick 前缀（"" / "S_" / "T_"）
            pre_grad:      P2-grad 前的回调（如 reload_decode_cache_to_gpu）
            clean_decode:  P2-grad 后的清理回调（如 release_spatial_cache）
            clean_for_vjp: P2 整体结束后的清理回调（必需）

        降级链路:
          submitted=False / wait 失败 → guidance 梯度缺失，VJP 仅用 reg 梯度
          P2-grad OOM → ctx_invalidate + skip_vjp=True
          tracker 无 reg_grads → skip_vjp=True
        """
        with TrainModeGuard(ops.get_model(system)):
            rgb_grad = ctx_p2_wait(
                ctx, self.state, self.global_step,
                system, profiler, prefix=prefix,
            )
            if rgb_grad is not None:
                ctx_p2_grad(
                    ops, ctx, self.state, self.global_step,
                    system, profiler, rgb_grad,
                    prefix=prefix, pre_grad=pre_grad,
                    clean_decode=clean_decode,
                )
            else:
                # ★ P2 跳过时不再无条件 skip_vjp：
                #   reg 梯度在 P1 已预计算（tracker.reg_grads），
                #   VJP 循环可以仅用 reg 梯度回传，不浪费 P1 的计算。
                #   _vjp_loader 已正确处理 guid_grad=None 的情况。
                if not ctx.tracker.reg_grads:
                    # 没有 reg 梯度也没有 guidance 梯度 → 确实无梯度可用
                    ctx.skip_vjp = True
        clean_for_vjp()

    def _drain_stage_vjp(
        self,
        ops: StageOps,
        ctx: StageContext,
        system: Any,
        profiler: AsyncPhaseProfiler,
        *,
        prefix: str = "",
        stage_name: str = "",
        chunk_size: int = 4,
        clean_p1_grad: Callable,
        log_prefix: str = "",
        collect_profiler: bool = True,
    ) -> Dict[str, Any]:
        """
        通用 VJP building block: VJP 循环 → θ.grad → clean → 合并日志。

        Args:
            ops:              StageOps 策略对象
            ctx:              目标 StageContext
            prefix:           profiler tick 前缀（"" / "S_" / "T_"）
            stage_name:       日志中的阶段名称（"" / "Shape" / "Tex"）
            chunk_size:       VJP loop 批大小
            clean_p1_grad:    VJP 后的清理回调（必需）
            log_prefix:       build_autograd_step_log 的 key 前缀（"" / "shape/" / "tex/"）
            collect_profiler: 是否在此调用 profiler.tick("end") + profiler.collect()

        skip_vjp=True 时跳过整个 VJP 循环（零梯度贡献），
        仅执行 clean_p1_grad 释放 tracker。
        """
        label = f"{stage_name} " if stage_name else "该 MB "
        if not ctx.skip_vjp:
            with TrainModeGuard(ops.get_model(system)):
                profiler.tick(f"{prefix}P1_grad")
                phase3_log = ctx_vjp_loop(
                    ops, ctx, self.state, self.global_step, system,
                    chunk_size=chunk_size,
                )
        else:
            profiler.tick(f"{prefix}P1_skip")
            phase3_log = ctx_build_vjp_log(ctx, reg_weight=ops.get_reg_weight(system))  # 即使跳过 VJP 也收集 loss/reg
            # ★ 动态构建原因描述，准确反映 guidance / reg 状态
            reasons = []
            if not ctx.submitted:
                reasons.append("guidance 未提交")
            else:
                reasons.append("guidance 不可用")
            if not ctx.tracker.reg_grads:
                reasons.append("无 reg 梯度")
            logging.info(
                f"[Step {self.global_step}] 跳过 {label}VJP — "
                f"{'且'.join(reasons)}，{label}零梯度贡献"
            )
        clean_p1_grad()

        if collect_profiler:
            profiler.tick("end")

        # 合并日志
        merged = build_autograd_step_log(
            ctx.guidance_log, ops.get_reg_weight(system), phase3_log,
            prefix=log_prefix,
        )
        if collect_profiler:
            merged.update(profiler.collect(
                self.global_step, print_freq=int(system.cfg.freq.profiler),
            ))
        return merged

    # ════════════════════════════════════════════════════════
    # 便利方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        _reclaim()


# =====================================================================
# OnestepContext — 单个阶段的 Onestep 异步状态
# =====================================================================

@dataclass
class OnestepContext:
    """
    单个 stage 在一个 Onestep micro-batch 中的异步生命周期状态。

    平行于 StageContext（VJP 用），但使用 VelocityTracker 而非 RolloutTracker。
    策略（ops）始终由调用方显式传入 OnestepPendingJob._drain_onestep_stage 方法。

    字段：
      vel_tracker    — VelocityTracker（含 v_student / v_proxy / reg_grad）
      t_val          — 采样时间步（日志用）
      submitted      — P4a + submit 是否成功
      guidance_log   — guidance 阶段的日志字典
    """
    vel_tracker: VelocityTracker
    t_val: float = 0.0
    submitted: bool = False
    guidance_log: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# OnestepPendingJob — Onestep 异步流水线 micro-batch 基类
# =====================================================================

@dataclass
class OnestepPendingJob:
    """
    Onestep 异步流水线 micro-batch 基类 — 支持单阶段和多阶段继承。

    平行于 PendingJob（VJP 用），提供 Onestep 特有的 building block。

    不持有 StageOps（ops 始终由调用方显式传入）。
    不持有 OnestepContext — 子类按需定义（单阶段：ctx，双阶段：shape_ctx + tex_ctx）。

    提供一个参数化的 building block 方法：
      _drain_onestep_stage(ops, ctx, ...)  — P4b-wait → P4c → clean → P5 relay → log

    子类职责：
      - 定义自己的 ctx 字段（OnestepContext）
      - 定义公开 drain 方法，调用 building block 并传入正确的 ctx / callbacks
      - 实现清理回调（clean_decode, clean_for_relay）

    使用方式::

      # 单阶段子类
      class ShapeOnestepJob(OnestepPendingJob):
          ctx: OnestepContext
          def drain(self, ops, system, profiler):
              log = self._drain_onestep_stage(ops, self.ctx, system, profiler,
                  clean_decode=lambda: self.state.release_shape_spatial_cache(),
                  clean_for_relay=self._clean_for_relay)
              self.ctx = None; self._reclaim()
              return log

      # 多阶段子类
      class DualOnestepJob(OnestepPendingJob):
          shape_ctx: OnestepContext; tex_ctx: OnestepContext
          def drain_shape(self, ops, system, profiler):
              return self._drain_onestep_stage(ops, self.shape_ctx, ..., prefix="S_", ...)
    """

    state: Trellis2StateBase
    global_step: int = 0
    batch_size: int = 0

    # ════════════════════════════════════════════════════════
    # Building Block — Onestep drain 全流程
    # ════════════════════════════════════════════════════════

    def _drain_onestep_stage(
        self,
        ops: StageOps,
        ctx: OnestepContext,
        system: Any,
        profiler: AsyncPhaseProfiler,
        *,
        prefix: str = "",
        clean_decode: Optional[Callable] = None,
        clean_for_relay: Callable,
        log_prefix: str = "",
        collect_profiler: bool = True,
    ) -> Dict[str, Any]:
        """
        通用 Onestep drain building block:
          P4b-wait → P4c(decode+backward) → clean → P5(relay) → log。

        子类通过传入不同的 ctx / prefix / callbacks 来定制行为。

        Args:
            ops:              StageOps 策略对象（提供 decode_render_dict / get_model 等）
            ctx:              目标 OnestepContext
            prefix:           profiler tick 前缀（"" / "S_" / "T_"）
            clean_decode:     P4c 后的 decode 清理回调（如 release_spatial_cache）
            clean_for_relay:  P4c 整体结束后的清理回调（detach, offload, gc，必需）
            log_prefix:       日志 key 前缀（"" / "shape/" / "tex/"）
            collect_profiler: 是否在 P5 后收集 profiler.tick("end") + profiler.collect()

        降级链路:
          submitted=False / wait 失败 → 无 rgb_grad → 跳过 P4c
          P4c OOM → v_proxy.grad=None → relay 仅用 reg_grad
          reg_grad=None + v_proxy.grad=None → 跳过 relay（零梯度贡献）

        Returns:
            日志字典（含 guidance loss / reg loss / noise/t / profiler 计时）
        """
        device = system.accelerator.device
        model = ops.get_model(system)
        tracker = ctx.vel_tracker
        rgb_grad = None

        # ── P4b-wait: 等 guidance GPU → rgb_grad ──────────────────
        if ctx.submitted:
            profiler.tick(f"{prefix}P4b_wait")
            try:
                result = system.guidance.wait_and_get(target_device=device)
                rgb_grad = result.rgb_grad.detach()  # (B, V, H, W, C)

                # 挂载 vis
                self.state.views_edited.image_tensor = result.edited_imgs
                self.state.views_edited.trackers = result.trackers

                # guidance 日志
                if result.loss_dict:
                    ctx.guidance_log.update({
                        f"loss/{k}": v for k, v in result.loss_dict.items()
                    })
                ctx.guidance_log["loss/guidance"] = result.loss_scalar

                # guidance GPU 计时
                if result.guid_wall_start is not None:
                    profiler.set_guid_timing(
                        result.guid_wall_start, result.guid_wall_end,
                    )
                del result
            except torch.cuda.OutOfMemoryError as e:
                e.__traceback__ = None  # 断开 traceback → frame locals 引用链
                logging.warning(
                    f"[Step {self.global_step}] {prefix}P4b-wait OOM → reg-only relay"
                )
                _reclaim()
            except Exception as e:
                logging.warning(
                    f"[Step {self.global_step}] {prefix}P4b-wait failed: {e} → reg-only relay"
                )

        # ── P4c: with-grad decode + backward → v_proxy.grad ──────
        if rgb_grad is not None:
            profiler.tick(f"{prefix}P4c_decode_grad")
            comp_rgb = None
            try:
                with TrainModeGuard(model):
                    render_out = ops.decode_render_dict(self.state, system)
                    comp_rgb = render_out["color"]  # autograd → z0_hat → v_proxy
                    comp_rgb.backward(rgb_grad)  # → v_proxy.grad = guidance_grad
                    del comp_rgb, render_out
            except torch.cuda.OutOfMemoryError as e:
                e.__traceback__ = None
                logging.warning(
                    f"[Step {self.global_step}] {prefix}P4c OOM → reg-only relay"
                )
                del comp_rgb
            finally:
                del rgb_grad
                if clean_decode is not None:
                    clean_decode()
                _reclaim()

        # ── clean_for_relay ──────────────────────────────────────
        clean_for_relay()

        # ── clip guidance grads（在 relay 之前裁剪，relay 消费裁剪后的梯度）──
        clip_log = tracker.clip_guidance_grads(ops.get_guidance_grad_max_norm(system))

        # ── P5: relay → θ.grad ──────────────────────────────────
        skip_relay = (
            tracker.v_proxy.grad is None
            and tracker.reg_grad is None
        )
        if not skip_relay:
            profiler.tick(f"{prefix}P5_relay")
            # ★ 如果 v_proxy.grad is None（guidance 不可用），仅用 reg_grad
            if tracker.v_proxy.grad is None:
                tracker.v_proxy.grad = torch.zeros_like(tracker.v_proxy)
            _no_sync = model.no_sync() if dist.is_initialized() else nullcontext()
            with TrainModeGuard(model):
                with _no_sync:
                    tracker.relay_and_backward()
        else:
            profiler.tick(f"{prefix}P5_skip")
            logging.info(
                f"[Step {self.global_step}] 跳过 {prefix}P5 relay — "
                f"无 guidance 梯度且无 reg 梯度，零梯度贡献"
            )

        if collect_profiler:
            profiler.tick("end")

        # ── 构建日志 ─────────────────────────────────────────────
        reg_weight = ops.get_reg_weight(system)
        log: Dict[str, Any] = {}
        log.update({
            f"{log_prefix}{k}": v
            for k, v in ctx.guidance_log.items()
        })
        # ★ clip 日志（clip 已在 P5 relay 之前完成）
        log.update({f"{log_prefix}{k}": v for k, v in clip_log.items()})
        log.update({
            f"{log_prefix}{k}": v
            for k, v in tracker.collect_log(reg_weight=reg_weight).items()
        })
        log[f"{log_prefix}noise/t"] = ctx.t_val
        if collect_profiler:
            log.update(profiler.collect(
                self.global_step,
                print_freq=int(system.cfg.freq.profiler),
            ))

        # ── 清理 tracker 计算图 ──────────────────────────────────
        del tracker.v_student, tracker.v_proxy

        return log

    # ════════════════════════════════════════════════════════
    # 便利方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        _reclaim()


# =====================================================================
# ContrastiveStageContext — stage-list 驱动的 per-stage 运行时状态
# =====================================================================

@dataclass
class ContrastiveStageContext:
    """
    Contrastive 流水线中一个 stage 的运行时状态。

    与 OnestepContext 不同点：
    - 持有 ops（stage-list 驱动，ops 不由调用方每次传入）
    - 没有 submitted / guidance_log（contrastive 不做 per-stage submit）
    - 保存 z0_hat（student x0 预测，有梯度到 v_proxy）

    字段：
      ops        — StageOps 策略对象（ShapeOps / TexOps / ShapeHQOps …）
      vel_tracker — VelocityTracker（P3 setup_proxy 后填入）
      t_val      — 采样时间步
      zt_feats   — (N, C) 加噪后特征（detached，re-noise 用）
      z0_hat     — (N, C) student x0 预测 = zt - t*v_proxy（有梯度到 v_proxy）
    """
    ops: StageOps
    vel_tracker: Optional[VelocityTracker] = None
    t_val: float = 0.0
    zt_feats: Optional[torch.Tensor] = None  # (N, C), detached
    z0_hat: Optional[torch.Tensor] = None    # (N, C), 有梯度到 v_proxy


# =====================================================================
# ContrastivePendingJob — Contrastive 异步流水线 micro-batch 基类
# =====================================================================

@dataclass
class ContrastivePendingJob:
    """
    Contrastive 异步流水线 micro-batch 基类 — stage-list 驱动。

    与 OnestepPendingJob 的核心区别：
    - drain 不做 decode（loss 在 x0 空间，3-arm contrastive）
    - 没有 per-stage submit（shared_prefix 统一提交一次 FlowEdit edit）
    - stage_entries: List[ContrastiveStageContext] 驱动，扩展新 stage 零改动

    梯度传播路径：
      contrastive_loss(z0_hat, pos, neg) → z0_hat → v_proxy → (relay) → v_student → θ.grad
      其中 z0_hat = zt - t·v_proxy（student x0 预测）
      pos = zt_stu - t·v_teacher_tgt（teacher 用 tgt 条件去噪 re-noised student x0）
      neg = zt_stu - t·v_teacher_src（teacher 用 src 条件去噪 re-noised student x0）

    扩展 shape-hq 时只需::

      stage_entries = [
          ContrastiveStageContext(ops=Trellis2ShapeOps()),
          ContrastiveStageContext(ops=Trellis2ShapeHQOps()),  # ← 新增一行
          ContrastiveStageContext(ops=Trellis2TexOpsFromShape()),
      ]

    子类职责：
    - 定义 create() 工厂方法（shared_prefix + per-stage create）
    - 定义 drain()（wait_edit + DINOv3 encode + per-stage drain）
    - 定义清理回调
    """

    state: Trellis2StateBase
    stage_entries: list  # List[ContrastiveStageContext]
    global_step: int = 0
    batch_size: int = 0
    src_image_pils: Optional[list] = None  # teacher PBR render → PIL
    edit_submitted: bool = False

    # ════════════════════════════════════════════════════════
    # 方案 2: 显存预释放（create(N) 前释放 prev 的大头）
    # ════════════════════════════════════════════════════════

    def pre_release_for_overlap(self) -> None:
        """create(N) 前释放 prev 的 rollout/decode 大项，降低两 Job 共存时的显存峰值。

        drain 只依赖：
          - edit_submitted / src_image_pils (CPU PIL)
          - vel_tracker (v_student, v_proxy, z0_hat, zt_feats, reg_grad, t_val)
          - state.views_conditioned.cond/uncond embeds (extract_embeddings / override_embeddings)
          - state.views_edited (wait_edit 时写入)
          - state.views_conditioned.image_pils + views_generated.pbr_tensor (save_tex_train)

        可以安全释放：shape.subs, shape.meshes, dense.coords, spatial cache
        注意：shape.z0 / tex.z0 不能释放 — drain 的 teacher denoise 需要 SparseTensor 的 coords/layout。
        """
        s = self.state
        # decode 产物
        s.shape.subs = None
        s.shape.meshes = None
        # dense sampling
        s.dense.coords = None
        # 兜底 spatial cache（create 末尾通常已释放，这里确保 GC 可达）
        s.release_shape_spatial_cache()
        s.release_tex_spatial_cache()
        _reclaim()

    # ════════════════════════════════════════════════════════
    # Building Block — encode images
    # ════════════════════════════════════════════════════════

    def _encode_images(
        self,
        system: Any,
        profiler: AsyncPhaseProfiler,
        image_pils: list,
        tick_label: str,
    ) -> Dict[int, torch.Tensor]:
        """
        DINOv3 encode images at all needed resolutions.

        遍历 stage_entries 收集所有 flow_resolution，对每个 resolution 调用
        pipeline.prepare_image_conditions 编码。

        Returns:
            {resolution: (B, S, C) tensor} 字典
        """
        profiler.tick(tick_label)
        pipeline = system.pipeline
        needed_resolutions = {
            entry.ops.get_flow_resolution(system)
            for entry in self.stage_entries
        }
        embeds: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for res in needed_resolutions:
                cond = pipeline.prepare_image_conditions(
                    image_pils, resolution=res,
                )
                embeds[res] = cond["cond"]  # (B, S, C)
        return embeds

    # ════════════════════════════════════════════════════════
    # Building Block — per-stage create
    # ════════════════════════════════════════════════════════

    def _create_stage(
        self,
        idx: int,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        对 stage_entries[idx] 执行:
          add_noise → student velocity → setup tracker → z0_hat → (可选) P3.5 velocity reg。

        ★ teacher rollout 已在 shared_prefix 完成，z₀ 已在 state 中。
        ★ create 不做 decode、不做 submit — 仅计算 student velocity + reg。

        Side Effects:
            entry.vel_tracker: VelocityTracker（v_student + v_proxy + 可选 reg_grad）
            entry.t_val: 采样时间步
            entry.zt_feats: (N, C) 加噪后特征（detached，re-noise 用）
            entry.z0_hat: (N, C) student x0 预测（有梯度到 v_proxy）
        """
        entry = self.stage_entries[idx]
        ops = entry.ops
        stage_name = ops.get_stage_name()
        prefix = f"{stage_name[0].upper()}_"

        # add_noise
        profiler.tick(f"{prefix}add_noise")
        z0_norm = ops.normalize_slat(ops.get_latent(self.state), system)
        z0_feats = z0_norm.feats.detach()
        t_val = ops.sample_timestep(system)
        zt_feats = ops.add_noise(z0_feats, t_val)

        # predict_cfg_velocity (student, with grad)
        profiler.tick(f"{prefix}stu_velocity")
        student_denoise_cfg = ops.get_student_denoise_cfg(system)
        with self.state.disable_uncond_embeddings(not student_denoise_cfg):
            v_student = ops.predict_cfg_velocity(self.state, system, zt_feats, t_val)

        tracker = VelocityTracker()
        tracker.setup_proxy(v_student)

        # ★ student x0 预测（有梯度通过 v_proxy leaf）
        z0_hat = zt_feats - t_val * tracker.v_proxy  # (N, C)

        entry.vel_tracker = tracker
        entry.t_val = t_val
        entry.zt_feats = zt_feats.detach()  # ★ re-noise 用
        entry.z0_hat = z0_hat               # ★ loss 的 anchor（有梯度到 v_proxy）

        # P3.5: velocity reg（可选）
        return self._velocity_reg(entry, zt_feats, system, profiler, prefix)

    # ════════════════════════════════════════════════════════
    # Building Block — P3.5 velocity reg
    # ════════════════════════════════════════════════════════

    def _velocity_reg(
        self,
        entry: 'ContrastiveStageContext',
        zt_feats: torch.Tensor,
        system: Any,
        profiler: AsyncPhaseProfiler,
        prefix: str,
    ) -> Dict[str, Any]:
        """
        P3.5 velocity reg: teacher velocity(src cond) → MSE → reg_grad。

        reg_weight == 0 时返回空 dict。
        reg_grad 存入 tracker，v_proxy.grad 清零 — relay 时与 guidance grad 合并。

        Returns:
            日志字典（bare keys）：loss/reg, grad_norm/reg
        """
        ops = entry.ops
        tracker = entry.vel_tracker
        reg_weight = ops.get_reg_weight(system)
        if reg_weight <= 0:
            return {}

        profiler.tick(f"{prefix}P3.5_reg")
        student_denoise_cfg = ops.get_student_denoise_cfg(system)
        with self.state.disable_uncond_embeddings(not student_denoise_cfg):
            v_teacher = ops.predict_cfg_velocity_teacher(
                self.state, system, zt_feats, entry.t_val,
            )  # (N, C), detached
        reg_loss = reg_weight * F.mse_loss(tracker.v_proxy, v_teacher)
        reg_loss.backward()  # → v_proxy.grad = reg_grad
        tracker.reg_grad = tracker.v_proxy.grad.detach().clone()
        tracker.reg_loss_val = reg_loss.item()
        tracker.v_proxy.grad = None  # ★ 清零，contrastive loss backward 将重新填充

        log: Dict[str, Any] = {
            "loss/reg": reg_loss.item(),
            "grad_norm/reg": tracker.reg_grad.norm().item(),
        }
        del v_teacher, reg_loss
        torch.cuda.empty_cache()
        return log

    # ════════════════════════════════════════════════════════
    # Building Block — per-stage drain (contrastive) — 编排器
    # ════════════════════════════════════════════════════════

    def _drain_contrastive_stage(
        self,
        idx: int,
        system: Any,
        profiler: AsyncPhaseProfiler,
        tgt_embeds: Optional[Dict[int, torch.Tensor]],
        src_embeds: Optional[Dict[int, torch.Tensor]],
        reg_log: Dict[str, Any],
        *,
        ada: bool = True,
        eps: float = 1e-4,
        collect_profiler: bool = False,
    ) -> Dict[str, Any]:
        """
        对 stage_entries[idx] 执行 contrastive drain — thin orchestrator。

        tgt_embeds/src_embeds 为 None 时自动退化为 reg-only relay（OOM 兜底路径）。

        流程 (正常):
          1. resolve embeds
          2. re-noise student x0 → zt_stu
          3. teacher denoise zt_stu with tgt cond → z0_tea_tgt (positive)
          4. teacher denoise zt_stu with src cond → z0_tea_src (negative)
          5. 3-arm contrastive loss(z0_hat, pos, neg) → backward → v_proxy.grad
          6. clip + relay → θ.grad
          7. log + cleanup

        流程 (reg-only fallback):
          1. v_proxy.grad = 0 → relay 只传 reg_grad → θ.grad
          2. log + cleanup

        ★ loss 在 x0 空间：z0_hat = zt - t·v_proxy（有梯度到 v_proxy）
        ★ teacher 从 student 预测的 x0 重新加噪后去噪（不是原始 zt_feats）
        """
        entry = self.stage_entries[idx]
        ops = entry.ops
        stage_name = ops.get_stage_name()
        prefix = f"{stage_name[0].upper()}_"

        has_contrastive = tgt_embeds is not None and src_embeds is not None

        if has_contrastive:
            # ── 正常 contrastive 路径 ──
            flow_res = ops.get_flow_resolution(system)
            tgt_emb = tgt_embeds[flow_res]
            src_emb = src_embeds[flow_res]

            # 2. re-noise student x0 → zt_stu
            profiler.tick(f"{prefix}re_noise")
            zt_stu = ops.add_noise(entry.z0_hat.detach(), entry.t_val)

            # 3. teacher denoise (tgt cond) → z0_tea_tgt (positive)
            z0_tea_tgt = self._teacher_denoise_x0(
                entry, zt_stu, tgt_emb, system, profiler, prefix, "tea_tgt",
            )

            # 4. teacher denoise (src cond) → z0_tea_src (negative)
            z0_tea_src = self._teacher_denoise_x0(
                entry, zt_stu, src_emb, system, profiler, prefix, "tea_src",
            )
            del zt_stu
            torch.cuda.empty_cache()

            # 5. contrastive loss → v_proxy.grad
            contrastive_weight = ops.get_guidance_weight(system)
            loss_log = self._contrastive_loss_backward(
                entry, z0_tea_tgt, z0_tea_src,
                contrastive_weight, ada, eps, profiler, prefix,
            )
        else:
            # ── reg-only fallback: v_proxy.grad = 0, relay 只传 reg_grad ──
            entry.vel_tracker.v_proxy.grad = torch.zeros_like(entry.vel_tracker.v_proxy)
            entry.z0_hat = None
            entry.zt_feats = None
            loss_log = {"loss/contrastive": 0.0, "fallback/reg_only": 1.0}

        # 6. clip + relay → θ.grad
        relay_log = self._clip_and_relay(entry, system, profiler, prefix)

        # 7. 合并日志 + cleanup
        log_prefix = f"{stage_name}/"
        if collect_profiler:
            profiler.tick("end")

        log: Dict[str, Any] = {}
        for sub_log in (reg_log, loss_log, relay_log):
            log.update({f"{log_prefix}{k}": v for k, v in sub_log.items()})
        log[f"{log_prefix}noise/t"] = entry.t_val
        if collect_profiler:
            log.update(profiler.collect(
                self.global_step,
                print_freq=int(system.cfg.freq.profiler),
            ))

        # cleanup
        del entry.vel_tracker.v_student, entry.vel_tracker.v_proxy
        return log

    # ════════════════════════════════════════════════════════
    # Helpers — contrastive drain 子步骤
    # ════════════════════════════════════════════════════════

    def _teacher_denoise_x0(
        self,
        entry: 'ContrastiveStageContext',
        zt_stu: torch.Tensor,
        cond_emb: torch.Tensor,
        system: Any,
        profiler: AsyncPhaseProfiler,
        prefix: str,
        tick_label: str,
    ) -> torch.Tensor:
        """
        override_embeddings(cond_emb) → teacher CFG velocity → x0。

        x0 = zt_stu - t * v_teacher。返回 detached z0_teacher。
        """
        ops = entry.ops
        flow_res = ops.get_flow_resolution(system)
        profiler.tick(f"{prefix}{tick_label}")
        teacher_cfg = bool(system.cfg.contrastive.teacher_cfg)
        with self.state.override_embeddings(cond_emb, resolution=flow_res), \
             self.state.disable_uncond_embeddings(not teacher_cfg):
            v_teacher = ops.predict_cfg_velocity_teacher(
                self.state, system, zt_stu, entry.t_val,
            )  # (N, C), detached
        z0_tea = zt_stu - entry.t_val * v_teacher  # (N, C)
        return z0_tea.detach()

    def _contrastive_loss_backward(
        self,
        entry: 'ContrastiveStageContext',
        z0_tea_tgt: torch.Tensor,
        z0_tea_src: torch.Tensor,
        contrastive_weight: float,
        ada: bool,
        eps: float,
        profiler: AsyncPhaseProfiler,
        prefix: str,
    ) -> Dict[str, Any]:
        """
        3-arm contrastive loss(z0_hat, pos=z0_tea_tgt, neg=z0_tea_src) → backward → v_proxy.grad。

        让 student 预测的 x0 靠近 teacher_tgt（正样本），远离 teacher_src（负样本）。

        Returns:
            日志字典（bare keys）：loss/contrastive, grad_norm/contrastive
        """
        profiler.tick(f"{prefix}contra_loss")
        loss = contrastive_weight * contrastive_loss_step(
            entry.z0_hat.unsqueeze(0),      # (1, N, C), 有梯度到 v_proxy
            z0_tea_tgt.unsqueeze(0),         # (1, N, C), detached
            z0_tea_src.unsqueeze(0),         # (1, N, C), detached
            ada=ada, eps=eps,
        )
        loss.backward()  # → v_proxy.grad = contrastive_grad

        log: Dict[str, Any] = {
            "loss/contrastive": loss.item(),
            "grad_norm/contrastive": entry.vel_tracker.v_proxy.grad.norm().item(),
        }

        del z0_tea_tgt, z0_tea_src, loss
        entry.z0_hat = None     # 释放
        entry.zt_feats = None   # 释放
        torch.cuda.empty_cache()
        return log

    def _clip_and_relay(
        self,
        entry: 'ContrastiveStageContext',
        system: Any,
        profiler: AsyncPhaseProfiler,
        prefix: str,
    ) -> Dict[str, Any]:
        """
        裁剪 v_proxy.grad → relay_and_backward → θ.grad。

        Returns:
            日志字典（bare keys）：grad_clip/clipped_ratio, grad_norm/guidance, grad_norm/ratio
        """
        ops = entry.ops
        tracker = entry.vel_tracker
        model = ops.get_model(system)
        log: Dict[str, Any] = {}

        # clip
        max_norm = ops.get_guidance_grad_max_norm(system)
        clipped = False
        if max_norm > 0:
            norm = tracker.v_proxy.grad.norm()
            if norm > max_norm:
                tracker.v_proxy.grad = tracker.v_proxy.grad * (max_norm / (norm + 1e-8))
                clipped = True
        log["grad_clip/clipped_ratio"] = float(clipped)

        # 裁剪后记录 guidance grad norm + ratio
        guid_norm = tracker.v_proxy.grad.norm().item()
        log["grad_norm/guidance"] = guid_norm
        if tracker.reg_grad is not None:
            reg_norm = tracker.reg_grad.norm().item()
            log["grad_norm/ratio"] = guid_norm / max(reg_norm, 1e-8)

        # relay → θ.grad
        profiler.tick(f"{prefix}relay")
        _no_sync = model.no_sync() if dist.is_initialized() else nullcontext()
        with TrainModeGuard(model):
            with _no_sync:
                tracker.relay_and_backward()
        return log

    # ════════════════════════════════════════════════════════
    # Building Block — DINO similarity + adaptive swap
    # ════════════════════════════════════════════════════════

    def _dino_similarity_and_swap(
        self,
        src_embeds: Dict[int, torch.Tensor],
        tgt_embeds: Dict[int, torch.Tensor],
        *,
        adaptive_swap: bool = False,
    ) -> Dict[str, Any]:
        """
        DINO cosine similarity 监控 + adaptive src/tgt 对调。

        1. 对每个 resolution，计算 src / tgt 与 input embed 的 cosine similarity
        2. adaptive_swap=True 时，用最后一个 resolution 的 sim 做判据，
           per-sample 对调 src / tgt（当 sim_src > sim_tgt 时交换）

        直接修改传入的 embed dict（in-place 替换 value）。

        Returns:
            日志字典：sim/{res}/src_input, sim/{res}/tgt_input,
                     sim/{res}/tgt_gt_src, sim/swap_rate
        """
        log: Dict[str, Any] = {}
        sim_src = sim_tgt = None

        # ── per-resolution cosine similarity ──
        for res in src_embeds:
            c_input, _ = self.state.extract_embeddings(resolution=res)
            e_input = c_input.mean(dim=1)  # (B, C)
            e_src = src_embeds[res].mean(dim=1)
            e_tgt = tgt_embeds[res].mean(dim=1)

            sim_src = torch.nn.functional.cosine_similarity(e_src, e_input, dim=-1)
            sim_tgt = torch.nn.functional.cosine_similarity(e_tgt, e_input, dim=-1)

            log[f"sim/{res}/src_input"] = sim_src.mean().item()
            log[f"sim/{res}/tgt_input"] = sim_tgt.mean().item()
            log[f"sim/{res}/tgt_gt_src"] = (sim_tgt > sim_src).float().mean().item()

        # ── adaptive swap: sim_src > sim_tgt → 交换 ──
        swap_mask = sim_src > sim_tgt  # (B,)
        log["sim/swap_rate"] = swap_mask.float().mean().item()
        if adaptive_swap and swap_mask.any():
            mask = swap_mask[:, None, None]  # (B, 1, 1)
            for res in src_embeds:
                c_src = src_embeds[res]
                c_tgt = tgt_embeds[res]
                src_embeds[res] = torch.where(mask, c_tgt, c_src)
                tgt_embeds[res] = torch.where(mask, c_src, c_tgt)
        return log

    # ════════════════════════════════════════════════════════
    # 便利方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        _reclaim()
