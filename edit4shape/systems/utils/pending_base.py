"""
PendingJob + StageContext — 异步 Guidance 流水线的公共抽象。

分层设计：
  StageContext         — per-stage 异步状态（tracker + flags + log，不含 ops）
  ctx_* 自由函数       — 参数化的通用阶段步骤（drain / VJP / unpack），ops 显式传入
  PendingJob           — micro-batch 基类（支持单/多阶段继承）

关注点分离：
  StageOps (策略)      — 数据无关的阶段策略，始终由调用方显式传入
  StageContext (状态)  — per-sample 异步生命周期，不绑定任何策略
  PendingJob (基类)    — 不持有 ctx，提供参数化 building block 方法

使用方式::

  # 单阶段子类 — 定义 ctx + 清理方法，组合 building block
  class ShapePendingJob(PendingJob):
      ctx: StageContext
      def drain_guidance(self, ops, system, profiler):
          self._drain_stage_guidance(ops, self.ctx, system, profiler,
              clean_decode=self._clean_p2_decode,
              clean_for_vjp=self._clean_for_vjp)

  # 多阶段子类 — 定义 shape_ctx + tex_ctx，各自组合 building block
  class DualPendingJob(PendingJob):
      shape_ctx: StageContext; tex_ctx: StageContext
      def drain_shape_guidance(self, ops, system, profiler):
          self._drain_stage_guidance(ops, self.shape_ctx, ..., prefix="S_", ...)

导出清单（供子类导入）：
  StageContext, PendingJob,
  ctx_p2_wait, ctx_p2_grad, ctx_vjp_loop,
  ctx_unpack_result, ctx_invalidate, ctx_build_vjp_log, ctx_clean_tracker,
  _log_mem, _reclaim
"""

# =====================================================================
# 标准库导入
# =====================================================================
import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# =====================================================================
# 第三方库导入
# =====================================================================
import torch

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.guidance.pipeline_parallel import AsyncGuidanceResult
from edit4shape.generators.trellis2.state import Trellis2State as Trellis2StateBase
from edit4shape.generators.trellis2.rollout import RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity, _vjp_loader
from edit4shape.systems.base import TrainModeGuard
from edit4shape.systems.utils import AsyncPhaseProfiler
from edit4shape.systems.utils.logging import build_autograd_step_log
from edit4shape.systems.utils.stage_ops import StageOps, StageSkipError


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


def ctx_build_vjp_log(ctx: StageContext) -> Dict[str, Any]:
    """构建 VJP 阶段日志（loss/reg + grad_norm/*）。"""
    return ctx.tracker.collect_log()


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
      - get_slat(state) → shape_slat / tex_slat
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

    slat = ops.get_slat(state)
    shape_cond = ops.get_shape_cond(state)
    model = ops.get_model(system)

    log = ctx_build_vjp_log(ctx)  # ★ VJP 前收集 loss/reg + grad_norm/*（.grad 尚未被消费）

    with model.no_sync():
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
            phase3_log = ctx_build_vjp_log(ctx)  # 即使跳过 VJP 也收集 loss/reg
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
