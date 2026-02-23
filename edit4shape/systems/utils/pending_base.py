"""
PendingMicroBatch 基类 + StageContext — 异步 Guidance 流水线的公共抽象。

分层设计：
  StageContext         — per-stage 异步状态（ops + tracker + flags + log）
  ctx_* 自由函数       — 参数化的通用阶段步骤（drain / VJP / unpack）
  PendingMicroBatchBase — 单阶段 micro-batch 基类（内部使用 StageContext + ctx_*）

使用方式::

  # 单阶段子类 — 继承 PendingMicroBatchBase，只需实现 3 个清理方法
  class ShapePendingMB(PendingMicroBatchBase):
      _clean_p2_decode / _clean_for_vjp / _clean_p1_grad

  # 多阶段 — 直接使用 StageContext + ctx_* 自由函数
  class DualStagePendingMB:
      shape_ctx: StageContext
      tex_ctx: StageContext
      def drain_shape_guidance(...):
          ctx_p2_wait(self.shape_ctx, ...) → ctx_p2_grad(...)

导出清单（供子类/多阶段类导入）：
  StageContext, PendingMicroBatchBase,
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
# StageContext — 单个阶段的异步状态
# =====================================================================

@dataclass
class StageContext:
    """
    单个 stage 在一个 micro-batch 中的异步状态。

    绑定 StageOps 实例 + RolloutTracker + 控制 flags + guidance 日志。
    ctx_* 自由函数通过此结构参数化地操作任意阶段，
    消除 shape/tex 两套 drain/VJP 的重复代码。

    字段：
      ops            — StageOps 实现（提供 model/stage_name/slat/decode 等查询）
      tracker        — RolloutTracker（含 input/output trajectory, timesteps, reg_grads）
      submitted      — P2-no-grad + submit 是否成功
      skip_vjp       — 是否跳过 VJP（P2 OOM / guidance 不可用时置 True）
      guidance_log   — guidance 阶段的日志字典
    """
    ops: StageOps
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
    """构建 VJP 阶段日志（reg loss）。"""
    log: Dict[str, Any] = {}
    if ctx.tracker.reg_loss_val is not None:
        log["loss/reg"] = ctx.tracker.reg_loss_val
    return log


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

    通过 ctx.ops.decode_render_dict 获取 comp_rgb，
    用 rgb_grad 做 backward → 填充 output_trajectory[t].grad。

    OOM / StageSkipError: ctx_invalidate + skip_vjp=True → 显存日志。
    Finally: del 大张量 → clean_decode → gc + empty_cache → 显存日志。

    Args:
        pre_grad:     在 decode 前执行的回调（如 reload_decode_cache_to_gpu）
        clean_decode: decode 后的清理回调（如 release_spatial_cache）。
                      不需要调用 _reclaim — 本函数末尾统一调用。
    """
    profiler.tick(f"{prefix}P2_grad")
    comp_rgb = None
    if pre_grad is not None:
        pre_grad()
    try:
        comp_rgb = ctx.ops.decode_render_dict(state, system)["color"]
        comp_rgb.backward(rgb_grad)
    except (torch.cuda.OutOfMemoryError, StageSkipError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
        ctx_invalidate(ctx)
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
    ctx: StageContext,
    state: Trellis2StateBase,
    global_step: int,
    system: Any,
    chunk_size: int = 4,
) -> Dict[str, Any]:
    """
    通用 VJP loop — 逐步/批量重算 f_θ，合并 guidance + reg 梯度 → θ.grad 累积。
    显存 O(1)，不随步数增长。

    通过 ctx.ops 获取：
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
    ops = ctx.ops
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

    return ctx_build_vjp_log(ctx)


# =====================================================================
# PendingMicroBatchBase — 单阶段 micro-batch 基类
# =====================================================================

@dataclass
class PendingMicroBatchBase:
    """
    单阶段异步流水线 micro-batch 基类。

    内部使用 StageContext + ctx_* 自由函数，消除 shape/tex 的重复代码。
    子类只需实现 3 个清理方法（阶段特有的 GPU 资源释放策略）。

    生命周期：
      .create(batch, ...)           ← 子类实现（阶段特有的 rollout + submit）
      .drain_guidance(system, ...)  ← 基类：ctx_p2_wait → ctx_p2_grad → _clean_for_vjp
      .drain_vjp(system, ...)       ← 基类：ctx_vjp_loop → _clean_p1_grad → 合并日志

    子类必须实现（3 个清理方法）：
      _clean_p2_decode()  — 释放 decode+render 中间产物
      _clean_for_vjp()    — P2 结束后释放 VJP 不需要的数据
      _clean_p1_grad()    — VJP 完成后释放 tracker 数据

    与旧版本的差异：
      旧版本需要实现 7 个抽象方法（_get_model / _get_reg_weight / _decode_and_render /
      _p1_grad / _clean_p2_decode / _clean_for_vjp / _clean_p1_grad），
      新版本通过 StageContext.ops 委托前 4 个到 StageOps → 只需 3 个清理方法。
    """

    state: Trellis2StateBase
    ctx: StageContext
    global_step: int = 0
    batch_size: int = 0

    # ════════════════════════════════════════════════════════
    # 公开 API
    # ════════════════════════════════════════════════════════

    def drain_guidance(
        self,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        P2 全流程: wait → grad → 清理。

        降级链路:
          submitted=False / wait 失败 → skip_vjp=True
          P2-grad OOM → ctx_invalidate + skip_vjp=True

        Postcondition:
          _clean_for_vjp() 执行完毕，GPU 显存水位降低。
        """
        with TrainModeGuard(self.ctx.ops.get_model(system)):
            rgb_grad = ctx_p2_wait(
                self.ctx, self.state, self.global_step,
                system, profiler,
            )
            if rgb_grad is not None:
                ctx_p2_grad(
                    self.ctx, self.state, self.global_step,
                    system, profiler, rgb_grad,
                    clean_decode=self._clean_p2_decode,
                )
            else:
                self.ctx.skip_vjp = True
        self._clean_for_vjp()

    def drain_vjp(
        self,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        P1-grad VJP → θ.grad 累积 → 清理 tracker → 返回合并日志。

        skip_vjp=True 时跳过整个 VJP 循环（零梯度贡献），
        仅执行 _clean_p1_grad 释放 tracker。
        """
        ctx = self.ctx
        if not ctx.skip_vjp:
            with TrainModeGuard(ctx.ops.get_model(system)):
                profiler.tick("P1_grad")
                phase3_log = ctx_vjp_loop(
                    ctx, self.state, self.global_step, system,
                )
        else:
            profiler.tick("P1_skip")
            phase3_log = {}
            logging.info(
                f"[Step {self.global_step}] 跳过 VJP — "
                f"guidance 不可用或 P2 OOM，该 MB 零梯度贡献"
            )
        self._clean_p1_grad()
        profiler.tick("end")

        # 合并日志
        merged = build_autograd_step_log(
            ctx.guidance_log, ctx.ops.get_reg_weight(system), phase3_log,
        )
        merged.update(profiler.collect(
            self.global_step, print_freq=int(system.cfg.freq.profiler),
        ))
        return merged

    # ════════════════════════════════════════════════════════
    # 抽象方法（子类必须实现 — 仅 3 个清理方法）
    # ════════════════════════════════════════════════════════

    def _clean_p2_decode(self) -> None:
        """释放 decode+render 中间产物（P2-no-grad / P2-grad 共用）。"""
        raise NotImplementedError

    def _clean_for_vjp(self) -> None:
        """P2 结束后释放 VJP 不需要的数据（推荐末尾调用 self._reclaim()）。"""
        raise NotImplementedError

    def _clean_p1_grad(self) -> None:
        """VJP 完成后释放 tracker 数据（推荐末尾调用 self._reclaim()）。"""
        raise NotImplementedError

    # ════════════════════════════════════════════════════════
    # 便利方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        _reclaim()
