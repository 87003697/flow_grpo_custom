"""
PendingMicroBatch 基类 — 异步 Guidance 流水线的公共 OOM 处理 / 显存日志 / 清理逻辑。

子类：
  - trellis2_shape_autograd_async.PendingMicroBatch (Shape Normal 渲染)
  - trellis2_tex_autograd_async.PendingMicroBatch   (Tex PBR 渲染)

共享的完整方法（子类无需覆写）：
  drain_guidance / drain_vjp         ← 公开 API
  _p2_wait / _p2_grad               ← Phase 步骤
  _log_mem / _reclaim                ← 显存工具
  _invalidate_guidance               ← OOM 降级
  _unpack_guidance_result            ← Guidance 结果解包
  _build_vjp_log                     ← VJP 日志构建

子类必须实现：
  _get_model / _get_reg_weight       ← 阶段配置
  _decode_and_render / _p1_grad      ← 阶段特有 Phase 步骤
  _clean_p2_decode / _clean_for_vjp / _clean_p1_grad  ← 阶段特有清理

OOM 策略（基类已实现，子类无需覆写）：
  _on_p2_grad_oom          ← P2-grad OOM → skip_vjp=True
  _on_guidance_unavailable ← guidance 不可用 → skip_vjp=True
"""

# =====================================================================
# 标准库导入
# =====================================================================
import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
from edit4shape.systems.base import TrainModeGuard
from edit4shape.systems.utils import AsyncPhaseProfiler


@dataclass
class PendingMicroBatchBase:
    """
    异步流水线 micro-batch 基类。

    封装两个 async 版本（shape / tex）完全相同的：
    - OOM 处理（traceback 清理 + 显存日志 + gc 回收）
    - Guidance 结果解包（vis 挂载 + 日志构建 + profiler 计时）
    - P2 阶段（wait + grad + 清理编排）
    - VJP 阶段（skip_vjp 检查 + 日志合并）

    生命周期（子类实现 create，基类实现 drain_guidance / drain_vjp）：
      .create(batch, ...)           ← 子类实现（阶段特有的 rollout + submit）
      .drain_guidance(system, ...)  ← 基类：_p2_wait → _p2_grad → _clean_for_vjp
      .drain_vjp(system, ...)       ← 基类：_p1_grad → _clean_p1_grad → 合并日志
    """

    state: Trellis2StateBase
    tracker: RolloutTracker
    global_step: int = 0
    batch_size: int = 0
    submitted: bool = False
    guidance_log: Dict[str, Any] = field(default_factory=dict)
    skip_vjp: bool = False  # shape 始终 False；tex 在 P2 不可用时置 True

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
          submitted=False / wait 失败 → _on_guidance_unavailable()
          P2-grad OOM → _invalidate_guidance() + _on_p2_grad_oom()

        Postcondition:
          _clean_for_vjp() 执行完毕，GPU 显存水位降低。
        """
        with TrainModeGuard(self._get_model(system)):
            rgb_grad = self._p2_wait(system, profiler)
            if rgb_grad is not None:
                self._p2_grad(rgb_grad, system, profiler)
            else:
                self._on_guidance_unavailable()
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
        cfg = system.cfg
        if not self.skip_vjp:
            with TrainModeGuard(self._get_model(system)):
                profiler.tick("P1_grad")
                phase3_log = self._p1_grad(system)
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
        merged = {**self.guidance_log, **phase3_log}
        merged["loss/total"] = (
            self.guidance_log.get("loss/guidance", 0.0)
            + self._get_reg_weight(system) * phase3_log.get("loss/reg", 0.0)
        )
        merged.update(profiler.collect(self.global_step, print_freq=int(cfg.freq.profiler)))
        return merged

    # ════════════════════════════════════════════════════════
    # 公共辅助
    # ════════════════════════════════════════════════════════

    def _log_mem(self, tag: str, *, warn: bool = False) -> None:
        """记录 GPU 显存状态；allocated > 25 GiB 自动附加 memory_summary。"""
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        msg = (
            f"[Step {self.global_step}] {tag} | "
            f"allocated={alloc:.2f} GiB, reserved={resv:.2f} GiB"
        )
        (logging.warning if warn else logging.info)(msg)
        if alloc > 25.0:
            logging.warning(
                f"[Step {self.global_step}] ⚠️ Unusual memory after {tag}!\n"
                f"{torch.cuda.memory_summary(abbreviated=True)}"
            )

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        gc.collect()
        torch.cuda.empty_cache()

    def _invalidate_guidance(self) -> None:
        """OOM 降级：清空 output_trajectory 梯度 + guidance 日志。"""
        for out_t in self.tracker.output_trajectory:
            out_t.grad = None
        self.guidance_log = {}

    def _unpack_guidance_result(
        self,
        result: AsyncGuidanceResult,
        profiler: AsyncPhaseProfiler,
    ) -> torch.Tensor:
        """
        解包 AsyncGuidanceResult → rgb_grad + 挂载 vis + 填充 guidance_log + profiler 计时。

        Returns:
            rgb_grad: detached 梯度张量 (B, V, H, W, 3)
        """
        rgb_grad = result.rgb_grad.detach()  # (B, V, H, W, 3)

        # 挂载可视化数据
        self.state.views_edited.image_tensor = result.edited_imgs
        self.state.views_edited.trackers = result.trackers

        # 构建日志
        log: Dict[str, Any] = {}
        if result.loss_dict:
            log.update({f"loss/{k}": v for k, v in result.loss_dict.items()})
        log["loss/guidance"] = result.loss_scalar

        # guidance GPU 计时
        if result.guid_wall_start is not None:
            profiler.set_guid_timing(result.guid_wall_start, result.guid_wall_end)

        self.guidance_log = log
        return rgb_grad

    def _build_vjp_log(self) -> Dict[str, Any]:
        """构建 VJP 阶段日志（reg loss）。"""
        log: Dict[str, Any] = {}
        if self.tracker.reg_loss_val is not None:
            log["loss/reg"] = self.tracker.reg_loss_val
        return log

    # ════════════════════════════════════════════════════════
    # 公共 Phase 步骤
    # ════════════════════════════════════════════════════════

    def _p2_wait(
        self,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> Optional[torch.Tensor]:
        """
        P2-wait: 阻塞等待 guidance GPU → rgb_grad。返回 None 表示不可用。

        OOM: 断开 traceback 引用链 → 显存日志 → gc 回收 → None。
        其他异常: 日志告警 → None。
        """
        if not self.submitted:
            return None

        profiler.tick("P2_wait")
        try:
            result = system.guidance.wait_and_get(
                target_device=system.accelerator.device,
            )
            return self._unpack_guidance_result(result, profiler)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
            self._log_mem("P2-wait OOM → reg-only", warn=True)
            self._reclaim()
            return None
        except Exception as e:
            logging.warning(
                f"[Step {self.global_step}] P2-wait failed: {e} → reg-only"
            )
            return None

    def _p2_grad(
        self,
        rgb_grad: torch.Tensor,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        P2-grad: 重跑 decode+render（带梯度）→ backward。

        OOM: 断开 traceback → _invalidate_guidance → _on_p2_grad_oom 钩子 → 显存日志。
        Finally: del 大张量 → _clean_p2_decode → gc + empty_cache → 显存日志。
        """
        profiler.tick("P2_grad")
        comp_rgb = None
        try:
            comp_rgb = self._decode_and_render(system)["color"]
            comp_rgb.backward(rgb_grad)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
            self._invalidate_guidance()
            self._on_p2_grad_oom()
            self._log_mem("P2-grad OOM", warn=True)
        finally:
            del comp_rgb, rgb_grad
            self._clean_p2_decode()
            self._reclaim()
            self._log_mem("P2-grad cleanup done")

    # ════════════════════════════════════════════════════════
    # OOM 策略
    # ════════════════════════════════════════════════════════

    def _on_p2_grad_oom(self) -> None:
        """P2-grad OOM → 跳过 VJP（避免超大样本导致 NCCL timeout）。"""
        self.skip_vjp = True

    def _on_guidance_unavailable(self) -> None:
        """Guidance 不可用 → 跳过 VJP（避免超大样本导致 NCCL timeout）。"""
        self.skip_vjp = True

    # ════════════════════════════════════════════════════════
    # 抽象方法（子类必须实现）
    # ════════════════════════════════════════════════════════

    def _get_model(self, system: Any):
        """返回当前阶段的 DDP model（用于 TrainModeGuard / no_sync）。"""
        raise NotImplementedError

    def _get_reg_weight(self, system: Any) -> float:
        """返回 reg loss 权重。"""
        raise NotImplementedError

    def _decode_and_render(self, system: Any) -> Dict[str, Any]:
        """调用阶段特定的 decode+render，返回含 'color' 键的 dict。"""
        raise NotImplementedError

    def _p1_grad(self, system: Any) -> Dict[str, Any]:
        """P1-grad VJP 循环。返回日志 dict（通常通过 _build_vjp_log 构建）。"""
        raise NotImplementedError

    def _clean_p2_decode(self) -> None:
        """释放 decode+render 中间产物（P2-no-grad / P2-grad 共用）。"""
        raise NotImplementedError

    def _clean_for_vjp(self) -> None:
        """P2 结束后释放 VJP 不需要的数据（推荐末尾调用 self._reclaim()）。"""
        raise NotImplementedError

    def _clean_p1_grad(self) -> None:
        """VJP 完成后释放 tracker 数据（推荐末尾调用 self._reclaim()）。"""
        raise NotImplementedError
