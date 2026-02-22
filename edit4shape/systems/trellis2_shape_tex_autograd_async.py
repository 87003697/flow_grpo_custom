"""
Trellis2 Shape+Tex 双阶段训练系统 — Autograd + 异步 Guidance 流水线版本（双阶段异步）。

核心类 PendingMicroBatch 统一管理一个 micro-batch 中 Shape + Tex 两阶段的完整计算生命周期：
- 两个独立的 proxy chain（shape_slat, tex_slat）
- 两个独立的 submitted / skip_vjp / guidance_log / tracker
- 共享一个 Trellis2State（生命周期由统一的清理方法管理）

curr = .create_shape(batch, ...)           ← Shape P1 + P2-ng + submit_S
    ├── dense_sampling → rollout → P2-ng (Normal) → submit_async
    └── submit 入 guidance FIFO 队列

_flush_shape(prev)                         ← Shape guid drain + vis + VJP + log
    ├── drain_shape_guidance → vis → drain_shape_vjp → log
    └── subs/meshes 保留（Tex P2-grad 还需要）

curr.create_tex(...)                       ← Tex P1 + P2-ng + submit_T
    ├── rollout_tex → P2-ng (PBR) → submit_async
    └── submit 入 guidance FIFO 队列

_flush_tex(prev)                           ← Tex guid drain + VJP + vis + log
    ├── drain_tex_guidance → drain_tex_vjp → vis → log
    └── subs/meshes 在 Tex P2-grad 后释放

prev = curr

每次迭代执行顺序（稳态，交替流水线）：
  ── Shape 半周期 ──
  1. curr = .create_shape(batch, ...)    ← Shape P1 + P2-ng + submit_S
  2. _flush_shape(prev)                  ← Shape guid drain + vis + VJP + log
  ── Tex 半周期 ──
  3. curr.create_tex(...)                ← Tex P1 + P2-ng + submit_T
  4. _flush_tex(prev)                    ← Tex guid drain + VJP + vis + log
  5. prev = curr

★ 交替异步优势：
  每次 submit 后，guidance 有另一个阶段的完整 drain（~38s）时间并行执行。
  相比统一 create 方案，submit 到下一次 submit 的间隔更短，
  guidance GPU 空闲时间从 ~10-20s 降低到 ~11s。

★ Proxy chain 管理关键：
  create_shape() 中 shape_slat 的 proxy chain 必须存活到 drain_shape_guidance 结束。
  不调用 _detach_shape_outputs（会断开 shape proxy chain）。
  tex rollout 所需的 shape_slat_norm 在 rollout_shape 中已 detach（line ~216 of
  rollout/shape.py），不影响 shape proxy chain。

★ 显存管理：
  create_shape() 后 curr 的 GPU 残留：
    shape proxy chain + shape tracker
    + cond/uncond embed + subs/meshes + cameras + vis tensors
  create_tex() 后 curr 的 GPU 残留增加：
    + tex proxy chain + tex tracker  ≈ 总计 1-3 GiB
  flush 逐步降低显存：
    shape_for_vjp:  释放 shape spatial_cache + detach shape_slat
                    + 释放 uncond + vis → CPU
                    + subs/meshes → CPU（降低 Shape VJP 显存水位）
    shape_p1_grad:  释放 shape tracker
    tex_p2_grad:    subs/meshes CPU → GPU（Tex decode+render 需要）
    tex_for_vjp:    释放 subs/meshes + tex spatial_cache + detach + offload
    tex_p1_grad:    释放 tex tracker + shape_slat_norm

DDP 安全：
- VJP 循环在 model.no_sync() 下执行，backward 不触发 DDP all-reduce
- 梯度同步由 _sync_grads_and_step() 在 optimizer.step 前手动 all-reduce(AVG)
- Shape 和 Tex 各自独立 sync + step
- 各 rank OOM 导致 VJP 迭代次数不同也不会死锁

正确性保证：
  Decoder (LayerNorm + SiLU, 无 Dropout/BatchNorm) 和 Renderer (纯数学运算)
  在 no_grad 和 grad 模式下行为完全一致，重跑 decode+render 得到相同 comp_rgb。

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ P2-grad guidance only  ↑ P1-ng autograd.grad        ↑ P1-grad 合并 VJP

特性：
- accum≥2 时收益最大：curr 的 guidance 与 prev 的全部 drain 全程并行
- accum=1 时退化为同步版（无并行窗口，但正确性不变）
- OOM 安全：Shape/Tex 独立降级（skip_shape_vjp / skip_tex_vjp）
- 评估路径仍使用 evaluate（内部自行调用 shape_forward + tex_forward）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import gc
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

# =====================================================================
# 第三方库导入
# =====================================================================
from absl import app
from ml_collections import config_flags

import torch
import torch.distributed as dist
from accelerate import Accelerator

# =====================================================================
# 从 trellis2_shape_tex.py 导入双阶段共享组件
# =====================================================================
from edit4shape.systems.trellis2_shape_tex import (
    Trellis2System,         # ★ 含 shape + tex 两个 StageSystem
    Trellis2State,          # ★ 含 tex_slat, pbr_tensor 等字段
    build_system,
    evaluate,
    decode_and_render_pbr,  # ★ PBR 渲染（Tex 阶段）
)

# =====================================================================
# Shape 阶段核心函数
# =====================================================================
from edit4shape.systems.trellis2_shape import decode_and_render_normal

from edit4shape.systems.trellis2_shape_autograd import (
    dense_sampling_no_grad,
    shape_phase1_rollout,
)

# =====================================================================
# Tex 阶段核心函数
# =====================================================================
from edit4shape.systems.trellis2_tex_autograd import tex_phase1_rollout

# =====================================================================
# Rollout & VJP
# =====================================================================
from edit4shape.generators.trellis2.rollout import RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity, _vjp_loader

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler
from edit4shape.systems.utils.logging import build_autograd_step_log

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from edit4shape.guidance.pipeline_parallel import AsyncGuidanceResult

# =====================================================================
# 数据加载器
# =====================================================================
from edit4shape.systems.trellis2 import build_dataloaders

# =====================================================================
# absl 配置
# =====================================================================
# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。


# =====================================================================
# PendingMicroBatch — Shape+Tex 双阶段异步流水线 micro-batch
# =====================================================================
#
# 不继承 PendingMicroBatchBase：基类假设单阶段（单 tracker / 单 submitted /
# 单 skip_vjp / 单 guidance_log），双阶段需要全部拆分为独立字段。
# 复用基类的 _log_mem / _reclaim 逻辑通过直接实现。
#
# 与 shape-only async / tex-only async 的核心差异：
#   1. 两个 proxy chain（shape_slat, tex_slat）共存，清理互不干扰
#   2. 两次 guidance submit / wait（FIFO 顺序：shape 先于 tex）
#   3. 两套独立的 skip flag + OOM 降级
#   4. Shape clean 保留 subs/meshes 给 Tex P2-grad
#   5. 两套独立的 optimizer sync + step
# =====================================================================

@dataclass
class PendingMicroBatch:
    """
    Shape+Tex 双阶段异步流水线 micro-batch（交替流水线版本）。

    统一管理 Shape 和 Tex 两个阶段的计算生命周期：
    - 两个独立的 proxy chain（shape_slat, tex_slat）
    - 两个独立的 tracker / submitted / skip_vjp / guidance_log
    - 共享一个 Trellis2State（生命周期由分阶段清理方法管理）

    生命周期（交替流水线）：
      .create_shape(batch, ...)        ← Shape P1 + P2-ng + submit_S
        ↓ _flush_shape(prev)           ← prev Shape guid drain + vis + VJP + log
      .create_tex(...)                 ← Tex P1 + P2-ng + submit_T
        ↓ _flush_tex(prev)             ← prev Tex guid drain + VJP + vis + log

    各阶段后的 GPU 状态：
      create_shape() 后：
          GPU: shape proxy chain, shape tracker,
               cond+uncond embed, subs, meshes, vis tensors
      create_tex() 后：
          GPU: + tex proxy chain, tex tracker
      _flush_shape(prev) 后：
          GPU(prev): shape tracker 已释放, subs/meshes 保留,
                     uncond 已释放, vis 已 offload CPU
      _flush_tex(prev) 后：
          GPU(prev): minimal residual

    ★ Guidance FIFO 约束：
      create_shape 中 submit_shape 先于 create_tex 中 submit_tex，
      _flush_shape 中 shape_wait 先于 _flush_tex 中 tex_wait，
      顺序严格一致。
    """

    state: Trellis2State
    shape_tracker: RolloutTracker
    tex_tracker: Optional[RolloutTracker] = None
    global_step: int = 0
    batch_size: int = 0

    # 双提交 flags
    shape_submitted: bool = False
    tex_submitted: bool = False

    # 双 skip flags（独立降级）
    skip_shape_vjp: bool = False
    skip_tex_vjp: bool = False

    # 双 guidance logs
    shape_guidance_log: Dict[str, Any] = field(default_factory=dict)
    tex_guidance_log: Dict[str, Any] = field(default_factory=dict)

    # ════════════════════════════════════════════════════════
    # 公开 API — create（拆分为 Shape / Tex 两个半周期）
    # ════════════════════════════════════════════════════════

    @classmethod
    def create_shape(
        cls,
        batch: Dict[str, Any],
        system: Trellis2System,
        global_step: int,
        profiler: AsyncPhaseProfiler,
    ) -> "PendingMicroBatch":
        """
        工厂方法（Shape 半周期）：Shape P1 + Shape P2-ng + submit_S。

        submit_shape 后，shape guidance 立即在 guidance GPU 开始，
        与后续操作（prev Shape drain/VJP）全程并行。

        ★ Proxy chain 管理：
          - shape_slat 在 rollout_shape 后有 proxy chain（连接 output_trajectory）
          - 不调用 _detach_shape_outputs（会断开 shape proxy chain）

        OOM 安全降级：
          - Shape P2-ng OOM → shape_submitted=False, subs/meshes 可能为 None
        """
        gen_seed_shape = int(system.cfg.seed) + global_step

        state = Trellis2State()
        state.attach_batch(
            batch, pipeline=system.pipeline,
            resolution=system.tex.config.cond_resolution,
        )

        batch_size = len(batch['image_pils'])

        # ── Shape P1: dense sampling + rollout ────────────────
        with TrainModeGuard(system.shape.model):
            profiler.tick("S_dense_sampling")
            dense_sampling_no_grad(state, system)

            profiler.tick("S_P1_rollout")
            shape_tracker = shape_phase1_rollout(state, system, gen_seed_shape)

        # 创建实例（submitted=False，P2 成功后置 True；tex_tracker 稍后由 create_tex 填充）
        inst = cls(
            state=state,
            shape_tracker=shape_tracker,
            global_step=global_step,
            batch_size=batch_size,
        )

        # ── Shape P2-no-grad + submit ────────────────────────
        try:
            profiler.tick("S_P2_no_grad")
            with torch.no_grad():
                shape_render = decode_and_render_normal(
                    state.features.shape_slat,
                    state.cameras,
                    system.pipeline,
                    system.shape.renderer,
                    system.accelerator.device,
                    resolution=system.pipeline.target_resolution,
                )
            # 存储 subs/meshes 供 Tex P2 使用
            state.features.subs = shape_render["subs"]
            state.features.meshes = shape_render["meshes"]
            comp_rgb = shape_render["color"]  # (B, V, H, W, 3)
            state.views_generated.shape_tensor = comp_rgb.detach()

            profiler.tick("S_P2_submit")
            system.guidance.submit_async(
                comp_rgb,
                state.views_conditioned.image_pils,
                guidance_weight=system.cfg.shape.train.loss.guidance,
                guidance_cfg=system.cfg.shape.guidance,
                rank=system.accelerator.process_index,
            )
            inst.shape_submitted = True
            del comp_rgb, shape_render
        except torch.cuda.OutOfMemoryError:
            logging.warning(
                f"[Step {global_step}] Shape P2-no-grad OOM → shape reg-only"
            )
            profiler.reset()
        finally:
            # 释放 shape decoder 的 spatial_cache（保留 subs/meshes 给 Tex）
            state.release_shape_spatial_cache()
            torch.cuda.empty_cache()

        return inst

    def create_tex(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Tex 半周期：Tex P1 + Tex P2-ng + submit_T（原地修改 self）。

        ★ 调用时机：在 create_shape 和 prev 的 Shape flush 之后调用。
          shape_slat_norm 已在 rollout_shape 中 detach，不影响 shape proxy chain。

        submit_tex 后，tex guidance 立即在 guidance GPU 开始，
        与后续操作（prev Tex drain/VJP）全程并行。

        OOM 安全降级：
          - Shape P2 OOM → meshes=None → Tex P2 也降级
          - Tex P2-ng OOM → tex_submitted=False
        """
        gen_seed_tex = int(system.cfg.seed) + self.global_step + 1000
        state = self.state

        # ── Tex P1: rollout ──────────────────────────────────
        # ★ shape_slat_norm 已在 rollout_shape 中 detach，不影响 shape proxy chain
        with TrainModeGuard(system.tex.model):
            profiler.tick("T_P1_rollout")
            tex_tracker = tex_phase1_rollout(state, system, gen_seed_tex)
        self.tex_tracker = tex_tracker

        # ── Tex P2-no-grad + submit ─────────────────────────
        try:
            # ★ Shape P2 OOM 时 meshes 可能为 None → Tex P2 也降级
            if state.features.meshes is None:
                raise RuntimeError("meshes 不可用 (Shape P2 OOM)")

            profiler.tick("T_P2_no_grad")
            with torch.no_grad():
                tex_render = decode_and_render_pbr(
                    state.features.meshes,
                    state.features.tex_slat,
                    state.features.subs,
                    state.cameras,
                    system.pipeline,
                    system.tex.renderer,
                    system.accelerator.device,
                    resolution=system.pipeline.target_resolution,
                )
            comp_rgb = tex_render["color"]  # (B, V, H, W, 3)
            state.views_generated.pbr_tensor = comp_rgb.detach()

            profiler.tick("T_P2_submit")
            system.guidance.submit_async(
                comp_rgb,
                state.views_conditioned.image_pils,
                guidance_weight=system.cfg.tex.train.loss.guidance,
                guidance_cfg=system.cfg.tex.guidance,
                rank=system.accelerator.process_index,
            )
            self.tex_submitted = True
            del comp_rgb, tex_render
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logging.warning(
                f"[Step {self.global_step}] Tex P2-no-grad failed: {e} → tex reg-only"
            )
            profiler.reset()
        finally:
            # 释放 tex decoder 的 spatial_cache
            state.release_tex_spatial_cache()
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # 公开 API — drain guidance
    # ════════════════════════════════════════════════════════

    def drain_shape_guidance(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Shape P2: wait → P2-grad → clean。

        降级链路：
          shape_submitted=False / wait 失败 → skip_shape_vjp=True
          P2-grad OOM → _invalidate_shape_guidance() + skip_shape_vjp=True

        Postcondition：
          shape spatial_cache 已释放。
          subs/meshes 保留（供后续 tex P2-grad 使用）。
        """
        with TrainModeGuard(system.shape.model):
            rgb_grad = self._shape_p2_wait(system, profiler)
            if rgb_grad is not None:
                self._shape_p2_grad(rgb_grad, system, profiler)
            else:
                self.skip_shape_vjp = True
        self._clean_shape_for_vjp()

    def drain_tex_guidance(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Tex P2: wait → P2-grad → clean。

        降级链路：
          tex_submitted=False / wait 失败 → skip_tex_vjp=True
          P2-grad OOM → _invalidate_tex_guidance() + skip_tex_vjp=True

        Postcondition：
          subs/meshes 已释放。features 已 detach。vis 已 offload 到 CPU。
        """
        with TrainModeGuard(system.tex.model):
            rgb_grad = self._tex_p2_wait(system, profiler)
            if rgb_grad is not None:
                self._tex_p2_grad(rgb_grad, system, profiler)
            else:
                self.skip_tex_vjp = True
        self._clean_tex_for_vjp()

    # ════════════════════════════════════════════════════════
    # 公开 API — drain VJP
    # ════════════════════════════════════════════════════════

    def drain_shape_vjp(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Shape VJP → θ_shape.grad 累积 → clean shape tracker → 返回日志。

        skip_shape_vjp=True 时跳过 VJP（零梯度贡献），
        仅执行 _clean_shape_p1_grad 释放 tracker。
        """
        if not self.skip_shape_vjp:
            with TrainModeGuard(system.shape.model):
                profiler.tick("S_P1_grad")
                phase3_log = self._shape_vjp_loop(system)
        else:
            profiler.tick("S_P1_skip")
            phase3_log = {}
            logging.info(
                f"[Step {self.global_step}] 跳过 Shape VJP — "
                f"guidance 不可用或 P2 OOM，Shape 零梯度贡献"
            )
        self._clean_shape_p1_grad()

        # 合并日志（key 前缀 "shape/"）
        return build_autograd_step_log(
            self.shape_guidance_log,
            system.cfg.shape.train.loss.reg,
            phase3_log,
            prefix="shape/",
        )

    def drain_tex_vjp(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Tex VJP → θ_tex.grad 累积 → clean tex tracker → 返回日志 + profiler 计时。

        skip_tex_vjp=True 时跳过 VJP（零梯度贡献），
        仅执行 _clean_tex_p1_grad 释放 tracker。

        ★ profiler.collect() 在此调用（最后一个 drain 方法），收集整个步的计时数据。
        """
        if not self.skip_tex_vjp:
            with TrainModeGuard(system.tex.model):
                profiler.tick("T_P1_grad")
                phase3_log = self._tex_vjp_loop(system)
        else:
            profiler.tick("T_P1_skip")
            phase3_log = {}
            logging.info(
                f"[Step {self.global_step}] 跳过 Tex VJP — "
                f"guidance 不可用或 P2 OOM，Tex 零梯度贡献"
            )
        self._clean_tex_p1_grad()
        profiler.tick("end")

        # 合并日志（key 前缀 "tex/"）+ profiler 计时
        merged = build_autograd_step_log(
            self.tex_guidance_log,
            system.cfg.tex.train.loss.reg,
            phase3_log,
            prefix="tex/",
        )
        merged.update(profiler.collect(
            self.global_step,
            print_freq=int(system.cfg.freq.profiler),
        ))
        return merged

    # ════════════════════════════════════════════════════════
    # Phase 步骤 — Shape P2
    # ════════════════════════════════════════════════════════

    def _shape_p2_wait(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Optional[torch.Tensor]:
        """
        Shape P2-wait: 阻塞等待 guidance GPU → rgb_grad。返回 None 表示不可用。

        OOM: 断开 traceback 引用链 → 显存日志 → gc 回收 → None。
        其他异常: 日志告警 → None。
        """
        if not self.shape_submitted:
            return None

        profiler.tick("S_P2_wait")
        try:
            result = system.guidance.wait_and_get(
                target_device=system.accelerator.device,
            )
            return self._unpack_shape_guidance_result(result, profiler)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
            self._log_mem("Shape P2-wait OOM → skip shape VJP", warn=True)
            self._reclaim()
            return None
        except Exception as e:
            logging.warning(
                f"[Step {self.global_step}] Shape P2-wait failed: {e} → skip shape VJP"
            )
            return None

    def _shape_p2_grad(
        self,
        rgb_grad: torch.Tensor,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Shape P2-grad: 重跑 decode+render Normal（带梯度）→ backward。

        ★ 重跑的 decode 产生新的 subs/meshes（局部变量），不覆盖 state.features 中
          为 Tex P2-grad 保留的 no_grad 版本。

        OOM: 断开 traceback → _invalidate_shape_guidance → skip_shape_vjp=True。
        Finally: 释放 shape spatial_cache（重跑 decode 填充的）→ gc + empty_cache。
        """
        profiler.tick("S_P2_grad")
        comp_rgb = None
        try:
            # ★ 重跑 decode+render（带梯度）：产生新的 subs/meshes 作为局部变量
            #   backward 通过 shape_slat proxy chain → 填充 output_trajectory[t].grad
            render_out = decode_and_render_normal(
                self.state.features.shape_slat,
                self.state.cameras,
                system.pipeline,
                system.shape.renderer,
                system.accelerator.device,
                resolution=system.pipeline.target_resolution,
            )
            comp_rgb = render_out["color"]  # (B, V, H, W, 3)
            comp_rgb.backward(rgb_grad)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None  # 断开 traceback → frame locals 引用链
            self._invalidate_shape_guidance()
            self.skip_shape_vjp = True
            self._log_mem("Shape P2-grad OOM → skip shape VJP", warn=True)
        finally:
            del comp_rgb, rgb_grad
            # 释放重跑 decode 时填充到 shape_slat._spatial_cache 中的数据
            self.state.release_shape_spatial_cache()
            self._reclaim()
            self._log_mem("Shape P2-grad cleanup done")

    # ════════════════════════════════════════════════════════
    # Phase 步骤 — Tex P2
    # ════════════════════════════════════════════════════════

    def _tex_p2_wait(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Optional[torch.Tensor]:
        """
        Tex P2-wait: 阻塞等待 guidance GPU → rgb_grad。返回 None 表示不可用。
        """
        if not self.tex_submitted:
            return None

        profiler.tick("T_P2_wait")
        try:
            result = system.guidance.wait_and_get(
                target_device=system.accelerator.device,
            )
            return self._unpack_tex_guidance_result(result, profiler)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None
            self._log_mem("Tex P2-wait OOM → skip tex VJP", warn=True)
            self._reclaim()
            return None
        except Exception as e:
            logging.warning(
                f"[Step {self.global_step}] Tex P2-wait failed: {e} → skip tex VJP"
            )
            return None

    def _tex_p2_grad(
        self,
        rgb_grad: torch.Tensor,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Tex P2-grad: 重跑 decode+render PBR（带梯度）→ backward。

        ★ 使用 state.features.meshes/subs（来自 create() 的 no_grad 版本，
          充当常数参与计算），只有 tex_slat 有 proxy chain → 梯度仅回传到 tex_slat。

        OOM: 断开 traceback → _invalidate_tex_guidance → skip_tex_vjp=True。
        Finally: 释放 tex spatial_cache → gc + empty_cache。
        """
        profiler.tick("T_P2_grad")
        comp_rgb = None
        # ★ subs/meshes 在 _clean_shape_for_vjp 中已 offload 到 CPU，搬回 GPU
        self.state.reload_decode_cache_to_gpu(system.accelerator.device)
        try:
            render_out = decode_and_render_pbr(
                self.state.features.meshes,      # no_grad, reloaded from CPU
                self.state.features.tex_slat,    # ★ proxy chain
                self.state.features.subs,        # no_grad, reloaded from CPU
                self.state.cameras,
                system.pipeline,
                system.tex.renderer,
                system.accelerator.device,
                resolution=system.pipeline.target_resolution,
            )
            comp_rgb = render_out["color"]  # (B, V, H, W, 3)
            comp_rgb.backward(rgb_grad)
        except torch.cuda.OutOfMemoryError as e:
            e.__traceback__ = None
            self._invalidate_tex_guidance()
            self.skip_tex_vjp = True
            self._log_mem("Tex P2-grad OOM → skip tex VJP", warn=True)
        finally:
            del comp_rgb, rgb_grad
            self.state.release_tex_spatial_cache()
            self._reclaim()
            self._log_mem("Tex P2-grad cleanup done")

    # ════════════════════════════════════════════════════════
    # VJP Loops
    # ════════════════════════════════════════════════════════

    def _shape_vjp_loop(self, system: Trellis2System) -> Dict[str, Any]:
        """
        Shape VJP loop — 逐步/批量重算 f_θ，合并 guidance + reg 梯度 → θ_shape.grad 累积。
        显存 O(1)，不随步数增长。

        DDP 安全：整个 VJP 循环在 model.no_sync() 下执行。
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        stage_config = pipeline.get_stage_config("shape")
        flow_res = stage_config["flow_resolution"]
        reg_weight = system.cfg.shape.train.loss.reg

        cond_emb, _ = self.state.extract_embeddings(resolution=flow_res)
        cond_emb = cond_emb.to(device)  # (B, S, C)

        model = system.shape.model
        chunk_size = 4

        with model.no_sync():
            for x_t, t_batch, cond_k, v_grad, sc_k in _vjp_loader(
                self.shape_tracker, self.state.features.shape_slat,
                cond_emb, None, reg_weight, device, chunk_size,
            ):
                try:
                    cond_pred = _predict_velocity(
                        pipeline, x_t, t_batch, cond_k,
                        "shape", flow_res, sc_k,
                    )  # SparseTensor
                    cond_pred.feats.backward(v_grad)
                except torch.cuda.OutOfMemoryError:
                    logging.warning(
                        f"[Step {self.global_step}] Shape P1-grad OOM → partial grad"
                    )
                    break

        log: Dict[str, Any] = {}
        if self.shape_tracker.reg_loss_val is not None:
            log["loss/reg"] = self.shape_tracker.reg_loss_val
        return log

    def _tex_vjp_loop(self, system: Trellis2System) -> Dict[str, Any]:
        """
        Tex VJP loop — 逐步/批量重算 f_θ，合并 guidance + reg 梯度 → θ_tex.grad 累积。
        显存 O(1)，不随步数增长。

        ★ 与 shape VJP 的差异：
          - stage = "tex"
          - 传入 shape_cond = shape_slat_norm 作为 tex flow model 的 concat_cond

        DDP 安全：整个 VJP 循环在 model.no_sync() 下执行。
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        stage_config = pipeline.get_stage_config("tex")
        flow_res = stage_config["flow_resolution"]
        reg_weight = system.cfg.tex.train.loss.reg

        cond_emb, _ = self.state.extract_embeddings(resolution=flow_res)
        cond_emb = cond_emb.to(device)  # (B, S, C)

        # ★ Tex 独有：shape_slat_norm 作为 tex flow model 的 concat_cond
        shape_cond = self.state.features.shape_slat_norm

        model = system.tex.model
        chunk_size = 4

        with model.no_sync():
            for x_t, t_batch, cond_k, v_grad, sc_k in _vjp_loader(
                self.tex_tracker, self.state.features.tex_slat,
                cond_emb, shape_cond, reg_weight, device, chunk_size,
            ):
                try:
                    cond_pred = _predict_velocity(
                        pipeline, x_t, t_batch, cond_k,
                        "tex", flow_res, sc_k,
                    )  # SparseTensor
                    cond_pred.feats.backward(v_grad)
                except torch.cuda.OutOfMemoryError:
                    logging.warning(
                        f"[Step {self.global_step}] Tex P1-grad OOM → partial grad"
                    )
                    break

        log: Dict[str, Any] = {}
        if self.tex_tracker.reg_loss_val is not None:
            log["loss/reg"] = self.tex_tracker.reg_loss_val
        return log

    # ════════════════════════════════════════════════════════
    # Guidance 结果解包
    # ════════════════════════════════════════════════════════

    def _unpack_shape_guidance_result(
        self,
        result: AsyncGuidanceResult,
        profiler: AsyncPhaseProfiler,
    ) -> torch.Tensor:
        """解包 Shape guidance 结果 → rgb_grad + 挂载 vis + 填充 shape_guidance_log。"""
        rgb_grad = result.rgb_grad.detach()  # (B, V, H, W, 3)

        # 挂载可视化数据（shape-edited images）
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

        self.shape_guidance_log = log
        return rgb_grad

    def _unpack_tex_guidance_result(
        self,
        result: AsyncGuidanceResult,
        profiler: AsyncPhaseProfiler,
    ) -> torch.Tensor:
        """
        解包 Tex guidance 结果 → rgb_grad + 挂载 vis + 填充 tex_guidance_log。

        ★ 注意：这会覆盖 views_edited（之前由 shape guidance 设置）。
          调用方需在此之前保存 shape vis。
        """
        rgb_grad = result.rgb_grad.detach()  # (B, V, H, W, 3)

        # ★ 覆盖 views_edited（tex-edited images 替换 shape-edited images）
        self.state.views_edited.image_tensor = result.edited_imgs
        self.state.views_edited.trackers = result.trackers

        # 构建日志
        log: Dict[str, Any] = {}
        if result.loss_dict:
            log.update({f"loss/{k}": v for k, v in result.loss_dict.items()})
        log["loss/guidance"] = result.loss_scalar

        # guidance GPU 计时（覆盖 shape 的计时，OK）
        if result.guid_wall_start is not None:
            profiler.set_guid_timing(result.guid_wall_start, result.guid_wall_end)

        self.tex_guidance_log = log
        return rgb_grad

    # ════════════════════════════════════════════════════════
    # Guidance 降级
    # ════════════════════════════════════════════════════════

    def _invalidate_shape_guidance(self) -> None:
        """OOM 降级：清空 shape output_trajectory 梯度 + shape guidance 日志。"""
        for out_t in self.shape_tracker.output_trajectory:
            out_t.grad = None
        self.shape_guidance_log = {}

    def _invalidate_tex_guidance(self) -> None:
        """OOM 降级：清空 tex output_trajectory 梯度 + tex guidance 日志。"""
        for out_t in self.tex_tracker.output_trajectory:
            out_t.grad = None
        self.tex_guidance_log = {}

    # ════════════════════════════════════════════════════════
    # 清理方法
    # ════════════════════════════════════════════════════════

    def _clean_shape_for_vjp(self) -> None:
        """
        Shape P2 结束后清理。

        ★ 保留 subs/meshes（Tex P2-grad 还需要）
        ★ 单独 detach shape_slat（P2-grad backward 已消费完 proxy chain）
        ★ 不 detach tex_slat（Tex P2-grad backward 还需要 proxy chain）
        ★ 提前释放 uncond / vis — VJP 和 Tex P2-grad 都不需要
        """
        s = self.state
        # shape spatial_cache 已在 _shape_p2_grad 的 finally 中释放（兜底）
        s.release_shape_spatial_cache()
        # ★ 不释放 subs/meshes — Tex P2-grad 还需要
        # ★ 单独 detach shape_slat（P2-grad backward 已消费完其 proxy chain，
        #   VJP 只需 .coords 做 .replace()，不需要 grad_fn）
        if s.features.shape_slat is not None:
            s.features.shape_slat = s.features.shape_slat.detach()
        # ★ 不 detach tex_slat — Tex P2-grad backward 还需要 proxy chain
        s.regularization.reg_loss = None
        s.release_uncond_embeddings()    # VJP 和 Tex P2-grad 都不需要
        s.offload_vis_to_cpu()           # vis tensor → CPU（save 在 CPU 也能工作）
        s.offload_decode_cache_to_cpu()  # ★ subs/meshes → CPU（降低 Shape VJP 显存水位）
        self._reclaim()

    def _clean_tex_for_vjp(self) -> None:
        """
        Tex P2 结束后清理：释放 VJP 不需要的所有 GPU 数据。

        ★ 释放 subs/meshes（Shape VJP 和 Tex VJP 都不需要）
        ★ detach features（proxy chain 已消费完毕）
        ★ 保留 shape_slat_norm（Tex VJP 需要作为 concat_cond）
        ★ 保留 tex_slat coords（Tex VJP 通过 .replace() 构建 x_t）
        """
        s = self.state
        # tex spatial_cache 已在 _tex_p2_grad 的 finally 中释放（兜底）
        s.release_tex_spatial_cache()
        s.prepare_for_tex_vjp()          # 释放 subs/meshes + tex_spatial_cache（兜底）
        s.detach_features()              # proxy chain → detached（shape + tex 都 detach）
        # ★ shape_slat 可置 None（VJP 只需 shape_slat.coords → detach 后仍可用）
        # ★ 保留 shape_slat_norm（Tex VJP 需要）
        # ★ 保留 tex_slat（VJP 需要 .coords 通过 .replace() 构建 x_t）
        s.regularization.reg_loss = None
        s.release_uncond_embeddings()    # VJP 只需 cond
        s.offload_vis_to_cpu()           # vis tensor → CPU
        self._reclaim()

    def _clean_shape_p1_grad(self) -> None:
        """Shape VJP 完成后：释放 shape tracker + VJP 产生的 spatial cache。"""
        self.state.release_shape_spatial_cache()
        del self.shape_tracker.input_trajectory[:], self.shape_tracker.output_trajectory[:]
        del self.shape_tracker.timesteps[:], self.shape_tracker.reg_grads[:]
        self._reclaim()

    def _clean_tex_p1_grad(self) -> None:
        """
        Tex VJP 完成后：释放 tex tracker + shape_slat_norm + VJP 产生的 spatial cache。

        ★ shape_slat_norm 在 Tex VJP 后不再需要，可以安全释放。
        """
        self.state.release_tex_spatial_cache()
        self.state.release_shape_decode_cache()  # 释放 subs/meshes/shape_slat_norm（兜底）
        del self.tex_tracker.input_trajectory[:], self.tex_tracker.output_trajectory[:]
        del self.tex_tracker.timesteps[:], self.tex_tracker.reg_grads[:]
        self._reclaim()

    # ════════════════════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════════════════════

    def _log_mem(self, tag: str, *, warn: bool = False) -> None:
        """记录 GPU 显存状态。"""
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        msg = (
            f"[Step {self.global_step}] {tag} | "
            f"allocated={alloc:.2f} GiB, reserved={resv:.2f} GiB"
        )
        (logging.warning if warn else logging.info)(msg)

    @staticmethod
    def _reclaim() -> None:
        """gc.collect + empty_cache 二连。"""
        gc.collect()
        torch.cuda.empty_cache()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd + 异步 Guidance 双阶段版本）。

    同时训练 Shape 和 Tex 两个 Flow Model：
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理
    训练策略使用三阶段 Autograd + 双阶段异步 Guidance 流水线（显存 O(1)）。

    流程: Dense Sampling → Shape Rollout → Normal 渲染 → Tex Rollout → PBR 渲染
         → 异步 Guidance × 2 → VJP × 2

    配置文件示例：
        python -m edit4shape.systems.trellis2_shape_tex_autograd_async \\
            --config=config/trellis2_shape_tex_distillation.py
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator（含 wandb 日志）
    # =====================================================
    use_wandb = cfg.use_wandb
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)

    # 初始化 wandb trackers
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape+tex-distillation-async",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(
        visuals_train_dir, target_h=cfg.renderer.resolution,
        vis_freq=vis_freq, accelerator=accelerator,
    )

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(
        cfg, accelerator,
        guidance_factory=partial(create_guidance, use_pp=True),  # ★ 异步 Guidance
    )
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_optimizers(accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（双阶段异步 Guidance 流水线）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    profiler = AsyncPhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    if accum_steps < 2 and accelerator.is_main_process:
        logging.warning(
            "[AsyncPipeline] gradient_accumulation_steps=%d, "
            "异步流水线需要 accum≥2 才有并行收益。"
            "当前退化为同步模式，建议增大 accum_steps。",
            accum_steps,
        )

    def _flush_shape(pending: PendingMicroBatch) -> None:
        """
        Shape 完整 drain: guid(P2-wait + P2-grad) → vis → VJP → log。

        ★ Shape vis 必须在 _flush_tex 之前保存
          （drain_tex_guidance 会覆盖 views_edited）。
        ★ subs/meshes 保留（Tex P2-grad 还需要）。
        """
        step, bs = pending.global_step, pending.batch_size

        # ── Shape guidance drain ──────────────────────────
        pending.drain_shape_guidance(system, profiler)

        # ★ Shape vis（在 Tex guidance 覆盖 views_edited 之前保存）
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)

        # ── Shape VJP ─────────────────────────────────────
        shape_log = pending.drain_shape_vjp(system, profiler)
        shape_logger.log_step(shape_log, bs, step, epoch)

    def _flush_tex(pending: PendingMicroBatch) -> None:
        """
        Tex 完整 drain: guid(P2-wait + P2-grad) → VJP → vis → log → reclaim。

        前提：_flush_shape 已完成。
        Tex P2-grad 使用 subs/meshes 后释放 → Tex VJP 在最干净状态运行。
        """
        step, bs = pending.global_step, pending.batch_size

        # ── Tex guidance drain ────────────────────────────
        # Postcondition: subs/meshes 已释放，features 已 detach
        pending.drain_tex_guidance(system, profiler)

        # ── Tex VJP ───────────────────────────────────────
        tex_log = pending.drain_tex_vjp(system, profiler)

        # ★ Tex vis（VJP 后 vis 已 offload 到 CPU，save 仍可工作）
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)

        tex_logger.log_step(tex_log, bs, step, epoch)
        pending._reclaim()

    def _sync_grads_and_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_accumulated: int,
    ) -> None:
        """
        手动 all-reduce 梯度 → 除以实际累积数 → optimizer.step → zero_grad。

        VJP 循环在 model.no_sync() 下执行，不触发 DDP 自动 all-reduce。
        因此需要在 optimizer.step() 前手动做一次跨 rank 梯度同步。
        单卡 / 非分布式环境下跳过 all-reduce，直接 step。

        Args:
            model: DDP 模型
            optimizer: 对应的优化器
            n_accumulated: 本次 step 实际累积的 micro-batch 数
        """
        # 1. 跨 rank 梯度同步
        if dist.is_initialized():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # 2. 除以实际累积的 micro-batch 数，得到平均梯度
        if n_accumulated > 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(n_accumulated)
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        prev: Optional[PendingMicroBatch] = None      # 双缓冲：上一个已 submit 的 MB

        for batch in train_loader:
            global_step += 1

            # ── Shape 半周期：curr Shape 前向 + prev Shape drain ──
            # submit_S 后 shape guidance 立即在 guidance GPU 开始，
            # 与下面的 prev Shape drain/VJP 全程并行。
            curr = PendingMicroBatch.create_shape(batch, system, global_step, profiler)
            if prev is not None:
                _flush_shape(prev)

            # ── Tex 半周期：curr Tex 前向 + prev Tex drain ────────
            # submit_T 后 tex guidance 立即在 guidance GPU 开始，
            # 与下面的 prev Tex drain/VJP 全程并行。
            curr.create_tex(system, profiler)
            if prev is not None:
                _flush_tex(prev)

            # ── prev ← curr ─────────────────────────────────
            prev = curr
            # ★ 老 prev 延迟释放：SparseTensor._spatial_cache 中的
            #   GPU 索引张量在此刻才真正解引用，需要 gc + empty_cache
            #   确保在下一个 create_shape 前回收。
            curr._reclaim()

            # ── Optimizer Step（在 accum 边界） ─────────────
            if global_step % accum_steps == 0:
                if prev is not None:
                    _flush_shape(prev)
                    _flush_tex(prev)
                    prev = None
                _sync_grads_and_step(system.shape.model, system.shape.optimizer, accum_steps)
                _sync_grads_and_step(system.tex.model, system.tex.optimizer, accum_steps)

        # ── epoch 结束：消化残留的 prev ─────────────────────
        if prev is not None:
            _flush_shape(prev)
            _flush_tex(prev)
            prev = None
        # ★ 独立于 prev：只要不在 accum 边界，就有待 step 的残留梯度
        #   （即使最后几个 MB 全 OOM → prev=None，之前 flush 的梯度仍需 step）
        remainder = global_step % accum_steps
        if remainder != 0:
            _sync_grads_and_step(system.shape.model, system.shape.optimizer, remainder)
            _sync_grads_and_step(system.tex.model, system.tex.optimizer, remainder)

        # ---- 周期性评估（epoch 级别）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
