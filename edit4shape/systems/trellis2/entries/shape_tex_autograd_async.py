"""
Trellis2 Shape+Tex 双阶段训练系统 — Autograd + 异步 Guidance 流水线版本（双阶段异步）。

核心类 PendingJob 统一管理一个 micro-batch 中 Shape + Tex 两阶段的完整计算生命周期：
- 两个 StageContext（shape_ctx, tex_ctx），各含独立的 tracker + flags + log
- 共享一个 Trellis2State（生命周期由统一的清理方法管理）
- drain/VJP 通过 ctx_* 自由函数参数化，消除 shape/tex 重复代码

curr = .create_shape(batch, ...)           ← Shape P1 + P2-ng + submit_S
    ├── ShapeOps.pre_rollout → rollout → decode_render(no_grad) → submit_async
    └── submit 入 guidance FIFO 队列

_flush_shape(prev)                         ← Shape guid drain + vis + VJP + log
    ├── ctx_p2_wait → ctx_p2_grad → vis → ctx_vjp_loop → log
    └── subs/meshes 保留（Tex P2-grad 还需要）

curr.create_tex(...)                       ← Tex P1 + P2-ng + submit_T
    ├── TexOpsFromShape.rollout → decode_render(no_grad) → submit_async
    └── submit 入 guidance FIFO 队列

_flush_tex(prev)                           ← Tex guid drain + VJP + vis + log
    ├── ctx_p2_wait → ctx_p2_grad → ctx_vjp_loop → vis → log
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
- OOM 安全：Shape/Tex 独立降级（skip_vjp per ctx）
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
import logging
from dataclasses import dataclass
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
# 项目内部导入
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import evaluate as _evaluate
from edit4shape.systems.trellis2.stage_ops import ShapeOps, TexOpsFromShape
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

# =====================================================================
# 基类 + StageContext 导入
# =====================================================================
from edit4shape.systems.utils.pending_base import (
    PendingJob as _PendingJobBase,
    StageContext,
    ctx_clean_tracker,
)
from edit4shape.systems.utils.stage_ops import StageSkipError

# =====================================================================
# absl 配置
# =====================================================================
# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。


# =====================================================================
# PendingJob — Shape+Tex 双阶段异步流水线 micro-batch
# =====================================================================
#
# 继承 PendingJob 基类，获得 building block 方法：
#   _drain_stage_guidance / _drain_stage_vjp
#
# 本类实现：
#   shape_ctx / tex_ctx                    ← 双阶段 StageContext
#   create_shape / create_tex              ← 工厂方法
#   drain_shape_guidance / drain_tex_guidance  ← 组合 building block
#   drain_shape_vjp / drain_tex_vjp        ← 组合 building block
#   _clean_shape_* / _clean_tex_*          ← 各阶段清理回调
#
# 与 shape-only async / tex-only async 的核心差异：
#   1. 两个 proxy chain（shape_slat, tex_slat）共存，清理互不干扰
#   2. 两次 guidance submit / wait（FIFO 顺序：shape 先于 tex）
#   3. 两套独立的 skip flag + OOM 降级
#   4. Shape clean 保留 subs/meshes 给 Tex P2-grad
#   5. 两套独立的 optimizer sync + step
# =====================================================================

@dataclass
class PendingJob(_PendingJobBase):
    """
    Shape+Tex 双阶段异步流水线 micro-batch（交替流水线版本）。

    继承 PendingJob 基类，通过 building block 组合 Shape 和 Tex 两阶段的
    drain/VJP 逻辑，消除大量重复代码。

    ops（ShapeOps / TexOpsFromShape）不存储在实例上 — 由调用方显式传入 drain 方法，
    与单阶段 PendingJob 风格一致。

    生命周期（交替流水线）：
      .create_shape(batch, ...)             ← Shape P1 + P2-ng + submit_S
        ↓ _flush_shape(prev)                ← prev Shape guid drain + vis + VJP + log
      .create_tex(...)                      ← Tex P1 + P2-ng + submit_T
        ↓ _flush_tex(prev)                  ← prev Tex guid drain + VJP + vis + log

    ★ Guidance FIFO 约束：
      create_shape 中 submit_shape 先于 create_tex 中 submit_tex，
      _flush_shape 中 shape_wait 先于 _flush_tex 中 tex_wait，
      顺序严格一致。
    """

    shape_ctx: Optional[StageContext] = None  # 由 create_shape() 设置
    tex_ctx: Optional[StageContext] = None    # 由 create_tex() 设置

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
    ) -> "PendingJob":
        """
        工厂方法（Shape 半周期）：Shape P1 + Shape P2-ng + submit_S。

        使用 ShapeOps 驱动所有 Shape 特有逻辑：
          pre_rollout(dense_sampling) → rollout → decode_render(no_grad) → submit

        submit_shape 后，shape guidance 立即在 guidance GPU 开始，
        与后续操作（prev Shape drain/VJP）全程并行。

        ★ Proxy chain 管理：
          - shape_slat 在 rollout_shape 后有 proxy chain（连接 output_trajectory）
          - 不调用 _detach_shape_outputs（会断开 shape proxy chain）

        OOM 安全降级：
          - Shape P2-ng OOM → shape_submitted=False, subs/meshes 可能为 None
        """
        ops = ShapeOps()
        gen_seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()

        state = Trellis2State()
        state.attach_batch(
            batch, pipeline=system.pipeline,
            resolution=system.tex.config.cond_resolution,
        )

        batch_size = len(batch['image_pils'])

        # ── Shape P1: pre_rollout + rollout ────────────────
        with TrainModeGuard(ops.get_model(system)):
            profiler.tick("S_dense_sampling")
            ops.pre_rollout(state, system, global_step)

            profiler.tick("S_P1_rollout")
            tracker = ops.rollout(state, system, gen_seed)

        # 创建实例（tex_ctx 稍后由 create_tex 填充）
        shape_ctx = StageContext(tracker=tracker)
        inst = cls(
            state=state,
            shape_ctx=shape_ctx,
            global_step=global_step,
            batch_size=batch_size,
        )

        # ── Shape P2-no-grad + submit ────────────────────────
        try:
            profiler.tick("S_P2_no_grad")
            with torch.no_grad():
                comp_rgb = ops.decode_render(state, system)

            profiler.tick("S_P2_submit")
            system.guidance.submit_async(
                comp_rgb,
                state.views_conditioned.image_pils,
                guidance_weight=ops.get_guidance_weight(system),
                guidance_cfg=ops.get_guidance_cfg(system),
                rank=system.accelerator.process_index,
            )
            shape_ctx.submitted = True
            del comp_rgb
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

        使用 TexOpsFromShape 驱动所有 Tex 特有逻辑：
          rollout → decode_render(no_grad) → submit

        ★ 调用时机：在 create_shape 和 prev 的 Shape flush 之后调用。
          shape_slat_norm 已在 rollout_shape 中 detach，不影响 shape proxy chain。

        submit_tex 后，tex guidance 立即在 guidance GPU 开始，
        与后续操作（prev Tex drain/VJP）全程并行。

        OOM 安全降级：
          - Shape P2 OOM → meshes=None → Tex decode_render 抛出 StageSkipError
          - Tex P2-ng OOM → tex_submitted=False
        """
        ops = TexOpsFromShape()
        gen_seed = int(system.cfg.seed) + self.global_step + ops.get_seed_offset()
        state = self.state

        # ── Tex P1: rollout ──────────────────────────────────
        # ★ shape_slat_norm 已在 rollout_shape 中 detach，不影响 shape proxy chain
        with TrainModeGuard(ops.get_model(system)):
            profiler.tick("T_P1_rollout")
            tracker = ops.rollout(state, system, gen_seed)

        self.tex_ctx = StageContext(tracker=tracker)

        # ── Tex P2-no-grad + submit ─────────────────────────
        try:
            profiler.tick("T_P2_no_grad")
            with torch.no_grad():
                # ★ TexOpsFromShape.decode_render 内部检查 meshes==None → StageSkipError
                comp_rgb = ops.decode_render(state, system)

            profiler.tick("T_P2_submit")
            system.guidance.submit_async(
                comp_rgb,
                state.views_conditioned.image_pils,
                guidance_weight=ops.get_guidance_weight(system),
                guidance_cfg=ops.get_guidance_cfg(system),
                rank=system.accelerator.process_index,
            )
            self.tex_ctx.submitted = True
            del comp_rgb
        except (torch.cuda.OutOfMemoryError, StageSkipError) as e:
            logging.warning(
                f"[Step {self.global_step}] Tex P2-no-grad failed: {e} → tex reg-only"
            )
            profiler.reset()
        finally:
            # 释放 tex decoder 的 spatial_cache
            state.release_tex_spatial_cache()
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # 公开 API — drain（组合基类 building block）
    # ════════════════════════════════════════════════════════

    def drain_shape_guidance(
        self,
        ops: ShapeOps,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Shape P2: wait → P2-grad → clean。

        Postcondition：
          shape spatial_cache 已释放。
          subs/meshes 保留（供后续 tex P2-grad 使用）。
        """
        self._drain_stage_guidance(
            ops, self.shape_ctx, system, profiler,
            prefix="S_",
            clean_decode=lambda: self.state.release_shape_spatial_cache(),
            clean_for_vjp=self._clean_shape_for_vjp,
        )

    def drain_tex_guidance(
        self,
        ops: TexOpsFromShape,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Tex P2: wait → P2-grad → clean。

        Postcondition：
          subs/meshes 已释放。features 已 detach。vis 已 offload 到 CPU。
        """
        self._drain_stage_guidance(
            ops, self.tex_ctx, system, profiler,
            prefix="T_",
            pre_grad=lambda: self.state.reload_decode_cache_to_gpu(
                system.accelerator.device,
            ),
            clean_decode=lambda: self.state.release_tex_spatial_cache(),
            clean_for_vjp=self._clean_tex_for_vjp,
        )

    def drain_shape_vjp(
        self,
        ops: ShapeOps,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """Shape VJP → θ_shape.grad 累积 → clean shape tracker → 返回日志。"""
        return self._drain_stage_vjp(
            ops, self.shape_ctx, system, profiler,
            prefix="S_",
            stage_name="Shape",
            chunk_size=6,
            clean_p1_grad=self._clean_shape_p1_grad,
            log_prefix="shape/",
            collect_profiler=False,
        )

    def drain_tex_vjp(
        self,
        ops: TexOpsFromShape,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Tex VJP → θ_tex.grad 累积 → clean tex tracker → 返回日志 + profiler 计时。

        ★ profiler.collect() 在此调用（最后一个 drain 方法），收集整个步的计时数据。
        """
        return self._drain_stage_vjp(
            ops, self.tex_ctx, system, profiler,
            prefix="T_",
            stage_name="Tex",
            chunk_size=6,
            clean_p1_grad=self._clean_tex_p1_grad,
            log_prefix="tex/",
            collect_profiler=True,
        )

    # ════════════════════════════════════════════════════════
    # 清理方法（阶段特有的 GPU 资源释放策略）
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
        # shape spatial_cache 已在 _drain_stage_guidance 的 clean_decode 中释放（兜底）
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
        # tex spatial_cache 已在 _drain_stage_guidance 的 clean_decode 中释放（兜底）
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
        ctx_clean_tracker(self.shape_ctx)
        self._reclaim()

    def _clean_tex_p1_grad(self) -> None:
        """
        Tex VJP 完成后：释放 tex tracker + shape_slat_norm + VJP 产生的 spatial cache。

        ★ shape_slat_norm 在 Tex VJP 后不再需要，可以安全释放。
        """
        self.state.release_tex_spatial_cache()
        self.state.release_shape_decode_cache()  # 释放 subs/meshes/shape_slat_norm（兜底）
        ctx_clean_tracker(self.tex_ctx)
        self._reclaim()


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
        python -m edit4shape.systems.trellis2.shape_tex_autograd_async \\
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
    system = _build_system(
        cfg, accelerator,
        guidance_factory=partial(create_guidance, use_pp=True),  # ★ 异步 Guidance
        mode="shape_tex",
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
        eval_log = _evaluate(
            system, epoch=start_epoch, global_step=global_step,
            eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
            with_tex=True,
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

    shape_ops = ShapeOps()              # 无状态策略对象，训练循环持有，drain 时传入
    tex_ops = TexOpsFromShape()          # 无状态策略对象，训练循环持有，drain 时传入

    # ★ 自适应梯度裁剪（TRELLIS.2 默认参数：max_norm=1.0, clip_percentile=95）
    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95)

    def _flush_shape(pending: PendingJob) -> None:
        """
        Shape 完整 drain: guid(P2-wait + P2-grad) → vis → VJP → log。

        ★ Shape vis 必须在 _flush_tex 之前保存
          （drain_tex_guidance 会覆盖 views_edited）。
        ★ subs/meshes 保留（Tex P2-grad 还需要）。
        """
        step, bs = pending.global_step, pending.batch_size

        # ── Shape guidance drain ──────────────────────────
        pending.drain_shape_guidance(shape_ops, system, profiler)

        # ★ Shape vis（在 Tex guidance 覆盖 views_edited 之前保存）
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)

        # ── Shape VJP ─────────────────────────────────────
        shape_log = pending.drain_shape_vjp(shape_ops, system, profiler)
        shape_logger.log_step(shape_log, bs, step, epoch)

    def _flush_tex(pending: PendingJob) -> None:
        """
        Tex 完整 drain: guid(P2-wait + P2-grad) → VJP → vis → log → reclaim。

        前提：_flush_shape 已完成。
        Tex P2-grad 使用 subs/meshes 后释放 → Tex VJP 在最干净状态运行。
        """
        step, bs = pending.global_step, pending.batch_size

        # ── Tex guidance drain ────────────────────────────
        # Postcondition: subs/meshes 已释放，features 已 detach
        pending.drain_tex_guidance(tex_ops, system, profiler)

        # ── Tex VJP ───────────────────────────────────────
        tex_log = pending.drain_tex_vjp(tex_ops, system, profiler)

        # ★ Tex vis（VJP 后 vis 已 offload 到 CPU，save 仍可工作）
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)

        tex_logger.log_step(tex_log, bs, step, epoch)
        pending._reclaim()

    def _sync_grads_and_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_accumulated: int,
        grad_clipper: AdaptiveGradClipper = None,
    ) -> None:
        """
        手动 all-reduce 梯度 → 除以实际累积数 → grad clip → optimizer.step → zero_grad。

        VJP 循环在 model.no_sync() 下执行，不触发 DDP 自动 all-reduce。
        因此需要在 optimizer.step() 前手动做一次跨 rank 梯度同步。
        单卡 / 非分布式环境下跳过 all-reduce，直接 step。

        Args:
            model: DDP 模型
            optimizer: 对应的优化器
            n_accumulated: 本次 step 实际累积的 micro-batch 数
            grad_clipper: 自适应梯度裁剪器（可选）
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
        # 3. 自适应梯度裁剪
        if grad_clipper is not None:
            grad_clipper(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        prev: Optional[PendingJob] = None      # 双缓冲：上一个已 submit 的 MB

        for batch in train_loader:
            global_step += 1

            # ── Shape 半周期：curr Shape 前向 + prev Shape drain ──
            # submit_S 后 shape guidance 立即在 guidance GPU 开始，
            # 与下面的 prev Shape drain/VJP 全程并行。
            curr = PendingJob.create_shape(batch, system, global_step, profiler)
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
                _sync_grads_and_step(system.shape.model, system.shape.optimizer, accum_steps, shape_grad_clipper)
                _sync_grads_and_step(system.tex.model, system.tex.optimizer, accum_steps, tex_grad_clipper)

        # ── epoch 结束：消化残留的 prev ─────────────────────
        if prev is not None:
            _flush_shape(prev)
            _flush_tex(prev)
            prev = None
        # ★ 独立于 prev：只要不在 accum 边界，就有待 step 的残留梯度
        #   （即使最后几个 MB 全 OOM → prev=None，之前 flush 的梯度仍需 step）
        remainder = global_step % accum_steps
        if remainder != 0:
            _sync_grads_and_step(system.shape.model, system.shape.optimizer, remainder, shape_grad_clipper)
            _sync_grads_and_step(system.tex.model, system.tex.optimizer, remainder, tex_grad_clipper)

        # ---- 周期性评估（epoch 级别）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = _evaluate(
                system, epoch=epoch, global_step=global_step,
                eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
                with_tex=True,
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
