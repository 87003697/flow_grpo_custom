"""
Trellis2 Tex 训练系统 — Autograd + 异步 Guidance 流水线版本。

核心类 PendingJob（继承 PendingJob）管理一个 micro-batch 的完整计算生命周期：

  PendingJob.create(batch, ...)     ← P0 + P1-ng + P2-ng + submit
      ├── pre_rollout: shape_frozen_prepare(no_grad) + detach
      ├── rollout:     rollout_tex → tracker (proxy chain)
      ├── P2-no-grad:  decode_render(no_grad) → comp_rgb + vis
      ├── P2-submit:   submit to guidance GPU
      └── _clean_p2_decode: 释放 tex decode cache

  prev.drain_guidance(...)                  ← 基类：P2-wait + P2-grad + clean
      ├── ctx_p2_wait:  等 guidance GPU → rgb_grad
      ├── ctx_p2_grad:  decode_render_dict(grad) → backward
      │   └── finally: _clean_p2_decode
      └── _clean_for_vjp: detach + release subs/meshes + offload + gc

  prev.drain_vjp(...)                       ← 基类：ctx_vjp_loop → θ.grad
      ├── ctx_vjp_loop: VJP loop (no_sync) → θ.grad 本地累积
      └── _clean_p1_grad: 释放 tracker

每次迭代执行顺序（稳态）：
  1. curr = .create(batch, ...)       ← 无梯度前向 + submit（guidance GPU 尽早开始）
  2. prev.drain_guidance(...)         ← P2-wait + P2-grad + clean
  3. prev.drain_vjp(...) + log + vis  ← VJP + 日志 + 可视化（guidance GPU 并行处理 curr）
  4. prev = curr

★ 异步优势：
  create(curr) 先于 drain_guidance(prev) 执行，guidance GPU 在 P2-grad + VJP
  全程并行处理 curr，异步窗口 ≈ P2-grad + VJP（而非仅 VJP）。
  代价：P2-grad 时 GPU 额外持有 curr 的残留数据（~0.5-1.5 GiB），
  已有 OOM 保护兜底。

★ 显存管理：
  _clean_for_vjp 调用 state 的原子方法（detach_features / release_uncond_embeddings
  / offload_vis_to_cpu）释放 proxy chain / uncond embed / vis→CPU，
  使 VJP 阶段显存水位大幅下降。

两层 proxy：
  1. cond_pred proxy (P1) — 显存隔离，不保留 flow model 计算图
  2. comp_rgb  proxy (异步) — 计算并行，train/guidance GPU 各自独立 backward

正确性保证：
  Decoder (LayerNorm + SiLU, 无 Dropout/BatchNorm) 和 Renderer (纯数学运算)
  在 no_grad 和 grad 模式下行为完全一致，重跑 decode+render 得到相同 comp_rgb。

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ P2-grad guidance only  ↑ P1-ng autograd.grad        ↑ P1-grad 合并 VJP

与 shape_autograd_async 的差异：
- 新增 Phase 0: shape_frozen_prepare（no_grad shape forward + detach）
- rollout_tex 替代 rollout_shape，rollout 需要 shape_slat_norm 作为条件
- decode_and_render_pbr 替代 decode_and_render_normal（需要 meshes + subs）
- VJP 传入 shape_cond = shape_slat_norm 作为 tex flow model 的 concat_cond
- 清理策略：_clean_for_vjp 保留 shape_slat_norm 供 VJP，_clean_p1_grad 释放之
- P2 OOM / guidance 不可用 → skip_vjp=True（覆写 _on_p2_grad_oom / _on_guidance_unavailable）

DDP 安全：
- VJP 循环在 model.no_sync() 下执行，backward 不触发 DDP all-reduce
- 梯度同步由 _sync_grads_and_step() 在 optimizer.step 前手动 all-reduce(AVG)
- 各 rank OOM 导致 VJP 迭代次数不同时不会死锁

特性：
- accum≥2 时收益最大：curr 的 guidance 与 prev 的 P2-grad + VJP 全程并行
- accum=1 时退化为同步版（无并行窗口，但正确性不变）
- 评估路径仍使用单阶段 forward（trellis2_tex_forward）
- OOM 安全：P2-no-grad/P2-grad OOM 均可降级到跳过 VJP

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
# Guidance 模块
# =====================================================================
from functools import partial
from edit4shape.guidance import create_guidance

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    TrainModeGuard,
    build_run_paths,
)
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler

# =====================================================================
# 从 system.py / forward.py 导入共享组件
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import (
    evaluate as _evaluate,
)
from edit4shape.systems.trellis2.stage_ops import Trellis2TexOps
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
# PendingJob — Tex 异步流水线 micro-batch
# =====================================================================
#
# 继承 PendingJob 基类，获得 building block 方法：
#   _drain_stage_guidance / _drain_stage_vjp
#
# 本类实现：
#   ctx                                    ← StageContext 字段
#   create                                 ← 工厂方法（使用 Trellis2TexOps）
#   drain_guidance / drain_vjp             ← 组合 building block 的公开 API
#   _clean_p2_decode / _clean_for_vjp / _clean_p1_grad  ← 3 个清理回调
#
# ★ 与 shape 版本的关键差异（全部编码在 Trellis2TexOps 中）：
#   1. pre_rollout = shape_frozen_prepare（shape forward + detach）
#   2. rollout = rollout_tex（需要 shape_slat_norm 作为条件）
#   3. decode_render_dict = decode_and_render_pbr（需要 meshes + subs）
#   4. get_shape_cond = shape_slat_norm（VJP 需要 concat_cond）
# =====================================================================

@dataclass
class PendingJob(_PendingJobBase):
    """
    Tex 异步流水线 micro-batch — 继承基类 building block。

    生命周期:
      .create(batch, ...)                ← 本类：shape_frozen_prepare + rollout_tex + submit
      .drain_guidance(ops, system, ...)  ← 本类：组合 _drain_stage_guidance
      .drain_vjp(ops, system, ...)       ← 本类：组合 _drain_stage_vjp

    各阶段后的 GPU 状态:
      create() 后:
          GPU: proxy chain, tracker, cond+uncond embed, shape_slat_norm, subs, meshes, vis tensor
      drain_guidance() 后:
          GPU: detached tex_slat, tracker, cond embed, shape_slat_norm
      drain_vjp() 后:
          GPU: tracker 已清空，仅剩 detached tex_slat + cond embed

    ★ comp_rgb 不存储：create 中仅用于 submit，drain_guidance 中重算。
    ★ shape_slat_norm 必须保留到 VJP 结束（作为 tex flow model 的 concat_cond）。
    """

    ctx: Optional[StageContext] = None  # 由 create() 设置

    # ════════════════════════════════════════════════════════
    # 公开 API — create
    # ════════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        batch: Dict[str, Any],
        system: Trellis2System,
        global_step: int,
        profiler: AsyncPhaseProfiler,
    ) -> "PendingJob":
        """
        工厂方法：P0 + P1-no-grad + P2-no-grad + submit → 创建 PendingJob。

        使用 Trellis2TexOps 驱动所有阶段特有逻辑：
          pre_rollout(shape_frozen_prepare) → rollout → decode_render(no_grad) → submit

        OOM 安全降级：
          P2-no-grad OOM → submitted=False → drain_guidance 跳过 P2-grad，跳过 VJP。
        """
        ops = Trellis2TexOps()
        gen_seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()

        with TrainModeGuard(ops.get_model(system)):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.tex.config.cond_resolution)

            # ── Phase 0: Shape 冻结前置（no_grad shape forward + detach） ──
            profiler.tick("shape_frozen_prepare")
            ops.pre_rollout(state, system, global_step)

            # ── P1: Tex rollout（proxy chain） ──────────────────
            profiler.tick("P1_rollout")
            tracker = ops.rollout(state, system, gen_seed)

            # 创建实例
            batch_size = len(batch['image_pils'])
            ctx = StageContext(tracker=tracker)
            inst = cls(state=state, ctx=ctx,
                       global_step=global_step, batch_size=batch_size)

            # ── P2-no-grad + submit ──────────────────────────────
            try:
                profiler.tick("P2_no_grad")
                with torch.no_grad():
                    comp_rgb = ops.decode_render(state, system)

                profiler.tick("P2_submit_async")
                system.guidance.submit_async(
                    comp_rgb,
                    state.views_conditioned.image_pils,
                    guidance_weight=ops.get_guidance_weight(system),
                    guidance_cfg=ops.get_guidance_cfg(system),
                    rank=system.accelerator.process_index,
                )
                ctx.submitted = True
                del comp_rgb
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    f"[Step {global_step}] P2-no-grad OOM → reg-only"
                )
                profiler.reset()
            except StageSkipError as e:
                logging.warning(
                    f"[Step {global_step}] P2-no-grad skipped: {e} → reg-only"
                )
                profiler.reset()
            finally:
                inst._clean_p2_decode()

        return inst

    # ════════════════════════════════════════════════════════
    # 公开 API — drain（组合基类 building block）
    # ════════════════════════════════════════════════════════

    def drain_guidance(
        self,
        ops: Trellis2TexOps,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """P2 全流程: wait → grad → 清理。"""
        self._drain_stage_guidance(
            ops, self.ctx, system, profiler,
            clean_decode=self._clean_p2_decode,
            clean_for_vjp=self._clean_for_vjp,
        )

    def drain_vjp(
        self,
        ops: Trellis2TexOps,
        system: Any,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """P1-grad VJP → θ.grad 累积 → 清理 tracker → 返回合并日志。"""
        return self._drain_stage_vjp(
            ops, self.ctx, system, profiler,
            clean_p1_grad=self._clean_p1_grad,
        )

    # ════════════════════════════════════════════════════════
    # 清理方法（3 个 — 阶段特有的回调）
    # ════════════════════════════════════════════════════════

    def _clean_p2_decode(self) -> None:
        """
        释放 tex decode+render 的中间产物（P2-no-grad / P2-grad 共用）。

        ★ 与 shape 版本的差异：仅清理 tex_slat._spatial_cache。
          subs/meshes 来自 Phase 0（detached），P2-grad 复用，不在此释放。
        """
        self.state.release_tex_spatial_cache()
        torch.cuda.empty_cache()

    def _clean_for_vjp(self) -> None:
        """
        P2 整体结束后：释放 VJP 不需要的 GPU 数据，降低显存水位。

        ★ 与 shape 版本的差异：
          - 必须保留 shape_slat_norm（VJP 需要作为 tex flow model 的 concat_cond）
          - 必须保留 tex_slat（VJP 需要其 coords 通过 .replace() 构建 x_t）
          - 释放 subs/meshes/shape_slat（VJP 不需要）
        """
        s = self.state
        # ★ 顺序：先 in-place 清 spatial cache + 释放 subs/meshes，再 detach
        s.release_shape_spatial_cache()  # 释放 shape 侧 spatial cache（兜底）
        s.prepare_for_tex_vjp()          # 释放 tex_spatial_cache + subs/meshes

        s.detach_features()              # proxy chain → detached（同时 detach tex_slat 和 shape_slat）

        # ★ 释放 shape_slat（VJP 只需要 shape_slat_norm，不需要 shape_slat）
        s.features.shape_slat = None
        # ★ 保留 shape_slat_norm（VJP 需要作为 tex flow model 的 concat_cond）
        # ★ 保留 tex_slat（VJP 需要其 coords 通过 .replace() 构建 x_t）

        s.regularization.reg_loss = None # reg 梯度已在 tracker.reg_grads
        s.release_uncond_embeddings()    # VJP 只需 cond
        s.offload_vis_to_cpu()           # vis tensor → CPU
        self._reclaim()

    def _clean_p1_grad(self) -> None:
        """
        VJP 完成后：释放 tracker 全部轨迹数据 + VJP 产生的 spatial cache。

        ★ 与 shape 版本的差异：
          额外释放 shape_slat_norm（VJP 后不再需要）。
        """
        self.state.release_tex_spatial_cache()
        # ★ VJP 后 shape_slat_norm 不再需要
        self.state.release_shape_decode_cache()
        ctx_clean_tracker(self.ctx)
        self._reclaim()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口（三阶段 Autograd + 异步 Guidance 版本）。

    只训练 Tex Flow Model，使用 PBR 渲染监督纹理。
    Shape 阶段使用冻结的模型生成几何。
    训练策略使用三阶段 Autograd + 异步 Guidance 流水线（显存 O(1)）。

    流程: Shape Forward (frozen) → Tex Rollout → PBR 渲染

    配置文件示例：
        python -m edit4shape.systems.trellis2.tex_autograd_async --config=configs/trellis2_tex.py
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
            project_name="trellis2-tex-distillation",  # ★ tex
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.render_base.resolution, vis_freq=vis_freq, accelerator=accelerator)

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = _build_system(cfg, accelerator, guidance_factory=partial(create_guidance, use_pp=True), mode="tex")
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
    # Step 8: 训练循环（Autograd + 异步 Guidance 流水线）
    # =====================================================
    tex_ops = Trellis2TexOps()  # 无状态策略对象，训练循环持有，drain 时传入
    # ★ 自适应梯度裁剪（TRELLIS.2 默认参数：max_norm=1.0, clip_percentile=95）
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")  # ★ train_tex
    profiler = AsyncPhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    if accum_steps < 2 and accelerator.is_main_process:
        logging.warning(
            "[AsyncPipeline] gradient_accumulation_steps=%d, "
            "异步流水线需要 accum≥2 才有并行收益。"
            "当前退化为同步模式，建议增大 accum_steps。",
            accum_steps,
        )

    def _flush_vjp(pending: PendingJob) -> None:
        """drain_vjp → log → vis → empty_cache（避免重复 3 次）。"""
        step, bs = pending.global_step, pending.batch_size
        log = pending.drain_vjp(tex_ops, system, profiler)
        tex_logger.log_step(log, bs, step, epoch)  # ★ tex_logger
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)  # ★ save_tex_train
        pending._reclaim()

    def _sync_grads_and_step(n_accumulated: int) -> None:
        """
        手动 all-reduce 梯度 → 除以实际累积数 → grad clip → optimizer.step → zero_grad。

        VJP 循环在 model.no_sync() 下执行，不触发 DDP 自动 all-reduce。
        因此需要在 optimizer.step() 前手动做一次跨 rank 梯度同步。
        单卡 / 非分布式环境下跳过 all-reduce，直接 step。

        Args:
            n_accumulated: 本次 step 实际累积的 micro-batch 数。
                           正常 accum 边界 = accum_steps；
                           epoch 尾部残留 = global_step % accum_steps。
        """
        model = system.tex.model  # ★ tex.model
        optimizer = system.tex.optimizer  # ★ tex.optimizer
        is_distributed = dist.is_initialized()
        has_nan = False
        for p in model.parameters():
            if p.grad is None:
                continue
            # 1. 跨 rank 梯度同步
            if is_distributed:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            # 2. 除以实际累积的 micro-batch 数，得到平均梯度
            if n_accumulated > 1:
                p.grad.div_(n_accumulated)
            # 3. NaN/Inf 检测（发现后仍继续循环，保证所有 rank all-reduce 行为一致）
            if not has_nan and not torch.isfinite(p.grad).all():
                has_nan = True
        # 4. NaN 拦截：跳过本次更新，防止 NaN 污染模型参数
        if has_nan:
            logging.warning("[NaN Guard] 检测到 NaN/Inf 梯度，跳过本次 optimizer step")
            optimizer.zero_grad()
            return
        # 5. 自适应梯度裁剪
        grad_clipper(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        prev: Optional[PendingJob] = None      # 双缓冲：上一个已 submit 的 MB

        for batch in train_loader:
            global_step += 1

            # ── Step 1: curr 前向 + submit（guidance GPU 尽早开始）──
            curr = PendingJob.create(batch, system, global_step, profiler)

            # ── Step 2: prev 的 P2（wait + P2-grad + clean）────────
            # ★ guidance GPU 已在处理 curr，与 P2-grad 并行。
            #   代价：P2-grad 时 GPU 额外持有 curr 残留数据（~0.5-1.5 GiB）。
            if prev is not None:
                prev.drain_guidance(tex_ops, system, profiler)

            # ── Step 3: prev 的 P1-grad (VJP) + log + vis ────────
            # ★ guidance GPU 继续处理 curr，与 VJP 并行。
            if prev is not None:
                _flush_vjp(prev)

            # ── prev ← curr ──────────────────────────────────────
            prev = curr
            # ★ 老 prev 延迟释放：SparseTensor._spatial_cache 中的
            #   GPU 索引张量在此刻才真正解引用，需要 gc + empty_cache
            #   确保在下一个 create 前回收。
            curr._reclaim()

            # ── Optimizer Step（在 accum 边界） ──────────────────
            if global_step % accum_steps == 0:
                if prev is not None:
                    prev.drain_guidance(tex_ops, system, profiler)
                    _flush_vjp(prev)
                    prev = None
                _sync_grads_and_step(accum_steps)

        # ── epoch 结束：消化残留的 prev ──────────────────────────
        if prev is not None:
            prev.drain_guidance(tex_ops, system, profiler)
            _flush_vjp(prev)
            prev = None
        # ★ 独立于 prev：只要不在 accum 边界，就有待 step 的残留梯度
        #   （即使最后几个 MB 全 OOM → prev=None，之前 flush 的梯度仍需 step）
        remainder = global_step % accum_steps
        if remainder != 0:
            _sync_grads_and_step(remainder)

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
