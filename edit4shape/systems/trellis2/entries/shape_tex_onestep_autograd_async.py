"""
Trellis2 Shape+Tex 双阶段训练系统 — Onestep + 异步 Guidance 流水线版本（双阶段交替异步）。

核心类 PendingJob 统一管理一个 micro-batch 中 Shape + Tex 两阶段的完整计算生命周期。
两个 VelocityTracker（shape_tracker, tex_tracker）分别追踪 shape 和 tex 的 velocity 梯度，
共享一个 Trellis2State（生命周期由统一的清理方法管理）。

_flush_shape(prev)                         ← Shape drain_shape + vis + log
    ├── P4b-wait → P4c-grad → clean → P5 relay → θ_shape.grad
    └── vis 保存（在 Tex guidance 覆盖 views_edited 之前）

curr = .create_shape(batch, ...)           ← Shape P0-P3.5 + P4a + submit_S
    ├── ShapeOps: pre_rollout → pretrained_rollout → add_noise
    │   → predict_cfg_velocity → VelocityTracker → P4a decode + submit
    └── submit 入 guidance FIFO 队列

_flush_tex(prev)                           ← Tex drain_tex + vis + log
    ├── P4b-wait → P4c-grad → clean → P5 relay → θ_tex.grad
    └── vis 保存

curr.create_tex(...)                       ← Tex P1-P3.5 + P4a + submit_T
    ├── TexOpsFromShape: pretrained_rollout → add_noise
    │   → predict_cfg_velocity → VelocityTracker → P4a decode + submit
    └── submit 入 guidance FIFO 队列

prev = curr

每次迭代执行顺序（稳态，S/T 交错流水线）：
  1. _flush_shape(prev)                  ← Shape drain + vis + log + maybe step_S
  2. curr = .create_shape(batch, ...)    ← Shape P0-P4a + submit_S
  3. _flush_tex(prev)                    ← Tex drain + vis + log + maybe step_T
  4. curr.create_tex(...)                ← Tex P1-P4a + submit_T
  5. prev = curr

★ S/T 交错流水线优势（accum_steps=1 也有完整异步并行）：
  - S[N] 在 step 2 submit，在下一轮 step 1 wait
    并行窗口 = flush_T(prev) + create_T(curr) ≈ tex drain + tex create
  - T[N] 在 step 4 submit，在下一轮 step 3 wait
    并行窗口 = flush_S(next) + create_S(next) ≈ shape drain + shape create

★ 与 shape_tex_autograd_async（VJP 版本）的差异：
  - 不使用 RolloutTracker / proxy chain / VJP 循环
  - 使用 VelocityTracker 做 velocity 空间的梯度追踪
  - P5 relay 是单次 v_student.backward（不是多步 VJP loop）
  - drain 更简洁：无需 StageContext / ctx_* 系列函数

DDP 安全：
- P5 relay backward 在 model.no_sync() 下执行，不触发 DDP all-reduce
- 梯度同步由 _sync_grads_and_step() 在 optimizer.step 前手动 all-reduce(AVG)
- Shape 和 Tex 各自独立 sync + step

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import os, sys
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

# =====================================================================
# TRELLIS.2 参考实现路径设置（必须在 trellis2 模块导入之前）
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

# =====================================================================
# 第三方库导入
# =====================================================================
from absl import app
from ml_collections import config_flags

import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import VelocityTracker
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import evaluate as _evaluate
from edit4shape.systems.trellis2.stage_ops import ShapeOps, TexOpsFromShape
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler
from edit4shape.systems.utils.stage_ops import StageSkipError

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper

# =====================================================================
# 基类 + OnestepContext 导入
# =====================================================================
from edit4shape.systems.utils.pending_base import (
    OnestepPendingJob as _OnestepBase,
    OnestepContext,
    _reclaim,
)


# =====================================================================
# 辅助函数
# =====================================================================

def _onestep_create_stage(
    ops,
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: AsyncPhaseProfiler,
    prefix: str = "",
) -> tuple:
    """
    通用 Onestep 阶段创建（P1-P3.5 + P4a + submit）。

    抽取 Shape / Tex 共同的 Onestep 流程。调用方负责 P0（pre_rollout）。

    Args:
        ops:    Trellis2StageOps 实例（ShapeOps / TexOpsFromShape）
        state:  Trellis2State（已完成 pre_rollout）
        system: Trellis2System
        global_step: 全局步数
        profiler: AsyncPhaseProfiler
        prefix: profiler tick 前缀（"S_" / "T_"）

    Returns:
        (tracker, t_val, submitted): VelocityTracker, 时间步, 是否 submit 成功
    """
    seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()
    stage_name = ops.get_stage_name()

    # ── P1: pretrained rollout (teacher, no_grad) → clean z₀ ──
    profiler.tick(f"{prefix}P1_pretrained_rollout")
    ops.pretrained_rollout(state, system, seed)

    # ── P2: 加噪 z₀ → zₜ ──
    profiler.tick(f"{prefix}P2_add_noise")
    z0_norm = ops.normalize_slat(ops.get_slat(state), system)
    z0_feats = z0_norm.feats.detach()
    t_val = ops.sample_timestep(system)
    zt_feats = ops.add_noise(z0_feats, t_val)

    # ── P3: predict CFG velocity (student, with grad) ──
    profiler.tick(f"{prefix}P3_velocity")
    v_student = ops.predict_cfg_velocity(state, system, zt_feats, t_val)

    tracker = VelocityTracker()
    tracker.setup_proxy(v_student)

    # ẑ₀ = zₜ - t·v_proxy
    z0_hat_norm = zt_feats - t_val * tracker.v_proxy
    z0_hat_denorm = ops.denormalize_slat(
        ops.get_slat(state).replace(z0_hat_norm), system,
    )

    # 更新 state 中的 slat
    slat = ops.get_slat(state)
    new_slat = slat.replace(z0_hat_denorm.feats)
    if stage_name == "shape":
        state.features.shape_slat = new_slat
    else:
        state.features.tex_slat = new_slat

    new_slat._spatial_cache.clear()
    torch.cuda.empty_cache()

    # ── P3.5: teacher velocity + reg backward（可选） ──
    reg_weight = ops.get_reg_weight(system)
    if reg_weight > 0:
        profiler.tick(f"{prefix}P3.5_reg")
        v_teacher = ops.predict_cfg_velocity_teacher(
            state, system, zt_feats, t_val,
        )
        reg_loss = reg_weight * F.mse_loss(tracker.v_proxy, v_teacher)
        reg_loss.backward()
        tracker.reg_grad = tracker.v_proxy.grad.detach().clone()
        tracker.reg_loss_val = reg_loss.item()
        tracker.v_proxy.grad = None
        del v_teacher, reg_loss
        torch.cuda.empty_cache()

    # ── P4a: no_grad decode → submit ──
    submitted = False
    try:
        profiler.tick(f"{prefix}P4a_decode_no_grad")
        with torch.no_grad():
            render_out = ops.decode_render_dict(state, system)
        comp_rgb_detached = render_out["color"].detach()
        # 挂载 vis
        if stage_name == "shape":
            state.views_generated.shape_tensor = comp_rgb_detached
            # ★ 保存 subs/meshes（Shape decode 产物，Tex P4a/P4c decode 需要）
            # 对齐 VJP 版本的 ops.decode_render() 行为
            state.features.subs = render_out.get("subs")
            state.features.meshes = render_out.get("meshes")
        else:
            state.views_generated.pbr_tensor = comp_rgb_detached
        del render_out

        profiler.tick(f"{prefix}P4a_submit")
        system.guidance.submit_async(
            comp_rgb_detached,
            state.views_conditioned.image_pils,
            guidance_weight=ops.get_guidance_weight(system),
            guidance_cfg=ops.get_guidance_cfg(system),
            rank=system.accelerator.process_index,
        )
        submitted = True
        del comp_rgb_detached
    except torch.cuda.OutOfMemoryError:
        logging.warning(
            f"[Step {global_step}] {prefix}P4a OOM → reg-only relay"
        )
        profiler.reset()
    except StageSkipError as e:
        logging.warning(
            f"[Step {global_step}] {prefix}P4a skipped: {e} → reg-only relay"
        )
        profiler.reset()
    finally:
        # 释放 decode cache + spatial cache
        if stage_name == "shape":
            state.release_shape_spatial_cache()
        else:
            state.release_tex_spatial_cache()
        torch.cuda.empty_cache()

    return tracker, t_val, submitted


# =====================================================================
# PendingJob — Shape+Tex 双阶段 Onestep 异步流水线 micro-batch
# =====================================================================

@dataclass
class PendingJob(_OnestepBase):
    """
    Shape+Tex 双阶段 Onestep 异步流水线 micro-batch（S/T 交错流水线版本）。

    继承 OnestepPendingJob 基类，通过 _drain_onestep_stage building block
    实现 drain 逻辑，消除与单阶段版本的重复代码。

    生命周期（S/T 交错流水线）：
      _flush_shape(prev)                    ← prev Shape drain + vis + log
      .create_shape(batch, ...)             ← Shape P0-P4a + submit_S
      _flush_tex(prev)                      ← prev Tex drain + vis + log
      .create_tex(...)                      ← Tex P1-P4a + submit_T

    ★ Guidance FIFO 约束：
      create_shape 中 submit_shape 先于 create_tex 中 submit_tex，
      _flush_shape 中 shape_wait 先于 _flush_tex 中 tex_wait，
      顺序严格一致。
    """

    # 双阶段 OnestepContext
    shape_ctx: Optional[OnestepContext] = None
    tex_ctx: Optional[OnestepContext] = None

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
        工厂方法（Shape 半周期）：Shape P0-P3.5 + P4a + submit_S。

        submit_shape 后，shape guidance 立即在 guidance GPU 开始，
        与后续操作（prev Shape drain）全程并行。
        """
        ops = ShapeOps()

        state = Trellis2State()
        state.attach_batch(
            batch, pipeline=system.pipeline,
            resolution=system.tex.config.cond_resolution,
        )

        batch_size = len(batch['image_pils'])

        with TrainModeGuard(ops.get_model(system)):
            # ── P0: dense_sampling ──
            profiler.tick("S_dense_sampling")
            ops.pre_rollout(state, system, global_step)

            # ── P1-P3.5 + P4a + submit ──
            tracker, t_val, submitted = _onestep_create_stage(
                ops, state, system, global_step, profiler, prefix="S_",
            )

        inst = cls(
            state=state,
            global_step=global_step,
            batch_size=batch_size,
            shape_ctx=OnestepContext(
                vel_tracker=tracker,
                t_val=t_val,
                submitted=submitted,
            ),
        )
        return inst

    def create_tex(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Tex 半周期：Tex P1-P3.5 + P4a + submit_T（原地修改 self）。

        ★ 调用时机：在 create_shape 和 prev 的 Shape flush 之后调用。
        ★ TexOpsFromShape 的 pre_rollout = no-op（shape 产物由 create_shape 提供）。
        """
        ops = TexOpsFromShape()

        with TrainModeGuard(ops.get_model(system)):
            # TexOpsFromShape 的 pre_rollout 是 no-op（shape 产物由 create_shape 提供）

            # ── P1-P3.5 + P4a + submit ──
            tracker, t_val, submitted = _onestep_create_stage(
                ops, self.state, system, self.global_step, profiler, prefix="T_",
            )

        self.tex_ctx = OnestepContext(
            vel_tracker=tracker,
            t_val=t_val,
            submitted=submitted,
        )

    # ════════════════════════════════════════════════════════
    # 公开 API — drain（Shape / Tex 分别 drain，委托基类 building block）
    # ════════════════════════════════════════════════════════

    def drain_shape(
        self,
        ops: ShapeOps,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Shape drain: P4b-wait + P4c + P5 → θ_shape.grad。

        Postcondition：
          shape spatial cache 已释放。
          subs/meshes 保留（供后续 Tex P4c 使用 → offload to CPU → drain_tex 时 reload）。
        """
        log = self._drain_onestep_stage(
            ops, self.shape_ctx, system, profiler,
            prefix="S_",
            clean_decode=lambda: self.state.release_shape_spatial_cache(),
            clean_for_relay=self._clean_shape_for_relay,
            log_prefix="shape/",
            collect_profiler=False,  # 由 drain_tex 最后收集
        )
        self.shape_ctx = None
        self.state.release_shape_spatial_cache()  # relay 可能产生的 spatial cache
        self._reclaim()
        return log

    def drain_tex(
        self,
        ops: TexOpsFromShape,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Tex drain: P4b-wait + P4c + P5 → θ_tex.grad。

        Postcondition：
          subs/meshes 已释放。features 已 detach。vis 已 offload。
        ★ profiler.collect() 在此调用（最后一个 drain），收集整个步的计时数据。

        ★ drain_shape 已将 subs/meshes offload 到 CPU，此处先 reload 到 GPU。
        """
        # ★ Reload subs/meshes from CPU（_clean_shape_for_relay 中 offload 到了 CPU）
        self.state.reload_decode_cache_to_gpu(system.accelerator.device)
        log = self._drain_onestep_stage(
            ops, self.tex_ctx, system, profiler,
            prefix="T_",
            clean_decode=lambda: self.state.release_tex_spatial_cache(),
            clean_for_relay=self._clean_tex_for_relay,
            log_prefix="tex/",
            collect_profiler=True,  # 最后一个 drain，收集整个步计时
        )
        self.tex_ctx = None
        self.state.release_tex_spatial_cache()
        self.state.release_shape_decode_cache()  # shape_slat_norm 等残留
        self._reclaim()
        return log

    # ════════════════════════════════════════════════════════
    # 清理回调
    # ════════════════════════════════════════════════════════

    def _clean_shape_for_relay(self) -> None:
        """
        Shape P4c 结束后清理：释放 relay 不需要的 GPU 数据。

        ★ 单独 detach shape_slat（释放 P4c decode→slat→v_proxy 计算图）
          不 detach tex_slat（Tex P4c 还需要 proxy chain）
        ★ 不释放 subs/meshes — Tex P4c 还需要（先 offload → drain_tex 时 reload）
        """
        s = self.state
        s.release_shape_spatial_cache()
        if s.features.shape_slat is not None:
            s.features.shape_slat = s.features.shape_slat.detach()
        s.release_uncond_embeddings()
        s.offload_vis_to_cpu()
        # ★ subs/meshes → CPU（降低 Shape relay + create_S 的显存水位，drain_tex 时 reload）
        s.offload_decode_cache_to_cpu()
        self._reclaim()

    def _clean_tex_for_relay(self) -> None:
        """
        Tex P4c 结束后清理：释放 relay 不需要的 GPU 数据。
        """
        s = self.state
        s.release_tex_spatial_cache()
        s.prepare_for_tex_vjp()     # 释放 subs/meshes + tex_spatial_cache（兜底）
        s.detach_features()          # proxy chain → detached
        s.features.shape_slat = None
        s.release_uncond_embeddings()
        s.offload_vis_to_cpu()
        self._reclaim()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    同时训练 Shape 和 Tex 两个 Flow Model，使用 Onestep 策略 + 双阶段异步 Guidance 流水线。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.shape_tex_onestep_autograd_async \\
            --config=configs/trellis2_shape_tex_onestep.py
    """
    del argv
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    Trellis2System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
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

    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape+tex-onestep-async",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )

    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(
        visuals_train_dir, target_h=cfg.render_base.resolution,
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
        guidance_factory=partial(create_guidance, use_pp=True),
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
    # Step 8: 训练循环（双阶段 Onestep + 异步 Guidance 流水线）
    # =====================================================
    shape_ops = ShapeOps()
    tex_ops = TexOpsFromShape()
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    profiler = AsyncPhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)

    def _flush_shape(pending: PendingJob) -> None:
        """
        Shape 完整 drain + vis + log。

        ★ Shape vis 必须在 _flush_tex 之前保存
          （drain_tex 会覆盖 views_edited）。
        """
        step, bs = pending.global_step, pending.batch_size
        shape_log = pending.drain_shape(shape_ops, system, profiler)
        shape_logger.log_step(shape_log, bs, step, epoch)

        # ★ Shape vis（在 Tex guidance 覆盖 views_edited 之前保存）
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)

    def _flush_tex(pending: PendingJob) -> None:
        """Tex 完整 drain + vis + log + reclaim。"""
        step, bs = pending.global_step, pending.batch_size
        tex_log = pending.drain_tex(tex_ops, system, profiler)

        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)

        tex_logger.log_step(tex_log, bs, step, epoch)
        _reclaim()

    def _sync_grads_and_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_accumulated: int,
        grad_clipper: AdaptiveGradClipper = None,
    ) -> None:
        """手动 all-reduce → NaN 拦截 → grad clip → step → zero_grad。"""
        is_distributed = dist.is_initialized()
        has_nan = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if is_distributed:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            if n_accumulated > 1:
                p.grad.div_(n_accumulated)
            if not has_nan and not torch.isfinite(p.grad).all():
                has_nan = True
        if has_nan:
            logging.warning("[NaN Guard] 检测到 NaN/Inf 梯度，跳过本次 optimizer step")
            optimizer.zero_grad()
            return
        if grad_clipper is not None:
            grad_clipper(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        prev: Optional[PendingJob] = None
        shape_accum = 0
        tex_accum = 0

        for batch in train_loader:
            global_step += 1

            # ── 1. Shape drain(prev) + maybe step ─────────────
            #    S[prev] guidance 已在上一轮 create_S 后全程并行完成
            if prev is not None:
                _flush_shape(prev)
                shape_accum += 1
                if shape_accum >= accum_steps:
                    _sync_grads_and_step(
                        system.shape.model, system.shape.optimizer,
                        shape_accum, shape_grad_clipper,
                    )
                    shape_accum = 0

            # ── 2. Create curr Shape (submit_S → guid GPU 开始) ──
            curr = PendingJob.create_shape(batch, system, global_step, profiler)

            # ── 3. Tex drain(prev) + maybe step ───────────────
            #    T[prev] guidance 已在上一轮 create_T 后全程并行完成
            if prev is not None:
                _flush_tex(prev)
                tex_accum += 1
                if tex_accum >= accum_steps:
                    _sync_grads_and_step(
                        system.tex.model, system.tex.optimizer,
                        tex_accum, tex_grad_clipper,
                    )
                    tex_accum = 0

            # ── 4. Create curr Tex (submit_T → guid GPU 开始) ──
            curr.create_tex(system, profiler)

            # ── prev ← curr ──────────────────────────────────
            prev = curr
            _reclaim()

        # ── epoch 结束：消化残留的 prev ─────────────────────
        if prev is not None:
            _flush_shape(prev)
            shape_accum += 1
            _flush_tex(prev)
            tex_accum += 1
            prev = None
        if shape_accum > 0:
            _sync_grads_and_step(
                system.shape.model, system.shape.optimizer,
                shape_accum, shape_grad_clipper,
            )
        if tex_accum > 0:
            _sync_grads_and_step(
                system.tex.model, system.tex.optimizer,
                tex_accum, tex_grad_clipper,
            )

        # ---- 周期性评估 ----
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
