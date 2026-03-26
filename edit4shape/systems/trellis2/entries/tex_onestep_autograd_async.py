"""
Trellis2 Tex 训练系统 — Onestep + 异步 Guidance 流水线版本。

核心类 PendingJob 管理一个 micro-batch 的完整计算生命周期：

  PendingJob.create(batch, ...)     ← P0 + P1-P3.5 + P4a + submit
      ├── pre_rollout: shape_frozen_prepare (no_grad shape forward + detach)
      ├── pretrained_rollout: teacher tex rollout (no_grad) → clean tex z₀
      ├── add_noise: z₀ → zₜ
      ├── predict_cfg_velocity (student, with grad) → v_student
      │   → setup VelocityTracker proxy → v_proxy (leaf)
      │   → ẑ₀ = zₜ - t·v_proxy → denormalize → update tex_slat
      ├── (可选) teacher velocity → reg backward → reg_grad
      ├── P4a: decode_render_pbr (no_grad) → comp_rgb_detached + vis
      ├── submit to guidance GPU
      └── clean tex decode cache + spatial cache

  prev.drain(...)                              ← P4b-wait + P4c + P5 + log
      ├── P4b-wait:  等 guidance GPU → rgb_grad
      ├── P4c:       decode_render_dict(grad) → backward(rgb_grad) → v_proxy.grad
      ├── clean_for_relay: detach + release subs/meshes + offload vis + gc
      ├── P5: relay_and_backward (no_sync) → v_student.backward → θ_tex.grad
      └── final cleanup

与 shape_onestep_autograd_async 的差异：
  - Phase 0: shape_frozen_prepare（no_grad shape forward + detach）
  - rollout_tex 替代 rollout_shape，需要 shape_slat_norm 作为条件
  - decode_render_pbr 替代 decode_render_normal（需要 meshes + subs）
  - 清理策略：_clean_for_relay 额外释放 subs/meshes + shape_slat
  - 训练的是 tex.model，不是 shape.model

DDP 安全：
- P5 relay backward 在 model.no_sync() 下执行，不触发 DDP all-reduce
- 梯度同步由 _sync_grads_and_step() 在 optimizer.step 前手动 all-reduce(AVG)

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
from functools import partial
from edit4shape.guidance import create_guidance
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import VelocityTracker
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import evaluate as _evaluate
from edit4shape.systems.trellis2.stage_ops import Trellis2TexOps
from edit4shape.systems.utils.stage_ops import StageSkipError
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
# PendingJob — Tex Onestep 异步流水线 micro-batch
# =====================================================================

@dataclass
class PendingJob(_OnestepBase):
    """
    Tex Onestep 异步流水线 micro-batch。

    继承 OnestepPendingJob 基类，通过 _drain_onestep_stage building block
    实现 drain 逻辑。

    生命周期:
      .create(batch, ...)           ← P0(shape_frozen_prepare) + P1-P3.5 + P4a + submit
      .drain(ops, system, ...)      ← P4b-wait + P4c + P5 + log（委托基类）

    各阶段后的 GPU 状态:
      create() 后:
          GPU: v_student 计算图 (tex flow model 激活), OnestepContext,
               new_slat (with grad), shape_slat_norm, subs, meshes,
               cond+uncond embed, vis tensor
               ★ tex decode spatial cache 已释放
      drain() 后:
          GPU: model parameters only, all intermediate data released

    ★ 与 shape 版本的关键差异（编码在 Trellis2TexOps 中）：
      1. pre_rollout = shape_frozen_prepare（shape forward + detach）
      2. rollout = rollout_tex（需要 shape_slat_norm 作为条件）
      3. decode_render_dict = decode_and_render_pbr（需要 meshes + subs）
      4. clean_for_relay 额外释放 subs/meshes + shape_slat
    """

    ctx: Optional[OnestepContext] = None

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
        工厂方法：P0 + P1-P3.5 + P4a + submit → 创建 PendingJob。

        使用 Trellis2TexOps 驱动所有阶段特有逻辑。

        GPU 存活时间线：
          P0:    shape_frozen_prepare (no_grad shape forward + detach)
          P1:    pretrained rollout → clean tex z₀（no_grad, teacher context）
          P2:    加噪 → zₜ
          P3:    predict_cfg_velocity → v_student（有 autograd 图到 θ_tex）
          P3.5:  teacher velocity → reg backward → reg_grad（可选）
          P4a:   decode_pbr(no_grad) → submit → clean tex decode cache
        """
        ops = Trellis2TexOps()
        seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()
        device = system.accelerator.device

        with TrainModeGuard(ops.get_model(system)):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.tex.config.cond_resolution)

            # ── P0: shape_frozen_prepare ─────────────────────────
            profiler.tick("shape_frozen_prepare")
            ops.pre_rollout(state, system, global_step)

            # ── P1: pretrained rollout (teacher, no_grad) → clean tex z₀ ──
            profiler.tick("P1_pretrained_rollout")
            ops.pretrained_rollout(state, system, seed)

            # ── P2: 加噪 z₀ → zₜ ────────────────────────────────
            profiler.tick("P2_add_noise")
            z0_norm = ops.normalize_slat(ops.get_latent(state), system)
            z0_feats = z0_norm.feats.detach()  # (N, C)
            t_val = ops.sample_timestep(system)
            zt_feats = ops.add_noise(z0_feats, t_val)  # (N, C)

            # ── P3: predict CFG velocity (student, with grad) ────
            profiler.tick("P3_velocity")
            v_student = ops.predict_cfg_velocity(state, system, zt_feats, t_val)

            tracker = VelocityTracker()
            tracker.setup_proxy(v_student)

            # ẑ₀ = zₜ - t·v_proxy
            z0_hat_norm = zt_feats - t_val * tracker.v_proxy  # (N, C)
            z0_hat_denorm = ops.denormalize_slat(
                ops.get_latent(state).replace(z0_hat_norm), system,
            )

            # 更新 state 中的 tex_slat
            slat = ops.get_latent(state)
            new_slat = slat.replace(z0_hat_denorm.feats)
            state.tex.z0 = new_slat

            # ★ 清理不再需要的中间张量 + spatial cache，为 P3.5/P4a 腾出显存
            del z0_norm, z0_feats, z0_hat_norm, z0_hat_denorm, slat
            new_slat._spatial_cache.clear()
            torch.cuda.empty_cache()

            # ── P3.5: teacher velocity + reg backward（可选） ────
            reg_weight = ops.get_reg_weight(system)
            if reg_weight > 0:
                profiler.tick("P3.5_reg")
                v_teacher = ops.predict_cfg_velocity_teacher(
                    state, system, zt_feats, t_val,
                )
                reg_loss = reg_weight * F.mse_loss(tracker.v_proxy, v_teacher)
                reg_loss.backward()
                tracker.reg_grad = tracker.v_proxy.grad.detach().clone()
                tracker.reg_loss_val = reg_loss.item()
                tracker.v_proxy.grad = None  # ★ 清零
                del v_teacher, reg_loss
                torch.cuda.empty_cache()
            # ★ P3.5 完成后 zt_feats 不再需要
            del zt_feats

            # 创建实例
            batch_size = len(batch['image_pils'])
            inst = cls(
                state=state,
                global_step=global_step,
                batch_size=batch_size,
                ctx=OnestepContext(
                    vel_tracker=tracker,
                    t_val=t_val,
                    submitted=False,
                ),
            )

            # ── P4a: no_grad decode → submit ─────────────────────
            try:
                profiler.tick("P4a_decode_no_grad")
                with torch.no_grad():
                    render_out = ops.decode_render_dict(state, system)
                comp_rgb_detached = render_out["color"].detach()
                # 挂载 vis
                state.views_generated.pbr_tensor = comp_rgb_detached
                del render_out

                profiler.tick("P4a_submit")
                system.guidance.submit_async(
                    comp_rgb_detached,
                    state.views_conditioned.image_pils,
                    guidance_weight=ops.get_guidance_weight(system),
                    guidance_cfg=ops.get_guidance_cfg(system),
                    rank=system.accelerator.process_index,
                )
                inst.ctx.submitted = True
                del comp_rgb_detached
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    f"[Step {global_step}] P4a OOM → reg-only relay"
                )
                profiler.reset()
            except StageSkipError as e:
                logging.warning(
                    f"[Step {global_step}] P4a skipped: {e} → reg-only relay"
                )
                profiler.reset()
            finally:
                # 释放 tex decode cache + spatial cache
                state.release_tex_spatial_cache()
                torch.cuda.empty_cache()

        return inst

    # ════════════════════════════════════════════════════════
    # 公开 API — drain（委托基类 building block）
    # ════════════════════════════════════════════════════════

    def drain(
        self,
        ops: Trellis2TexOps,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        P4b-wait + P4c + P5 + log → θ_tex.grad 累积 → 返回日志。

        委托 OnestepPendingJob._drain_onestep_stage building block。
        """
        log = self._drain_onestep_stage(
            ops, self.ctx, system, profiler,
            clean_decode=lambda: self.state.release_tex_spatial_cache(),
            clean_for_relay=self._clean_for_relay,
        )
        self.ctx = None
        self.state.release_tex_spatial_cache()
        self.state.release_shape_decode_cache()  # shape_slat_norm 等残留
        self._reclaim()
        return log

    def _clean_for_relay(self) -> None:
        """
        P4c 结束后清理：释放 relay 不需要的 GPU 数据。

        ★ 与 shape 版本的差异：
          - 释放 subs/meshes（relay 不需要）
          - 释放 shape_slat（relay 不需要）
          - 保留 tex_slat（detach 后不影响 relay）
        """
        s = self.state
        s.release_shape_spatial_cache()  # 兜底
        s.prepare_for_tex_vjp()          # 释放 subs/meshes + tex_spatial_cache
        s.detach_features()              # 释放 P4c 的 decode→slat→v_proxy 计算图
        s.features.shape_slat = None     # relay 不需要 shape_slat
        s.release_uncond_embeddings()
        s.offload_vis_to_cpu()
        self._reclaim()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    只训练 Tex Flow Model，使用 Onestep 策略 + 异步 Guidance 流水线。
    Shape 阶段使用冻结的模型生成几何。

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.tex_onestep_autograd_async \\
            --config=configs/trellis2_tex_onestep.py
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
            project_name="trellis2-tex-onestep-async",
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
        mode="tex",
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
    # Step 8: 训练循环（Onestep + 异步 Guidance 流水线）
    # =====================================================
    tex_ops = Trellis2TexOps()
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
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

    def _flush(pending: PendingJob) -> None:
        """drain → log → vis → reclaim。"""
        step, bs = pending.global_step, pending.batch_size
        log = pending.drain(tex_ops, system, profiler)
        tex_logger.log_step(log, bs, step, epoch)
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)
        _reclaim()

    def _sync_grads_and_step(n_accumulated: int) -> None:
        """手动 all-reduce → NaN 拦截 → grad clip → step → zero_grad。"""
        model = system.tex.model
        optimizer = system.tex.optimizer
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
        grad_clipper(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        prev: Optional[PendingJob] = None

        for batch in train_loader:
            global_step += 1

            curr = PendingJob.create(batch, system, global_step, profiler)

            if prev is not None:
                _flush(prev)

            prev = curr
            _reclaim()

            if global_step % accum_steps == 0:
                if prev is not None:
                    _flush(prev)
                    prev = None
                _sync_grads_and_step(accum_steps)

        # ── epoch 结束：消化残留 ──
        if prev is not None:
            _flush(prev)
            prev = None
        remainder = global_step % accum_steps
        if remainder != 0:
            _sync_grads_and_step(remainder)

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
