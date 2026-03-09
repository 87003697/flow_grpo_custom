"""
Trellis2 Shape 训练系统 — Onestep + 异步 Guidance 流水线版本。

核心类 PendingJob 管理一个 micro-batch 的完整计算生命周期：

  PendingJob.create(batch, ...)     ← P0-P3.5 + P4a + submit
      ├── pre_rollout: dense sampling → state.coords
      ├── pretrained_rollout: teacher (no_grad) → clean z₀
      ├── add_noise: z₀ → zₜ
      ├── predict_cfg_velocity (student, with grad) → v_student
      │   → setup VelocityTracker proxy → v_proxy (leaf)
      │   → ẑ₀ = zₜ - t·v_proxy → denormalize → update slat
      ├── (可选) teacher velocity → reg backward → reg_grad
      ├── P4a: decode_render(no_grad) → comp_rgb_detached + vis
      ├── submit to guidance GPU
      └── clean decode cache + spatial cache

  prev.drain(...)                              ← P4b-wait + P4c + P5 + log
      ├── P4b-wait:  等 guidance GPU → rgb_grad
      ├── P4c:       decode_render_dict(grad) → backward(rgb_grad) → v_proxy.grad
      ├── clean_for_relay: detach + offload vis + release uncond + gc
      ├── P5: relay_and_backward (no_sync) → v_student.backward → θ.grad
      └── final cleanup

每次迭代执行顺序（稳态）：
  1. curr = .create(batch, ...)       ← P0-P4a + submit（guidance GPU 尽早开始）
  2. prev.drain(...)                  ← P4b-wait + P4c + P5 + log + vis
  3. prev = curr
  4. (accum 边界) → drain 残留 + all-reduce + optimizer step

★ 异步优势：
  create(curr) 先于 drain(prev) 执行，guidance GPU 在 P4c + P5
  全程并行处理 curr，异步窗口 ≈ P4c + P5（relay backward 含 flow model 反传）。
  代价：drain 时 GPU 额外持有 curr 的 v_student 计算图（~1-2 GiB），
  已有 OOM 保护兜底。

★ 与 shape_autograd_async（VJP 版本）的差异：
  - 不使用 RolloutTracker / proxy chain / VJP 循环
  - 使用 VelocityTracker 做 velocity 空间的梯度追踪
  - P5 relay 是单次 v_student.backward（不是多步 VJP loop）
  - 显存模式：v_student 计算图常驻（VJP 版只有 detached proxy）

DDP 安全：
- P5 relay backward 在 model.no_sync() 下执行，不触发 DDP all-reduce
- 梯度同步由 _sync_grads_and_step() 在 optimizer.step 前手动 all-reduce(AVG)
- 各 rank OOM 导致梯度缺失不会死锁

特性：
- accum≥2 时收益最大：curr 的 guidance 与 prev 的 P4c + P5 全程并行
- accum=1 时退化为同步版（无并行窗口，但正确性不变）
- 评估路径仍使用单阶段 forward（trellis2_shape_forward）
- OOM 安全：P4a/P4c OOM 均可降级到 reg-only relay

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
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
from edit4shape.systems.trellis2.stage_ops import ShapeOps
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
# 统一接口：build_system / evaluate
# =====================================================================

def build_system(cfg, accelerator, guidance_factory):
    """构建 Shape-only 训练系统。"""
    return _build_system(cfg, accelerator, guidance_factory, mode="shape")


def evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir):
    """Shape-only 评估。"""
    return _evaluate(system, epoch, global_step, eval_loader, visuals_eval_dir, with_tex=False)


# =====================================================================
# PendingJob — Shape Onestep 异步流水线 micro-batch
# =====================================================================

@dataclass
class PendingJob(_OnestepBase):
    """
    Shape Onestep 异步流水线 micro-batch。

    继承 OnestepPendingJob 基类，通过 _drain_onestep_stage building block
    实现 drain 逻辑，消除与 tex/shape_tex 版本的重复代码。

    生命周期:
      .create(batch, ...)           ← P0-P3.5 + P4a + submit
      .drain(ops, system, ...)      ← P4b-wait + P4c + P5 + log（委托基类）

    各阶段后的 GPU 状态:
      create() 后:
          GPU: v_student 计算图 (flow model 激活), OnestepContext,
               new_slat (with grad through v_proxy), cond+uncond embed, vis tensor
               ★ decode spatial cache 已释放
      drain() 后:
          GPU: model parameters only, all intermediate data released

    ★ comp_rgb 不存储：create 中仅用于 submit，drain 中重算。
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
        工厂方法：P0-P3.5 + P4a + submit → 创建 PendingJob。

        使用 ShapeOps 驱动所有阶段特有逻辑。

        GPU 存活时间线：
          P0-P1: pretrained rollout → clean z₀（no_grad, teacher context）
          P2:    加噪 → zₜ
          P3:    predict_cfg_velocity → v_student（有 autograd 图到 θ）
          P3.5:  teacher velocity → reg backward → reg_grad（可选）
          P4a:   decode(no_grad) → submit → clean decode cache

        OOM 安全降级：
          P4a OOM → submitted=False → drain 跳过 P4c，relay 仅用 reg_grad。
        """
        ops = ShapeOps()
        seed = int(system.cfg.seed) + global_step + ops.get_seed_offset()
        device = system.accelerator.device

        with TrainModeGuard(ops.get_model(system)):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.shape.config.cond_resolution)

            # ── P0: pre_rollout（dense_sampling） ────────────────
            profiler.tick("P0_pre_rollout")
            ops.pre_rollout(state, system, global_step)

            # ── P1: pretrained rollout (teacher, no_grad) → clean z₀ ──
            profiler.tick("P1_pretrained_rollout")
            ops.pretrained_rollout(state, system, seed)

            # ── P2: 加噪 z₀ → zₜ ────────────────────────────────
            profiler.tick("P2_add_noise")
            z0_norm = ops.normalize_slat(ops.get_slat(state), system)
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
                ops.get_slat(state).replace(z0_hat_norm), system,
            )

            # 更新 state 中的 slat
            slat = ops.get_slat(state)
            new_slat = slat.replace(z0_hat_denorm.feats)
            state.features.shape_slat = new_slat

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
                tracker.v_proxy.grad = None  # ★ 清零，给 P4c 的 guidance 梯度腾位
                del v_teacher, reg_loss
                torch.cuda.empty_cache()
            # ★ P3.5 完成后 zt_feats 不再需要
            del zt_feats

            # 创建实例
            batch_size = len(batch['image_pils'])
            submitted = False
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
                state.views_generated.shape_tensor = comp_rgb_detached
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
                # 释放 decode cache + spatial cache
                state.release_shape_decode_cache()
                state.release_shape_spatial_cache()
                torch.cuda.empty_cache()

        return inst

    # ════════════════════════════════════════════════════════
    # 公开 API — drain（委托基类 building block）
    # ════════════════════════════════════════════════════════

    def drain(
        self,
        ops: ShapeOps,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        P4b-wait + P4c + P5 + log → θ.grad 累积 → 返回日志。

        委托 OnestepPendingJob._drain_onestep_stage building block。

        OOM 降级链路：
          P4b-wait 失败 → rgb_grad=None → 跳过 P4c → relay 仅用 reg_grad
          P4c OOM → v_proxy.grad=None → relay 仅用 reg_grad
          reg_grad=None + v_proxy.grad=None → 跳过 relay（零梯度贡献）
        """
        log = self._drain_onestep_stage(
            ops, self.ctx, system, profiler,
            clean_decode=lambda: (
                self.state.release_shape_decode_cache(),
                self.state.release_shape_spatial_cache(),
            ),
            clean_for_relay=self._clean_for_relay,
        )
        self.ctx = None
        self.state.release_shape_spatial_cache()  # relay 可能产生的 spatial cache
        self._reclaim()
        return log

    def _clean_for_relay(self) -> None:
        """
        P4c 结束后清理：释放 relay 不需要的 GPU 数据。

        ★ detach_features 释放 P4c 的 decode→slat→v_proxy 计算图
          （v_student 计算图不受影响，relay 仍可 backward 到 θ）
        """
        self.state.detach_features()
        self.state.release_uncond_embeddings()
        self.state.offload_vis_to_cpu()
        self._reclaim()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    训练 Shape Flow Model，使用 Onestep 策略 + 异步 Guidance 流水线。

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.shape_onestep_autograd_async \\
            --config=configs/trellis2_shape_onestep.py
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

    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape-onestep-async",
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
            system, epoch=start_epoch, global_step=global_step,
            eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（Onestep + 异步 Guidance 流水线）
    # =====================================================
    shape_ops = ShapeOps()  # 无状态策略对象，训练循环持有，drain 时传入
    grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
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
        log = pending.drain(shape_ops, system, profiler)
        shape_logger.log_step(log, bs, step, epoch)
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)
        _reclaim()

    def _sync_grads_and_step(n_accumulated: int) -> None:
        """
        手动 all-reduce 梯度 → 除以实际累积数 → NaN 拦截 → grad clip → step → zero_grad。

        P5 relay backward 在 model.no_sync() 下执行，不触发 DDP 自动 all-reduce。
        因此需要在 optimizer.step() 前手动做一次跨 rank 梯度同步。

        Args:
            n_accumulated: 本次 step 实际累积的 micro-batch 数。
        """
        model = system.shape.model
        optimizer = system.shape.optimizer
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

        prev: Optional[PendingJob] = None  # 双缓冲：上一个已 submit 的 MB

        for batch in train_loader:
            global_step += 1

            # ── Step 1: curr 前向 + submit（guidance GPU 尽早开始） ──
            curr = PendingJob.create(batch, system, global_step, profiler)

            # ── Step 2: prev drain (P4c + P5) + log + vis ───────
            # ★ guidance GPU 已在处理 curr，与 P4c + P5 并行。
            if prev is not None:
                _flush(prev)

            # ── prev ← curr ─────────────────────────────────────
            prev = curr
            # ★ 老 prev 延迟释放：SparseTensor._spatial_cache 中的
            #   GPU 索引张量在此刻才真正解引用，需要 gc + empty_cache
            #   确保在下一个 create 前回收。
            _reclaim()

            # ── Optimizer Step（在 accum 边界） ──────────────────
            if global_step % accum_steps == 0:
                if prev is not None:
                    _flush(prev)
                    prev = None
                _sync_grads_and_step(accum_steps)

        # ── epoch 结束：消化残留的 prev ─────────────────────────
        if prev is not None:
            _flush(prev)
            prev = None
        # ★ 独立于 prev：只要不在 accum 边界，就有待 step 的残留梯度
        remainder = global_step % accum_steps
        if remainder != 0:
            _sync_grads_and_step(remainder)

        # ---- 周期性评估（epoch 级别）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, epoch=epoch, global_step=global_step,
                eval_loader=eval_loader, visuals_eval_dir=visuals_eval_dir,
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
