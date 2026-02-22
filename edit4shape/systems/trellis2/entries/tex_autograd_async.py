"""
Trellis2 Tex 训练系统 — Autograd + 异步 Guidance 流水线版本。

核心类 PendingMicroBatch（继承 PendingMicroBatchBase）管理一个 micro-batch 的完整计算生命周期：

  PendingMicroBatch.create(batch, ...)     ← P0 + P1-ng + P2-ng + submit
      ├── _p0_shape_frozen_prepare: shape forward(no_grad) + detach all
      ├── _p1_rollout:        rollout_tex → tracker (proxy chain)
      ├── _p2_no_grad:        _decode_and_render(no_grad) → comp_rgb + vis
      ├── _p2_submit:         submit to guidance GPU
      └── _clean_p2_decode:   释放 tex decode cache

  prev.drain_guidance(...)                  ← 基类：P2-wait + P2-grad + clean
      ├── _p2_wait:  等 guidance GPU → rgb_grad
      ├── _p2_grad:  _decode_and_render(grad) → backward
      │   └── finally: _clean_p2_decode
      └── _clean_for_vjp: detach + release subs/meshes + offload + gc

  prev.drain_vjp(...)                       ← 基类：P1-grad VJP → θ.grad
      ├── _p1_grad:  VJP loop (no_sync) → θ.grad 本地累积
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
    Trellis2System, build_system as _build_system,
)
from edit4shape.systems.trellis2.forward import (
    decode_and_render_pbr,  # ★ PBR 渲染（而非 Normal）
    evaluate as _evaluate,
)

# =====================================================================
# 从 trellis2_tex_autograd.py 导入可复用的 Phase 函数
# =====================================================================
from edit4shape.systems.trellis2.phases import (
    shape_frozen_prepare_no_grad,   # ★ Phase 0: shape forward(no_grad) + detach
    tex_phase1_rollout,             # ★ Phase 1: rollout_tex + tracker
)

# =====================================================================
# Rollout & VJP
# =====================================================================
from edit4shape.generators.trellis2.rollout import RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity, _vjp_loader

# =====================================================================
# 基类导入
# =====================================================================
from edit4shape.systems.utils.pending_base import PendingMicroBatchBase

# =====================================================================
# absl 配置
# =====================================================================
# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。


# =====================================================================
# PendingMicroBatch — Tex 异步流水线 micro-batch
# =====================================================================
#
# 继承 PendingMicroBatchBase，获得公共的：
#   drain_guidance / drain_vjp / _p2_wait / _p2_grad / _log_mem / _reclaim
#
# 本类实现 tex 特有的：
#   create / _p0_shape_frozen_prepare / _p1_rollout / _p2_no_grad / _p2_submit
#   _decode_and_render / _p1_grad
#   _clean_p2_decode / _clean_for_vjp / _clean_p1_grad
#
# ★ 与 shape 版本的关键差异：
#   1. 新增 Phase 0: _p0_shape_frozen_prepare（shape forward + detach）
#   2. _p1_rollout 使用 rollout_tex（需要 shape_slat_norm 作为条件）
#   3. _decode_and_render 调用 decode_and_render_pbr（需要 meshes + subs）
#   4. _p1_grad VJP 传入 shape_cond（tex flow model 的 concat_cond）
#   5. 清理策略：_clean_for_vjp 保留 shape_slat_norm 供 VJP，_clean_p1_grad 释放之
#   6. P2 OOM / guidance 不可用 → skip_vjp=True（避免超大样本导致 NCCL timeout）
# =====================================================================

@dataclass
class PendingMicroBatch(PendingMicroBatchBase):
    """
    Tex 异步流水线 micro-batch — 继承公共 OOM/日志/清理逻辑。

    生命周期:
      .create(batch, ...)           ← 本类：shape_frozen_prepare + rollout_tex + submit
      .drain_guidance(system, ...)  ← 基类：_p2_wait → _p2_grad → _clean_for_vjp
      .drain_vjp(system, ...)       ← 基类：_p1_grad → _clean_p1_grad → 合并日志

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

    # ════════════════════════════════════════════════════════
    # 抽象方法实现
    # ════════════════════════════════════════════════════════

    def _get_model(self, system: Trellis2System):
        return system.tex.model

    def _get_reg_weight(self, system: Trellis2System) -> float:
        return system.cfg.tex.train.loss.reg

    def _decode_and_render(self, system: Trellis2System) -> Dict[str, Any]:
        """
        调用 decode_and_render_pbr，返回 render_out dict。

        ★ 与 shape 版本的差异：PBR 渲染需要 meshes + subs（来自 Phase 0）。
        """
        return decode_and_render_pbr(  # ★ PBR 而非 Normal
            self.state.features.meshes,      # ★ Phase 0 产出的 meshes
            self.state.features.tex_slat,    # ★ tex_slat 而非 shape_slat
            self.state.features.subs,        # ★ Phase 0 产出的 subs
            self.state.cameras,
            system.pipeline,
            system.tex.renderer,             # ★ tex.renderer（PBR + envmap）
            system.accelerator.device,
            resolution=system.pipeline.target_resolution,
        )

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
    ) -> "PendingMicroBatch":
        """
        工厂方法：P0 + P1-no-grad + P2-no-grad + submit → 创建 PendingMicroBatch。

        流水线前向阶段：
          attach_batch → shape_frozen_prepare → rollout_tex → decode+render_pbr(no_grad) → submit

        OOM 安全降级：
          P2-no-grad OOM → submitted=False → drain_guidance 跳过 P2-grad，跳过 VJP。
        """
        gen_seed = int(system.cfg.seed) + global_step + 1000  # ★ +1000 避免与 shape seed 冲突

        with TrainModeGuard(system.tex.model):  # ★ tex.model
            state = Trellis2State()  # ★ 来自 trellis2_tex，有 tex_slat/pbr_tensor 等字段
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.tex.config.cond_resolution)  # ★ tex.config

            # ── Phase 0: Shape 冻结前置（no_grad shape forward + detach） ──
            cls._p0_shape_frozen_prepare(state, system, global_step, profiler)

            # ── P1: Tex rollout（proxy chain） ──────────────────────
            tracker = cls._p1_rollout(state, system, gen_seed, profiler)

            # 创建实例（submitted=False，P2 成功后置 True）
            batch_size = len(batch['image_pils'])
            inst = cls(state=state, tracker=tracker,
                       global_step=global_step, batch_size=batch_size)

            # ── P2-no-grad + submit ──────────────────────────────
            try:
                comp_rgb = inst._p2_no_grad(system, profiler)
                inst._p2_submit(comp_rgb, system, profiler)
                inst.submitted = True
                del comp_rgb
            except torch.cuda.OutOfMemoryError:
                logging.warning(
                    f"[Step {global_step}] P2-no-grad OOM → reg-only"
                )
                profiler.reset()
            finally:
                inst._clean_p2_decode()

        return inst

    # ════════════════════════════════════════════════════════
    # Phase 步骤（tex 特有）
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _p0_shape_frozen_prepare(
        state: Trellis2State,
        system: Trellis2System,
        global_step: int,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """
        Phase 0: Shape 冻结前置 — no_grad 下执行 Shape forward + 全量 detach。

        完成后 state 具有：
        - coords: 稀疏坐标（detached）
        - features.shape_slat / shape_slat_norm: shape latent（detached）
        - features.subs / meshes: 几何中间结果（detached）
        """
        profiler.tick("shape_frozen_prepare")
        shape_frozen_prepare_no_grad(state, system, global_step)

    @staticmethod
    def _p1_rollout(
        state: Trellis2State,
        system: Trellis2System,
        gen_seed: int,
        profiler: AsyncPhaseProfiler,
    ) -> RolloutTracker:
        """P1: Tex rollout → 填充 state.features.tex_slat (proxy chain)，返回 tracker。"""
        profiler.tick("P1_rollout")
        return tex_phase1_rollout(state, system, gen_seed)

    def _p2_no_grad(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> torch.Tensor:
        """P2-no-grad: decode+render PBR（不保留 autograd 图）→ 返回 comp_rgb。"""
        profiler.tick("P2_no_grad")
        with torch.no_grad():
            comp_rgb = self._decode_and_render(system)["color"]

        self.state.views_generated.pbr_tensor = comp_rgb.detach()  # ★ pbr_tensor 而非 shape_tensor
        return comp_rgb

    def _p2_submit(
        self,
        comp_rgb: torch.Tensor,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> None:
        """P2-submit: 将 comp_rgb 异步提交给 guidance GPU。"""
        profiler.tick("P2_submit_async")
        guidance_weight = system.cfg.tex.train.loss.guidance  # ★ cfg.tex
        system.guidance.submit_async(
            comp_rgb,
            self.state.views_conditioned.image_pils,
            guidance_weight=guidance_weight,
            guidance_cfg=system.cfg.tex.guidance,  # ★ cfg.tex
            rank=system.accelerator.process_index,
        )

    def _p1_grad(self, system: Trellis2System) -> Dict[str, Any]:
        """
        P1-grad: VJP loop — 逐步/批量重算 f_θ，合并 guidance + reg 梯度 → θ.grad 累积。
        显存 O(1)，不随步数增长。

        ★ 与 shape 版本的核心差异：
        - stage = "tex"（而非 "shape"）
        - 传入 shape_cond = shape_slat_norm 作为 tex flow model 的 concat_cond
        - 使用 tex_slat 作为 x_t 的坐标载体

        DDP 安全：
        - 整个 VJP 循环在 model.no_sync() 下执行，backward 只做本地累积
        - 各 rank OOM 导致迭代次数不同也不会死锁
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        stage_config = pipeline.get_stage_config("tex")  # ★ "tex"
        flow_res = stage_config["flow_resolution"]
        reg_weight = system.cfg.tex.train.loss.reg  # ★ cfg.tex

        cond_emb, _ = self.state.extract_embeddings(resolution=flow_res)
        cond_emb = cond_emb.to(device)  # (B, S, C)

        # ★ Tex 独有：shape_slat_norm 作为 tex flow model 的 concat_cond
        shape_cond = self.state.features.shape_slat_norm

        model = system.tex.model  # ★ tex.model
        chunk_size = 4 # hard coded

        with model.no_sync():
            for x_t, t_batch, cond_k, v_grad, sc_k in _vjp_loader(
                self.tracker, self.state.features.tex_slat,
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
                        f"[Step {self.global_step}] P1-grad OOM → partial grad"
                    )
                    break

        return self._build_vjp_log()

    # ════════════════════════════════════════════════════════
    # 清理方法
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
        del self.tracker.input_trajectory[:], self.tracker.output_trajectory[:]
        del self.tracker.timesteps[:], self.tracker.reg_grads[:]
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
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    from edit4shape.systems.trellis2.system import build_dataloaders
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

    def _flush_vjp(pending: PendingMicroBatch) -> None:
        """drain_vjp → log → vis → empty_cache（避免重复 3 次）。"""
        step, bs = pending.global_step, pending.batch_size
        log = pending.drain_vjp(system, profiler)
        tex_logger.log_step(log, bs, step, epoch)  # ★ tex_logger
        if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
            visual_io.save_tex_train(state=pending.state, epoch=epoch, step=step)  # ★ save_tex_train
        pending._reclaim()

    def _sync_grads_and_step(n_accumulated: int) -> None:
        """
        手动 all-reduce 梯度 → 除以实际累积数 → optimizer.step → zero_grad。

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

            # ── Step 1: curr 前向 + submit（guidance GPU 尽早开始）──
            curr = PendingMicroBatch.create(batch, system, global_step, profiler)

            # ── Step 2: prev 的 P2（wait + P2-grad + clean）────────
            # ★ guidance GPU 已在处理 curr，与 P2-grad 并行。
            #   代价：P2-grad 时 GPU 额外持有 curr 残留数据（~0.5-1.5 GiB）。
            if prev is not None:
                prev.drain_guidance(system, profiler)

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
                    prev.drain_guidance(system, profiler)
                    _flush_vjp(prev)
                    prev = None
                _sync_grads_and_step(accum_steps)

        # ── epoch 结束：消化残留的 prev ──────────────────────────
        if prev is not None:
            prev.drain_guidance(system, profiler)
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
