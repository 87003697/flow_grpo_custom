"""
Trellis2 Shape+Tex 对比学习训练系统 — Contrastive + FlowEdit 异步流水线。

核心类 ContrastiveJob 管理一个 micro-batch 中 Shape + Tex 两阶段的完整生命周期。
使用 stage-list 驱动，扩展新 stage（如 shape-hq）只需添加一个 ContrastiveStageContext。

与 Distillation 异步的核心区别：
  - loss 在 velocity 空间（不在像素空间），drain 不需要 decode
  - GPU-2 只做 FlowEdit 编辑（不算 loss/grad），取回的是图片不是梯度
  - 每个 iter 只 submit 一次（teacher PBR 渲染图），不是每个 stage 各 submit 一次

训练循环（create → drain 两步交替）：

  curr = ContrastiveJob.create(batch, ...)
    ├── shared_prefix:
    │   ├── P0: dense_sampling
    │   ├── P_teacher: teacher 完整 shape+tex rollout (no_grad)
    │   ├── P_render: teacher decode + PBR render → src_image
    │   └── submit_edit_async(src_image) → GPU-2 FlowEdit 开始
    └── per-stage create (×2):
        └── add_noise → student velocity (with grad) → VelocityTracker

  prev.drain(system, profiler)
    ├── P_wait: wait_edit(prev) → tgt_image（取回编辑后图片）
    ├── P_enc: DINOv3(src_pils, 512/1024) → src_embeds, DINOv3(tgt_pils, 512/1024) → tgt_embeds
    └── per-stage drain (×2):
        ├── re-noise student x0: zt_stu = add_noise(z0_hat.detach(), t)
        ├── teacher denoise zt_stu with tgt cond → z0_tea_tgt (positive)
        ├── teacher denoise zt_stu with src cond → z0_tea_src (negative)
        ├── contrastive_loss(z0_hat, z0_tea_tgt, z0_tea_src).backward()
        └── relay_and_backward() → θ.grad

GPU-2 并行窗口分析（pipelined）：
  create(N) 提交 FlowEdit(N) → 在 per-stage create(N) + drain(N-1) 期间并行
  drain(N) 的 wait_edit(N) 时 FlowEdit(N) 早已完成（零等待）

依赖：
- TRELLIS.2 参考实现
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import os, sys
import gc
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

# =====================================================================
# TRELLIS.2 参考实现路径设置
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
from PIL import Image

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import VelocityTracker
from edit4shape.systems.trellis2.system import (
    Trellis2System, build_system as _build_system, build_dataloaders,
)
from edit4shape.systems.trellis2.forward import (
    trellis2_shape_forward,
    trellis2_tex_forward,
    decode_and_render_pbr,
    decode_and_render_normal,
    decode_and_render_normal_filled,
    dense_sampling_no_grad,
    detach_shape_outputs_for_tex,
)
from edit4shape.systems.trellis2.stage_ops import Trellis2ShapeOps, Trellis2TexOpsFromShape
from edit4shape.systems.base import TrainModeGuard, build_run_paths
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler
from edit4shape.systems.utils.pending_base import (
    ContrastiveStageContext,
    ContrastivePendingJob as _ContrastiveBase,
    _reclaim,
)
from edit4shape.systems.trellis2.forward import _detach_shape_outputs

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from edit4shape.guidance.pipeline_parallel import AsyncEditResult
from trellis2.utils.grad_clip_utils import AdaptiveGradClipper


# =====================================================================
# ContrastiveJob — Shape+Tex 对比学习异步流水线 micro-batch
# =====================================================================

@dataclass
class ContrastiveJob(_ContrastiveBase):
    """
    Shape+Tex 对比学习异步流水线 micro-batch。

    stage-list 驱动，当前配置：
      [Trellis2ShapeOps, Trellis2TexOpsFromShape]

    扩展 shape-hq 只需在 stage_entries 中插入一行。
    """

    # ════════════════════════════════════════════════════════
    # 公开 API — create / drain
    # ════════════════════════════════════════════════════════

    @classmethod
    def create(
        cls,
        batch: Dict[str, Any],
        system: Trellis2System,
        global_step: int,
        profiler: AsyncPhaseProfiler,
    ) -> "ContrastiveJob":
        """
        工厂方法：shared_prefix + per-stage create。

        执行流程：
          1. State 初始化 + DINOv3 encode
          2. Rollout + render: rollout(shape+tex) → decode → PBR render
          3. submit_edit_async → GPU-2 FlowEdit
          4. Per-stage create: add_noise → student velocity → tracker
        """
        cfg = system.cfg

        # ── Stage entries（stage-list 驱动）──
        stage_entries = [
            ContrastiveStageContext(ops=Trellis2ShapeOps()),
            ContrastiveStageContext(ops=Trellis2TexOpsFromShape()),
        ]

        # ── 1. State 初始化 ──
        state = Trellis2State()
        state.attach_batch(
            batch, pipeline=system.pipeline,
            resolution=system.tex.config.cond_resolution,
        )
        batch_size = len(batch['image_pils'])

        # ── 2. Rollout + render → src_render ──
        src_render, src_image_pils = cls._rollout_and_render(
            state, system, profiler, stage_entries,
        )

        # ── 3. Submit edit ──
        edit_submitted = cls._submit_edit(
            system, profiler, src_render, batch, global_step,
        )
        del src_render

        # ── 4. Per-stage create: student velocity ──
        inst = cls(
            state=state,
            stage_entries=stage_entries,
            global_step=global_step,
            batch_size=batch_size,
            src_image_pils=src_image_pils,
            edit_submitted=edit_submitted,
        )

        reg_logs = []
        for i, entry in enumerate(stage_entries):
            with TrainModeGuard(entry.ops.get_model(system)):
                reg_logs.append(inst._create_stage(i, system, profiler))
        inst._reg_logs = reg_logs

        return inst

    def drain(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Dict[str, Any]:
        """
        Drain: wait_edit → encode src+tgt → per-stage contrastive drain。

        Returns:
            合并后的日志字典
        """
        ada = bool(system.cfg.contrastive.ada)
        eps = float(system.cfg.contrastive.eps)
        merged_log: Dict[str, Any] = {}

        # ── 1. Wait FlowEdit → tgt PILs ──
        tgt_image_pils = self._wait_edit(system, profiler)
        if tgt_image_pils is None:
            self._cleanup_on_skip()
            raise RuntimeError(
                f"[Step {self.global_step}] FlowEdit 返回 None — "
                f"edit_submitted={self.edit_submitted}，无法继续 contrastive drain"
            )

        # ── 2. DINOv3 encode src + tgt ──
        src_embeds = self._encode_images(system, profiler, self.src_image_pils, "encode_src")
        tgt_embeds = self._encode_images(system, profiler, tgt_image_pils, "encode_tgt")
        del tgt_image_pils

        # ── 2.5. DINO similarity + adaptive swap ──
        profiler.tick("dino_sim")
        merged_log.update(self._dino_similarity_and_swap(
            src_embeds, tgt_embeds,
            adaptive_swap=bool(system.cfg.contrastive.adaptive_swap),
        ))

        # ── 3. Per-stage contrastive drain ──
        n_stages = len(self.stage_entries)
        for i in range(n_stages):
            stage_log = self._drain_contrastive_stage(
                i, system, profiler, tgt_embeds, src_embeds,
                self._reg_logs[i],
                ada=ada, eps=eps,
                collect_profiler=(i == n_stages - 1),
            )
            merged_log.update(stage_log)

        # ── 4. Cleanup ──
        self._cleanup()
        return merged_log

    # ════════════════════════════════════════════════════════
    # 内部 Helper — create 阶段
    # ════════════════════════════════════════════════════════

    @classmethod
    def _rollout_and_render(
        cls,
        state: Trellis2State,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
        stage_entries: list,
    ):
        """
        Rollout (shape+tex) → decode → PBR render。

        使用 pretrained_rollout（可能是 teacher 或 student，取决于 rollout_mode 配置）。

        Returns:
            (src_render, src_image_pils):
                src_render — (B, V, H, W, 3) tensor
                src_image_pils — List[PIL.Image]
        """
        cfg = system.cfg
        device = system.accelerator.device
        seed = int(cfg.seed)

        profiler.tick("dense_sampling")
        dense_sampling_no_grad(state, system)

        profiler.tick("tea_shape_rollout")
        stage_entries[0].ops.pretrained_rollout(state, system, seed)

        # Shape decode → subs + meshes（Tex decode 和 PBR render 需要）
        profiler.tick("tea_shape_decode")
        renderer_type = cfg.shape.renderer.type
        if renderer_type == "mesh_filled":
            shape_decode_fn = decode_and_render_normal_filled
        else:
            shape_decode_fn = decode_and_render_normal
        with torch.no_grad():
            shape_out = shape_decode_fn(
                state.shape.z0,
                state.cameras,
                system.pipeline,
                system.shape.renderer,
                device,
                resolution=system.pipeline.target_resolution,
                decode_only=True,
                bg_color=tuple(cfg.shape.renderer.bg_color),
                grad_shrink_scale=1.0,
                max_hole_perimeter=cfg.shape.renderer.max_hole_perimeter,
                is_training=False,  # no_grad 下关闭 checkpoint，避免无意义的额外显存占用
            )
        state.shape.subs = shape_out["subs"]
        state.shape.meshes = shape_out["meshes"]

        detach_shape_outputs_for_tex(state)

        profiler.tick("tea_tex_rollout")
        stage_entries[1].ops.pretrained_rollout(state, system, seed)

        profiler.tick("tea_decode_render")
        with torch.no_grad():
            render_out = decode_and_render_pbr(
                state.shape.meshes, state.tex.z0, state.shape.subs,
                state.cameras, system.pipeline, system.tex.renderer, device,
                resolution=system.pipeline.target_resolution,
                bg_color=tuple(cfg.tex.renderer.bg_color),
                grad_shrink_scale=1.0,
            )
        src_render = render_out["color"]  # (B, V, H, W, 3)
        state.views_generated.pbr_tensor = src_render.detach()
        _to_pil = Trellis2VisualIO.to_pil
        src_image_pils = [_to_pil(src_render[b, 0]) for b in range(src_render.shape[0])]

        del render_out
        state.release_shape_spatial_cache()
        state.release_tex_spatial_cache()
        torch.cuda.empty_cache()

        return src_render, src_image_pils

    @staticmethod
    def _submit_edit(
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
        src_render: torch.Tensor,
        batch: Dict[str, Any],
        global_step: int,
    ) -> bool:
        """Submit FlowEdit to GPU-2. Returns True if successful."""
        try:
            profiler.tick("submit_edit")
            system.guidance.submit_edit_async(
                src_render,
                batch['image_pils'],
                guidance_cfg=system.cfg.tex.guidance,
                rank=system.accelerator.process_index,
            )
            return True
        except Exception as e:
            logging.warning(f"[Step {global_step}] submit_edit_async failed: {e}")
            return False

    # ════════════════════════════════════════════════════════
    # 内部 Helper — drain 阶段
    # ════════════════════════════════════════════════════════

    def _wait_edit(
        self,
        system: Trellis2System,
        profiler: AsyncPhaseProfiler,
    ) -> Optional[List[Image.Image]]:
        """Wait for FlowEdit result. Returns tgt PIL list or None."""
        if not self.edit_submitted:
            return None

        profiler.tick("wait_edit")
        try:
            edit_result: AsyncEditResult = system.guidance.wait_edit(
                target_device=system.accelerator.device,
            )
            # 挂载 vis
            self.state.views_edited.image_tensor = edit_result.edited_imgs
            self.state.views_edited.trackers = edit_result.trackers

            # guid GPU 计时
            if edit_result.guid_wall_start is not None:
                profiler.set_guid_timing(
                    edit_result.guid_wall_start, edit_result.guid_wall_end,
                )

            _to_pil = Trellis2VisualIO.to_pil
            edited = edit_result.edited_imgs  # (B, V, C, H, W)
            pils = [_to_pil(edited[b, 0].permute(1, 2, 0)) for b in range(edited.shape[0])]
            del edit_result
            return pils
        except Exception as e:
            logging.warning(
                f"[Step {self.global_step}] wait_edit failed: {e} → 跳过 contrastive"
            )
            return None

    def _cleanup_on_skip(self) -> None:
        """FlowEdit 不可用时：清理 tracker + gc。"""
        for entry in self.stage_entries:
            if entry.vel_tracker is not None:
                del entry.vel_tracker.v_student, entry.vel_tracker.v_proxy
        _reclaim()

    def _cleanup(self) -> None:
        """Drain 结束后：detach + release + offload + gc。"""
        self.state.detach_features()
        self.state.release_uncond_embeddings()
        self.state.offload_vis_to_cpu()
        _reclaim()


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。

    同时训练 Shape 和 Tex 两个 Flow Model，使用 Contrastive 策略 +
    FlowEdit 异步流水线。
    - Teacher rollout → decode → PBR render → FlowEdit 编辑
    - 3-arm contrastive loss: student x0 靠近 teacher_tgt(positive)、远离 teacher_src(negative)

    配置文件示例：
        python -m edit4shape.systems.trellis2.entries.shape_tex_contrastive_async \
            --config=configs/trellis2_shape_tex_contrastive.py
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
            project_name="trellis2-shape+tex-contrastive",
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
        from edit4shape.systems.trellis2.forward import evaluate as _evaluate
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
    # Step 8: 训练循环（Contrastive + FlowEdit 异步流水线）
    # =====================================================
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    profiler = AsyncPhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.gradient_accumulation_steps)

    shape_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)
    tex_grad_clipper = AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=10)

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

        prev: Optional[ContrastiveJob] = None
        accum_count = 0

        for batch in train_loader:
            global_step += 1

            # ── 1. CREATE(N): teacher rollout + submit_edit + student velocity ──
            curr = ContrastiveJob.create(batch, system, global_step, profiler)

            # ── 2. DRAIN(N-1): wait_edit + DINOv3 encode + contrastive loss ──
            if prev is not None:
                train_log = prev.drain(system, profiler)
                train_logger.log_step(train_log, prev.batch_size, prev.global_step, epoch)

                # vis
                if accelerator.is_main_process and (prev.global_step % visual_io.vis_freq == 0):
                    visual_io.save_tex_train(state=prev.state, epoch=epoch, step=prev.global_step)

                accum_count += 1
                if accum_count >= accum_steps:
                    _sync_grads_and_step(
                        system.shape.model, system.shape.optimizer,
                        accum_count, shape_grad_clipper,
                    )
                    _sync_grads_and_step(
                        system.tex.model, system.tex.optimizer,
                        accum_count, tex_grad_clipper,
                    )
                    accum_count = 0

                del prev
                _reclaim()

            prev = curr

        # ── epoch 结束：消化残留的 prev ──
        if prev is not None:
            train_log = prev.drain(system, profiler)
            train_logger.log_step(train_log, prev.batch_size, prev.global_step, epoch)
            accum_count += 1
            del prev
            _reclaim()

        if accum_count > 0:
            _sync_grads_and_step(
                system.shape.model, system.shape.optimizer,
                accum_count, shape_grad_clipper,
            )
            _sync_grads_and_step(
                system.tex.model, system.tex.optimizer,
                accum_count, tex_grad_clipper,
            )

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            from edit4shape.systems.trellis2.forward import evaluate as _evaluate
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
