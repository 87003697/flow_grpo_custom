"""
Trellis v1 StageOps 具体实现 — 单阶段（SLAT Flow Model）。

ABC (StageOps) 定义在 edit4shape.systems.utils.stage_ops，
本文件提供 Trellis v1 特定的实现，使其可接入 autograd_template 编排模板。

实现：
  TrellisOps      — 通用单阶段，根据 cfg.renderer.type 分发 mesh/gs 渲染
  TrellisMeshOps  — 强制 Mesh Normal 渲染（覆写 decode_render_dict）
  TrellisGsOps    — 强制 GS Color 渲染（覆写 decode_render_dict）

使用方式：
  from edit4shape.systems.trellis.stage_ops import TrellisOps
  from edit4shape.systems.trellis.autograd_template import trellis_three_phase_step

  trellis_three_phase_step(TrellisOps(), state, system, ...)

设计原则：
  - 同一个 TrellisOps 在 standard / autograd / bilevel 入口中一行不改
  - 子类仅需覆写 decode_render_dict 即可切换渲染策略
  - 清理策略由编排层的 clean_for_vjp 回调注入
  - vjp_loop 手动合并 reg_grads（适配 3-sub-step Phase 2）
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from edit4shape.guidance.pipelines.utils.loss_functions import contrastive_loss_step

# ABC 从 utils 导入（模型无关的抽象层）
from edit4shape.systems.utils.stage_ops import StageOps  # noqa: F401 — re-export

# Phase 函数 & 渲染
from edit4shape.systems.trellis.forward import decode_and_render_mesh, decode_and_render_gs
from edit4shape.systems.trellis.phases import phase3_rollout_grad_backward
from edit4shape.generators.trellis.rollout import rollout_sparse, rollout_dense, RolloutTracker
from edit4shape.generators.trellis.rollout.base import (
    predict_sparse_velocity_with_cfg,
    prepare_embeddings,
    predict_dense_velocity_with_cfg,
)


# =====================================================================
# 通用实现 — 根据 cfg.renderer.type 分发
# =====================================================================

class TrellisOps(StageOps):
    """
    Trellis v1 单阶段 Ops — SLAT Flow Model。

    计算链：
      dense_sampling → rollout_sparse → decode_and_render_{mesh|gs} → guidance → VJP

    Phase 2 保持 3-sub-step 显存优化：
      P2a: no_grad decode → detached comp_rgb
      P2b: guidance-only backward → rgb_grad
      P2c: with-grad decode → backward(rgb_grad) → cond_proxy.grad（仅 guidance）

    Phase 3 手动合并 reg_grads：
      v_grad = guidance_grad + reg_weight * reg_grad → VJP
    """

    # ═══════════════════════════════════════════════════════
    # 配置查询
    # ═══════════════════════════════════════════════════════

    def get_model(self, system):
        """返回 DDP 包装的 slat_flow_model。"""
        return system.pipeline.pipe.models['slat_flow_model']

    def get_stage_name(self) -> str:
        return "slat"

    def get_reg_weight(self, system) -> float:
        return system.cfg.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.train.guidance

    # ═══════════════════════════════════════════════════════
    # Async 友好查询
    # ═══════════════════════════════════════════════════════

    def get_latent(self, state):
        return state.stage2.z0

    # get_shape_cond → 继承默认 None（单模型无 shape cond）

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        根据 cfg.renderer.type 分发到 mesh 或 gs 渲染。

        子类可覆写此方法以实现自定义渲染策略
        （如 TrellisMeshOps / TrellisGsOps / 自定义混合渲染）。
        """
        latents = state.stage2.z0
        device = system.accelerator.device

        renderer_type = system.cfg.renderer.type
        renderer = system.renderers[renderer_type]  # 从 renderers dict 查找
        if renderer_type == "gs":
            return decode_and_render_gs(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras,
                system.pipeline, renderer, device,
            )
            render_out["color"] = render_out["normal"]
            return render_out

    # ═══════════════════════════════════════════════════════
    # Phase 函数
    # ═══════════════════════════════════════════════════════

    def pre_rollout(self, state, system, global_step) -> None:
        """Phase 0: Dense Sampling → 填充 state.coords。

        根据 cfg.train.rollout_mode 选择 student / pretrained dense rollout。
        若 student 产生 empty coords（decode 阈值 > 0 全为 False），
        自动 fallback 到 pretrained teacher 并设 state.dense_fallback = True。
        """
        pipeline = system.pipeline
        cfg = system.cfg
        device = system.accelerator.device
        generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.dense_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}")

        with ctx, torch.no_grad():
            rollout_dense(state, cfg, system, device, generator=generator)

        batch_size = state.stage1.z0.shape[0]
        coords = pipeline.dense.decode_to_coords(
            state.stage1.z0, batch_size=batch_size,
        )

        # ── Fallback: student empty coords → teacher ──
        state.dense_fallback = False
        if coords.numel() == 0 and rollout_mode == "student":
            logging.warning(
                "[pre_rollout] Student dense rollout produced empty coords, "
                "falling back to pretrained teacher."
            )
            generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
            with system.strategy.dense_teacher_context(), torch.no_grad():
                rollout_dense(state, cfg, system, device, generator=generator)
            coords = pipeline.dense.decode_to_coords(
                state.stage1.z0, batch_size=batch_size,
            )
            state.dense_fallback = True

        state.coords = coords

    def rollout(self, state, system, seed) -> RolloutTracker:
        """
        Phase 1: rollout_sparse → proxy chain + tracker。

        对齐 StageOps 签名：接收 seed（由编排函数计算），
        内部创建 Generator 并调用 rollout_sparse。
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)
        tracker = RolloutTracker()

        rollout_sparse(
            state, cfg, system, device,
            generator=generator,
            is_training=False,   # 模型推理 no_grad
            tracker=tracker,     # ★ 记录 proxy 轨迹
        )
        torch.cuda.empty_cache()
        return tracker

    def decode_render(self, state, system) -> torch.Tensor:
        """
        Phase 2a: decode + render → comp_rgb（含 autograd 图，连接到 proxy chain）。

        调用 decode_render_dict + 挂载可视化数据。
        """
        render_out = self.decode_render_dict(state, system)
        comp_rgb = render_out["color"]  # (B, V, H, W, C)
        state.views_generated.image_tensor = comp_rgb.detach()
        return comp_rgb

    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: VJP → θ.grad 累积（委托 phases.phase3_rollout_grad_backward）。

        ★ 适配 3-sub-step Phase 2：cond_proxy.grad 仅含 guidance 梯度，
        phase3_rollout_grad_backward 内部手动合并 reg_weight * reg_grads[i]。
        """
        reg_weight = self.get_reg_weight(system)
        log = tracker.collect_log(reg_weight=reg_weight)  # loss/reg + grad_norm/*（VJP 前收集）

        phase3_rollout_grad_backward(
            state, system, system.cfg, system.accelerator.device, tracker,
        )

        return log


# =====================================================================
# 显式渲染策略子类
# =====================================================================

class TrellisMeshOps(TrellisOps):
    """强制使用 Mesh Normal 渲染，不受 cfg.renderer.type 控制。"""

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        render_out = decode_and_render_mesh(
            state.stage2.z0, state.cameras,
            system.pipeline, system.renderers["mesh"], system.accelerator.device,
        )
        render_out["color"] = render_out["normal"]
        return render_out


class TrellisGsOps(TrellisOps):
    """强制使用 GS Color 渲染，不受 cfg.renderer.type 控制。"""

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        return decode_and_render_gs(
            state.stage2.z0, state.cameras,
            system.pipeline, system.renderers["gs"], system.accelerator.device,
        )


# =====================================================================
# 双路渲染 — Mesh Normal + GS Color 同时提供 guidance
# =====================================================================

class TrellisHybridOps(TrellisOps):
    """
    双路渲染 Ops：Mesh Normal + GS Color，各自独立提供 guidance。

    设计：
      - decode_render_dict(renderer_key=...) 按指定渲染器分发
      - get_render_passes() 返回多路渲染配置，供 hybrid 编排循环使用
      - 继承 TrellisOps 的 rollout / pre_rollout / vjp_loop（共享 SLAT Flow Model）

    使用方式：
      from edit4shape.systems.trellis.stage_ops import TrellisHybridOps
      from edit4shape.systems.trellis.autograd_template import trellis_hybrid_three_phase_step

      trellis_hybrid_three_phase_step(TrellisHybridOps(), state, system, ...)

    配置要求（cfg.train 下）：
      guidance_normal:      Mesh Normal guidance 配置
      guidance_color:       GS Color guidance 配置
      loss.guidance_normal: Mesh Normal guidance 权重
      loss.guidance_color:  GS Color guidance 权重
    """

    def decode_render_dict(self, state, system, renderer_key: str = "gs") -> Dict[str, Any]:
        """
        按 renderer_key 分发到指定渲染器。

        Args:
            state: TrellisState
            system: TrellisSystem（需 renderers 包含 "mesh" 和 "gs"）
            renderer_key: "mesh" 或 "gs"

        Returns:
            渲染输出字典，"color" key 统一为各渲染器的主要输出：
              mesh → normal,  gs → color
        """
        latents = state.stage2.z0
        device = system.accelerator.device
        renderer = system.renderers[renderer_key]

        if renderer_key == "gs":
            return decode_and_render_gs(
                latents, state.cameras, system.pipeline, renderer, device,
            )
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, renderer, device,
            )
            render_out["color"] = render_out["normal"]
            return render_out

    def get_render_passes(self, system) -> List[Tuple[str, Any, float]]:
        """
        返回多路渲染配置列表。

        Returns:
            [(renderer_key, guidance_cfg, guidance_weight), ...] 的列表。
            编排函数会循环处理每一路：P2a → P2b → P2c。
        """
        cfg = system.cfg
        return [
            ("mesh", cfg.train.guidance_normal, cfg.train.loss.guidance_normal),
            ("gs", cfg.train.guidance_color, cfg.train.loss.guidance_color),
        ]


# =====================================================================
# FlowEdit 训练 — Pretrained Rollout + Finetuned 单步去噪
# =====================================================================

class TrellisFlowEditOps(TrellisOps):
    """
    FlowEdit Ops：Rollout + Finetuned 单步去噪 + 2D FlowEdit Guidance。

    训练流程：
      1. pre_rollout()           — Dense Sampling（复用 TrellisOps）
      2. rollout()               — no_grad 完整 rollout → z₀（pretrained 或 student）
      3. add_noise(z₀, t)       — 随机采样 t，flow matching 加噪 → zₜ（归一化域）
      4. finetune_denoise(zₜ,t) — finetuned model 单步预测速度 → ẑ₀（有梯度）
      5. denormalize(ẑ₀)        — 反归一化到 decoder 输入空间
      6. decode_render_dict()    — decoder → 渲染 comp_rgb（复用 TrellisOps）

    不需要 VJP、proxy chain、no_sync hack，标准 autograd 即可。

    梯度传播路径：
      loss → rgb_grad → decoder(frozen, 有计算图) → ẑ₀ → finetune_denoise → θ.grad

    配置要求（cfg.train 下）：
      rollout_mode:  "pretrained" | "student"（P1 使用哪个模型 rollout）
      noise.t_min:   时间步采样下界（默认 0.02）
      noise.t_max:   时间步采样上界（默认 0.98）
    """

    # ═══════════════════════════════════════════════════════
    # Rollout（根据 cfg.train.rollout_mode 选择模型）
    # ═══════════════════════════════════════════════════════

    def rollout(self, state, system, seed) -> None:
        """
        完整 rollout → clean z₀，no_grad。

        根据 cfg.train.rollout_mode 选择使用哪个模型：
          - "pretrained"：sparse_teacher_context()，使用 pretrained 权重（off-policy）
          - "student"：直接使用当前 finetuned 权重（on-policy）

        完全 no_grad，不需要任何 proxy chain。

        Side Effects:
            - state.stage2.z0: 挂载 rollout 输出的 SparseTensor（反归一化后）
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.sparse_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}, expected 'pretrained' or 'student'")

        with ctx, torch.no_grad():
            rollout_sparse(
                state, cfg, system, device,
                generator=generator,
                is_training=False,
                tracker=None,  # 不需要 proxy chain
            )
        # rollout_sparse 已经做了反归一化并挂载到 state.stage2.z0
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # 加噪：z₀ → zₜ（归一化域操作）
    # ═══════════════════════════════════════════════════════

    def normalize_latent(self, state, system) -> torch.Tensor:
        """
        将 state.stage2.z0.feats 从反归一化域 → 归一化域。

        rollout_sparse 输出的 slat 已经反归一化（denorm_feats = feats * std + mean），
        加噪/去噪需要在归一化域进行（与训练时一致）。

        Returns:
            normalized_feats: (N, C) 归一化后的特征
        """
        norm = system.pipeline.pipe.slat_normalization
        device = system.accelerator.device
        std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
        denorm_feats = state.stage2.z0.feats  # (N, C)
        return (denorm_feats - mean) / std  # (N, C)

    def add_noise(self, z0_feats: torch.Tensor, t: float, generator=None) -> torch.Tensor:
        """
        Flow matching 加噪：zₜ = (1-t) * z₀ + t * ε

        Args:
            z0_feats: (N, C) 归一化域的 clean features
            t: 标量时间步 ∈ (0, 1)
            generator: 随机数生成器

        Returns:
            zt_feats: (N, C) 加噪后的特征
        """
        noise = torch.randn_like(z0_feats)
        zt = (1.0 - t) * z0_feats + t * noise
        return zt

    def sample_timestep(self, system) -> float:
        """
        从 inference scheduler 的时间步序列中随机采样一个时间步。

        使用 scheduler 的实际时间步（经过 mu-shift 和 rescale_t），
        而非简单的 Uniform(t_min, t_max)，确保对齐 inference 分布。

        Returns:
            t_val: 采样到的时间步值（范围 [0, 1]，与 scheduler.timesteps 一致）
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        slat_steps, _, slat_rescale_t, _, _, _ = pipeline.sparse.get_runtime_params()

        scheduler = pipeline.sparse.scheduler()
        scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
        timesteps = scheduler.timesteps  # Tensor, 从大到小排列, [0, 1] 范围

        # 去掉最后一个（通常接近 0）
        timesteps = timesteps[:-1]

        # 可选：限制采样范围（t_min / t_max 在 config 中已经是 [0,1] 范围）
        cfg = system.cfg
        t_min = float(cfg.train.noise.get("t_min", 0.02))
        t_max = float(cfg.train.noise.get("t_max", 0.98))
        mask = (timesteps >= t_min) & (timesteps <= t_max)
        valid_timesteps = timesteps[mask]
        if len(valid_timesteps) == 0:
            valid_timesteps = timesteps  # fallback

        # 随机选一个
        idx = torch.randint(0, len(valid_timesteps), (1,)).item()
        return float(valid_timesteps[idx].item())

    # ═══════════════════════════════════════════════════════
    # Velocity 预测（student / teacher）
    # ═══════════════════════════════════════════════════════

    def _predict_velocity_impl(
        self, state, system, zt_feats: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        内部共享：构建 SparseTensor 输入 + predict_sparse_velocity_with_cfg → v_feats。

        调用方决定是否在 sparse_teacher_context / no_grad 下调用。

        Args:
            state: TrellisState（需要 state.stage2.z0 提供 coords）
            system: TrellisSystem
            zt_feats: (N, C) 归一化域特征
            t_val: 时间步值（[0, 1000] 范围）
            cond: (B, S, C) 条件编码（可选，不传则从 state 读取）
            uncond: (B, S, C) 无条件编码（可选，不传则从 state 读取）

        Returns:
            v_feats: (N, C) 预测的速度特征
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        _, slat_guidance, _, slat_cfg_min, slat_cfg_max, _ = pipeline.sparse.get_runtime_params()

        # 构建 SparseTensor 输入
        slat = state.stage2.z0
        x_t = slat.replace(zt_feats)  # SparseTensor with zt_feats, 共享 coords

        # 条件编码
        cond_emb, uncond_emb = prepare_embeddings(state, device, cond, uncond)

        velocity = predict_sparse_velocity_with_cfg(
            pipeline, x_t, t_val, cond_emb, uncond_emb,
            slat_guidance, slat_cfg_min, slat_cfg_max, device,
        )  # SparseTensor
        return velocity.feats  # (N, C)

    def predict_velocity_student(self, state, system, zt_feats: torch.Tensor, t_val: float) -> torch.Tensor:
        """
        Finetuned (student) model 速度预测（有 autograd 图到 θ）。

        Returns:
            v_feats: (N, C) 有 autograd 图
        """
        return self._predict_velocity_impl(state, system, zt_feats, t_val)

    def predict_velocity_teacher(
        self, state, system, zt_feats: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pretrained (teacher) model 速度预测（sparse_teacher_context + no_grad，detached）。

        Args:
            cond: (B, S, C) 条件编码（可选，不传则从 state 读取）
            uncond: (B, S, C) 无条件编码（可选，不传则从 state 读取）

        Returns:
            v_feats: (N, C) detached
        """
        with system.strategy.sparse_teacher_context(), torch.no_grad():
            v = self._predict_velocity_impl(state, system, zt_feats, t_val, cond=cond, uncond=uncond)
        return v.detach()

    def denormalize_feats(self, feats: torch.Tensor, system) -> torch.Tensor:
        """
        反归一化：归一化域特征 → decoder 输入域。

        Args:
            feats: (N, C) 归一化域特征
            system: TrellisSystem

        Returns:
            denorm_feats: (N, C) 反归一化后的特征
        """
        norm = system.pipeline.pipe.slat_normalization
        device = system.accelerator.device
        std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
        return feats * std + mean  # (N, C)


# =====================================================================
# Contrastive FlowEdit Ops — 在 latent 空间做对比学习
# =====================================================================

class TrellisContrastiveOps(TrellisFlowEditOps):
    """
    Contrastive FlowEdit Ops：拓展 P4/P5 为 latent 空间对比学习。

    继承 TrellisFlowEditOps 的 P0–P3.5（dense sampling、pretrained rollout、
    加噪、velocity 预测、velocity reg）。

    新增方法（P4/P5）：
      decode_render_teacher   — 渲染 Teacher z₀ → src tensor（挂载到 state）
      edit_views              — FlowEdit 编辑 src → tgt tensor（从 state 读取 src）
      encode_conditions       — tensor→PIL→preprocess→DINOv2 编码 → c_src, c_tgt
      predict_x0_teacher_with_cond — Teacher 单步去噪 with custom condition
      contrastive_loss        — 对比 loss（latent 空间）

    梯度传播路径：
      contrastive_loss → z0_hat_norm → v_proxy → (relay) → v_student → θ.grad
    """

    def decode_render_teacher(self, state, system) -> None:
        """
        P4a: 渲染 Teacher z₀ → src 图像 tensor。

        前置条件: state.stage2.z0 == z0_teacher（P1 结束后未被修改）。

        Side Effects:
            - state.views_generated.image_tensor: (B, V, H, W, C) detached
        """
        state.stage2.z0._spatial_cache.clear()
        torch.cuda.empty_cache()

        with torch.no_grad():
            render_out = self.decode_render_dict(state, system)
        state.views_generated.image_tensor = render_out["color"].detach()  # (B, V, H, W, C)

        state.stage2.z0._spatial_cache.clear()
        del render_out
        torch.cuda.empty_cache()

    def edit_views(self, state, system) -> None:
        """
        P4b: FlowEdit 编辑 src → tgt。

        从 state.views_generated.image_tensor 读取 src，
        使用 system.guidance.compute_guidance 做 2D FlowEdit 编辑，
        结果统一为 (B, V, H, W, C) 挂载到 state.views_edited。

        Side Effects:
            - state.views_edited.image_tensor: (B, V, C, H, W) detached
            - state.trackers.guidance:         List[StateTracker]
        """
        src_tensor = state.views_generated.image_tensor  # (B, V, H, W, C)
        guidance_cfg = self.get_guidance_cfg(system)
        accelerator = system.accelerator

        guidance_result = system.guidance.edit(
            src_tensor,
            state.views_conditioned.image_pils,
            guidance_cfg=guidance_cfg,
            rank=accelerator.process_index,
        )

        edited_imgs = guidance_result.edited_imgs  # (B, V, C, H, W)
        state.views_edited.image_tensor = edited_imgs.detach()

        state.trackers.guidance = guidance_result.trackers

        del guidance_result
        torch.cuda.empty_cache()

    def encode_conditions(self, state, system) -> None:
        """
        P4c: tensor → PIL → preprocess → DINOv2 编码 → c_src, c_tgt。

        从 state 读取 image_tensor（均为 (B, V, H, W, C)），取 view 0 转 PIL，
        经 preprocess_image（rembg 去背 + bbox crop 居中 + resize 518）后
        由 encode_image（DINOv2）编码为 patch token。

        Side Effects:
            - state.views_generated.image_condition: (B, S, C)
            - state.views_edited.image_condition:    (B, S, C)
        """
        pipe = system.pipeline.pipe

        # src: (B, V, H, W, C), tgt: (B, V, C, H, W)
        src_tensor = state.views_generated.image_tensor  # (B, V, H, W, C)
        tgt_tensor = state.views_edited.image_tensor     # (B, V, C, H, W)
        src_pils = [
            Image.fromarray((src_tensor[b, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            for b in range(src_tensor.shape[0])
        ]
        tgt_pils = [
            Image.fromarray((tgt_tensor[b, 0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            for b in range(tgt_tensor.shape[0])
        ]

        with torch.no_grad():
            src_proc = [pipe.preprocess_image(img) for img in src_pils]
            tgt_proc = [pipe.preprocess_image(img) for img in tgt_pils]
            c_src = pipe.encode_image(src_proc)  # (B, S, C)
            c_tgt = pipe.encode_image(tgt_proc)  # (B, S, C)

        state.views_generated.image_condition = c_src
        state.views_edited.image_condition = c_tgt

    @staticmethod
    def compute_dino_similarity(state) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
        """
        计算 src / tgt 与 input 的 DINO cosine similarity。

        读取:
            state.views_conditioned.cond_embed       — (B, S, C) input 图像 DINO embedding
            state.views_generated.image_condition     — (B, S, C) src (Teacher 渲染)
            state.views_edited.image_condition        — (B, S, C) tgt (FlowEdit 编辑)

        Returns:
            log_dict: sim/src_input, sim/tgt_input, sim/tgt_gt_src
            sim_src: (B,) per-sample cosine similarity
            sim_tgt: (B,) per-sample cosine similarity
        """
        c_input = state.views_conditioned.cond_embed       # (B, S, C)
        c_src = state.views_generated.image_condition       # (B, S, C)
        c_tgt = state.views_edited.image_condition          # (B, S, C)

        # mean pool over patch tokens → (B, C)
        e_input = c_input.mean(dim=1)
        e_src = c_src.mean(dim=1)
        e_tgt = c_tgt.mean(dim=1)

        sim_src = torch.nn.functional.cosine_similarity(e_src, e_input, dim=-1)  # (B,)
        sim_tgt = torch.nn.functional.cosine_similarity(e_tgt, e_input, dim=-1)  # (B,)

        log_dict = {
            "sim/src_input": sim_src.mean().item(),
            "sim/tgt_input": sim_tgt.mean().item(),
            "sim/tgt_gt_src": (sim_tgt > sim_src).float().mean().item(),
        }
        return log_dict, sim_src, sim_tgt

    @staticmethod
    def adaptive_swap_conditions(state, sim_src: torch.Tensor, sim_tgt: torch.Tensor) -> float:
        """
        Per-sample 对调 c_src / c_tgt：当 sim_src > sim_tgt 时交换。

        修改:
            state.views_generated.image_condition
            state.views_edited.image_condition

        Returns:
            swap_rate: 被交换的 sample 比例
        """
        swap_mask = sim_src > sim_tgt  # (B,)
        swap_rate = swap_mask.float().mean().item()

        if swap_mask.any():
            c_src = state.views_generated.image_condition  # (B, S, C)
            c_tgt = state.views_edited.image_condition     # (B, S, C)
            mask = swap_mask[:, None, None]                # (B, 1, 1)
            new_src = torch.where(mask, c_tgt, c_src)
            new_tgt = torch.where(mask, c_src, c_tgt)
            state.views_generated.image_condition = new_src
            state.views_edited.image_condition = new_tgt

        return swap_rate

    def predict_x0_teacher_with_cond(
        self,
        state,
        system,
        zt_feats: torch.Tensor,
        t_val: float,
        cond_source: str = "edited",
    ) -> torch.Tensor:
        """
        P5a/b: Teacher 单步去噪 with condition from state → ẑ₀。

        临时替换 state 中的条件编码，调用 predict_velocity_teacher，
        然后恢复原始条件。

        Args:
            zt_feats: (N, C) 加噪特征（detached）
            t_val: 时间步
            cond_source: 从 state 哪个子容器读取 image_condition。
                         "edited"    → state.views_edited.image_condition
                         "generated" → state.views_generated.image_condition

        Returns:
            x0: (N, C) detached
        """
        if cond_source == "edited":
            cond_emb = state.views_edited.image_condition
        elif cond_source == "generated":
            cond_emb = state.views_generated.image_condition
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}, expected 'edited' or 'generated'")

        # teacher_cfg=True  → uncond=zeros，走 CFG
        # teacher_cfg=False → uncond=None，跳过 CFG
        teacher_cfg = system.cfg.train.loss.contrastive.teacher_cfg
        uncond_emb = torch.zeros_like(cond_emb) if teacher_cfg else None
        v_teacher = self.predict_velocity_teacher(
            state, system, zt_feats, t_val,
            cond=cond_emb,
            uncond=uncond_emb,
        )
        x0 = zt_feats - t_val * v_teacher  # (N, C), detached
        return x0

    @staticmethod
    def contrastive_loss(
        z0_stu: torch.Tensor,
        z0_tea_tgt: torch.Tensor,
        z0_tea_src: torch.Tensor,
        ada: bool = True,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        P5c: 对比 loss — latent 空间。

        让 student 预测靠近 teacher_tgt（正样本），远离 teacher_src（负样本）。

        Args:
            z0_stu:     (N, C) 有梯度（通过 v_proxy）
            z0_tea_tgt: (N, C) detached, positive
            z0_tea_src: (N, C) detached, negative
            ada: 自适应归一化
            eps: ada epsilon

        Returns:
            scalar loss
        """
        return contrastive_loss_step(
            z0_stu.unsqueeze(0),       # (1, N, C)
            z0_tea_tgt.unsqueeze(0),   # (1, N, C)
            z0_tea_src.unsqueeze(0),   # (1, N, C)
            ada=ada,
            eps=eps,
        )


# =====================================================================
# Dense Ops — Stage 1 (sparse_structure_flow_model)
# =====================================================================

class TrellisDenseOps(TrellisFlowEditOps):
    """
    Dense (Stage 1) Ops — sparse_structure_flow_model。

    操作 dense Tensor (B, C, R, R, R) 而非 SparseTensor。
    无 normalization（Stage 1 不存在 ss_normalization）。
    不提供 decode_render — 复用 Sparse stage 的渲染结果。
    """

    # ═══════════════════════════════════════════════════════
    # 配置查询
    # ═══════════════════════════════════════════════════════

    def get_model(self, system):
        """返回 DDP 包装的 sparse_structure_flow_model。"""
        return system.pipeline.pipe.models['sparse_structure_flow_model']

    def get_stage_name(self) -> str:
        return "ss"

    def get_latent(self, state):
        return state.stage1.z0

    # ═══════════════════════════════════════════════════════
    # Rollout
    # ═══════════════════════════════════════════════════════

    def rollout(self, state, system, seed) -> None:
        """
        Dense rollout → clean z_s，no_grad。

        Side Effects:
            - state.stage1.z0: (B, C, R, R, R) dense latent
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device="cpu").manual_seed(seed)

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.dense_teacher_context()
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown rollout_mode: {rollout_mode!r}")

        with ctx, torch.no_grad():
            rollout_dense(
                state, cfg, system, device,
                generator=generator,
            )
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # Normalize / Denormalize — Stage 1 无 normalization
    # ═══════════════════════════════════════════════════════

    def normalize_latent(self, state, system) -> torch.Tensor:
        """Stage 1 无 normalization，直接返回 z0。"""
        return state.stage1.z0  # (B, C, R, R, R)

    def denormalize_feats(self, feats: torch.Tensor, system) -> torch.Tensor:
        """Stage 1 无 normalization，身份映射。"""
        return feats

    # ═══════════════════════════════════════════════════════
    # 加噪 / 采样时间步
    # ═══════════════════════════════════════════════════════

    def sample_timestep(self, system) -> float:
        """
        从 Stage 1 的 scheduler 时间步序列中随机采样。
        """
        pipeline = system.pipeline
        ss_steps, _, ss_rescale_t, _, _ = pipeline.dense.get_runtime_params()

        t_seq, _ = pipeline.dense.scheduler(ss_steps, ss_rescale_t)
        # 去掉最后一个（接近 0）
        t_seq = t_seq[:-1]

        cfg = system.cfg
        t_min = float(cfg.train.noise.t_min)
        t_max = float(cfg.train.noise.t_max)
        valid = [t for t in t_seq if t_min <= t <= t_max]
        if not valid:
            valid = list(t_seq)

        idx = torch.randint(0, len(valid), (1,)).item()
        return float(valid[idx])

    # ═══════════════════════════════════════════════════════
    # Velocity 预测 — dense Tensor
    # ═══════════════════════════════════════════════════════

    def _predict_velocity_impl(
        self, state, system, zt: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dense velocity 预测 with CFG。

        Args:
            zt: (B, C, R, R, R)
            t_val: 时间步 ∈ [0, 1]

        Returns:
            velocity: (B, C, R, R, R)
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        _, ss_guidance, _, ss_cfg_min, ss_cfg_max = pipeline.dense.get_runtime_params()

        cond_emb, uncond_emb = prepare_embeddings(state, device, cond, uncond)

        return predict_dense_velocity_with_cfg(
            pipeline, zt, t_val, cond_emb, uncond_emb,
            ss_guidance, ss_cfg_min, ss_cfg_max, device,
        )  # (B, C, R, R, R)

    def predict_velocity_student(self, state, system, zt: torch.Tensor, t_val: float) -> torch.Tensor:
        return self._predict_velocity_impl(state, system, zt, t_val)

    def predict_velocity_teacher(
        self, state, system, zt: torch.Tensor, t_val: float,
        cond: Optional[torch.Tensor] = None,
        uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with system.strategy.dense_teacher_context(), torch.no_grad():
            v = self._predict_velocity_impl(state, system, zt, t_val, cond=cond, uncond=uncond)
        return v.detach()

    # ═══════════════════════════════════════════════════════
    # 不提供 decode / render
    # ═══════════════════════════════════════════════════════

    def decode_render_dict(self, state, system):
        raise NotImplementedError("TrellisDenseOps 不提供 decode_render，应复用 Sparse stage 渲染结果。")


# =====================================================================
# Dense Contrastive Ops — 在 Dense latent 空间做对比学习
# =====================================================================

class TrellisDenseContrastiveOps(TrellisDenseOps):
    """
    Dense Contrastive Ops：在 Dense (B,C,R,R,R) latent 空间做对比学习。

    复用 Sparse stage 的 c_src / c_tgt（DINOv2 条件编码），
    不自行渲染/编辑/编码。

    方法来源：
      rollout / normalize / add_noise / sample_timestep / velocity → TrellisDenseOps
      predict_x0_teacher_with_cond → 本类（dense 版本）
      contrastive_loss → TrellisContrastiveOps（复用）
    """

    def predict_x0_teacher_with_cond(
        self,
        state,
        system,
        zt: torch.Tensor,
        t_val: float,
        cond_source: str = "edited",
    ) -> torch.Tensor:
        """
        Teacher 单步去噪 with condition → ẑ₀ (dense)。

        Args:
            zt: (B, C, R, R, R) 加噪 latent
            t_val: 时间步
            cond_source: "edited" | "generated"

        Returns:
            x0: (B, C, R, R, R) detached
        """
        if cond_source == "edited":
            cond_emb = state.views_edited.image_condition
        elif cond_source == "generated":
            cond_emb = state.views_generated.image_condition
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}")

        # teacher_cfg=True  → uncond=zeros，走 CFG
        # teacher_cfg=False → uncond=None，跳过 CFG
        teacher_cfg = system.cfg.train.loss.contrastive.teacher_cfg
        uncond_emb = torch.zeros_like(cond_emb) if teacher_cfg else None
        v_teacher = self.predict_velocity_teacher(
            state, system, zt, t_val,
            cond=cond_emb,
            uncond=uncond_emb,
        )
        return zt - t_val * v_teacher  # (B, C, R, R, R), detached

    @staticmethod
    def contrastive_loss(
        z0_stu: torch.Tensor,
        z0_tea_tgt: torch.Tensor,
        z0_tea_src: torch.Tensor,
        ada: bool = True,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Dense 对比 loss — 在 (B, C, R, R, R) 空间操作。

        将 5D tensor 展平为 (B, C*R*R*R) 后调用 contrastive_loss_step。
        """
        B = z0_stu.shape[0]
        return contrastive_loss_step(
            z0_stu.reshape(B, 1, -1),       # (B, 1, C*R³)
            z0_tea_tgt.reshape(B, 1, -1),   # (B, 1, C*R³)
            z0_tea_src.reshape(B, 1, -1),   # (B, 1, C*R³)
            ada=ada,
            eps=eps,
        )
