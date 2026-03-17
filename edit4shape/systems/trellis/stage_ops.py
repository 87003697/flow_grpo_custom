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

import torch

# ABC 从 utils 导入（模型无关的抽象层）
from edit4shape.systems.utils.stage_ops import StageOps  # noqa: F401 — re-export

# Phase 函数 & 渲染
from edit4shape.systems.trellis.forward import decode_and_render_mesh, decode_and_render_gs
from edit4shape.systems.trellis.phases import dense_sampling_no_grad, phase3_rollout_grad_backward
from edit4shape.generators.trellis.rollout import rollout_sparse, RolloutTracker
from edit4shape.generators.trellis.rollout.base import predict_velocity_with_cfg


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

    def get_seed_offset(self) -> int:
        return 0

    def get_reg_weight(self, system) -> float:
        return system.cfg.train.loss.reg

    def get_guidance_weight(self, system) -> float:
        return system.cfg.train.loss.guidance

    def get_guidance_cfg(self, system):
        return system.cfg.train.guidance

    def get_gs_reg_config(self, system) -> Dict[str, float]:
        """
        返回 GS 表示正则化权重（reg_vol / reg_opacity）。

        从 cfg.train.loss.gs_reg 读取；未配置时返回 0（不启用）。
        """
        cfg = system.cfg
        gs_reg = cfg.train.loss.get("gs_reg", {})
        return {
            "lambda_vol": float(gs_reg.get("vol", 0.0)),
            "lambda_opacity": float(gs_reg.get("opacity", 0.0)),
        }

    # ═══════════════════════════════════════════════════════
    # Async 友好查询
    # ═══════════════════════════════════════════════════════

    def get_slat(self, state):
        return state.features.slat

    # get_shape_cond → 继承默认 None（单模型无 shape cond）

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        """
        根据 cfg.renderer.type 分发到 mesh 或 gs 渲染。

        子类可覆写此方法以实现自定义渲染策略
        （如 TrellisMeshOps / TrellisGsOps / 自定义混合渲染）。
        """
        latents = state.features.slat
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
        """Phase 0: Dense Sampling → 填充 state.coords。"""
        dense_sampling_no_grad(state, system, system.accelerator.device)

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
            state.features.slat, state.cameras,
            system.pipeline, system.renderers["mesh"], system.accelerator.device,
        )
        render_out["color"] = render_out["normal"]
        return render_out


class TrellisGsOps(TrellisOps):
    """强制使用 GS Color 渲染，不受 cfg.renderer.type 控制。"""

    def decode_render_dict(self, state, system) -> Dict[str, Any]:
        return decode_and_render_gs(
            state.features.slat, state.cameras,
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
        latents = state.features.slat
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
          - "pretrained"：teacher_context()，使用 pretrained 权重（off-policy）
          - "student"：直接使用当前 finetuned 权重（on-policy）

        完全 no_grad，不需要任何 proxy chain。

        Side Effects:
            - state.features.slat: 挂载 rollout 输出的 SparseTensor（反归一化后）
        """
        device = system.accelerator.device
        cfg = system.cfg
        generator = torch.Generator(device=device).manual_seed(seed)

        rollout_mode = str(cfg.train.rollout_mode)
        if rollout_mode == "pretrained":
            ctx = system.strategy.teacher_context()
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
        # rollout_sparse 已经做了反归一化并挂载到 state.features.slat
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════
    # 加噪：z₀ → zₜ（归一化域操作）
    # ═══════════════════════════════════════════════════════

    def normalize_slat(self, state, system) -> torch.Tensor:
        """
        将 state.features.slat.feats 从反归一化域 → 归一化域。

        rollout_sparse 输出的 slat 已经反归一化（denorm_feats = feats * std + mean），
        加噪/去噪需要在归一化域进行（与训练时一致）。

        Returns:
            normalized_feats: (N, C) 归一化后的特征
        """
        norm = system.pipeline.pipe.slat_normalization
        device = system.accelerator.device
        std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
        mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
        denorm_feats = state.features.slat.feats  # (N, C)
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
        _, _, slat_steps, _, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()

        scheduler = pipeline.scheduler()
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

    def _predict_velocity_impl(self, state, system, zt_feats: torch.Tensor, t_val: float) -> torch.Tensor:
        """
        内部共享：构建 SparseTensor 输入 + predict_velocity_with_cfg → v_feats。

        调用方决定是否在 teacher_context / no_grad 下调用。

        Args:
            state: TrellisState（需要 state.features.slat 提供 coords）
            system: TrellisSystem
            zt_feats: (N, C) 归一化域特征
            t_val: 时间步值（[0, 1000] 范围）

        Returns:
            v_feats: (N, C) 预测的速度特征
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        _, _, _, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()
        cfg_min, cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]

        # 构建 SparseTensor 输入
        slat = state.features.slat
        x_t = slat.replace(zt_feats)  # SparseTensor with zt_feats, 共享 coords

        # 条件编码
        cond_emb, uncond_emb = state.extract_embeddings()
        cond_emb = cond_emb.to(device)
        uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None

        velocity = predict_velocity_with_cfg(
            pipeline, x_t, t_val, cond_emb, uncond_emb,
            slat_guidance, cfg_min, cfg_max, device,
        )  # SparseTensor
        return velocity.feats  # (N, C)

    def predict_velocity_student(self, state, system, zt_feats: torch.Tensor, t_val: float) -> torch.Tensor:
        """
        Finetuned (student) model 速度预测（有 autograd 图到 θ）。

        Returns:
            v_feats: (N, C) 有 autograd 图
        """
        return self._predict_velocity_impl(state, system, zt_feats, t_val)

    def predict_velocity_teacher(self, state, system, zt_feats: torch.Tensor, t_val: float) -> torch.Tensor:
        """
        Pretrained (teacher) model 速度预测（teacher_context + no_grad，detached）。

        Returns:
            v_feats: (N, C) detached
        """
        with system.strategy.teacher_context(), torch.no_grad():
            v = self._predict_velocity_impl(state, system, zt_feats, t_val)
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
