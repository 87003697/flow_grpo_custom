"""
Trellis2StageOps — Trellis2 特有的 StageOps 中间基类。

在模型无关的 StageOps ABC 与 Trellis2 具体实现（Trellis2ShapeOps / Trellis2TexOps）之间
插入一层公共逻辑，避免代码重复，同时不污染模型无关的 StageOps。

继承层次：
    StageOps (ABC, 模型无关)
    └── Trellis2StageOps (本文件, Trellis2 公共逻辑)
        ├── Trellis2ShapeOps
        └── Trellis2TexOps

提供的公共方法：
  - vjp_loop:            通用 VJP 循环（shape/tex 完全相同的逻辑）
  - get_flow_resolution:  获取 flow_resolution
  - normalize_slat:       SparseTensor 归一化
  - denormalize_slat:     SparseTensor 反归一化
  - get_sigma_min:        获取 sigma_min
  - pretrained_rollout:   teacher rollout（onestep 训练用）
  - add_noise:            flow matching 加噪
  - sample_timestep:      从 scheduler 时间步中采样
  - predict_cfg_velocity: CFG velocity 预测（student，有 autograd 图）
  - predict_cfg_velocity_teacher: CFG velocity 预测（teacher，detached）

使用方式：
    子类只需实现 StageOps 的 abstract 方法 + 填充 stage-specific 参数，
    vjp_loop / onestep 相关方法从本基类继承即可。
"""

from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from edit4shape.systems.utils.stage_ops import StageOps
from edit4shape.generators.trellis2.rollout.base import (
    _predict_velocity,
    trellis2_cfg_sparse,
)

if TYPE_CHECKING:
    from edit4shape.generators.trellis2.rollout import RolloutTracker, VelocityTracker
    from trellis2.modules.sparse import SparseTensor


class Trellis2StageOps(StageOps):
    """
    Trellis2 公共 StageOps — 提供 vjp_loop 和工具方法的默认实现。

    子类（Trellis2ShapeOps / Trellis2TexOps）只需覆写：
      - StageOps 的抽象方法（get_model, get_stage_name, rollout, decode_render 等）
    无需再重复 vjp_loop / normalize / denormalize 等与具体渲染无关的逻辑。
    """

    # ═══════════════════════════════════════════════════════
    # 工具方法 — 依赖 Trellis2 pipeline 接口
    # ═══════════════════════════════════════════════════════

    def get_flow_resolution(self, system) -> int:
        """返回当前阶段的 flow_resolution。"""
        stage_name = self.get_stage_name()
        return system.pipeline.get_stage_config(stage_name)["flow_resolution"]

    def normalize_slat(self, slat, system):
        """SparseTensor → 归一化域。"""
        return system.pipeline.normalize(slat, self.get_stage_name())

    def denormalize_slat(self, slat, system):
        """SparseTensor → 反归一化域。"""
        return system.pipeline.denormalize(slat, self.get_stage_name())

    def get_sigma_min(self, system) -> float:
        """获取当前阶段的 sigma_min。"""
        stage_name = self.get_stage_name()
        sampler_attr = f"{stage_name}_slat_sampler"
        return getattr(system.pipeline.pipe, sampler_attr).sigma_min

    def get_rollout_mode(self, system) -> str:
        """获取 Rollout 模式: "pretrained" (off-policy) | "student" (on-policy)。"""
        return str(system.cfg[self.get_stage_name()].train.rollout_mode)

    def get_student_denoise_cfg(self, system) -> bool:
        """获取 Student Denoise CFG 开关: True = 使用 CFG, False = 跳过 uncond forward。"""
        return bool(system.cfg[self.get_stage_name()].train.student_denoise_cfg)

    # ═══════════════════════════════════════════════════════
    # VJP Loop — 通用实现（shape/tex 逻辑完全相同）
    # ═══════════════════════════════════════════════════════

    def vjp_loop(self, state, system, tracker: RolloutTracker) -> Dict[str, Any]:
        """
        Phase 3: 通用 VJP 循环 — 逐步重算 f_θ，用 cond_proxy.grad 做 VJP → θ.grad 累积。
        显存 O(1)，不随步数增长。

        ★ 子类无需覆写：shape/tex 的 VJP 逻辑完全相同，
        差异（stage_name, slat, shape_cond）通过 self 的查询方法自动获取。

        DDP 安全：
          整个 VJP 循环在 model.no_sync() 下执行，backward 只做本地 θ.grad 累积，
          不触发 DDP all-reduce。梯度同步由 entry 层的 sync_grads_and_step() 负责。

        流程（每步 t）:
          1. t_val, x_t, v_grad ← tracker 直接读取
          2. cond_pred = f_θ(x_t, t, cond, shape_cond) — 唯一需要重算的，有 θ 梯度
          3. (v_grad * cond_pred).sum().backward() — VJP，图立即释放

        Returns:
            日志字典（clip + collect_log 合并）
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        stage_name = self.get_stage_name()
        flow_res = self.get_flow_resolution(system)
        reg_weight = self.get_reg_weight(system)
        model = self.get_model(system)

        # ---- 日志收集（在 VJP 前，此时 .grad 已由 P2c backward 填充） ----
        log = tracker.clip_guidance_grads(self.get_guidance_grad_max_norm(system))
        log.update(tracker.collect_log(reg_weight=reg_weight))

        # ---- 条件编码（只需 cond，不需要 uncond） ----
        cond_emb, _ = state.extract_embeddings(resolution=flow_res)
        cond_emb = cond_emb.to(device)  # (B, S, C)

        # ---- shape 条件（tex 需要，shape 返回 None） ----
        shape_cond = self.get_shape_cond(state)

        T = len(tracker.input_trajectory)
        B = cond_emb.shape[0]  # ()

        # ★ no_sync：VJP backward 只做本地累积，不触发 DDP all-reduce。
        #   防止某个 rank OOM 跳过 VJP 时，其他 rank 的 DDP hooks 死等。
        with model.no_sync():
            for i in range(T):
                # 1. 从 tracker 直接读取：t, x_t, v_grad
                t_val = tracker.timesteps[i]  # float64 精度
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)

                x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
                slat = self.get_latent(state)
                x_t = slat.replace(x_t_feats)  # SparseTensor（无梯度）

                # 2. 重算 cond_pred = f_θ(x_t, t, cond, shape_cond)（仅对 θ 有梯度，x_t detached）
                cond_pred = _predict_velocity(
                    pipeline, x_t, t_batch, cond_emb,
                    stage_name, flow_res, shape_cond
                )  # SparseTensor

                # 3. VJP: (v_grad)^T · ∂f_θ/∂θ → θ.grad +=
                v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
                if v_grad is not None:
                    (v_grad * cond_pred.feats).sum().backward()  # 图立即释放，显存 O(1)

        # ---- 释放 tracker 数据 ----
        del tracker.input_trajectory[:], tracker.output_trajectory[:]
        del tracker.timesteps[:]
        torch.cuda.empty_cache()

        return log

    # ═══════════════════════════════════════════════════════
    # Onestep 训练 — 公共方法
    # ═══════════════════════════════════════════════════════

    def pretrained_rollout(self, state, system, seed) -> None:
        """
        完整 rollout → clean z₀，no_grad。

        根据 cfg.{stage}.train.rollout_mode 选择使用哪个模型：
          - "pretrained"：teacher_context()，使用 pretrained 权重（off-policy）
          - "student"：直接使用当前 finetuned 权重（on-policy）

        完全 no_grad，不需要任何 proxy chain。

        Side Effects:
            - state.shape.z0 / state.tex.z0: 挂载 rollout 输出的 SparseTensor（反归一化后）
        """
        stage_name = self.get_stage_name()
        flow_res = self.get_flow_resolution(system)

        rollout_mode = self.get_rollout_mode(system)
        if rollout_mode == "pretrained":
            ctx = system.strategy.teacher_context(stage_name, flow_res)
        elif rollout_mode == "student":
            ctx = contextlib.nullcontext()
        else:
            raise ValueError(
                f"Unknown rollout_mode: {rollout_mode!r}, expected 'pretrained' or 'student'"
            )

        with ctx, torch.no_grad():
            self._pretrained_rollout_impl(state, system, seed)
        torch.cuda.empty_cache()

    @abstractmethod
    def _pretrained_rollout_impl(self, state, system, seed) -> None:
        """
        子类实现具体的 rollout 逻辑。

        在 no_grad 上下文内调用。外层 pretrained_rollout() 已根据
        rollout_mode 选择了 teacher_context 或 nullcontext。
        Shape: 调用 rollout_shape；Tex: 调用 rollout_tex。
        """
        ...

    def add_noise(self, z0_feats: torch.Tensor, t: float) -> torch.Tensor:
        """
        Flow matching 加噪：zₜ = (1-t) * z₀ + t * ε

        Args:
            z0_feats: (N, C) 归一化域的 clean features
            t: 标量时间步 ∈ (0, 1)

        Returns:
            zt_feats: (N, C) 加噪后的特征
        """
        noise = torch.randn_like(z0_feats)  # (N, C)
        zt = (1.0 - t) * z0_feats + t * noise  # (N, C)
        return zt

    def sample_timestep(self, system) -> float:
        """
        从 inference scheduler 的时间步序列中随机采样一个时间步。

        使用 scheduler 的实际时间步（经过 rescale_t），
        确保对齐 inference 分布。

        Returns:
            t_val: 采样到的时间步值（范围 [0, 1]）
        """
        stage_name = self.get_stage_name()
        pipeline = system.pipeline
        device = system.accelerator.device

        sampler_params = pipeline.get_sampler_params(stage_name)
        steps = int(sampler_params["steps"])

        scheduler = pipeline.scheduler(stage_name)
        scheduler.set_timesteps(steps, device=device)
        timesteps = scheduler.timesteps  # Tensor, 从大到小排列, [0, 1] 范围

        # 去掉最后一个（通常接近 0）
        timesteps = timesteps[:-1]

        # 限制采样范围（cfg.{stage}.train.noise.t_min / t_max）
        stage_cfg = system.cfg[stage_name]
        t_min = float(stage_cfg.train.noise.t_min)
        t_max = float(stage_cfg.train.noise.t_max)
        mask = (timesteps >= t_min) & (timesteps <= t_max)
        valid_timesteps = timesteps[mask]
        if len(valid_timesteps) == 0:
            valid_timesteps = timesteps  # fallback

        # 随机选一个
        idx = torch.randint(0, len(valid_timesteps), (1,)).item()
        return float(valid_timesteps[idx].item())

    def predict_cfg_velocity(
        self, state, system, zt_feats: torch.Tensor, t_val: float
    ) -> torch.Tensor:
        """
        Student CFG velocity 预测（有 autograd 图到 θ）。

        执行 cond + uncond 预测，CFG 混合后返回 velocity feats。

        Args:
            state: Trellis2State
            system: Trellis2System
            zt_feats: (N, C) 归一化域特征
            t_val: 时间步 [0, 1]

        Returns:
            v_feats: (N, C) CFG velocity，有 autograd 图
        """
        return self._predict_cfg_velocity_impl(state, system, zt_feats, t_val)

    def predict_cfg_velocity_teacher(
        self, state, system, zt_feats: torch.Tensor, t_val: float
    ) -> torch.Tensor:
        """
        Teacher CFG velocity 预测（teacher_context + no_grad，detached）。

        Returns:
            v_feats: (N, C) CFG velocity，detached
        """
        stage_name = self.get_stage_name()
        flow_res = self.get_flow_resolution(system)
        with system.strategy.teacher_context(stage_name, flow_res), torch.no_grad():
            v = self._predict_cfg_velocity_impl(state, system, zt_feats, t_val)
        return v.detach()  # (N, C)

    def _predict_cfg_velocity_impl(
        self, state, system, zt_feats: torch.Tensor, t_val: float
    ) -> torch.Tensor:
        """
        内部共享：cond + uncond 预测 + CFG 混合 → v_feats。

        调用方决定是否在 teacher_context / no_grad 下调用。

        Args:
            state: Trellis2State
            system: Trellis2System
            zt_feats: (N, C) 归一化域特征
            t_val: 时间步 [0, 1]

        Returns:
            v_feats: (N, C) CFG velocity
        """
        pipeline = system.pipeline
        device = system.accelerator.device
        stage_name = self.get_stage_name()
        flow_res = self.get_flow_resolution(system)

        # 获取 CFG 参数
        sampler_params = pipeline.get_sampler_params(stage_name)
        cfg_strength = float(sampler_params["guidance_strength"])
        cfg_rescale = float(sampler_params["guidance_rescale"])
        cfg_min, cfg_max = pipeline.get_cfg_interval(stage_name)
        sigma_min = self.get_sigma_min(system)

        # 条件编码
        cond_emb, uncond_emb = state.extract_embeddings(resolution=flow_res)
        cond_emb = cond_emb.to(device)  # (B, S, C)
        uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)

        # shape 条件
        shape_cond = self.get_shape_cond(state)

        # 构建 SparseTensor 输入
        slat = self.get_latent(state)
        x_t = slat.replace(zt_feats)  # SparseTensor with zt_feats

        B = cond_emb.shape[0]  # ()
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.float32)  # (B,)

        # cond 预测（step-level gradient checkpoint：释放 flow model 中间激活，backward 时重算）
        # ★ 对齐 VJP rollout 的 checkpoint 策略（rollout_shape.py L130），
        #   避免 v_student 计算图保留全部中间激活（~10-20 GiB）。
        #   teacher 调用方已在 no_grad 下，checkpoint 自动退化为普通 forward。
        cond_pred_feats = ckpt(
            lambda *a: _predict_velocity(*a).feats,
            pipeline, x_t, t_batch, cond_emb,
            stage_name, flow_res, shape_cond,
            use_reentrant=False,
        )  # (N, C), 有 autograd 图到 θ（中间激活已释放）
        cond_pred = x_t.replace(cond_pred_feats)  # SparseTensor

        # CFG 混合
        use_cfg = cfg_min <= t_val <= cfg_max
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, x_t, t_batch, uncond_emb,
                    stage_name, flow_res, shape_cond
                )  # SparseTensor
            velocity = trellis2_cfg_sparse(
                cond_pred, uncond_pred, cfg_strength,
                guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                sigma_min=sigma_min,
            )  # SparseTensor
        else:
            velocity = cond_pred  # SparseTensor

        return velocity.feats  # (N, C)
