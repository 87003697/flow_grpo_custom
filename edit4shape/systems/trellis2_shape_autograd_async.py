"""
Trellis2 Shape 训练系统 — 五阶段 Autograd + 异步 Guidance 流水线版本。

基于 trellis2_shape_autograd.py 的三阶段架构，采用「前向无梯度 + 后向重计算」策略：
- forward_no_grad(): P1-no-grad + P2-no-grad + submit → 立即释放 decode cache
- backward_with_grad(): P2-wait + P2-grad (重跑 decode+render) + P1-grad (flow VJP)

五阶段流水线（双缓冲 prev/curr）：
  curr: dense → P1-ng → P2-ng (no_grad decode+render) → submit_async → 释放 decode cache
  prev: P2-wait → P2-grad (重跑 decode+render, 带梯度) → backward → P1-grad (flow VJP)

★ 显存优势：
  原始方案：decode cache（meshes/subs/autograd图）从 P2a 持续存活到 P3 结束。
  新方案：  P2-no-grad 后 decode cache 立即释放，P2-grad 重计算时 GPU 上无 curr 数据。
  结果：显存峰值从「curr decode cache + prev P3 重算」降低到「单次 decode+render 峰值」。

两层 proxy：
  1. cond_pred proxy (P1) — 显存隔离，不保留 flow model 计算图
  2. comp_rgb  proxy (异步) — 计算并行，train/guidance GPU 各自独立 backward

正确性保证：
  Decoder (LayerNorm + SiLU, 无 Dropout/BatchNorm) 和 Renderer (纯数学运算)
  在 no_grad 和 grad 模式下行为完全一致，重跑 decode+render 得到相同 comp_rgb。

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ P2-grad guidance only  ↑ P1-ng autograd.grad        ↑ P1-grad 合并 VJP

特性：
- accum≥2 时收益最大：N-1 个 MB 的 guidance 与下一个 MB 的前向并行
- accum=1 时退化为同步版（无并行窗口，但正确性不变）
- 评估路径仍使用单阶段 forward（trellis2_shape_forward）
- OOM 安全：P2-no-grad/P2-grad OOM 均可降级到 P1-grad reg-only

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import argparse
import csv
import json
import logging
import os
import random
import sys
import importlib.util
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, List, Literal

# =====================================================================
# 第三方库导入
# =====================================================================
from PIL import Image
import numpy as np
import requests
import yaml
import ml_collections
from absl import app
from ml_collections import config_flags

import torch
from accelerate import Accelerator
from torch.utils.data import DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点，节省显存
from tqdm import tqdm

# =====================================================================
# 项目内部导入
# =====================================================================


# =====================================================================
# TRELLIS.2 参考实现路径设置（必须在 trellis2.* 导入之前）
# =====================================================================
repo_root = os.path.abspath(os.getcwd())
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)

# SparseTensor: TRELLIS.2 中用于表示稀疏 3D 特征的核心数据结构
from trellis2.modules.sparse import SparseTensor
# Chunked Forward 支持（自定义实现，已从 _reference_codes 迁移）
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin

# =====================================================================
# Guidance 模块
# =====================================================================
from functools import partial
from edit4shape.guidance import create_guidance
from edit4shape.guidance.pipeline_parallel import AsyncGuidanceResult

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    BaseState,
    build_run_paths,
)
from edit4shape.generators.trellis2.training_adpter import Trellis2CheckpointIO
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, AsyncPhaseProfiler
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import rollout_shape, RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity

# =====================================================================
# 从 trellis2_shape.py 导入共享组件
# _CONFIG 不再从此处 import，各入口点自行在 if __name__ == "__main__" 中定义
# =====================================================================
from edit4shape.systems.trellis2_shape import (
    StageSystem,
    build_system,
    decode_and_render_normal,
    decode_and_render_normal_mesh,
    decode_and_render_normal_hybrid,
    trellis2_shape_forward,
    evaluate,
)

# =====================================================================
# Renderer 导入（使用 trellis2 的可微渲染器）
# =====================================================================
from trellis2.renderers import MeshRenderer
from trellis2.representations.mesh import Mesh

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape"]


# =====================================================================
# 从 training_adpter 导入 StageConfig
# =====================================================================
from edit4shape.generators.trellis2.training_adpter import StageConfig


# =====================================================================
# Trellis2 系统组件类
# =====================================================================

@dataclass
class Trellis2System:
    """
    Trellis2 Shape 训练系统。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    
    渲染器配置（使用 trellis2 的 nvdiffrast 可微渲染器）：
    - shape.renderer: MeshRenderer (直接渲染 normal，支持梯度)
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.shape.model      # Shape Flow Model
        system.shape.renderer   # MeshRenderer (Normal)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # Shape 阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    
    # 共享组件
    guidance: Any = None
    
    # 训练策略（LoRA / Full / Frozen）
    strategy: Any = None
    
    # ★ 三阶段 Autograd：将 cfg 和 accelerator 挂在 system 上，
    #   Phase 函数只需 (state, system) 即可访问所有训练配置和组件
    cfg: Any = None                  # ml_collections.ConfigDict
    accelerator: Accelerator = None  # Accelerate 加速器
    
    @staticmethod
    def setup_env_and_seed(cfg: Any) -> None:
        """设置随机种子与确定性运行环境。"""
        import random
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def prepare_lora(self, cfg: Any, adapter: str = "base", **kwargs) -> "Trellis2System":
        """准备 LoRA 适配器"""
        for module in [self.pipeline, self.guidance]:
            if module is not None and hasattr(module, "set_adapter"):
                module.set_adapter(adapter)
        return self
    
    def prepare_optimizers(self, accelerator: Accelerator) -> "Trellis2System":
        """
        通过 strategy.prepare() 统一做 DDP 包裹 + 回写 pipeline。
        
        与 V1 System.prepare_models_and_optimizers() 对齐：
        模型和优化器一起 prepare → DDP 包裹 + 注册到 accelerator，
        使 save_state/load_state 自动管理模型权重。
        """
        if self.strategy is not None and self.shape.optimizer is not None:
            shape_config = self.shape.config
            self.shape.model, self.shape.optimizer = self.strategy.prepare(
                accelerator, shape_config.model_stage, shape_config.flow_resolution, self.shape.optimizer
            )
        return self


# =====================================================================
# 三阶段 Autograd — Phase 函数
# =====================================================================

def dense_sampling_no_grad(
    state: Trellis2State,
    system: Trellis2System,
) -> None:
    """
    Dense Sampling（no_grad）。填充 state.coords。
    
    从现有 trellis2_shape_forward 的 Dense Sampling 段提取。
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    ss_params = pipeline.get_ss_params()
    
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_512_embed,       # 始终用 512
            "neg_cond": state.views_conditioned.uncond_512_embed,  # 始终用 512
        }
        coords = pipeline.dense_sampling(
            cond_dict,
            steps=int(ss_params["steps"]),
            resolution=stage_config["ss_resolution"],
        )  # (N, 4)
    state.coords = coords


def shape_phase1_rollout(
    state: Trellis2State,
    system: Trellis2System,
    gen_seed: int,
) -> RolloutTracker:
    """
    Shape Phase 1: 无梯度 rollout + 记录 proxy 轨迹。
    
    - 创建 RolloutTracker 并传入 rollout_shape()
    - rollout_shape 在每步记录 input/output proxy，用 proxy 推进 scheduler
    - 最终 state.features.shape_slat 含 proxy chain（不含模型计算图）
    
    Args:
        state: 已填充 coords 和条件编码的状态
        system: 训练系统
        gen_seed: rollout 随机种子
    
    Returns:
        RolloutTracker: 已填充 input_trajectory / output_trajectory
    """
    cfg = system.cfg
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    device = system.accelerator.device
    
    tracker = RolloutTracker()
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    
    # ⚠️ 不可包裹 torch.no_grad()：rollout_shape 内部 is_training=False
    #    已用 no_grad 做模型推理，但 tracker 的 proxy 需要 autograd 图
    #    （scheduler 用 proxy 推进 → slat 依赖 proxy chain）
    rollout_shape(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=gen,
        is_training=False,   # 模型推理 no_grad
        tracker=tracker,     # ★ 记录 proxy 轨迹
    )
    # state.features.shape_slat: SparseTensor（有 proxy chain，不含模型图）
    
    return tracker


# =====================================================================
# PendingMicroBatch — 流水线双缓冲数据容器
# =====================================================================

@dataclass
class PendingMicroBatch:
    """
    记录一个已提交 guidance 但尚未 backward 的 micro-batch 的全部上下文。

    五阶段生命周期:
        1. forward_no_grad():
           P1-no-grad (rollout) + P2-no-grad (decode+render) + submit
           → decode cache 立即释放，GPU 上仅保留 slat proxy chain + tracker
        2. backward_with_grad():
           P2-wait → P2-grad (重跑 decode+render 带梯度) → backward
           → P1-grad (flow VJP) → 释放 tracker
        3. 释放（del pending）

    ★ 关键：comp_rgb 不存储在 pending 中。
      forward_no_grad 中的 comp_rgb 仅用于 submit，之后立即丢弃。
      backward_with_grad 中重新计算 comp_rgb（带 autograd 图）后 backward。
    """
    state: Trellis2State                  # 该 MB 的 Trellis2State（含 slat proxy chain / cameras 等）
    tracker: RolloutTracker               # Phase 1 记录的 proxy 轨迹
    image_pils: List[Image.Image]         # 条件图像（submit 时用）
    submitted: bool = False               # 是否已成功 submit 给 guidance GPU


# =====================================================================
# 异步 Phase 2: submit + wait_guidance
# =====================================================================

def phase2_submit_guidance_async(
    comp_rgb: torch.Tensor,
    image_pils: List[Image.Image],
    system: Trellis2System,
) -> None:
    """
    Phase 2 前半：将 comp_rgb 异步提交给 guidance GPU。

    调用 PipelineParallelMixin.submit_async()：
      comp_rgb → detach → copy to guidance GPU → 独立 backward → 入队（FIFO）
    本函数立即返回，不阻塞 train GPU。

    ★ comp_rgb 可以来自 no_grad 前向（仅需值用于提交，submit_async 内部会 detach）。

    Args:
        comp_rgb: (B, V, H, W, 3) Normal 渲染图（不要求有 autograd 图）
        image_pils: 条件图像列表
        system: 训练系统（访问 guidance + cfg）
    """
    guidance_weight = system.cfg.train.loss.guidance
    system.guidance.submit_async(
        comp_rgb,
        image_pils,
        guidance_weight=guidance_weight,
        rank=system.accelerator.process_index,
    )


def phase2_wait_guidance(
    state: Trellis2State,
    system: Trellis2System,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Phase 2 中段：阻塞等待 guidance GPU 结果，返回 rgb_grad 和日志。

    ★ 不做 backward — backward 由 backward_with_grad() 在重跑 decode+render 后执行。
    
    挂载可视化数据到 state（与同步版 attach_guidance_result 对齐）。

    Args:
        state: 当前 MB 的 Trellis2State（挂载 edited 可视化数据）
        system: 训练系统

    Returns:
        rgb_grad: (B, V, H, W, 3) guidance 反传的梯度（detached）
        guidance_log: 日志字典（loss/guidance + 细分 loss + _guid_timing）
    """
    device = system.accelerator.device

    # 1. 阻塞等待 guidance 结果
    async_result: AsyncGuidanceResult = system.guidance.wait_and_get(
        target_device=device,
    )

    # 2. 提取 rgb_grad（后续用于 comp_rgb.backward）
    rgb_grad = async_result.rgb_grad.detach()  # (B, V, H, W, 3)

    # 3. 挂载可视化数据到 state
    state.views_edited.image_tensor = async_result.edited_imgs   # (B,V,C,H,W) or None
    state.views_edited.trackers = async_result.trackers           # List[StateTracker] or None

    # 4. 构建日志（reg 日志由 P1-grad 从 tracker.reg_loss_val 提供）
    guidance_log: Dict[str, Any] = {}
    if async_result.loss_dict:
        guidance_log.update({f"loss/{k}": v for k, v in async_result.loss_dict.items()})
    guidance_log["loss/guidance"] = async_result.loss_scalar
    
    # 4b. 保存 guidance GPU 计时供 profiler 分析
    guidance_log["_guid_timing"] = (async_result.guid_wall_start,
                                    async_result.guid_wall_end)

    del async_result
    return rgb_grad, guidance_log


def phase3_rollout_grad_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
) -> Dict[str, Any]:
    """
    Phase 3: 纯 VJP — 逐步重算 f_θ，合并 guidance + reg 梯度做 VJP → θ.grad 累积。
    显存 O(1)，不随步数增长。
    
    梯度来源：
    - guidance: output_trajectory[t].grad（Phase 2 backward 填充，含 CFG 因子）
    - reg:     tracker.reg_grads[t]（Phase 1 autograd.grad 提前计算）
    
    合并: v_grad = guidance_grad + reg_weight * reg_grad
    
    降级链路：
    - 正常: guidance + reg → 完整 VJP
    - Phase 2 OOM: guidance_grad=None → reg-only VJP
    - 极端: 两者皆无 → 跳过该步
    
    流程（每步 t）:
      1. t_val, x_t ← tracker 直接读取
      2. v_grad = guidance_grad + reg_weight * reg_grad（合并）
      3. cond_pred = f_θ(x_t, t) — 唯一需要重算的，有 θ 梯度
      4. (v_grad * cond_pred).sum().backward() — VJP，图立即释放
    
    Returns:
        日志字典（包含 loss/reg）
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_config = pipeline.get_stage_config("shape")
    flow_res = stage_config["flow_resolution"]
    reg_weight = system.cfg.train.loss.reg
    
    # ---- 条件编码（只需 cond，不需要 uncond） ----
    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    
    T = len(tracker.input_trajectory)
    assert len(tracker.reg_grads) == T, (
        f"reg_grads 长度 ({len(tracker.reg_grads)}) != 轨迹长度 ({T})，Phase 1 未正确计算 reg 梯度"
    )
    
    for i in range(T):
        # 1. 合并梯度：reg（Phase 1 预计算，必有）+ guidance（Phase 2，OOM 时为 None）
        reg_grad = tracker.reg_grads[i]  # (N, C)，Phase 1 保证非 None
        v_grad = reg_weight * reg_grad  # (N, C)
        guid_grad = tracker.output_trajectory[i].grad  # (N, C) or None（Phase 2 OOM）
        if guid_grad is not None:
            v_grad = v_grad + guid_grad  # (N, C)
        
        # 2. 从 tracker 读取 t, x_t
        t_val = tracker.timesteps[i]  # float64 精度
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.shape_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 3. 重算 cond_pred = f_θ(x_t, t)（仅对 θ 有梯度，x_t detached）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_val, cond_emb,
            "shape", flow_res, None
        )  # SparseTensor
        
        # 4. VJP: (v_grad)^T · ∂f_θ/∂θ → θ.grad +=
        (v_grad * cond_pred.feats).sum().backward()  # 图立即释放，显存 O(1)
    
    # ---- 释放 tracker 数据 ----
    del tracker.input_trajectory[:], tracker.output_trajectory[:]
    del tracker.timesteps[:], tracker.reg_grads[:]
    torch.cuda.empty_cache()
    
    # ---- 日志：reg loss 值来自 tracker ----
    phase3_log: Dict[str, Any] = {}
    if tracker.reg_loss_val is not None:
        phase3_log["loss/reg"] = tracker.reg_loss_val
    
    return phase3_log


# =====================================================================
# 五阶段流水线 — 前向（无梯度）+ 后向（有梯度）
# =====================================================================

def forward_no_grad(
    batch: Dict[str, Any],
    system: Trellis2System,
    global_step: int,
    profiler: AsyncPhaseProfiler = None,
) -> PendingMicroBatch:
    """
    P1-no-grad + P2-no-grad + submit：无梯度前向，提交 guidance 后立即释放 decode cache。

    五阶段流水线的前半部分：
      attach_batch → dense_sampling → P1 rollout → P2 decode+render (no_grad) → submit
    
    ★ 显存优势：
      P2 decode+render 在 torch.no_grad() 下执行，不保留 autograd 图。
      comp_rgb 仅用于异步提交 guidance（submit_async 内部会 detach），
      之后 decode cache（meshes / subs / shape_slat_norm）立即释放。
      GPU 上仅保留：slat proxy chain（~轻量）+ tracker（~1-2 GB）。

    正确性保证：
      Decoder（LayerNorm + SiLU，无 Dropout / BatchNorm）、Renderer（纯数学运算）
      在 no_grad 和 grad 模式下行为完全一致。
      backward_with_grad() 重跑 decode+render 可得到完全相同的 comp_rgb。

    OOM 安全降级：
      P2-no-grad OOM 时释放 decode 缓存，返回 submitted=False 的 PendingMicroBatch，
      backward_with_grad() 将跳过 P2-grad，仅做 P1-grad reg-only VJP。

    Returns:
        PendingMicroBatch: submitted=True（已提交 guidance）或 False（P2 OOM 降级）
    """
    gen_seed = int(system.cfg.seed) + global_step
    cfg = system.cfg
    pipeline = system.pipeline
    device = system.accelerator.device

    with TrainModeGuard(system.shape.model):
        state = Trellis2State()
        state.attach_batch(batch, pipeline=pipeline,
                           resolution=system.shape.config.cond_resolution)

        # ── P1-no-grad: dense sampling + rollout ──────────────────────
        profiler.tick("dense_sampling")
        dense_sampling_no_grad(state, system)

        profiler.tick("P1_rollout")
        tracker = shape_phase1_rollout(state, system, gen_seed)

        # ── P2-no-grad: decode + render（不保留 autograd 图） ─────────
        profiler.tick("P2_no_grad")
        try:
            with torch.no_grad():
                render_out = decode_and_render_normal(
                    state.features.shape_slat,
                    state.cameras,
                    pipeline,
                    system.shape.renderer,
                    device,
                    resolution=pipeline.target_resolution,
                    normal_mode=cfg.renderer.normal_mode,
                )
                comp_rgb = render_out["color"]  # (B, V, H, W, 3) 无 autograd 图

            # 存可视化数据（detach，不占 autograd 显存）
            state.views_generated.shape_tensor = comp_rgb.detach()

            # 异步提交 guidance
            profiler.tick("P2_submit_async")
            phase2_submit_guidance_async(comp_rgb, state.views_conditioned.image_pils, system)
            submitted = True

            # ★ 立即释放 decode cache（meshes / subs / shape_slat_norm）和临时变量
            del comp_rgb, render_out
            state.release_shape_decode_cache()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # P2-no-grad OOM：保留 state + tracker（含 reg_grads），降级 reg-only
            logging.warning(
                f"[Step {global_step}] P2-no-grad OOM → backward_with_grad reg-only"
            )
            state.release_shape_decode_cache()
            torch.cuda.empty_cache()
            profiler.reset()
            submitted = False

    return PendingMicroBatch(
        state=state,
        tracker=tracker,
        image_pils=state.views_conditioned.image_pils,
        submitted=submitted,
    )


def backward_with_grad(
    prev: PendingMicroBatch,
    system: Trellis2System,
    profiler: AsyncPhaseProfiler = None,
    global_step: int = 0,
) -> Dict[str, Any]:
    """
    P2-wait + P2-grad + P1-grad：有梯度后向，消化上一个 micro-batch。

    五阶段流水线的后半部分：
      P2-wait:  阻塞等待 guidance GPU 结果 → 获取 rgb_grad
      P2-grad:  重跑 decode+render（带梯度）→ comp_rgb.backward(rgb_grad)
                → output_trajectory[t].grad 填充 → 释放 decode cache
      P1-grad:  flow VJP → θ.grad 累积

    ★ 显存优势：
      P2-grad 执行时 GPU 上没有 curr 的 decode cache（forward_no_grad 已释放）。
      与原始方案相比，decode cache 不再跨越 forward/backward 边界存活。

    降级链路：
      1. submitted=True  → 正常：P2-wait + P2-grad + P1-grad
      2. submitted=False → P2-no-grad OOM，跳过 P2，仅 P1-grad reg-only
      3. P2-wait 失败    → 降级同 2
      4. P2-grad OOM     → 降级同 2

    ★ 无论何种情况，P1-grad 始终执行（tracker.reg_grads 在 P1-no-grad 已预计算）。
    """
    cfg = system.cfg
    pipeline = system.pipeline
    device = system.accelerator.device

    with TrainModeGuard(system.shape.model):
        guidance_log: Dict[str, Any] = {}
        rgb_grad = None

        if prev.submitted:
            # ── P2-wait: 等 guidance GPU 结果 ─────────────────────
            profiler.tick("P2_wait")
            try:
                rgb_grad, guidance_log = phase2_wait_guidance(prev.state, system)
                # 把 guidance GPU 计时传给 profiler（用于双 GPU 利用率分析）
                guid_timing = guidance_log.pop("_guid_timing", None)
                if guid_timing:
                    profiler.set_guid_timing(*guid_timing)
            except Exception as e:
                logging.warning(
                    f"[Step {global_step}] P2-wait failed: {e} → P1-grad reg-only"
                )
                guidance_log = {}
                rgb_grad = None

        if rgb_grad is not None:
            # ── P2-grad: 重跑 decode+render（带梯度）→ backward ──
            profiler.tick("P2_grad")
            try:
                render_out = decode_and_render_normal(
                    prev.state.features.shape_slat,
                    prev.state.cameras,
                    pipeline,
                    system.shape.renderer,
                    device,
                    resolution=pipeline.target_resolution,
                    normal_mode=cfg.renderer.normal_mode,
                )
                comp_rgb = render_out["color"]  # (B, V, H, W, 3) 有 autograd 图
                comp_rgb.backward(rgb_grad)
                # 释放 decode cache + 临时变量
                del comp_rgb, render_out, rgb_grad
                prev.state.release_shape_decode_cache()
                prev.state.regularization.reg_loss = None  # 清空引用
                torch.cuda.empty_cache()
            except Exception as e:
                # P2-grad OOM：降级 reg-only
                logging.warning(
                    f"[Step {global_step}] P2-grad failed: {e} → P1-grad reg-only"
                )
                del rgb_grad
                # ★ 清空可能被 backward 部分填充的 .grad，避免 P1-grad 混用
                for out_t in prev.tracker.output_trajectory:
                    out_t.grad = None
                prev.state.release_shape_decode_cache()
                prev.state.regularization.reg_loss = None  # 释放死图节点
                torch.cuda.empty_cache()
                guidance_log = {}

        # ★ P2-grad 结束后（无论成败），释放 proxy chain — P1-grad 不需要它
        if prev.state.features.shape_slat is not None:
            prev.state.features.shape_slat = prev.state.features.shape_slat.detach()
            torch.cuda.empty_cache()

        # ── P1-grad: flow VJP → θ.grad 累积 ─────────────────────
        profiler.tick("P1_grad")
        phase3_log = phase3_rollout_grad_backward(prev.state, system, prev.tracker)

    profiler.tick("end")

    # 合并日志
    merged = {**guidance_log, **phase3_log}
    merged["loss/total"] = (
        guidance_log.get("loss/guidance", 0.0)
        + cfg.train.loss.reg * phase3_log.get("loss/reg", 0.0)
    )

    # Profiler: 收集计时并合入日志
    merged.update(profiler.collect(global_step, print_freq=int(system.cfg.freq.profiler)))
    return merged


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    训练 Shape Flow Model，使用 Normal 渲染监督几何。
    
    流程: Dense Sampling → Shape Rollout → Normal 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_shape --config=configs/trellis2_shape.py
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
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=["wandb"] if use_wandb else None,
    )
    
    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    
    # 初始化 wandb trackers
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trellis2-shape-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    from edit4shape.systems.trellis2 import build_dataloaders
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)
    
    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator, guidance_factory=partial(create_guidance, use_pp=True))
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
            system,
            epoch=start_epoch,
            global_step=global_step,
            eval_loader=eval_loader,
            visuals_eval_dir=visuals_eval_dir,
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return
    
    # =====================================================
    # Step 8: 训练循环（五阶段 Autograd + 异步 Guidance 流水线）
    # =====================================================
    #
    # 五阶段流水线（双缓冲 prev / curr）：
    #   每个 micro-batch 的生命周期：
    #     P1-no-grad: flow 前向（不带梯度）→ tracker
    #     P2-no-grad: decode+render（不带梯度）→ submit 给 guidance → 释放 decode cache
    #        ... guidance GPU 异步计算 ...
    #     P2-wait:    等 guidance 结果 → rgb_grad
    #     P2-grad:    重跑 decode+render（带梯度）→ backward(rgb_grad) → 释放 decode cache
    #     P1-grad:    flow VJP → θ.grad 累积
    #
    #   双缓冲执行顺序（稳态）：
    #     curr = forward_no_grad(batch)              # P1-ng + P2-ng + submit（快，无 autograd 图）
    #     if prev: flush_prev(prev, ...)             # P2-wait + P2-grad + P1-grad（消化上一个 MB）
    #     prev = curr
    #
    # ★ 显存优势：P2-grad 执行时 GPU 上没有 curr 的 decode cache
    #   （forward_no_grad 已释放），decode cache 不再跨越 forward/backward 边界存活。
    # ★ 异步收益：curr.P2-no-grad + submit 与 prev 的 guidance 并行。
    # ★ OOM 安全：forward_no_grad 内部 catch P2-no-grad OOM → submitted=False，
    #   backward_with_grad 自动跳过 P2，仅做 P1-grad reg-only VJP。
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    profiler = AsyncPhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    accum_steps = int(cfg.train.gradient_accumulation_steps)
    
    if accum_steps < 2 and accelerator.is_main_process:
        logging.warning(
            "[AsyncPipeline] gradient_accumulation_steps=%d, "
            "异步流水线需要 accum≥2 才有并行收益。"
            "当前退化为同步模式，建议增大 accum_steps。",
            accum_steps,
        )
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)
        
        prev: Optional[PendingMicroBatch] = None      # 双缓冲：上一个已 submit 的 MB
        prev_step: int = 0                             # prev 对应的 global_step
        prev_batch_size: int = 0                       # prev 对应的 batch_size

        def flush_prev(pending: PendingMicroBatch, step: int, bs: int) -> None:
            """backward_with_grad → log → vis → 释放。"""
            log = backward_with_grad(pending, system, profiler=profiler, global_step=step)
            shape_logger.log_step(log, bs, step, epoch)
            if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)
            torch.cuda.empty_cache()

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            # ── curr 前向: P1-ng + P2-ng + submit（无梯度，快速）────
            curr = forward_no_grad(batch, system, global_step, profiler=profiler)
            
            # ── 消化 prev（P2-wait + P2-grad + P1-grad） ─────────
            if prev is not None:
                flush_prev(prev, prev_step, prev_batch_size)
            
            # ── prev ← curr（P2-no-grad OOM 时 submitted=False，降级 reg-only）
            prev, prev_step, prev_batch_size = curr, global_step, batch_size
            
            # ── Optimizer Step（在 accum 边界） ──────────────────
            if global_step % accum_steps == 0:
                if prev is not None:
                    flush_prev(prev, prev_step, prev_batch_size)
                    prev = None
                system.shape.optimizer.step()
                system.shape.optimizer.zero_grad()
        
        # ── epoch 结束：消化残留的 prev ──────────────────────────
        if prev is not None:
            flush_prev(prev, prev_step, prev_batch_size)
            prev = None
        # ★ 独立于 prev：只要不在 accum 边界，就有待 step 的残留梯度
        #   （即使最后几个 MB 全 OOM → prev=None，之前 flush 的梯度仍需 step）
        if global_step % accum_steps != 0:
            system.shape.optimizer.step()
            system.shape.optimizer.zero_grad()

        # ---- 周期性评估（epoch 级别）----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
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
