"""
Trellis2 Shape 训练系统 — 三阶段 Autograd + 异步 Guidance 流水线版本。

基于 trellis2_shape_autograd.py 的三阶段架构，
额外在 **comp_rgb 层面做 proxy** 实现 train GPU / guidance GPU 的 autograd 图解耦，
使 guidance 计算与下一个 micro-batch 的 P1+P2a 并行。

异步流水线（双缓冲 prev/curr）：
  curr: dense → P1 → P2a → submit_async  (不等，立即返回)
  prev: wait → comp_rgb.backward(rgb_grad) → P3  (prev 的 guidance 已与 curr 前向并行跑完)

两层 proxy：
  1. cond_pred proxy (Phase 1) — 显存隔离，不保留 flow model 计算图
  2. comp_rgb  proxy (异步)   — 计算并行，train/guidance GPU 各自独立 backward

数学等价性：
  ∂L/∂θ = (∂L/∂comp_rgb) · (∂comp_rgb/∂v_t^cond) · (∂v_t^cond/∂θ)
           ↑ guidance GPU    ↑ train GPU backward    ↑ Phase 3 逐步重算

特性：
- accum≥2 时收益最大：N-1 个 MB 的 guidance 与下一个 MB 的前向并行
- accum=1 时退化为同步版（无并行窗口，但正确性不变）
- Phase 1/2a/3 与同步版完全相同，仅 Phase 2 和训练循环不同
- 评估路径仍使用单阶段 forward（trellis2_shape_forward）

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
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import rollout_shape, RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity
import torch.nn.functional as F

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
        
        与 trellis_distill 分支的 System.prepare_models_and_optimizers() 对齐：
        模型和优化器一起 prepare，DDP 模型回写到 pipeline，确保 forward 走 DDP。
        """
        if self.strategy is not None and self.shape.optimizer is not None:
            shape_config = self.shape.config
            self.shape.model, self.shape.optimizer = self.strategy.prepare(
                accelerator, "shape", shape_config.flow_resolution, self.shape.optimizer
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


def shape_phase2a_decode_render(
    state: Trellis2State,
    system: Trellis2System,
) -> torch.Tensor:
    """
    Shape Phase 2a: slat → decode → render → comp_rgb (normals)。
    
    直接使用带 proxy chain 的 slat（不创建 slat_proxy），
    decode/render 的 autograd 图连接到 proxy chain 上，
    后续 loss.backward() 一路反传到 output_trajectory[t].grad。
    
    Args:
        state: 已填充 shape_slat 的状态
        system: 训练系统
    
    Returns:
        comp_rgb: (B, V, H, W, 3) Normal 渲染图（有 autograd 图，连接到 proxy chain）
    """
    cfg = system.cfg
    pipeline = system.pipeline
    device = system.accelerator.device
    target_res = pipeline.target_resolution
    normal_mode = cfg.renderer.normal_mode
    
    # ★ 直接使用带 proxy chain 的 slat — 无需 slat_proxy 中间层
    # loss.backward() 将一路反传：renderer → decoder → slat → scheduler → CFG → cond_proxy.grad
    render_out = decode_and_render_normal(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=target_res,
        normal_mode=normal_mode,
    )
    comp_rgb = render_out["color"]  # (B, V, H, W, 3)
    
    # 挂载可视化数据（detach 避免保留计算图）
    state.views_generated.shape_tensor = comp_rgb.detach()
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]
    state.simplify_meshes()
    
    return comp_rgb


# =====================================================================
# PendingMicroBatch — 流水线双缓冲数据容器
# =====================================================================

@dataclass
class PendingMicroBatch:
    """
    记录一个已提交 guidance 但尚未 backward 的 micro-batch 的全部上下文。

    生命周期:
        1. 创建 → 填充 state / tracker / comp_rgb / image_pils（P1 + P2a 阶段）
        2. 调用 phase2_submit_guidance_async() → guidance GPU 开始异步计算
        3. 下一个 MB 的前向阶段结束后，调用 phase2_wait_and_backward() 取回梯度
        4. 调用 phase3_rollout_grad_backward() 完成 Phase 3
        5. 释放（del pending）
    """
    state: Trellis2State                  # 该 MB 的 Trellis2State（含 slat / cameras 等）
    tracker: RolloutTracker               # Phase 1 记录的 proxy 轨迹
    comp_rgb: torch.Tensor                # Phase 2a 输出（有 autograd 图，连接到 proxy chain）
    image_pils: List[Image.Image]         # 条件图像（submit 时用）


# =====================================================================
# 异步 Phase 2: submit + wait_and_backward
# =====================================================================

def phase2_submit_guidance_async(
    pending: PendingMicroBatch,
    system: Trellis2System,
) -> None:
    """
    Phase 2 前半：将 comp_rgb 异步提交给 guidance GPU。

    调用 PipelineParallelMixin.submit_async()：
      comp_rgb → detach → copy to guidance GPU → 独立 backward → 入队（FIFO）
    本函数立即返回，不阻塞 train GPU。

    Args:
        pending: 已填充 comp_rgb 和 image_pils 的 PendingMicroBatch
        system: 训练系统（访问 guidance + cfg）
    """
    guidance_weight = system.cfg.train.loss.guidance
    system.guidance.submit_async(
        pending.comp_rgb,
        pending.image_pils,
        guidance_weight=guidance_weight,
        rank=system.accelerator.process_index,
    )


def phase2_wait_and_backward(
    pending: PendingMicroBatch,
    system: Trellis2System,
) -> Dict[str, Any]:
    """
    Phase 2 后半：等待 guidance 结果 → comp_rgb.backward(rgb_grad) → 释放图。

    rgb_grad = ∂(weight*L)/∂comp_rgb（已在 guidance GPU 侧算好并搬回 train GPU）。
    comp_rgb.backward(rgb_grad) 沿 train 侧 autograd 图反传：
      comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    → 填充 output_trajectory[t].grad（含 CFG 缩放因子），与同步版语义完全一致。

    Args:
        pending: 已 submit 的 PendingMicroBatch
        system: 训练系统

    Returns:
        guidance_log: 日志字典（loss/guidance + 细分 loss）
    """
    device = system.accelerator.device

    # 1. 阻塞等待 guidance 结果
    async_result: AsyncGuidanceResult = system.guidance.wait_and_get(
        target_device=device,
    )

    # 2. backward: 用 rgb_grad 驱动 train 侧反传
    #    comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    pending.comp_rgb.backward(async_result.rgb_grad)  # 填充 output_trajectory[t].grad

    # 3. 构建日志
    guidance_log: Dict[str, Any] = {}
    if async_result.loss_dict:
        guidance_log.update({f"loss/{k}": v for k, v in async_result.loss_dict.items()})
    guidance_log["loss/guidance"] = async_result.loss_scalar

    # 4. 释放 comp_rgb 计算图
    del pending.comp_rgb, async_result
    pending.comp_rgb = None  # 标记已释放
    torch.cuda.empty_cache()

    return guidance_log


def phase3_rollout_grad_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
) -> Dict[str, Any]:
    """
    Phase 3（通用）: 逐步从 tracker 读取 input/grad/teacher，重算 f_θ 并即时 backward。
    θ.grad 逐步累积。显存 O(1)，不随步数增长。
    
    每步只做一次 student cond forward + 一次 backward（guidance 梯度项 + reg 合并）。
    Teacher velocity 已在 Phase 1 预计算并存于 tracker，无需再跑 teacher 模型。
    
    ★ 关键简化：output_trajectory 存的是 cond_pred proxy（不是 CFG 后的 velocity）。
    Phase 2 backward 沿 scheduler → CFG → cond_proxy chain 反传，
    cond_proxy.grad 已自动包含 CFG 缩放因子。
    因此 Phase 3 只需重算 cond_pred，**不需要 uncond 计算或 CFG 混合**。
    
    流程（每步 t）:
      1. t_val, x_t, v_grad, teacher_feats ← tracker 直接读取
      2. cond_pred = f_θ(x_t, t) — 唯一需要重算的，有 θ 梯度
      3. combined = (v_grad * cond_pred).sum() + λ * reg(cond_pred, teacher) — 合并 loss
      4. combined.backward() — 单次 backward，图立即释放
    
    Returns:
        日志字典（包含 reg loss 等指标）
    """
    cfg = system.cfg
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_config = pipeline.get_stage_config("shape")
    flow_res = stage_config["flow_resolution"]
    reg_weight = cfg.train.loss.reg
    
    # ---- 条件编码（只需 cond，不需要 uncond） ----
    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    
    # ---- 正则化配置（仅 v 模式，teacher 已在 Phase 1 预计算） ----
    reg_enabled = len(tracker.teacher_trajectory) > 0
    
    reg_loss_sum = 0.0
    T = len(tracker.input_trajectory)
    phase3_log: Dict[str, Any] = {}
    
    for i in range(T):
        # 1. 从 tracker 直接读取：t, x_t, v_grad, teacher
        t_val = tracker.timesteps[i]  # float64 精度
        
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.shape_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 2. 重算 cond_pred = f_θ(x_t, t)（仅对 θ 有梯度，x_t detached）
        # ★ 只需 cond forward，无需 uncond / CFG 混合
        # v_grad = cond_proxy.grad 已包含 CFG 缩放因子（Phase 2 沿 CFG chain 反传得到）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_val, cond_emb,
            "shape", flow_res, None
        )  # SparseTensor
        
        # 3. 合并 loss：guidance 梯度项 + 正则化项（共享同一次 forward 的计算图）
        v_grad = tracker.output_trajectory[i].grad  # (N, C), Phase 2 自动填充（含 CFG 因子）
        combined = (v_grad * cond_pred.feats).sum()  # ()  guidance 梯度项
        
        if reg_enabled:
            teacher_feats = tracker.teacher_trajectory[i]  # (N, C), Phase 1 预计算
            reg_loss = F.mse_loss(cond_pred.feats, teacher_feats.detach())  # cond v MSE
            combined = combined + reg_weight * reg_loss / T  # () ★ 除以 T，使 reg 梯度为步平均值
            reg_loss_sum = reg_loss_sum + reg_loss.item()
        
        # 4. 单次 backward — 图立即释放，显存 O(1)
        combined.backward()
    
    # ---- 日志 ----
    num_steps = max(1, T)
    if reg_enabled and reg_loss_sum > 0:
        phase3_log["loss/reg"] = reg_loss_sum / num_steps
    
    # ---- 释放 tracker 数据 ----
    del tracker.input_trajectory[:], tracker.output_trajectory[:]
    del tracker.timesteps[:], tracker.teacher_trajectory[:]
    torch.cuda.empty_cache()
    
    return phase3_log


# =====================================================================
# 三阶段 Autograd — 流水线辅助函数
# =====================================================================

def build_next(
    batch: Dict[str, Any],
    system: Trellis2System,
    global_step: int,
) -> PendingMicroBatch:
    """
    为一个 micro-batch 执行 P1 + P2a 的前向，返回已就绪的 PendingMicroBatch。

    流程：
      attach_batch → dense_sampling → P1 rollout → P2a decode+render
    结果 comp_rgb 含 autograd 图（连接到 proxy chain），可直接用于 submit/backward。
    """
    gen_seed = int(system.cfg.seed) + global_step
    
    state = Trellis2State()
    state.attach_batch(batch, pipeline=system.pipeline,
                       resolution=system.shape.config.cond_resolution)
    
    # Dense Sampling（no_grad）
    dense_sampling_no_grad(state, system)
    
    # Phase 1: Rollout no_grad + 记录 proxy 轨迹
    tracker = shape_phase1_rollout(state, system, gen_seed)
    
    # Phase 2a: Decode + Render
    comp_rgb = shape_phase2a_decode_render(state, system)
    
    return PendingMicroBatch(
        state=state,
        tracker=tracker,
        comp_rgb=comp_rgb,
        image_pils=state.views_conditioned.image_pils,
    )


def drain_prev(
    prev: PendingMicroBatch,
    system: Trellis2System,
) -> Dict[str, Any]:
    """
    对已 submit 的 prev：wait → backward → Phase 3，返回合并日志。

    用于:
    - 每个 micro-batch 的 prev 消化（只要 prev 存在就执行）
    """
    # Phase 2 后半: 等待 guidance 结果 → comp_rgb.backward(rgb_grad)
    guidance_log = phase2_wait_and_backward(prev, system)
    
    # Phase 3: 逐步重算 + 即时 backward → θ.grad +=
    phase3_log = phase3_rollout_grad_backward(prev.state, system, prev.tracker)
    
    # 合并日志
    merged = {**guidance_log, **phase3_log}
    total = guidance_log.get("loss/guidance", 0.0)
    if "loss/reg" in phase3_log:
        total += system.cfg.train.loss.reg * phase3_log["loss/reg"]
    merged["loss/total"] = total
    
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
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base")
    system = system.prepare_optimizers(accelerator)
    
    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = Trellis2CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, system, stages=["shape"], mode="train")
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
    # Step 8: 训练循环（三阶段 Autograd + 异步 Guidance 流水线）
    # =====================================================
    #
    # 双缓冲 prev / curr 流水线：
    #   for each micro-batch:
    #     curr = build_next(batch)               # P1 + P2a 前向
    #     phase2_submit_guidance_async(curr)      # 提交 guidance（不等）
    #     if prev: flush_prev(prev, ...)          # 消化上一个 MB（wait + backward + P3 + log）
    #     prev = curr
    #   after accum boundary / epoch end:
    #     if prev: flush_prev(prev, ...)          # 消化最后一个 MB
    #
    # ★ 关键：只要 prev 存在就走 flush_prev，
    #   不要求 accum_steps 是 2 的倍数。
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
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
            """drain → log → vis → 释放。"""
            with TrainModeGuard(system.shape.model):
                log = drain_prev(pending, system)
            shape_logger.log_step(log, bs, step, epoch)
            if accelerator.is_main_process and (step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=pending.state, epoch=epoch, step=step)
            torch.cuda.empty_cache()

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            # ── curr 前向: P1 + P2a + submit ──────────────────────
            with TrainModeGuard(system.shape.model):
                curr = build_next(batch, system, global_step)
                phase2_submit_guidance_async(curr, system)
            
            # ── 消化 prev（只要 prev 存在） ──────────────────────
            if prev is not None:
                flush_prev(prev, prev_step, prev_batch_size)
            
            # ── prev ← curr ──────────────────────────────────────
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
            ckpt_io.save(system, epoch, global_step, stages=["shape"])


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    _CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")
    app.run(main)
