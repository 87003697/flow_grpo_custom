"""
Trellis2 Shape 训练系统 — 三阶段 Autograd 版本。

采用三阶段反向传播策略，将 rollout / decoder+renderer 的计算图隔离，
任意时刻只有一个阶段的计算图驻留显存，大幅降低显存峰值。

三阶段流程：
  Phase 1: rollout no_grad → shape_slat（proxy chain，不含模型图）
           + 计算 reg_loss（连着 cond_proxy 的图）→ state.regularization.reg_loss
  Phase 2: (guidance_loss + reg_weight * reg_loss).backward()
           → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）→ 释放所有图
  Phase 3: 纯 VJP — 逐步重算 f_θ → (v_grad * cond_pred).sum().backward()
           → flow model θ.grad +=，显存 O(1)，完全不感知 reg

数学等价性：
  ∂L/∂θ = Σ_t (∂L_guid/∂v_t^cond + λ · ∂L_reg/∂v_t^cond)^T · (∂v_t^cond/∂θ)
           ↑ Phase 2 一次性 backward 算出                       ↑ Phase 3 逐步 VJP

特性：
- 无 slat_proxy 中间层：loss.backward() 一路反传穿过 renderer → decoder → slat → scheduler → CFG → cond_proxy
- 三阶段显存隔离：峰值仅为 decode/render 图 + proxy chain
- 支持 mesh 与 mesh_peeled 两种 Normal 渲染模式
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


# _CONFIG 在 if __name__ == "__main__" 块中定义，
# 避免被其他模块 import 时重复注册 absl flag。

# =====================================================================
# TRELLIS.2 参考实现路径设置
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
from edit4shape.systems.utils import MetricLogger, Trellis2VisualIO, PhaseProfiler
from edit4shape.generators.trellis2.state import Trellis2State
from edit4shape.generators.trellis2.rollout import rollout_shape, RolloutTracker
from edit4shape.generators.trellis2.rollout.base import _predict_velocity
from edit4shape.systems.trellis2_shape import (
    StageSystem,
    build_system,
    decode_and_render_normal,
    trellis2_shape_forward,
    evaluate,
)

# =====================================================================
# Renderer 导入（使用 trellis2 的可微渲染器）
# =====================================================================
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
    # ★ 直接使用带 proxy chain 的 slat — 无需 slat_proxy 中间层
    # loss.backward() 将一路反传：renderer → decoder → slat → scheduler → CFG → cond_proxy.grad
    render_out = decode_and_render_normal(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=target_res,
    )
    comp_rgb = render_out["color"]  # (B, V, H, W, 3)
    
    # 挂载可视化数据（detach 避免保留计算图）
    state.views_generated.shape_tensor = comp_rgb.detach()
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]
    
    return comp_rgb


def phase2_guidance_and_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
    comp_rgb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Phase 2（同步版）: guidance + reg 合并 backward → 填充 tracker 梯度 → 释放图。
    
    reg_loss 在 Phase 1 的 rollout 中已计算并存于 state.regularization.reg_loss，
    其计算图连着 cond_proxy（通过 velocity → CFG → cond_proxy chain）。
    与 guidance_loss 合并后一次性 backward：
      cond_proxy.grad = ∂L_guid/∂cond_proxy + reg_weight * ∂L_reg/∂cond_proxy
    Phase 3 直接用 cond_proxy.grad 做 VJP，完全不感知 reg。
    
    流程:
    1. guidance = compute_guidance(comp_rgb, ...)
    2. total_loss = guidance.loss * weight + reg_weight * reg_loss
    3. accelerator.backward(total_loss)
       → 一路反传到 output_trajectory[t].grad（含 CFG 因子 + reg 梯度）
    4. 构建日志
    5. 释放所有计算图 + empty_cache()
    
    Returns:
        日志字典（包含 guidance loss、reg loss 等指标）
    """
    cfg = system.cfg
    accelerator = system.accelerator
    device = accelerator.device
    guidance_weight = cfg.shape.train.loss.guidance
    reg_weight = cfg.shape.train.loss.reg
    
    # 1. Guidance 前向（同步阻塞）
    guidance_result = system.guidance.compute_guidance(
        comp_rgb,
        state.views_conditioned.image_pils,
        guidance_cfg=cfg.shape.guidance,
        rank=accelerator.process_index,
    )
    state.attach_guidance_result(guidance_result)
    
    # 2. 合并 loss: guidance + reg（reg_loss 的图连着 cond_proxy，backward 自然传播）
    # comp_rgb ← renderer ← decoder ← slat ← scheduler ← CFG ← cond_proxy
    # reg_loss ← MSE ← velocity ← CFG ← cond_proxy
    # → 两路梯度汇聚到 cond_proxy.grad
    total_loss = guidance_result.loss.to(device) * guidance_weight  # ()
    reg_loss = state.regularization.reg_loss
    if reg_loss is not None:
        total_loss = total_loss + reg_weight * reg_loss  # ()
    
    accelerator.backward(total_loss)
    
    # 3. 构建日志
    guidance_log: Dict[str, Any] = {}
    if guidance_result.loss_dict:
        guidance_log.update({
            f"loss/{k}": v.item()
            for k, v in guidance_result.loss_dict.items()
            if v is not None
        })
    guidance_log["loss/guidance"] = (guidance_result.loss.to(device) * guidance_weight).item()
    if reg_loss is not None:
        guidance_log["loss/reg"] = reg_loss.item()
    
    # 4. 释放所有计算图（decode/render + proxy chain + reg 图一次性释放）
    del comp_rgb, total_loss, guidance_result, reg_loss
    state.regularization.reg_loss = None  # 图已释放，清空引用
    
    # 5. 释放 Shape 解码中间产物（Shape-only 训练中 Tex 阶段不会执行）
    state.release_shape_decode_cache()
    
    torch.cuda.empty_cache()
    
    return guidance_log


def phase3_rollout_grad_backward(
    state: Trellis2State,
    system: Trellis2System,
    tracker: RolloutTracker,
) -> Dict[str, Any]:
    """
    Phase 3: 纯 VJP — 逐步重算 f_θ，用 cond_proxy.grad 做 VJP → θ.grad 累积。
    显存 O(1)，不随步数增长。
    
    ★ Phase 3 完全不感知 reg：
    reg_loss 已在 Phase 1 计算（连着 cond_proxy），Phase 2 合并 backward 后，
    cond_proxy.grad 已同时包含 guidance 梯度和 reg 梯度。
    Phase 3 只需做 VJP: (cond_proxy.grad)^T · ∂f_θ/∂θ。
    
    流程（每步 t）:
      1. t_val, x_t, v_grad ← tracker 直接读取
      2. cond_pred = f_θ(x_t, t) — 唯一需要重算的，有 θ 梯度
      3. (v_grad * cond_pred).sum().backward() — VJP，图立即释放
    
    Returns:
        空日志字典（reg 日志已由 Phase 2 提供）
    """
    pipeline = system.pipeline
    device = system.accelerator.device
    stage_config = pipeline.get_stage_config("shape")
    flow_res = stage_config["flow_resolution"]
    
    # ---- 条件编码（只需 cond，不需要 uncond） ----
    cond_emb, _ = state.extract_embeddings(resolution=flow_res)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    
    T = len(tracker.input_trajectory)
    
    for i in range(T):
        # 1. 从 tracker 直接读取：t, x_t, v_grad
        t_val = tracker.timesteps[i]  # float64 精度
        
        x_t_feats = tracker.input_trajectory[i]  # (N, C), detached
        x_t = state.features.shape_slat.replace(x_t_feats)  # SparseTensor（无梯度）
        
        # 2. 重算 cond_pred = f_θ(x_t, t)（仅对 θ 有梯度，x_t detached）
        # ★ 只需 cond forward，无需 uncond / CFG 混合 / reg 计算
        # v_grad = cond_proxy.grad 已包含 CFG 因子 + reg 梯度（Phase 2 一次性 backward 得到）
        cond_pred = _predict_velocity(
            pipeline, x_t, t_val, cond_emb,
            "shape", flow_res, None
        )  # SparseTensor
        
        # 3. VJP: (v_grad)^T · ∂f_θ/∂θ → θ.grad +=
        # v_grad 为 None 时说明 Phase 2 OOM，跳过（无梯度贡献）
        v_grad = tracker.output_trajectory[i].grad  # (N, C) or None
        if v_grad is not None:
            (v_grad * cond_pred.feats).sum().backward()  # 图立即释放，显存 O(1)
    
    # ---- 释放 tracker 数据 ----
    del tracker.input_trajectory[:], tracker.output_trajectory[:]
    del tracker.timesteps[:]
    torch.cuda.empty_cache()
    
    return {}


# =====================================================================
# 三阶段 Autograd — 编排函数
# =====================================================================

def three_phase_shape_step(
    state: Trellis2State,
    system: Trellis2System,
    global_step: int,
    profiler: PhaseProfiler = None,
) -> Dict[str, Any]:
    """
    Shape-only 三阶段训练步（同步 Guidance 版本）。
    
    编排:
      dense_sampling → Phase 1 (rollout + tracker) → Phase 2a (decode + render)
      → Phase 2 (guidance + backward，一路反传到 cond_proxy.grad)
      → Phase 3 (逐步重算 + backward) → 返回日志
    
    Args:
        state: 已 attach_batch 的状态
        system: 训练系统
        global_step: 全局步数
        profiler: PhaseProfiler，用于测量各阶段耗时（enabled=False 时为空操作）
    
    Returns:
        合并的日志字典
    """
    gen_seed = int(system.cfg.seed) + global_step
    
    profiler.tick("dense_sampling")
    dense_sampling_no_grad(state, system)
    
    profiler.tick("P1_rollout")
    tracker = shape_phase1_rollout(state, system, gen_seed)
    
    profiler.tick("P2a_decode_render")
    comp_rgb = None
    try:
        comp_rgb = shape_phase2a_decode_render(state, system)
        
        profiler.tick("P2_guidance_backward")
        guidance_log = phase2_guidance_and_backward(state, system, tracker, comp_rgb)
    except torch.cuda.OutOfMemoryError:
        # Phase 2a/2 OOM（decode/render 或 guidance backward 显存不足）
        # → Phase 3 降级，该 micro-batch 零梯度贡献。
        # 安全性：Phase 2a/2 不经过模型参数，不触发 DDP hooks，不会导致分布式死锁。
        logging.warning(f"[Step {global_step}] Phase 2a/2 OOM → Phase 3 降级 reg-only")
        del comp_rgb
        state.release_shape_decode_cache()
        torch.cuda.empty_cache()
        guidance_log = {}
    
    profiler.tick("P3_grad_backward")
    phase3_log = phase3_rollout_grad_backward(state, system, tracker)
    
    profiler.tick("end")
    
    # 合并日志 + 计算 loss/total（与 trellis2_shape.py 对齐）
    # reg 日志现在在 guidance_log 里（Phase 2 合并 backward 时记录）
    merged = {**guidance_log, **phase3_log}
    total = guidance_log.get("loss/guidance", 0.0)
    if "loss/reg" in guidance_log:
        total += system.cfg.shape.train.loss.reg * guidance_log["loss/reg"]
    merged["loss/total"] = total
    
    # Profiler: 收集计时并合入日志（外部无需感知 profiler）
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
    # Step 8: 训练循环（三阶段 Autograd — velocity-level proxy）
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    profiler = PhaseProfiler(enabled=True, verbose=accelerator.is_main_process)
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline,
                               resolution=system.shape.config.cond_resolution)
            
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_log = three_phase_shape_step(state, system, global_step, profiler=profiler)
                
                # Optimizer Step
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
            
            # Logging & Visualization
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_shape_train(state=state, epoch=epoch, step=global_step)

            # 释放当前 step 残留引用
            del state, shape_log
            torch.cuda.empty_cache()

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
