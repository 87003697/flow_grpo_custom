"""
Trellis2 Shape+Tex 双阶段训练系统。

本模块实现了基于 TRELLIS.2 架构的 3D 生成系统训练，同时训练 Shape 和 Tex 两个阶段。
核心流程：
- Stage 1 (Shape): 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss
- Stage 2 (Tex): Tex Rollout -> MeshWithVoxel -> PBR Voxel 渲染 -> Guidance Loss

特性：
- 双阶段同时训练：Shape 阶段用 Normal 渲染监督几何，Tex 阶段用 PBR Voxel 渲染监督纹理
- 每个 batch 分两步计算 Guidance Loss
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、tex_slat、相机参数、条件编码等）
2. Trellis2System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape / rollout_tex: 执行 Shape/Tex 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（渲染 Mesh Normal）
5. trellis2_tex_forward: Tex 阶段前向传播（使用 PbrMeshRenderer 渲染 PBR）
6. evaluate: 评估循环，生成 mesh 并保存可视化结果
7. main: 训练主循环（依次执行 Shape Guidance 和 Tex Guidance）

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- Shape 阶段: MeshRenderer 直接渲染 normal（支持梯度）
- Tex 阶段: PbrMeshRenderer 渲染 PBR + IBL 着色（支持梯度）

依赖：
- TRELLIS.2 参考实现 (_reference_codes/TRELLIS.2)
- Accelerate 分布式训练库
- nvdiffrast (可微光栅化渲染)
- nvdiffrec_render (PBR IBL 着色)
"""

# =====================================================================
# 环境变量设置（必须在 torch 导入之前）
# =====================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点，节省显存
from tqdm import tqdm

# =====================================================================
# 项目内部导入
# =====================================================================
from edit4shape.datasets.trellis import TrellisDataConfig, TrellisDataModule

# 使用 absl 的 config_flags 管理配置文件
_CONFIG = config_flags.DEFINE_config_file("config", help_string="Path to the config file.")

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
from edit4shape.generators.trellis2.chunked import MemoryMonitor

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
from edit4shape.systems.utils import MetricLogger, VisualIO

# =====================================================================
# Renderer 导入（使用 trellis2 的可微渲染器）
# =====================================================================
from trellis2.renderers import MeshRenderer, PbrMeshRenderer, EnvMap
from trellis2.representations.mesh import Mesh

# =====================================================================
# 类型定义
# =====================================================================
Stage = Literal["shape", "tex"]


# =====================================================================
# 从 trellis2_shape / generators 中导入共用组件（避免代码重复）
# =====================================================================
from edit4shape.generators.trellis2.rollout import rollout_shape
from edit4shape.generators.trellis2.rollout.base import (
    trellis2_cfg_sparse,
    _predict_velocity,
    _compute_regularization,
)
from edit4shape.systems.trellis2_shape import (
    StageSystem,
    decode_and_render_normal,
    trellis2_shape_forward,
    build_dataloaders,
)

# =====================================================================
# 从 trellis2_tex 导入 Tex 阶段组件（避免代码重复）
# =====================================================================
from edit4shape.systems.trellis2_tex import (
    # 扩展版 State（含 tex 字段）
    Trellis2State,
    # Tex 阶段核心函数
    rollout_tex,
    decode_and_render_pbr,
    trellis2_tex_forward,
)

# =====================================================================
# 从 training_adpter 导入 StageConfig
# =====================================================================
from edit4shape.generators.trellis2.training_adpter import StageConfig




@dataclass
class Trellis2System:
    """
    Trellis2 双阶段系统。
    
    组件结构：
    - pipeline: 共享的生成管道
    - shape: Shape 阶段（model, optimizer, renderer, config）
    - tex: Tex 阶段（model, optimizer, renderer, config）
    - guidance: 共享 Guidance
    
    渲染器配置（使用 trellis2 的 nvdiffrast 可微渲染器）：
    - shape.renderer: MeshRenderer (直接渲染 normal，支持梯度)
    - tex.renderer: PbrMeshRenderer (渲染 PBR + IBL 着色，支持梯度)
    
    使用示例：
        system = build_system(cfg, accelerator, guidance_factory)
        system = system.prepare_lora(cfg)
        system = system.prepare_optimizers(accelerator)
        
        # 访问组件
        system.shape.model      # Shape Flow Model
        system.shape.renderer   # MeshRenderer (Normal)
        system.tex.renderer     # PbrMeshRenderer (PBR)
        system.guidance         # 共享 Guidance
    """
    
    pipeline: Any = None
    
    # 分阶段系统
    shape: StageSystem = field(default_factory=StageSystem)
    tex: StageSystem = field(default_factory=StageSystem)
    
    # 共享组件
    guidance: Any = None
    
    # 训练策略（LoRA / Full / Frozen）
    strategy: Any = None
    
    # 运行时上下文（与 trellis2_shape.py 对齐）
    cfg: Any = None
    accelerator: Accelerator = None
    
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
        """准备双阶段优化器（使用 accelerator.prepare）"""
        if self.shape.optimizer is not None:
            self.shape.optimizer = accelerator.prepare(self.shape.optimizer)
        if self.tex.optimizer is not None:
            self.tex.optimizer = accelerator.prepare(self.tex.optimizer)
        return self




# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def build_system(
    cfg: ml_collections.ConfigDict, 
    accelerator: Accelerator,
    guidance_factory: callable,
) -> Trellis2System:
    """
    构建完整的 Trellis2 系统。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        Trellis2System: 包含所有组件的系统实例
    """
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, _build_single_optimizer,
    )
    from edit4shape.systems.base import compute_guidance_device
    from edit4shape.systems.utils.strategy import create_trellis2_strategy
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 注入 Chunked Decoder（强制启用自适应显存分块） ----
    shape_decoder = pipeline.pipe.models['shape_slat_decoder']
    tex_decoder = pipeline.pipe.models['tex_slat_decoder']
    ChunkedDecoderMixin.inject_to(shape_decoder)
    ChunkedDecoderMixin.inject_to(tex_decoder)
    logging.info("[Trellis2] Shape/Tex decoder 已启用 chunked forward（自适应显存）")
    
    # ---- 2. Renderer 配置（Shape 和 Tex 共用） ----
    render_opts = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
        "chunk_size": 8000000,  # 分块渲染：800万面片/chunk，避免 nvdiffrast 2^24 限制，保持可微
    }
    
    # ---- 3. 获取阶段配置（训练和评估都需要） ----
    shape_config = get_stage_config(pipeline_type, "shape")
    tex_config = get_stage_config(pipeline_type, "tex")
    
    # ---- 4. 构建 StageSystem（使用 trellis2 可微渲染器） ----
    # Shape 阶段：MeshRenderer 渲染 normal（nvdiffrast，支持梯度）
    shape_renderer = MeshRenderer(rendering_options=render_opts, device=device)
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    # Tex 阶段：PbrMeshRenderer 渲染 PBR（nvdiffrast，支持梯度）
    tex_renderer = PbrMeshRenderer(rendering_options=render_opts, device=device)
    # 加载环境贴图（使用现有函数处理 EXR 格式）
    from edit4shape.renderers.ovoxel_trellis2 import load_envmap
    logging.info(f"[PbrMeshRenderer] 加载环境贴图: {cfg.renderer.envmap_path}")
    tex_renderer.envmap = load_envmap(cfg.renderer.envmap_path, device=device)
    tex_stage = StageSystem(
        config=tex_config,
        renderer=tex_renderer,
    )
    
    # ---- 5. 训练模式：同时训练 Shape 和 Tex ----
    guidance = None
    strategy = None
    
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        train_mode = cfg.train.mode  # "lora" | "full" | "frozen"
        train_device = accelerator.device
        teacher_device = compute_guidance_device(accelerator.device)
        
        strategy = create_trellis2_strategy(
            mode=train_mode,
            pipeline=pipeline,
            train_device=train_device,
            teacher_device=teacher_device,
            pipeline_type=pipeline_type,
            stages=["shape", "tex"],
            lora_cfg=cfg.lora,
            pretrained_path=cfg.pretrained.model,
        )
        
        strategy.setup()
        strategy.prepare(accelerator)
        
        # 统一获取学生模型和构建优化器
        shape_model = strategy.get_student("shape", shape_config.flow_resolution)
        optimizer_shape = _build_single_optimizer(shape_model, cfg.train.optimizer)
        shape_stage.model = shape_model
        shape_stage.optimizer = optimizer_shape
        
        tex_model = strategy.get_student("tex", tex_config.flow_resolution)
        optimizer_tex = _build_single_optimizer(tex_model, cfg.train.optimizer)
        tex_stage.model = tex_model
        tex_stage.optimizer = optimizer_tex
        
        # 启用 Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        pipeline._set_decoder_checkpointing("tex_slat_decoder", enable=True)
        pipeline._set_flow_model_checkpointing("shape", shape_config.flow_resolution, enable=True)
        pipeline._set_flow_model_checkpointing("tex", tex_config.flow_resolution, enable=True)
        logging.info("[Trellis2] 已启用 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        tex=tex_stage,
        guidance=guidance,
        strategy=strategy,
        cfg=cfg,
        accelerator=accelerator,
    )




# =====================================================================
# 评估
# =====================================================================

@torch.no_grad()
def evaluate(
    system: Trellis2System,
    epoch: int,
    global_step: int,
    eval_loader: Any,
    visuals_eval_dir: Path,
) -> Dict[str, Any]:
    """
    评估函数：执行推理并保存可视化结果。
    
    完整的评估流程：
    1. 从图像提取条件编码
    2. 执行 Dense Sampling 生成稀疏结构
    3. 执行 Sparse Sampling 生成特征
    4. 解码为 3D 表示（mesh 或 GS）
    5. 渲染多视角图像并保存
    6. 导出 mesh 文件
    
    输出目录结构：
    visuals_eval_dir/
    └── epoch_{N}/
        ├── sample_name_1/
        │   ├── color.png      # 渲染的颜色图
        │   ├── normal.png     # 渲染的法线图（mesh 模式）
        │   └── mesh.obj       # 导出的网格文件
        └── sample_name_2/
            └── ...
    
    Args:
        system: 系统组件
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 输出目录
    
    Returns:
        dict: 评估日志
    """
    if eval_loader is None:
        return {}
    
    cfg = system.cfg
    if cfg is None:
        raise ValueError("system.cfg is required: ensure build_system() populates cfg.")
    accelerator = system.accelerator
    if accelerator is None:
        raise ValueError("system.accelerator is required: ensure build_system() populates accelerator.")
    
    pipeline = system.pipeline
    visual_io = VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution)
    
    # 获取需要设置为 eval 模式的模型
    models_to_eval = [
        system.shape.model,
        system.tex.model,
        pipeline.pipe.models['shape_slat_decoder'],
        pipeline.pipe.models['tex_slat_decoder'],
    ]
    
    # 过滤 None（eval_only 模式下 model 可能为 None）
    models_to_eval = [m for m in models_to_eval if m is not None]
    
    with EvalModeGuard(*models_to_eval):
        for batch_idx, batch in enumerate(eval_loader):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=pipeline, resolution=system.tex.config.cond_resolution)
            
            # Shape Forward (渲染 Normal)
            _ = trellis2_shape_forward(
                system, state, global_step,
                is_training=False
            )
            
            # Tex Forward (渲染 RGB)
            render_out = trellis2_tex_forward(
                system, state, global_step,
                is_training=False
            )
            
            visual_io.save_batch_eval(
                state=state,
                epoch=epoch,
                render_out=render_out,
                pipeline=pipeline,
                export_mesh=False,
            )
    
    return {"eval_done": 1.0}


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    同时训练 Shape 和 Tex 两个 Flow Model。
    - Shape 阶段使用 Normal 渲染监督几何
    - Tex 阶段使用 PBR 渲染监督纹理
    
    流程: Dense Sampling → Shape Rollout → Tex Rollout → RGB 渲染
    
    配置文件示例：
        python -m edit4shape.systems.trellis2_shape+tex --config=config/trellis2_shape+tex_distillation.py
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
            project_name="trellis2-shape+tex-distillation",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq, accelerator=accelerator)
    
    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
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
    start_epoch = ckpt_io.load(cfg.checkpoint, system, stages=["shape", "tex"], mode="train")
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
    # Step 8: 训练循环
    # =====================================================
    shape_logger = MetricLogger(accelerator, logs_dir / "train_shape.csv")
    tex_logger = MetricLogger(accelerator, logs_dir / "train_tex.csv")
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
        # ---- 计算总 loss ----
        # guidance.loss 在 Guidance 设备上，需要移到训练设备
        guidance_loss = state.guidance.loss.to(accelerator.device) * cfg.train.loss.guidance  # ()
        total = guidance_loss  # ()
        if state.regularization.reg_loss is not None:
            total = total + cfg.train.loss.reg * state.regularization.reg_loss  # ()
        
        # ---- 反向传播 ----
        accelerator.backward(total)
        
        # ---- 构建日志（直接复用 loss_dict）----
        logs = {f"loss/{k}": v.item() for k, v in (state.guidance.loss_dict or {}).items() if v is not None}
        logs["loss/total"] = total.item()
        if state.regularization.reg_loss is not None:
            logs["loss/reg"] = state.regularization.reg_loss.item()
        if state.regularization.reg_metric is not None:
            logs["loss/reg_metric"] = state.regularization.reg_metric
        return logs
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.tex.config.cond_resolution)
            
            # # ★ DEBUG: 开启 detect_anomaly 以获取详细的 backward 错误信息
            # with torch.autograd.set_detect_anomaly(True):
            
            # ============================================
            # Stage 1: Shape Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_render_out = trellis2_shape_forward(
                            system, state, global_step,
                            is_training=True
                        )
                    shape_normal = shape_render_out["color"]  # (B, V, H, W, 3) - Normal 图
                    
                    # Shape Guidance（使用 Normal 监督几何）
                    shape_guidance_result = system.guidance.compute_guidance(
                        shape_normal,
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(shape_guidance_result)
                    
                    # Shape Loss & Backward
                    shape_log = _compute_loss_and_backward(state)
                
                if accelerator.sync_gradients:
                    system.shape.optimizer.step()
                    system.shape.optimizer.zero_grad()
        
            # ============================================
            # Stage 2: Tex Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.tex.model):
                with TrainModeGuard(system.tex.model):
                    tex_render_out = trellis2_tex_forward(
                        system, state, global_step,
                        is_training=True
                    )
                    tex_rgb = tex_render_out["color"]  # (B, V, H, W, 3) - RGB 图
                    
                    # Tex Guidance（使用 RGB 监督纹理）
                    tex_guidance_result = system.guidance.compute_guidance(
                        tex_rgb,
                        state.views_conditioned.image_pils,
                        rank=accelerator.process_index,
                    )
                    state.attach_guidance_result(tex_guidance_result)
                    
                    # Tex Loss & Backward
                    tex_log = _compute_loss_and_backward(state)
                
                if accelerator.sync_gradients:
                    system.tex.optimizer.step()
                    system.tex.optimizer.zero_grad()
            
            # 每步结束后：卸载不需要的特征到 CPU + 清理显存缓存
            state.offload_features()
            torch.cuda.empty_cache()
        
            # ============================================
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            tex_logger.log_step(tex_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用最终的 RGB 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(state=state, epoch=epoch, step=global_step)
        
        # ============================================
        # Epoch 结束后：周期性评估和检查点保存
        # ============================================
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
        
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, epoch, global_step, stages=["shape", "tex"])


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)
