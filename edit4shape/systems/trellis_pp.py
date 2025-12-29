"""
Trellis 单 renderer 版（适配 Gen2Turbo Trellis 逻辑）。

本模块实现了基于 TRELLIS 架构的 3D 生成系统，支持从单张图像生成 3D 模型。
核心流程：图像条件编码 -> 结构采样 (Dense Sampling) -> 特征采样 (Sparse Sampling) -> 解码 -> 渲染

特性：
- 单 renderer，训练/推理共用统一 rollout。
- 必需稠密结构 coords，若缺失直接报错。
- 统一步数 num_steps_sparse，训练/推理一致。
- 全程 CFG：每步都跑 cond/uncond，再 mix_cfg。

主要组件：
1. TrellisState: 存储生成状态（坐标、特征、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_sparse: 执行稀疏特征的去噪采样过程
4. train_edit4shape: 训练循环，支持 Flow Matching 训练
5. evaluate: 评估循环，生成 mesh 并保存可视化结果

依赖：
- TRELLIS 参考实现 (_reference_codes/TRELLIS)
- Accelerate 分布式训练库
- nvdiffrast (mesh 渲染) 或 Gaussian Splatting (GS 渲染)
"""

# =====================================================================
# 标准库导入
# =====================================================================
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# =====================================================================
# 第三方库导入
# =====================================================================
from PIL import Image
import numpy as np
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
# TRELLIS 参考实现路径设置
# 将 TRELLIS 参考代码目录加入 Python 路径，以便导入其模块
# =====================================================================
import os
import sys
repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)

# SparseTensor: TRELLIS 中用于表示稀疏 3D 特征的核心数据结构
# 包含 coords (坐标) 和 feats (特征) 两个主要属性
from trellis.modules.sparse import SparseTensor

# =====================================================================
# Guidance 模块（使用流水线并行版本）
# =====================================================================

from edit4shape.guidance.base import GuidanceResult


def create_guidance_pp(cfg, train_device):
    """
    创建流水线并行版本的 Guidance 实例。
    
    使用 local_pp.py 中支持异步接口的 LocalGuidance。
    传递完整配置，以便从 cfg.guidance.flowedit 读取算法参数，
    从 cfg.train.loss 读取 loss 权重。
    """
    from edit4shape.guidance.backends.local_pp import LocalGuidance
    return LocalGuidance(cfg, train_device)


# =====================================================================
# 从 trellis.py 和 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    mix_cfg,
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    System,
    CheckpointIO,
    build_run_paths,
    TrellisState,
)
# from edit4shape.systems.trellis import TrellisState
from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

def _compute_train_device() -> torch.device:
    """
    计算当前 worker 的训练设备。
    
    Pipeline 并行规则：每个 worker 占用 2 张连续的卡
    - LOCAL_RANK 0: train=cuda:0, guidance=cuda:1
    - LOCAL_RANK 1: train=cuda:2, guidance=cuda:3
    - ...
    
    Returns:
        torch.device: 训练设备
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_idx = local_rank * 2
    
    if train_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"训练需要 cuda:{train_idx}，但本节点只有 {torch.cuda.device_count()} 个 GPU。"
            f"Pipeline 并行需要每 worker 2 张卡，当前 LOCAL_RANK={local_rank}。"
        )
    
    return torch.device(f"cuda:{train_idx}")


def build_system(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> System:
    """
    构建完整的 Trellis 系统。
    
    根据配置创建所有必要的组件，包括：
    1. Pipeline: 负责条件编码、结构/特征采样、解码的核心生成管道
    2. Renderer: 将 3D 表示渲染为 2D 图像的渲染器
    3. Guidance: 训练时的指导模块（如 SDS loss）
    4. Optimizer: 模型参数优化器
    
    Args:
        cfg: 完整配置对象，包含以下关键配置：
            - cfg.renderer: 渲染器配置（类型、分辨率、近远裁剪面等）
            - cfg.train.optimizer: 优化器配置
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 分布式训练加速器
    
    Returns:
        System: 包含所有组件的系统实例
    """
    # ---- 0. 计算正确的训练设备 ----
    # Pipeline 并行：每个 LOCAL_RANK 使用 2 张卡 (train + guidance)
    train_device = _compute_train_device()
    
    # ---- 1. 构建 Pipeline (核心生成管道) ----
    # Pipeline 封装了 TRELLIS 的所有生成逻辑，包括：
    # - 图像条件编码 (DINOv2 等)
    # - 结构采样 (Dense Structure Sampling)
    # - 特征采样 (Sparse Latent Sampling, SLAT)
    # - 解码器 (Mesh/GS 解码)
    # 注意：传递 train_device 确保 Pipeline 加载到流水线并行方案的正确设备上
    # 避免与 Guidance 设备冲突（Guidance 在 train_device + 1）
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    pipeline = build_pipeline_from_reference(cfg, accelerator, device=train_device)

    # ---- 2. 构建 Renderer (3D 渲染器) ----
    # 根据配置选择渲染方式：
    # - "mesh": 基于 nvdiffrast 的可微分网格光栅化
    # - "gs": 基于 3D Gaussian Splatting 的渲染
    renderer_type = cfg.renderer.type  # "mesh" 或 "gs"
    
    if renderer_type == "gs":
        # ---- Gaussian Splatting 渲染器 ----
        # 优势：渲染速度快，支持实时渲染
        # 适用场景：预览、快速迭代
        from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
        rendering_options = {
            "resolution": cfg.renderer.resolution,  # 渲染分辨率 (像素)
            "near": cfg.renderer.near,  # 近裁剪面距离
            "far": cfg.renderer.far,    # 远裁剪面距离
            "ssaa": cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            "bg_color": cfg.renderer.bg_color,  # 背景色模式
        }
        renderer = GaussianRenderer(rendering_options)
    else:
        # ---- Mesh 光栅化渲染器 (nvdiffrast) ----
        # 优势：支持精确的几何渲染，法线/深度图质量高
        # 适用场景：训练、精细渲染
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
        renderer_cfg = TrellisRendererConfig(
            resolution=cfg.renderer.resolution,  # 渲染分辨率 (像素)
            ssaa=cfg.renderer.ssaa,    # 超采样抗锯齿倍数
            near=cfg.renderer.near,  # 近裁剪面距离
            far=cfg.renderer.far,    # 远裁剪面距离
        )
        renderer = TrellisMeshRasterizer(cfg=renderer_cfg, device=str(train_device))

    # ---- 3. 构建 Guidance 和 Optimizer ----
    # 仅在训练模式下创建 Guidance 和优化器
    guidance = None
    optimizer = None

    if not cfg.eval_only:
        # 创建 Guidance（使用流水线并行版本，支持异步接口）
        # FlowEdit 模型自动放在 train_device + 1 (由 local_pp._compute_guidance_device 计算)
        guidance = create_guidance_pp(cfg, train_device=train_device)
        
        # 为 SLAT (Sparse Latent) 模型创建优化器
        from edit4shape.generators.trellis.training_adpter import build_optimizer_for_slat
        slat_model = pipeline.pipe.models["slat_flow_model"]  # 获取 SLAT flow 模型
        optimizer = build_optimizer_for_slat(slat_model, cfg.train.optimizer)

    return System(pipeline=pipeline, renderer=renderer, guidance=guidance, optimizer=optimizer)


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练和评估的 DataLoader。
    
    使用 TrellisDataModule 统一管理数据加载，支持：
    - 多视角相机采样（训练时随机，评估时固定）
    - 分布式数据分片
    - 图像预处理和条件准备
    
    Args:
        cfg: 配置对象，需包含：
            - cfg.data.train: 训练数据配置（batch_size, dir, n_view, yaw_range 等）
            - cfg.data.eval: 评估数据配置（batch_size, dir, n_view, yaw 等）
            - cfg.renderer.resolution: 渲染分辨率
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 加速器，提供分布式信息
    
    Returns:
        tuple: (train_loader, eval_loader)
            - train_loader: 训练数据加载器（eval_only 时为 None）
            - eval_loader: 评估数据加载器
    """
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig
    
    # ---- 构建训练相机配置 ----
    # 训练时相机参数在指定范围内随机采样，增加数据多样性
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,                    # 每个样本采样的视角数
        yaw_range=list(cfg.data.train.yaw_range),        # 偏航角范围 [min, max]
        pitch_range=list(cfg.data.train.pitch_range),    # 俯仰角范围 [min, max]
        r_range=list(cfg.data.train.r_range),            # 相机距离范围 [min, max]
        fov_range=list(cfg.data.train.fov_range),        # 视场角范围 [min, max]
    )
    
    # ---- 构建评估相机配置 ----
    # 评估时使用固定相机参数，确保结果可比较
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,    # 评估视角数
        yaw=cfg.data.eval.yaw,          # 固定偏航角
        pitch=cfg.data.eval.pitch,      # 固定俯仰角
        r=cfg.data.eval.r,              # 固定相机距离
        fov=cfg.data.eval.fov,          # 固定视场角
    )
    
    # ---- 构建完整数据配置 ----
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,           # 训练批次大小
        eval_batch_size=cfg.data.eval.batch_size,       # 评估批次大小
        width=cfg.renderer.resolution,   # 渲染宽度
        height=cfg.renderer.resolution,  # 渲染高度
        image_dataset_dir=cfg.data.train.dir if not cfg.eval_only else cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )

    # ---- 创建 DataModule 并设置分布式 ----
    dm = TrellisDataModule(
        dm_cfg, 
        num_replicas=accelerator.num_processes,  # 分布式进程数
        rank=accelerator.process_index           # 当前进程排名
    )
    dm.setup()

    # ---- 返回 DataLoader ----
    train_loader = dm.train_dataloader() if not cfg.eval_only else None
    eval_loader = dm.eval_dataloader()
    return train_loader, eval_loader


# =====================================================================
# Rollout - 核心采样循环（训练/评估共用）
# =====================================================================

def rollout_sparse(
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    system: System,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    执行稀疏特征的去噪采样循环（Rollout）。
    
    这是 TRELLIS 的核心生成函数，实现了 Stage 2 (SLAT) 的采样过程：
    1. 初始化高斯噪声潜变量
    2. 迭代去噪：每步执行 条件预测 -> CFG 混合 -> 调度器步进
    3. 应用归一化，得到最终特征
    
    该函数同时支持训练和推理两种模式：
    - 训练模式：启用梯度检查点 (Gradient Checkpointing) 节省显存
    - 推理模式：使用 no_grad 加速推理
    
    采样流程图：
    噪声 z_T -> 模型预测 v(z_t, t, c) -> CFG 混合 -> 调度器步进 -> z_{t-1} -> ... -> z_0
    
    Args:
        state: TrellisState 状态对象，包含条件编码、坐标等
        cfg: 配置对象
        system: 系统组件（pipeline、renderer 等）
        device: 运行设备
        generator: 随机数生成器（用于可复现性）
        is_training: 是否为训练模式
    
    Returns:
        dict: 包含以下键值：
            - "latents": SparseTensor, 最终的稀疏特征
            - "coords": (N,4), 稀疏坐标 [batch_idx, x, y, z]
    """
    pipeline = system.pipeline
    # 获取采样器运行时参数
    # ss_steps: 结构采样步数, slat_steps: 特征采样步数
    # slat_guidance: CFG 强度, slat_rescale_t: 时间步重缩放
    ss_steps, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 提取条件/无条件嵌入 ----
    cond_embeddings, uncond_embeddings = state.extract_embeddings()  # (B,S,C),(B,S,C)
    cond_embeddings = cond_embeddings.to(device)  # (B,S,C) - 移动到目标设备
    if uncond_embeddings is not None:
        uncond_embeddings = uncond_embeddings.to(device)  # (B,S,C)

    # =====================================================
    # Stage 1: 结构生成 (Structure Generation / Dense Sampling)
    # 生成稀疏 3D 坐标，定义几何结构的位置（训练时已外部完成）
    # =====================================================
    assert state.coords is not None, "state.coords 缺失：训练/推理需先完成稠密结构生成。"  # (N,4)
    coords = state.coords  # (N,4) - N = B * T，T 为每个样本的点数
    
    batch_size = cond_embeddings.shape[0]  # () - 批次大小
    if generator is None:
        # 创建可复现的随机数生成器
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    
    # =====================================================
    # Stage 2: 特征采样初始化 (SLAT Initialization)
    # 初始化高斯噪声潜变量
    # =====================================================
    in_channels = pipeline.pipe.models['slat_flow_model'].in_channels  # 特征通道数
    latents_sparse = pipeline.init_latents(
        coords=coords, 
        in_channels=in_channels, 
        generator=generator
    )  # SparseTensor: feats 形状 (N,C)

    # 提取 feats 张量用于后续操作（模型参数有梯度，无需对输入 latent 开梯度）
    latents_feats = latents_sparse.feats  # (N,C)

    # =====================================================
    # Scheduler 配置
    # 设置时间步序列（从 T 到 0 的递减序列）
    # =====================================================
    scheduler = pipeline.scheduler()  # 获取调度器实例
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    # CFG 区间：只在 [slat_cfg_min, slat_cfg_max] 时间范围内应用 CFG
    slat_cfg_min, slat_cfg_max = pipeline.pipe.slat_sampler_params["cfg_interval"]  # float

    # =====================================================
    # 定义拆分后的去噪函数
    # 分离 cond/uncond 分支便于控制梯度流
    # =====================================================
    
    def _expand_t_to_batch(t_scalar, batch_size, device):
        """
        将标量时间步扩展为 batch 维度。
        模型期望 t 形状为 (B,)，每个样本对应一个时间步。
        """
        if torch.is_tensor(t_scalar):
            t_val = float(t_scalar.item()) if t_scalar.dim() == 0 else float(t_scalar)  # ()
        else:
            t_val = float(t_scalar)  # ()
        return torch.full((batch_size,), t_val, device=device, dtype=torch.float32)  # (B,)

    def get_cond_pred(current_feats, t_tensor, cond_emb):
        """
        条件分支预测。
        
        在训练时需要保持梯度，使用 Gradient Checkpointing 减少显存。
        """
        x_t = SparseTensor(coords=coords, feats=current_feats)  # 重建 SparseTensor
        t_batch = _expand_t_to_batch(t_tensor, cond_emb.shape[0], current_feats.device)  # (B,)
        cond_out = pipeline.sparse_sampling_step(
            x_t, t_batch, cond_emb, uncond_embeddings=None, guidance_scale=0.0
        )  # SparseTensor
        return cond_out.feats  # (N,C)

    def get_uncond_pred(current_feats, t_tensor, uncond_emb):
        """
        无条件分支预测。
        
        始终在 no_grad 下执行，因为 CFG 只需要梯度流经条件分支。
        """
        x_t = SparseTensor(coords=coords, feats=current_feats)  # 重建 SparseTensor
        t_batch = _expand_t_to_batch(t_tensor, uncond_emb.shape[0], current_feats.device)  # (B,)
        uncond_out = pipeline.sparse_sampling_step(
            x_t, t_batch, uncond_emb, uncond_embeddings=None, guidance_scale=0.0
        )  # SparseTensor
        return uncond_out.feats  # (N,C)

    # =====================================================
    # 执行去噪循环 (Denoising Loop)
    # 从 T 步迭代到 0 步，逐步去除噪声
    # =====================================================
    timesteps_list = list(scheduler.timesteps)
    # 最后一个时间步不需要推理（已经是 z_0）
    steps_to_run = timesteps_list[:-1] if len(timesteps_list) > 1 else timesteps_list
    
    # 训练时显示进度条
    if is_training:
        steps_to_run = tqdm(
            steps_to_run, 
            desc="Rollout", 
            leave=False, 
            disable=not Accelerator().is_main_process
        )

    # 仅在训练时启用梯度检查点
    use_ckpt = is_training

    for t in steps_to_run:
        t_val = float(t) if torch.is_tensor(t) else float(t)  # ()
        # 判断当前时间步是否在 CFG 区间内
        apply_cfg = slat_cfg_min <= t_val <= slat_cfg_max  # bool

        # ---- Step 1: 条件分支预测 ----
        if use_ckpt:
            # 训练模式：使用 Gradient Checkpointing 节省显存
            # 前向时不保存中间激活，反向时重新计算
            cond_pred = checkpoint(
                get_cond_pred,
                latents_feats,
                t,
                cond_embeddings,
                use_reentrant=False  # 推荐使用 non-reentrant 模式
            )  # (N,C)
        else:
            # 推理模式：使用 no_grad 减少内存占用和计算
            with torch.no_grad():
                cond_pred = get_cond_pred(latents_feats, t, cond_embeddings)  # (N,C)

        # ---- Step 2: 无条件分支预测 ----
        # 始终在 no_grad 下执行（CFG 只需条件分支的梯度）
        uncond_pred = None
        if apply_cfg and uncond_embeddings is not None:
            with torch.no_grad():
                uncond_pred = get_uncond_pred(latents_feats, t, uncond_embeddings)  # (N,C)

        # ---- Step 3: CFG 混合 ----
        # 公式: v = v_cond + scale * (v_cond - v_uncond)
        if apply_cfg:
            velocity_preds = mix_cfg(
                cond_pred=cond_pred,
                uncond_pred=uncond_pred,
                scale=float(slat_guidance),
                uncond_mode=True  # detach uncond 分支
            )  # (N,C)
        else:
            # CFG 区间外直接使用条件预测
            velocity_preds = cond_pred  # (N,C)

        # ---- Step 4: 调度器步进 ----
        # 根据预测的速度场更新潜变量
        x_t_sparse = SparseTensor(coords=coords, feats=latents_feats)  # 当前状态
        v_pred_sparse = SparseTensor(coords=coords, feats=velocity_preds)  # 预测速度场
        
        step_out = scheduler.step(v_pred_sparse, t, x_t_sparse)
        latents_feats = step_out.prev_sample.feats  # (N,C) - 更新为下一时刻状态

    # =====================================================
    # 后处理：应用 SLAT 归一化
    # 将归一化的特征恢复到原始尺度
    # 参考：TRELLIS/trellis/pipelines/trellis_image_to_3d.py:248-250
    # =====================================================
    slat_norm = pipeline.pipe.slat_normalization
    std = torch.tensor(slat_norm['std'])[None].to(latents_feats.device)  # (1,C) - 标准差
    mean = torch.tensor(slat_norm['mean'])[None].to(latents_feats.device)  # (1,C) - 均值
    latents_feats = latents_feats * std + mean  # (N,C) - 反归一化

    # =====================================================
    # 构建返回结果
    # =====================================================
    final_latents = SparseTensor(coords=coords, feats=latents_feats)
    return {"latents": final_latents, "coords": coords}


# =====================================================================
# 渲染工具函数 - Mesh 渲染
# =====================================================================

def decode_and_render_mesh(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # TrellisMeshRasterizer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Mesh 并渲染多视角图像。
    
    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: Mesh 渲染器实例
        device: 运行设备
    
    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "normal": (B,V,H,W,3) 法线图
            - "depth": (B,V,H,W,1) 深度图
            - "meshes": list[len=B] of MeshExtractResult
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['mesh'])  # dict
    meshes = outputs['mesh']  # list[len=B] of MeshExtractResult
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    batch_size, num_views = extr_all.shape[:2]  # (), ()
    
    # ---- 逐样本逐视角渲染 ----
    all_renders: Dict[str, List[torch.Tensor]] = {}
    
    for i, mesh in enumerate(meshes):
        view_renders: Dict[str, List[torch.Tensor]] = {}
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)
            
            # Mesh 渲染器返回 dict of (H,W,C)
            render_out = renderer.render(mesh, ext_iv, intr_iv)  # dict
            
            for k, val in render_out.items():
                view_renders.setdefault(k, []).append(val)  # (H,W,C)
        
        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        for k, v_list in view_renders.items():
            stacked = torch.stack(v_list, dim=0)  # (V,H,W,C)
            all_renders.setdefault(k, []).append(stacked)
    
    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {}
    for k, b_list in all_renders.items():
        result[k] = torch.stack(b_list, dim=0)  # (B,V,H,W,C)
    
    result["meshes"] = meshes  # 保留 mesh 供导出
    return result


# =====================================================================
# 渲染工具函数 - Gaussian Splatting 渲染
# =====================================================================

def decode_and_render_gs(
    latents: Any,  # SparseTensor
    cameras: Any,  # TrellisState.Cameras
    pipeline: Any,
    renderer: Any,  # GaussianRenderer
    device: torch.device,
) -> Dict[str, Any]:
    """
    解码潜变量为 Gaussian Splatting 并渲染多视角图像。
    
    Args:
        latents: SparseTensor, rollout 输出的稀疏特征
        cameras: TrellisState.Cameras, 相机参数容器
        pipeline: 生成 pipeline，提供 decode 方法
        renderer: GS 渲染器实例
        device: 运行设备
    
    Returns:
        dict: 渲染输出，包含：
            - "color": (B,V,H,W,3) 渲染的颜色图
            - "gaussians": list[len=B] of Gaussian 对象
    """
    # ---- 解码 ----
    outputs = pipeline.decode(latents, formats=['gaussian'])  # dict
    gaussians = outputs['gaussian']  # list[len=B] of Gaussian
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.intrinsics.to(device)  # (B,V,3,3)
    batch_size, num_views = extr_all.shape[:2]  # (), ()
    
    # ---- 逐样本逐视角渲染 ----
    all_colors: List[torch.Tensor] = []
    
    for i, gs in enumerate(gaussians):
        view_colors: List[torch.Tensor] = []
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4,4)
            intr_iv = intr_all[i, v]  # (3,3)
            
            # GS 渲染器返回 color: (C,H,W)
            render_out = renderer.render(gs, ext_iv, intr_iv)  # dict
            color = render_out['color']  # (C,H,W)
            color = color.permute(1, 2, 0)  # (H,W,C)
            view_colors.append(color)
        
        # 堆叠视角维度: list[V] of (H,W,C) -> (V,H,W,C)
        stacked = torch.stack(view_colors, dim=0)  # (V,H,W,C)
        all_colors.append(stacked)
    
    # 堆叠 batch 维度: list[B] of (V,H,W,C) -> (B,V,H,W,C)
    result: Dict[str, Any] = {
        "color": torch.stack(all_colors, dim=0),  # (B,V,H,W,C)
        "gaussians": gaussians,  # 保留 GS 供其他用途
    }
    return result


# =====================================================================
# 保存工具函数 - Mesh 输出
# =====================================================================

def save_mesh_outputs(
    render_out: Dict[str, Any],
    image_names: List[str],
    save_dir: Path,
    pipeline: Any,
    export_mesh: bool = True,
) -> None:
    """
    保存 Mesh 渲染结果到磁盘。
    
    Args:
        render_out: decode_and_render_mesh 的输出
        image_names: 样本名称列表
        save_dir: 输出目录
        pipeline: 用于导出 mesh 的 pipeline
        export_mesh: 是否导出 mesh 文件
    """
    meshes = render_out.get("meshes", [])
    
    for i, name in enumerate(image_names):
        name = os.path.splitext(name)[0]
        sample_dir = save_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各渲染通道（取第一个视角）
        for k, v in render_out.items():
            if k == "meshes":
                continue
            img = v[i, 0]  # (H,W,C) - 第 i 个样本的第 0 个视角
            img_np = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np[..., 0]  # (H,W)
            Image.fromarray(img_np).save(str(sample_dir / f"{k}.png"))
        
        # 导出 mesh
        if export_mesh and i < len(meshes):
            out_path = sample_dir / "mesh.obj"
            pipeline.export_mesh_obj(meshes[i], str(out_path))
            print(f"Saved mesh to {out_path}")


# =====================================================================
# 保存工具函数 - GS 输出
# =====================================================================

def save_gs_outputs(
    render_out: Dict[str, Any],
    image_names: List[str],
    save_dir: Path,
) -> None:
    """
    保存 GS 渲染结果到磁盘。
    
    Args:
        render_out: decode_and_render_gs 的输出
        image_names: 样本名称列表
        save_dir: 输出目录
    """
    for i, name in enumerate(image_names):
        name = os.path.splitext(name)[0]
        sample_dir = save_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存颜色图（取第一个视角）
        color = render_out["color"][i, 0]  # (H,W,C)
        img_np = (color.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
        Image.fromarray(img_np).save(str(sample_dir / "color.png"))


# =====================================================================
# 训练 - Trellis 前向传播（流水线用）
# =====================================================================

def train_trellis_forward(
    system: System,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> torch.Tensor:
    """
    Trellis 前向传播：生成渲染图像（不执行 Guidance 和 backward）。
    
    用于流水线并行：只执行到 comp_rgb，然后异步提交给 Guidance。
    
    注意：调用此函数时需要在外层使用 TrainModeGuard。
    
    Args:
        system: 系统组件（pipeline、renderer、optimizer）
        state: TrellisState 状态对象（已挂载 batch 数据）
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
    
    Returns:
        torch.Tensor: 渲染图像 comp_rgb (B,V,H,W,C)
    """
    device = _compute_train_device()  # 使用 Pipeline 并行的设备计算规则
    pipeline = system.pipeline
    
    # ---- 1. Dense Sampling（结构生成）----
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()
    with torch.no_grad():
        cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
    state.coords = coords
    
    # ---- 2. Rollout：执行稀疏特征采样 ----
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    
    rollout_out = rollout_sparse(
        state, cfg, system, device,
        generator=generator,
        is_training=True,
    )
    latents = rollout_out["latents"]  # SparseTensor
    
    # ---- 3. 解码 & 渲染 ----
    renderer_type = cfg.renderer.type
    
    if renderer_type == "gs":
        render_out = decode_and_render_gs(
            latents, state.cameras, system.pipeline, system.renderer, device
        )
    else:
        render_out = decode_and_render_mesh(
            latents, state.cameras, system.pipeline, system.renderer, device
        )
    
    comp_rgb = render_out["color"]  # (B,V,H,W,C)
    state.views_generated.image_tensor = comp_rgb
    
    return comp_rgb


# =====================================================================
# 训练 - 核心训练循环（单步，兼容接口）
# =====================================================================

def train_edit4shape(
    system: System,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    global_step: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    单步训练函数（核心训练循环）。
    
    实现了基于 Flow Matching 的 3D 生成训练流程：
    1. 执行 rollout_sparse 生成潜变量（带梯度）
    2. 解码潜变量得到 3D 表示（mesh/GS）
    3. 渲染多视角图像
    4. 计算损失（重建损失 + guidance 损失）
    5. 反向传播更新参数
    
    训练策略：
    - 使用 Gradient Checkpointing 减少显存占用
    - 每步使用不同的随机种子以增加数据多样性
    - 支持梯度累积（通过 accelerator 配置）
    
    Args:
        system: 系统组件（pipeline、renderer、optimizer）
        state: TrellisState 状态对象（已挂载 batch 数据）
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
    
    Returns:
        tuple: (训练日志字典, 渲染输出字典)
    """
    device = _compute_train_device()  # 使用 Pipeline 并行的设备计算规则
    optimizer = system.optimizer

    # =====================================================
    # 1. 准备阶段
    # =====================================================
    # 注意：optimizer.zero_grad() 移到反向传播后，配合 accelerator.accumulate() 使用
    
    # =====================================================
    # 2. 显式结构生成 (Dense Sampling)
    # 与评估流程保持一致，先生成稠密坐标再进入 SLAT 采样
    # =====================================================
    pipeline = system.pipeline
    ss_steps, _, _, _, _, _ = pipeline.get_sampler_runtime_params()  # () - 解析结构采样步数
    with torch.no_grad():
        cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
        coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4) - 稠密采样得到稀疏坐标
    state.coords = coords  # (N,4) - 挂载坐标供后续 rollout 使用
    
    # =====================================================
    # 3. 训练核心逻辑（在 TrainModeGuard 下执行）
    # Pipeline 加载时默认将所有模型设为 eval（见 base.py Pipeline.__init__）
    # 需要将 flow model 和解码器切换到 train 模式以启用可微分路径
    # =====================================================
    pipe_models = pipeline.pipe.models
    with TrainModeGuard(
        pipe_models.get('slat_flow_model'),      # 我们训练的目标模型
        pipe_models.get('slat_decoder_mesh'),    # 使 mesh_extractor(x, training=True) 启用可微分 FlexiCubes
        pipe_models.get('slat_decoder_gs'),      # GS 解码器保持一致性
    ):
        # ---- Rollout：执行稀疏特征采样 ----
        # 每步使用不同的随机种子，确保训练数据多样性
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
        
        rollout_out = rollout_sparse(
            state, cfg, system, device, 
            generator=generator, 
            is_training=True,  # 启用梯度和 Checkpointing
        )
        
        # latents 是 SparseTensor，其 feats 包含完整的计算图
        latents = rollout_out["latents"]
        
        # ---- 解码 & 渲染 ----
        # 根据 renderer 类型选择解码格式并渲染多视角图像
        renderer_type = cfg.renderer.type
        
        if renderer_type == "gs":
            render_out = decode_and_render_gs(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color": (B,V,H,W,C), "gaussians": list
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        
        comp_rgb = render_out["color"]  # (B,V,H,W,C) - 渲染的颜色图
        state.views_generated.image_tensor = comp_rgb  # 挂载生成图用于可视化
        
        # ---- FlowEdit Guidance & 损失计算 ----
        # 使用 system.guidance（在 build_system 中初始化，支持 local/http 后端）
        guidance_result = system.guidance.compute_guidance(
            comp_rgb, 
            state.views_conditioned.image_pils,
            rank=accelerator.process_index,
        )
        state.views_edited.image_tensor = guidance_result.edited_imgs  # 存入 state
        
        # ---- 反向传播 ----
        # local 后端：loss 是真正的可微分 loss，autograd 自动处理
        # http 后端：loss 是 SpecifyGradient 返回的伪 loss，使用预计算梯度
        total_loss = 0
        if guidance_result.loss_ssim is not None:
            total_loss = total_loss + guidance_result.loss_ssim
        if guidance_result.loss_lpips is not None:
            total_loss = total_loss + guidance_result.loss_lpips
        if guidance_result.loss_latent_mse is not None:
            total_loss = total_loss + guidance_result.loss_latent_mse
        
        # 使用 accelerator.backward() 支持混合精度和分布式训练
        accelerator.backward(total_loss)
        
        # 仅在梯度同步时（累积完成）执行优化器步骤
        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()
    # TrainModeGuard 退出后自动恢复模型的原始模式
    
    # 构建日志（loss tensor 直接用 .detach() 记录）
    logs = {"loss/total": total_loss.detach()}
    if guidance_result.loss_ssim is not None:
        logs["loss/ssim"] = guidance_result.loss_ssim.detach()
    if guidance_result.loss_lpips is not None:
        logs["loss/lpips"] = guidance_result.loss_lpips.detach()
    if guidance_result.loss_latent_mse is not None:
        logs["loss/latent_mse"] = guidance_result.loss_latent_mse.detach()
    
    return logs


# =====================================================================
# 评估 - 推理与可视化保存
# =====================================================================

@torch.no_grad()
def evaluate(
    system: System,
    cfg: ml_collections.ConfigDict,
    accelerator: Accelerator,
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
        cfg: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 可视化输出目录
    
    Returns:
        dict: 评估日志字典
    """
    if eval_loader is None:
        return {}
    
    pipeline = system.pipeline
    # 获取采样参数
    ss_steps, _, slat_steps, slat_guidance, _, _ = pipeline.get_sampler_runtime_params()
    
    # ---- 创建输出目录 ----
    save_dir = visuals_eval_dir / f"epoch_{epoch}"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logs: Dict[str, Any] = {}
    
    # =====================================================
    # 使用 EvalModeGuard 确保所有模型处于评估模式
    # =====================================================
    pipe_models = pipeline.pipe.models
    with EvalModeGuard(
        pipe_models.get('slat_flow_model'),
        pipe_models.get('slat_decoder_mesh'),
        pipe_models.get('slat_decoder_gs'),
    ):
        # =====================================================
        # 遍历评估数据集
        # =====================================================
        for batch_idx, batch in enumerate(eval_loader):
            # 每个 batch 创建独立状态，避免跨 batch 残留
            state = TrellisState()
            
            # ---- 提取输入图像和名称，挂载 batch 数据 ----
            image_names = [os.path.basename(p) for p in batch['paths']]  # list[len=B]
            state.attach_batch(batch, pipeline=pipeline)  # 自动从 image_pils 生成条件编码并挂载
            
            # =====================================================
            # Step 2: Dense Sampling（结构生成）
            # 根据条件生成稀疏 3D 坐标
            # =====================================================
            cond_dict = {"cond": state.views_conditioned.cond_embed, "neg_cond": state.views_conditioned.uncond_embed}
            coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
            state.coords = coords  # 保存到 state 供后续使用

            # =====================================================
            # Step 3: Sparse Sampling（特征生成）
            # 在稀疏坐标上执行去噪采样
            # =====================================================
            eval_device = _compute_train_device()  # 使用 Pipeline 并行的设备计算规则
            rollout_out = rollout_sparse(state, cfg, system, eval_device)  # dict
            latents = rollout_out["latents"]  # SparseTensor

            # =====================================================
            # Step 4: 解码 & 渲染 & 保存
            # 根据 renderer 类型选择解码格式
            # =====================================================
            renderer_type = cfg.renderer.type
            
            if renderer_type == "gs":
                # ---- Gaussian Splatting 分支 ----
                render_out = decode_and_render_gs(
                    latents, state.cameras, pipeline, system.renderer, eval_device
                )  # dict with "color": (B,V,H,W,C), "gaussians": list
                if accelerator.is_main_process:
                    save_gs_outputs(render_out, image_names, save_dir)
            else:
                # ---- Mesh Rasterizer 分支 ----
                render_out = decode_and_render_mesh(
                    latents, state.cameras, pipeline, system.renderer, eval_device
                )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
                if accelerator.is_main_process:
                    save_mesh_outputs(render_out, image_names, save_dir, pipeline, export_mesh=True)

    return {"eval_done": 1.0}


# =====================================================================
# 主函数入口
# =====================================================================

def main(argv) -> None:
    """
    程序主入口。
    
    完整的训练/评估流程：
    1. 解析配置文件
    2. 设置环境与随机种子
    3. 初始化 Accelerator（分布式/混合精度）
    4. 创建运行目录
    5. 构建数据加载器
    6. 构建系统组件（pipeline, renderer, optimizer）
    7. 加载检查点（如有）
    8. 执行训练循环或评估
    
    配置文件示例：
        python -m edit4shape.systems.trellis --config=configs/trellis.py
    
    关键配置项：
        - cfg.eval_only: True 时仅执行评估
        - cfg.num_epochs: 训练总 epoch 数
        - cfg.eval_freq: 评估频率（每 N 个 epoch）
        - cfg.save_freq: 保存检查点频率
        - cfg.checkpoint: 恢复训练的检查点路径
    
    Args:
        argv: 命令行参数（由 absl.app.run 传入，本函数不使用）
    """
    del argv  # absl.app.run 会传入 argv；本函数不使用
    cfg = _CONFIG.value

    # =====================================================
    # Step 1: 环境设置
    # =====================================================
    System.setup_env_and_seed(cfg)

    # =====================================================
    # Step 2: 初始化 Accelerator
    # 配置混合精度训练和梯度累积
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,  # "no", "fp16", "bf16"
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )

    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    vis_freq = int(cfg.freq.save.visual)
    visual_io = VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq)

    # =====================================================
    # Step 4: 构建数据加载器
    # =====================================================
    train_loader, eval_loader = build_dataloaders(cfg, accelerator)

    # =====================================================
    # Step 5: 构建系统组件
    # =====================================================
    system = build_system(cfg, accelerator)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)
    system = system.prepare_models_and_optimizers(cfg, accelerator)

    # =====================================================
    # Step 6: 检查点管理
    # =====================================================
    ckpt_root = run_root / "checkpoints"
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.get('checkpoint'), mode="train")
    global_step = int(ckpt_io.start_global_step)

    # =====================================================
    # Step 7: 评估模式（仅评估不训练）
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator, 
            epoch=start_epoch, 
            global_step=global_step, 
            eval_loader=eval_loader, 
            visuals_eval_dir=visuals_eval_dir
        )
        eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
        eval_logger.accumulate(eval_log, 1)
        eval_logger.flush(global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环（流水线并行版）
    # =====================================================
    # 初始化训练日志记录器（自动处理梯度累积）
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    grad_accum_steps = cfg.train.gradient_accumulation_steps
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        # 设置分布式采样器的 epoch（确保各进程数据不同）
        train_loader.sampler.set_epoch(epoch)
        
        # 将 dataloader 转为 iterator，便于手动控制 gradient accumulation
        data_iter = iter(train_loader)
        
        # 流水线主循环
        pipe_models = system.pipeline.pipe.models
        epoch_done = False
        
        def _process_result(result: Any, state: TrellisState) -> None:
            """处理 guidance 结果：backward + 记录日志"""
            state.views_edited.image_tensor = result.edited_imgs
            
            # 构建 loss 和日志
            # 注意：result 中的 loss 在 guidance 设备上，需要移动到训练设备
            train_device = _compute_train_device()  # 使用 Pipeline 并行的设备计算规则
            loss = torch.tensor(0.0, device=train_device)
            logs: Dict[str, Any] = {}
            if result.loss_ssim is not None:
                loss_ssim = result.loss_ssim.to(train_device)  # (,) 移动到训练设备
                loss = loss + loss_ssim
                logs["loss/ssim"] = loss_ssim
            if result.loss_lpips is not None:
                loss_lpips = result.loss_lpips.to(train_device)  # (,) 移动到训练设备
                loss = loss + loss_lpips
                logs["loss/lpips"] = loss_lpips
            if result.loss_latent_mse is not None:
                loss_latent_mse = result.loss_latent_mse.to(train_device)  # (,) 移动到训练设备
                loss = loss + loss_latent_mse
                logs["loss/latent_mse"] = loss_latent_mse
            logs["loss/total"] = loss
            
            # backward（除以 grad_accum_steps 做平均）
            accelerator.backward(loss / grad_accum_steps)
            
            # 累积日志（MetricLogger 会自动处理平均）
            train_logger.accumulate(logs, batch_size=1)
        
        while not epoch_done:
            n_micro = 0
            pending_states: List[TrellisState] = []
            
            with TrainModeGuard(
                pipe_models.get('slat_flow_model'),
                pipe_models.get('slat_decoder_mesh'),
                pipe_models.get('slat_decoder_gs'),
            ):
                for mb_idx in range(grad_accum_steps):
                    # ---- 1. 获取下一个 batch ----
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        epoch_done = True
                        break
                    
                    global_step += 1
                    n_micro += 1
                    
                    # ---- 2. 准备当前 micro-batch ----
                    state = TrellisState()
                    state.attach_batch(batch, pipeline=system.pipeline)  # 自动从 image_pils 生成条件编码并挂载
                    pending_states.append(state)
                    
                    # ---- 3. Trellis 前向（在默认 stream 上）----
                    comp_rgb = train_trellis_forward(
                        system, state, cfg, accelerator, epoch, global_step
                    )
                    
                    # ---- 4. 异步提交 guidance（不阻塞）----
                    # 使用 FIFO 队列，可安全地先提交再等待
                    system.guidance.submit_async(comp_rgb, state.views_conditioned.image_pils)
                    
                    # ---- 5. 等待并处理已完成的 guidance（流水线并行）----
                    # 当队列中有 2 个任务时，说明前一个已经有足够时间完成，可以取出处理
                    # 这样实现真正的双缓冲：当前 batch 的 guidance 在后台执行，
                    # 同时处理上一个 batch 的 backward
                    if mb_idx > 0:
                        result = system.guidance.wait_and_get()
                        _process_result(result, pending_states[mb_idx - 1])
                
                # ---- 6. 清空队列：处理所有剩余的 pending guidance ----
                # 已处理的数量 = mb_idx（因为 mb_idx > 0 时处理了 mb_idx - 1）
                # 如果循环正常结束（没有 break），remaining_idx = n_micro - 1
                # 需要处理 pending_states[n_micro - queue_size : n_micro]
                processed_count = max(0, n_micro - 1) if n_micro > 0 else 0  # 已经处理了多少个
                while system.guidance.has_pending() and processed_count < len(pending_states):
                    result = system.guidance.wait_and_get()
                    _process_result(result, pending_states[processed_count])
                    processed_count += 1
                
                # ---- 7. 优化器步进 ----
                if n_micro > 0:
                    system.optimizer.step()
                    system.optimizer.zero_grad()
            
            # ---- 8. 日志和可视化 ----
            if n_micro > 0:
                # 使用最后一个 state 做可视化
                if pending_states and accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                    visual_io.save_batch(
                        state=pending_states[-1],
                        epoch=epoch,
                        step=global_step,
                    )
                
                # flush 日志（分布式聚合 + 发射）
                train_logger.flush(global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator, 
                epoch=epoch, 
                global_step=global_step, 
                eval_loader=eval_loader, 
                visuals_eval_dir=visuals_eval_dir
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            # 使用最后一个 state 保存
            last_state = pending_states[-1] if pending_states else TrellisState()
            ckpt_io.save(system, last_state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
