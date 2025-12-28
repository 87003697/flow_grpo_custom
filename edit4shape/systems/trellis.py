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
import argparse
import csv
import json
import os
import random
import sys
import importlib.util
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

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
# FlowEdit 客户端
# =====================================================================

from edit4shape.guidance.flowedit import FlowEditClient


# =====================================================================
# 实用函数 - CFG 混合与调度器辅助
# =====================================================================

def mix_cfg(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, scale: float, uncond_mode: str = "detach") -> torch.Tensor:
    """
    Classifier-Free Guidance (CFG) 混合函数。
    
    CFG 是一种在扩散模型中增强条件生成质量的技术，通过放大条件预测与无条件预测的差异来实现。
    公式: output = cond_pred + scale * (cond_pred - uncond_pred)
    
    Args:
        cond_pred: 条件预测结果，形状 (B,T,C) 或 (N,C)
                   B=batch_size, T=token_num, C=channel_dim
        uncond_pred: 无条件预测结果，形状与 cond_pred 相同，可为 None
        scale: CFG 缩放因子，通常 > 1.0 以增强条件效果（如 7.5）
        uncond_mode: 梯度处理模式
            - "detach": 对 uncond_pred 断开梯度（默认，只训练条件分支）
            - "mirror": 对 cond_pred 断开梯度（反向训练）
            - "none": 保持两者梯度

    Returns:
        混合后的预测结果，形状与输入相同
    """
    if uncond_pred is None:
        return cond_pred  # (B,T,C) - 无 uncond 时直接返回条件预测
    if uncond_mode == "detach":
        uncond_pred = uncond_pred.detach()  # (B,T,C) - 阻止梯度回传到无条件分支
    if uncond_mode == "mirror":
        cond_pred = cond_pred.detach()  # (B,T,C) - 阻止梯度回传到条件分支
    return cond_pred + scale * (cond_pred - uncond_pred)  # (B,T,C) - CFG 公式


# =====================================================================
# TrellisState - 生成状态管理类
# =====================================================================

@dataclass
class TrellisState:
    """
    Trellis 生成过程的状态容器。
    
    存储整个生成流程中的所有中间状态，包括：
    - 稀疏结构坐标和特征
    - 相机参数（用于渲染）
    - 条件编码（用于条件生成）
    - 生成/编辑的视角缓存
    
    该类设计为可变数据容器，在生成流程中逐步填充各个字段。
    """

    @dataclass
    class Conditions:
        """条件编码容器。"""
        cond: Any = None      # 条件嵌入
        neg_cond: Any = None  # 无条件嵌入（用于 CFG）

    @dataclass
    class Cameras:
        """
        相机参数容器。
        
        存储渲染所需的相机矩阵，包括：
        - c2w: camera-to-world 变换矩阵 (4,4)
        - w2c: world-to-camera 变换矩阵 (4,4)
        - intrinsics: 相机内参矩阵 (3,3)
        - mvp: model-view-projection 矩阵 (4,4)
        
        支持两套分辨率的相机参数：
        - mesh_*: 高分辨率，用于 mesh 渲染
        - sdf_*: 低分辨率，用于 SDF 计算
        """

    @dataclass
    class ViewsGenerated:
        """生成视角缓存占位类。存储从 3D 表示渲染出的多视角图像。"""

    @dataclass
    class ViewsEdited:
        """编辑后视角缓存。存储经过编辑（如 FlowEdit 风格迁移）后的视角图像。"""
        images: Any = None  # (B,V,C,H,W) 编辑后的图像张量

    @dataclass
    class ViewsConditioned:
        """条件视角缓存。存储输入的条件图像（用于 FlowEdit 等场景）。"""
        images: Any = None  # list[len=B] of PIL.Image 条件图像
        paths: Any = None   # list[len=B] of str 条件图像路径

    @dataclass
    class Guidance:
        """Guidance 缓存占位类。存储用于监督的指导信号（如参考图像）。"""

    # ============== 核心状态字段 ==============
    coords: Any = None  # 稀疏结构坐标 (N,4)，4维为 [batch_idx, x, y, z]
    feats: Any = None   # 稀疏特征 (N,C)，C 为特征维度
    
    # ============== 子状态容器 ==============
    cameras: Cameras = field(default_factory=Cameras)  # 相机参数
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)  # 生成视角
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)  # 编辑视角
    views_conditioned: ViewsConditioned = field(default_factory=ViewsConditioned)  # 条件视角
    conditions: Conditions = field(default_factory=Conditions)  # 条件编码
    guidance: Guidance = field(default_factory=Guidance)  # 指导信号
    
    # ============== 数据挂载字段 ==============
    guidances_data: Any = None  # 挂载 batch["Guidances"]，包含监督信号

    def attach_batch(self, batch: Dict[str, Any]) -> "TrellisState":
        """
        从数据批次中提取并挂载条件、相机等信息。
        
        该方法解析 DataLoader 返回的 batch 字典，将各项数据
        挂载到 TrellisState 的相应字段中，便于后续处理。
        
        Args:
            batch: DataLoader 返回的批次数据，可能包含：
                - "Conditions": 条件编码字典 {"cond": ..., "neg_cond": ...}
                - "Guidances": 指导信号（如参考图像）
                - "mesh_*": 高分辨率相机参数
                - "sdf_*": 低分辨率相机参数
                - "camera_positions": 相机位置
                - "light_positions": 光源位置
        
        Returns:
            self: 支持链式调用
        
        Raises:
            ValueError: 当 Conditions 缺失或格式错误时
        """
        # ---- 1. 条件编码处理 ----
        if "Conditions" in batch:
            cond_dict = batch["Conditions"] or {}
            cond = cond_dict.get("cond")
            if cond is None:
                raise ValueError("batch['Conditions'] 缺少 cond，无法构造条件。")
            neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))  # 用于 CFG 无条件分支
            self.conditions.cond = cond
            self.conditions.neg_cond = neg_cond
        elif self.conditions.cond is None:
            raise ValueError("batch['Conditions'] 为空且 state.conditions 未设置，无法构造条件。")

        # ---- 2. 指导信号处理 ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]

        # ---- 3. 高分辨率相机参数 (mesh 渲染用) ----
        if "mesh_c2w" in batch:
            self.cameras.mesh_c2w = batch["mesh_c2w"]  # (B,V,4,4) camera-to-world
            self.cameras.mesh_w2c = batch["mesh_w2c"]  # (B,V,4,4) world-to-camera
            self.cameras.mesh_mvp = batch["mesh_mvp_mtx"]  # (B,V,4,4) MVP 矩阵
            self.cameras.mesh_positions = batch["mesh_camera_positions"]  # (B,V,3)
            self.cameras.mesh_intrinsics = batch["mesh_intrinsics"]  # (B,V,3,3)
        
        # ---- 4. 低分辨率相机参数 (SDF 计算用) ----
        if "sdf_c2w" in batch:
            self.cameras.sdf_c2w = batch["sdf_c2w"]  # (B,V,4,4)
            self.cameras.sdf_w2c = batch["sdf_w2c"]  # (B,V,4,4)

        # ---- 5. 共享相机/光源参数 ----
        if "camera_positions" in batch:
            self.cameras.camera_positions = batch["camera_positions"]  # (B,V,3)
        if "light_positions" in batch:
            self.cameras.light_positions = batch["light_positions"]  # (B,V,3)
        
        return self

    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 conditions 中提取条件和无条件嵌入。
        
        处理不同格式的条件输入（list 或 Tensor），统一输出为标准张量格式。
        
        Returns:
            tuple: (cond_embeddings, uncond_embeddings)
                - cond_embeddings: 条件嵌入 (B,S,C) 或 (B,C)
                - uncond_embeddings: 无条件嵌入，形状同上
        
        Raises:
            ValueError: 当 conditions 为空时
        """
        condition_utils = {"cond": self.conditions.cond, "neg_cond": self.conditions.neg_cond}
        if condition_utils.get("cond") is None:
            raise ValueError("TrellisState.conditions 为空，无法提取 embeddings。")
        
        # ---- 处理条件嵌入 ----
        cond_embeddings = condition_utils.get('cond')  # list 或 Tensor
        if isinstance(cond_embeddings, list):
            cond_embeddings = torch.cat(cond_embeddings, dim=0)  # (B,S,C) - 合并列表
        if isinstance(cond_embeddings, torch.Tensor) and cond_embeddings.dim() == 4 and cond_embeddings.shape[1] == 1:
            cond_embeddings = cond_embeddings.squeeze(1)  # (B,S,C) - 移除多余维度

        # ---- 处理无条件嵌入 ----
        uncond_embeddings = condition_utils.get('neg_cond')  # list 或 Tensor
        if isinstance(uncond_embeddings, list):
            uncond_embeddings = torch.cat(uncond_embeddings, dim=0)  # (B,S,C)
        if isinstance(uncond_embeddings, torch.Tensor) and uncond_embeddings.dim() == 4 and uncond_embeddings.shape[1] == 1:
            uncond_embeddings = uncond_embeddings.squeeze(1)  # (B,S,C)
        
        return cond_embeddings, uncond_embeddings



# =====================================================================
# System - 系统组件容器类
# =====================================================================

@dataclass
class System:
    """
    系统核心组件容器。
    
    封装了 Trellis 系统的四大核心组件：
    1. pipeline: 生成管道，负责条件编码、结构采样、特征采样、解码等核心生成逻辑
    2. renderer: 渲染器，将 3D 表示（mesh/GS）渲染为 2D 图像
    3. guidance: 指导模块，提供训练监督信号（如 SDS loss）
    4. optimizer: 优化器，用于模型参数更新
    
    该类还提供了环境设置、LoRA 适配、分布式训练准备等工具方法。
    """

    pipeline: Any = None   # 生成管道 (TrellisRefAdapter 等)
    renderer: Any = None   # 渲染器 (TrellisMeshRasterizer / GaussianRenderer)
    guidance: Any = None   # 指导模块 (SDS/VSD 等)
    optimizer: Any = None  # 优化器 (AdamW 等)

    @staticmethod
    def setup_env_and_seed(cfg: ml_collections.ConfigDict) -> None:
        """
        设置随机种子与确定性运行环境。
        
        确保实验可复现性，设置以下随机源的种子：
        - Python random
        - NumPy random
        - PyTorch CPU/CUDA
        - cuDNN 确定性模式
        
        Args:
            cfg: 配置对象，需包含 seed 字段
        """
        seed = int(cfg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 确定性卷积算法
        torch.backends.cudnn.benchmark = False     # 禁用自动调优以保证确定性

    def prepare_lora(
        self,
        cfg: ml_collections.ConfigDict,
        adapter: str = "base",
        load_path: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> "System":
        """
        准备 LoRA (Low-Rank Adaptation) 适配器。
        
        LoRA 是一种参数高效微调方法，通过低秩矩阵分解减少可训练参数量。
        此方法检测支持 LoRA 的组件并进行配置。
        
        Args:
            cfg: 配置对象
            adapter: 适配器名称，默认 "base"
            load_path: LoRA 权重加载路径，可选
            clone_from: 从已有适配器克隆，可选
        
        Returns:
            self: 支持链式调用
        """
        # 筛选支持 LoRA 的模块（需有 set_adapter 方法）
        target_modules = [m for m in [self.pipeline, self.guidance] if hasattr(m, "set_adapter")]
        for module in target_modules:
            if load_path and hasattr(module, "load_adapter"):
                module.load_adapter(load_path, adapter_name=adapter)
            module.set_adapter(adapter)
        return self

    def prepare_models_and_optimizers(self, cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> "System":
        """
        使用 Accelerate 包装模型和优化器以支持分布式训练。
        
        Accelerator.prepare() 会自动：
        - 将模型分布到多 GPU
        - 包装优化器以支持梯度同步
        - 处理混合精度训练
        
        Args:
            cfg: 配置对象
            accelerator: Accelerate 加速器实例
        
        Returns:
            self: 支持链式调用
        """
        if accelerator is None:
            return self
        
        items = []
        # 仅包装 nn.Module 类型的 pipeline（TrellisRefAdapter 是封装类，不直接包装）
        if isinstance(self.pipeline, torch.nn.Module):
            items.append(("pipeline", self.pipeline))
        if self.optimizer is not None:
            items.append(("optimizer", self.optimizer))
            
        if not items:
            return self

        # 调用 accelerator.prepare 进行分布式包装
        prepared = accelerator.prepare(*[obj for _, obj in items])
        
        # 单参数时 prepare 返回单个对象而非列表
        if len(items) == 1:
            prepared = [prepared]
            
        # 将包装后的对象替换回 System 属性
        for (name, _), wrapped in zip(items, prepared):
            setattr(self, name, wrapped)
        return self


# =====================================================================
# 构建函数 - 系统组件工厂
# =====================================================================

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
            - cfg.camera: 相机配置（分辨率、视角范围等）
            - cfg.renderer: 渲染器配置（类型、近远裁剪面等）
            - cfg.train.optimizer: 优化器配置
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 分布式训练加速器
    
    Returns:
        System: 包含所有组件的系统实例
    """
    # ---- 1. 构建 Pipeline (核心生成管道) ----
    # Pipeline 封装了 TRELLIS 的所有生成逻辑，包括：
    # - 图像条件编码 (DINOv2 等)
    # - 结构采样 (Dense Structure Sampling)
    # - 特征采样 (Sparse Latent Sampling, SLAT)
    # - 解码器 (Mesh/GS 解码)
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    pipeline = build_pipeline_from_reference(cfg, accelerator)

    # ---- 2. 构建 Renderer (3D 渲染器) ----
    # 根据配置选择渲染方式：
    # - "mesh": 基于 nvdiffrast 的可微分网格光栅化
    # - "gs": 基于 3D Gaussian Splatting 的渲染
    cam = cfg.camera
    renderer_type = cfg.renderer.get("type", "mesh")  # 默认使用 mesh 渲染
    
    if renderer_type == "gs":
        # ---- Gaussian Splatting 渲染器 ----
        # 优势：渲染速度快，支持实时渲染
        # 适用场景：预览、快速迭代
        from edit4shape.renderers.gaussian_splatting_trellis import GaussianRenderer
        rendering_options = {
            "resolution": cam.get("render_resolution", 512),  # 渲染分辨率 (像素)
            "near": cfg.renderer.get("near", 0.8),  # 近裁剪面距离
            "far": cfg.renderer.get("far", 1.6),    # 远裁剪面距离
            "ssaa": cfg.renderer.get("ssaa", 1),    # 超采样抗锯齿倍数
            "bg_color": cfg.renderer.get("bg_color", "random"),  # 背景色模式
        }
        renderer = GaussianRenderer(rendering_options)
    else:
        # ---- Mesh 光栅化渲染器 (nvdiffrast) ----
        # 优势：支持精确的几何渲染，法线/深度图质量高
        # 适用场景：训练、精细渲染
        from edit4shape.renderers.sparseflex_trellis import TrellisMeshRasterizer, TrellisRendererConfig
        renderer_cfg = TrellisRendererConfig(
            resolution=cam.get("render_resolution", 512),  # 渲染分辨率 (像素)
            ssaa=cfg.renderer.get("ssaa", 1),    # 超采样抗锯齿倍数
            near=cfg.renderer.get("near", 0.8),  # 近裁剪面距离
            far=cfg.renderer.get("far", 1.6),    # 远裁剪面距离
        )
        renderer = TrellisMeshRasterizer(cfg=renderer_cfg, device=str(accelerator.device))

    # ---- 3. 构建 Guidance 和 Optimizer ----
    # 仅在训练模式下创建优化器
    guidance = None  # TODO: 添加 SDS/VSD 等指导模块
    optimizer = None

    if not cfg.eval_only:
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
            - cfg.camera: 相机配置（视角范围、分辨率等）
            - cfg.batch_size: 训练批次大小
            - cfg.eval_batch_size: 评估批次大小
            - cfg.train_data_dir: 训练数据目录
            - cfg.eval_data_dir: 评估数据目录
            - cfg.eval_only: 是否仅评估模式
        accelerator: Accelerate 加速器，提供分布式信息
    
    Returns:
        tuple: (train_loader, eval_loader)
            - train_loader: 训练数据加载器（eval_only 时为 None）
            - eval_loader: 评估数据加载器
    """
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig
    
    cam = cfg.camera
    
    # ---- 构建训练相机配置 ----
    # 训练时相机参数在指定范围内随机采样，增加数据多样性
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cam.train.n_view,          # 每个样本采样的视角数
        yaw_range=list(cam.train.yaw_range),    # 偏航角范围 [min, max]
        pitch_range=list(cam.train.pitch_range), # 俯仰角范围 [min, max]
        r_range=list(cam.train.r_range),        # 相机距离范围 [min, max]
        fov_range=list(cam.train.fov_range),    # 视场角范围 [min, max]
    )
    
    # ---- 构建评估相机配置 ----
    # 评估时使用固定相机参数，确保结果可比较
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cam.eval.n_view,  # 评估视角数
        yaw=cam.eval.yaw,        # 固定偏航角
        pitch=cam.eval.pitch,    # 固定俯仰角
        r=cam.eval.r,            # 固定相机距离
        fov=cam.eval.fov,        # 固定视场角
    )
    
    # ---- 构建完整数据配置 ----
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.batch_size,           # 训练批次大小
        eval_batch_size=cfg.eval_batch_size, # 评估批次大小
        width=cam.render_resolution,   # 渲染宽度
        height=cam.render_resolution,  # 渲染高度
        image_dataset_dir=cfg.train_data_dir if not cfg.eval_only else cfg.eval_data_dir,
        eval_image_path=cfg.eval_data_dir,
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
    extr_all = cameras.mesh_w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.mesh_intrinsics.to(device)  # (B,V,3,3)
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
    extr_all = cameras.mesh_w2c.to(device)  # (B,V,4,4)
    intr_all = cameras.mesh_intrinsics.to(device)  # (B,V,3,3)
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
# 训练 - 核心训练循环
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
    device = accelerator.device
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
        cond_dict = {"cond": state.conditions.cond, "neg_cond": state.conditions.neg_cond}
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
        renderer_type = cfg.renderer.get("type", "mesh")
        
        if renderer_type == "gs":
            render_out = decode_and_render_gs(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color": (B,V,H,W,C), "gaussians": list
        else:
            render_out = decode_and_render_mesh(
                latents, state.cameras, system.pipeline, system.renderer, device
            )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
        
        comp_rgb = render_out["color"]  # (B,V,H,W,C) - 渲染的颜色图
        state.views_generated.images = comp_rgb  # 挂载生成图用于可视化
        
        # ---- FlowEdit Guidance & 损失计算 ----
        flowedit_client = FlowEditClient(cfg.guidance)
        guidance_result = flowedit_client.compute_guidance(
            comp_rgb, 
            state.views_conditioned.images,
            rank=accelerator.process_index,
        )
        state.views_edited.images = guidance_result.edited_imgs  # 存入 state
        
        # ---- 反向传播（使用 SpecifyGradient 绑定的梯度）----
        # 累加所有 loss（SpecifyGradient 返回的伪 loss）
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
    
    # 构建日志
    logs = {"loss/total": total_loss.detach()}
    if guidance_result.loss_ssim is not None:
        logs["loss/ssim"] = guidance_result.loss_ssim.detach()
    if guidance_result.loss_lpips is not None:
        logs["loss/lpips"] = guidance_result.loss_lpips.detach()
    if guidance_result.loss_latent_mse is not None:
        logs["loss/latent_mse"] = guidance_result.loss_latent_mse.detach()
    if guidance_result.avg_ssim is not None:
        logs["metric/ssim"] = guidance_result.avg_ssim
    if guidance_result.avg_lpips is not None:
        logs["metric/lpips"] = guidance_result.avg_lpips
    if guidance_result.avg_latent_mse is not None:
        logs["metric/latent_mse"] = guidance_result.avg_latent_mse
    
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
            
            # ---- 提取输入图像和名称 ----
            # batch['pixel_values'] 是 PIL.Image 列表
            images = batch['pixel_values']  # list[len=B] of PIL.Image
            image_names = [os.path.basename(p) for p in batch['image_path']]  # list[len=B]
            
            # =====================================================
            # Step 1: 准备条件编码
            # 使用 DINOv2 等编码器将图像编码为条件嵌入
            # =====================================================
            batch["Conditions"] = pipeline.prepare_image_conditions(images)  # dict with cond/neg_cond
            state.attach_batch(batch)  # 挂载相机参数等
            
            # =====================================================
            # Step 2: Dense Sampling（结构生成）
            # 根据条件生成稀疏 3D 坐标
            # =====================================================
            cond_dict = {"cond": state.conditions.cond, "neg_cond": state.conditions.neg_cond}
            coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)  # (N,4)
            state.coords = coords  # 保存到 state 供后续使用

            # =====================================================
            # Step 3: Sparse Sampling（特征生成）
            # 在稀疏坐标上执行去噪采样
            # =====================================================
            rollout_out = rollout_sparse(state, cfg, system, accelerator.device)  # dict
            latents = rollout_out["latents"]  # SparseTensor

            # =====================================================
            # Step 4: 解码 & 渲染 & 保存
            # 根据 renderer 类型选择解码格式
            # =====================================================
            renderer_type = cfg.renderer.get("type", "mesh")
            
            if renderer_type == "gs":
                # ---- Gaussian Splatting 分支 ----
                render_out = decode_and_render_gs(
                    latents, state.cameras, pipeline, system.renderer, accelerator.device
                )  # dict with "color": (B,V,H,W,C), "gaussians": list
                if accelerator.is_main_process:
                    save_gs_outputs(render_out, image_names, save_dir)
            else:
                # ---- Mesh Rasterizer 分支 ----
                render_out = decode_and_render_mesh(
                    latents, state.cameras, pipeline, system.renderer, accelerator.device
                )  # dict with "color"/"normal"/"depth": (B,V,H,W,C), "meshes": list
                if accelerator.is_main_process:
                    save_mesh_outputs(render_out, image_names, save_dir, pipeline, export_mesh=True)

    return {"eval_done": 1.0}


def build_run_paths(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[Path, Path, Path, Path]:
    """
    创建实验运行目录结构并保存配置。
    
    目录结构：
    {logdir}/{run_name}/
    ├── config.yaml          # 保存的配置文件
    ├── run_command.txt      # 启动命令
    ├── logs/                # 训练/评估日志 (CSV)
    ├── checkpoints/         # 模型检查点 (由 CheckpointIO 管理)
    └── visualizations/
        ├── train/           # 训练过程可视化
        └── eval/            # 评估结果可视化
    
    Args:
        cfg: 配置对象，需包含 logdir 和 run_name
        accelerator: Accelerate 加速器
    
    Returns:
        tuple: (run_root, logs_dir, visuals_train_dir, visuals_eval_dir)
    """
    run_root = Path(cfg.logdir) / (cfg.run_name if cfg.run_name else "trellis_run")
    logs_dir = run_root / "logs"
    visuals_train_dir = run_root / "visualizations" / "train"
    visuals_eval_dir = run_root / "visualizations" / "eval"
    
    # 仅主进程创建目录和保存配置（避免并发冲突）
    if accelerator.is_main_process:
        run_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        visuals_train_dir.mkdir(parents=True, exist_ok=True)
        visuals_eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件（YAML 格式）
        with (run_root / "config.yaml").open("w", encoding="utf-8") as f:
            f.write(yaml.dump(cfg.to_dict(), sort_keys=False))
        
        # 保存启动命令（便于复现）
        with (run_root / "run_command.txt").open("w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))
    
    return run_root, logs_dir, visuals_train_dir, visuals_eval_dir


# =====================================================================
# CheckpointIO - 检查点读写管理
# =====================================================================

@dataclass
class CheckpointIO:
    """
    检查点（Checkpoint）读写封装类。
    
    使用 Accelerate 的 save_state/load_state 进行分布式安全的检查点操作。
    
    检查点目录结构：
    ckpt_dir/
    └── checkpoint_{epoch}_{global_step}/
        ├── state.json         # Accelerate 状态文件
        ├── meta.json          # 自定义元数据 (epoch, global_step)
        ├── optimizer.bin      # 优化器状态
        └── pytorch_model.bin  # 模型权重
    
    Attributes:
        accelerator: Accelerate 加速器实例
        ckpt_dir: 检查点根目录
        start_epoch: 加载后的起始 epoch
        start_global_step: 加载后的起始 global_step
    """

    accelerator: Accelerator
    ckpt_dir: Path
    start_epoch: int = 0
    start_global_step: int = 0

    def save(self, system: System, state: TrellisState, cfg: ml_collections.ConfigDict, epoch: int, global_step: int) -> None:
        """
        保存检查点。
        
        保存内容包括：
        - 模型权重
        - 优化器状态
        - 学习率调度器状态
        - 随机数状态
        - 元数据 (epoch, global_step)
        
        Args:
            system: 系统组件
            state: TrellisState 状态对象
            cfg: 配置对象
            epoch: 当前 epoch
            global_step: 当前全局步数
        """
        target = self.ckpt_dir / f"checkpoint_{epoch}_{global_step}"
        target.mkdir(parents=True, exist_ok=True)
        
        # 同步所有进程
        self.accelerator.wait_for_everyone()
        
        # 使用 Accelerate 保存状态（自动处理分布式）
        self.accelerator.save_state(str(target))
        
        # 仅主进程保存元数据
        if self.accelerator.is_main_process:
            meta = {"epoch": int(epoch), "global_step": int(global_step)}
            with (target / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 等待所有进程完成
        self.accelerator.wait_for_everyone()

    def load(self, path: str, mode: str = "train") -> int:
        """
        加载检查点。
        
        Args:
            path: 检查点目录路径（如 "checkpoints/checkpoint_5_1000"）
            mode: 加载模式
                - "train": 从下一个 epoch 继续训练
                - "eval": 从 epoch 0 开始（仅评估）
        
        Returns:
            int: 起始 epoch（训练模式下为 loaded_epoch + 1）
        """
        cp = path
        # 路径无效时返回 0
        if not (isinstance(cp, str) and cp):
            self.start_epoch = 0
            return 0
        
        root = Path(cp)
        # 验证检查点目录结构
        if not (root.is_dir() and (root / "state.json").exists() and root.name.startswith("checkpoint_")):
            self.start_epoch = 0
            self.start_global_step = 0
            return 0
        
        # 同步所有进程
        self.accelerator.wait_for_everyone()
        
        # 使用 Accelerate 加载状态
        self.accelerator.load_state(str(root))
        
        self.accelerator.wait_for_everyone()
        
        # 读取元数据
        meta_path = root / "meta.json"
        assert meta_path.exists(), f"meta.json missing in {root}"
        meta = json.load(meta_path.open("r", encoding="utf-8")) or {}
        epoch_val = meta["epoch"]  # int
        step_val = meta["global_step"]  # int
        
        # 训练模式从下一个 epoch 开始
        self.start_epoch = int(epoch_val) + 1 if mode == "train" else 0
        self.start_global_step = int(step_val)
        return self.start_epoch


# =====================================================================
# ModeGuard - 模块模式上下文管理器
# =====================================================================

class ModeGuard:
    """
    模块模式上下文管理器。
    
    用于临时将模块切换到指定模式（train 或 eval），并在退出时恢复原状态。
    支持同时管理多个模块，确保 BatchNorm、Dropout 等层在不同模式下行为正确。
    
    Usage:
        # 切换到训练模式
        with ModeGuard(model1, model2, training=True):
            output = model1(input)  # 这里处于 train 模式
        
        # 切换到评估模式
        with ModeGuard(model1, model2, training=False):
            output = model1(input)  # 这里处于 eval 模式
        
        # 退出后自动恢复原来的 training 状态
    
    Attributes:
        modules: 要管理的模块列表
        training: 目标模式（True=train, False=eval）
        states: 保存的原始训练状态
    """

    def __init__(self, *modules: Any, training: bool = False):
        """
        初始化。
        
        Args:
            *modules: 要管理的 nn.Module 实例（自动过滤 None）
            training: 目标模式，True 为训练模式，False 为评估模式
        """
        self.modules = [m for m in modules if m is not None]
        self.training = training
        self.states = []  # 保存进入前的训练状态

    def __enter__(self):
        """进入上下文：保存状态并切换到目标模式。"""
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.train(self.training)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复原始训练状态。"""
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


def TrainModeGuard(*modules: Any) -> ModeGuard:
    """训练模式守卫。等价于 ModeGuard(..., training=True)。"""
    return ModeGuard(*modules, training=True)


def EvalModeGuard(*modules: Any) -> ModeGuard:
    """评估模式守卫。等价于 ModeGuard(..., training=False)。"""
    return ModeGuard(*modules, training=False)


# MetricLogger 从 utils 导入
from edit4shape.systems.utils import MetricLogger, append_csv_row, VisualIO


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
    visual_io = VisualIO(visuals_train_dir, target_h=cfg.camera.render_resolution, vis_freq=vis_freq)

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
    # Step 8: 训练循环
    # =====================================================
    # 初始化训练日志记录器（自动处理梯度累积）
    train_logger = MetricLogger(accelerator, logs_dir / "train.csv")
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        # 设置分布式采样器的 epoch（确保各进程数据不同）
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            
            # 创建新状态并挂载 batch 数据
            state = TrellisState()
            
            # 从 batch 提取图像并准备条件编码
            images = batch['pixel_values']  # list[len=B] of PIL.Image
            batch["Conditions"] = system.pipeline.prepare_image_conditions(images)  # dict with cond/neg_cond
            state = state.attach_batch(batch)
            
            # 存储条件图像供 FlowEdit 使用（作为所有视角的 target 参考）
            state.views_conditioned.images = images  # list[len=B] of PIL.Image
            state.views_conditioned.paths = batch.get("image_path")  # list[len=B] of str
            
            # 使用 accumulate 上下文管理器处理梯度累积
            with accelerator.accumulate(system.pipeline.pipe.models['slat_flow_model']):
                train_log = train_edit4shape(system, state, cfg, accelerator, epoch, global_step)
            
            # 仅主进程按频率保存三联图
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch(
                    state=state,
                    epoch=epoch,
                    step=global_step,
                )
            
            # 自动累积并在 sync_gradients 时发射平均日志
            train_logger.log_step(train_log, len(images), global_step, epoch)

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
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
