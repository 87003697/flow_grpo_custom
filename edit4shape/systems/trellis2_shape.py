"""
Trellis2 Shape 训练系统（专注于 Shape 阶段训练）。

本模块实现了基于 TRELLIS.2 架构的 3D 几何生成系统训练，支持从单张图像生成 3D 模型。
核心流程：
- 图像条件 -> Dense Sampling -> Shape Rollout -> Mesh -> Normal 渲染 -> Guidance Loss

特性：
- 专注 Shape 阶段训练：使用 Normal 渲染监督几何
- 不使用 Low VRAM 模式
- 支持 1024 非 cascade 模式

主要组件：
1. Trellis2State: 存储生成状态（shape_slat、相机参数、条件编码等）
2. System: 封装 pipeline、renderer、guidance、optimizer 等核心组件
3. rollout_shape: 执行 Shape 阶段的去噪采样
4. trellis2_shape_forward: Shape 阶段前向传播（渲染 Mesh Normal）
5. evaluate: 评估循环，生成 mesh 并保存可视化结果
6. main: 训练主循环

渲染器（使用 trellis2 的 nvdiffrast 可微渲染器）：
- MeshRenderer 直接渲染 normal（支持梯度）

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

# =====================================================================
# Guidance 模块
# =====================================================================
from edit4shape.guidance import create_guidance
from edit4shape.systems.base import SpecifyGradient

# =====================================================================
# 从 base.py 导入通用组件
# =====================================================================
from edit4shape.systems.base import (
    ModeGuard,
    TrainModeGuard,
    EvalModeGuard,
    BaseState,
    CheckpointIO,
    build_run_paths,
    SpecifyGradient,
)
from edit4shape.systems.utils import MetricLogger, append_csv_row, Trellis2VisualIO, LossDict

# =====================================================================
# Renderer 导入（使用伪 GT Mesh 方案的 MeshRenderer）
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
# CFG 函数（对齐 TRELLIS.2 参考实现）
# =====================================================================

def _sparse_pred_to_xstart(
    x_t: SparseTensor,
    t: float,
    pred: SparseTensor,
    sigma_min: float,
) -> SparseTensor:
    """
    从 velocity 预测 x0（对齐参考实现 FlowEulerSampler._pred_to_xstart）。
    
    公式: x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
    
    Args:
        x_t: SparseTensor，当前 latent
        t: 时间步 [0, 1]
        pred: SparseTensor，velocity 预测
        sigma_min: flow matching sigma_min 参数
    
    Returns:
        SparseTensor: 预测的 x_0
    """
    return (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * pred


def _sparse_xstart_to_pred(
    x_t: SparseTensor,
    t: float,
    x_0: SparseTensor,
    sigma_min: float,
) -> SparseTensor:
    """
    从 x0 转换为 velocity（对齐参考实现 FlowEulerSampler._xstart_to_pred）。
    
    公式: v = ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)
    
    Args:
        x_t: SparseTensor，当前 latent
        t: 时间步 [0, 1]
        x_0: SparseTensor，预测的 x_0
        sigma_min: flow matching sigma_min 参数
    
    Returns:
        SparseTensor: velocity 预测
    """
    # 使用 `/` 运算符，对齐参考实现（避免乘法 `* (1.0 / ...)` 的潜在精度差异）
    return ((1 - sigma_min) * x_t - x_0) / (sigma_min + (1 - sigma_min) * t)


def trellis2_cfg_sparse(
    cond_pred: SparseTensor,
    uncond_pred: SparseTensor,
    guidance_strength: float,
    guidance_rescale: float = 0.0,
    x_t: Optional[SparseTensor] = None,
    t: Optional[float] = None,
    sigma_min: float = 0.0,
) -> SparseTensor:
    """
    Classifier-Free Guidance (CFG) 函数，完全对齐 TRELLIS.2 参考实现。
    
    在 SparseTensor 上进行 CFG 混合，使用 SparseTensor 的运算符和 std 方法，
    确保与参考实现 ClassifierFreeGuidanceSamplerMixin._inference_model 完全一致。
    
    CFG 公式（加权平均）：
        pred = guidance_strength * cond_pred + (1 - guidance_strength) * uncond_pred
    
    CFG Rescale（对齐参考实现）：
        使用 SparseTensor.std(dim=[1], keepdim=True) 进行 std 计算。
    
    Args:
        cond_pred: SparseTensor，条件 velocity 预测
        uncond_pred: SparseTensor，无条件 velocity 预测
        guidance_strength: CFG 强度，通常 > 1.0
        guidance_rescale: CFG rescale 强度，0.0 表示不 rescale
        x_t: SparseTensor，当前 latent（rescale 需要）
        t: 当前时间步 [0, 1]（rescale 需要）
        sigma_min: flow matching sigma_min 参数（rescale 需要）
    
    Returns:
        SparseTensor: CFG 后的 velocity 预测
    """
    if guidance_strength == 1.0:
        return cond_pred  # SparseTensor
    
    if guidance_strength == 0.0:
        return uncond_pred  # SparseTensor
    
    # CFG 加权平均公式（在 SparseTensor 上进行，对齐参考实现）
    # 参考: pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
    pred = guidance_strength * cond_pred + (1 - guidance_strength) * uncond_pred  # SparseTensor
    
    # CFG Rescale（对齐参考实现 ClassifierFreeGuidanceSamplerMixin）
    if guidance_rescale > 0 and x_t is not None and t is not None:
        # 从 velocity 预测 x0（在 SparseTensor 上进行）
        x_0_pos = _sparse_pred_to_xstart(x_t, t, cond_pred, sigma_min)  # SparseTensor
        x_0_cfg = _sparse_pred_to_xstart(x_t, t, pred, sigma_min)  # SparseTensor
        
        # 使用 SparseTensor.std（继承自 VarLenTensor.std）
        # 参考实现: x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
        # 对于 SparseTensor，ndim = 2（batch + channels），所以 dim=[1]
        std_pos = x_0_pos.std(dim=[1], keepdim=True)  # (B, 1) 普通 tensor
        std_cfg = x_0_cfg.std(dim=[1], keepdim=True)  # (B, 1) 普通 tensor
        
        # Rescale（SparseTensor * 普通 tensor 会通过 __elemwise__ 正确广播）
        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)  # SparseTensor
        x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg  # SparseTensor
        
        # 转换回 velocity
        pred = _sparse_xstart_to_pred(x_t, t, x_0, sigma_min)  # SparseTensor
    
    return pred  # SparseTensor


# =====================================================================
# Debug Tracker - 极简调试跟踪器
# =====================================================================

class DebugTracker:
    """
    极简调试跟踪器，用于 rollout 过程中的中间变量跟踪。
    
    只需在循环中添加一行 log() 调用，调试完删除即可。
    自动对 Tensor 进行 detach + cpu 处理以节省显存。
    
    使用示例:
        tracker = DebugTracker()
        for t in timesteps:
            ...
            tracker.log(t=t_val, latents=latents, velocity=velocity)
            ...
        
        # 分析
        print(tracker["latents"])  # 所有步的 latents 列表
        print(tracker["velocity"])  # 所有步的 velocity 列表
        print(len(tracker))  # 总步数
    """
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    def log(self, **kwargs) -> None:
        """
        记录任意 key-value。Tensor 会自动 detach + cpu。
        
        Args:
            **kwargs: 任意键值对，如 t=0.5, latents=latents, velocity=velocity
        """
        processed = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                processed[k] = v.detach().cpu()
            else:
                processed[k] = v
        self.data.append(processed)
    
    def __getitem__(self, key: str) -> List[Any]:
        """获取所有步中某个 key 的值列表，如 tracker["latents"]"""
        return [d.get(key) for d in self.data if key in d]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def clear(self) -> None:
        """清空所有记录"""
        self.data = []
    
    def __repr__(self) -> str:
        keys = set()
        for d in self.data:
            keys.update(d.keys())
        return f"DebugTracker(steps={len(self.data)}, keys={keys})"


# =====================================================================
# Trellis2 系统组件类
# =====================================================================

@dataclass
class StageSystem:
    """
    单个阶段的系统组件。
    
    封装 Shape 阶段的 model、optimizer、renderer 和配置。
    
    属性:
        model: Flow Model
        optimizer: 优化器
        renderer: MeshRenderer（直接渲染 normal，支持梯度）
        config: StageConfig 配置
    """
    model: Any = None       # Flow Model
    optimizer: Any = None   # Optimizer
    renderer: Any = None    # MeshRenderer
    config: StageConfig = field(default_factory=StageConfig)


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
        """准备 Shape 优化器（使用 accelerator.prepare）"""
        if self.shape.optimizer is not None:
            self.shape.optimizer = accelerator.prepare(self.shape.optimizer)
        return self


# =====================================================================
# Trellis2State - Trellis2 专用状态类
# =====================================================================

@dataclass
class Trellis2State(BaseState):
    """
    Trellis2 生成过程的状态容器。
    
    扩展 BaseState 以支持 TRELLIS.2 的 Shape 阶段生成：
    - shape_slat: 几何阶段的稀疏潜变量
    - subs: 解码中间结果
    
    属性说明:
        coords (torch.Tensor): 稀疏结构坐标，形状 (N, 4)。
                               N 为总点数 (batch_size * num_points)。
                               第 0 列为 batch 索引，后 3 列为 (x, y, z) 坐标。
        
        features (Trellis2State.Features): 特征容器。
            - shape_slat (SparseTensor): Shape 阶段输出的稀疏特征
            - subs (List[SparseTensor]): Shape 解码中间结果
        
        cameras (BaseState.Cameras): 相机参数容器。
            - c2w (torch.Tensor): (B, V, 4, 4) 相机到世界变换矩阵。
            - w2c (torch.Tensor): (B, V, 4, 4) 世界到相机变换矩阵。
            - intrinsics (torch.Tensor): (B, V, 3, 3) 内参矩阵。
            
        views_conditioned (Trellis2State.ViewsConditioned): 条件信息容器（覆盖基类，支持双分辨率）。
            - image_pils (List[PIL.Image]): 输入的条件图像列表。
            - cond_512_embed (torch.Tensor): (B, S, C) 512 分辨率条件嵌入（Dense Sampling 使用）。
            - uncond_512_embed (torch.Tensor): (B, S, C) 512 分辨率无条件嵌入。
            - cond_1024_embed (torch.Tensor): (B, S, C) 1024 分辨率条件嵌入（Shape Rollout 使用）。
            - uncond_1024_embed (torch.Tensor): (B, S, C) 1024 分辨率无条件嵌入。
            
        views_generated (Trellis2State.ViewsGenerated): Shape 阶段生成结果容器。
            - shape_tensor (torch.Tensor): (B, V, H, W, C) Shape 阶段 Normal 图
            
        views_edited (BaseState.ViewsEdited): 编辑结果容器。
            - image_tensor (torch.Tensor): (B, V, C, H, W) 经过 Guidance 编辑后的图像。
            
        regularization (Trellis2State.Regularization): 正则化信息容器。
            - reg_loss: 正则化 loss（用于反向传播）
            - reg_metric: 正则化 metric（用于日志记录）
            
        guidance (Trellis2State.Guidance): Guidance 结果容器。
            - loss_ssim: SSIM loss
            - loss_lpips: LPIPS loss
            - loss_latent_mse: Latent MSE loss
    """
    
    @dataclass
    class Features:
        """特征容器。存储 Shape 阶段的稀疏特征。"""
        # Denormalized 版本（用于 decode）
        shape_slat: Any = None      # SparseTensor, Shape 阶段输出（denormalized）
        # Normalized 版本（用于作为条件输入其他模型）
        shape_slat_norm: Any = None # SparseTensor, Shape latent（normalized）
        # 解码中间结果
        subs: Any = None            # List[SparseTensor], Shape 解码中间结果
        meshes: Any = None          # List[Mesh], Shape 解码输出的 mesh
    
    @dataclass
    class Regularization:
        """正则化信息容器。存储 VSD/KL 正则化的 loss 和 metric。"""
        reg_loss: Any = None    # 正则化 loss（用于反向传播）
        reg_metric: Any = None  # 正则化 metric（用于日志记录）
    
    @dataclass
    class Guidance:
        """Guidance 结果容器。存储 FlowEdit 的各项 loss。"""
        loss_ssim: Any = None         # SSIM loss（标量张量）
        loss_lpips: Any = None        # LPIPS loss（标量张量）
        loss_latent_mse: Any = None   # Latent MSE loss（标量张量）
    
    @dataclass
    class ViewsGenerated:
        """Shape 阶段生成结果容器。"""
        shape_tensor: Any = None  # (B, V, H, W, C) Shape 阶段 Normal 图
    
    @dataclass
    class ViewsConditioned:
        """
        条件视角缓存（覆盖基类，支持双分辨率条件编码）。
        
        对齐 TRELLIS.2 参考实现：
        - Dense Sampling 始终使用 512 分辨率条件编码
        - Shape Rollout 使用对应 pipeline 分辨率的条件编码
        """
        image_pils: Any = None          # list[len=B] of PIL.Image
        paths: Any = None               # list[len=B] of str
        # 512 分辨率条件编码（Dense Sampling 始终使用）
        cond_512_embed: Any = None      # (B, S, C) 512 分辨率条件嵌入
        uncond_512_embed: Any = None    # (B, S, C) 512 分辨率无条件嵌入
        # 1024 分辨率条件编码（Shape Rollout 使用）
        cond_1024_embed: Any = None     # (B, S, C) 1024 分辨率条件嵌入
        uncond_1024_embed: Any = None   # (B, S, C) 1024 分辨率无条件嵌入
    
    # batch key -> state 属性的映射（类常量）
    _CAMERA_KEYS: ClassVar[List[str]] = ["c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions"]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]
    
    # ============== Trellis2 专用子状态容器 ==============
    features: Features = field(default_factory=Features)
    regularization: Regularization = field(default_factory=Regularization)
    guidance: Guidance = field(default_factory=Guidance)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)  # 覆盖 BaseState
    views_conditioned: ViewsConditioned = field(default_factory=ViewsConditioned)  # 覆盖 BaseState

    def attach_batch(self, batch: Dict[str, Any], pipeline: Any = None, resolution: int = 1024) -> "Trellis2State":
        """
        从数据批次中提取并挂载所有数据到 state。
        
        Args:
            batch: DataLoader 返回的批次数据，包含图像、相机参数等。
            pipeline: 必须提供，用于调用 prepare_image_conditions 从 image_pils 生成条件嵌入。
            resolution: 条件编码分辨率（512 或 1024）
        
        Returns:
            self: 支持链式调用
        """
        # ---- 1. views_conditioned（图像、路径、嵌入） ----
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])
        
        # 从 image_pils 生成双分辨率条件编码（对齐 TRELLIS.2 参考实现）
        if "image_pils" in batch and pipeline is not None:
            # 始终生成 512 分辨率的条件编码（Dense Sampling 始终需要）
            cond_512 = pipeline.prepare_image_conditions(batch["image_pils"], resolution=512)
            self.views_conditioned.cond_512_embed = cond_512["cond"]  # (B, S, C)
            self.views_conditioned.uncond_512_embed = cond_512["neg_cond"]  # (B, S, C)
            
            # 生成目标分辨率的条件编码（用于 Shape Rollout）
            if resolution == 512:
                # 复用 512 的结果
                self.views_conditioned.cond_1024_embed = self.views_conditioned.cond_512_embed
                self.views_conditioned.uncond_1024_embed = self.views_conditioned.uncond_512_embed
            else:
                # 生成 1024 分辨率
                cond_1024 = pipeline.prepare_image_conditions(batch["image_pils"], resolution=resolution)
                self.views_conditioned.cond_1024_embed = cond_1024["cond"]  # (B, S, C)
                self.views_conditioned.uncond_1024_embed = cond_1024["neg_cond"]  # (B, S, C)
        
        # ---- 2. 指导信号 (Guidance 数据) ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]
        
        # ---- 3. 相机参数 ----
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])
        
        return self
    
    def extract_embeddings(self, resolution: int = 1024) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        提取指定分辨率的条件和无条件嵌入（覆盖基类方法）。
        
        Args:
            resolution: 条件编码分辨率，512 或 1024
        
        Returns:
            tuple: (cond_embed, uncond_embed)
        """
        if resolution == 512:
            cond = self.views_conditioned.cond_512_embed
            uncond = self.views_conditioned.uncond_512_embed
        else:
            cond = self.views_conditioned.cond_1024_embed
            uncond = self.views_conditioned.uncond_1024_embed
        
        if cond is None:
            raise ValueError(f"views_conditioned.cond_{resolution}_embed 未设置，请先调用 attach_batch")
        
        return cond, uncond

    def attach_guidance_result(self, guidance_result: Any) -> "Trellis2State":
        """
        将 GuidanceResult 挂载到 state。
        
        Args:
            guidance_result: GuidanceResult 对象，包含编辑后图像和各项 loss。
        
        Returns:
            self: 支持链式调用
        """
        # Loss 挂载到 guidance
        self.guidance.loss_ssim = guidance_result.loss_ssim
        self.guidance.loss_lpips = guidance_result.loss_lpips
        self.guidance.loss_latent_mse = guidance_result.loss_latent_mse
        # 编辑后图像挂载到 views_edited
        self.views_edited.image_tensor = guidance_result.edited_imgs
        return self

    def simplify_meshes(self, max_faces: int = 16777216) -> "Trellis2State":
        """
        简化 state 中的 meshes，避免 nvdiffrast 的面片数量限制。
        
        nvdiffrast 的 CUDA 光栅化器最多支持 2^24 = 16,777,216 个三角面片。
        当 mesh 面片数超过限制时，调用 mesh.simplify() 进行简化。
        
        注意：simplify() 是不可微的操作，使用 torch.no_grad() 包裹。
        
        Args:
            max_faces: 最大面片数量，默认 16777216（nvdiffrast 限制）
        
        Returns:
            self: 支持链式调用
        """
        if self.features.meshes is None:
            return self
        
        for mesh in self.features.meshes:
            if mesh.faces.shape[0] > max_faces:
                with torch.no_grad():
                    mesh.simplify(max_faces)
        
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
    构建完整的 Trellis2 Shape 系统。
    
    Args:
        cfg: 完整配置对象
        accelerator: Accelerate 分布式训练加速器
        guidance_factory: Guidance 工厂函数
    
    Returns:
        Trellis2System: 包含所有组件的系统实例
    """
    from edit4shape.generators.trellis2.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.generators.trellis2.training_adpter import (
        get_stage_config, set_stage_trainable, build_optimizer_for_stage, register_sparse_linear_with_peft
    )
    
    pipeline_type = cfg.pipeline_type
    device = str(accelerator.device)
    
    # ---- 1. Pipeline ----
    pipeline = build_pipeline_from_reference(cfg, accelerator)
    
    # ---- 2. Renderer 配置 ----
    render_opts = {
        "resolution": cfg.renderer.resolution,
        "ssaa": cfg.renderer.ssaa,
        "near": cfg.renderer.near,
        "far": cfg.renderer.far,
        "chunk_size": 8000000,  # 分块渲染：800万面片/chunk，避免 nvdiffrast 2^24 限制，保持可微
    }
    
    # ---- 3. 获取 Shape 阶段配置 ----
    shape_config = get_stage_config(pipeline_type, "shape")
    
    # ---- 4. 构建 StageSystem（使用可微 MeshRenderer） ----
    # Shape 阶段：MeshRenderer 渲染 normal（伪 GT Mesh 方案，可微）
    shape_renderer = MeshRenderer(rendering_options=render_opts)
    shape_stage = StageSystem(
        config=shape_config,
        renderer=shape_renderer,
    )
    
    # ---- 5. 训练模式：设置 model 和 optimizer ----
    guidance = None
    if not cfg.eval_only:
        guidance = guidance_factory(cfg, train_device=accelerator.device)
        
        register_sparse_linear_with_peft()
        set_stage_trainable(pipeline, pipeline_type, ["shape"])
        
        # 获取模型
        shape_stage.model = pipeline.get_flow_model(shape_config.model_stage, shape_config.flow_resolution)
        
        # 创建优化器
        optimizer_shape = build_optimizer_for_stage(
            pipeline, pipeline_type, "shape", cfg.train.optimizer
        )
        shape_stage.optimizer = optimizer_shape
        
        # 启用 Decoder Gradient Checkpointing
        pipeline._set_decoder_checkpointing("shape_slat_decoder", enable=True)
        print("[Trellis2] 已启用 shape_slat_decoder 的 gradient checkpointing")

    return Trellis2System(
        pipeline=pipeline,
        shape=shape_stage,
        guidance=guidance,
    )


def build_dataloaders(cfg: ml_collections.ConfigDict, accelerator: Accelerator) -> Tuple[DataLoader, DataLoader]:
    """
    构造训练和评估的 DataLoader。
    
    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器
    
    Returns:
        tuple: (train_loader, eval_loader)
    """
    from edit4shape.datasets.trellis import TrellisCameraTrainConfig, TrellisCameraEvalConfig, AdaptiveDistanceConfig
    
    # ---- 构建训练相机配置 ----
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=AdaptiveDistanceConfig(
            enabled=cfg.data.train.adaptive_distance.enabled,
            fill_ratio=cfg.data.train.adaptive_distance.fill_ratio,
        ),
    )
    
    # ---- 构建评估相机配置 ----
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw=cfg.data.eval.yaw,
        pitch=cfg.data.eval.pitch,
        r=cfg.data.eval.r,
        fov=cfg.data.eval.fov,
        adaptive_distance=AdaptiveDistanceConfig(
            enabled=cfg.data.eval.adaptive_distance.enabled,
            fill_ratio=cfg.data.eval.adaptive_distance.fill_ratio,
        ),
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
# Rollout 辅助函数
# =====================================================================

def _predict_velocity(
    pipeline: Any,
    x_t: SparseTensor,
    t: float,
    cond_emb: torch.Tensor,
    stage: Stage,
    resolution: int,
    shape_cond: Optional[SparseTensor] = None,
) -> SparseTensor:
    """
    Velocity 预测（用于 checkpoint 包裹）。
    
    返回 SparseTensor 以保持完整的 SparseTensor 流程，对齐参考实现。
    
    Args:
        pipeline: Trellis2RefAdapter
        x_t: SparseTensor，当前 latent
        t: 时间步标量，范围 [0, 1]
        cond_emb: (B, S, C) 条件嵌入
        stage: "shape"
        resolution: 512 或 1024
        shape_cond: SparseTensor，shape 条件（已归一化）
    
    Returns:
        SparseTensor: velocity 预测（保持完整的 SparseTensor 类型）
    """
    # t 已经是 0-1 范围，直接传给 sampling_step（内部会乘 1000）
    out = pipeline.sampling_step(
        x_t, t, cond_emb, stage, resolution, shape_cond=shape_cond
    )  # SparseTensor
    
    return out  # SparseTensor


def _compute_regularization(
    x0_student: torch.Tensor,
    x0_teacher: torch.Tensor,
    latents: torch.Tensor,
    t_norm: float,
    reg_type: str,
    weight_mode: str = "uniform",
) -> Tuple[torch.Tensor, float]:
    """
    计算正则化 loss（VSD / KL）。
    
    Args:
        x0_student: (N, C) 学生模型预测的 x0
        x0_teacher: (N, C) 教师模型预测的 x0（无梯度）
        latents: (N, C) 当前步的 x_t
        t_norm: 归一化时间步 (0~1)
        reg_type: "vsd" | "kl"
        weight_mode: "uniform" | "t" | "ada"
    
    Returns:
        (loss, metric): loss 用于反向传播，metric 用于日志
    """
    diff = x0_student - x0_teacher  # (N, C)
    
    if weight_mode == "t":
        diff = t_norm * diff  # (N, C)
    elif weight_mode == "ada":
        diff = diff / (x0_teacher.abs().mean() + 0.01).detach()  # (N, C)
    
    if reg_type == "vsd":
        metric = 0.5 * (diff ** 2).mean().item()
        loss = SpecifyGradient.apply(latents, diff)
    elif reg_type == "kl":
        var = t_norm ** 2 + 1e-3
        loss = (0.5 * diff ** 2 / var).mean()
        metric = loss.item()
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")
    
    return loss, metric


# =====================================================================
# Rollout - Shape 阶段
# =====================================================================

def rollout_shape(
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    system: Trellis2System,
    device: torch.device,
    resolution: int = 1024,
    generator: Optional[torch.Generator] = None,
    is_training: bool = False,
) -> DebugTracker:
    """
    Shape 阶段去噪采样。
    
    Args:
        state: Trellis2State，包含条件编码、坐标等
        cfg: 配置对象
        system: 系统组件
        device: 运行设备
        resolution: 模型分辨率（512 或 1024）
        generator: 随机数生成器
        is_training: 是否为训练模式
    
    Returns:
        DebugTracker: 包含每步中间变量的跟踪器
    
    Side Effects:
        - state.features.shape_slat: 挂载反归一化后的 SparseTensor
        - state.regularization: 挂载 reg_loss 和 reg_metric
    """
    tracker = DebugTracker()
    pipeline = system.pipeline
    stage = "shape"
    
    # ---- 1. 获取采样参数 ----
    sampler_params = pipeline.get_sampler_params(stage)
    steps = int(sampler_params["steps"])
    cfg_strength = float(sampler_params["guidance_strength"])
    cfg_rescale = float(sampler_params["guidance_rescale"])
    cfg_min, cfg_max = pipeline.get_cfg_interval(stage)
    sigma_min = pipeline.pipe.shape_slat_sampler.sigma_min  # 从 sampler 获取 sigma_min
    
    # ---- 2. 初始化 ----
    # Shape Rollout 使用 flow_resolution 对应的条件编码（对齐参考实现）
    cond_emb, uncond_emb = state.extract_embeddings(resolution=resolution)
    cond_emb = cond_emb.to(device)  # (B, S, C)
    uncond_emb = uncond_emb.to(device) if uncond_emb is not None else None  # (B, S, C)
    
    assert state.coords is not None, "state.coords 缺失"
    
    # ★ generator 处理：
    # - None: 使用全局种子（与参考实现一致，适用于 eval）
    # - 显式传入: 使用传入的 Generator（适用于可控的训练）
    # 注意：不自动创建 Generator，保持与参考实现的行为一致
    # ★ 使用 SparseTensor 贯穿整个流程，对齐参考实现
    x_t = pipeline.init_latents(
        coords=state.coords,
        stage=stage,
        resolution=resolution,
        generator=generator,  # 可以是 None，使用全局种子
    )  # SparseTensor
    
    # ---- 3. Scheduler 配置 ----
    scheduler = pipeline.scheduler(stage)
    scheduler.set_timesteps(steps, device=device)
    
    # ---- 4. 正则化配置 ----
    reg_type = cfg.reg.type
    weight_mode = cfg.reg.weight_mode
    reg_enabled = reg_type != "none" and is_training
    
    reg_loss_sum = 0.0
    reg_metric_sum = 0.0
    
    # ---- 5. 去噪循环 ----
    # 使用基于索引的 API 确保时间步精度与参考实现完全一致
    step_indices = scheduler.get_timesteps_for_loop()  # [0, 1, ..., steps-1]
    steps_iter = tqdm(step_indices, desc="Shape Rollout", leave=False,
                      disable=not is_training or not Accelerator().is_main_process)
    
    for step_idx in steps_iter:
        # 使用精确的 numpy float64 时间步值（对齐参考实现）
        t_val = scheduler.get_precise_t(step_idx)  # float64 精度
        t_norm = t_val  # 直接使用，scheduler.timesteps 已经是 0-1 范围
        use_cfg = cfg_min <= t_norm <= cfg_max
        
        # ---- cond 预测（使用 SparseTensor 流程） ----
        if is_training:
            cond_pred = checkpoint(
                _predict_velocity, pipeline, x_t,
                t_val, cond_emb, stage, resolution, None,
                use_reentrant=False
            )  # SparseTensor
        else:
            with torch.no_grad():
                cond_pred = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, None
                )  # SparseTensor
        
        # ---- uncond 预测 + CFG 混合（在 SparseTensor 上进行） ----
        if use_cfg and uncond_emb is not None:
            with torch.no_grad():
                uncond_pred = _predict_velocity(
                    pipeline, x_t, t_val, uncond_emb,
                    stage, resolution, None
                )  # SparseTensor
            velocity = trellis2_cfg_sparse(
                cond_pred, uncond_pred, cfg_strength,
                guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                sigma_min=sigma_min
            )  # SparseTensor
        else:
            velocity = cond_pred  # SparseTensor
        
        # ---- 正则化（VSD / KL）----
        if reg_enabled:
            with pipeline.disable_lora_context(stage, resolution), torch.no_grad():
                teacher_cond = _predict_velocity(
                    pipeline, x_t, t_val, cond_emb,
                    stage, resolution, None
                )  # SparseTensor
                if use_cfg and uncond_emb is not None:
                    teacher_uncond = _predict_velocity(
                        pipeline, x_t, t_val, uncond_emb,
                        stage, resolution, None
                    )  # SparseTensor
                    teacher_vel = trellis2_cfg_sparse(
                        teacher_cond, teacher_uncond, cfg_strength,
                        guidance_rescale=cfg_rescale, x_t=x_t, t=t_val,
                        sigma_min=sigma_min
                    )  # SparseTensor
                else:
                    teacher_vel = teacher_cond  # SparseTensor
            
            # 正则化在 feats 上计算
            # 使用正确的 x0 公式（对齐参考实现 FlowEulerSampler._pred_to_xstart）：
            # x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
            coeff = sigma_min + (1 - sigma_min) * t_val  # scalar
            x0_stu = (1 - sigma_min) * x_t.feats - coeff * velocity.feats  # (N, C)
            x0_tea = (1 - sigma_min) * x_t.feats - coeff * teacher_vel.feats  # (N, C)
            
            reg_loss, reg_metric = _compute_regularization(
                x0_stu, x0_tea, x_t.feats, t_norm,
                reg_type=reg_type, weight_mode=weight_mode
            )
            reg_loss_sum = reg_loss_sum + reg_loss
            reg_metric_sum = reg_metric_sum + reg_metric
        
        # ---- Scheduler 步进（使用 SparseTensor 流程） ----
        # scheduler.step_by_index 直接接收 SparseTensor，返回 SparseTensor
        x_t = scheduler.step_by_index(velocity, step_idx, x_t).prev_sample  # SparseTensor
        
        # ---- 记录调试信息 ----
        tracker.log(
            t=t_val,
            latents=x_t.feats,  # (N, C)
            velocity=velocity.feats,  # (N, C)
            cond_pred=cond_pred.feats,  # (N, C)
            uncond_pred=uncond_pred.feats if use_cfg and uncond_emb is not None else None,  # (N, C)
        )
    
    # ---- 6. 反归一化 ----
    # x_t 已经是 SparseTensor，直接使用
    shape_slat_normalized = x_t  # SparseTensor
    shape_slat = pipeline.denormalize(shape_slat_normalized, stage)  # SparseTensor
    
    # ---- 7. 挂载到 state（同时保存 normalized 和 denormalized 版本）----
    state.features.shape_slat = shape_slat  # denormalized，保留梯度用于 decode
    
    # shape_slat_norm 备用，直接 detach 切断依赖
    norm_detached = shape_slat_normalized.detach()
    norm_detached.clear_spatial_cache()
    state.features.shape_slat_norm = norm_detached
    
    num_steps = max(1, len(step_indices))
    state.regularization.reg_loss = reg_loss_sum / num_steps if reg_enabled else None
    state.regularization.reg_metric = reg_metric_sum / num_steps if reg_enabled else None
    
    return tracker


# =====================================================================
# 渲染工具函数 - Normal 渲染
# =====================================================================

def decode_and_render_normal(
    shape_slat: SparseTensor,
    cameras: Any,  # Trellis2State.Cameras
    pipeline: Any,
    renderer: Any,  # MeshRenderer（nvdiffrast，支持梯度）
    device: torch.device,
    resolution: int = 1024,
    use_checkpointing: bool = True,  # 使用 gradient checkpointing 减少显存
) -> Dict[str, Any]:
    """
    解码 shape_slat 为 Mesh 并使用 MeshRenderer 渲染 Normal 图。
    
    使用 nvdiffrast 可微渲染器直接渲染 normal，支持梯度反向传播。
    只调用 decode_shape（Normal 渲染不需要纹理信息）。
    支持 gradient checkpointing 以减少显存使用。
    
    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器
        pipeline: Trellis2RefAdapter
        renderer: MeshRenderer（nvdiffrast）
        device: 运行设备
        resolution: 输出分辨率
        use_checkpointing: 是否使用 gradient checkpointing（默认 True）
    
    Returns:
        dict: {
            "color": (B, V, H, W, 3) Normal 图
            "subs": List[SparseTensor]
            "meshes": List[Mesh]
        }
    """
    
    # ---- 解码 Shape（Normal 渲染只需要 Mesh） ----
    # 注意：decoder 的 gradient checkpointing 在 build_system 中已全局启用
    shape_result = pipeline.decode_shape(shape_slat, resolution)
    meshes = shape_result["meshes"]  # List[Mesh]
    subs = shape_result["subs"]  # List[SparseTensor]
    
    # ---- 获取相机参数 ----
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # ---- 渲染辅助函数 ----
    # 中性 Normal 背景（朝向相机，RGB = [0.5, 0.5, 1.0]）
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)  # (3,)
    
    def _render_normal(mesh, ext, intr):
        out = renderer.render(mesh, ext, intr, return_types=["normal", "mask"])
        normal = out["normal"].permute(1, 2, 0)  # (H, W, 3)
        mask = out["mask"].unsqueeze(-1)  # (H, W, 1)
        # 使用 mask 混合背景颜色
        normal = normal * mask + bg_color * (1 - mask)  # (H, W, 3)
        return normal
    
    # ---- 使用 MeshRenderer 渲染 normal（nvdiffrast，支持梯度） ----
    all_normals: List[torch.Tensor] = []
    
    for i, mesh in enumerate(meshes):
        view_normals: List[torch.Tensor] = []
        mesh = mesh.to(device)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            if use_checkpointing:
                normal = checkpoint(_render_normal, mesh, ext_iv, intr_iv, use_reentrant=False)
            else:
                normal = _render_normal(mesh, ext_iv, intr_iv)
            
            view_normals.append(normal)  # (H, W, 3)
        
        stacked = torch.stack(view_normals, dim=0)  # (V, H, W, 3)
        all_normals.append(stacked)
    
    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
    
    return {
        "color": normals,   # (B, V, H, W, 3) Normal 图
        "subs": subs,       # List[SparseTensor]
        "meshes": meshes,   # List[Mesh]
    }


def decode_and_render_normal_voxel(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,
    device: torch.device,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """
    使用可微 VoxelRenderer 渲染 Normal 图（绕过 Mesh 提取）。
    
    流程:
        1. 调用 FDG Decoder 父类获取原始特征 h.feats (N, 7)
        2. 构建 VoxelProxy（position 和 opacity 可微）
        3. 渲染深度 → depth_to_normal → Normal
    
    梯度流: Loss → Normal → Depth → VoxelProxy → h.feats → Decoder → Flow Model
    
    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: DiffVoxelRenderer
        device: 运行设备
        resolution: 输出分辨率
    
    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": None}
    """
    from edit4shape.renderers.ovoxel_trellis2 import VoxelProxy
    
    # 调用 Decoder 父类的 forward（绕过 Mesh 提取）
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    parent_class = decoder.__class__.__bases__[0]  # SparseUnetVaeDecoder
    h, subs = parent_class.forward(decoder, shape_slat, return_subs=True)  # h.feats: (N, 7)
    
    # 构建 VoxelProxy（从原始特征）
    voxel_proxy = VoxelProxy.from_fdg_decoder(h.feats, h.coords, resolution, decoder.voxel_margin)
    
    # 批量渲染
    extr = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    normals = renderer.render_batch(voxel_proxy, extr, intr).normal  # (B, V, H, W, 3)
    
    return {"color": normals, "subs": list(subs), "meshes": None}


def decode_and_render_normal_mesh_pseudo_gt(
    shape_slat: SparseTensor,
    cameras: Any,
    pipeline: Any,
    renderer: Any,  # MeshRenderer
    device: torch.device,
    resolution: int = 1024,
    use_checkpointing: bool = True,
) -> Dict[str, Any]:
    """
    使用"伪 GT intersected"方案渲染 Normal（可微 Mesh 路径）。
    
    核心思路：
    1. 用模型预测的 h.feats[3:6] > 0 作为 intersected（detach，固定拓扑）
    2. dual_vertices (h.feats[0:3]) 和 quad_lerp (h.feats[6:7]) 参与梯度
    3. 调用 flexible_dual_grid_to_mesh(train=True) 生成可微 Mesh
    4. 使用 MeshRenderer 渲染 Normal
    
    可训练特征：4/7 通道（57%）
    - ✅ h.feats[0:3] (dual_vertices) → sigmoid → mesh 顶点位置
    - ❌ h.feats[3:6] (intersected) → 硬阈值 + detach → 拓扑（不可训练）
    - ✅ h.feats[6:7] (quad_lerp) → softplus → 三角化权重
    
    梯度流：
    Loss → Normal → mesh_vertices → sigmoid(h.feats[0:3]) + softplus(h.feats[6:7]) → Decoder
    
    Args:
        shape_slat: SparseTensor，shape 特征（已反归一化）
        cameras: 相机参数容器，包含 w2c 和 intrinsics
        pipeline: Trellis2RefAdapter
        renderer: MeshRenderer（nvdiffrast）
        device: 运行设备
        resolution: 输出分辨率
        use_checkpointing: 是否使用 gradient checkpointing
    
    Returns:
        dict: {"color": (B, V, H, W, 3), "subs": List[SparseTensor], "meshes": List[Mesh]}
    """
    from o_voxel.convert import flexible_dual_grid_to_mesh
    import torch.nn.functional as F
    
    decoder = pipeline.pipe.models['shape_slat_decoder']
    decoder.set_resolution(resolution)
    
    # 调用父类 forward 获取原始特征
    parent_class = decoder.__class__.__bases__[0]  # SparseUnetVaeDecoder
    h, subs = parent_class.forward(decoder, shape_slat, return_subs=True)  # h.feats: (N, 7)
    
    voxel_margin = decoder.voxel_margin
    
    # ========== 分解 h.feats ==========
    # 1. dual_vertices: sigmoid 变换后的顶点偏移（可微）
    vertices_sp = h.replace(
        (1 + 2 * voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - voxel_margin
    )
    
    # 2. intersected: 硬阈值 + detach（伪 GT，不可微）
    # 这是关键：用模型自己的预测作为固定拓扑
    pseudo_gt_intersected = h.replace(
        (h.feats[..., 3:6] > 0).detach()  # detach 切断梯度
    )
    
    # 3. quad_lerp: softplus 变换（可微）
    quad_lerp_sp = h.replace(F.softplus(h.feats[..., 6:7]))
    
    # ========== 为每个 batch 构建 Mesh ==========
    meshes = []
    for v, i, q in zip(vertices_sp, pseudo_gt_intersected, quad_lerp_sp):
        vertices, faces = flexible_dual_grid_to_mesh(
            v.coords[:, 1:],  # (N, 3) voxel 坐标
            v.feats,          # (N, 3) dual_vertices（可微）
            i.feats,          # (N, 3) intersected（detached bool）
            q.feats,          # (N, 1) quad_lerp（可微）
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=resolution,
            train=True,       # 启用可微路径
        )
        meshes.append(Mesh(vertices, faces))
    
    # ========== 渲染 Normal ==========
    extr_all = cameras.w2c.to(device)  # (B, V, 4, 4)
    intr_all = cameras.intrinsics.to(device)  # (B, V, 3, 3)
    batch_size, num_views = extr_all.shape[:2]
    
    # 中性 Normal 背景（朝向相机，RGB = [0.5, 0.5, 1.0]）
    bg_color = torch.tensor([0.5, 0.5, 1.0], device=device)
    
    def _render_normal(mesh, ext, intr):
        out = renderer.render(mesh, ext, intr, return_types=["normal", "mask"])
        normal = out["normal"].permute(1, 2, 0)  # (H, W, 3)
        mask = out["mask"].unsqueeze(-1)  # (H, W, 1)
        return normal * mask + bg_color * (1 - mask)
    
    all_normals = []
    for i, mesh in enumerate(meshes):
        view_normals = []
        mesh = mesh.to(device)
        
        for v in range(num_views):
            ext_iv = extr_all[i, v]  # (4, 4)
            intr_iv = intr_all[i, v]  # (3, 3)
            
            if use_checkpointing:
                normal = checkpoint(_render_normal, mesh, ext_iv, intr_iv, use_reentrant=False)
            else:
                normal = _render_normal(mesh, ext_iv, intr_iv)
            
            view_normals.append(normal)
        
        all_normals.append(torch.stack(view_normals, dim=0))
    
    normals = torch.stack(all_normals, dim=0)  # (B, V, H, W, 3)
    
    return {"color": normals, "subs": list(subs), "meshes": meshes}


# =====================================================================
# 前向传播 - Shape 阶段
# =====================================================================

def trellis2_shape_forward(
    system: Trellis2System,
    state: Trellis2State,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
    global_step: int,
    is_training: bool = True,
) -> Dict[str, Any]:
    """
    Shape 阶段前向传播: Dense Sampling → Shape Rollout → Mesh Normal 渲染
    
    使用"伪 GT intersected"方案渲染 Normal：
    1. 模型预测的 h.feats[3:6] > 0 作为 intersected（detach，固定拓扑）
    2. dual_vertices (h.feats[0:3]) 和 quad_lerp (h.feats[6:7]) 参与梯度
    3. 调用 flexible_dual_grid_to_mesh(train=True) 生成可微 Mesh
    4. 使用 MeshRenderer 渲染 Normal
    
    梯度流：
    Loss → Normal → mesh_vertices → sigmoid(h.feats[0:3]) + softplus(h.feats[6:7]) → Decoder
    
    Args:
        system: 系统组件
        state: Trellis2State 状态对象
        cfg: 配置对象
        device: 运行设备
        global_step: 全局步数
        is_training: 是否为训练模式
    
    Returns:
        render_out: 渲染输出字典，包含：
            - "color": (B, V, H, W, 3) Normal 图
            - "subs": List[SparseTensor]
            - "meshes": List[Mesh]（伪 GT Mesh 方案生成的可微 Mesh）
    
    Side Effects:
        - state.coords: 挂载稀疏坐标
        - state.features.shape_slat: 挂载 shape latent
        - state.features.subs: 挂载解码中间结果
        - state.regularization: 挂载 reg_loss 和 reg_metric
        - state.views_generated.shape_tensor: 挂载 Normal 渲染图像
    """
    pipeline = system.pipeline
    stage_config = pipeline.get_stage_config("shape")
    
    # Dense Sampling
    # Dense Sampling - 始终使用 512 分辨率的条件编码（对齐 TRELLIS.2 参考实现）
    ss_params = pipeline.get_ss_params()
    with torch.no_grad():
        cond_dict = {
            "cond": state.views_conditioned.cond_512_embed,      # 始终用 512
            "neg_cond": state.views_conditioned.uncond_512_embed  # 始终用 512
        }
        coords = pipeline.dense_sampling(
            cond_dict, steps=int(ss_params["steps"]), resolution=stage_config["ss_resolution"]
        )  # (N, 4)
    state.coords = coords
    
    # Shape Rollout
    # eval 时使用全局种子（对齐参考实现），train 时使用独立 Generator
    generator = None if not is_training else torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    rollout_shape(
        state, cfg, system, device,
        resolution=stage_config["flow_resolution"],
        generator=generator,
        is_training=is_training,
    )
    
    # 解码 + Normal 渲染（使用伪 GT Mesh 方案）
    render_out = decode_and_render_normal_mesh_pseudo_gt(
        state.features.shape_slat,
        state.cameras,
        pipeline,
        system.shape.renderer,
        device,
        resolution=pipeline.target_resolution,
    )
    
    # 挂载结果
    state.features.subs = render_out["subs"]
    state.features.meshes = render_out["meshes"]  # 伪 GT Mesh 方案生成 Mesh
    state.views_generated.shape_tensor = render_out["color"]  # (B, V, H, W, C) Normal 图
    
    # 简化超大 mesh，避免 nvdiffrast 面片数量限制
    state.simplify_meshes()
    
    return render_out


# =====================================================================
# 评估
# =====================================================================

@torch.no_grad()
def evaluate(
    system: Trellis2System,
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
    3. 执行 Shape Rollout 生成特征
    4. 解码为 Mesh
    5. 渲染 Normal 图并保存
    6. 导出 mesh 文件
    
    输出目录结构：
    visuals_eval_dir/
    └── epoch_{N}/
        ├── sample_name_1/
        │   ├── normal.png     # 渲染的法线图
        │   └── mesh.obj       # 导出的网格文件
        └── sample_name_2/
            └── ...
    
    Args:
        system: 系统组件
        cfg: 配置对象
        accelerator: Accelerator
        epoch: 当前 epoch
        global_step: 全局步数
        eval_loader: 评估数据加载器
        visuals_eval_dir: 输出目录
    
    Returns:
        dict: 评估日志
    """
    if eval_loader is None:
        return {}
    
    pipeline = system.pipeline
    visual_io = Trellis2VisualIO(visuals_eval_dir, target_h=cfg.renderer.resolution)
    
    # 获取需要设置为 eval 模式的模型
    models_to_eval = [
        system.shape.model,
        pipeline.pipe.models['shape_slat_decoder'],
    ]
    
    # 过滤 None（eval_only 模式下 model 可能为 None）
    models_to_eval = [m for m in models_to_eval if m is not None]
    
    with EvalModeGuard(*models_to_eval):
        for batch_idx, batch in enumerate(eval_loader):
            state = Trellis2State()
            state.attach_batch(batch, pipeline=pipeline, resolution=system.shape.config.cond_resolution)
            
            # Shape Forward (渲染 Normal)
            render_out = trellis2_shape_forward(
                system, state, cfg, accelerator.device, global_step,
                is_training=False
            )
            
            if accelerator.is_main_process:
                # 伪 GT Mesh 方案生成 Mesh，可以导出
                visual_io.save_batch_eval(
                    state=state,
                    epoch=epoch,
                    render_out=render_out,
                    pipeline=pipeline,
                    export_mesh=False,  # 默认不导出 Mesh
                )
    
    return {"eval_done": 1.0}


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
    # Step 2: 初始化 Accelerator
    # 配置混合精度训练和梯度累积
    # =====================================================
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )
    
    # =====================================================
    # Step 3: 创建运行目录
    # =====================================================
    run_root, logs_dir, visuals_train_dir, visuals_eval_dir = build_run_paths(cfg, accelerator)
    vis_freq = int(cfg.freq.save.visual)
    visual_io = Trellis2VisualIO(visuals_train_dir, target_h=cfg.renderer.resolution, vis_freq=vis_freq)
    
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
    ckpt_io = CheckpointIO(accelerator, ckpt_root)
    start_epoch = ckpt_io.load(cfg.checkpoint, mode="train")
    global_step = int(ckpt_io.start_global_step)
    
    # =====================================================
    # Step 7: 评估模式
    # =====================================================
    if cfg.eval_only:
        eval_log = evaluate(
            system, cfg, accelerator,
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
    
    def _compute_loss_and_backward(state: Trellis2State) -> Dict[str, Any]:
        """计算 loss 并反向传播。返回日志字典供 logger 使用。"""
        losses = LossDict(device=accelerator.device)
        guidance_weights = system.guidance.get_loss_weights()
        
        losses.add("ssim", state.guidance.loss_ssim, weight=guidance_weights["ssim"])
        losses.add("lpips", state.guidance.loss_lpips, weight=guidance_weights["lpips"])
        losses.add("latent_mse", state.guidance.loss_latent_mse, weight=guidance_weights["latent_mse"])
        losses.add("reg", state.regularization.reg_loss, weight=cfg.train.loss.reg)
        
        # ---- 反向传播 ----
        total_loss = losses.total()
        accelerator.backward(total_loss)
        
        # ---- 构建日志 ----
        logs = losses.to_logs()  # {"loss/ssim": ..., "loss/total": ...}
        if state.regularization.reg_metric is not None:
            logs["loss/reg_metric"] = state.regularization.reg_metric
        return logs
    
    for epoch in range(start_epoch, int(cfg.num_epochs)):
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            global_step += 1
            batch_size = len(batch['image_pils'])
            
            state = Trellis2State()
            state.attach_batch(batch, pipeline=system.pipeline, resolution=system.shape.config.cond_resolution)
            
            # ============================================
            # Shape Forward → Backward → Update
            # ============================================
            with accelerator.accumulate(system.shape.model):
                with TrainModeGuard(system.shape.model):
                    shape_render_out = trellis2_shape_forward(
                            system, state, cfg, accelerator.device, global_step,
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
            # Logging
            # ============================================
            shape_logger.log_step(shape_log, batch_size, global_step, epoch)
            
            # 保存可视化（使用 Normal 渲染结果）
            if accelerator.is_main_process and (global_step % visual_io.vis_freq == 0):
                visual_io.save_batch_train(state=state, epoch=epoch, step=global_step)
        
        # ============================================
        # Epoch 结束后：周期性评估和检查点保存
        # ============================================
        if cfg.freq.eval and (epoch % int(cfg.freq.eval) == 0):
            eval_log = evaluate(
                system, cfg, accelerator,
                epoch=epoch,
                global_step=global_step,
                eval_loader=eval_loader,
                visuals_eval_dir=visuals_eval_dir,
            )
            eval_logger = MetricLogger(accelerator, logs_dir / "test.csv")
            eval_logger.accumulate(eval_log, 1)
            eval_logger.flush(global_step, epoch)
        
        if cfg.freq.save.ckpt and (epoch % int(cfg.freq.save.ckpt) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    app.run(main)


# =====================================================================
# 模块导出列表（供 trellis2_tex.py、trellis2_shape+tex.py 等模块复用）
# =====================================================================
__all__ = [
    # CFG 函数
    "_sparse_pred_to_xstart",
    "_sparse_xstart_to_pred",
    "trellis2_cfg_sparse",
    # 调试工具
    "DebugTracker",
    # 共用组件类
    "StageSystem",
    "Trellis2State",
    # Rollout 辅助函数
    "_predict_velocity",
    "_compute_regularization",
    # Shape 阶段核心函数
    "rollout_shape",
    "decode_and_render_normal",
    "decode_and_render_normal_mesh_pseudo_gt",  # 伪 GT Mesh 方案
    "trellis2_shape_forward",
    # 数据加载
    "build_dataloaders",
]
