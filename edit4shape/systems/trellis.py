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


def scheduler_step_at_index(scheduler: Any, t: torch.Tensor, latents: torch.Tensor, noise_pred: torch.Tensor) -> Any:
    """
    扩散调度器的安全单步执行函数。
    
    扩散模型的调度器 (scheduler) 负责管理去噪过程中的时间步进，
    将噪声预测转换为下一时刻的潜变量。此函数兼容不同类型的调度器实现。
    
    Args:
        scheduler: 扩散调度器实例（如 FlowMatchEulerDiscreteScheduler）
        t: 当前时间步，标量张量 ()
        latents: 当前潜变量，形状 (B,T,C) 或 SparseTensor
        noise_pred: 模型预测的噪声/速度场，形状与 latents 相同

    Returns:
        SchedulerOutput 对象，包含:
            - prev_sample: 下一时刻的潜变量
            - pred_original_sample: 预测的原始样本（部分调度器支持）
    """
    # 某些调度器需要先设置当前步骤索引
    if hasattr(scheduler, "index_for_timestep"):
        _ = scheduler.index_for_timestep(t, scheduler.timesteps)  # () - 设置内部状态
    return scheduler.step(noise_pred, t, latents)  # (obj: prev_sample/pred_original_sample)


def stage2_rollout_step(
    pipeline: Any,
    scheduler: Any,
    latents: torch.Tensor,
    coords: torch.Tensor,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: Optional[torch.Tensor],
    step_index: int,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stage 2 (SLAT 特征采样) 的单步 rollout。
    
    执行扩散模型的单步去噪过程，包括条件/无条件推理和 CFG 混合。
    此函数封装了完整的单步流程，便于在训练/推理中复用。
    
    Args:
        pipeline: 生成 pipeline，提供 denoise 方法
        scheduler: 扩散调度器
        latents: 当前时刻的潜变量 (B,T,C)
        coords: 稀疏坐标 (B,T,4)，4 维为 [batch_idx, x, y, z]
        cond_embeddings: 条件嵌入 (B,S,C)，S=序列长度
        uncond_embeddings: 无条件嵌入 (B,S,C)，可为 None
        step_index: 当前步骤索引 (0, 1, 2, ...)
        cfg: 配置对象，包含 guidance_scale 和 uncond_mode_rollout

    Returns:
        tuple: (next_feats, velocity_preds, final_feats_ft)
            - next_feats: 下一时刻的潜变量 (B,T,C)
            - velocity_preds: 速度场预测 (B,T,C)
            - final_feats_ft: 预测的最终特征 (B,T,C)
    """
    batch_size = latents.shape[0]  # 标量 ()，获取 batch 大小
    t = scheduler.timesteps[step_index]  # 标量 ()，当前时间步
    t_expanded = t.expand(batch_size)  # (B,) - 扩展为 batch 维度

    # 条件分支推理：使用条件嵌入进行去噪预测
    cond_pred = pipeline.denoise(
        noisy_input=latents,  # (B,T,C) - 当前含噪潜变量
        timesteps=t_expanded,  # (B,) - 时间步
        cond_embeddings=cond_embeddings,  # (B,S,C) - 条件嵌入
        coords=coords,  # (B,T,4) - 空间坐标
    )  # (B,T,C) - 条件预测的速度场

    # 无条件分支推理（仅在提供 uncond_embeddings 时执行）
    uncond_pred = None  # (B,T,C) 或 None
    if uncond_embeddings is not None:
        uncond_pred = pipeline.denoise(
            noisy_input=latents,  # (B,T,C)
            timesteps=t_expanded,  # (B,)
            uncond_embeddings=uncond_embeddings,  # (B,S,C) - 无条件嵌入
            coords=coords,  # (B,T,4)
        )  # (B,T,C) - 无条件预测的速度场

    # CFG 混合：结合条件和无条件预测
    velocity_preds = mix_cfg(
        cond_pred=cond_pred,  # (B,T,C)
        uncond_pred=uncond_pred,  # (B,T,C) 或 None
        scale=float(cfg.guidance_scale),  # 标量 () - CFG 缩放因子
        uncond_mode=cfg.uncond_mode_rollout,  # str - 梯度处理模式
    )  # (B,T,C) - 混合后的速度场

    # 调度器步进：根据速度场更新潜变量
    step_out = scheduler_step_at_index(scheduler, t, latents, velocity_preds)  # (obj 包含 prev_sample/pred_original_sample)
    next_feats = step_out.prev_sample  # (B,T,C) - 下一时刻潜变量
    final_feats_ft = getattr(step_out, "pred_original_sample", velocity_preds)  # (B,T,C) - 预测的最终样本

    return next_feats, velocity_preds, final_feats_ft


def _zeros_like(value: torch.Tensor) -> torch.Tensor:
    """创建与输入张量相同设备和数据类型的零标量。"""
    return torch.zeros((), device=value.device, dtype=value.dtype)  # () - 标量张量


# =====================================================================
# 正则化函数 - KL 散度与 Score Distillation
# 这些函数目前为占位实现，实际项目中需要替换为具体算法
# =====================================================================

def compute_kl_step_regularization(
    scheduler: Any,
    batch_size: int,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_mode: str,
    latents_ori: torch.Tensor,
    t: torch.Tensor,
    final_pred_ft: torch.Tensor,
    pipeline: Any,
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KL 散度正则化（占位函数）。
    
    KL 正则用于约束生成分布与先验分布的差异，防止模式坍塌。
    实际实现需要计算当前预测与参考模型预测之间的 KL 散度。
    
    Args:
        scheduler: 扩散调度器
        batch_size: 批次大小
        cond_embeddings: 条件嵌入 (B,S,C)
        uncond_embeddings: 无条件嵌入 (B,S,C)
        guidance_scale: CFG 缩放因子
        uncond_mode: 梯度处理模式
        latents_ori: 原始潜变量 (B,T,C)
        t: 当前时间步 ()
        final_pred_ft: 最终预测特征 (B,T,C)
        pipeline: 生成 pipeline
        coords: 稀疏坐标 (B,T,4)

    Returns:
        tuple: (reg_scalar, grad_norm)
            - reg_scalar: 正则化损失标量 ()
            - grad_norm: 梯度范数 () (用于监控)
    """
    reg_scalar = _zeros_like(final_pred_ft)  # () - 占位返回零
    grad_norm = _zeros_like(final_pred_ft)  # ()
    return reg_scalar, grad_norm


def compute_score_distillation_step_regularization(
    method: str,
    scheduler: Any,
    batch_size: int,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    uncond_mode: str,
    pipeline: Any,
    final_latent_ft: torch.Tensor,
    latents_x_t: torch.Tensor,
    t: torch.Tensor,
    weight_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Score Distillation Sampling (SDS) / Classifier Score Distillation (CSD) 正则化（占位函数）。
    
    SDS 是一种用于 3D 生成的蒸馏方法，利用预训练的 2D 扩散模型指导 3D 表示优化。
    CSD 是 SDS 的变体，使用分类器梯度进行指导。
    
    Args:
        method: 蒸馏方法 ("sds" 或 "csd")
        scheduler: 扩散调度器
        batch_size: 批次大小
        cond_embeddings: 条件嵌入 (B,S,C)
        uncond_embeddings: 无条件嵌入 (B,S,C)
        guidance_scale: CFG 缩放因子
        uncond_mode: 梯度处理模式
        pipeline: 生成 pipeline
        final_latent_ft: 最终潜变量特征 (B,T,C)
        latents_x_t: 当前时刻潜变量 (B,T,C)
        t: 当前时间步 ()
        weight_mode: 权重计算模式

    Returns:
        tuple: (reg_scalar, grad_norm)
            - reg_scalar: 正则化损失标量 ()
            - grad_norm: 梯度范数 ()
    """
    reg_scalar = _zeros_like(final_latent_ft)  # () - 占位返回零
    grad_norm = _zeros_like(final_latent_ft)  # ()
    return reg_scalar, grad_norm


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
        """条件编码占位类。用于存储图像/文本条件的编码结果。"""

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
        """编辑后视角缓存占位类。存储经过编辑（如风格迁移）后的视角图像。"""

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
    conditions: Conditions = field(default_factory=Conditions)  # 条件编码
    guidance: Guidance = field(default_factory=Guidance)  # 指导信号
    
    # ============== 数据挂载字段 ==============
    space_cache: Any = None  # 空间缓存（用于加速推理）
    conditions_data: Any = None  # 挂载 batch["Conditions"]，包含 cond/neg_cond
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
            # neg_cond 用于 CFG 的无条件分支
            neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))
            self.conditions_data = {"cond": cond, "neg_cond": neg_cond}
        elif self.conditions_data is None:
            # evaluate 路径会预先通过 pipeline.prepare_image_conditions 写入
            raise ValueError("batch['Conditions'] 为空且 state.conditions_data 未设置，无法构造条件。")

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
        从 conditions_data 中提取条件和无条件嵌入。
        
        处理不同格式的条件输入（list 或 Tensor），统一输出为标准张量格式。
        
        Returns:
            tuple: (cond_embeddings, uncond_embeddings)
                - cond_embeddings: 条件嵌入 (B,S,C) 或 (B,C)
                - uncond_embeddings: 无条件嵌入，形状同上
        
        Raises:
            ValueError: 当 conditions_data 为空时
        """
        condition_utils = self.conditions_data
        if condition_utils is None:
            raise ValueError("TrellisState.conditions_data 为空，无法提取 embeddings。")
        
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
        ray_height=cam.ray_height,     # 光线采样高度（用于 SDF）
        ray_width=cam.ray_width,       # 光线采样宽度（用于 SDF）
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
    # 生成稀疏 3D 坐标，定义几何结构的位置
    # =====================================================
    condition_utils = state.conditions_data
    if state.coords is not None:
        # 复用已有的 coords（某些场景下预先生成）
        coords = state.coords  # (N,4) - N = B * T，T 为每个样本的点数
    else:
        # 首次生成：调用 dense_sampling 生成稀疏坐标
        # 训练时 Stage 1 通常不需要梯度
        with torch.no_grad():
            coords = pipeline.dense_sampling(condition_utils, steps=ss_steps)  # (N,4)
        state.coords = coords
    
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
# Loss 与 Guidance - 损失计算与指导信号
# =====================================================================

def compute_guidance(
    guidance_module: Any,
    out: Dict[str, Any],
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    计算 Guidance 损失。
    
    Guidance 模块提供训练信号，可以是：
    - 图像级损失（L1/L2/Perceptual）
    - SDS/VSD 蒸馏损失
    - CLIP 相似度损失
    
    本函数汇总 guidance_module 输出的所有 loss_* 项，
    并乘以对应的 lambda_* 权重进行加权求和。
    
    Args:
        guidance_module: 指导模块实例，需实现 __call__ 方法
        out: 渲染输出，需包含 "comp_rgb" 键
        state: TrellisState 状态对象
        cfg: 配置对象，包含 loss.lambda_* 权重
        step: 当前步骤索引（用于日志命名）
    
    Returns:
        tuple: (guidance_loss, log_items)
            - guidance_loss: 加权后的总损失标量 ()
            - log_items: 用于日志记录的字典
    """
    # ---- 提取渲染图像 ----
    # comp_rgb 形状可能是 (B,H,W,C) 或 (B,V,H,W,C)，需转换为 (B,C,H,W)
    guidance_rgb = out["comp_rgb"].permute(0, 3, 1, 2)  # (B,3,H,W)
    
    # TODO: 替换 batch_data 逻辑
    batch_extra = {}
    
    # ---- 调用 Guidance 模块 ----
    guidance_out = guidance_module(
        guidance_rgb,
        conditions=getattr(state, "guidances_data", None),
        **batch_extra,
    )
    
    # ---- 聚合损失 ----
    guidance_loss = torch.zeros((), device=guidance_rgb.device, dtype=guidance_rgb.dtype)  # () - 初始化为零
    log_items: Dict[str, Any] = {}
    
    for name, value in guidance_out.items():
        # 记录所有输出项用于日志
        log_items[f"guidance/{name}_{step}"] = value
        
        # 处理 loss_* 项：乘以对应权重并累加
        if name.startswith("loss_"):
            lambda_name = name.replace("loss_", "lambda_")  # loss_xxx -> lambda_xxx
            weight = float(cfg.loss.get(lambda_name, 1.0))  # 默认权重 1.0
            guidance_loss = guidance_loss + value * weight  # ()
    
    # ---- 额外的蒸馏损失 ----
    if cfg.lambda_distill > 0.0:
        distill_loss = guidance_out.get("loss_distill", None)
        if distill_loss is not None:
            guidance_loss = guidance_loss + cfg.lambda_distill * distill_loss  # ()
            log_items["loss/distill"] = distill_loss
    
    return guidance_loss, log_items


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
) -> Dict[str, torch.Tensor]:
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
        dict: 训练日志字典，包含各项损失值
    """
    device = accelerator.device
    optimizer = system.optimizer

    # =====================================================
    # 1. 准备阶段
    # =====================================================
    optimizer.zero_grad()  # 清空梯度
    
    # =====================================================
    # 2. Rollout：执行稀疏特征采样
    # 使用 is_training=True 启用 Gradient Checkpointing
    # =====================================================
    # 每步使用不同的随机种子，确保训练数据多样性
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + global_step)
    
    rollout_out = rollout_sparse(
        state, cfg, system, device, 
        generator=generator, 
        is_training=True,  # 启用梯度和 Checkpointing
    )
    
    # latents 是 SparseTensor，其 feats 包含完整的计算图
    latents = rollout_out["latents"]
    
    # =====================================================
    # 3. TODO: 解码 & 渲染 & 损失计算
    # 后续需要实现：
    # - pipeline.decode(latents) -> mesh/GS
    # - renderer.render(mesh, cameras) -> images
    # - compute_guidance(images, ...) -> loss
    # - loss.backward()
    # - optimizer.step()
    # =====================================================
    
    # 清空梯度（占位，实际训练时移除此行）
    optimizer.zero_grad()
    return {}


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
        coords = pipeline.dense_sampling(state.conditions_data, steps=ss_steps)  # (N,4)
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
            outputs = pipeline.decode(latents, formats=['gaussian'])  # dict
            gaussians = outputs['gaussian']  # list[len=B] of Gaussian 对象
            
            if accelerator.is_main_process:
                # 获取相机参数（使用第一个视角渲染）
                extr_all = state.cameras.mesh_w2c  # (B,V,4,4) world-to-camera
                intr_all = state.cameras.mesh_intrinsics  # (B,V,3,3) 内参
                
                for i, gs in enumerate(gaussians):
                    # 提取第 i 个样本的第 0 个视角
                    ext_i = extr_all[i, 0].to(accelerator.device)  # (4,4)
                    intr_i = intr_all[i, 0].to(accelerator.device)  # (3,3)
                    
                    # 渲染 GS
                    render_out = system.renderer.render(gs, ext_i, intr_i)  # color: (3,H,W)
                    name = os.path.splitext(image_names[i])[0]
                    
                    # 转换图像格式：(C,H,W) -> (H,W,C) -> numpy uint8
                    img_chw = render_out['color']  # (3,H,W)
                    img_hwc = img_chw.permute(1, 2, 0)  # (H,W,3)
                    img_np = (img_hwc.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,3)
                    
                    # 保存图像
                    img_dir = save_dir / name
                    img_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img_np).save(str(img_dir / "color.png"))
        else:
            # ---- Mesh Rasterizer 分支 ----
            outputs = pipeline.decode(latents, formats=['mesh'])  # dict
            meshes = outputs['mesh']  # list[len=B] of MeshExtractResult

            # ---- 渲染图像 ----
            if accelerator.is_main_process:
                extr_all = state.cameras.mesh_w2c  # (B,V,4,4)
                intr_all = state.cameras.mesh_intrinsics  # (B,V,3,3)

                for i, mesh in enumerate(meshes):
                    ext_i = extr_all[i, 0].to(accelerator.device)  # (4,4)
                    intr_i = intr_all[i, 0].to(accelerator.device)  # (3,3)
                    
                    # Mesh 渲染器返回多个通道（color, normal, depth 等）
                    render_out = system.renderer.render(mesh, ext_i, intr_i)  # dict of (H,W,C)
                    name = os.path.splitext(image_names[i])[0]
                    
                    # 保存每个渲染通道
                    for k, v in render_out.items():
                        img_np = (v.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # (H,W,C)
                        if img_np.ndim == 3 and img_np.shape[-1] == 1:
                            img_np = img_np[..., 0]  # (H,W) - 单通道压缩
                        img_dir = save_dir / name
                        img_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(img_np).save(str(img_dir / f"{k}.png"))

            # ---- 导出 Mesh 文件 ----
            if accelerator.is_main_process:
                for i, mesh in enumerate(meshes):
                    name = os.path.splitext(image_names[i])[0]  # 去掉 .png 扩展名
                    mesh_dir = save_dir / name
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    out_path = mesh_dir / "mesh.obj"
                    pipeline.export_mesh_obj(mesh, str(out_path))
                    print(f"Saved mesh to {out_path}")

    return {"eval_done": 1.0}


# =====================================================================
# 记录与工具函数
# =====================================================================

def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """
    追加写入 CSV 日志文件。
    
    如果文件不存在，先写入表头；如果存在，追加数据行。
    
    Args:
        path: CSV 文件路径
        row: 要写入的数据行（字典格式）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()  # 首次写入时添加表头
        writer.writerow(row)


def save_visualizations(visuals: Dict[str, Any], out_dir: Path, prefix: str) -> None:
    """
    保存可视化结果（占位函数）。
    
    TODO: 根据 visuals 内容类型实现具体保存逻辑（图像、视频等）。
    
    Args:
        visuals: 可视化内容字典
        out_dir: 输出目录
        prefix: 文件名前缀
    """
    if not visuals:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in visuals.items():
        placeholder = out_dir / f"{prefix}_{name}.txt"
        with placeholder.open("w", encoding="utf-8") as f:
            f.write("TODO: save visualization content here.")


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
# EvalModeGuard - 评估模式上下文管理器
# =====================================================================

class EvalModeGuard:
    """
    评估模式上下文管理器。
    
    用于临时将模块切换到评估模式（eval），并在退出时恢复原状态。
    支持同时管理多个模块，确保 BatchNorm、Dropout 等层在评估时行为正确。
    
    Usage:
        with EvalModeGuard(model1, model2):
            # 这里 model1 和 model2 处于 eval 模式
            output = model1(input)
        # 退出后自动恢复原来的 training 状态
    
    Attributes:
        modules: 要管理的模块列表
        states: 保存的原始训练状态
    """

    def __init__(self, *modules: Any):
        """
        初始化。
        
        Args:
            *modules: 要管理的 nn.Module 实例（自动过滤 None）
        """
        self.modules = [m for m in modules if m is not None]
        self.states = []  # 保存进入前的训练状态

    def __enter__(self):
        """进入上下文：保存状态并切换到 eval 模式。"""
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复原始训练状态。"""
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


# =====================================================================
# 指标记录器 - 训练/评估指标聚合与日志
# =====================================================================

class MetricLoggerBase:
    """
    指标记录器基类。
    
    提供通用的日志发射和分布式聚合功能，供训练/评估记录器继承使用。
    
    日志输出：
    - CSV 文件：持久化存储，便于后续分析
    - Accelerator.log：集成 tensorboard/wandb 等实验追踪工具
    """

    @staticmethod
    def emit_logs(log_dict: Optional[Dict[str, Any]], accelerator: Accelerator, csv_path: Path, global_step: int, epoch: int) -> None:
        """
        发射日志到 CSV 和实验追踪器。
        
        Args:
            log_dict: 日志字典（键值对），值可以是 Tensor 或标量
            accelerator: Accelerate 加速器
            csv_path: CSV 文件路径
            global_step: 全局步数
            epoch: 当前 epoch
        """
        if not log_dict:
            return
        
        # 仅主进程写入 CSV
        if accelerator.is_main_process:
            row = {"global_step": global_step, "epoch": epoch}
            # 将 Tensor 转换为 float
            row.update({k: float(v) if isinstance(v, torch.Tensor) else v for k, v in log_dict.items()})
            append_csv_row(csv_path, row)
        
        # 所有进程发射到实验追踪器
        accelerator.log(log_dict, step=global_step)

    @staticmethod
    def distributed_mean(values_np: Any, accelerator: Accelerator) -> float:
        """
        计算分布式均值。
        
        收集所有进程的值并计算平均值。
        
        Args:
            values_np: 本地值（可以是 numpy array 或标量）
            accelerator: Accelerate 加速器
        
        Returns:
            float: 所有进程的平均值
        """
        tensor = torch.tensor(values_np, device=accelerator.device)
        reduced = accelerator.gather(tensor)  # 收集所有进程的值
        return float(reduced.mean().item())


class TrainMetricLogger(MetricLoggerBase):
    """
    训练指标聚合器。
    
    累积一个 epoch 内的训练损失，并在 epoch 结束时计算平均值。
    支持主损失 (total_loss) 和多个附加损失项。
    
    Usage:
        logger = TrainMetricLogger()
        for batch in train_loader:
            loss = compute_loss(...)
            logger.update(loss, batch_size, guidance_loss=g_loss, reg_loss=r_loss)
        
        log_dict = logger.to_global_log_dict(accelerator)
        logger.emit_logs(log_dict, accelerator, csv_path, step, epoch)
        logger.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置累积器，准备新一轮聚合。"""
        self.sum_total = 0.0  # 总损失累积
        self.count = 0.0      # 样本数累积
        self.extras: Dict[str, float] = {}  # 附加损失项累积

    def update(self, total_loss: torch.Tensor, batch_size: int, **kwargs: torch.Tensor) -> None:
        """
        更新累积值。
        
        Args:
            total_loss: 当前 batch 的总损失
            batch_size: 当前 batch 大小
            **kwargs: 附加损失项（如 guidance_loss, reg_loss 等）
        """
        bs = float(batch_size)
        self.sum_total += float(total_loss.detach().item()) * bs  # 加权累积
        self.count += bs
        
        # 累积附加损失项
        for k, v in kwargs.items():
            self.extras.setdefault(k, 0.0)
            self.extras[k] += float(v.detach().item()) * bs

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        """
        计算平均损失并返回日志字典。
        
        Args:
            accelerator: Accelerate 加速器（预留分布式聚合）
        
        Returns:
            dict: 平均损失字典，格式如 {"loss/total": 0.5, "loss/guidance": 0.1}
        """
        if self.count <= 0.0:
            return None
        
        base = {"loss/total": self.sum_total / self.count}
        for k, v in self.extras.items():
            base[f"loss/{k}"] = v / self.count
        return base


class EvalMetricLogger(MetricLoggerBase):
    """
    评估指标聚合器。
    
    累积评估过程中的各项指标（如 PSNR、SSIM、LPIPS 等），
    并在评估结束时计算平均值。
    
    与 TrainMetricLogger 的区别：
    - 支持动态数量的指标项（不预设固定的 total_loss）
    - 每个指标独立计数（某些指标可能在部分样本上未计算）
    
    Usage:
        logger = EvalMetricLogger()
        for batch in eval_loader:
            metrics = evaluate_batch(...)
            logger.update(metrics, batch_size)
        
        log_dict = logger.to_global_log_dict(accelerator)
        logger.emit_logs(log_dict, accelerator, csv_path, step, epoch)
        logger.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置累积器。"""
        self.sums: Dict[str, float] = {}   # 各指标的累积和
        self.counts: Dict[str, float] = {}  # 各指标的样本计数

    def update(self, metrics: Dict[str, torch.Tensor], batch_size: int) -> None:
        """
        更新累积值。
        
        Args:
            metrics: 指标字典，如 {"psnr": tensor, "ssim": tensor}
            batch_size: 当前 batch 大小
        """
        bs = float(batch_size)
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.detach().item()) * bs
            self.counts[k] = self.counts.get(k, 0.0) + bs

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        """
        计算各指标的平均值。
        
        Args:
            accelerator: Accelerate 加速器
        
        Returns:
            dict: 平均指标字典，如 {"psnr": 25.0, "ssim": 0.95}
        """
        if len(self.sums) == 0:
            return None
        
        out: Dict[str, float] = {}
        for k, v in self.sums.items():
            denom = self.counts.get(k, 0.0)
            if denom > 0.0:
                out[k] = v / denom
        
        return out if len(out) > 0 else None


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
        EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, start_epoch)
        return

    # =====================================================
    # Step 8: 训练循环
    # =====================================================
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
            
            # 执行单步训练
            train_log = train_edit4shape(system, state, cfg, accelerator, epoch, global_step)
            TrainMetricLogger.emit_logs(train_log, accelerator, logs_dir / "train.csv", global_step, epoch)

        # ---- 周期性评估 ----
        if cfg.eval_freq and (epoch % int(cfg.eval_freq) == 0):
            eval_log = evaluate(
                system, cfg, accelerator, 
                epoch=epoch, 
                global_step=global_step, 
                eval_loader=eval_loader, 
                visuals_eval_dir=visuals_eval_dir
            )
            EvalMetricLogger.emit_logs(eval_log, accelerator, logs_dir / "test.csv", global_step, epoch)

        # ---- 周期性保存检查点 ----
        if cfg.save_freq and (epoch % int(cfg.save_freq) == 0):
            ckpt_io.save(system, state, cfg, epoch, global_step)


# =====================================================================
# 程序入口点
# =====================================================================
if __name__ == "__main__":
    # 使用 absl.app.run 启动，支持 --config 等命令行参数
    app.run(main)
