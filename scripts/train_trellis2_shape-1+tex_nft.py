#!/usr/bin/env python3
"""
Trellis Stage 2 GRPO Training Script
=====================================

本脚本实现了 Trellis 3D 生成模型的 Stage 2 强化学习微调（基于 DiffusionNFT 算法）。

核心设计思路：
--------------
1. **两阶段架构**：
   - Stage 1：冻结的稀疏结构生成器（生成 3D 坐标）
   - Stage 2：可训练的 SLatFlowModel（在给定坐标上生成形状特征）

2. **DiffusionNFT 算法**：
   - 基于 GRPO (Group Relative Policy Optimization) 的强化学习框架
   - 对每张输入图像生成 K 个候选 3D mesh
   - 通过奖励模型评分，计算相对优势（advantage）
   - 使用正负样本对比学习优化生成质量

3. **稀疏张量处理**：
   - 使用 SparseTensor(coords, feats) 表示 3D 结构
   - coords: (N, 4) - 稀疏坐标 (batch_idx, x, y, z)
   - feats: (N, C) - 每个坐标点的特征向量

4. **奖励信号来源**：
   - 相机法线一致性（camera_normal）
   - Uni3D 语义相似度
   - 其他可扩展的奖励函数

训练流程：
----------
1. 采样阶段：对每个 batch 生成 K 个候选 mesh 并计算奖励
2. 优势计算：按图像分组计算相对优势（winrate 或 similarity）
3. 更新阶段：使用 DiffusionNFT 损失函数更新 Stage 2 模型

约束条件：
----------
- 仅训练 Stage 2 的 SLatFlowModel
- 使用 LoRA 进行参数高效微调
- 支持分布式训练（多 GPU）
"""

import os
import sys
import gc
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence

# ===== CUDA 内存优化配置 =====
# 启用可扩展段分配策略，减少显存碎片化
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ===== 结束 CUDA 配置 =====


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
import hashlib

import ml_collections
from absl import app
from ml_collections import config_flags
import torch.distributed as dist
from PIL import Image
import wandb

# ===== 项目路径配置 =====
# 设置项目根目录，确保模块导入正确
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加 VGGTObj 参考渲染器路径（用于法线渲染可视化）
_vggt_root = project_root / "_reference_codes" / "VGGTObj"
if str(_vggt_root) not in sys.path:
    sys.path.insert(0, str(_vggt_root))

# 添加 TRELLIS.2 参考代码路径（核心 3D 生成模型）
_trellis2_root = project_root / "_reference_codes" / "TRELLIS.2"
if str(_trellis2_root) not in sys.path:
    sys.path.insert(0, str(_trellis2_root))

# o-voxel 已通过 pip 安装（见 init_env_trellis2.sh），无需再从源码目录导入，避免遮挡已编译好的扩展 _C

# ===== 核心模块导入 =====

# Trellis2 Pipeline：封装了完整的图像到 3D 生成流程
from flow_grpo.diffusers_patch.trellis2_pipeline_with_logprob import (
    Trellis2PipelineWithLogProb,
)

# 稀疏张量操作：Trellis 使用稀疏表示来高效处理 3D 数据
from trellis2.modules.sparse import SparseTensor
from flow_grpo.diffusers_patch.trellis2_sparse_tensor import (
    prepare_sparse_tensor_batch,   # 将多个稀疏张量合并为批次
    sparse_batch_mse,              # 计算稀疏张量之间的 MSE
    sparse_clone_with_feats,       # 克隆稀疏张量并替换特征
    compute_sparse_weighted_mse,   # 计算加权 MSE（用于策略损失）
)

# EMA 指数移动平均：用于稳定训练
from flow_grpo.ema import EMAModuleWrapper

# 奖励模型：评估生成 mesh 的质量
from reward_models.rewards_mesh import MeshScorer

# Accelerate：分布式训练加速框架
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

# 配置文件解析
_CONFIG = config_flags.DEFINE_config_file("config")

# LoRA 相关：参数高效微调
from peft import LoraConfig, get_peft_model, PeftModel
from flow_grpo.peft_sparse.sparse_lora_trellis2 import register_sparse_linear_with_peft
from dataclasses import dataclass
import itertools


# =============================================================================
# 辅助函数：时间步采样、模型包装、随机数生成
# =============================================================================

def compute_timestep_usage(num_steps: int, fraction: float) -> int:
    """计算训练时使用的扩散时间步数量。
    
    在 Flow Matching 训练中，我们不需要对所有时间步进行反向传播，
    而是采样一个子集来降低计算成本。
    
    Args:
        num_steps: 总扩散步数（如 50）
        fraction: 从总步数中采样的比例（如 0.5 → 25 步）
    
    Returns:
        used_steps: 用于训练的时间步数
    """
    total_steps = max(1, int(num_steps))
    frac = max(0.0, float(fraction))
    used_steps = max(1, int(frac * total_steps))
    return used_steps


def _unwrap_model(model: nn.Module) -> nn.Module:
    """提取加速器/并行包装内的真实模型。
    
    当模型被 DDP/FSDP 包装后，原始模型存储在 .module 属性中。
    此函数用于获取底层模型以访问其特定方法。
    """
    return model.module if hasattr(model, "module") else model


def set_model_adapter(model: nn.Module, adapter_name: str) -> None:
    """为 LoRA/PEFT 模型设置当前激活的 adapter。
    
    在使用多个 LoRA adapter 时，需要显式指定使用哪一个。
    """
    target = _unwrap_model(model)
    if hasattr(target, "set_adapter"):
        target.set_adapter(adapter_name)


def setup_backend_determinism() -> None:
    """配置 PyTorch 后端为确定性模式。
    
    关闭 cuDNN benchmark 并启用确定性算法，
    确保相同输入产生相同输出，便于调试和复现。
    """
    torch.backends.cudnn.benchmark = False  # 禁用自动算法选择（会引入随机性）
    torch.backends.cudnn.deterministic = True  # 强制使用确定性算法


def create_eval_generator(device: torch.device, seed: int) -> torch.Generator:
    """创建评估用的固定随机数生成器。
    
    所有 rank 使用完全相同的种子，确保评估时生成结果一致，
    便于公平比较不同 checkpoint 的性能。
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def create_train_generator_for_batch(
    device: torch.device,
    epoch: int,
    batch_idx: int,
    image_paths: List[str],
) -> torch.Generator:
    """为训练采样创建稳定的随机数生成器（用于 same_latent 模式）。
    
    设计目的：
    - 当 same_latent=True 时，对同一张图像的 K 个候选使用相同的初始噪声
    - 种子基于 (epoch, batch_idx, 图像路径哈希) 计算，确保可复现
    - 不同 epoch 的同一张图会使用不同噪声，增加采样多样性
    
    Args:
        device: 目标设备
        epoch: 当前训练轮次
        batch_idx: 当前批次索引
        image_paths: 批次中所有图像的路径列表
    
    Returns:
        配置好种子的 torch.Generator
    """
    # 将所有图像路径拼接后计算哈希，确保批次内容决定种子
    joined = "||".join(image_paths)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    batch_hash = int.from_bytes(digest[:4], byteorder="big", signed=False)
    # 组合 epoch 和 batch_idx 作为基础种子
    base_seed = (epoch * 10000 + int(batch_idx)) % (2**31)
    seed = (base_seed + batch_hash) % (2**31)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


# =============================================================================
# DiffusionNFT 指标记录器
# =============================================================================

class DiffusionNFTMetricLogger:
    """DiffusionNFT 训练指标聚合器。
    
    跟踪以下关键指标：
    - policy_loss: 总策略损失（self + cross 的加权和）
    - policy_loss_self: 自参考策略损失（与自身参考模型对比）
    - policy_loss_cross: 交叉策略损失（跨样本对比，当前已禁用）
    - positive_loss: 正样本（高优势）的重建损失
    - negative_loss: 负样本（低优势）的重建损失
    - kl_loss: 与参考模型的 KL 散度（防止偏离太远）
    - reward_mean: 平均奖励值
    - adv_mean: 平均优势值
    
    指标在 epoch 结束时通过分布式 all_reduce 聚合到主进程。
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置所有累计值，在每个 epoch 开始时调用。"""
        self.sum_policy = 0.0       # 总策略损失累计
        self.sum_policy_self = 0.0  # 自参考策略损失累计
        self.sum_policy_cross = 0.0 # 交叉策略损失累计（当前未使用）
        self.sum_positive = 0.0     # 正样本损失累计
        self.sum_negative = 0.0     # 负样本损失累计
        self.sum_kl = 0.0           # KL 散度累计
        self.count = 0.0            # 样本数累计（用于计算平均）
        self.reward_mean = 0.0      # 平均奖励（单独设置）
        self.adv_mean = 0.0         # 平均优势（单独设置）

    def update(
        self,
        policy_loss: torch.Tensor,
        policy_loss_self: torch.Tensor,
        policy_loss_cross: torch.Tensor,
        positive_loss: torch.Tensor,
        negative_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        batch_size: int,
    ) -> None:
        """更新指标累计值（每个训练步调用）。
        
        使用加权累加（乘以 batch_size），以便最终计算正确的平均值。
        """
        bs_val = float(batch_size)
        self.sum_policy += float(policy_loss.detach().item()) * bs_val
        self.sum_policy_self += float(policy_loss_self.detach().item()) * bs_val
        self.sum_policy_cross += float(policy_loss_cross.detach().item()) * bs_val
        self.sum_positive += float(positive_loss.detach().item()) * bs_val
        self.sum_negative += float(negative_loss.detach().item()) * bs_val
        self.sum_kl += float(kl_loss.detach().item()) * bs_val
        self.count += bs_val

    def set_reward_and_adv_means(self, reward_mean: float, adv_mean: float) -> None:
        """设置奖励和优势的全局平均值（在采样阶段计算后调用）。"""
        self.reward_mean = float(reward_mean)
        self.adv_mean = float(adv_mean)

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        """将本地累计值聚合为全局平均值，返回日志字典。
        
        通过 all_reduce 操作将所有 GPU 的累计值求和，
        然后除以总样本数得到全局平均值。
        
        Returns:
            包含所有指标平均值的字典，或 None（如果没有样本）
        """
        # 将所有累计值打包成张量以便一次性 reduce
        local = torch.tensor(
            [
                self.sum_policy,
                self.sum_policy_self,
                self.sum_policy_cross,
                self.sum_positive,
                self.sum_negative,
                self.sum_kl,
                self.count,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )  # 形状: (7,)
        
        # 分布式聚合：所有 rank 的值求和
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        
        denom = float(local[6].item())  # 总样本数
        if denom <= 0.0:
            return None
        
        return {
            "epoch/policy_loss": float(local[0].item() / denom),
            "epoch/policy_loss_self": float(local[1].item() / denom),
            "epoch/policy_loss_cross": float(local[2].item() / denom),
            "epoch/positive_loss": float(local[3].item() / denom),
            "epoch/negative_loss": float(local[4].item() / denom),
            "epoch/kl_loss": float(local[5].item() / denom),
            "epoch/reward_mean": float(self.reward_mean),
            "epoch/adv_mean": float(self.adv_mean),
        }


# =============================================================================
# 优势计算函数
# =============================================================================

def name_to_stable_id(name: str) -> int:
    """将字符串名称映射为跨进程稳定的 63-bit 正整型 ID。
    
    使用 MD5 哈希确保：
    - 相同名称在任何进程上映射到相同 ID
    - 不同名称映射到不同 ID（极低碰撞概率）
    - 结果为正整数，可安全用于分组操作
    """
    h = hashlib.md5(name.encode("utf-8")).digest()  # 16 字节哈希
    return int.from_bytes(h[:8], byteorder="big", signed=False) & 0x7fffffffffffffff


def compute_advantages_per_image(
    image_names: List[str],
    rewards_np_local: np.ndarray,
    accelerator: Accelerator,
    epoch: int,
) -> np.ndarray:
    """按图像分组计算 Z-score 标准化的优势值。
    
    GRPO 的核心思想：对于同一张输入图像生成的 K 个候选 mesh，
    通过组内标准化来计算相对优势。这样可以：
    - 消除不同图像之间的绝对奖励差异
    - 突出同一图像内哪个候选更好
    
    计算公式：
        advantage[i] = (reward[i] - mean(group)) / (std(group) + eps)
    
    形状约定：
        - N: 当前进程样本数（通常 N = B_local * K）
        - K: 每张图像的候选数
        - B: 图像数（N = B * K）

    Args:
        image_names: 每个样本对应的图像名称列表，长度 N
        rewards_np_local: 本地奖励向量，形状 (N,)
        accelerator: Accelerate 加速器
        epoch: 当前 epoch（用于日志）
    
    Returns:
        本地优势向量，形状 (N,)，顺序与输入对齐
    """
    device = accelerator.device

    # 将图像名称转换为稳定的数值 ID，用于分组
    image_ids_list = [name_to_stable_id(n) for n in image_names]
    image_ids_local_tensor = torch.tensor(image_ids_list, device=device, dtype=torch.long)  # (N,)
    rewards_local_tensor = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # (N,)

    # 按图像 ID 排序，将同一图像的 K 个候选聚集在一起
    sort_vals, sort_idx = torch.sort(image_ids_local_tensor)  # (N,), (N,)
    rewards_sorted = rewards_local_tensor.index_select(0, sort_idx)  # (N,)
    
    # 获取每个图像的候选数（应该都是 K）
    unique_ids, counts = torch.unique(sort_vals, return_counts=True)  # (B,), (B,)
    B = int(unique_ids.numel())  # 图像数
    K = int(counts[0].item())    # 每图候选数
    # 验证所有图像的候选数一致
    assert int(counts.min().item()) == K and int(counts.max().item()) == K
    
    # 重塑为 (B, K) 便于按组计算统计量
    scores_bk = rewards_sorted.reshape(B, K)  # (B, K)
    mean_b = scores_bk.mean(dim=1, keepdim=True)  # (B, 1) - 每组均值
    std_b = scores_bk.std(dim=1, keepdim=True)    # (B, 1) - 每组标准差
    
    # Z-score 标准化：(x - mean) / std
    advantages_bk = (scores_bk - mean_b) / (std_b + 1e-8)  # (B, K)
    advantages_sorted = advantages_bk.reshape(B * K)  # (N,)
    
    # 逆排序，恢复原始顺序
    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(B * K, device=advantages_sorted.device, dtype=torch.long)
    advantages_local_tensor = advantages_sorted.index_select(0, inv_idx)  # (N,)

    return advantages_local_tensor.detach().cpu().numpy().astype(np.float64)

class EvalModeGuard:
    """上下文管理器：临时将模型切换到评估模式，退出时恢复原状态。
    
    用于在训练循环中进行采样，确保 BatchNorm/Dropout 等层
    使用评估模式的行为，同时不影响后续训练。
    
    Example:
        with EvalModeGuard(model1, model2):
            # model1, model2 处于 eval 模式
            outputs = model1(inputs)
        # 退出后自动恢复原始训练状态
    """
    def __init__(self, *modules: nn.Module):
        self.modules = [m for m in modules if m is not None]
        self.states: List[bool] = []  # 保存原始 training 状态

    def __enter__(self):
        # 记录原始状态并切换到 eval
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始训练状态
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)


def distributed_mean(values_np: np.ndarray, accelerator: Accelerator) -> float:
    """分布式计算全局均值。
    
    将所有 GPU 上的值汇总后计算平均，用于日志记录。
    """
    if values_np.size == 0:
        return 0.0
    vals = torch.as_tensor(values_np, device=accelerator.device, dtype=torch.float32)  # (N,)
    total = vals.sum()  # 标量
    count = torch.tensor(float(vals.numel()), device=accelerator.device, dtype=torch.float32)
    # 跨 GPU 聚合
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    denom = float(count.item())
    if denom <= 0.0:
        return 0.0
    return float((total / max(denom, 1e-8)).item())


# =============================================================================
# 样本数据结构
# =============================================================================

@dataclass
class TrellisSample:
    """单个生成样本的完整信息，用于训练阶段的策略更新。
    
    每个 TrellisSample 代表一次 3D 生成的结果，包含：
    - 生成的稀疏 latent（形状和纹理，用于计算策略损失）
    - 条件信息（用于重放生成过程）
    - 奖励信息（用于计算优势）
    - 来源信息（用于分组和日志）
    """
    x0_shape_sparse: SparseTensor      # 形状 3D 稀疏 latent，形状: (N_points, C_shape)
    x0_tex_sparse: SparseTensor        # 纹理 3D 稀疏 latent，形状: (N_points, C_tex)
    cond_patches: torch.Tensor         # 图像条件 patch embeddings，形状: (1, P, C)
    reward_components: Dict[str, float]  # 各奖励组件的分数，如 {"camera_normal": 0.8, "uni3d": 0.7}
    reward_avg: float            # 加权平均奖励
    advantage: float             # 计算出的优势值（正=好于组内平均，负=差于平均）
    image_name: str              # 来源图像的文件名（用于分组）
    image_path: str              # 来源图像的完整路径（用于日志）


@dataclass
class PolicyLossResult:
    """策略损失计算的结果容器。
    
    DiffusionNFT 算法将生成结果分为正负样本并分别计算损失。
    """
    policy_vec: torch.Tensor  # 每个样本的策略损失向量，形状: (B,)
    pos_mean: torch.Tensor    # 正样本（高优势）的平均损失
    neg_mean: torch.Tensor    # 负样本（低优势）的平均损失


class TrellisSampleCollection:
    """TrellisSample 的集合管理器，支持按图像分组存储和批次迭代。
    
    核心功能：
    1. **分组存储**：按图像名称分组，便于计算组内优势
    2. **样本筛选**：保留 top/bottom k 个样本，减少训练噪声
    3. **批次迭代**：支持按固定 batch_size 迭代训练
    4. **优势计算**：封装奖励聚合和优势计算逻辑
    
    内部结构：
        _samples: OrderedDict[image_id -> List[TrellisSample]]
        使用 OrderedDict 确保迭代顺序稳定
    """

    def __init__(self):
        # 按图像 ID 分组存储样本
        self._samples: "OrderedDict[int, List[TrellisSample]]" = OrderedDict()
        # =========================================================================
        # 规范化参数缓存（用于 tex 模型训练时计算 shape_norm）
        # =========================================================================
        # 背景：Trellis2 的 tex 模型需要 shape latent 作为 concat_cond 条件输入。
        # 但 tex 模型期望的是**规范化后的** shape latent（mean=0, std=1 空间），
        # 而我们保存的 x0_shape_sparse 是**反规范化的**（原始 latent 空间）。
        # 
        # 规范化公式：shape_norm = (shape_latent - mean) / std
        # 反规范化公式：shape_latent = shape_norm * std + mean
        #
        # 这里缓存 mean 和 std 参数，避免每次训练都从 pipeline 获取。
        # =========================================================================
        self._shape_norm_mean: Optional[torch.Tensor] = None  # 形状: (C_shape,)
        self._shape_norm_std: Optional[torch.Tensor] = None   # 形状: (C_shape,)

    def set_normalization_params(self, shape_mean: torch.Tensor, shape_std: torch.Tensor) -> None:
        """设置 shape latent 的规范化参数。
        
        在采样阶段从 pipeline.shape_slat_normalization 获取后调用此方法缓存参数。
        参数保存在 CPU 上以节省 GPU 显存，使用时再移动到目标设备。
        
        Args:
            shape_mean: shape latent 的均值向量，形状 (C_shape,)
            shape_std: shape latent 的标准差向量，形状 (C_shape,)
        """
        self._shape_norm_mean = shape_mean.cpu()  # 形状: (C_shape,)
        self._shape_norm_std = shape_std.cpu()    # 形状: (C_shape,)

    def compute_shape_norm(self, x0_shape_batch: SparseTensor, device: torch.device, dtype: torch.dtype) -> SparseTensor:
        """计算规范化的 shape latent，用于 tex 模型的 concat_cond。
        
        Tex 模型的输入结构：
        - xt_tex: 噪声化的 tex latent，形状 (N_total, C_tex)
        - concat_cond: 规范化的 shape latent，形状 (N_total, C_shape)
        - 模型内部会拼接：[xt_tex, concat_cond] → (N_total, C_tex + C_shape)
        
        因此我们需要将反规范化的 x0_shape_batch 转换为规范化空间。
        
        计算公式：
            shape_norm = (x0_shape - mean) / std
        
        Args:
            x0_shape_batch: 反规范化的 shape latent batch，SparseTensor
                - feats 形状: (N_total, C_shape)
            device: 目标设备（GPU）
            dtype: 目标数据类型
            
        Returns:
            规范化后的 shape latent SparseTensor，feats 形状: (N_total, C_shape)
            
        Raises:
            RuntimeError: 若规范化参数未设置
        """
        if self._shape_norm_mean is None or self._shape_norm_std is None:
            raise RuntimeError("规范化参数未设置，请先调用 set_normalization_params()")
        
        # 将规范化参数移动到目标设备
        mean = self._shape_norm_mean.to(device=device, dtype=dtype)  # (C_shape,)
        std = self._shape_norm_std.to(device=device, dtype=dtype)    # (C_shape,)
        
        # 应用规范化变换：(x - mean) / std
        # 使用 detach() 确保不追踪 shape latent 的梯度（它是固定的条件）
        shape_norm_feats = (x0_shape_batch.feats.detach() - mean) / std  # (N_total, C_shape)
        
        # 创建新的 SparseTensor，保持坐标和 layout 不变，只替换 feats
        return sparse_clone_with_feats(x0_shape_batch, shape_norm_feats)

    def add(self, sample: TrellisSample) -> None:
        """添加单个样本到对应图像的分组中。"""
        key = name_to_stable_id(sample.image_name)
        if key not in self._samples:
            self._samples[key] = []
        self._samples[key].append(sample)

    def extend(self, samples: List[TrellisSample]) -> None:
        """批量添加多个样本。"""
        for sample in samples:
            self.add(sample)

    def __len__(self) -> int:
        """返回所有样本的总数。"""
        return sum(len(v) for v in self._samples.values())

    def __iter__(self):
        """迭代所有样本（扁平化）。"""
        for samples in self._samples.values():
            for sample in samples:
                yield sample

    def as_list(self) -> List[TrellisSample]:
        """返回所有样本的扁平列表。"""
        return [sample for samples in self._samples.values() for sample in samples]

    def iter_batches(self, batch_size: int):
        """按固定大小生成批次迭代器（用于训练循环）。"""
        flat = self.as_list()
        for start in range(0, len(flat), batch_size):
            yield flat[start:start + batch_size]

    def __getitem__(self, item):
        flat = self.as_list()
        return flat[item]

    def clear(self) -> None:
        """清空所有样本（在 epoch 结束时调用）。"""
        self._samples.clear()

    def valid_ratio(self) -> float:
        """计算有效样本比例（优势非零的样本占比）。
        
        用于监控：如果大量样本优势为零，说明组内样本差异小，
        可能需要增加采样多样性。
        """
        flat = self.as_list()
        total = len(flat)
        if total == 0:
            return 0.0
        non_zero = sum(1 for s in flat if abs(s.advantage) > 0.0)
        return float(non_zero) / float(total)

    def compute_rewards_and_advantages(
        self,
        reward_weights: Dict[str, float],
        accelerator: Accelerator,
        epoch: int,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """计算所有样本的奖励和优势值，并更新样本属性。
        
        使用加权平均奖励 + Z-score 标准化优势。
        
        Args:
            reward_weights: 各奖励组件的权重，如 {"camera_normal": 0.5, "uni3d": 0.5}
            accelerator: Accelerate 加速器
            epoch: 当前 epoch
            
        Returns:
            (image_names, rewards, advantages): 图像名列表、奖励数组、优势数组
        """
        flat_samples = self.as_list()
        N_local = len(flat_samples)
        image_names = [s.image_name for s in flat_samples]
        if N_local == 0:
            return image_names, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        # 使用已有的加权平均奖励
        rewards_local = np.array([s.reward_avg for s in flat_samples], dtype=np.float64)
        
        # 计算 Z-score 标准化优势
        advantages_local = compute_advantages_per_image(
            image_names=image_names,
            rewards_np_local=rewards_local,
            accelerator=accelerator,
            epoch=epoch,
        )

        # 将计算结果写回样本对象
        for sample, reward_val, adv_val in zip(flat_samples, rewards_local.tolist(), advantages_local.tolist()):
            sample.reward_avg = float(reward_val)
            sample.advantage = float(adv_val)

        return image_names, rewards_local, advantages_local

    @staticmethod
    def move_batch_samples(
        batch_samples: List[TrellisSample],
        device: torch.device,
        dtype: torch.dtype,
        adv_clip_max: float,
    ) -> Tuple[torch.Tensor, SparseTensor, SparseTensor, torch.Tensor]:
        """将一批样本移动到目标设备，并准备训练所需的张量。
        
        Args:
            batch_samples: 当前批次的样本列表
            device: 目标设备（GPU）
            dtype: 目标数据类型
            adv_clip_max: 优势裁剪的最大值
            
        Returns:
            (cond_batched, x0_shape_batch, x0_tex_batch, routing_probs):
                - cond_batched: 条件 embeddings，形状 (B, P, C)
                - x0_shape_batch: 批量形状稀疏 latent
                - x0_tex_batch: 批量纹理稀疏 latent
                - routing_probs: 路由权重 [0,1]，用于正负样本加权
        """
        # 拼接条件 embeddings
        cond_batched = torch.cat([s.cond_patches.to(device=device, dtype=dtype) for s in batch_samples], dim=0)
        # 合并形状稀疏 latent
        shape_sparse_list = [s.x0_shape_sparse.to(device=device, dtype=dtype) for s in batch_samples]
        x0_shape_batch: SparseTensor = prepare_sparse_tensor_batch(shape_sparse_list, batch_size=len(batch_samples))
        # 合并纹理稀疏 latent
        tex_sparse_list = [s.x0_tex_sparse.to(device=device, dtype=dtype) for s in batch_samples]
        x0_tex_batch: SparseTensor = prepare_sparse_tensor_batch(tex_sparse_list, batch_size=len(batch_samples))
        # 计算路由权重（将优势映射到 [0,1]）
        routing_vals = torch.tensor([s.advantage for s in batch_samples], device=device, dtype=torch.float32)
        routing_probs = compute_routing_weights(routing_vals, adv_clip_max)
        return cond_batched, x0_shape_batch, x0_tex_batch, routing_probs

    @staticmethod
    def _split_sparse_batch(batch: SparseTensor) -> List[SparseTensor]:
        """将批量 SparseTensor 拆分为单样本列表（保留梯度流）。
        
        用于需要逐样本处理的场景。
        """
        splits: List[SparseTensor] = []
        for sl in batch.layout:
            feats_slice = batch.feats[sl]
            coords_slice = batch.coords[sl].clone()
            coords_slice[:, 0] = 0  # 重置 batch 索引为 0
            splits.append(
                SparseTensor(
                    feats=feats_slice,
                    coords=coords_slice,
                    layout=[slice(0, coords_slice.shape[0])],
                )
            )
        return splits

    @staticmethod
    def _group_indices_by_image(batch_samples: List[TrellisSample]) -> Dict[str, List[int]]:
        """按图像名称分组样本索引（用于组内操作）。"""
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, sample in enumerate(batch_samples):
            groups[sample.image_name].append(idx)
        return groups

    @staticmethod
    def compute_sparse_policy_loss(
        batch_samples: List[TrellisSample],
        x0_pos: SparseTensor,
        x0_neg: SparseTensor,
        x0_ref: SparseTensor,
        routing_probs: torch.Tensor,
        nft_beta: float,
        mode: str,
    ) -> PolicyLossResult:
        """计算 DiffusionNFT 的稀疏策略损失。
        
        DiffusionNFT 核心思想：
        - 正样本方向：让模型预测更接近 x0_pos（高优势样本的去噪方向）
        - 负样本方向：让模型预测远离 x0_neg（低优势样本的去噪方向）
        - 使用 routing_probs 作为权重混合正负损失
        
        损失公式：
            policy = routing * pos_loss + (1 - routing) * neg_loss
            
        其中 routing ∈ [0,1]，高优势样本 routing 接近 1，低优势样本接近 0。
        
        Args:
            batch_samples: 当前批次样本
            x0_pos: 正样本去噪预测（往高优势方向插值）
            x0_neg: 负样本去噪预测（往低优势方向插值）
            x0_ref: 参考点（当前噪声状态）
            routing_probs: 路由权重 [0,1]
            nft_beta: NFT 插值强度
            mode: 损失模式（仅支持 "self"）
            
        Returns:
            PolicyLossResult 包含策略损失向量和正负损失均值
        """
        mode_norm = (mode or "self").lower()
        beta_denom = max(float(nft_beta), 1e-6)

        if mode_norm == "self":
            # 计算正负样本与参考点的 MSE
            pos_vec = compute_sparse_weighted_mse(x0_pos, x0_ref)  # (B,)
            neg_vec = compute_sparse_weighted_mse(x0_neg, x0_ref)  # (B,)
            routing = routing_probs.to(pos_vec.dtype)
            # 加权混合：高优势样本强调正损失，低优势样本强调负损失
            policy_vec = routing * (pos_vec / beta_denom) + (1.0 - routing) * (neg_vec / beta_denom)
        else:
            raise ValueError(f"Sparse policy cross 模式已移除，当前仅支持 self，收到: {mode}")

        pos_mean = pos_vec.mean()
        neg_mean = neg_vec.mean()
        return PolicyLossResult(policy_vec=policy_vec, pos_mean=pos_mean, neg_mean=neg_mean)

    @staticmethod
    def build_samples_from_generation(
        meshes: List[Any],
        all_shape_latents: List[SparseTensor],
        all_tex_latents: List[SparseTensor],
        cond_batch: torch.Tensor,
        rewards: Sequence[float],
        reward_parts_local: Dict[str, Union[np.ndarray, torch.Tensor]],
        batch_meta: List[dict],
        batch_paths: Sequence[str],
        k: int,
    ) -> List[TrellisSample]:
        """从生成结果构建 TrellisSample 列表。
        
        在采样阶段完成后调用，将生成的 mesh、latent、条件和奖励
        打包成 TrellisSample 对象，供后续训练使用。
        
        Args:
            meshes: 生成的 mesh 列表，长度 BK（B 张图 × K 个候选）
            all_shape_latents: 对应的形状稀疏 latent 列表
            all_tex_latents: 对应的纹理稀疏 latent 列表
            cond_batch: 条件 embeddings，形状 (B, P, C)
            rewards: 平均奖励列表，长度 BK
            reward_parts_local: 各奖励组件的分数字典
            batch_meta: 图像元数据列表，长度 B
            batch_paths: 图像路径列表，长度 B
            k: 每张图的候选数
            
        Returns:
            TrellisSample 列表，长度 BK
        """
        BK = len(meshes)
        samples: List[TrellisSample] = []

        for s in range(BK):
            # 取对应的形状稀疏 latent（若数量不足则复用最后一个）
            shape_src = all_shape_latents[s] if s < len(all_shape_latents) else all_shape_latents[-1]
            # 复制到 CPU 以节省 GPU 显存
            shape_coords_cpu = shape_src.coords.clone().detach().cpu()
            shape_coords_cpu[:, 0] = 0  # 重置 batch 索引
            shape_feats_cpu = shape_src.feats.detach().cpu()
            shape_latent_cpu = SparseTensor(feats=shape_feats_cpu, coords=shape_coords_cpu, layout=[slice(0, shape_feats_cpu.shape[0])])

            # 取对应的纹理稀疏 latent
            tex_src = all_tex_latents[s] if s < len(all_tex_latents) else all_tex_latents[-1]
            tex_coords_cpu = tex_src.coords.clone().detach().cpu()
            tex_coords_cpu[:, 0] = 0  # 重置 batch 索引
            tex_feats_cpu = tex_src.feats.detach().cpu()
            tex_latent_cpu = SparseTensor(feats=tex_feats_cpu, coords=tex_coords_cpu, layout=[slice(0, tex_feats_cpu.shape[0])])

            # 取对应图像的条件（s // k 得到图像索引）
            cond_patches_s = cond_batch[s // k:s // k + 1].detach().cpu()
            # 收集各奖励组件
            reward_components = {**{rk: float(rv[s]) for rk, rv in reward_parts_local.items()}}
            sample = TrellisSample(
                x0_shape_sparse=shape_latent_cpu,
                x0_tex_sparse=tex_latent_cpu,
                cond_patches=cond_patches_s,
                reward_components=reward_components,
                reward_avg=float(rewards[s]),
                advantage=0.0,  # 稍后由 compute_rewards_and_advantages 填充
                image_name=batch_meta[s // k]["image_name"],
                image_path=batch_meta[s // k].get("image_path", batch_paths[s // k]),
            )
            samples.append(sample)
        return samples


def compute_routing_weights(advantages: torch.Tensor, adv_clip_max: float) -> torch.Tensor:
    """将优势值映射为 [0, 1] 的路由权重。
    
    DiffusionNFT 使用路由权重来决定正负样本损失的混合比例：
    - 高优势（>0）→ 权重接近 1 → 强调正样本损失
    - 低优势（<0）→ 权重接近 0 → 强调负样本损失
    - 零优势 → 权重 = 0.5 → 平等对待
    
    映射公式：
        clipped = clamp(advantage, -max, max)
        weight = clipped / max / 2 + 0.5
        
    Args:
        advantages: 优势值张量
        adv_clip_max: 裁剪边界（防止极端值）
        
    Returns:
        路由权重，范围 [0, 1]
    """
    adv_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    normalized = (adv_clip / adv_clip_max) / 2.0 + 0.5
    return torch.clamp(normalized, 0.0, 1.0)


# =============================================================================
# 数据集与数据加载器
# =============================================================================

class Image3DDataset(Dataset):
    """3D 生成训练/评估用的图像数据集。
    
    功能特点：
    1. 支持常见图像格式（jpg, png, bmp）
    2. 自动处理 RGBA 透明通道（Trellis 支持带 alpha 的输入）
    3. 可选预加载法线缓存（用于相机法线奖励计算）
    
    目录结构要求：
        image_dir/
        ├── image1.png
        ├── image2.jpg
        └── ...
        
        或者：
        image_dir/
        └── images/
            ├── image1.png
            └── ...
    """
    
    def __init__(self, image_dir: str, normal_cache_dir: Optional[str] = None, normal_resolution: Optional[int] = None):
        """初始化数据集。
        
        Args:
            image_dir: 图像目录路径
            normal_cache_dir: 法线缓存目录（可选，用于 camera_normal 奖励）
            normal_resolution: 法线图分辨率（与缓存配合使用）
        """
        self.image_dir = Path(image_dir)
        # 兼容 image_dir/images/ 子目录结构
        if (self.image_dir / "images").exists():
            self.image_dir = self.image_dir / "images"
        
        # 收集所有支持格式的图像文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(sorted(self.image_dir.glob(ext)))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # 法线缓存配置（用于加速相机法线奖励计算）
        self.normal_cache_dir = str(normal_cache_dir) if normal_cache_dir is not None else None
        self.normal_resolution = int(normal_resolution) if normal_resolution is not None else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """加载单张图像及其元数据。
        
        Returns:
            dict 包含:
                - image: PIL 图像对象（RGB 或 RGBA）
                - image_path: 图像完整路径
                - metadata: 元数据字典，包含 image_name，可能包含 normal_path 和 normal_pil
        """
        image_path = str(self.image_files[idx])
        image_pil = Image.open(image_path)
        
        # RGBA 图像保留 alpha 通道（Trellis 可利用透明信息）
        if image_pil.mode == 'RGBA':
            image = image_pil
        else:
            image = image_pil.convert('RGB')
        
        meta = {"image_name": self.image_files[idx].name}
        
        # 若配置了法线缓存，则预加载对应的法线图
        if self.normal_cache_dir is not None and self.normal_resolution is not None:
            stem = self.image_files[idx].stem  # 不含扩展名的文件名
            normal_path = str(Path(self.normal_cache_dir) / f"R{self.normal_resolution}" / f"{stem}.png")
            meta["normal_path"] = normal_path
            # 预加载法线图 PIL（供 scorer 直接使用）
            normal_pil = Image.open(normal_path).convert('RGB')
            meta["normal_pil"] = normal_pil
        
        return {
            "image": image,
            "image_path": image_path,
            "metadata": meta,
        }

    @staticmethod
    def collate_fn(examples):
        """自定义 collate 函数：不进行张量堆叠，保留 PIL 列表。
        
        因为 Trellis pipeline 期望接收 PIL 图像列表，所以不使用默认的
        张量堆叠行为。
        """
        images = [ex["image"] for ex in examples]
        image_paths = [ex["image_path"] for ex in examples]
        metadata = [ex["metadata"] for ex in examples]
        return images, image_paths, metadata


def dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    """从配置创建训练数据加载器。
    
    配置要求：
        - config.train_data_dir: 训练图像目录
        - config.camera_normal_train.cache_dir: 训练集法线缓存目录
        - config.camera_normal_train.normal_resolution: 法线图分辨率
        - config.sample.input_batch_size: 每个 GPU 的批大小
    
    Returns:
        配置好分布式采样器的 DataLoader
    """
    train_root = str(config.train_data_dir)
    normal_cache_dir = str(config.camera_normal_train.cache_dir)
    normal_resolution = int(config.camera_normal_train.normal_resolution)
    dataset = Image3DDataset(train_root, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)
    
    # 分布式采样器：每个 GPU 处理数据集的不同分片
    batch_size = int(config.sample.input_batch_size)
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,   # 训练时打乱
        drop_last=True, # 丢弃不完整批次
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=Image3DDataset.collate_fn,
    )
    return loader


def eval_dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    """从配置创建评估数据加载器。
    
    与训练加载器的区别：
        - 使用评估数据集和法线缓存
        - 不打乱数据顺序
        - 不丢弃不完整批次
        - 使用更少的 worker
    """
    eval_root = str(config.eval_data_dir)
    normal_cache_dir = str(config.camera_normal_eval.cache_dir)
    normal_resolution = int(config.camera_normal_eval.normal_resolution)
    eval_dataset = Image3DDataset(eval_root, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)

    eval_bs = int(config.sample.test_batch_size)
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,    # 评估时不打乱
        drop_last=False,  # 保留所有样本
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_bs,
        sampler=eval_sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=Image3DDataset.collate_fn,
    )
    return eval_loader


# =============================================================================
# 优化器构建
# =============================================================================

def build_optimizer(params, config: ml_collections.ConfigDict):
    """根据配置构建优化器。
    
    支持的优化器类型：
    - adam_8bit: 8-bit AdamW（使用 bitsandbytes，显存友好）
    - adamw: 标准 AdamW
    - adan: Adan 优化器（3 元 betas）
    - 其他 timm 支持的优化器
    
    配置要求：
        config.train.optimizer.type: 优化器类型
        config.train.optimizer.lr: 学习率
        config.train.optimizer.beta1, beta2: Adam betas
        config.train.optimizer.eps: 数值稳定性 epsilon
        config.train.optimizer.weight_decay: 权重衰减
    
    Args:
        params: 可训练参数列表
        config: 配置对象
        
    Returns:
        配置好的优化器实例
    """
    opt = config.train.optimizer
    opt_type = str(opt.type).lower()

    if opt_type == 'adam_8bit':
        # 8-bit Adam：显存占用减少约 75%，适合大模型
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            params,
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2),
            eps=opt.eps,
            weight_decay=opt.weight_decay,
        )
    else:
        # 使用 timm 的优化器工厂
        from timm.optim.optim_factory import create_optimizer_v2
        # Adan 优化器需要 3 元 betas
        if opt_type == 'adan':
            betas = (0.98, 0.92, 0.99)
        else:
            betas = (opt.beta1, opt.beta2)
        return create_optimizer_v2(
            params,
            opt=opt_type,
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            betas=betas,
            eps=opt.eps,
        )


# 注：旧的 compute_advantages 函数已移除（依赖 tracking/global_std），
# 现在使用 TrellisSampleCollection.compute_rewards_and_advantages


# =============================================================================
# 评估流程
# =============================================================================

def eval_trellis(
    pipeline: Trellis2PipelineWithLogProb,
    test_dataloader: DataLoader,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    mesh_scorer: MeshScorer,
    generator: Optional[torch.Generator] = None,
    export_dir: Optional[str] = None,
    write_mesh: bool = False,
):
    """Trellis 模型评估流程。
    
    对评估数据集的每张图像生成一个 mesh，计算各项奖励指标，
    并可选保存可视化预览和 OBJ 文件。
    
    评估流程（与训练不同，每图仅生成 1 个候选）：
    1. 编码图像条件
    2. Stage1：生成稀疏结构坐标
    3. Stage2：生成形状 latent
    4. 解码为 mesh
    5. 计算奖励并聚合
    
    Args:
        pipeline: Trellis 生成 pipeline
        test_dataloader: 评估数据加载器
        config: 配置对象
        accelerator: Accelerate 加速器
        epoch: 当前 epoch（用于命名导出目录）
        mesh_scorer: 奖励评估器
        generator: 随机数生成器（用于可复现性）
        export_dir: 可视化导出目录（可选）
        write_mesh: 是否导出 OBJ 文件
        
    Returns:
        all_rewards_np: Dict[str, ndarray]，各奖励项的聚合数组
    """
    all_rewards: Dict[str, List[np.ndarray]] = defaultdict(list)

    # 获取需要切换到 eval 模式的模块
    dense_eval_module = pipeline.get_flow_module("structure")
    sparse_eval_module = pipeline.get_flow_module("shape_slat")

    with EvalModeGuard(dense_eval_module, sparse_eval_module):
        for eval_batch in tqdm(
            test_dataloader,
            desc="Eval:",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            images, image_paths, metadata = eval_batch
            
            with torch.inference_mode():  # 关闭梯度追踪，节省显存
                # Step 1: 图像条件编码
                cond_batch = pipeline.prepare_image_conditions(images)  # (B, P, C)
                neg_filled = torch.zeros_like(cond_batch)  # CFG 负向条件为零
                
                # ============================================================
                # Step 2: 批量 Stage1 - 生成所有样本的稀疏结构坐标
                # ============================================================
                B = int(cond_batch.shape[0])
                coords_all, _ = pipeline.stage_1(
                    cond={"cond": cond_batch, "neg_cond": neg_filled},
                    ss_resolution=64,       # 稀疏结构分辨率
                    num_samples=B,
                )  # (N_total, 4) - 所有样本的坐标

                # ============================================================
                # Step 3: 批量 Stage2 - 生成所有样本的形状特征
                # ============================================================
                shape_slat_batched, _ = pipeline.stage_2_shape(
                    cond={
                        "cond": cond_batch,
                        "neg_cond": neg_filled,
                    },
                    coords=coords_all,
                    resolution=1024,
                )  # SparseTensor(N_total, C_shape)

                # ============================================================
                # Step 3.5: 批量纹理 latent 生成
                # ============================================================
                tex_slat_batched, _ = pipeline.stage_2_tex(
                    cond={
                        "cond": cond_batch,
                        "neg_cond": neg_filled,
                    },
                    shape_slat=shape_slat_batched,
                )  # tex_slat_batched: SparseTensor(N_total, C_tex)
                
                # 清理不再需要的中间张量
                del cond_batch, neg_filled, coords_all
                torch.cuda.empty_cache()

                # ============================================================
                # Step 4: 逐样本解码为 mesh（使用 layout 正确分割）
                # ============================================================
                meshes_batch = []
                for b, sl in enumerate(shape_slat_batched.layout):
                    # 提取形状 latent
                    sample_feats = shape_slat_batched.feats[sl]
                    sample_coords = shape_slat_batched.coords[sl].clone()
                    sample_coords[:, 0] = 0  # 重置 batch 索引为 0
                    sample_slat = SparseTensor(
                        feats=sample_feats,
                        coords=sample_coords,
                        layout=[slice(0, sample_coords.shape[0])],
                    )
                    
                    # 提取纹理 latent
                    tex_feats = tex_slat_batched.feats[sl]
                    tex_coords = tex_slat_batched.coords[sl].clone()
                    tex_coords[:, 0] = 0  # 重置 batch 索引为 0
                    tex_slat = SparseTensor(
                        feats=tex_feats,
                        coords=tex_coords,
                        layout=[slice(0, tex_coords.shape[0])],
                    )
                    
                    # 解码为 mesh
                    mesh_obj = pipeline.export_mesh(sample_slat, tex_slat=tex_slat, resolution=1024)
                    meshes_batch.append(mesh_obj)
                    
                    # 立即清理单样本的中间张量
                    del sample_slat, tex_slat, sample_feats, sample_coords, tex_feats, tex_coords
                
                # 清理 batched latent 张量
                del shape_slat_batched, tex_slat_batched
                torch.cuda.empty_cache()

            # 可选：导出可视化和 OBJ 文件
            if export_dir is not None:
                epoch_dir = os.path.join(export_dir, f"eval_epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                
                # 保存 PBR 渲染预览图
                for idx, (mesh, img_path) in enumerate(zip(meshes_batch, image_paths)):
                    name = Path(img_path).stem
                    result = pipeline.render_pbr_snapshot(mesh, resolution=512, nviews=4)
                    panel = pipeline.make_pbr_vis_panel(result, resolution=512)[0]
                    Image.fromarray(panel).save(os.path.join(epoch_dir, f"{name}_{idx}_tex.png"))
                    
                    # 可选：导出 OBJ
                    if write_mesh:
                        import trimesh
                        mesh_trimesh = trimesh.Trimesh(
                            vertices=mesh.vertices.cpu().numpy(),
                            faces=mesh.faces.cpu().numpy(),
                        )
                        mesh_trimesh.export(os.path.join(epoch_dir, f"{name}_{idx}.obj"))
                
                # 配置相机法线奖励的可视化输出目录
                if hasattr(mesh_scorer, "_camera_normal") and (mesh_scorer._camera_normal is not None):
                    cn = mesh_scorer._camera_normal
                    cn.cfg.save_vis = True
                    cn.cfg.vis_dir = epoch_dir

            # Step 5: 计算奖励
            rewards_dict, _ = mesh_scorer.score(meshes_batch, images, metadata, dict(config.reward_fn))
            
            # 跨 GPU 聚合奖励
            for key, value in rewards_dict.items():
                gathered = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards[key].append(gathered)

            # 清理 GPU 缓存
            del meshes_batch
            torch.cuda.empty_cache()
            gc.collect()  # 强制 Python 垃圾回收

    # 合并所有 batch 的奖励
    all_rewards_np = {key: (np.concatenate(v) if len(v) > 0 else np.array([])) for key, v in all_rewards.items()}
    return all_rewards_np




# =============================================================================
# Pipeline 构建与模型准备
# =============================================================================

def build_pipeline(config: ml_collections.ConfigDict, accelerator: Accelerator) -> Trellis2PipelineWithLogProb:
    """构建 Trellis 生成 Pipeline 并移动到目标设备。
    
    加载预训练权重，包括：
    - DINOv2 图像编码器
    - Stage1 稀疏结构生成模型
    - Stage2 形状/纹理生成模型
    - 解码器
    
    Args:
        config: 配置对象，需要 config.pretrained.pipeline_path
        accelerator: Accelerate 加速器
        
    Returns:
        配置好的 Trellis2PipelineWithLogProb 实例
    """
    # 优先使用本地 DINOv3 权重（避免访问需要认证的 HuggingFace 仓库）
    dino_local = project_root / "_reference_codes" / "TRELLIS.2" / "pretrained_weights" / "dinov3-vitl16-pretrain-lvd1689m" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    dino_local_path = str(dino_local) if dino_local.exists() else None

    pipeline = Trellis2PipelineWithLogProb.from_pretrained(
        config.pretrained.pipeline_path,
        dino_local_path=dino_local_path,
    )
    
    # 兼容性设置：部分工具函数期望 pipeline.ref
    pipeline.ref = pipeline
    
    # 将所有模型移动到 GPU，避免 CPU/GPU 混用
    pipeline.to(accelerator.device)
    for m in pipeline.models.values():
        m.to(accelerator.device)
    return pipeline


def get_trainable_model(pipeline: Trellis2PipelineWithLogProb) -> nn.Module:
    """获取 Trellis 中可训练的稀疏形状分支模型（SLatFlowModel）。
    
    按优先级查找：
    1. shape_slat_flow_model_1024（高分辨率版本）
    2. shape_slat_flow_model（默认版本）
    3. shape_slat_flow_model_512（低分辨率版本）
    
    Returns:
        SLatFlowModel 实例
        
    Raises:
        ValueError: 如果找不到任何形状分支模型
    """
    models = pipeline.get_trainable_models()
    slat_model = (
        models.get("shape_slat_flow_model_1024")
        or models.get("shape_slat_flow_model")
        or models.get("shape_slat_flow_model_512")
    )
    if slat_model is None:
        raise ValueError(f"未找到可训练形状分支模型，已有 keys={list(models.keys())}")
    return slat_model


def apply_lora_if_needed(slat_model: nn.Module, config: ml_collections.ConfigDict) -> nn.Module:
    """根据配置为 SLatFlowModel 应用 LoRA 参数高效微调。
    
    LoRA (Low-Rank Adaptation) 只训练少量新增参数，显著降低显存和训练成本。
    
    目标模块（注意力投影层）：
    - to_qkv: Query-Key-Value 联合投影
    - to_q, to_kv: 分离的 Q 和 KV 投影
    - to_out: 输出投影
    
    Args:
        slat_model: 原始 SLatFlowModel
        config: 配置对象，需要 config.use_lora, config.lora.lora_rank
        
    Returns:
        包装后的 PeftModel（若启用 LoRA）或原始模型
    """
    if not bool(config.use_lora):
        return slat_model
    
    # 注册稀疏线性层的 PEFT 支持
    register_sparse_linear_with_peft()
    
    # LoRA 目标模块：注意力层的投影矩阵
    target_modules = [
        "to_qkv",
        "to_q",
        "to_kv",
        "to_out",
    ]
    
    # LoRA 超参数
    lora_r = int(config.lora.lora_rank)  # 低秩矩阵的秩
    lora_alpha = lora_r                   # 缩放因子
    lora_dropout = 0.1                    # Dropout 率
    lora_bias_mode = "none"               # 不训练偏置
    
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha * 2,  # 实际缩放 = alpha / r
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias_mode,
    )
    
    # 若提供预训练 LoRA 路径，则加载；否则从头初始化
    lora_path = (config.train.lora_path if 'lora_path' in config.train else None)
    if isinstance(lora_path, str) and len(lora_path) > 0:
        slat_model = PeftModel.from_pretrained(slat_model, lora_path)
        set_model_adapter(slat_model, "default")
    else:
        slat_model = get_peft_model(slat_model, lora_cfg)
        set_model_adapter(slat_model, "default")
    return slat_model


def prepare_optimizer_and_wrap(
    slat_model: nn.Module,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
) -> tuple[nn.Module, optim.Optimizer, list]:
    """为 Stage2 模型构建优化器并用 Accelerate 包装。
    
    仅优化需要梯度的参数（LoRA 层）。
    
    Returns:
        (wrapped_model, optimizer, trainable_params): 包装后的模型、优化器、参数列表
    """
    sparse_trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    optimizer_stage2 = build_optimizer(sparse_trainable_params, config)
    slat_model, optimizer_stage2 = accelerator.prepare(slat_model, optimizer_stage2)
    return slat_model, optimizer_stage2, sparse_trainable_params


def enable_gradient_checkpointing_if_needed(slat_model: nn.Module, accelerator: Accelerator, config: ml_collections.ConfigDict) -> None:
    """按配置启用梯度检查点以节省显存。
    
    梯度检查点通过在反向传播时重新计算中间激活，
    用计算换显存，对于大模型训练很有帮助。
    """
    use_gc = bool(getattr(config, "gradient_checkpointing", False))
    if use_gc:
        unwrapped = accelerator.unwrap_model(slat_model)
        for blk in unwrapped.blocks:
            blk.use_checkpoint = True


# =============================================================================
# 检查点管理
# =============================================================================

def load_checkpoint(accelerator: "Accelerator", config: ml_collections.ConfigDict, mode: str = "train") -> int:
    """加载训练检查点。
    
    支持两种模式：
    - train: 恢复训练，返回下一个 epoch 编号
    - eval: 仅加载权重进行评估
    
    检查点路径查找逻辑：
    1. 若 config.checkpoint 直接指向 checkpoint_X 目录，则使用它
    2. 若指向包含多个 checkpoint_X 子目录的父目录，则选择最新的
    
    多卡同步：
    使用 main_process_first 和 wait_for_everyone 确保所有 rank
    选择相同的检查点并同步加载，避免竞态条件。
    
    Args:
        accelerator: Accelerate 加速器
        config: 配置对象，需要 config.checkpoint
        mode: "train" 或 "eval"
        
    Returns:
        起始 epoch 编号（eval 模式返回 0）
    """
    def pick_dir(path_str: Optional[str]) -> Optional[Path]:
        """选择要加载的检查点目录。"""
        if not (isinstance(path_str, str) and path_str):
            return None
        root = Path(path_str)
        if not root.exists() or not root.is_dir():
            return None
        # 直接是检查点目录
        if (root / "state.json").exists() or root.name.startswith("checkpoint_"):
            return root
        # 是包含检查点的父目录，选择最新的
        cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")]
        if not cands:
            return None
        cands.sort(key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
        return cands[-1]

    cp = (config.checkpoint if hasattr(config, "checkpoint") else None)

    if mode == "eval":
        # 评估模式：只加载权重，不恢复训练状态
        with accelerator.main_process_first():
            chosen = pick_dir(cp)
        accelerator.wait_for_everyone()
        if chosen:
            accelerator.load_state(str(chosen))
            accelerator.print(f"🔁 Eval-only: loaded {str(chosen)}")
        accelerator.wait_for_everyone()
        return 0

    # 训练恢复模式
    with accelerator.main_process_first():
        chosen = pick_dir(cp)
    accelerator.wait_for_everyone()
    if not chosen:
        return 0  # 无检查点，从头开始
    
    # 加载检查点并同步
    accelerator.load_state(str(chosen))
    accelerator.wait_for_everyone()
    
    # 从目录名解析 epoch 编号
    name = chosen.name
    tail = name.split("_")[-1]
    start_epoch = (int(tail) + 1) if tail.isdigit() else 0
    accelerator.print(f"🔁 Resumed: {str(chosen)} → start_epoch={start_epoch}")
    return start_epoch


def run_eval_only(
    pipeline: Trellis2PipelineWithLogProb,
    config: ml_collections.ConfigDict,
    accelerator: "Accelerator",
    mesh_scorer: MeshScorer,
    run_logger: "RunLogger",
    ema: Optional[EMAModuleWrapper] = None,
    trainable_params: Optional[list] = None,
) -> None:
    """仅评估模式：加载检查点、运行评估、记录结果后退出。
    
    用于 config.eval_only=True 时的独立评估流程。
    若启用 EMA，会使用 EMA 权重进行评估。
    
    Args:
        pipeline: Trellis pipeline
        config: 配置对象
        accelerator: Accelerate 加速器
        mesh_scorer: 奖励评估器
        run_logger: 日志记录器
        ema: EMA 包装器（可选）
        trainable_params: 可训练参数列表（用于 EMA 切换）
    """
    accelerator.wait_for_everyone()
    load_checkpoint(accelerator, config, mode="eval")
    
    # 创建评估数据加载器
    eval_loader = eval_dataloader_from_config(config, accelerator)
    eval_loader.sampler.set_epoch(0)
    gen = create_eval_generator(accelerator.device, int(config.seed))
    
    # 设置导出目录
    dirs = RunDirs.from_config(config)
    export_dir = str(dirs.viz_dir)

    # 开启相机法线可视化
    if hasattr(mesh_scorer, "_camera_normal") and (mesh_scorer._camera_normal is not None):
        cn = mesh_scorer._camera_normal
        cn.cfg.save_vis = True

    # 若启用 EMA，临时切换到 EMA 权重进行评估
    if bool(config.train.ema) and ema is not None and trainable_params is not None:
        ema.copy_ema_to(trainable_params, store_temp=True)  # 保存当前权重并切换到 EMA
        all_rewards_np = eval_trellis(
            pipeline, eval_loader, config, accelerator, epoch=0, mesh_scorer=mesh_scorer, generator=gen, export_dir=export_dir, write_mesh=True
        )
        ema.copy_temp_to(trainable_params)  # 恢复原始权重
    else:
        all_rewards_np = eval_trellis(
            pipeline, eval_loader, config, accelerator, epoch=0, mesh_scorer=mesh_scorer, generator=gen, export_dir=export_dir, write_mesh=True
        )
    
    # 记录评估结果
    if accelerator.is_main_process:
        run_logger.log_eval_rewards(0, all_rewards_np)

    return


def create_ema_if_needed(trainable_params: list, accelerator: Accelerator, config: ml_collections.ConfigDict) -> Optional[EMAModuleWrapper]:
    """按配置创建 EMA（指数移动平均）包装器。
    
    EMA 通过维护参数的滑动平均来稳定训练，常用于评估和最终模型导出。
    
    Args:
        trainable_params: 可训练参数列表
        accelerator: Accelerate 加速器
        config: 配置对象，需要 config.train.ema 和 config.train.ema_decay
        
    Returns:
        EMAModuleWrapper 实例，或 None（若未启用）
    """
    if bool(config.train.ema):
        ema_decay = float(config.train.ema_decay)  # 典型值：0.999
        return EMAModuleWrapper(trainable_params, decay=ema_decay, device=accelerator.device)
    return None


class TrainState:
    """可检查点化的训练状态容器。
    
    用于跟踪全局训练步数，在保存/恢复检查点时自动序列化。
    通过 accelerator.register_for_checkpointing 注册后，
    会自动参与 Accelerate 的状态保存/加载。
    """
    
    def __init__(self, global_step: int = 0):
        self.global_step = int(global_step)  # 全局训练步数

    def state_dict(self) -> dict:
        """序列化为字典（用于保存）。"""
        return {"global_step": int(self.global_step)}

    def load_state_dict(self, state: dict) -> None:
        """从字典恢复（用于加载）。"""
        self.global_step = int(state.get("global_step", 0))






# =============================================================================
# 主函数
# =============================================================================

def main(_):
    """Trellis Stage 2 DiffusionNFT 训练主函数。
    
    整体训练流程：
    ================
    
    1. **初始化阶段**
       - 解析配置
       - 创建 Accelerate 加速器（分布式训练支持）
       - 构建 Trellis Pipeline 和奖励评估器
       - 应用 LoRA 并准备优化器
       - 加载检查点（若有）
    
    2. **每个 Epoch**
       a) **采样阶段**（所有模型 eval 模式，无梯度）
          - 对每个 batch 的图像生成 K 个候选 mesh
          - 使用奖励模型评估每个 mesh 的质量
          - 收集样本到 TrellisSampleCollection
       
       b) **优势计算**
          - 按图像分组计算相对优势（winrate 或 z-score）
          - 可选筛选 top/bottom k 个样本
       
       c) **训练阶段**（Stage2 模型 train 模式）
          - 对采样的稀疏时间步进行 DiffusionNFT 损失计算
          - 包括策略损失（正负样本对比）和 KL 正则化
          - 梯度累积和优化器更新
          - EMA 参数更新
       
       d) **日志与保存**
          - 记录训练指标到 W&B
          - 周期性评估和可视化
          - 保存检查点
    
    DiffusionNFT 算法关键点：
    =========================
    - 正样本：高优势样本的去噪预测方向
    - 负样本：低优势样本的去噪预测方向
    - 策略损失：加权混合正负样本损失，权重由优势决定
    - KL 正则化：防止模型偏离预训练分布太远
    """
    config: ml_collections.ConfigDict = _CONFIG.value
    
    # 验证配置：DiffusionNFT 要求使用 LoRA
    assert config.use_lora, "DiffusionNFT 训练脚本要求 config.use_lora=True"

    # =========================================================================
    # 第一部分：初始化 Accelerator 和基础设施
    # =========================================================================
    
    # 计算实际使用的时间步数（用于梯度累积步数设置）
    sparse_step_count = compute_timestep_usage(
        num_steps=int(config.sample.num_steps),
        fraction=float(config.train.timestep_fraction),
    )

    # 确定实验名称
    run_name = config.run_name if len(config.run_name) > 0 else f"trellis_{int(time.time())}"
    
    # 创建 Accelerator 加速器
    # 梯度累积步数 = 配置值 × 稀疏时间步数（因为每个时间步都有一个反向传播）
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=os.path.join(config.logdir, run_name)),
        log_with=["wandb"],
        gradient_accumulation_steps=max(1, int(config.train.gradient_accumulation_steps * sparse_step_count)),
    )
    
    # 设置随机种子和确定性后端
    set_seed(int(config.seed))
    setup_backend_determinism()

    # 初始化 W&B 日志跟踪器（仅主进程）
    config.run_name = run_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo-trellis",
            config=dict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # =========================================================================
    # 第二部分：构建 Pipeline 和奖励评估器
    # =========================================================================
    
    # 加载 Trellis 生成 Pipeline
    pipeline = build_pipeline(config, accelerator)
    device = accelerator.device
    
    # 配置相机法线奖励参数
    cam_cfg = dict(config.camera_normal) if "camera_normal" in config else {}
    if "camera_normal_train" in config:
        cam_cfg.setdefault("cache_dir", str(config.camera_normal_train.cache_dir))
        cam_cfg.setdefault("normal_resolution", int(config.camera_normal_train.normal_resolution))

    # 创建奖励评估器（按需加载各项奖励模型）
    mesh_scorer = MeshScorer(
        device=device,
        verbose=bool(config.verbose),
        score_fns_cfg=dict(config.reward_fn),
        camera_normal_cfg=cam_cfg,
    )

    # =========================================================================
    # 第三部分：仅评估模式（提前返回）
    # =========================================================================
    
    if bool(config.eval_only):
        # 准备模型（应用 LoRA）
        slat_model = get_trainable_model(pipeline)
        slat_model = apply_lora_if_needed(slat_model, config)
        slat_model = accelerator.prepare(slat_model)
        pipeline.set_shape_flow_model(slat_model)
        
        # 准备日志和 EMA
        dirs = RunDirs.from_config(config)
        run_logger = RunLogger(accelerator, dirs)
        trainable_params_eval = [p for p in slat_model.parameters() if p.requires_grad]
        ema_eval = create_ema_if_needed(trainable_params_eval, accelerator, config)
        if ema_eval is not None:
            accelerator.register_for_checkpointing(ema_eval)
        
        # 运行评估并退出
        run_eval_only(pipeline, config, accelerator, mesh_scorer, run_logger, ema=ema_eval, trainable_params=trainable_params_eval)
        return

    # =========================================================================
    # 第四部分：训练模型准备（同时训练 shape 和 tex 模型）
    # =========================================================================
    # 
    # Trellis2 的 Stage2 包含两个 SLatFlowModel：
    # - shape_slat_flow_model: 生成形状 latent (C=32)
    # - tex_slat_flow_model: 生成纹理 latent (C=32)，依赖 shape_norm 作为条件
    # 
    # 本脚本同时训练两个模型，使用独立的优化器和分阶段反向传播策略。
    # =========================================================================
    
    # 获取 Shape 模型并应用 LoRA
    # shape_slat_flow_model_1024 或 shape_slat_flow_model
    shape_model = get_trainable_model(pipeline)
    shape_model = apply_lora_if_needed(shape_model, config)

    # 获取 Tex 模型并应用 LoRA
    # tex_slat_flow_model_1024 或 tex_slat_flow_model
    tex_model = pipeline.get_tex_flow_model()
    tex_model = apply_lora_if_needed(tex_model, config)

    # 获取可训练参数（LoRA 层的参数）
    shape_trainable_params = [p for p in shape_model.parameters() if p.requires_grad]
    tex_trainable_params = [p for p in tex_model.parameters() if p.requires_grad]
    all_trainable_params = shape_trainable_params + tex_trainable_params  # 用于 EMA

    # 构建两个独立的优化器
    # 独立优化器的优势：
    # - 可以为两个模型设置不同的学习率
    # - 可以使用不同的学习率调度策略
    # - 分阶段训练时，每个优化器独立更新
    optimizer_shape = build_optimizer(shape_trainable_params, config)
    optimizer_tex = build_optimizer(tex_trainable_params, config)

    # 用 Accelerate 包装模型和优化器
    # accelerator.prepare() 会自动处理：
    # - 模型的 DDP/FSDP 包装（多 GPU 训练）
    # - 优化器的分布式同步
    # - 混合精度训练的自动缩放
    shape_model, tex_model, optimizer_shape, optimizer_tex = accelerator.prepare(
        shape_model, tex_model, optimizer_shape, optimizer_tex
    )

    # 设置 LoRA adapter 和梯度检查点
    # 必须在 accelerator.prepare() 之后设置，因为模型可能被包装
    set_model_adapter(shape_model, "default")
    set_model_adapter(tex_model, "default")
    enable_gradient_checkpointing_if_needed(shape_model, accelerator, config)
    enable_gradient_checkpointing_if_needed(tex_model, accelerator, config)

    # 将 LoRA 包装后的模型传回 pipeline
    # 这样 pipeline 的采样方法会使用训练中的 LoRA 权重
    pipeline.set_shape_flow_model(shape_model)
    pipeline.set_tex_flow_model(tex_model)

    # 创建 EMA 和训练状态，注册到 Accelerate 以支持检查点保存
    ema_stage2 = create_ema_if_needed(all_trainable_params, accelerator, config)
    if ema_stage2 is not None:
        accelerator.register_for_checkpointing(ema_stage2)
    
    train_state = TrainState(global_step=0)
    accelerator.register_for_checkpointing(train_state)
    
    # 加载检查点（若有）
    start_epoch = load_checkpoint(accelerator, config, mode="train")

    # =========================================================================
    # 第五部分：数据加载器和调度器
    # =========================================================================
    
    # 创建训练数据加载器
    train_loader = dataloader_from_config(config, accelerator)
    train_loader.sampler.set_epoch(start_epoch)

    # 配置日志/保存/可视化调度
    dirs = RunDirs.from_config(config)
    schedule = LogSaveSchedule(
        log_every_epochs=int(config.train.log_freq),
        eval_every_epochs=int(config.eval_freq),
        save_every_epochs=int(config.save_freq),
        viz_every_epochs=int(config.save_freq),
        save_visualizations=bool(config.save_visualizations),
    )
    run_logger = RunLogger(accelerator, dirs)
    saver = CheckpointSaver(accelerator, dirs)
    viz = TwoStageViz()  # 两阶段可视化缓冲区

    # =========================================================================
    # 训练主循环
    # =========================================================================
    
    for epoch in range(start_epoch, config.num_epochs):
        # 本 epoch 的指标聚合器
        epoch_logger_s2 = DiffusionNFTMetricLogger()
        
        # =====================================================================
        # 阶段 1：采样阶段 - 生成候选 mesh 并计算奖励
        # =====================================================================
        
        # 用于收集本 epoch 所有样本
        all_samples = TrellisSampleCollection()
        
        # =====================================================================
        # 设置规范化参数（用于 tex 模型训练时计算 shape_norm）
        # =====================================================================
        # Trellis2 的 shape/tex latent 都经过规范化处理：
        # - 采样时：模型输出在规范化空间，然后反规范化得到最终 latent
        # - 训练时：我们保存的是反规范化后的 latent（原始空间）
        # - tex 模型需要规范化的 shape latent 作为 concat_cond
        # 
        # 这里从 pipeline 获取规范化参数并缓存到样本集合中，
        # 使样本集合自包含，便于序列化和后续使用。
        # =====================================================================
        all_samples.set_normalization_params(
            shape_mean=torch.tensor(pipeline.shape_slat_normalization["mean"]),  # (C_shape,)
            shape_std=torch.tensor(pipeline.shape_slat_normalization["std"]),    # (C_shape,)
        )
        max_train_batches = int(config.sample.num_batches_per_epoch)
        
        # 设置采样器随机种子（每 epoch 不同以增加多样性）
        train_loader.sampler.set_epoch(epoch)
        
        # 使用有限迭代器控制每 epoch 的 batch 数量
        loader_iter = (
            itertools.islice(train_loader, max_train_batches)
            if max_train_batches > 0 else train_loader
        )
        
        # 遍历训练 batch 进行采样
        for batch_idx, (batch_images, batch_paths, batch_meta) in enumerate(tqdm(loader_iter, disable=not accelerator.is_main_process)):
            
            # 采样时使用 eval 模式和 inference_mode，关闭梯度追踪
            with EvalModeGuard(shape_model, tex_model):
                with torch.inference_mode():
                    # Step 1: 图像条件编码
                    cond_batch = pipeline.prepare_image_conditions(batch_images)  # (B, P, C)
                    k = int(config.sample.num_meshes_per_image)  # 每图生成 K 个候选
                    
                    # 可选：使用稳定随机生成器（same_latent 模式）
                    use_same_latent = config.sample.same_latent
                    generator = (
                        create_train_generator_for_batch(accelerator.device, int(epoch), int(batch_idx), list(batch_paths))
                        if use_same_latent else None
                    )
                    
                    # 将条件扩展到 BK 个候选（每图 K 个）
                    cond_bk = cond_batch.repeat_interleave(k, dim=0)  # (BK, P, C)
                    BK = cond_bk.shape[0]  # 总候选数

                    # ============================================================
                    # Step 2: 批量 Stage1 - 生成所有 BK 个样本的稀疏结构坐标
                    # ============================================================
                    coords_all, _ = pipeline.stage_1(
                        cond={
                            "cond": cond_bk,
                            "neg_cond": torch.zeros_like(cond_bk),
                        },
                        ss_resolution=64,
                        num_samples=BK,
                    )  # coords_all: (N_total, 4)，其中 coords_all[:, 0] 是 batch 索引

                    # ============================================================
                    # Step 3: 批量 Stage2 - 生成所有 BK 个样本的形状特征
                    # ============================================================
                    shape_slat_batched, _ = pipeline.stage_2_shape(
                        cond={
                            "cond": cond_bk,
                            "neg_cond": torch.zeros_like(cond_bk),
                        },
                        coords=coords_all,
                        resolution=1024,
                    )  # shape_slat_batched: SparseTensor(N_total, C)

                    # ============================================================
                    # Step 3.5: 批量纹理 latent 生成
                    # ============================================================
                    tex_slat_batched, _ = pipeline.stage_2_tex(
                        cond={
                            "cond": cond_bk,
                            "neg_cond": torch.zeros_like(cond_bk),
                        },
                        shape_slat=shape_slat_batched,
                    )  # tex_slat_batched: SparseTensor(N_total, C_tex)

                    # ============================================================
                    # Step 4: 逐样本解码为 mesh（两种类型）
                    # ============================================================
                    meshes = []           # 完整 mesh（带纹理）
                    meshes_shape = []     # Shape only（无纹理）
                    shape_slat_list = []
                    tex_slat_list = []
                    for sample_idx, sl in enumerate(shape_slat_batched.layout):
                        # 提取形状 latent
                        sample_feats = shape_slat_batched.feats[sl]
                        sample_coords = shape_slat_batched.coords[sl].clone()
                        sample_coords[:, 0] = 0  # 重置 batch 索引为 0
                        sample_slat = SparseTensor(
                            feats=sample_feats,
                            coords=sample_coords,
                            layout=[slice(0, sample_coords.shape[0])],
                        )
                        shape_slat_list.append(sample_slat)
                        
                        # 提取纹理 latent
                        tex_feats = tex_slat_batched.feats[sl]
                        tex_coords = tex_slat_batched.coords[sl].clone()
                        tex_coords[:, 0] = 0  # 重置 batch 索引为 0
                        tex_slat = SparseTensor(
                            feats=tex_feats,
                            coords=tex_coords,
                            layout=[slice(0, tex_coords.shape[0])],
                        )
                        tex_slat_list.append(tex_slat)
                        
                        # 解码 Shape only mesh（无纹理）
                        mesh_shape = pipeline.export_mesh(shape_slat=sample_slat, tex_slat=None, resolution=1024)
                        meshes_shape.append(mesh_shape)
                        
                        # 解码完整 mesh（带纹理）
                        mesh_full = pipeline.export_mesh(shape_slat=sample_slat, tex_slat=tex_slat, resolution=1024)
                        meshes.append(mesh_full)

                    # 保存 latent 用于后续训练
                    all_shape_latents = shape_slat_list
                    all_tex_latents = tex_slat_list

            # Step 5: 计算奖励
            # 扩展元数据和图像列表以匹配 BK 个 mesh
            repeated_meta = []
            for meta_item, path in zip(batch_meta, batch_paths):
                m = dict(meta_item)
                m["image_path"] = path
                repeated_meta.extend([m] * k)
            repeated_images = []
            for img in batch_images:
                repeated_images.extend([img] * k)
            
            # 调用奖励评估器
            rewards_dict, meta_out = mesh_scorer.score(meshes, repeated_images, repeated_meta, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # 加权平均奖励 (BK,)
            reward_parts_local = {k: v for k, v in rewards_dict.items() if k != "avg"}
            
            # 缓存第一个 batch 用于可视化
            if batch_idx == 0:
                repeated_paths = []
                for p in batch_paths:
                    repeated_paths.extend([p] * k)
                num_samples_to_cache = min(2, len(meshes))
                viz.update(
                    meshes_shape=meshes_shape[:num_samples_to_cache],
                    meshes_tex=meshes[:num_samples_to_cache],
                    image_paths=repeated_paths[:num_samples_to_cache],
                )

            # Step 6: 构建样本并加入集合
            all_samples.extend(
                TrellisSampleCollection.build_samples_from_generation(
                        meshes=meshes,
                        all_shape_latents=all_shape_latents,
                        all_tex_latents=all_tex_latents,
                        cond_batch=cond_batch,
                        rewards=rewards,
                        reward_parts_local=reward_parts_local,
                        batch_meta=batch_meta,
                        batch_paths=batch_paths,
                        k=k,
                    ),
            )

            # 清理显存
            del meshes, meshes_shape, all_shape_latents, all_tex_latents
            torch.cuda.empty_cache()

        # =====================================================================
        # 阶段 2：优势计算 - 按图像分组计算相对优势
        # =====================================================================
        
        accelerator.wait_for_everyone()  # 同步所有进程
        
        # 计算奖励和优势值（写入每个样本，使用 Z-score 标准化）
        all_samples.compute_rewards_and_advantages(
            reward_weights=dict(config.reward_fn),
            accelerator=accelerator,
            epoch=epoch,
        )

        # 提取奖励和优势用于统计
        filtered_samples = all_samples.as_list()
        if len(filtered_samples) == 0:
            rewards_local = np.zeros(0, dtype=np.float64)
            advantages_local = np.zeros(0, dtype=np.float64)
        else:
            rewards_local = np.array([s.reward_avg for s in filtered_samples], dtype=np.float64)
            advantages_local = np.array([s.advantage for s in filtered_samples], dtype=np.float64)

        # 计算全局平均值（跨 GPU 聚合）
        accelerator.wait_for_everyone()
        reward_mean_global = distributed_mean(rewards_local, accelerator)
        adv_mean_global = distributed_mean(advantages_local, accelerator)
        epoch_logger_s2.set_reward_and_adv_means(reward_mean_global, adv_mean_global)

        # 记录采样统计到日志
        valid_samples_ratio = all_samples.valid_ratio()
        actual_train_bs = config.train.batch_size 
        run_logger.log_sampling_stats(
            epoch=epoch,
            actual_batch_size=actual_train_bs,
            num_sub_batches=len(all_samples),
            valid_ratio=float(valid_samples_ratio),
        )

        # =====================================================================
        # 阶段 3：DiffusionNFT 训练 - 策略梯度更新
        # =====================================================================
        
        # 切换到训练模式
        set_model_adapter(shape_model, "default")
        set_model_adapter(tex_model, "default")
        shape_model.train()
        tex_model.train()

        # 准备时间步采样（均匀采样子集）
        steps_sparse = int(config.sample.num_steps)  # 总扩散步数
        frac = float(config.train.timestep_fraction)
        used_sparse = compute_timestep_usage(steps_sparse, frac)
        train_step_indices = np.linspace(0, steps_sparse - 1, used_sparse, dtype=np.int32)
        
        # DiffusionNFT 超参数
        nft_beta = float(config.nft_beta)         # NFT 插值强度
        kl_beta = float(config.train.beta)        # KL 正则化系数
        adv_clip_max = float(config.train.adv_clip_max)  # 优势裁剪边界
        max_grad_norm = float(config.train.max_grad_norm)  # 梯度裁剪

        # 内部训练循环：可多次遍历采样的样本
        for inner_epoch in range(int(config.train.num_inner_epochs)):
            batch_iter = tqdm(
                all_samples.iter_batches(actual_train_bs),
                total=(len(all_samples) + actual_train_bs - 1) // actual_train_bs,
                disable=not accelerator.is_main_process,
                desc=f"Stage2 Batches (inner {inner_epoch})",
                leave=False,
            )
            
            # 遍历样本批次
            for batch_idx, batch_samples in enumerate(batch_iter):
                # 将样本移动到 GPU 并准备训练数据（同时获取 shape 和 tex 的 sparse batch）
                cond_batched, x0_shape_batch, x0_tex_batch, routing_probs = TrellisSampleCollection.move_batch_samples(
                    batch_samples=batch_samples,
                    device=accelerator.device,
                    dtype=torch.float32,
                    adv_clip_max=adv_clip_max,
                )
                batch_size = len(batch_samples)

                # =====================================================================
                # 预计算 shape_norm 用于 tex 模型训练
                # =====================================================================
                # Tex 模型的 forward 签名：
                #   tex_model(x, t, cond, concat_cond, guidance_scale)
                # 
                # 其中 concat_cond 是规范化的 shape latent，在模型内部会与 x 拼接：
                #   if concat_cond is not None:
                #       x = sparse_cat([x, concat_cond], dim=-1)  # 64 = 32 + 32
                #
                # 由于 shape_norm 在整个时间步循环中保持不变（基于固定的 x0_shape_batch），
                # 我们在循环外预计算一次以提高效率。
                # =====================================================================
                shape_norm = all_samples.compute_shape_norm(
                    x0_shape_batch, 
                    device=accelerator.device, 
                    dtype=x0_shape_batch.feats.dtype
                )  # SparseTensor, feats 形状: (N_total, C_shape)

                # 遍历采样的时间步
                step_iter = tqdm(
                    train_step_indices,
                    total=len(train_step_indices),
                    disable=not accelerator.is_main_process,
                    desc=f"Stage2 Steps (batch {batch_idx})",
                    leave=False,
                )
                
                for t_idx in step_iter:
                    # 当前时间步的值和归一化值
                    t_value = float(t_idx)  # 直接使用索引值作为时间步
                    t_norm_value = t_value / float(steps_sparse)  # 归一化到 [0, 1)
                    t = torch.full((batch_size,), t_value, device=accelerator.device, dtype=torch.float32)  # (B,)

                    # =================================================================
                    # 阶段 1: Shape 模型训练（独立反向传播）
                    # =================================================================
                    # 
                    # 显存优化策略：分阶段训练
                    # ----------------------------
                    # 同时训练 Shape 和 Tex 两个大模型会导致 OOM，因为需要同时保留
                    # 两个模型的计算图。解决方案是分阶段训练：
                    # 
                    # 1. Shape 前向 → 损失计算 → 反向传播 → 优化器更新 → 清理计算图
                    # 2. Tex 前向 → 损失计算 → 反向传播 → 优化器更新 → 清理计算图
                    # 
                    # 这样每次只保留一个模型的计算图，显存占用减半。
                    # 两个模型使用独立的优化器，可以有不同的学习率配置。
                    # =================================================================
                    
                    # Flow Matching 前向过程（Shape）
                    t_norm_per_point_shape = torch.full((x0_shape_batch.feats.shape[0], 1), t_norm_value, 
                                                   device=x0_shape_batch.feats.device, dtype=x0_shape_batch.feats.dtype)  # (N_total, 1)
                    noise_feats_shape = torch.randn_like(x0_shape_batch.feats)  # (N_total, C_shape)
                    xt_feats_shape = x0_shape_batch.feats * (1.0 - t_norm_per_point_shape) + noise_feats_shape * t_norm_per_point_shape  # (N_total, C_shape)
                    xt_shape_sparse = sparse_clone_with_feats(x0_shape_batch, xt_feats_shape)  # 批量 SparseTensor
                    
                    # Shape 模型前向传播
                    with accelerator.autocast():
                        shape_output = shape_model(xt_shape_sparse, t, cond_batched, None, guidance_scale=1.0)  # 批量 SparseTensor

                    # Shape KL 正则化
                    if kl_beta > 0.0:
                        with torch.no_grad():
                            unwrapped_shape = accelerator.unwrap_model(shape_model)
                            with unwrapped_shape.disable_adapter():
                                shape_output_ref = unwrapped_shape(xt_shape_sparse, t, cond_batched, None, guidance_scale=1.0)
                    else:
                        shape_output_ref = None

                    # Shape DiffusionNFT 损失计算
                    t_norm_full_shape = torch.full_like(xt_shape_sparse.feats, t_norm_value)  # (N_total, C_shape)
                    shape_ref_feats = shape_output_ref.feats if shape_output_ref is not None else shape_output.feats
                    shape_positive_feats = nft_beta * shape_output.feats + (1.0 - nft_beta) * shape_ref_feats
                    shape_negative_feats = (1.0 + nft_beta) * shape_ref_feats - nft_beta * shape_output.feats
                    shape_positive_sparse = sparse_clone_with_feats(shape_output, shape_positive_feats)
                    shape_negative_sparse = sparse_clone_with_feats(shape_output, shape_negative_feats)
                    x0_shape_pos_feats = xt_shape_sparse.feats - shape_positive_sparse.feats * t_norm_full_shape
                    x0_shape_neg_feats = xt_shape_sparse.feats - shape_negative_sparse.feats * t_norm_full_shape
                    x0_shape_pos = sparse_clone_with_feats(xt_shape_sparse, x0_shape_pos_feats)
                    x0_shape_neg = sparse_clone_with_feats(xt_shape_sparse, x0_shape_neg_feats)

                    policy_shape = TrellisSampleCollection.compute_sparse_policy_loss(
                        batch_samples=batch_samples,
                        x0_pos=x0_shape_pos,
                        x0_neg=x0_shape_neg,
                        x0_ref=x0_shape_batch,
                        routing_probs=routing_probs,
                        nft_beta=nft_beta,
                        mode="self",
                    )
                    shape_policy_loss = (policy_shape.policy_vec * adv_clip_max).mean()
                    shape_kl_loss = sparse_batch_mse(shape_output, shape_output_ref).mean() if shape_output_ref is not None else torch.zeros(1, device=accelerator.device)
                    
                    # Shape 总损失并反向传播
                    shape_total_loss = shape_policy_loss + kl_beta * shape_kl_loss
                    accelerator.backward(shape_total_loss)
                    
                    # Shape 优化器更新
                    if accelerator.sync_gradients:
                        if max_grad_norm > 0.0:
                            accelerator.clip_grad_norm_(shape_model.parameters(), max_grad_norm)
                        optimizer_shape.step()
                        optimizer_shape.zero_grad(set_to_none=True)
                    
                    # 保存 shape 指标用于日志（detach 确保不保留计算图）
                    shape_pos_mean = policy_shape.pos_mean.detach()
                    shape_neg_mean = policy_shape.neg_mean.detach()
                    shape_policy_loss_val = shape_policy_loss.detach()
                    shape_kl_loss_val = shape_kl_loss.detach()
                    
                    # =========================================================
                    # 清理 Shape 计算图，释放显存供 Tex 模型使用
                    # =========================================================
                    # 关键：在开始 Tex 训练前必须清理 Shape 的所有中间张量，
                    # 否则显存会累积导致 OOM。
                    # =========================================================
                    del shape_output, shape_output_ref, xt_shape_sparse
                    del shape_positive_sparse, shape_negative_sparse
                    del x0_shape_pos, x0_shape_neg
                    torch.cuda.empty_cache()

                    # =================================================================
                    # 阶段 2: Tex 模型训练（独立反向传播）
                    # =================================================================
                    # 
                    # Tex 模型与 Shape 模型的关键区别：
                    # ---------------------------------
                    # 1. **输入维度**：Tex 输入需要与 shape_norm 拼接
                    #    - xt_tex: (N_total, 32) - tex latent 的噪声版本
                    #    - concat_cond: (N_total, 32) - 规范化的 shape latent
                    #    - 模型内部拼接后: (N_total, 64)
                    # 
                    # 2. **条件依赖**：Tex 生成依赖于 Shape 生成结果
                    #    - 采样时：先生成 shape_slat，再基于它生成 tex_slat
                    #    - 训练时：使用预计算的 shape_norm 作为固定条件
                    # 
                    # 3. **训练目标**：学习给定 shape 条件下的 tex 分布
                    # =================================================================
                    
                    # Flow Matching 前向过程（Tex）
                    t_norm_per_point_tex = torch.full((x0_tex_batch.feats.shape[0], 1), t_norm_value, 
                                                   device=x0_tex_batch.feats.device, dtype=x0_tex_batch.feats.dtype)  # (N_total, 1)
                    noise_feats_tex = torch.randn_like(x0_tex_batch.feats)  # (N_total, C_tex)
                    xt_feats_tex = x0_tex_batch.feats * (1.0 - t_norm_per_point_tex) + noise_feats_tex * t_norm_per_point_tex  # (N_total, C_tex)
                    xt_tex_sparse = sparse_clone_with_feats(x0_tex_batch, xt_feats_tex)  # 批量 SparseTensor
                    
                    # Tex 模型前向传播（需要 shape_norm 作为 concat_cond）
                    # 参数说明：
                    #   - xt_tex_sparse: 噪声化的 tex latent，形状 (N_total, C_tex=32)
                    #   - t: 时间步，形状 (B,)
                    #   - cond_batched: 图像条件 embeddings，形状 (B, P, C)
                    #   - shape_norm: 规范化的 shape latent（concat_cond），形状 (N_total, C_shape=32)
                    #   - 模型输出: velocity 预测，形状 (N_total, C_tex=32)
                    with accelerator.autocast():
                        tex_output = tex_model(xt_tex_sparse, t, cond_batched, shape_norm, guidance_scale=1.0)  # SparseTensor(N_total, C_tex)

                    # Tex KL 正则化
                    if kl_beta > 0.0:
                        with torch.no_grad():
                            unwrapped_tex = accelerator.unwrap_model(tex_model)
                            with unwrapped_tex.disable_adapter():
                                tex_output_ref = unwrapped_tex(xt_tex_sparse, t, cond_batched, shape_norm, guidance_scale=1.0)
                    else:
                        tex_output_ref = None

                    # Tex DiffusionNFT 损失计算
                    t_norm_full_tex = torch.full_like(xt_tex_sparse.feats, t_norm_value)  # (N_total, C_tex)
                    tex_ref_feats = tex_output_ref.feats if tex_output_ref is not None else tex_output.feats
                    tex_positive_feats = nft_beta * tex_output.feats + (1.0 - nft_beta) * tex_ref_feats
                    tex_negative_feats = (1.0 + nft_beta) * tex_ref_feats - nft_beta * tex_output.feats
                    tex_positive_sparse = sparse_clone_with_feats(tex_output, tex_positive_feats)
                    tex_negative_sparse = sparse_clone_with_feats(tex_output, tex_negative_feats)
                    x0_tex_pos_feats = xt_tex_sparse.feats - tex_positive_sparse.feats * t_norm_full_tex
                    x0_tex_neg_feats = xt_tex_sparse.feats - tex_negative_sparse.feats * t_norm_full_tex
                    x0_tex_pos = sparse_clone_with_feats(xt_tex_sparse, x0_tex_pos_feats)
                    x0_tex_neg = sparse_clone_with_feats(xt_tex_sparse, x0_tex_neg_feats)

                    policy_tex = TrellisSampleCollection.compute_sparse_policy_loss(
                        batch_samples=batch_samples,
                        x0_pos=x0_tex_pos,
                        x0_neg=x0_tex_neg,
                        x0_ref=x0_tex_batch,
                        routing_probs=routing_probs,
                        nft_beta=nft_beta,
                        mode="self",
                    )
                    tex_policy_loss = (policy_tex.policy_vec * adv_clip_max).mean()
                    tex_kl_loss = sparse_batch_mse(tex_output, tex_output_ref).mean() if tex_output_ref is not None else torch.zeros(1, device=accelerator.device)

                    # Tex 总损失并反向传播
                    tex_total_loss = tex_policy_loss + kl_beta * tex_kl_loss
                    accelerator.backward(tex_total_loss)
                    
                    # Tex 优化器更新
                    if accelerator.sync_gradients:
                        if max_grad_norm > 0.0:
                            accelerator.clip_grad_norm_(tex_model.parameters(), max_grad_norm)
                        optimizer_tex.step()
                        optimizer_tex.zero_grad(set_to_none=True)
                    
                    accelerator.wait_for_everyone()

                    # =================================================================
                    # 合并日志指标
                    # =================================================================
                    
                    policy_loss_self = shape_policy_loss_val + tex_policy_loss.detach()
                    policy_loss = policy_loss_self
                    policy_loss_cross = torch.tensor(0.0, device=accelerator.device)
                    kl_loss = shape_kl_loss_val + tex_kl_loss.detach()
                    pos_mean = (shape_pos_mean + policy_tex.pos_mean.detach()) / 2.0
                    neg_mean = (shape_neg_mean + policy_tex.neg_mean.detach()) / 2.0

                    # 更新训练状态和 EMA
                    train_state.global_step += 1
                    if bool(config.train.ema) and ema_stage2 is not None:
                        ema_stage2.step(all_trainable_params, train_state.global_step)

                    # 记录指标
                    epoch_logger_s2.update(
                        policy_loss,
                        policy_loss_self,
                        policy_loss_cross,
                        pos_mean,
                        neg_mean,
                        kl_loss,
                        batch_size=len(batch_samples),
                    )
                    
                    # 清理 tex 计算图释放显存
                    del tex_output, tex_output_ref, xt_tex_sparse
                    del tex_positive_sparse, tex_negative_sparse
                    del x0_tex_pos, x0_tex_neg
                    torch.cuda.empty_cache()

        # =====================================================================
        # 阶段 4：Epoch 结束处理 - 日志、评估、保存
        # =====================================================================
        
        accelerator.wait_for_everyone()
        
        # 记录训练指标到 W&B
        if (epoch % max(1, schedule.log_every_epochs) == 0):
            run_logger.log_epoch_metrics_prefixed(epoch, epoch_logger_s2, "stage2")

        # 周期性评估
        if int(config.eval_freq) > 0 and (epoch % int(config.eval_freq) == 0):
            accelerator.wait_for_everyone()
            eval_loader = eval_dataloader_from_config(config, accelerator)
            eval_loader.sampler.set_epoch(epoch)
            
            # 创建固定生成器确保评估可复现
            gen = create_eval_generator(accelerator.device, int(config.seed))
            trainable = None
            if bool(config.train.ema) and ema_stage2 is not None:
                trainable = all_trainable_params
            
            # 使用 EMA 权重进行评估（如启用）
            if bool(config.train.ema) and ema_stage2 is not None:
                ema_stage2.copy_ema_to(trainable, store_temp=True)  # 切换到 EMA
                all_rewards_np = eval_trellis(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
                ema_stage2.copy_temp_to(trainable)  # 恢复原始权重
            else:
                all_rewards_np = eval_trellis(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
            accelerator.wait_for_everyone()
            run_logger.log_eval_rewards(epoch, all_rewards_np)

        # 周期性保存检查点
        if (epoch % int(schedule.save_every_epochs) == 0):
            saver.save_epoch(
                epoch=epoch,
                config=config,
                ema=(ema_stage2 if bool(config.train.ema) and ema_stage2 is not None else None),
                use_lora=bool(config.use_lora),
            )

        # 可视化输出（仅主进程）
        if schedule.save_visualizations and (epoch % int(schedule.viz_every_epochs) == 0) and viz.tex.meshes is not None:
            viz_dir = dirs.viz_dir / f"epoch_{epoch+1}"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存两阶段 PBR 预览并上传到 W&B
            viz.save_and_log(pipeline, accelerator, viz_dir, epoch)

        # 清理本 epoch 的样本缓存
        all_samples.clear()
        gc.collect()


# =============================================================================
# 辅助数据类：调度、目录、缓冲、日志、保存
# =============================================================================

@dataclass
class LogSaveSchedule:
    """日志和保存的频率调度配置。"""
    log_every_epochs: int      # 多少个 epoch 记录一次训练指标
    eval_every_epochs: int     # 多少个 epoch 评估一次
    save_every_epochs: int     # 多少个 epoch 保存一次检查点
    viz_every_epochs: int      # 多少个 epoch 保存一次可视化
    save_visualizations: bool  # 是否保存可视化


@dataclass
class RunDirs:
    """实验运行的目录结构。
    
    目录布局：
        {logdir}/{run_name}/
        ├── checkpoints/        # 检查点
        │   ├── checkpoint_0/
        │   ├── checkpoint_5/
        │   └── ...
        └── generated_meshes/   # 可视化输出
            ├── epoch_1/
            ├── eval_epoch_0/
            └── ...
    """
    run_dir: Path    # 实验根目录
    ckpt_dir: Path   # 检查点目录
    viz_dir: Path    # 可视化输出目录

    @staticmethod
    def from_config(config) -> "RunDirs":
        """从配置创建目录结构。"""
        run_name_dir = config.run_name if isinstance(config.run_name, str) and len(config.run_name) > 0 else "trellis_s2"
        run_dir = Path(config.logdir) / run_name_dir
        return RunDirs(
            run_dir=run_dir,
            ckpt_dir=run_dir / "checkpoints",
            viz_dir=run_dir / "generated_meshes",
        )


@dataclass
class StageVizBuffer:
    """单阶段可视化缓冲区。"""
    stage_name: str                          # "shape" | "tex"
    meshes: Optional[List] = None            # MeshWithVoxel 列表
    image_paths: Optional[List[str]] = None

    def update(self, meshes: List, image_paths: List[str]) -> None:
        self.meshes = meshes
        self.image_paths = image_paths

    def save_previews(self, pipeline, base_dir: Path, resolution: int = 512) -> List[str]:
        """渲染并保存 PBR 预览图。"""
        if not self.meshes:
            return []
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        files = []
        for idx, (mesh, img_path) in enumerate(zip(self.meshes, self.image_paths)):
            name = Path(img_path).stem
            path = base_dir / f"{name}_{idx}_{self.stage_name}.png"
            
            result = pipeline.render_pbr_snapshot(mesh, resolution=resolution, nviews=4)
            panel = pipeline.make_pbr_vis_panel(result, resolution=resolution)[0]
            Image.fromarray(panel).save(path)
            files.append(str(path))
        
        return files

    def log_to_wandb(self, accelerator, epoch: int, files: List[str]) -> None:
        """上传预览图到 W&B。"""
        if not accelerator.is_main_process or not files:
            return
        accelerator.log({
            f"mesh/{self.stage_name}": [wandb.Image(f) for f in files]
        }, step=epoch + 1)


@dataclass 
class TwoStageViz:
    """两阶段可视化管理器（shape + tex）。"""
    shape: StageVizBuffer = None
    tex: StageVizBuffer = None
    
    def __post_init__(self):
        self.shape = StageVizBuffer("shape")
        self.tex = StageVizBuffer("tex")

    def update(
        self,
        meshes_shape: List,
        meshes_tex: List,
        image_paths: List[str],
    ) -> None:
        """更新两个阶段的缓冲区。"""
        self.shape.update(meshes_shape, image_paths)
        self.tex.update(meshes_tex, image_paths)

    def save_and_log(self, pipeline, accelerator, base_dir: Path, epoch: int, resolution: int = 512) -> None:
        """保存并上传所有阶段的预览。"""
        for stage in [self.shape, self.tex]:
            files = stage.save_previews(pipeline, base_dir, resolution)
            stage.log_to_wandb(accelerator, epoch, files)


class RunLogger:
    """W&B 日志记录器，封装各类指标的记录逻辑。
    
    所有日志方法都会检查是否为主进程，避免重复记录。
    """
    
    def __init__(self, accelerator: Accelerator, dirs: RunDirs):
        self.accelerator = accelerator
        self.dirs = dirs

    def print(self, msg: str):
        """打印消息（仅主进程）。"""
        self.accelerator.print(msg)

    def log_sampling_stats(self, epoch: int, actual_batch_size: int, num_sub_batches: int, valid_ratio: float):
        """记录采样阶段的统计信息。"""
        self.accelerator.log(
            {
                "actual_batch_size": int(actual_batch_size),
                "num_sub_batches": int(num_sub_batches),
                "valid_samples_ratio": float(valid_ratio),
            },
            step=epoch,
        )

    def log_epoch_metrics(self, epoch: int, epoch_logger: "DiffusionNFTMetricLogger"):
        """记录 epoch 训练指标。"""
        log_dict = epoch_logger.to_global_log_dict(self.accelerator)
        if self.accelerator.is_main_process and log_dict is not None:
            self.accelerator.log(log_dict, step=epoch + 1)

    def log_epoch_metrics_prefixed(self, epoch: int, epoch_logger: "DiffusionNFTMetricLogger", prefix: str):
        """记录 epoch 训练指标（带前缀，如 "stage2/"）。"""
        log_dict = epoch_logger.to_global_log_dict(self.accelerator)
        if self.accelerator.is_main_process and log_dict is not None:
            renamed = {}
            for k, v in log_dict.items():
                if k.startswith("epoch/"):
                    renamed[f"epoch/{prefix}/" + k[len("epoch/"):]] = v
                else:
                    renamed[f"{prefix}/" + k] = v
            self.accelerator.log(renamed, step=epoch + 1)

    def log_eval_rewards(self, epoch: int, all_rewards_np: Dict[str, np.ndarray]):
        """记录评估阶段的奖励指标。"""
        if self.accelerator.is_main_process:
            metrics = {f"eval_reward_{k}": (float(v.mean()) if v.size > 0 else 0.0) for k, v in all_rewards_np.items()}
            self.accelerator.log(metrics, step=epoch + 1)

    def log_mesh_previews(self, epoch: int, preview_files: List[str], image_paths: List[str]):
        """上传 mesh 预览图到 W&B。"""
        if self.accelerator.is_main_process and len(preview_files) > 0:
            import wandb
            self.accelerator.log(
                {
                    "mesh_previews": [
                        wandb.Image(preview_files[i], caption=os.path.basename(image_paths[i]))
                        for i in range(len(preview_files))
                    ]
                },
                step=epoch + 1,
            )


class CheckpointSaver:
    """检查点保存器，使用 Accelerate 的状态保存机制。
    
    保存内容包括：
    - 模型权重（包括 LoRA adapter）
    - 优化器状态
    - 学习率调度器状态（如有）
    - 随机数生成器状态
    - 自定义注册状态（EMA, TrainState）
    """
    
    def __init__(self, accelerator: Accelerator, dirs: RunDirs):
        self.accelerator = accelerator
        self.dirs = dirs

    def save_epoch(self, epoch: int, config: ml_collections.ConfigDict, ema: Optional[Any] = None, use_lora: bool = False):
        """保存指定 epoch 的检查点。
        
        多卡注意事项：
        - 所有 rank 都参与保存（保证 RNG 状态一致）
        - 使用 wait_for_everyone 同步避免竞态
        - 主进程负责目录创建和清理
        """
        # 等待所有 rank 对齐
        self.accelerator.wait_for_everyone()
        checkpoint_dir = self.dirs.ckpt_dir / f"checkpoint_{epoch}"
        
        # 仅主进程创建/清理目录
        if self.accelerator.is_main_process:
            self.dirs.ckpt_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        
        # 同步后保存
        self.accelerator.wait_for_everyone()
        # 所有 rank 都调用 save_state（保证 RNG 状态写入）
        self.accelerator.save_state(output_dir=str(checkpoint_dir))
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"💾 Saved (Accelerate): {str(checkpoint_dir)}")




if __name__ == "__main__":
    app.run(main)
