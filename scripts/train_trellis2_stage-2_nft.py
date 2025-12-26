#!/usr/bin/env python3
"""
Trellis Stage 2 GRPO Training Script

- 两阶段：Stage 1 冻结在线生成稀疏坐标；Stage 2 对 SLatFlowModel 做 GRPO
- 复用 Hunyuan3D/SD3 的 GRPO 训练框架与指标定义
- 稀疏张量：coords(N,4) + feats(N,C)，接入 Flow Matching + SDE + LogProb
- 约束：仅训练 Stage 2，无 try/except，无 fallback
"""

import os
import sys
import gc
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence

# ===== CUDA 内存优化配置 =====
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

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 使 VGGTObj 参考渲染器的相对导入可用
_vggt_root = project_root / "_reference_codes" / "VGGTObj"
if str(_vggt_root) not in sys.path:
    sys.path.insert(0, str(_vggt_root))

# TRELLIS.2 参考代码路径（替代 TRELLIS 1.x）
_trellis2_root = project_root / "_reference_codes" / "TRELLIS.2"
if str(_trellis2_root) not in sys.path:
    sys.path.insert(0, str(_trellis2_root))
# o-voxel 子包路径
_ovoxel_root = _trellis2_root / "o-voxel"
if str(_ovoxel_root) not in sys.path:
    sys.path.insert(0, str(_ovoxel_root))

# 参考渲染器与 mesh 适配器（用于可视化四视角法线渲染）
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from reward_models.camera_normal_scorer.render.adapter import to_mesh_extract, KiuiMeshLike

# 导入 Trellis2/GRPO 相关模块（使用 TRELLIS.2 代码路径）
from flow_grpo.diffusers_patch.trellis2_pipeline_with_logprob import (
    Trellis2PipelineWithLogProb,
)
# SparseTensor 使用 trellis2 原生实现
from trellis2.modules.sparse import SparseTensor
from flow_grpo.diffusers_patch.trellis2_sparse_tensor import (
    prepare_sparse_tensor_batch,
    sparse_batch_mse,
    sparse_clone_with_feats,
    compute_sparse_weighted_mse,
)
from flow_grpo.ema import EMAModuleWrapper
from reward_models.rewards_mesh import MeshScorer
# 工具函数改为直接使用 pipeline 的 preprocess_image

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config")
from peft import LoraConfig, get_peft_model, PeftModel
from flow_grpo.peft_sparse.sparse_lora_trellis2 import register_sparse_linear_with_peft
from dataclasses import dataclass
import itertools


def compute_timestep_usage(num_steps: int, fraction: float, keep_ratio: float) -> Tuple[int, int]:
    """统一计算时间步采样数量，返回 (used_steps, keep_steps)。"""
    total_steps = max(1, int(num_steps))
    frac = max(0.0, float(fraction))
    keep = max(0.0, float(keep_ratio))
    used_steps = max(1, int(frac * total_steps))
    keep_steps = max(1, int(keep * used_steps))
    keep_steps = min(used_steps, keep_steps)
    return used_steps, keep_steps


def _unwrap_model(model: nn.Module) -> nn.Module:
    """提取加速器/并行包装内的真实模型。"""
    return model.module if hasattr(model, "module") else model


def set_model_adapter(model: nn.Module, adapter_name: str) -> None:
    """为任意 LoRA/PEFT 模型设置当前 adapter。"""
    target = _unwrap_model(model)
    if hasattr(target, "set_adapter"):
        target.set_adapter(adapter_name)


def setup_backend_determinism() -> None:
    """配置后端为确定性模式，尽量减少非确定性波动。"""
    torch.backends.cudnn.benchmark = False  # 标量
    torch.backends.cudnn.deterministic = True  # 标量


def create_eval_generator(device: torch.device, seed: int) -> torch.Generator:
    """创建评估用固定生成器，所有 rank 使用完全相同种子。"""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def create_train_generator_for_batch(
    device: torch.device,
    epoch: int,
    batch_idx: int,
    image_paths: List[str],
) -> torch.Generator:
    """为训练采样创建稳定的生成器（same_latent）。

    - 基于 (epoch, batch_idx) 与当前批的图像路径稳定哈希生成种子，确保可复现。
    - 返回单个 torch.Generator，用于本批次 Stage1/Stage2 的随机数流。
    """
    joined = "||".join(image_paths)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    batch_hash = int.from_bytes(digest[:4], byteorder="big", signed=False)
    base_seed = (epoch * 10000 + int(batch_idx)) % (2**31)
    seed = (base_seed + batch_hash) % (2**31)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


class DiffusionNFTMetricLogger:
    """DiffusionNFT 版本的指标聚合器，跟踪 policy(self/cross)/kl 与正负损失均值。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sum_policy = 0.0  # 标量累计
        self.sum_policy_self = 0.0  # 标量累计
        self.sum_policy_cross = 0.0  # 标量累计
        self.sum_positive = 0.0  # 标量累计
        self.sum_negative = 0.0  # 标量累计
        self.sum_kl = 0.0  # 标量累计
        self.count = 0.0  # 标量累计
        self.reward_mean = 0.0  # 标量累计
        self.adv_mean = 0.0  # 标量累计

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
        bs_val = float(batch_size)
        self.sum_policy += float(policy_loss.detach().item()) * bs_val
        self.sum_policy_self += float(policy_loss_self.detach().item()) * bs_val
        self.sum_policy_cross += float(policy_loss_cross.detach().item()) * bs_val
        self.sum_positive += float(positive_loss.detach().item()) * bs_val
        self.sum_negative += float(negative_loss.detach().item()) * bs_val
        self.sum_kl += float(kl_loss.detach().item()) * bs_val
        self.count += bs_val

    def set_reward_and_adv_means(self, reward_mean: float, adv_mean: float) -> None:
        self.reward_mean = float(reward_mean)
        self.adv_mean = float(adv_mean)

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
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
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        denom = float(local[6].item())
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


def log_normal_similarity_pairs(accelerator: "Accelerator", pairs, step: int, prefix: str = "camera_normal", max_pairs: int = 4):
    """将法线相似度的配对（图像侧法线 vs 渲染法线）记录到 W&B。

    - pairs: List[Dict]，每项包含 keys: "image_path", "image_normal_pil", "rendered_normal_pil", "mesh_index", "score"
    - step: 日志步
    - prefix: 指标前缀，例如 "camera_normal" 或 "eval/camera_normal"
    - max_pairs: 限制上传条目数，避免过大日志
    """
    if (not accelerator.is_main_process) or (pairs is None) or (len(pairs) == 0):
        return
    panel_images = []
    limit = min(int(max_pairs), len(pairs))
    for rec in pairs[:limit]:
        left = rec["image_normal_pil"]  # PIL(R,R,3)
        right = rec["rendered_normal_pil"]  # PIL(R,R,3)
        W = left.width + right.width
        H = max(left.height, right.height)
        panel = Image.new("RGB", (W, H))
        panel.paste(left, (0, 0))
        panel.paste(right, (left.width, 0))
        cap = f"img: {rec['image_path']} | mesh_idx: {rec['mesh_index']} | score: {rec['score']:.4f}"
        panel_images.append(wandb.Image(panel, caption=cap))
    if len(panel_images) > 0:
        accelerator.log({
            f"{prefix}/pairs": panel_images
        }, step=step)


def name_to_stable_id(name: str) -> int:
    """将字符串名称映射为跨进程稳定的 63-bit 正整型 ID。"""
    h = hashlib.md5(name.encode("utf-8")).digest()  # 形状: 16字节
    return int.from_bytes(h[:8], byteorder="big", signed=False) & 0x7fffffffffffffff  # 形状: 标量


def compute_advantages_per_image(
    image_names: List[str],
    rewards_np_local: np.ndarray,
    accelerator: Accelerator,
    epoch: int,
) -> np.ndarray:
    """按图像分组计算优势，并记录与 Hunyuan3D 一致的统计。

    形状约定：
        - N: 当前进程样本数（通常 N = B_local*K）
        - K: 每图候选数
        - G: 进程数（全局样本总数为 G*N）

    返回：本地优势向量，形状 (N,)，顺序与 `image_names`/`rewards_np_local` 对齐。
    """
    device = accelerator.device  # 形状: 标量

    # 直接构造 torch.long 避免 numpy 溢出
    image_ids_list = [name_to_stable_id(n) for n in image_names]  # 长度 N
    image_ids_local_tensor = torch.tensor(image_ids_list, device=device, dtype=torch.long)  # 形状: (N,)
    rewards_local_tensor = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # 形状: (N,)

    # 仅基于本 rank 当前批，按图像分组 z-score；不进行全局 gather
    sort_vals, sort_idx = torch.sort(image_ids_local_tensor)  # 形状: (N,), (N,)
    rewards_sorted = rewards_local_tensor.index_select(0, sort_idx)  # 形状: (N,)
    unique_ids, counts = torch.unique(sort_vals, return_counts=True)  # 形状: (B,), (B,)
    B = int(unique_ids.numel())  # 形状: 标量
    K = int(counts[0].item())  # 形状: 标量
    assert int(counts.min().item()) == K and int(counts.max().item()) == K
    scores_bk = rewards_sorted.reshape(B, K)  # 形状: (B,K)
    mean_b = scores_bk.mean(dim=1, keepdim=True)  # 形状: (B,1)
    std_b = scores_bk.std(dim=1, keepdim=True)  # 形状: (B,1)
    advantages_bk = (scores_bk - mean_b) / (std_b + 1e-8)  # 形状: (B,K)
    advantages_sorted = advantages_bk.reshape(B * K)  # 形状: (N,)
    inv_idx = torch.empty_like(sort_idx)  # 形状: (N,)
    inv_idx[sort_idx] = torch.arange(B * K, device=advantages_sorted.device, dtype=torch.long)  # 形状: (N,)
    advantages_local_tensor = advantages_sorted.index_select(0, inv_idx)  # 形状: (N,)

    return advantages_local_tensor.detach().cpu().numpy().astype(np.float64)  # 形状: (N,)

class EvalModeGuard:
    """记录模块原始 training 状态，进入上下文时设为 eval，退出时恢复。"""
    def __init__(self, *modules: nn.Module):
        self.modules = [m for m in modules if m is not None]
        self.states: List[bool] = []

    def __enter__(self):
        self.states = [m.training for m in self.modules]
        for module in self.modules:
            module.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, was_training in zip(self.modules, self.states):
            module.train(was_training)




def compute_winrate_advantages_per_image(
    image_names: List[str],
    rewards_np_local: np.ndarray,
    accelerator: Accelerator,
    plus: bool = False,
) -> np.ndarray:
    """按图像计算"硬排名胜率优势"（winrate-0.5），分布式聚合后切回本地。

    形状约定：
        - N: 当前进程样本数（通常 N = B_local*K）
        - K: 每图候选数
        - G: 进程数（全局维度为 G*N）

    - 无 stat_tracker: 基于当前组内 K 个候选 严格胜出数/(K-1) - 0.5（平局计 0）
    - 有 stat_tracker: 将历史分数并入对手池 严格胜出数/(K-1+H) - 0.5（平局计 0）
    返回：优势 (N,)，与 `image_names`/`rewards_np_local` 对齐。
    """
    device = accelerator.device  # 形状: 标量

    # 名称 -> 稳定 id（仅本 rank）
    name_ids_list = [name_to_stable_id(n) for n in image_names]  # 形状: 长度 N
    image_ids_local = torch.tensor(name_ids_list, device=device, dtype=torch.long)  # 形状: (N,)
    rewards_local = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # 形状: (N,)

    # 仅用当前 K 候选做硬排名胜率（自动推断 K 并断言所有组大小一致），本 rank 本地
    sort_vals, sort_idx = torch.sort(image_ids_local)  # 形状: (N,), (N,)
    rewards_sorted = rewards_local.index_select(0, sort_idx)  # 形状: (N,)
    unique_ids_sorted, counts = torch.unique(sort_vals, return_counts=True)  # 形状: (≈N/K,), (≈N/K,)
    total = int(rewards_sorted.numel())  # 形状: 标量 (N)
    B = int(unique_ids_sorted.numel())  # 形状: 标量 (≈N/K)
    K = int(counts[0].item())  # 形状: 标量 (K)
    assert int(counts.min().item()) == K and int(counts.max().item()) == K
    assert total == B * K

    scores_bk = rewards_sorted.reshape(B, K)  # 形状: ((N)/K, K)
    diff = scores_bk.unsqueeze(2) - scores_bk.unsqueeze(1)  # 形状: ((N)/K, K, K)
    win_strict = (diff > 0).float()  # 形状: ((N)/K, K, K)
    wins = win_strict  # 形状: ((N)/K, K, K)
    eye = torch.eye(K, device=device, dtype=torch.float32).unsqueeze(0)  # 形状: (1,K,K)
    wins = wins * (1.0 - eye)  # 形状: ((G*N)/K, K, K)
    wr = wins.sum(dim=2) / max(1, K - 1)  # 形状: ((G*N)/K, K)
    if plus:
        adv_bk = wr  # 形状: ((G*N)/K, K)
    else:
        adv_bk = wr - 0.5  # 形状: ((G*N)/K, K)

    adv_sorted = adv_bk.reshape(total)  # 形状: (N)
    inv_idx = torch.empty_like(sort_idx)  # 形状: (N,)
    inv_idx[sort_idx] = torch.arange(total, device=device, dtype=torch.long)  # 形状: (N,)
    adv_local = adv_sorted.index_select(0, inv_idx)  # 形状: (N,)

    return adv_local.detach().cpu().numpy().astype(np.float64)  # 形状: (N,)



def distributed_mean(values_np: np.ndarray, accelerator: Accelerator) -> float:
    """分布式求均值；当输入为空时返回 0。"""
    if values_np.size == 0:
        return 0.0
    vals = torch.as_tensor(values_np, device=accelerator.device, dtype=torch.float32)  # 形状: (N,)
    total = vals.sum()  # 形状: ()
    count = torch.tensor(float(vals.numel()), device=accelerator.device, dtype=torch.float32)  # 形状: ()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    denom = float(count.item())
    if denom <= 0.0:
        return 0.0
    return float((total / max(denom, 1e-8)).item())


@dataclass
class TrellisSample:
    x0_sparse: SparseTensor
    cond_patches: torch.Tensor
    neg_patches: Optional[torch.Tensor]
    reward_components: Dict[str, float]
    reward_avg: float
    advantage: float
    image_name: str
    image_path: str


@dataclass
class PolicyLossResult:
    policy_vec: torch.Tensor
    pos_mean: torch.Tensor
    neg_mean: torch.Tensor


class TrellisSampleCollection:
    """管理 TrellisSample 的容器，提供筛选、批次与统计等工具。"""

    def __init__(self):
        self._samples: "OrderedDict[int, List[TrellisSample]]" = OrderedDict()

    def add(self, sample: TrellisSample) -> None:
        key = name_to_stable_id(sample.image_name)
        if key not in self._samples:
            self._samples[key] = []
        self._samples[key].append(sample)

    def extend(self, samples: List[TrellisSample]) -> None:
        for sample in samples:
            self.add(sample)

    def __len__(self) -> int:
        return sum(len(v) for v in self._samples.values())

    def __iter__(self):
        for samples in self._samples.values():
            for sample in samples:
                yield sample

    def as_list(self) -> List[TrellisSample]:
        return [sample for samples in self._samples.values() for sample in samples]

    def iter_batches(self, batch_size: int):
        flat = self.as_list()
        for start in range(0, len(flat), batch_size):
            yield flat[start:start + batch_size]

    def __getitem__(self, item):
        flat = self.as_list()
        return flat[item]

    def clear(self) -> None:
        self._samples.clear()

    def select_top_bottom(self, k: int) -> int:
        """针对每张图像仅保留 reward 最高与最低的 k 个样本，返回保留总数。"""
        k_val = int(k)
        if k_val <= 0 or len(self) == 0:
            return len(self)

        for key, image_samples in list(self._samples.items()):
            image_samples.sort(key=lambda s: s.reward_avg)
            if len(image_samples) <= 2 * k_val:
                continue
            else:
                self._samples[key] = image_samples[:k_val] + image_samples[-k_val:]

        return len(self)

    def select_top(self, k: int) -> int:
        """针对每张图像仅保留 reward 最高的 k 个样本，返回保留总数。"""
        k_val = int(k)
        if k_val <= 0 or len(self) == 0:
            return len(self)

        for key, image_samples in list(self._samples.items()):
            if len(image_samples) <= k_val:
                continue
            image_samples.sort(key=lambda s: s.reward_avg, reverse=True)
            self._samples[key] = image_samples[:k_val]

        return len(self)

    def valid_ratio(self) -> float:
        flat = self.as_list()
        total = len(flat)
        if total == 0:
            return 0.0
        non_zero = sum(1 for s in flat if abs(s.advantage) > 0.0)
        return float(non_zero) / float(total)

    def compute_rewards_and_advantages(
        self,
        reward_weights: Dict[str, float],
        adv_type: str,
        adv_from: str,
        accelerator: Accelerator,
        epoch: int,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        flat_samples = self.as_list()
        N_local = len(flat_samples)
        image_names = [s.image_name for s in flat_samples]
        if N_local == 0:
            return image_names, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        weights_dict = dict(reward_weights)
        enabled_keys = [k for k, v in weights_dict.items() if float(v) > 0.0]

        rewards_local = np.zeros(N_local, dtype=np.float64)
        advantages_local = np.zeros(N_local, dtype=np.float64)

        if adv_from in ("average",):
            rewards_local = np.array([s.reward_avg for s in flat_samples], dtype=np.float64)
            advantages_local = self.compute_advantage_vector(
                image_names=image_names,
                rewards_np=rewards_local,
                adv_type=adv_type,
                accelerator=accelerator,
                epoch=epoch,
            )
        elif adv_from in ("seperate",):
            for k in enabled_keys:
                w = float(weights_dict[k])
                v_k = np.array([s.reward_components[k] for s in flat_samples], dtype=np.float64)
                adv_k = self.compute_advantage_vector(
                    image_names=image_names,
                    rewards_np=v_k,
                    adv_type=adv_type,
                    accelerator=accelerator,
                    epoch=epoch,
                )
                rewards_local += w * v_k
                advantages_local += w * adv_k
        else:
            raise ValueError(f"Invalid adv_from: {adv_from}")

        for sample, reward_val, adv_val in zip(flat_samples, rewards_local.tolist(), advantages_local.tolist()):
            sample.reward_avg = float(reward_val)
            sample.advantage = float(adv_val)

        return image_names, rewards_local, advantages_local

    @staticmethod
    def compute_advantage_vector(
        image_names: List[str],
        rewards_np: np.ndarray,
        adv_type: str,
        accelerator: Accelerator,
        epoch: int,
    ) -> np.ndarray:
        if adv_type == "winrate":
            return compute_winrate_advantages_per_image(
                image_names=image_names,
                rewards_np_local=rewards_np,
                accelerator=accelerator,
            )
        if adv_type == "winrate_plus":
            return compute_winrate_advantages_per_image(
                image_names=image_names,
                rewards_np_local=rewards_np,
                accelerator=accelerator,
                plus=True,
            )
        if adv_type == "similarity":
            return compute_advantages_per_image(
                image_names=image_names,
                rewards_np_local=rewards_np,
                accelerator=accelerator,
                epoch=epoch,
            )
        raise ValueError(f"Invalid adv_type: {adv_type}")


    @staticmethod
    def move_batch_samples(
        batch_samples: List[TrellisSample],
        device: torch.device,
        dtype: torch.dtype,
        adv_clip_max: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], SparseTensor, torch.Tensor]:
        cond_batched = torch.cat([s.cond_patches.to(device=device, dtype=dtype) for s in batch_samples], dim=0)
        neg_sources = [s.neg_patches for s in batch_samples]
        neg_batched = (
            torch.cat([n.to(device=device, dtype=dtype) for n in neg_sources], dim=0)
            if all(n is not None for n in neg_sources)
            else None
        )
        sparse_list = [s.x0_sparse.to(device=device, dtype=dtype) for s in batch_samples]
        x0_batch: SparseTensor = prepare_sparse_tensor_batch(sparse_list, batch_size=len(batch_samples))
        routing_vals = torch.tensor([s.advantage for s in batch_samples], device=device, dtype=torch.float32)
        routing_probs = compute_routing_weights(routing_vals, adv_clip_max)
        return cond_batched, neg_batched, x0_batch, routing_probs

    @staticmethod
    def _split_sparse_batch(batch: SparseTensor) -> List[SparseTensor]:
        """将 batched SparseTensor 拆分为单样本列表，保留梯度。"""
        splits: List[SparseTensor] = []
        for sl in batch.layout:
            feats_slice = batch.feats[sl]
            coords_slice = batch.coords[sl].clone()
            coords_slice[:, 0] = 0
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
        mode_norm = (mode or "self").lower()
        beta_denom = max(float(nft_beta), 1e-6)

        if mode_norm == "self":
            pos_vec = compute_sparse_weighted_mse(x0_pos, x0_ref)
            neg_vec = compute_sparse_weighted_mse(x0_neg, x0_ref)
            routing = routing_probs.to(pos_vec.dtype)
            policy_vec = routing * (pos_vec / beta_denom) + (1.0 - routing) * (neg_vec / beta_denom)
        else:
            raise ValueError(f"Sparse policy cross 模式已移除，当前仅支持 self，收到: {mode}")

        pos_mean = pos_vec.mean()
        neg_mean = neg_vec.mean()
        return PolicyLossResult(policy_vec=policy_vec, pos_mean=pos_mean, neg_mean=neg_mean)

    @staticmethod
    def build_samples_from_generation(
        meshes: List[Any],
        all_latents: List[SparseTensor],
        cond_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        rewards: Sequence[float],
        reward_parts_local: Dict[str, Union[np.ndarray, torch.Tensor]],
        batch_meta: List[dict],
        batch_paths: Sequence[str],
        k: int,
    ) -> List[TrellisSample]:
        BK = len(meshes)
        samples: List[TrellisSample] = []

        for s in range(BK):
            # 取对应的稀疏 latent（若数量不足则复用最后一个）
            latent_src = all_latents[s] if s < len(all_latents) else all_latents[-1]
            coords_cpu = latent_src.coords.clone().detach().cpu()
            coords_cpu[:, 0] = 0
            feats_cpu = latent_src.feats.detach().cpu()
            final_latent_cpu = SparseTensor(feats=feats_cpu, coords=coords_cpu, layout=[slice(0, feats_cpu.shape[0])])

            cond_patches_s = cond_batch[s // k:s // k + 1].detach().cpu()
            neg_patches_s = (neg_batch[s // k:s // k + 1].detach().cpu() if (neg_batch is not None) else None)
            reward_components = {**{rk: float(rv[s]) for rk, rv in reward_parts_local.items()}}
            sample = TrellisSample(
                x0_sparse=final_latent_cpu,
                cond_patches=cond_patches_s,
                neg_patches=neg_patches_s,
                reward_components=reward_components,
                reward_avg=float(rewards[s]),
                advantage=0.0,
                image_name=batch_meta[s // k]["image_name"],
                image_path=batch_meta[s // k].get("image_path", batch_paths[s // k]),
            )
            samples.append(sample)
        return samples


def compute_routing_weights(advantages: torch.Tensor, adv_clip_max: float) -> torch.Tensor:
    """DiffusionNFT：将优势裁剪映射到 [0,1]。"""
    adv_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    normalized = (adv_clip / adv_clip_max) / 2.0 + 0.5
    return torch.clamp(normalized, 0.0, 1.0)


def save_meshes_for_preview(
    meshes,
    repeated_image_paths,
    rewards,
    epoch: int,
    save_dir: str,
    device_str: str = "cuda",
    repeated_image_pils=None,
    write_mesh: bool = False,
):
    """使用参考渲染器渲染三视角法线并保存 2×2 预览 PNG；可选导出 OBJ。

    - meshes: List[Any]
    - repeated_image_paths: 与 meshes 一一对应的图像路径列表（每图像重复 K 次）
    - rewards: 与 meshes 对齐的奖励（未使用，仅占位）
    - epoch: 当前 epoch 编号（未使用，仅占位）
    - save_dir: 输出目录
    - device_str: 渲染设备字符串（"cuda" 或 "cpu"）
    - repeated_image_pils: 与 repeated_image_paths 对齐的 PIL 列表
    - write_mesh: 是否导出 OBJ
    """


    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device_str)  # 形状: 标量设备
    renderer = RefMeshRenderer(img_size=512, device=device_str)  # 形状: 渲染器

    preview_files = []
    mesh_files = []

    # 固定三视角参数（右上/左下/右下）
    elevation = 15.0  # 形状: 标量
    distance = 3.0  # 形状: 标量
    fovy = 50.0  # 形状: 标量
    azimuths = [0.0, 120.0, 240.0]  # 形状: (3,)
    predefined_poses = [
        {"distance": distance, "fovy": fovy, "elevation": elevation, "azimuth": a} for a in azimuths
    ]  # 形状: 列表(3)

    # 扁平命名：基于文件名（与原逻辑一致）
    for idx, (mesh, img_path) in enumerate(zip(meshes, repeated_image_paths)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        safe_base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).rstrip()
        case_dir = os.path.join(save_dir, safe_base)
        os.makedirs(case_dir, exist_ok=True)
        preview_path = os.path.join(case_dir, f"preview_{idx}.png")

        # 统一几何提取（供导出与渲染复用）
        mesh_ex = to_mesh_extract(mesh, device)  # 形状: MeshExtractResult(vertices:(V,3), faces:(F,3))

        if write_mesh:
            import trimesh
            v_np = mesh_ex.vertices.detach().cpu().numpy()  # 形状: (V,3)
            f_np = mesh_ex.faces.detach().cpu().numpy().astype(np.int32)  # 形状: (F,3)
            tri = trimesh.Trimesh(vertices=v_np, faces=f_np)
            mesh_path = os.path.join(case_dir, f"mesh_{idx}.obj")
            tri.export(mesh_path)
            mesh_files.append(mesh_path)

        # 渲染三视角法线 + 左上原图
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: kiui Mesh(v:(V,3), f:(F,3))
        cams = renderer.sample_camera_poses(num_random_views=0, predefined_poses=predefined_poses)  # 形状: 列表(3)
        out = renderer.render_mesh(
            mesh_kiui,
            cams,
            return_depth=False,
            return_normals=True,
            return_positions=False,
            return_masks=True,
        )  # 形状: images(3,3,R,R), masks(3,R,R)

        images_t = out["images"]  # 形状: (3,3,R,R)
        R = images_t.shape[-1]  # 形状: 标量

        def to_pil(img_chw: torch.Tensor) -> Image.Image:
            img01 = img_chw.clamp(0, 1)  # 形状: (3,R,R)
            img255 = (img01 * 255.0).round().to(torch.uint8)  # 形状: (3,R,R)
            img_hwc = img255.permute(1, 2, 0).cpu().numpy()  # 形状: (R,R,3)
            return Image.fromarray(img_hwc)

        pil_renders = [to_pil(images_t[i]) for i in range(3)]  # 形状: 列表(3 × PIL(R,R,3))

        # 左上角放输入 RGB（方裁后缩放）
        rgb_in = repeated_image_pils[idx]  # 形状: PIL(H,W,3)
        w, h = rgb_in.size  # 形状: 标量, 标量
        side = min(w, h)  # 形状: 标量
        left = (w - side) // 2  # 形状: 标量
        top = (h - side) // 2  # 形状: 标量
        rgb_sq = rgb_in.crop((left, top, left + side, top + side)).resize((R, R), Image.BICUBIC)  # 形状: PIL(R,R,3)

        panel = Image.new("RGB", (R * 2, R * 2))  # 形状: (2R,2R,3)
        panel.paste(rgb_sq, (0, 0))              # 左上：输入 RGB
        panel.paste(pil_renders[0], (R, 0))      # 右上：az=0
        panel.paste(pil_renders[1], (0, R))      # 左下：az=120
        panel.paste(pil_renders[2], (R, R))      # 右下：az=240
        panel.save(preview_path)

        preview_files.append(preview_path)

    return mesh_files, preview_files


class Image3DDataset(Dataset):
    """最小图像数据集（与 Hunyuan3D 保持一致接口），可选返回 normal_path。"""
    def __init__(self, image_dir: str, normal_cache_dir: Optional[str] = None, normal_resolution: Optional[int] = None):
        self.image_dir = Path(image_dir)
        if (self.image_dir / "images").exists():
            self.image_dir = self.image_dir / "images"
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(sorted(self.image_dir.glob(ext)))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        # normal 缓存相关（若提供则在 metadata 返回 normal_path）
        self.normal_cache_dir = str(normal_cache_dir) if normal_cache_dir is not None else None
        self.normal_resolution = int(normal_resolution) if normal_resolution is not None else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = str(self.image_files[idx])
        image_pil = Image.open(image_path)
        # 新策略：若为 RGBA，直接保留 alpha 通道；否则转 RGB
        if image_pil.mode == 'RGBA':
            image = image_pil  # 保留 RGBA（保留 alpha 通道，供 trellis 使用）
        else:
            image = image_pil.convert('RGB')
        meta = {"image_name": self.image_files[idx].name}  # 形状: 标量
        # 若提供 normal 缓存信息，则返回 normal_path 以便 query 使用
        if self.normal_cache_dir is not None and self.normal_resolution is not None:
            stem = self.image_files[idx].stem  # 形状: 标量
            normal_path = str(Path(self.normal_cache_dir) / f"R{self.normal_resolution}" / f"{stem}.png")  # 形状: 标量
            meta["normal_path"] = normal_path  # 形状: 标量
            # 预加载 normal 的 PIL（供 scorer 构造 query 使用）
            normal_pil = Image.open(normal_path).convert('RGB')
            meta["normal_pil"] = normal_pil  # 形状: PIL(R,R,3)
        return {
            "image": image,
            "image_path": image_path,
            "metadata": meta,
        }

    @staticmethod
    def collate_fn(examples):
        images = [ex["image"] for ex in examples]
        image_paths = [ex["image_path"] for ex in examples]
        metadata = [ex["metadata"] for ex in examples]
        return images, image_paths, metadata


def dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    # 严格使用训练集根目录与法线缓存（不做回退）
    train_root = str(config.train_data_dir)
    normal_cache_dir = str(config.camera_normal_train.cache_dir)
    normal_resolution = int(config.camera_normal_train.normal_resolution)
    dataset = Image3DDataset(train_root, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)
    # 标准分布式采样器（仅本地优势：不做采样侧 K-repeat）
    batch_size = int(config.sample.input_batch_size)
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,
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
    # 严格使用评估集根目录与法线缓存（不做回退）
    eval_root = str(config.eval_data_dir)
    normal_cache_dir = str(config.camera_normal_eval.cache_dir)
    normal_resolution = int(config.camera_normal_eval.normal_resolution)
    eval_dataset = Image3DDataset(eval_root, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)

    eval_bs = int(config.sample.test_batch_size)  # 标量
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
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


def build_optimizer(params, config: ml_collections.ConfigDict):
    # 强制使用新的 optimizer 配置命名：config.train.optimizer.{type, lr, beta1, beta2, eps, weight_decay}
    # assert hasattr(config.train, 'optimizer'), "config.train.optimizer must exist"
    opt = config.train.optimizer
    # for k in ['type', 'lr', 'beta1', 'beta2', 'eps', 'weight_decay']:
    #     assert hasattr(opt, k), f"config.train.optimizer.{k} must be set"

    opt_type = str(opt.type).lower()

    if opt_type == 'adam_8bit':
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            params,
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2),
            eps=opt.eps,
            weight_decay=opt.weight_decay,
        )
    else:
        from timm.optim.optim_factory import create_optimizer_v2
        # 为 Adan 硬编码 3 元 betas；其他优化器使用 2 元 betas
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


# 已移除旧的 compute_advantages（依赖 tracking/global_std），trellis 不再使用


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
    """Trellis 评估流程：逐图生成 1 个 mesh 并聚合奖励。"""
    all_rewards: Dict[str, List[np.ndarray]] = defaultdict(list)

    dense_eval_module = pipeline._resolve_structure_flow_module()
    sparse_eval_module = pipeline._resolve_slat_flow_module()

    with EvalModeGuard(dense_eval_module, sparse_eval_module):
        for eval_batch in tqdm(
            test_dataloader,
            desc="Eval:",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            images, image_paths, metadata = eval_batch
            with torch.inference_mode():  # 关闭梯度，省显存
                # 直接使用原始 PIL 图像做条件编码（Trellis 内部完成预处理）
                cond_batch, neg_batch = pipeline.prepare_image_conditions(images)  # 形状: cond(B,P,C), neg_cond(B,P,C) 或 None
                neg_filled = neg_batch if neg_batch is not None else torch.zeros_like(cond_batch)  # 形状: (B,P,C)
                # Stage1：稀疏结构采样（替换旧的 stage1_with_logprob）
                coords_all, _ = pipeline.stage_1(
                    cond={"cond": cond_batch, "neg_cond": neg_filled},
                    ss_resolution=64,
                    num_samples=int(cond_batch.shape[0]),
                )  # 形状: (N_total, 4)
                coords_list = []
                for b in range(cond_batch.shape[0]):
                    coords_b = coords_all[coords_all[:, 0] == b]  # 形状: (N_b, 4)
                    coords_list.append(coords_b)

                meshes_batch = []
                for b, coords_b in enumerate(coords_list):
                    if coords_b.numel() == 0:
                        continue
                    shape_slat, _ = pipeline.stage_2_shape(
                        cond={
                            "cond": cond_batch[b:b+1],
                            "neg_cond": neg_filled[b:b+1],
                        },
                        coords=coords_b,
                        resolution=1024,
                    )  # 形状: SparseTensor(N_b, C_shape)
                    mesh_obj = pipeline.export_mesh(shape_slat, tex_slat=None, resolution=1024)
                    if isinstance(mesh_obj, (list, tuple)):
                        meshes_batch.extend(mesh_obj)
                        mesh_count = len(mesh_obj)
                    else:
                        meshes_batch.append(mesh_obj)
                        mesh_count = 1

            # 导出 OBJ 与预览（多卡：各 rank 负责自身分片；无需主进程限制）
            if export_dir is not None:
                epoch_dir = os.path.join(export_dir, f"eval_epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                # 1) 保存 mesh 预览和 OBJ 到 .../generated_meshes/eval_epoch_{epoch}/{safe_base}/
                # 在可视化前，将 RGBA 与白底合成为 RGB（不改动原始 images）
                assert isinstance(images, list) and all(isinstance(im, Image.Image) for im in images)
                images_preview = [
                    (Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), im).convert('RGB'))
                    if im.mode == 'RGBA' else im.convert('RGB')
                    for im in images
                ]
                save_meshes_for_preview(
                    meshes=meshes_batch,
                    repeated_image_paths=image_paths,
                    rewards=None,
                    epoch=epoch,
                    save_dir=os.path.join(export_dir, f"eval_epoch_{epoch}"),
                    device_str=accelerator.device.type,
                    repeated_image_pils=images_preview,
                    write_mesh=bool(write_mesh),
                )
                # 2) 将 camera_normal 的 vis_dir 对齐到与 mesh 相同的 {safe_base} 子目录（eval-only）
                if hasattr(mesh_scorer, "_camera_normal") and (mesh_scorer._camera_normal is not None):
                    cn = mesh_scorer._camera_normal
                    cn.cfg.save_vis = True
                    cn.cfg.vis_dir = epoch_dir

            rewards_dict, _ = mesh_scorer.score(meshes_batch, images, metadata, dict(config.reward_fn))
            for key, value in rewards_dict.items():
                gathered = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards[key].append(gathered)

            # 评估步后清理 GPU 缓存，防止累计占用
            del meshes_batch
            torch.cuda.empty_cache()

    all_rewards_np = {key: (np.concatenate(v) if len(v) > 0 else np.array([])) for key, v in all_rewards.items()}
    return all_rewards_np




def build_stage1_cond(
    pipeline: Trellis2PipelineWithLogProb,
    batch_paths: List[str],
    cond_batch: torch.Tensor,
    neg_batch: Optional[torch.Tensor],
    num_steps_dense: int,
    guidance_scale: float,
    generator: Optional[torch.Generator],
    k: int,
) -> Dict[str, Any]:
    """构造 stage2 输入的批字典，避免上层 list[dict] 冗余。

    返回键：
    - cond: (B,P,C)
    - neg_cond: (B,P,C) 或 None
    - coords: SparseTensor（批，候选级layout）
    """
    # 批量生成 B 张图像的稀疏坐标，取代逐图串行
    coords_list: List[torch.Tensor] = pipeline.forward_stage1(
        images=batch_paths,
        num_inference_steps=int(num_steps_dense),  # 形状: 标量
        guidance_scale=float(guidance_scale),      # 形状: 标量
        generator=generator,
    )  # 长度 B，每项 (N_i,4)

    # 使用现有工具函数批量构造稀疏：为每个 coords 创建空特征，然后用 prepare_sparse_tensor_batch 合批
    sparse_list = [SparseTensor(feats=torch.empty((c.shape[0], 1), device=c.device), coords=c) for c in coords_list for _ in range(k)]
    coords_batched = prepare_sparse_tensor_batch(sparse_list, batch_size=len(sparse_list))

    # 扩展 cond/neg_cond 到 (BK,P,C)
    cond_b = cond_batch.repeat_interleave(k, dim=0)
    neg_b = (None if (neg_batch is None) else neg_batch.repeat_interleave(k, dim=0))

    return {
        "cond": cond_b,               # 形状: (BK,P,C)
        "neg_cond": neg_b,            # 形状: (BK,P,C) 或 None
        "coords": coords_batched,     # 形状: 批稀疏（候选级layout）
    }

def build_pipeline(config: ml_collections.ConfigDict, accelerator: Accelerator) -> Trellis2PipelineWithLogProb:
    """构建并放置 Trellis Pipeline 到设备。"""
    # 优先使用本地 DINOv3（避免访问 gated repo）
    dino_local = project_root / "pretrained_weights" / "dinov3-vitl16-pretrain-lvd1689m" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    dino_local_path = str(dino_local) if dino_local.exists() else None

    pipeline = Trellis2PipelineWithLogProb.from_pretrained(
        config.pretrained.pipeline_path,
        dino_local_path=dino_local_path,
    )
    # 兼容旧逻辑：部分工具函数依赖 pipeline.ref
    pipeline.ref = pipeline
    # 统一将所有模型迁移到目标设备，避免采样/解码阶段的 CPU/GPU 混用
    pipeline.to(accelerator.device)
    for m in pipeline.models.values():
        m.to(accelerator.device)
    return pipeline


def get_trainable_model(pipeline: Trellis2PipelineWithLogProb) -> nn.Module:
    """获取 Trellis 可训练的稀疏形状分支模型。"""
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
    """根据配置为 SLatFlowModel 应用 LoRA（仅适配注意力投影层）。"""
    if not bool(config.use_lora):
        return slat_model
    register_sparse_linear_with_peft()
    target_modules = [
        "to_qkv",
        "to_q",
        "to_kv",
        "to_out",
    ]
    lora_r = int(config.lora.lora_rank)
    lora_alpha = lora_r
    lora_dropout = 0.1
    lora_bias_mode = "none"
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha * 2,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias_mode,
    )
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
    """仅为稀疏形状分支构建优化器并包装。"""
    sparse_trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    optimizer_stage2 = build_optimizer(sparse_trainable_params, config)
    slat_model, optimizer_stage2 = accelerator.prepare(slat_model, optimizer_stage2)
    return slat_model, optimizer_stage2, sparse_trainable_params

def enable_gradient_checkpointing_if_needed(slat_model: nn.Module, accelerator: Accelerator, config: ml_collections.ConfigDict) -> None:
    """按配置为所有 block 启用梯度检查点。"""
    use_gc = bool(getattr(config, "gradient_checkpointing", False))
    if use_gc:
        unwrapped = accelerator.unwrap_model(slat_model)
        for blk in unwrapped.blocks:
            blk.use_checkpoint = True


def load_checkpoint(accelerator: "Accelerator", config: ml_collections.ConfigDict, mode: str = "train") -> int:
    """简化版通用加载：仅读取 config.checkpoint，不做任何回退。

    多卡注意：增加显式同步，避免不同 rank 在共享文件系统上同时扫描/加载造成的竞态或卡死。
    """
    def pick_dir(path_str: Optional[str]) -> Optional[Path]:
        if not (isinstance(path_str, str) and path_str):
            return None
        root = Path(path_str)
        if not root.exists() or not root.is_dir():
            return None
        if (root / "state.json").exists() or root.name.startswith("checkpoint_"):
            return root
        cands = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")]
        if not cands:
            return None
        cands.sort(key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
        return cands[-1]

    # 仅使用统一字段 checkpoint（可为 checkpoint_XXXX 目录或其父目录）
    cp = (config.checkpoint if hasattr(config, "checkpoint") else None)

    if mode == "eval":
        # 仅 rank0 选路径，所有进程一致读取；前后各一次 barrier 确保同步
        with accelerator.main_process_first():
            chosen = pick_dir(cp)
        accelerator.wait_for_everyone()
        if chosen:
            accelerator.load_state(str(chosen))
            accelerator.print(f"🔁 Eval-only: loaded {str(chosen)}")
        accelerator.wait_for_everyone()
        return 0

    # 训练恢复同样加同步
    with accelerator.main_process_first():
        chosen = pick_dir(cp)
    accelerator.wait_for_everyone()
    if not chosen:
        return 0
    # 防止部分 rank 早/晚读取导致不一致
    accelerator.load_state(str(chosen))
    accelerator.wait_for_everyone()
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
    """仅评测并退出的封装。"""
    accelerator.wait_for_everyone()
    load_checkpoint(accelerator, config, mode="eval")
    eval_loader = eval_dataloader_from_config(config, accelerator)
    eval_loader.sampler.set_epoch(0)
    gen = create_eval_generator(accelerator.device, int(config.seed))
    # 导出目录（eval 专用）：与现有实验保持一致 -> {run_dir}/generated_meshes
    dirs = RunDirs.from_config(config)
    export_dir = str(dirs.viz_dir)

    # 开启 CameraNormalScorer 可视化（具体 vis_dir 在 eval_trellis 内按 epoch 设置）
    if hasattr(mesh_scorer, "_camera_normal") and (mesh_scorer._camera_normal is not None):
        cn = mesh_scorer._camera_normal
        cn.cfg.save_vis = True

    if bool(config.train.ema) and ema is not None and trainable_params is not None:
        ema.copy_ema_to(trainable_params, store_temp=True)
        all_rewards_np = eval_trellis(
            pipeline, eval_loader, config, accelerator, epoch=0, mesh_scorer=mesh_scorer, generator=gen, export_dir=export_dir, write_mesh=True
        )
        ema.copy_temp_to(trainable_params)
    else:
        all_rewards_np = eval_trellis(
            pipeline, eval_loader, config, accelerator, epoch=0, mesh_scorer=mesh_scorer, generator=gen, export_dir=export_dir, write_mesh=True
        )
    if accelerator.is_main_process:
        run_logger.log_eval_rewards(0, all_rewards_np)

    return



def create_ema_if_needed(trainable_params: list, accelerator: Accelerator, config: ml_collections.ConfigDict) -> Optional[EMAModuleWrapper]:
    """按配置创建 EMA 包装器。"""
    if bool(config.train.ema):
        ema_decay = float(config.train.ema_decay)
        return EMAModuleWrapper(trainable_params, decay=ema_decay, device=accelerator.device)
    return None


class TrainState:
    def __init__(self, global_step: int = 0):
        self.global_step = int(global_step)

    def state_dict(self) -> dict:
        return {"global_step": int(self.global_step)}

    def load_state_dict(self, state: dict) -> None:
        self.global_step = int(state.get("global_step", 0))






def main(_):
    config: ml_collections.ConfigDict = _CONFIG.value
    assert config.use_lora, "DiffusionNFT 训练脚本要求 config.use_lora=True"
    # if not bool(getattr(config, "use_lora", False)):
    #     raise ValueError("DiffusionNFT 训练脚本要求 config.use_lora=True")

    # 统一的时间步采样计算（供梯度累计等逻辑复用）
    _, sparse_step_count = compute_timestep_usage(
        num_steps=int(config.sample.num_steps),
        fraction=float(config.train.timestep_fraction),
        keep_ratio=float(config.train.timestep_keep_ratio),
    )

    # 基础加速器（梯度累计步数 = 配置值 × 稀疏时间步数）
    # 先确定 run_name，用于 Accelerate 的自动 checkpoint 命名
    run_name = config.run_name if len(config.run_name) > 0 else f"trellis_{int(time.time())}"
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=os.path.join(config.logdir, run_name)),
        log_with=["wandb"],
        gradient_accumulation_steps=max(1, int(config.train.gradient_accumulation_steps * sparse_step_count)),  # 标量
    )
    set_seed(int(config.seed))
    setup_backend_determinism()

    # 规范化加速器跟踪器初始化，确保 accelerator.log 正常写入 W&B
    # 规范 run_name，保持与 RunDirs 一致
    config.run_name = run_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo-trellis",
            config=dict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # 构建 Pipeline（评估与训练共享）
    pipeline = build_pipeline(config, accelerator)

    device = accelerator.device
    # 在初始化阶段按权重加载所需 scorer，避免无关模型初始化
    cam_cfg = dict(config.camera_normal) if "camera_normal" in config else {}
    if "camera_normal_train" in config:
        cam_cfg.setdefault("cache_dir", str(config.camera_normal_train.cache_dir))
        cam_cfg.setdefault("normal_resolution", int(config.camera_normal_train.normal_resolution))

    mesh_scorer = MeshScorer(
        device=device,
        verbose=bool(config.verbose),
        score_fns_cfg=dict(config.reward_fn),
        camera_normal_cfg=cam_cfg,
    )

    # eval_only 提前返回：应用 LoRA，并准备模型后再加载权重评测
    if bool(config.eval_only):
        slat_model = get_trainable_model(pipeline)
        slat_model = apply_lora_if_needed(slat_model, config)
        slat_model = accelerator.prepare(slat_model)
        pipeline.ref.models['slat_flow_model'] = slat_model
        dirs = RunDirs.from_config(config)
        run_logger = RunLogger(accelerator, dirs)
        trainable_params_eval = [p for p in slat_model.parameters() if p.requires_grad]
        ema_eval = create_ema_if_needed(trainable_params_eval, accelerator, config)
        if ema_eval is not None:
            accelerator.register_for_checkpointing(ema_eval)
        run_eval_only(pipeline, config, accelerator, mesh_scorer, run_logger, ema=ema_eval, trainable_params=trainable_params_eval)
        return

    # 构建训练对象（仅稀疏形状分支），应用可选 LoRA，并包装/构建优化器
    slat_model = get_trainable_model(pipeline)
    slat_model = apply_lora_if_needed(slat_model, config)

    slat_model, optimizer_stage2, sparse_trainable_params = prepare_optimizer_and_wrap(slat_model, config, accelerator)

    set_model_adapter(slat_model, "default")

    enable_gradient_checkpointing_if_needed(slat_model, accelerator, config)

    # 注册自定义持久化状态（EMA/TrainState）后再加载 checkpoint
    ema_stage2 = create_ema_if_needed(sparse_trainable_params, accelerator, config)
    if ema_stage2 is not None:
        accelerator.register_for_checkpointing(ema_stage2)
    # 不再创建或注册按图像统计追踪器（去除 tracking/global_std）
    train_state = TrainState(global_step=0)
    accelerator.register_for_checkpointing(train_state)
    start_epoch = load_checkpoint(accelerator, config, mode="train")

    # 数据与奖励（仅训练路径）
    train_loader = dataloader_from_config(config, accelerator)
    train_loader.sampler.set_epoch(start_epoch)


    # 集中化日志/保存调度与缓存
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
    viz = VizBuffer()

    # eval_only 模式：仅评测后退出
    if bool(config.eval_only):
        run_eval_only(pipeline, config, accelerator, mesh_scorer, run_logger)
        return

    for epoch in range(start_epoch, config.num_epochs):
        # 本 epoch 训练指标聚合器：仅记录 Stage2 指标
        epoch_logger_s2 = DiffusionNFTMetricLogger()
        # 采样阶段：对每张图像生成 K 个候选并打分
        all_samples = TrellisSampleCollection()
        max_train_batches = int(config.sample.num_batches_per_epoch)
        # 为本 epoch 设置采样器随机种子，使本 epoch 的前 N 个 batch 与其他 epoch 不同
        train_loader.sampler.set_epoch(epoch)
        # 使用有限迭代器：当 num_batches_per_epoch>0 时截断；否则保持原始无限流（依赖上层控制）
        loader_iter = (
            itertools.islice(train_loader, max_train_batches)
            if max_train_batches > 0 else train_loader
        )
        for batch_idx, (batch_images, batch_paths, batch_meta) in enumerate(tqdm(loader_iter, disable=not accelerator.is_main_process)):
            with EvalModeGuard(slat_model):
                with torch.inference_mode():  # 关闭梯度，省显存
                    # 条件编码
                    cond_batch, neg_batch = pipeline.prepare_image_conditions(batch_images)  # 形状: cond(B,P,C), neg_cond(B,P,C) 或 None
                    k = int(config.sample.num_meshes_per_image)  # 形状: 标量
                    # 依据 same_latent 开关，创建稳定生成器（批级别）
                    use_same_latent = config.sample.same_latent  # 形状: 标量
                    generator = (
                        create_train_generator_for_batch(accelerator.device, int(epoch), int(batch_idx), list(batch_paths))
                        if use_same_latent else None
                    )  # 形状: 生成器 或 None
                    # 展开到 BK
                    cond_bk = cond_batch.repeat_interleave(k, dim=0)  # 形状: (BK,P,C)
                    neg_bk = (None if (neg_batch is None) else neg_batch.repeat_interleave(k, dim=0))  # 形状: (BK,P,C) 或 None

                    # ===== 简化采样：直接三阶段 + 解码 =====
                    coords_list = []
                    slat_list = []
                    meshes = []
                    for coords_idx in range(cond_bk.shape[0]):
                        # Stage1: 稀疏结构（固定分辨率 64）
                        coords, _ = pipeline.stage_1(
                            cond={
                                "cond": cond_bk[coords_idx:coords_idx+1],
                                "neg_cond": torch.zeros_like(cond_bk[coords_idx:coords_idx+1]),
                            },
                            ss_resolution=64,
                            num_samples=1,
                        )
                        coords_list.append(coords)

                        # Stage2-shape
                        shape_slat, _ = pipeline.stage_2_shape(
                            cond={
                                "cond": cond_bk[coords_idx:coords_idx+1],
                                "neg_cond": torch.zeros_like(cond_bk[coords_idx:coords_idx+1]),
                            },
                            coords=coords,
                            resolution=1024,
                        )
                        slat_list.append(shape_slat)

                        # 解码 mesh：使用官方 decode_latent（纹理分支可选，这里传形状复用）
                        # 若无需纹理，可传 shape_slat 作为占位，decode_latent 会返回 MeshWithVoxel 列表
                        mesh_list = pipeline.decode_latent(
                            shape_slat=shape_slat,
                            tex_slat=shape_slat,  # 占位：无纹理时复用形状 slat
                            resolution=1024,
                        )
                        meshes.extend(mesh_list)

                    # 占位：兼容后续 TrellisSampleCollection 使用
                    latents_seq_dense = []  # 不再提供稠密时间序列
                    all_latents = slat_list  # 用形状 slat 列表占位

            # 打分与可视化
            repeated_meta = []
            for meta_item, path in zip(batch_meta, batch_paths):
                m = dict(meta_item)
                m["image_path"] = path
                repeated_meta.extend([m] * k)
            repeated_images = []
            for img in batch_images:
                repeated_images.extend([img] * k)
            rewards_dict, meta_out = mesh_scorer.score(meshes, repeated_images, repeated_meta, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # 形状: (BK,)
            reward_parts_local = {k: v for k, v in rewards_dict.items() if k != "avg"}
            if batch_idx == 0:
                repeated_paths = []
                repeated_pils = []
                for p, im in zip(batch_paths, batch_images):
                    repeated_paths.extend([p] * k)
                    repeated_pils.extend([im] * k)
                num_samples_to_cache = min(2, len(meshes))
                viz.update_from_batch(
                    meshes[:num_samples_to_cache],
                    repeated_paths[:num_samples_to_cache],
                    rewards[:num_samples_to_cache],
                    camera_normal_pairs_best=meta_out.get("camera_normal_pairs_best", None),
                    camera_normal_pairs_worst=meta_out.get("camera_normal_pairs_worst", None),
                    image_pils=repeated_pils[:num_samples_to_cache],
                    uni3d_pairs_best=meta_out.get("uni3d_pairs_best", None),
                    uni3d_pairs_worst=meta_out.get("uni3d_pairs_worst", None)
                )


            all_samples.extend(
                TrellisSampleCollection.build_samples_from_generation(
                        meshes=meshes,
                        all_latents=all_latents,
                        cond_batch=cond_batch,
                        neg_batch=neg_batch,
                        rewards=rewards,
                        reward_parts_local=reward_parts_local,
                        batch_meta=batch_meta,
                        batch_paths=batch_paths,
                        k=k,
                    ),
            )

            del meshes, all_latents
            torch.cuda.empty_cache()

        # 统计与优势（与 Hunyuan3D 一致：分布式聚合后按图像标准化）
        accelerator.wait_for_everyone()
        all_samples.compute_rewards_and_advantages(
            reward_weights=dict(config.reward_fn),
            adv_type=config.sample.adv_type,
            adv_from=config.sample.adv_from,
            accelerator=accelerator,
            epoch=epoch,
        )

        top_bottom_k = int(config.sample.top_bottom_k)  # 形状: 标量
        if top_bottom_k > 0:
            all_samples.select_top_bottom(top_bottom_k)
        top_k = int(config.sample.top_k)  # 形状: 标量
        if top_k > 0:
            all_samples.select_top(top_k)

        filtered_samples = all_samples.as_list()
        if len(filtered_samples) == 0:
            rewards_local = np.zeros(0, dtype=np.float64)
            advantages_local = np.zeros(0, dtype=np.float64)
        else:
            rewards_local = np.array([s.reward_avg for s in filtered_samples], dtype=np.float64)
            advantages_local = np.array([s.advantage for s in filtered_samples], dtype=np.float64)

        accelerator.wait_for_everyone()
        reward_mean_global = distributed_mean(rewards_local, accelerator)
        adv_mean_global = distributed_mean(advantages_local, accelerator)
        epoch_logger_s2.set_reward_and_adv_means(reward_mean_global, adv_mean_global)

        valid_samples_ratio = all_samples.valid_ratio()

        actual_train_bs = config.train.batch_size 
        run_logger.log_sampling_stats(
            epoch=epoch,
            actual_batch_size=actual_train_bs,
            num_sub_batches=len(all_samples),
            valid_ratio=float(valid_samples_ratio),
        )

        # ===== 训练阶段：DiffusionNFT =====
        set_model_adapter(slat_model, "default")
        slat_model.train()

        # 若无调度器属性，则使用均匀时间步占位
        steps_sparse = int(config.sample.num_steps)
        sparse_timesteps = torch.arange(steps_sparse, device=accelerator.device, dtype=torch.float32)
        rng = np.random.default_rng(int(config.seed) + epoch)
        frac = float(config.train.timestep_fraction)
        keep = float(config.train.timestep_keep_ratio)

        used_sparse, keep_sparse = compute_timestep_usage(steps_sparse, frac, keep)
        base_sparse = np.linspace(0, steps_sparse - 1, used_sparse, dtype=np.int32)
        train_step_indices_sparse = np.sort(rng.choice(base_sparse, size=keep_sparse, replace=False))
        nft_beta = float(config.nft_beta)
        kl_beta = float(config.train.beta)
        adv_clip_max = float(config.train.adv_clip_max)
        max_grad_norm = float(config.train.max_grad_norm)

        for inner_epoch in range(int(config.train.num_inner_epochs)):
            batch_iter = tqdm(
                all_samples.iter_batches(actual_train_bs),
                total=(len(all_samples) + actual_train_bs - 1) // actual_train_bs,
                disable=not accelerator.is_main_process,
                desc=f"Stage2 Batches (inner {inner_epoch})",
                leave=False,
            )
            for batch_idx, batch_samples in enumerate(batch_iter):

                cond_batched, _, x0_sparse_batch, routing_probs = TrellisSampleCollection.move_batch_samples(
                    batch_samples=batch_samples,
                    device=accelerator.device,
                    dtype=torch.float32,
                    adv_clip_max=adv_clip_max,
                )
                batch_size = len(batch_samples)

                step_iter = tqdm(
                    train_step_indices_sparse,
                    total=len(train_step_indices_sparse),
                    disable=not accelerator.is_main_process,
                    desc=f"Stage2 Steps (batch {batch_idx})",
                    leave=False,
                )
                for j in step_iter:
                    t_value = sparse_timesteps[int(j)].item()
                    t_norm_value = t_value / 1000.0
                    # t 传标量给模型，但混噪按点扩展；逐样本处理避免批维冲突
                    t = torch.full((batch_size,), t_value, device=accelerator.device, dtype=torch.float32)

                    model_out_list = []
                    teacher_out_list = []
                    xt_list = []
                    valid_idx = []
                    for b_idx, sl in enumerate(x0_sparse_batch.layout):
                        feats_b = x0_sparse_batch.feats[sl]
                        coords_b = x0_sparse_batch.coords[sl].clone()
                        coords_b[:, 0] = 0
                        t_norm_full = torch.full_like(feats_b, t_norm_value)
                        noise_b = torch.randn_like(feats_b)
                        xt_feats = feats_b * (1.0 - t_norm_full) + noise_b * t_norm_full
                        xt_b = SparseTensor(
                            feats=xt_feats,
                            coords=coords_b,
                            layout=[slice(0, feats_b.shape[0])],
                        )
                        if feats_b.shape[0] == 0:
                            continue
                        cond_b = cond_batched[b_idx:b_idx+1] if cond_batched is not None else None
                        neg_b = None  # 若需 CFG 可替换为 neg_batched[b_idx:b_idx+1] 或 zeros_like(cond_b)
                        if (cond_b is None) or (cond_b.numel() == 0):
                            continue

                        with accelerator.autocast():
                            out_b = slat_model(xt_b, t[b_idx:b_idx+1], cond_b, neg_b, guidance_scale=1.0)
                        xt_list.append(xt_b)
                        valid_idx.append(int(b_idx))
                        model_out_list.append(out_b)

                        if kl_beta > 0.0:
                            with torch.no_grad():
                                base_sparse = pipeline._resolve_slat_flow_module()
                                with base_sparse.disable_adapter():
                                    teacher_b = base_sparse(xt_b, t[b_idx:b_idx+1], cond_b, neg_b, guidance_scale=1.0)
                            teacher_out_list.append(teacher_b)

                    if len(xt_list) == 0:
                        continue

                    xt_sparse = prepare_sparse_tensor_batch(xt_list, batch_size=len(xt_list))
                    model_output = prepare_sparse_tensor_batch(model_out_list, batch_size=len(model_out_list))
                    model_output_ref = (
                        prepare_sparse_tensor_batch(teacher_out_list, batch_size=len(teacher_out_list))
                        if (kl_beta > 0.0 and len(teacher_out_list) > 0)
                        else None
                    )

                    t_norm_full_batch = torch.full_like(xt_sparse.feats, t_norm_value)
                    positive_sparse = sparse_clone_with_feats(
                        model_output,
                        nft_beta * model_output.feats + (1.0 - nft_beta) * (model_output_ref.feats if model_output_ref is not None else model_output.feats),
                    )
                    negative_sparse = sparse_clone_with_feats(
                        model_output,
                        (1.0 + nft_beta) * (model_output_ref.feats if model_output_ref is not None else model_output.feats) - nft_beta * model_output.feats,
                    )
                    x0_pos = sparse_clone_with_feats(
                        xt_sparse,
                        xt_sparse.feats - positive_sparse.feats * t_norm_full_batch,
                    )
                    x0_neg = sparse_clone_with_feats(
                        xt_sparse,
                        xt_sparse.feats - negative_sparse.feats * t_norm_full_batch,
                    )

                    # 使用有效样本子集对齐 layout
                    filtered_batch_samples = [batch_samples[i] for i in valid_idx]
                    routing_valid = routing_probs[valid_idx] if routing_probs.numel() == len(batch_samples) else routing_probs

                    policy_sparse_self = TrellisSampleCollection.compute_sparse_policy_loss(
                        batch_samples=filtered_batch_samples,
                        x0_pos=x0_pos,
                        x0_neg=x0_neg,
                        x0_ref=xt_sparse,
                        routing_probs=routing_valid,
                        nft_beta=nft_beta,
                        mode="self",
                    )
                    policy_loss_self = (policy_sparse_self.policy_vec * adv_clip_max).mean()
                    policy_loss = policy_loss_self
                    policy_loss_cross = torch.tensor(0.0, device=accelerator.device)

                    pos_mean = policy_sparse_self.pos_mean
                    neg_mean = policy_sparse_self.neg_mean
                    if model_output_ref is not None:
                        kl_vec = sparse_batch_mse(model_output, model_output_ref)
                        kl_loss = kl_vec.mean()
                    else:
                        kl_loss = torch.zeros(1, device=accelerator.device, dtype=model_output.feats.dtype)
                    total_loss = policy_loss + kl_beta * kl_loss

                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        if max_grad_norm > 0.0:
                            accelerator.clip_grad_norm_(slat_model.parameters(), max_grad_norm)
                        optimizer_stage2.step()
                        optimizer_stage2.zero_grad(set_to_none=True)
                    accelerator.wait_for_everyone()

                    train_state.global_step += 1
                    if bool(config.train.ema) and ema_stage2 is not None:
                        ema_stage2.step([p for p in slat_model.parameters() if p.requires_grad], train_state.global_step)

                    epoch_logger_s2.update(
                        policy_loss.detach(),
                        policy_loss_self.detach(),
                        policy_loss_cross.detach(),
                        pos_mean.detach(),
                        neg_mean.detach(),
                        kl_loss.detach(),
                        batch_size=len(batch_samples),
                    )

                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        # 本 epoch 结束：分别按命名空间记录一次到 W&B（步数用 epoch）
        if (epoch % max(1, schedule.log_every_epochs) == 0):
            run_logger.log_epoch_metrics_prefixed(epoch, epoch_logger_s2, "stage2")

        # 评估节奏对齐：所有进程共同参与评估以避免分布式 gather 阻塞
        if int(config.eval_freq) > 0 and (epoch % int(config.eval_freq) == 0):
            accelerator.wait_for_everyone()
            eval_loader = eval_dataloader_from_config(config, accelerator)
            eval_loader.sampler.set_epoch(epoch)
            # —— 评估固定生成器：所有 rank 使用完全相同的噪声序列（严格对齐） ——
            gen = create_eval_generator(accelerator.device, int(config.seed))
            trainable = None
            if bool(config.train.ema) and ema_stage2 is not None:
                trainable = [p for p in slat_model.parameters() if p.requires_grad]
            # 使用 EMA 权重评估（如启用）
            if bool(config.train.ema) and ema_stage2 is not None:
                ema_stage2.copy_ema_to(trainable, store_temp=True)
                all_rewards_np = eval_trellis(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
                ema_stage2.copy_temp_to(trainable)
            else:
                all_rewards_np = eval_trellis(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
            accelerator.wait_for_everyone()
            run_logger.log_eval_rewards(epoch, all_rewards_np)

        # 保存节奏对齐：每 epoch 末保存（频率由调度控制）
        if (epoch % int(schedule.save_every_epochs) == 0):
            saver.save_epoch(
                epoch=epoch,
                slat_model=slat_model,
                optimizer=optimizer_stage2,
                config=config,
                ema=(ema_stage2 if bool(config.train.ema) and ema_stage2 is not None else None),
                use_lora=bool(config.use_lora),
            )

        # 可视化与上传：独立于保存频率（仅主进程执行文件写入）
        if schedule.save_visualizations and (epoch % int(schedule.viz_every_epochs) == 0) and viz.meshes is not None:
            if accelerator.is_main_process:
                viz_dir = dirs.viz_dir / f"epoch_{epoch+1}"
                viz_dir.mkdir(parents=True, exist_ok=True)
                device_str = device.type
                mesh_files, preview_files = save_meshes_for_preview(
                    viz.meshes,
                    viz.image_paths,
                    viz.rewards,
                    epoch + 1,
                    str(viz_dir),
                    device_str=device_str,
                    repeated_image_pils=viz.image_pils,
                )
                run_logger.log_mesh_previews(epoch, preview_files, viz.image_paths)

            if viz.camera_normal_pairs_best is not None and len(viz.camera_normal_pairs_best) > 0:
                run_logger.log_normal_pairs(epoch, viz.camera_normal_pairs_best, prefix="camera_normal/best", max_pairs=4)
            if viz.camera_normal_pairs_worst is not None and len(viz.camera_normal_pairs_worst) > 0:
                run_logger.log_normal_pairs(epoch, viz.camera_normal_pairs_worst, prefix="camera_normal/worst", max_pairs=4)
            # 追加 Uni3D best/worst 配对日志
            if viz.uni3d_pairs_best is not None and len(viz.uni3d_pairs_best) > 0:
                run_logger.log_normal_pairs(epoch, viz.uni3d_pairs_best, prefix="uni3d/best", max_pairs=4)
            if viz.uni3d_pairs_worst is not None and len(viz.uni3d_pairs_worst) > 0:
                run_logger.log_normal_pairs(epoch, viz.uni3d_pairs_worst, prefix="uni3d/worst", max_pairs=4)

        all_samples.clear()
        gc.collect()


@dataclass
class LogSaveSchedule:
    log_every_epochs: int
    eval_every_epochs: int
    save_every_epochs: int
    viz_every_epochs: int
    save_visualizations: bool


@dataclass
class RunDirs:
    run_dir: Path
    ckpt_dir: Path
    viz_dir: Path

    @staticmethod
    def from_config(config) -> "RunDirs":
        run_name_dir = config.run_name if isinstance(config.run_name, str) and len(config.run_name) > 0 else "trellis_s2"
        run_dir = Path(config.logdir) / run_name_dir
        return RunDirs(
            run_dir=run_dir,
            ckpt_dir=run_dir / "checkpoints",
            viz_dir=run_dir / "generated_meshes",
        )


@dataclass
class VizBuffer:
    meshes: Optional[list] = None
    image_paths: Optional[list] = None
    rewards: Optional[np.ndarray] = None
    camera_normal_pairs_best: Optional[list] = None
    camera_normal_pairs_worst: Optional[list] = None
    image_pils: Optional[list] = None  # 形状: 列表(PIL(H,W,3)) 与 image_paths 对齐（经 K 次重复后子集）
    uni3d_pairs_best: Optional[list] = None
    uni3d_pairs_worst: Optional[list] = None

    def update_from_batch(self, meshes, image_paths, rewards, camera_normal_pairs_best=None, camera_normal_pairs_worst=None, image_pils=None, uni3d_pairs_best=None, uni3d_pairs_worst=None):
        self.meshes = meshes
        self.image_paths = image_paths
        self.rewards = rewards
        if camera_normal_pairs_best is not None and len(camera_normal_pairs_best) > 0:
            self.camera_normal_pairs_best = camera_normal_pairs_best
        if camera_normal_pairs_worst is not None and len(camera_normal_pairs_worst) > 0:
            self.camera_normal_pairs_worst = camera_normal_pairs_worst
        if image_pils is not None:
            assert isinstance(image_pils, list) and all(isinstance(im, Image.Image) for im in image_pils)
            processed = [
                (Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), im).convert('RGB'))
                if im.mode == 'RGBA' else im.convert('RGB')
                for im in image_pils
            ]
            self.image_pils = processed
        if uni3d_pairs_best is not None and len(uni3d_pairs_best) > 0:
            self.uni3d_pairs_best = uni3d_pairs_best
        if uni3d_pairs_worst is not None and len(uni3d_pairs_worst) > 0:
            self.uni3d_pairs_worst = uni3d_pairs_worst


class RunLogger:
    def __init__(self, accelerator: Accelerator, dirs: RunDirs):
        self.accelerator = accelerator
        self.dirs = dirs

    def print(self, msg: str):
        self.accelerator.print(msg)

    def log_sampling_stats(self, epoch: int, actual_batch_size: int, num_sub_batches: int, valid_ratio: float):
        self.accelerator.log(
            {
                "actual_batch_size": int(actual_batch_size),
                "num_sub_batches": int(num_sub_batches),
                "valid_samples_ratio": float(valid_ratio),
            },
            step=epoch,
        )

    def log_epoch_metrics(self, epoch: int, epoch_logger: "DiffusionNFTMetricLogger"):
        log_dict = epoch_logger.to_global_log_dict(self.accelerator)
        if self.accelerator.is_main_process and log_dict is not None:
            self.accelerator.log(log_dict, step=epoch + 1)

    def log_epoch_metrics_prefixed(self, epoch: int, epoch_logger: "DiffusionNFTMetricLogger", prefix: str):
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
        if self.accelerator.is_main_process:
            metrics = {f"eval_reward_{k}": (float(v.mean()) if v.size > 0 else 0.0) for k, v in all_rewards_np.items()}
            self.accelerator.log(metrics, step=epoch + 1)

    def log_mesh_previews(self, epoch: int, preview_files: List[str], image_paths: List[str]):
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

    def log_normal_pairs(self, epoch: int, pairs: list, prefix: str = "camera_normal", max_pairs: int = 4):
        log_normal_similarity_pairs(self.accelerator, pairs, step=epoch + 1, prefix=prefix, max_pairs=max_pairs)


class CheckpointSaver:
    def __init__(self, accelerator: Accelerator, dirs: RunDirs):
        self.accelerator = accelerator
        self.dirs = dirs

    def save_epoch(self, epoch: int, slat_model: nn.Module, optimizer: optim.Optimizer, config: ml_collections.ConfigDict, ema: Optional[Any] = None, use_lora: bool = False):
        # 等待所有 rank 对齐，避免并发 I/O 竞态
        self.accelerator.wait_for_everyone()
        checkpoint_dir = self.dirs.ckpt_dir / f"checkpoint_{epoch}"
        # 仅主进程创建/清理目录
        if self.accelerator.is_main_process:
            self.dirs.ckpt_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
        # 确保目录存在再进行保存
        self.accelerator.wait_for_everyone()
        # 关键：所有 rank 都 save_state，保证各自 RNG 状态被写入
        self.accelerator.save_state(output_dir=str(checkpoint_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.print(f"💾 Saved (Accelerate): {str(checkpoint_dir)}")




if __name__ == "__main__":
    app.run(main)
