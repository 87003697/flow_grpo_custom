#!/usr/bin/env python3
"""
Direct3D-S2 Stage 2 GRPO Training Script

- 两阶段：Stage 1 冻结在线生成稀疏坐标；Stage 2 对 SLatFlowModel 做 GRPO
- 复用 Hunyuan3D/SD3 的 GRPO 训练框架与指标定义
- 稀疏张量：coords(N,4) + feats(N,C)，接入 Flow Matching + SDE + LogProb
- 约束：仅训练 Stage 2，无 try/except，无 fallback
"""

import os, sys
import gc
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


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
from collections import defaultdict
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

# 参考渲染器与 mesh 适配器（用于可视化四视角法线渲染）
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from reward_models.camera_normal_scorer.render.adapter import to_mesh_extract, KiuiMeshLike

# 导入 Direct3D/GRPO 相关模块
from flow_grpo.diffusers_patch.direct3d_s2_pipeline_with_logprob import (
    Direct3DS2PipelineWithLogProb,
    SlatSamplerParams,
)
from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import (
    Stage2RuntimeConfig,
    compute_log_prob_direct3d_stage2,
    SparseTensor,
    prepare_sparse_tensor_batch,
)
from flow_grpo.ema import EMAModuleWrapper
from reward_models.rewards_mesh import MeshScorer
# 工具函数改为直接使用 pipeline 的 preprocess_image
# convert_direct3d_to_trimesh 不再在训练路径中直接使用

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

_CONFIG = config_flags.DEFINE_config_file("config")
from peft import LoraConfig, get_peft_model, PeftModel
from flow_grpo.peft_sparse.sparse_lora import register_sparse_linear_with_peft
from dataclasses import dataclass
import itertools


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


class EpochMetricLogger:
    """按 epoch 聚合训练指标并提供便捷的上报接口"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_loss = 0.0  # 标量累计
        self.sum_kl = 0.0    # 标量累计
        self.sum_adv = 0.0   # 标量累计
        self.sum_ratio = 0.0 # 标量累计
        self.min_ratio = float('inf')  # 本 epoch 各步最小 ratio（标量）
        self.max_ratio = float('-inf') # 本 epoch 各步最大 ratio（标量）
        self.min_adv = float('inf')    # 本 epoch 各步最小 advantage（标量）
        self.max_adv = float('-inf')   # 本 epoch 各步最大 advantage（标量）
        self.sum_approx_kl = 0.0  # 标量累计（样本加权）
        # 计数加权：分别统计均值类与 clipfrac 的样本总数
        self.count_total_means = 0.0   # 标量累计（用于 loss/kl/adv/ratio/approx_kl/policy_loss 的样本数）
        self.count_clip_low = 0.0      # 标量累计（被低端裁剪的样本数）
        self.count_clip_high = 0.0     # 标量累计（被高端裁剪的样本数）
        self.count_total_clip = 0.0    # 标量累计（clipfrac 的样本总数）
        self.sum_policy_loss = 0.0  # 标量累计（样本加权）
        self.num_steps = 0   # 标量累计
        self.reward_mean: Optional[float] = None  # 标量 ()
        # 新增：按子奖励键记录全局均值（仅主进程填充）
        self.reward_means_by_key: Dict[str, float] = {}

    def update(self, loss_tensor: torch.Tensor, kl_vec: torch.Tensor, adv_vec: torch.Tensor, ratio_vec: torch.Tensor, batch_size: Any):
        # 使用样本数加权，得到全局稳定的均值
        bs_val = float(batch_size) if not isinstance(batch_size, torch.Tensor) else float(batch_size.detach().item())  # 标量 ()
        self.sum_loss += float(loss_tensor.detach().item()) * bs_val  # 标量 ()
        self.sum_kl += float(kl_vec.detach().mean().item()) * bs_val  # 标量 ()
        self.sum_adv += float(adv_vec.detach().mean().item()) * bs_val  # 标量 ()
        self.sum_ratio += float(ratio_vec.detach().mean().item()) * bs_val  # 标量 ()
        self.count_total_means += bs_val  # 统计全局样本数（用于均值）
        # 记录本步 ratio 的极值（ratio_vec 形状 (B_sub,) -> 最小/最大均为标量）
        ratio_min_val = float(ratio_vec.detach().min().item())  # 标量 ()
        ratio_max_val = float(ratio_vec.detach().max().item())  # 标量 ()
        if ratio_min_val < self.min_ratio:
            self.min_ratio = ratio_min_val
        if ratio_max_val > self.max_ratio:
            self.max_ratio = ratio_max_val
        # 记录本步 advantage 的极值（adv_vec 形状 (B_sub,) -> 最小/最大均为标量）
        adv_min_val = float(adv_vec.detach().min().item())  # 标量 ()
        adv_max_val = float(adv_vec.detach().max().item())  # 标量 ()
        if adv_min_val < self.min_adv:
            self.min_adv = adv_min_val
        if adv_max_val > self.max_adv:
            self.max_adv = adv_max_val
        self.num_steps += 1  # 标量 ()

    def update_ppo_metrics(self, approx_kl: torch.Tensor, clipfrac_low: torch.Tensor, clipfrac_high: torch.Tensor, policy_loss: torch.Tensor, batch_size: Any):
        """聚合 PPO 诊断指标（样本加权的 approx_kl/policy_loss + 按样本计数的 clipfrac）"""
        bs_val = float(batch_size) if not isinstance(batch_size, torch.Tensor) else float(batch_size.detach().item())  # 标量 ()
        # approx_kl / policy_loss 按样本加权求和，最终除以样本总数得到稳定均值
        self.sum_approx_kl += float(approx_kl.detach().item()) * bs_val  # 标量 ()
        self.sum_policy_loss += float(policy_loss.detach().item()) * bs_val  # 标量 ()
        # clipfrac 使用按样本计数的全局比例
        self.count_clip_low += float(clipfrac_low.detach().item()) * bs_val  # 标量 ()
        self.count_clip_high += float(clipfrac_high.detach().item()) * bs_val  # 标量 ()
        self.count_total_clip += bs_val  # 标量 ()

    # 已废弃：请使用 update_reward({"avg": rewards_np_local}, accelerator)

    def update_reward(self, rewards_parts_np_local: Dict[str, np.ndarray], accelerator: Accelerator) -> None:
        """分布式聚合各子奖励的全局均值并缓存到 reward_means_by_key。"""
        means: Dict[str, float] = {}
        for k, arr in rewards_parts_np_local.items():
            t_local = torch.as_tensor(arr, device=accelerator.device, dtype=torch.float32)  # 形状 (N,)
            t_global = accelerator.gather(t_local)  # 形状 (G*N,)
            if accelerator.is_main_process:
                means[k] = float(t_global.mean().item())  # 标量 ()
        if accelerator.is_main_process:
            self.reward_means_by_key = means

    def to_log_dict(self) -> Optional[Dict[str, float]]:
        if self.num_steps == 0:
            return None
        # 使用样本总数作为归一化因子，确保跨子批大小稳定（与 clipfrac 的样本总数分开统计）
        denom_samples = float(self.count_total_means) if self.count_total_means > 0 else 1.0  # 标量 ()
        out = {
            "epoch/train_loss": float(self.sum_loss / denom_samples),   # 标量 ()
            "epoch/kl_mean": float(self.sum_kl / denom_samples),        # 标量 ()
            "epoch/adv_mean": float(self.sum_adv / denom_samples),      # 标量 ()
            "epoch/adv_min": float(self.min_adv),               # 标量 ()
            "epoch/adv_max": float(self.max_adv),               # 标量 ()
            "epoch/ratio_mean": float(self.sum_ratio / denom_samples),  # 标量 ()
            "epoch/ratio_min": float(self.min_ratio),           # 标量 ()
            "epoch/ratio_max": float(self.max_ratio),           # 标量 ()
            "epoch/approx_kl": float(self.sum_approx_kl / denom_samples),   # 标量 ()
            # clipfrac 使用按样本计数的全局比例，避免子批大小不同导致的偏差
            "epoch/clipfrac_low": float(self.count_clip_low / max(1.0, self.count_total_clip)),   # 标量 ()
            "epoch/clipfrac_high": float(self.count_clip_high / max(1.0, self.count_total_clip)), # 标量 ()
            "epoch/policy_loss": float(self.sum_policy_loss / denom_samples),  # 标量 ()
        }
        # 同步输出各子奖励均值（若已设置）
        for k, v in getattr(self, "reward_means_by_key", {}).items():
            out[f"epoch/reward_mean/{k}"] = float(v)
        # 兼容总分均值：从 avg 键读取
        if "avg" in getattr(self, "reward_means_by_key", {}):
            out["epoch/reward_mean"] = float(self.reward_means_by_key["avg"])  # 标量 ()
        return out

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        """分布式全局聚合并返回日志字典（所有进程均需调用）。"""
        # 本地张量（求和量与计数）
        vec_sum = torch.tensor([
            self.sum_loss,
            self.sum_kl,
            self.sum_adv,
            self.sum_ratio,
            self.count_total_means,
            self.sum_approx_kl,
            self.sum_policy_loss,
            self.count_clip_low,
            self.count_clip_high,
            self.count_total_clip,
            float(self.num_steps),
        ], device=accelerator.device, dtype=torch.float64)  # 形状 (11,)
        vec_min = torch.tensor([
            self.min_ratio,
            self.min_adv,
        ], device=accelerator.device, dtype=torch.float64)  # 形状 (2,)
        vec_max = torch.tensor([
            self.max_ratio,
            self.max_adv,
        ], device=accelerator.device, dtype=torch.float64)  # 形状 (2,)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vec_sum, op=dist.ReduceOp.SUM)  # 形状 (11,)
            dist.all_reduce(vec_min, op=dist.ReduceOp.MIN)  # 形状 (2,)
            dist.all_reduce(vec_max, op=dist.ReduceOp.MAX)  # 形状 (2,)

        # 非分布式或 world_size=1 时，以上操作等价于本地值
        denom_samples = float(vec_sum[4].item())  # 标量 ()
        if denom_samples <= 0.0:
            return None

        out = {
            "epoch/train_loss": float(vec_sum[0].item() / denom_samples),  # 标量 ()
            "epoch/kl_mean": float(vec_sum[1].item() / denom_samples),     # 标量 ()
            "epoch/adv_mean": float(vec_sum[2].item() / denom_samples),    # 标量 ()
            "epoch/adv_min": float(vec_min[1].item()),                     # 标量 ()
            "epoch/adv_max": float(vec_max[1].item()),                     # 标量 ()
            "epoch/ratio_mean": float(vec_sum[3].item() / denom_samples),  # 标量 ()
            "epoch/ratio_min": float(vec_min[0].item()),                   # 标量 ()
            "epoch/ratio_max": float(vec_max[0].item()),                   # 标量 ()
            "epoch/approx_kl": float(vec_sum[5].item() / denom_samples),   # 标量 ()
            "epoch/clipfrac_low": float(vec_sum[7].item() / max(1.0, float(vec_sum[9].item()))),   # 标量 ()
            "epoch/clipfrac_high": float(vec_sum[8].item() / max(1.0, float(vec_sum[9].item()))),  # 标量 ()
            "epoch/policy_loss": float(vec_sum[6].item() / denom_samples), # 标量 ()
        }
        # 主进程追加各子奖励均值（由 update_reward 预先聚合）
        if accelerator.is_main_process:
            for k, v in self.reward_means_by_key.items():
                out[f"epoch/reward_mean/{k}"] = float(v)
            if "avg" in self.reward_means_by_key:
                out["epoch/reward_mean"] = float(self.reward_means_by_key["avg"])  # 标量 ()
        return out


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
    # 将 (N,) 的 image_ids 与 rewards 按 image_id 排序并整形为 (B,K)
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

    # 名称 -> 稳定 id，并 gather 成全局
    name_ids_list = [name_to_stable_id(n) for n in image_names]  # 形状: 长度 N
    image_ids_local = torch.tensor(name_ids_list, device=device, dtype=torch.long)  # 形状: (N,)
    rewards_local = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # 形状: (N,)

    # 仅用本 rank 的 K 候选做硬排名胜率（自动推断 K 并断言所有组大小一致）
    sort_vals, sort_idx = torch.sort(image_ids_local)  # 形状: (N,), (N,)
    rewards_sorted = rewards_local.index_select(0, sort_idx)  # 形状: (N,)
    unique_ids_sorted, counts = torch.unique(sort_vals, return_counts=True)  # 形状: (≈N/K,), (≈N/K,)
    total = int(rewards_sorted.numel())  # 形状: 标量 (N)
    B = int(unique_ids_sorted.numel())  # 形状: 标量 (≈N/K)
    K = int(counts[0].item())  # 形状: 标量 (K)
    assert int(counts.min().item()) == K and int(counts.max().item()) == K
    assert total == B * K

    scores_bk = rewards_sorted.reshape(B, K)  # 形状: ((G*N)/K, K)
    diff = scores_bk.unsqueeze(2) - scores_bk.unsqueeze(1)  # 形状: ((G*N)/K, K, K)
    win_strict = (diff > 0).float()  # 形状: ((G*N)/K, K, K)
    wins = win_strict  # 形状: ((G*N)/K, K, K)
    eye = torch.eye(K, device=device, dtype=torch.float32).unsqueeze(0)  # 形状: (1,K,K)
    wins = wins * (1.0 - eye)  # 形状: ((G*N)/K, K, K)
    wr = wins.sum(dim=2) / max(1, K - 1)  # 形状: ((G*N)/K, K)
    if plus:
        adv_bk = wr  # 形状: ((G*N)/K, K)
    else:
        adv_bk = wr - 0.5  # 形状: ((G*N)/K, K)

    adv_sorted = adv_bk.reshape(total)  # 形状: (N)
    inv_idx = torch.empty_like(sort_idx)  # 形状: (G*N,)
    inv_idx[sort_idx] = torch.arange(total, device=device, dtype=torch.long)  # 形状: (N,)
    adv_local = adv_sorted.index_select(0, inv_idx)  # 形状: (N,)

    return adv_local.detach().cpu().numpy().astype(np.float64)  # 形状: (N,)

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
    - repeated_image_paths: 与 meshes 一一对应的图像路径列表
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
        if image_pil.mode == 'RGBA':
            image = image_pil  # 保留 RGBA（保留 alpha 通道，供 direct3d 使用）
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
    # 标准分布式采样器（本地优势模式）
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
    # 必填：config.train.optimizer 下必须提供以下字段
    opt = config.train.optimizer

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
        # 为 Adan 组装 3 元 betas，缺省 beta3=0.99；其他优化器使用 2 元 betas
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


# 已移除旧的 compute_advantages（依赖 tracking/global_std），direct3d 不再使用


def eval_direct3d(
    pipeline: Direct3DS2PipelineWithLogProb,
    test_dataloader: DataLoader,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    mesh_scorer: MeshScorer,
    generator: Optional[torch.Generator] = None,
    export_dir: Optional[str] = None,
    write_mesh: bool = False,
):
    """Direct3D 评估流程：逐图生成 1 个 mesh 并聚合奖励。"""
    all_rewards: Dict[str, List[np.ndarray]] = defaultdict(list)

    for eval_batch in tqdm(
        test_dataloader,
        desc="Eval:",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        images, image_paths, metadata = eval_batch
        with torch.inference_mode():  # 关闭梯度，省显存
            # 直接使用原始 PIL 图像做条件编码（Direct3D 内部完成预处理）
            cond_batch, neg_batch = pipeline.prepare_image_conditions(images)  # 形状: cond(B,P,C), neg_cond(B,P,C) 或 None

            # 使用 stage1_with_logprob 与训练策略一致（步数对齐、可设 deterministic）
            coords_list, _, _, _ = pipeline.stage1_with_logprob(
                cond_dict={"cond": cond_batch, "neg_cond": neg_batch},  # 形状: (B,P,C)/(B,P,C)|None
                num_inference_steps=int(config.sample.num_steps),  # 形状: 标量
                guidance_scale=float(config.sample.guidance_scale),  # 形状: 标量
                generator=generator,
                deterministic=bool(config.deterministic),  # 形状: 标量
                noise_level=float(config.slat_sampler_params.noise_level),
            )  # 返回 List[Tensor(N_i,4)]

            # 合批为稀疏（每图 1 候选 → BK=B）
            sparse_list = [SparseTensor(feats=torch.empty((c.shape[0], 1), device=c.device), coords=c) for c in coords_list]  # 形状: 列表(B × Sparse)
            coords_batched = prepare_sparse_tensor_batch(sparse_list, batch_size=len(sparse_list))  # 形状: batched 稀疏（候选级）

            sampler_params = SlatSamplerParams(
                mc_threshold=float(config.slat_sampler_params.mc_threshold),  # 形状: 标量
            )

            # 直接整批调用 Stage2（BK=B）
            meshes_batch, _, _, _ = pipeline.stage2_with_logprob(
                stage1_cond_dict={
                    "cond": cond_batch,        # 形状: (B,P,C)
                    "neg_cond": neg_batch,     # 形状: (B,P,C) 或 None
                    "coords": coords_batched,  # 形状: batched 稀疏
                },
                slat_sampler_params=sampler_params,
                num_inference_steps=int(config.sample.num_steps),  # 形状: 标量
                guidance_scale=float(config.sample.guidance_scale),  # 形状: 标量
                generator=generator,
                deterministic=bool(config.deterministic),  # 形状: 标量
                noise_level=float(config.slat_sampler_params.noise_level),
            )

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
                # 将 camera_normal 可视化与 mesh 导出对齐到相同 eval_epoch 目录，由 scorer 在内部根据 safe_base 建子目录
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


def repeat_image_conds(cond: torch.Tensor, k: int) -> torch.Tensor:
    # cond: (B, C) -> (B*k, C) 用于生成多个 candidates
    B, C = cond.shape  # (B, C)
    cond_expanded = cond.unsqueeze(1).expand(B, k, C).reshape(B * k, C)  # (B*k, C)
    return cond_expanded


def _move_batch_samples(batch_samples: list, device: torch.device, to_cpu: bool = False) -> None:
    """原地移动 batch 内样本的重资源字段（coords/slat/latents_seq）至 device 或 CPU。"""
    target = torch.device("cpu") if to_cpu else device
    for s in batch_samples:
        if "coords" in s and isinstance(s["coords"], torch.Tensor):
            s["coords"] = s["coords"].to(target)  # 形状: (N,4)
        if "slat" in s and hasattr(s["slat"], "to"):
            s["slat"] = s["slat"].to(target)  # 形状: 稀疏张量
        if "latents_seq" in s and isinstance(s["latents_seq"], (list, tuple)):
            s["latents_seq"] = [(t.to(target) if (t is not None and hasattr(t, "to")) else None) for t in s["latents_seq"]]  # 形状: [steps+1]

def build_stage1_cond(
    pipeline: Direct3DS2PipelineWithLogProb,
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

def build_pipeline(config: ml_collections.ConfigDict, accelerator: Accelerator) -> Direct3DS2PipelineWithLogProb:
    """构建并放置 Direct3D‑S2 Pipeline 到设备。"""
    pipeline = Direct3DS2PipelineWithLogProb.from_pretrained(
        config.pretrained.pipeline_path,
        subfolder=config.pretrained.subfolder,
        dtype=(torch.float16 if config.mixed_precision in ["fp16", "bf16"] else torch.float32),
        minimal_512_only=bool(config.pretrained.minimal_512_only),
        use_refiner=bool(config.pretrained.use_refiner),
    )
    pipeline.to(accelerator.device)
    return pipeline


def get_trainable_model_fp16(pipeline: Direct3DS2PipelineWithLogProb) -> nn.Module:
    """获取 Direct3D 可训练模块（Stage2 稀疏分支）。"""
    slat_model: nn.Module = pipeline.get_trainable_model_stage2()
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
    # 精简版：仅使用 config.lora.lora_rank；alpha 同 rank；dropout 固定 0.1；bias 固定 "none"
    lora_r = int(config.lora.lora_rank)
    lora_alpha = lora_r
    lora_dropout = 0.1
    lora_bias_mode = "none"
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias_mode,
    )
    lora_path = (config.train.lora_path if 'lora_path' in config.train else None)
    if isinstance(lora_path, str) and len(lora_path) > 0:
        slat_model = PeftModel.from_pretrained(slat_model, lora_path)
        slat_model.set_adapter("default")
    else:
        slat_model = get_peft_model(slat_model, lora_cfg)
    return slat_model


def prepare_optimizer_and_wrap(
    slat_model: nn.Module,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
    pipeline: Direct3DS2PipelineWithLogProb,
) -> tuple[nn.Module, optim.Optimizer, list]:
    """构建优化器，使用 accelerator.prepare 包装，并回写到 pipeline。"""
    trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, config)
    slat_model, optimizer = accelerator.prepare(slat_model, optimizer)
    # 回写到 pipeline 内部（统一由 pipeline 暴露）
    if hasattr(pipeline, "ref") and hasattr(pipeline.ref, "sparse_dit_512"):
        pipeline.ref.sparse_dit_512 = slat_model
    return slat_model, optimizer, trainable_params


def enable_gradient_checkpointing_if_needed(slat_model: nn.Module, accelerator: Accelerator, config: ml_collections.ConfigDict) -> None:
    """按配置为所有 block 启用梯度检查点。"""
    if bool(config.gradient_checkpointing):
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
    accelerator.load_state(str(chosen))
    accelerator.wait_for_everyone()
    name = chosen.name
    tail = name.split("_")[-1]
    start_epoch = (int(tail) + 1) if tail.isdigit() else 0
    accelerator.print(f"🔁 Resumed: {str(chosen)} → start_epoch={start_epoch}")
    return start_epoch
    
def run_eval_only(
    pipeline: Direct3DS2PipelineWithLogProb,
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

    # 开启 CameraNormalScorer 可视化（具体 vis_dir 在 eval_direct3d 内按 epoch 设置）
    if hasattr(mesh_scorer, "_camera_normal") and (mesh_scorer._camera_normal is not None):
        cn = mesh_scorer._camera_normal
        cn.cfg.save_vis = True

    if bool(config.train.ema) and ema is not None and trainable_params is not None:
        ema.copy_ema_to(trainable_params, store_temp=True)
        all_rewards_np = eval_direct3d(
            pipeline, eval_loader, config, accelerator, epoch=0, mesh_scorer=mesh_scorer, generator=gen, export_dir=export_dir, write_mesh=True
        )
        ema.copy_temp_to(trainable_params)
    else:
        all_rewards_np = eval_direct3d(
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

    # 训练时间步数量（与 SD3/Hunyuan3D 对齐，用于放大梯度累积步数）
    num_train_timesteps = int(float(config.sample.num_steps) * float(config.train.timestep_fraction) * float(config.train.timestep_keep_ratio))  # 标量

    # 基础加速器（将梯度累积步数乘以时间步数，对齐 SD3/Hunyuan3D）
    # 先确定 run_name，用于 Accelerate 的自动 checkpoint 命名
    run_name = config.run_name if len(config.run_name) > 0 else f"direct3d_s2_{int(time.time())}"
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=os.path.join(config.logdir, run_name)),
        log_with=["wandb"],
        gradient_accumulation_steps=int(config.train.gradient_accumulation_steps) * max(1, num_train_timesteps),  # 标量
    )
    set_seed(int(config.seed))
    setup_backend_determinism()

    # 规范化加速器跟踪器初始化，确保 accelerator.log 正常写入 W&B
    # 规范 run_name，保持与 RunDirs 一致
    config.run_name = run_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo-direct3d",
            config=dict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # 构建 Pipeline（评估与训练共享）
    pipeline = build_pipeline(config, accelerator)
    device = accelerator.device
    # 在初始化阶段按权重加载所需 scorer，避免无关模型初始化
    mesh_scorer = MeshScorer(
        device=device,
        verbose=bool(config.verbose),
        score_fns_cfg=dict(config.reward_fn),
        camera_normal_cfg=(dict(config.camera_normal) if 'camera_normal' in config else None),
    )

    # eval_only 提前返回：应用 LoRA 后评估，不做任何 DDP 包装/优化器等训练相关初始化
    if bool(config.eval_only):
        slat_model = get_trainable_model_fp16(pipeline)
        slat_model = apply_lora_if_needed(slat_model, config)
        if hasattr(pipeline, "ref") and hasattr(pipeline.ref, "sparse_dit_512"):
            pipeline.ref.sparse_dit_512 = slat_model
        dirs = RunDirs.from_config(config)
        run_logger = RunLogger(accelerator, dirs)
        trainable_params_eval = [p for p in slat_model.parameters() if p.requires_grad]
        ema_eval = create_ema_if_needed(trainable_params_eval, accelerator, config)
        if ema_eval is not None:
            accelerator.register_for_checkpointing(ema_eval)
        run_eval_only(pipeline, config, accelerator, mesh_scorer, run_logger, ema=ema_eval, trainable_params=trainable_params_eval)
        return

    # 构建训练对象（仅训练路径）
    slat_model = get_trainable_model_fp16(pipeline)
    slat_model = apply_lora_if_needed(slat_model, config)
    slat_model, optimizer, trainable_params = prepare_optimizer_and_wrap(slat_model, config, accelerator, pipeline)
    enable_gradient_checkpointing_if_needed(slat_model, accelerator, config)
    # 注册自定义持久化状态（EMA/TrainState/StatTracker）后再加载 checkpoint，确保被恢复
    ema = create_ema_if_needed(trainable_params, accelerator, config)
    if ema is not None:
        accelerator.register_for_checkpointing(ema)
    # 不再创建或注册按图像统计追踪器（去除 tracking/global_std）
    train_state = TrainState(global_step=0)
    accelerator.register_for_checkpointing(train_state)
    start_epoch = load_checkpoint(accelerator, config, mode="train")

    # 数据与奖励（仅训练路径）
    train_loader = dataloader_from_config(config, accelerator)


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
        epoch_start_time = time.perf_counter()
        # 本 epoch 训练指标聚合器（包含训练 loss/kl/adv/ratio 以及 reward 均值）
        epoch_logger = EpochMetricLogger()
        # 采样阶段：对每张图像生成 K 个候选并打分
        all_samples = []
        max_train_batches = int(config.sample.num_batches_per_epoch)
        # 为本 epoch 设置采样器随机种子，使本 epoch 的前 N 个 batch 与其他 epoch 不同
        train_loader.sampler.set_epoch(epoch)
        # 使用有限迭代器：当 num_batches_per_epoch>0 时截断；否则保持原始无限流（依赖上层控制）
        loader_iter = (
            itertools.islice(train_loader, max_train_batches)
            if max_train_batches > 0 else train_loader
        )
        for batch_idx, (batch_images, batch_paths, batch_meta) in enumerate(tqdm(loader_iter, disable=not accelerator.is_main_process)):
            with torch.inference_mode():  # 关闭梯度，省显存
                # 条件编码与阶段1/阶段2采样
                cond_batch, neg_batch = pipeline.prepare_image_conditions(batch_images)  # 形状: cond(B,P,C), neg_cond(B,P,C) 或 None
                k = int(config.sample.num_meshes_per_image)  # 形状: 标量
                # 依据 same_latent 开关，创建稳定生成器（批级别）
                use_same_latent = config.sample.same_latent  # 形状: 标量
                generator = (
                    create_train_generator_for_batch(accelerator.device, int(epoch), int(batch_idx), list(batch_paths))
                    if use_same_latent else None
                )  # 形状: 生成器 或 None
                stage1_cond_batch = build_stage1_cond(
                    pipeline=pipeline,
                    batch_paths=batch_paths,
                    cond_batch=cond_batch,
                    neg_batch=neg_batch,
                    num_steps_dense=int(getattr(config.sample, "num_inference_steps_dense", 20)),  # 形状: 标量
                    guidance_scale=float(config.sample.guidance_scale),  # 形状: 标量
                    generator=generator,
                    k=int(config.sample.num_meshes_per_image),
                )
                meshes, all_latents, all_log_probs, t_seq_out = pipeline.stage2_with_logprob(
                    num_inference_steps=int(config.sample.num_steps),  # 形状: 标量
                    guidance_scale=float(config.sample.guidance_scale),  # 形状: 标量
                    generator=generator,
                    deterministic=False,
                    slat_sampler_params=SlatSamplerParams(
                        mc_threshold=float(getattr(config.slat_sampler_params, "mc_threshold", 0.2)),  # 形状: 标量
                    ),
                    noise_level=float(config.slat_sampler_params.noise_level),
                    stage1_cond_dict=stage1_cond_batch,
                )

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
            reward_parts_local = {k: v for k, v in rewards_dict.items() if k != "avg"}  # 形状: 字典(各子项 -> (BK,))
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

            # 构建样本并将重资源转到 CPU（适配批输出：从 batched latents_seq/log_probs 提取候选切片）
            steps_eff = int(len(all_latents) - 1)  # 形状: 标量（有效步数 = len(latents_seq)-1）
            BK = len(meshes)  # 形状: 标量
            layouts_bk = all_latents[-1].layout  # 形状: 长度 BK
            t_seq = t_seq_out.detach().cpu().numpy()  # 形状: (steps_eff+1,)
            for s in range(BK):
                sl = layouts_bk[s]
                # 按步提取该候选的稀疏序列，并重置批索引到0
                latents_seq_cpu = []
                for j in range(steps_eff + 1):
                    batched_j = all_latents[j]
                    feats_j = batched_j.feats[sl].detach().cpu()  # 形状: (N_s, C)
                    coords_j = batched_j.coords[sl].clone().detach().cpu()  # 形状: (N_s, 4)
                    coords_j[:, 0] = 0  # 形状: (N_s, 4)
                    latents_seq_cpu.append(SparseTensor(feats=feats_j, coords=coords_j, layout=[slice(0, feats_j.shape[0])]))
                final_latent_cpu = latents_seq_cpu[-1]
                coords_cpu = final_latent_cpu.coords  # 形状: (N_s,4)
                # 该候选的每步对数概率向量
                old_log_probs_cpu = all_log_probs[:, s].detach().cpu()  # 形状: (steps_eff,)
                cond_patches_s = cond_batch[s // k:s // k + 1].detach().cpu()  # 形状: (1,P,C)
                neg_patches_s = (neg_batch[s // k:s // k + 1].detach().cpu() if (neg_batch is not None) else None)  # 形状: (1,P,C) 或 None
                all_samples.append({
                    "coords": coords_cpu,  # 形状: (N_s,4)
                    "slat": final_latent_cpu,  # 形状: SparseTensor(CPU)
                    "image_idx": 0,  # 形状: 标量
                    "latents_seq": latents_seq_cpu,  # 形状: [steps_eff+1]
                    "old_log_probs": old_log_probs_cpu,  # 形状: (steps_eff,)
                    "t_seq": t_seq,  # 形状: (steps_eff+1,)
                    "sampler_params": {
                        "deterministic": False,
                        "num_inference_steps": int(config.sample.num_steps),
                    },
                    "rewards":  {**{rk: float(rv[s]) for rk, rv in reward_parts_local.items()}, "avg": float(rewards[s])},  # 形状: 字典(子项标量 + avg 标量)
                    "image_name": batch_meta[s // k]["image_name"],
                    # 仅 patch 级条件（默认保留在 CPU）
                    "cond_patches": cond_patches_s,
                    "neg_patches": neg_patches_s,
                    "time_indices": np.arange(steps_eff, dtype=int),  # 形状: (steps_eff,)
                })

            del meshes, all_latents, all_log_probs
            torch.cuda.empty_cache()

        # 统计与优势（与 Hunyuan3D 一致：分布式聚合后按图像标准化）
        accelerator.wait_for_everyone()
        # 分布式聚合并缓存各子奖励的全局均值（由 EpochMetricLogger 统一输出）
        epoch_logger.update_reward(rewards_dict, accelerator)

        image_names = [s["image_name"] for s in all_samples]  # (N,)
        # 先按配置权重逐键累加总分，并记录每个 reward 的全局均值
        weights_dict = dict(config.reward_fn)
        enabled_keys = [k for k, v in weights_dict.items() if float(v) > 0.0]

        # 断言样本奖励键都在配置中（排除 'avg' 汇总项）
        first_rewards = all_samples[0]['rewards']
        for rk in first_rewards.keys():
            if rk == 'avg': continue
            assert rk in weights_dict, f"Missing reward weight in config for key: {rk}"

        N_local = len(all_samples)  # 形状: 标量

        # 计算优势
        adv_type = config.sample.adv_type  # 形状: 标量(字符串)
        adv_from = config.sample.adv_from  # 形状: 标量(字符串): 'seperate' 或 'average'
        rewards_local = np.zeros(N_local, dtype=np.float64)  # 形状: (N,)
        advantages_local = np.zeros(N_local, dtype=np.float64)  # 形状: (N,)

        if adv_from in ("average",):
            # 直接使用 scorer 的加权总分 avg（已随样本缓存）
            rewards_local = np.array([s['rewards']['avg'] for s in all_samples], dtype=np.float64)  # 形状: (N,)

            # 对 v_avg 一次性计算优势
            if adv_type == "winrate":
                advantages_local = compute_winrate_advantages_per_image(
                    image_names=image_names,
                    rewards_np_local=rewards_local,  # 形状: (N,)
                    accelerator=accelerator,
                )  # 形状: (N,)
            elif adv_type == "winrate_plus":
                advantages_local = compute_winrate_advantages_per_image(
                    image_names=image_names,
                    rewards_np_local=rewards_local,  # 形状: (N,)
                    accelerator=accelerator,
                    plus=True,
                )  # 形状: (N,)
            elif adv_type == "similarity":
                advantages_local = compute_advantages_per_image(
                    image_names=image_names,
                    rewards_np_local=rewards_local,  # 形状: (N,)
                    accelerator=accelerator,
                    epoch=epoch,
                )  # 形状: (N,)
            else:
                raise ValueError(f"Invalid adv_type: {adv_type}")
        elif adv_from in ("seperate",):
            # 逐子奖励各自求优势，再按权重相加（原逻辑）
            for k in enabled_keys:
                w = float(weights_dict[k])  # 形状: 标量
                v_k = np.array([s['rewards'][k] for s in all_samples], dtype=np.float64)  # 形状: (N,)
                if adv_type == "winrate":
                    adv_k = compute_winrate_advantages_per_image(
                        image_names=image_names,
                        rewards_np_local=v_k,
                        accelerator=accelerator,
                    )  # 形状: (N,)
                elif adv_type == "winrate_plus":
                    adv_k = compute_winrate_advantages_per_image(
                        image_names=image_names,
                        rewards_np_local=v_k,
                        accelerator=accelerator,
                        plus=True,
                    )  # 形状: (N,)
                elif adv_type == "similarity":
                    adv_k = compute_advantages_per_image(
                        image_names=image_names,
                        rewards_np_local=v_k,
                        accelerator=accelerator,
                        epoch=epoch,
                    )  # 形状: (N,)
                else:
                    raise ValueError(f"Invalid adv_type: {adv_type}")
                rewards_local += w * v_k  # 形状: (N,)
                advantages_local += w * adv_k  # 形状: (N,)
        else:
            raise ValueError(f"Invalid adv_from: {adv_from}")

        steps = int(all_samples[0]["old_log_probs"].shape[0])  # 形状: 标量（=steps_eff）
        old_log_probs = torch.stack([s["old_log_probs"] for s in all_samples], dim=0)  # shape: (N, steps)
        advantages = torch.from_numpy(advantages_local).to(torch.float32).unsqueeze(1).expand(-1, steps)  # shape: (N, steps)

        valid_samples_ratio = float((advantages.abs().sum(dim=1) != 0).float().mean().item()) if advantages.shape[0] > 0 else 0.0

        for idx, sample in enumerate(all_samples):
            sample.update({
                "old_log_probs_tensor": old_log_probs[idx].detach().clone(),  # shape: (steps,)
                "advantages_tensor": advantages[idx].detach().clone(),  # shape: (steps,)
            })

        actual_train_bs = max(1, int(getattr(config.train, "batch_size", 1)))  # 形状: 标量
        accelerator.wait_for_everyone()
        run_logger.log_sampling_stats(
            epoch=epoch,
            actual_batch_size=actual_train_bs,
            num_sub_batches=len(all_samples),
            valid_ratio=float(valid_samples_ratio),
        )

        # ===== 训练阶段：批量并行处理 =====
        slat_model.train()

        # 先用 fraction 取基础集合，再在其内随机不放回采样 keep_ratio（DDP 各 rank 一致，随 epoch 变化）
        steps_to_train = int(float(config.train.timestep_fraction) * steps)  # 形状: 标量
        base_indices = np.linspace(0, steps - 1, steps_to_train, dtype=int)  # 形状: (steps_to_train,)
        num_keep = int(round(float(config.train.timestep_keep_ratio) * steps_to_train))  # 形状: 标量
        rng = np.random.default_rng(int(config.seed) + int(epoch))  # 形状: 随机源
        train_step_indices = np.sort(rng.choice(base_indices, size=num_keep, replace=False).astype(int))  # 形状: (num_keep,)
        autocast_ctx = accelerator.autocast
        stage2_runtime_cfg = Stage2RuntimeConfig(
            guidance_scale=float(config.sample.guidance_scale),
            deterministic=bool(all_samples[0]["sampler_params"].get("deterministic", False)),
            compute_kl=bool(config.train.beta > 0.0),
            noise_level=float(config.slat_sampler_params.noise_level),
        )

        for inner_epoch in range(int(config.train.num_inner_epochs)):
            batch_iter = tqdm(
                range(0, len(all_samples), actual_train_bs),
                total=(len(all_samples) + actual_train_bs - 1) // actual_train_bs,
                disable=not accelerator.is_main_process,
                desc=f"Train Batches (inner {inner_epoch})",
                leave=False,
            )
            for batch_idx, batch_start in enumerate(batch_iter):
                batch_samples = all_samples[batch_start: batch_start + actual_train_bs]

                # 将当前 batch 的重资源字段搬到 GPU
                _move_batch_samples(batch_samples, accelerator.device, to_cpu=False)

                step_iter = tqdm(
                    train_step_indices,
                    total=len(train_step_indices),
                    disable=not accelerator.is_main_process,
                    desc=f"Inner Steps (batch {batch_idx})",
                    leave=False,
                )
                for j in step_iter:
                    j = int(j)
                    
                    with accelerator.accumulate(slat_model):
                        with autocast_ctx():
                            _, log_prob_vec, kl_vec = compute_log_prob_direct3d_stage2(
                                pipeline=pipeline,
                                samples=batch_samples,
                                j=j,
                                config=stage2_runtime_cfg,
                                detach_uncond=config.train.detach_uncond,
                            )  # 形状: (SparseTensor, (len(batch_samples),), (len(batch_samples),))

                        log_prob_val = log_prob_vec  # 形状: (len(batch_samples),)
                        kl_val = kl_vec  # 形状: (len(batch_samples),)
                        old_log_prob_vals = torch.stack(
                            [s["old_log_probs_tensor"][j] for s in batch_samples],
                            dim=0,
                        ).to(device=accelerator.device, dtype=log_prob_val.dtype)  # 形状: (len(batch_samples),)
                        adv_vals = torch.stack(
                            [s["advantages_tensor"][j] for s in batch_samples],
                            dim=0,
                        ).to(device=accelerator.device, dtype=log_prob_val.dtype)  # 形状: (len(batch_samples),)

                        adv_vals = torch.clamp(
                            adv_vals,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )  # 形状: (len(batch_samples),)
                        ratio = torch.exp(log_prob_val - old_log_prob_vals)  # 形状: (len(batch_samples),)

                        unclipped = -adv_vals * ratio  # 形状: (batch_size_actual,)
                        clipped = -adv_vals * torch.clamp(
                            ratio,
                            1.0 - float(config.train.clip_range),
                            1.0 + float(config.train.clip_range),
                        )  # 形状: (len(batch_samples),)

                        policy_loss_vec = torch.maximum(unclipped, clipped)  # 形状: (len(batch_samples),)
                        loss_vec = policy_loss_vec  # 形状: (len(batch_samples),)
                        if float(config.train.beta) > 0.0:
                            loss_vec = loss_vec + float(config.train.beta) * kl_val  # 形状: (batch_size_actual,)

                        loss_val = loss_vec.mean()  # 形状: ()

                        accelerator.backward(loss_val)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            slat_model.parameters(), config.train.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    # ===== 关键修复：防止快的 rank 跑太前导致死锁 =====
                    # 注意：必须在每个 step 后同步，不能只在 sync_gradients 时同步
                    # 否则在梯度累积的中间步骤，快的 rank 会跑太前
                    accelerator.wait_for_everyone()
                    # ===== 结束修复 =====
                    
                    train_state.global_step += 1
                    if bool(config.train.ema) and ema is not None:
                        ema.step([p for p in slat_model.parameters() if p.requires_grad], train_state.global_step)

                    with torch.no_grad():
                        ratio_detached = torch.exp(log_prob_val.detach() - old_log_prob_vals.detach())  # 形状: (batch_size_actual,)
                        unclipped_detached = -adv_vals.detach() * ratio_detached  # 形状: (batch_size_actual,)
                        clipped_detached = -adv_vals.detach() * torch.clamp(
                            ratio_detached,
                            1.0 - float(config.train.clip_range),
                            1.0 + float(config.train.clip_range),
                        )  # 形状: (batch_size_actual,)
                        policy_loss_detached = torch.maximum(unclipped_detached, clipped_detached)  # 形状: (batch_size_actual,)
                        loss_val_detached = policy_loss_detached  # 形状: (batch_size_actual,)
                        if float(config.train.beta) > 0.0:
                            loss_val_detached = loss_val_detached + float(config.train.beta) * kl_val.detach()  # 形状: (batch_size_actual,)

                        delta_detached = log_prob_val.detach() - old_log_prob_vals.detach()  # 形状: (batch_size_actual,)
                        approx_kl_detached = 0.5 * (delta_detached * delta_detached)  # 形状: (batch_size_actual,)
                        lower_bound = 1.0 - float(config.train.clip_range)  # 形状: 标量
                        upper_bound = 1.0 + float(config.train.clip_range)  # 形状: 标量
                        clipfrac_low_detached = (ratio_detached < lower_bound).float()  # 形状: (batch_size_actual,)
                        clipfrac_high_detached = (ratio_detached > upper_bound).float()  # 形状: (batch_size_actual,)

                        loss_val_detached_mean = loss_val_detached.mean()  # 形状: ()
                        approx_kl_detached_mean = approx_kl_detached.mean()  # 形状: ()
                        clipfrac_low_detached_mean = clipfrac_low_detached.mean()  # 形状: ()
                        clipfrac_high_detached_mean = clipfrac_high_detached.mean()  # 形状: ()
                        policy_loss_detached_mean = policy_loss_detached.mean()  # 形状: ()

                        epoch_logger.update(
                            loss_val_detached_mean,
                            kl_val.detach(),
                            adv_vals.detach(),
                            ratio_detached,
                            batch_size=len(batch_samples),
                        )
                        epoch_logger.update_ppo_metrics(
                            approx_kl_detached_mean,
                            clipfrac_low_detached_mean,
                            clipfrac_high_detached_mean,
                            policy_loss_detached_mean,
                            batch_size=len(batch_samples),
                        )

                # 将当前 batch 样本搬回 CPU 并清理缓存
                _move_batch_samples(batch_samples, accelerator.device, to_cpu=True)
                torch.cuda.empty_cache()


        accelerator.wait_for_everyone()
        # 本 epoch 结束：按调度记录一次到 W&B（步数用 epoch）
        if (epoch + 1) % max(1, schedule.log_every_epochs) == 0:
            run_logger.log_epoch_metrics(epoch, epoch_logger)

        # 评估节奏对齐：所有进程共同参与评估以避免分布式 gather 阻塞
        if int(config.eval_freq) > 0 and ((epoch + 1) % int(config.eval_freq) == 0):
            accelerator.wait_for_everyone()
            eval_loader = eval_dataloader_from_config(config, accelerator)
            eval_loader.sampler.set_epoch(epoch)
            # —— 评估固定生成器：所有 rank 使用完全相同的噪声序列（严格对齐） ——
            gen = create_eval_generator(accelerator.device, int(config.seed))
            # 使用 EMA 权重评估（如启用）
            if bool(config.train.ema) and ema is not None:
                trainable = [p for p in slat_model.parameters() if p.requires_grad]
                ema.copy_ema_to(trainable, store_temp=True)
                all_rewards_np = eval_direct3d(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
                ema.copy_temp_to(trainable)
            else:
                all_rewards_np = eval_direct3d(
                    pipeline, eval_loader, config, accelerator, epoch, mesh_scorer,
                    generator=gen, export_dir=str(dirs.viz_dir), write_mesh=False
                )
            accelerator.wait_for_everyone()
            run_logger.log_eval_rewards(epoch, all_rewards_np)

        # 保存节奏对齐：每 epoch 末保存（频率由调度控制）
        if (epoch + 1) % int(schedule.save_every_epochs) == 0:
            saver.save_epoch(
                epoch=epoch,
                slat_model=slat_model,
                optimizer=optimizer,
                config=config,
                ema=(ema if bool(config.train.ema) and ema is not None else None),
                use_lora=bool(config.use_lora),
            )

        # 可视化与上传：独立于保存频率（仅主进程执行文件写入）
        if schedule.save_visualizations and (epoch + 1) % int(schedule.viz_every_epochs) == 0 and viz.meshes is not None:
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
        run_name_dir = config.run_name if isinstance(config.run_name, str) and len(config.run_name) > 0 else "direct3d_s2"
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

    def log_epoch_metrics(self, epoch: int, epoch_logger: "EpochMetricLogger"):
        log_dict = epoch_logger.to_global_log_dict(self.accelerator)
        if self.accelerator.is_main_process and log_dict is not None:
            self.accelerator.log(log_dict, step=epoch + 1)

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