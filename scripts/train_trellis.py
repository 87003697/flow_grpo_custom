#!/usr/bin/env python3
"""
TRELLIS Stage 2 GRPO Training Script

- 两阶段：Stage 1 冻结在线生成稀疏坐标；Stage 2 对 SLatFlowModel 做 GRPO
- 复用 Hunyuan3D/SD3 的 GRPO 训练框架与指标定义
- 稀疏张量：coords(N,4) + feats(N,C)，接入 Flow Matching + SDE + LogProb
- 遵循 TRELLIS_DEV.md 约束：仅训练 Stage 2，无 try/except，无 fallback
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import hashlib
import torch.distributed as dist

import ml_collections
from absl import app
from ml_collections import config_flags
from PIL import Image
import wandb

# 项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS/GRPO 相关模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from flow_grpo.diffusers_patch.trellis_stage2_with_logprob import trellis_stage2_with_logprob
from flow_grpo.diffusers_patch.sparse_tensor_grpo import compute_log_prob_trellis_stage2, compute_log_prob_trellis_stage2_batched
from flow_grpo.stat_tracking import PerImageStatTracker
from flow_grpo.ema import EMAModuleWrapper
from reward_models.rewards_mesh import MeshScorer
from flow_grpo.diffusers_patch.trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
# 工具函数改为直接使用 pipeline 的 preprocess_image
# convert_trellis_to_trimesh 不再在训练路径中直接使用

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

    def update_reward_mean_from_local(self, rewards_np_local: np.ndarray, accelerator: Accelerator):
        """从当前进程的本地奖励向量计算全局均值并缓存。

        - rewards_np_local: 当前进程本地奖励，形状 (N,)
        - accelerator.gather 后得到形状 (G*N,)
        """
        reward_local_tensor = torch.as_tensor(rewards_np_local, device=accelerator.device, dtype=torch.float32)  # 形状 (N,)
        reward_global_tensor = accelerator.gather(reward_local_tensor)  # 形状 (G*N,)
        if accelerator.is_main_process:
            self.reward_mean = float(reward_global_tensor.mean().item())  # 标量 ()

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
        if self.reward_mean is not None:
            out["epoch/reward_mean"] = float(self.reward_mean)  # 标量 ()
        return out

    def to_global_log_dict(self, accelerator: Accelerator) -> Optional[Dict[str, float]]:
        """分布式全局聚合并返回日志字典（所有进程均需调用）。"""
        if self.num_steps == 0:
            return None
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
        if accelerator.is_main_process and self.reward_mean is not None:
            out["epoch/reward_mean"] = float(self.reward_mean)  # 标量 ()
        return out


def log_normal_similarity_pairs(accelerator: "Accelerator", pairs, step: int, prefix: str = "normal_similarity", max_pairs: int = 4):
    """将法线相似度的配对（图像侧法线 vs 渲染法线）记录到 W&B。

    - pairs: List[Dict]，每项包含 keys: "image_path", "image_normal_pil", "rendered_normal_pil", "mesh_index", "score"
    - step: 日志步
    - prefix: 指标前缀，例如 "normal_similarity" 或 "eval/normal_similarity"
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
    stat_tracker: Optional[PerImageStatTracker],
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

    rewards_global_tensor = accelerator.gather(rewards_local_tensor)  # 形状: (G*N,)
    image_ids_global_tensor = accelerator.gather(image_ids_local_tensor)  # 形状: (G*N,)

    # 计算全局优势（按图像分组或全局标准化），保持在 torch 上
    if stat_tracker is None:
        eps = 1e-8  # 标量
        mean = rewards_global_tensor.mean()  # ()
        std = rewards_global_tensor.std()  # ()
        advantages_global_tensor = (rewards_global_tensor - mean) / (std + eps)  # 形状: (G*N,)
    else:
        advantages_global_tensor = stat_tracker.update_torch(image_ids_global_tensor, rewards_global_tensor)  # 形状: (G*N,)

    # 记录 per-image 统计（仅主进程）
    if accelerator.is_main_process and stat_tracker is not None:
        group_size, trained_image_num = stat_tracker.get_stats()  # 标量, 标量
        unique_ids = torch.unique(image_ids_global_tensor)  # 形状: (≈(G*N)/K,)
        if unique_ids.numel() > 0:
            stds = []
            for uid in unique_ids.tolist():
                mask = (image_ids_global_tensor == uid)
                stds.append(rewards_global_tensor[mask].std().item())
            zero_std_ratio = float((torch.tensor(stds) == 0).sum().item() / len(stds))
        else:
            zero_std_ratio = 0.0  # 标量
        accelerator.log(
            {
                "group_size": float(group_size),
                "trained_image_num": int(trained_image_num),
                "zero_std_ratio": zero_std_ratio,
            },
            step=epoch,
        )

    # 切回当前进程本地片段
    local_n = rewards_np_local.shape[0]  # 标量 (N)
    world = accelerator.num_processes  # 标量 (G)
    rank = accelerator.process_index  # 标量
    assert advantages_global_tensor.numel() % world == 0, "Global advantages size not divisible by world size"
    per_rank = advantages_global_tensor.numel() // world  # 标量 (N)
    assert per_rank == local_n, "Local sample count mismatch across processes"
    advantages_local_tensor = advantages_global_tensor.reshape(world, per_rank)[rank]  # 形状: (N,)

    return advantages_local_tensor.detach().cpu().numpy().astype(np.float64)  # 形状: (N,)


def compute_winrate_advantages_per_image(
    image_names: List[str],
    rewards_np_local: np.ndarray,
    accelerator: Accelerator,
    stat_tracker: Optional[PerImageStatTracker],
) -> np.ndarray:
    """按图像计算“硬排名胜率优势”（winrate-0.5），分布式聚合后切回本地。

    形状约定：
        - N: 当前进程样本数（通常 N = B_local*K）
        - K: 每图候选数
        - G: 进程数（全局维度为 G*N）

    - 无 stat_tracker: 基于当前组内 K 个候选 wins/(K-1) - 0.5
    - 有 stat_tracker: 将历史分数并入对手池 wins/(K-1+H) - 0.5
    返回：优势 (N,)，与 `image_names`/`rewards_np_local` 对齐。
    """
    device = accelerator.device  # 形状: 标量

    # 名称 -> 稳定 id，并 gather 成全局
    name_ids_list = [name_to_stable_id(n) for n in image_names]  # 形状: 长度 N
    image_ids_local = torch.tensor(name_ids_list, device=device, dtype=torch.long)  # 形状: (N,)
    rewards_local = torch.as_tensor(rewards_np_local, device=device, dtype=torch.float32)  # 形状: (N,)

    image_ids_global = accelerator.gather(image_ids_local)  # 形状: (G*N,)
    rewards_global = accelerator.gather(rewards_local)  # 形状: (G*N,)

    if stat_tracker is None:
        # 仅用当前 K 候选做硬排名胜率（自动推断 K 并断言所有组大小一致）
        sort_vals, sort_idx = torch.sort(image_ids_global)  # 形状: (G*N,), (G*N,)
        rewards_sorted = rewards_global.index_select(0, sort_idx)  # 形状: (G*N,)
        unique_ids_sorted, counts = torch.unique(sort_vals, return_counts=True)  # 形状: (≈(G*N)/K,), (≈(G*N)/K,)
        total = int(rewards_sorted.numel())  # 形状: 标量 (G*N)
        B = int(unique_ids_sorted.numel())  # 形状: 标量 (≈(G*N)/K)
        K = int(counts[0].item())  # 形状: 标量 (K)
        assert int(counts.min().item()) == K and int(counts.max().item()) == K
        assert total == B * K

        scores_bk = rewards_sorted.reshape(B, K)  # 形状: ((G*N)/K, K)
        diff = scores_bk.unsqueeze(2) - scores_bk.unsqueeze(1)  # 形状: ((G*N)/K, K, K)
        win_strict = (diff > 0).float()  # 形状: ((G*N)/K, K, K)
        tie = (diff == 0).float()  # 形状: ((G*N)/K, K, K)
        wins = win_strict + 0.5 * tie  # 形状: ((G*N)/K, K, K)
        eye = torch.eye(K, device=device, dtype=torch.float32).unsqueeze(0)  # 形状: (1,K,K)
        wins = wins * (1.0 - eye)  # 形状: ((G*N)/K, K, K)
        wr = wins.sum(dim=2) / max(1, K - 1)  # 形状: ((G*N)/K, K)
        adv_bk = wr - 0.5  # 形状: ((G*N)/K, K)

        adv_sorted = adv_bk.reshape(total)  # 形状: (G*N)
        inv_idx = torch.empty_like(sort_idx)  # 形状: (G*N,)
        inv_idx[sort_idx] = torch.arange(total, device=device, dtype=torch.long)  # 形状: (G*N,)
        adv_global = adv_sorted.index_select(0, inv_idx)  # 形状: (G*N,)
    else:
        # 将历史分数并入对手池（并写回历史池，使用全局聚合后的分数以保持一致）
        adv_global = torch.zeros_like(rewards_global, dtype=torch.float32)  # 形状: (G*N,)
        unique_ids = torch.unique(image_ids_global)  # 形状: (≈(G*N)/K,)
        for uid in unique_ids.tolist():
            mask = (image_ids_global == uid)  # 形状: (G*N,)
            cur = rewards_global[mask]  # 形状: (K,)（该图像的当前 K 个候选）
            K = int(cur.shape[0])  # 形状: 标量 (K)
            hist_list = stat_tracker.stats.get(int(uid), []) if hasattr(stat_tracker, "stats") else []  # 形状: list[float]
            hist = (
                torch.tensor(hist_list, device=device, dtype=torch.float32)
                if len(hist_list) > 0 else torch.empty(0, device=device, dtype=torch.float32)
            )  # 形状: (H,) 或 (0,)
            pool = torch.cat([cur, hist], dim=0)  # 形状: (K+H,)
            diff = cur.unsqueeze(1) - pool.unsqueeze(0)  # 形状: (K, K+H)
            win_strict = (diff > 0).float()  # 形状: (K, K+H)
            tie = (diff == 0).float()  # 形状: (K, K+H)
            H = int(hist.shape[0])  # 形状: 标量
            self_mask = torch.zeros((K, K + H), device=device, dtype=torch.bool)  # 形状: (K,K+H)
            if K > 0:
                self_mask[:, :K] |= torch.eye(K, device=device, dtype=torch.bool)  # 形状: (K,K)
            wins = (win_strict + 0.5 * tie).masked_fill(self_mask, 0.0)  # 形状: (K, K+H)
            denom = max(1, K - 1 + H)  # 形状: 标量
            wr = wins.sum(dim=1) / denom  # 形状: (K,)
            adv = wr - 0.5  # 形状: (K,)
            adv_global[mask] = adv  # 形状: (G*N,)

            # —— 写回历史池：使用全局聚合分数，进程间一致 ——
            uid_int = int(uid)  # 形状: 标量
            stat_tracker.stats.setdefault(uid_int, []).extend(cur.detach().cpu().tolist())  # 形状: 追加 K 个 float
            stat_tracker.history_images.add(uid_int)  # 形状: 集合大小+1

    # 切回当前进程本地片段
    world = accelerator.num_processes  # 形状: 标量 (G)
    rank = accelerator.process_index  # 形状: 标量
    assert adv_global.numel() % world == 0
    per_rank = adv_global.numel() // world  # 形状: 标量 (N)
    start = rank * per_rank  # 形状: 标量
    end = start + per_rank  # 形状: 标量
    adv_local = adv_global[start:end]  # 形状: (N,)

    return adv_local.detach().cpu().numpy().astype(np.float64)  # 形状: (N,)

def save_meshes_for_preview(
    meshes,
    repeated_image_paths,
    rewards,
    epoch: int,
    save_dir: str,
    device_str: str = "cuda",
):
    """保存 mesh 为 OBJ 并渲染预览 PNG（对齐 Hunyuan3D 的可视化体验）。

    - meshes: List[kiui.Mesh]
    - repeated_image_paths: 与 meshes 一一对应的图像路径列表（每图像重复 K 次）
    - rewards: 与 meshes 对齐的奖励向量（可选，仅用于命名/扩展，当前未写入文件名）
    - epoch: 当前 epoch 编号
    - save_dir: 输出目录
    - device_str: 渲染设备字符串（"cuda" 或 "cpu"）
    """
    import os
    from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_for_training

    os.makedirs(save_dir, exist_ok=True)

    mesh_files = []
    preview_files = []

    for idx, (mesh, img_path) in enumerate(zip(meshes, repeated_image_paths)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        safe_base = "".join(c for c in base if c.isalnum() or c in (" ", "-", "_")).rstrip()

        mesh_path = os.path.join(save_dir, f"{safe_base}_mesh_{idx}.obj")
        preview_path = os.path.join(save_dir, f"{safe_base}_preview_{idx}.png")

        mesh.write(mesh_path)
        render_mesh_for_training(mesh_path, preview_path, device=device_str)

        mesh_files.append(mesh_path)
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
        image = Image.open(image_path).convert('RGB')
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


class DistributedImageRepeatSampler(Sampler):
    """分布式重复采样器（与 SD3/Hunyuan3D 对齐的 K-repeat 采样器）。

    - dataset: Image3DDataset
    - batch_size: 每卡输入图像数（形如 B）
    - k: 每图生成候选数（形如 K）
    - num_replicas: 总卡数（形如 G）
    - rank: 当前卡编号 [0, G)
    - seed: 随机种子
    """
    def __init__(self, dataset: Dataset, batch_size: int, k: int, num_replicas: int, rank: int, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)

        self.total_samples = self.num_replicas * self.batch_size  # 标量
        if self.total_samples < self.k:
            self.simple_repeat_mode = True
            self.m = self.total_samples  # 标量
        else:
            assert self.total_samples % self.k == 0, f"k can not div n*b, k{self.k}-num_replicas{self.num_replicas}-batch_size{self.batch_size}"
            self.simple_repeat_mode = False
            self.m = self.total_samples // self.k  # 标量

        self.epoch = 0

    def __iter__(self):
        # 使用同一个随机生成器跨多次 yield 前进，确保同一 epoch 内批次不同
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            if self.simple_repeat_mode:
                # 可用图像索引（随机顺序）
                available = torch.randperm(len(self.dataset), generator=g).tolist()  # 形状 (N,)
                # 填满所有卡的 batch
                repeated = [available[i % len(available)] for i in range(self.total_samples)]  # 形状 (G*B,)
                # 切分到各卡
                per_card = [repeated[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_replicas)]
                yield per_card[self.rank]
            else:
                indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()  # 形状 (m,)
                repeated = [idx for idx in indices for _ in range(self.k)]  # 形状 (G*B,)
                shuffled_idx = torch.randperm(len(repeated), generator=g).tolist()  # 形状 (G*B,)
                shuffled = [repeated[i] for i in shuffled_idx]  # 形状 (G*B,)
                per_card = [shuffled[i*self.batch_size:(i+1)*self.batch_size] for i in range(self.num_replicas)]
                yield per_card[self.rank]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    # 若启用 camera_normal，则传入 normal 缓存目录与分辨率，便于下游 query=normal_image 使用
    if 'camera_normal' in config:
        normal_cache_dir = str(config.camera_normal.cache_dir) if 'cache_dir' in config.camera_normal else None
        normal_resolution = int(config.camera_normal.normal_resolution) if 'normal_resolution' in config.camera_normal else None
        if isinstance(normal_cache_dir, str) and len(normal_cache_dir) > 0 and isinstance(normal_resolution, int) and normal_resolution > 0:
            dataset = Image3DDataset(config.data_dir, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)
        else:
            dataset = Image3DDataset(config.data_dir)
    else:
        dataset = Image3DDataset(config.data_dir)
    # 分布式 K-repeat 采样器（与 SD3/Hunyuan3D 对齐）
    batch_size = int(config.sample.input_batch_size)
    k = int(config.sample.num_meshes_per_image)
    sampler = DistributedImageRepeatSampler(
        dataset,
        batch_size=batch_size,
        k=k,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=int(config.seed),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=Image3DDataset.collate_fn,
    )
    return loader


def eval_dataloader_from_config(config: ml_collections.ConfigDict, accelerator: Accelerator) -> DataLoader:
    # 与训练集保持一致的 metadata 字段（包含 normal_path）
    if 'camera_normal' in config:
        normal_cache_dir = str(config.camera_normal.cache_dir) if 'cache_dir' in config.camera_normal else None
        normal_resolution = int(config.camera_normal.normal_resolution) if 'normal_resolution' in config.camera_normal else None
        if isinstance(normal_cache_dir, str) and len(normal_cache_dir) > 0 and isinstance(normal_resolution, int) and normal_resolution > 0:
            eval_dataset = Image3DDataset(config.data_dir, normal_cache_dir=normal_cache_dir, normal_resolution=normal_resolution)
        else:
            eval_dataset = Image3DDataset(config.data_dir)
    else:
        eval_dataset = Image3DDataset(config.data_dir)

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
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.adam_weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            params,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.adam_weight_decay,
        )
    return optimizer


def compute_advantages(rewards: np.ndarray, stat_tracker: PerImageStatTracker, image_names: List[str], use_global_std: bool) -> np.ndarray:
    """将字符串图像名映射为整数ID，并计算标准化优势。

    与 PerImageStatTracker 接口对齐：update(image_ids, rewards)
    """
    if stat_tracker is None:
        # 纯全局标准化：A = (r - mean) / (std + eps)
        rewards_arr = np.array(rewards, dtype=np.float64)
        mean = float(rewards_arr.mean())
        std = float(rewards_arr.std())
        eps = 1e-8
        advantages = (rewards_arr - mean) / (std + eps)
        return advantages.astype(np.float64)

    # 根据配置同步归一化策略
    stat_tracker.global_std = bool(use_global_std)

    # 将图像名映射为稳定的整数ID
    unique_names = list(set(image_names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    image_ids = np.array([name_to_id[name] for name in image_names], dtype=np.int64)

    # 计算优势（PerImageStatTracker 内部完成标准化）
    advantages = stat_tracker.update(image_ids, np.array(rewards, dtype=np.float64))
    return advantages


def eval_trellis(
    pipeline: TrellisStage2Pipeline,
    test_dataloader: DataLoader,
    config: ml_collections.ConfigDict,
    accelerator: Accelerator,
    epoch: int,
    mesh_scorer: MeshScorer,
    generator: Optional[torch.Generator] = None,
):
    """Trellis 评估流程（对齐 SD3/Hunyuan3D 的评估风格）"""
    model = pipeline.get_trainable_model()
    model.eval()

    all_rewards: Dict[str, List[np.ndarray]] = defaultdict(list)

    for eval_batch in tqdm(
        test_dataloader,
        desc="Eval:",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        images, image_paths, metadata = eval_batch

        # 预处理图像以与官方一致（背景移除 + 裁剪 + 518x518）
        images_proc = [pipeline.core_pipeline.preprocess_image(img) for img in images]

        # 编码图像条件（与训练一致）
        cond_dict = pipeline.prepare_image_conditions(images_proc)  # {'cond': (B, P, C), 'neg_cond': (B,P,C)

        # 生成mesh（评估每图只生成1个候选）
        meshes, _, _, _ = trellis_stage2_with_logprob(
            pipeline=pipeline,
            num_inference_steps=int(config.sample.num_steps),
            guidance_scale=float(config.sample.guidance_scale),
            generator=generator,
            kl_reward=0.0,
            deterministic=bool(config.deterministic),
            sparse_structure_sampler_params=dict(config.sparse_structure_sampler_params),
            slat_sampler_params=dict(config.slat_sampler_params),
            stage1_cond_dict=cond_dict,
            num_candidates=1,
            output_type="kiui",
        )

        meshes = [m.to(accelerator.device) for m in meshes]

        # 计算奖励并聚合到主进程
        rewards_dict, meta_out = mesh_scorer.score(meshes, images, metadata, dict(config.reward_fn))
        for key, value in rewards_dict.items():
            gathered = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(gathered)

    # 聚合并返回，由上层统一日志上报
    all_rewards_np = {key: (np.concatenate(v) if len(v) > 0 else np.array([])) for key, v in all_rewards.items()}
    return all_rewards_np


def repeat_image_conds(cond: torch.Tensor, k: int) -> torch.Tensor:
    # cond: (B, C) -> (B*k, C) 用于生成多个 candidates
    B, C = cond.shape  # (B, C)
    cond_expanded = cond.unsqueeze(1).expand(B, k, C).reshape(B * k, C)  # (B*k, C)
    return cond_expanded


def build_pipeline(config: ml_collections.ConfigDict, accelerator: Accelerator) -> TrellisStage2Pipeline:
    """构建并放置 Trellis Stage2 Pipeline 到设备，设置 verbose 环境变量。"""
    pipeline = TrellisStage2Pipeline(model_path=config.pretrained.model, verbose=bool(config.verbose))
    os.environ["TRELLIS_VERBOSE"] = "1" if bool(config.verbose) else "0"
    device = accelerator.device
    pipeline.to(device)
    if device.type == 'cuda':
        pipeline.cuda()
    return pipeline


def get_trainable_model_fp16(pipeline: TrellisStage2Pipeline) -> nn.Module:
    """获取可训练模型并切换到 FP16（如支持）。"""
    slat_model: nn.Module = pipeline.get_trainable_model()
    # 硬编码：默认使用 FP16 管理模型权重/模块（不考虑 FP32/BF16 权重管理）
    slat_model.convert_to_fp16()
    slat_model.use_fp16 = True
    slat_model.dtype = torch.float16
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
    pipeline: TrellisStage2Pipeline,
) -> tuple[nn.Module, optim.Optimizer, list]:
    """构建优化器，使用 accelerator.prepare 包装，并回写到 pipeline。"""
    trainable_params = [p for p in slat_model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, config)
    slat_model, optimizer = accelerator.prepare(slat_model, optimizer)
    if 'slat_flow_model' in pipeline.core_pipeline.models:
        pipeline.core_pipeline.models['slat_flow_model'] = slat_model
    return slat_model, optimizer, trainable_params


def enable_gradient_checkpointing_if_needed(slat_model: nn.Module, accelerator: Accelerator, config: ml_collections.ConfigDict) -> None:
    """按配置为所有 block 启用梯度检查点。"""
    if bool(config.gradient_checkpointing):
        unwrapped = accelerator.unwrap_model(slat_model)
        for blk in unwrapped.blocks:
            blk.use_checkpoint = True


def resume_checkpoint_if_needed(slat_model: nn.Module, optimizer: optim.Optimizer, accelerator: Accelerator, config: ml_collections.ConfigDict) -> None:
    """从 config.resume_from 恢复最小状态（如存在）。"""
    if isinstance(config.resume_from, str) and len(config.resume_from) > 0:
        ckpt_path = Path(config.resume_from) / "pytorch_model.bin"
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location="cpu")
            slat_model.load_state_dict(state.get("model", state))
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            accelerator.print(f"🔁 Resumed from {str(ckpt_path)}")


def create_ema_if_needed(trainable_params: list, accelerator: Accelerator, config: ml_collections.ConfigDict) -> Optional[EMAModuleWrapper]:
    """按配置创建 EMA 包装器。"""
    if bool(config.train.ema):
        ema_decay = float(config.train.ema_decay)
        return EMAModuleWrapper(trainable_params, decay=ema_decay, device=accelerator.device)
    return None


def main(_):
    config: ml_collections.ConfigDict = _CONFIG.value

    # 训练时间步数量（与 SD3/Hunyuan3D 对齐，用于放大梯度累积步数）
    num_train_timesteps = int(float(config.sample.num_steps) * float(config.train.timestep_fraction))  # 标量

    # 基础加速器（将梯度累积步数乘以时间步数，对齐 SD3/Hunyuan3D）
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=config.logdir),
        log_with=["wandb"],
        gradient_accumulation_steps=int(config.train.gradient_accumulation_steps) * max(1, num_train_timesteps),  # 标量
    )
    set_seed(int(config.seed))
    setup_backend_determinism()

    # 规范化加速器跟踪器初始化，确保 accelerator.log 正常写入 W&B
    if accelerator.is_main_process:
        run_name = config.run_name if len(config.run_name) > 0 else f"trellis_stage2_{int(time.time())}"
        accelerator.init_trackers(
            project_name="flow-grpo-trellis",
            config=dict(config),
            init_kwargs={"wandb": {"name": run_name}},
        )

    # 构建组件并准备训练对象
    pipeline = build_pipeline(config, accelerator)
    device = accelerator.device
    slat_model = get_trainable_model_fp16(pipeline)
    slat_model = apply_lora_if_needed(slat_model, config)
    slat_model, optimizer, trainable_params = prepare_optimizer_and_wrap(slat_model, config, accelerator, pipeline)
    enable_gradient_checkpointing_if_needed(slat_model, accelerator, config)
    resume_checkpoint_if_needed(slat_model, optimizer, accelerator, config)
    ema = create_ema_if_needed(trainable_params, accelerator, config)

    # 数据与奖励
    train_loader = dataloader_from_config(config, accelerator)
    # 在初始化阶段按权重加载所需 scorer，避免无关模型初始化
    mesh_scorer = MeshScorer(
        device=device,
        verbose=bool(config.verbose),
        score_fns_cfg=dict(config.reward_fn),
        camera_normal_cfg=(dict(config.camera_normal) if 'camera_normal' in config else None),
    )

    # 按配置启用/禁用按图像统计
    stat_tracker = PerImageStatTracker(global_std=config.sample.global_std) if bool(config.per_image_stat_tracking) else None


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

    for epoch in range(config.num_epochs):
        # 本 epoch 训练指标聚合器（包含训练 loss/kl/adv/ratio 以及 reward 均值）
        epoch_logger = EpochMetricLogger()
        # 采样阶段：对每张图像生成 K 个候选并打分
        all_samples = []
        accelerator.print(f"[Epoch {epoch}] Sampling...")
        max_train_batches = int(config.sample.num_batches_per_epoch)
        # 为本 epoch 设置采样器随机种子，使本 epoch 的前 N 个 batch 与其他 epoch 不同
        train_loader.batch_sampler.set_epoch(epoch)
        # 使用有限迭代器：当 num_batches_per_epoch>0 时截断；否则保持原始无限流（依赖上层控制）
        loader_iter = (
            itertools.islice(train_loader, max_train_batches)
            if max_train_batches > 0 else train_loader
        )
        for batch_idx, (batch_images, batch_paths, batch_meta) in enumerate(tqdm(loader_iter, disable=not accelerator.is_main_process)):
            # 提取图像条件（DINOv2 / 官方接口），先进行与官方一致的预处理
            batch_images_processed = [pipeline.core_pipeline.preprocess_image(img) for img in batch_images]
            cond_dict = pipeline.prepare_image_conditions(batch_images_processed)  # {'cond': (B, P, C), 'neg_cond': (B,P,C)}
            # 采样端不再计算向量级条件，统一使用 patch 级 cond/neg_cond（形状 (B, P, C)）

            # 统一仅保留 patch 级条件：采样端与训练端均用 patch 级 cond/neg_cond
            k = int(config.sample.num_meshes_per_image)

            meshes, all_latents, all_log_probs, all_kl = trellis_stage2_with_logprob(
                pipeline=pipeline,
                num_inference_steps=int(config.sample.num_steps),
                guidance_scale=float(config.sample.guidance_scale),
                kl_reward=float(config.sample.kl_reward),
                deterministic=False,
                sparse_structure_sampler_params=dict(config.sparse_structure_sampler_params),
                slat_sampler_params=dict(config.slat_sampler_params),
                stage1_cond_dict=cond_dict,
                num_candidates=int(config.sample.num_meshes_per_image),
                output_type="kiui",
                verbose=bool(config.verbose),
            )

            # 将 mesh 迁移到与 scorer 相同设备，避免 CPU/GPU 混用
            meshes = [m.to(device) for m in meshes]

            # 打分
            # 为每个样本补充 image_path（供 camera_normal 使用）
            repeated_meta = []
            for meta_item, path in zip(batch_meta, batch_paths):
                m = dict(meta_item)
                m["image_path"] = path
                repeated_meta.extend([m] * k)

            # 将图像按每张重复 K 次以与 meshes/repeated_meta 顺序完全对齐
            repeated_images = []
            for img in batch_images:
                repeated_images.extend([img] * k)
            rewards_dict, meta_out = mesh_scorer.score(meshes, repeated_images, repeated_meta, dict(config.reward_fn))
            rewards = rewards_dict["avg"]  # np.ndarray

            # 更新可视化缓存（仅首个 batch，避免过多显存/日志）
            if batch_idx == 0:
                # 展开 batch 的 image_paths（每图重复 K 次），与 meshes 对齐
                repeated_paths = []
                for p in batch_paths:
                    repeated_paths.extend([p] * k)
                num_samples_to_cache = min(2, len(meshes))
                cached_meshes = meshes[:num_samples_to_cache]
                cached_paths = repeated_paths[:num_samples_to_cache]
                cached_rewards = rewards[:num_samples_to_cache]
                viz.update_from_batch(
                    cached_meshes,
                    cached_paths,
                    cached_rewards,
                    meta_out.get("camera_normal_pairs_best", None),
                    meta_out.get("camera_normal_pairs_worst", None),
                )

            # 记录样本条目（逐样本切片 log_prob/latent），并构造 step 索引用于后续时间维随机打乱
            steps = int(config.sample.num_steps)
            for s in range(len(meshes)):
                # per-sample latent/logprob 切片
                latent_start = s * (steps + 1)
                latent_end = latent_start + (steps + 1)
                logprob_start = s * steps
                logprob_end = logprob_start + steps

                final_latent = all_latents[latent_end - 1]
                # 保存整条时序（单步重算所需）
                latents_seq = all_latents[latent_start:latent_end]  # 长度 steps+1
                sample_log_probs = all_log_probs[logprob_start:logprob_end]  # 长度 steps
                old_log_probs = torch.stack(sample_log_probs)  # 形状 [steps]

                # 对齐采样器的时间序列（与 trellis_flow_euler_sampler_with_logprob 一致）
                t_seq = np.linspace(1.0, 0.0, steps + 1) * 1000
                t_seq = float(config.slat_sampler_params.rescale_t) * t_seq / (1 + (float(config.slat_sampler_params.rescale_t) - 1) * t_seq / 1000)

                # 保存对应的条件（用于训练期重算）
                # 只保存 patch 级 cond/neg_cond（统一接口）
                cond_patches_s = cond_dict['cond'][s//k:s//k+1]  # 形状 (1, P, C)
                neg_patches_s = cond_dict['neg_cond'][s//k:s//k+1]  # 形状 (1, P, C)

                all_samples.append({
                    "coords": final_latent.coords,     # (N,4)
                    "slat": final_latent,             # SparseTensor
                    "image_idx": 0,                   # 重算时索引（与image_conds对齐，传入单样本条件）
                    "latents_seq": latents_seq,      # [steps+1] SparseTensor 序列
                    "old_log_probs": old_log_probs,  # [steps] 采样期每步 log_prob
                    "t_seq": t_seq,                  # [steps+1] 时间序列（numpy数组）
                    "sampler_params": {               # 采样期保存的关键信息（训练期严格一致性校验）
                        "deterministic": False,
                        "sigma_min": float(config.slat_sampler_params.sigma_min),
                        "rescale_t": float(config.slat_sampler_params.rescale_t),
                        "num_inference_steps": int(config.sample.num_steps),
                    },
                    "reward": float(rewards[s]),     # 标量
                    "image_name": batch_meta[s // k]["image_name"],
                    # 仅 patch 级条件
                    "cond_patches": cond_patches_s,
                    "neg_patches": neg_patches_s,
                    "time_indices": np.arange(steps, dtype=int),  # (steps,)
                })

            # 采样期不落盘：可视化统一在 epoch 末处理（已在上方 batch_idx==0 缓存首个 batch）

            # 迭代次数由 islice 控制，这里无需手动 break

        # 统计与优势（与 Hunyuan3D 一致：分布式聚合后按图像标准化）
        image_names = [s["image_name"] for s in all_samples]  # (N,)
        rewards_np_local = np.array([s["reward"] for s in all_samples], dtype=np.float64)  # (N,)
        adv_type = config.sample.adv_type
        if adv_type == "winrate":
            advantages_local_np = compute_winrate_advantages_per_image(
                image_names=image_names,
                rewards_np_local=rewards_np_local,
                accelerator=accelerator,
                stat_tracker=stat_tracker,
            )  # 形状: (N,)
        else:
            advantages_local_np = compute_advantages_per_image(
                image_names=image_names,
                rewards_np_local=rewards_np_local,
                accelerator=accelerator,
                stat_tracker=stat_tracker,
                epoch=epoch,
            )  # 形状: (N,)

        # 更新本 epoch 的奖励均值（分布式聚合后）
        epoch_logger.update_reward_mean_from_local(rewards_np_local, accelerator)

        # ===== 先拼后切：构造“字典的张量” =====
        # 聚合条件（patch 级）
        pos_cond_batched = torch.cat([s["cond_patches"] for s in all_samples], dim=0)  # (N, P, C)
        neg_cond_batched = torch.cat([s["neg_patches"] for s in all_samples], dim=0)  # (N, P, C)

        # 旧 log_prob 与优势（时间维复制）
        old_log_probs = torch.stack([s["old_log_probs"] for s in all_samples], dim=0).to(accelerator.device)  # (N, steps)
        steps = int(config.sample.num_steps)  # 标量
        adv_vec = torch.from_numpy(advantages_local_np).to(accelerator.device, dtype=torch.float32)  # (N,)
        advantages = adv_vec.unsqueeze(1).expand(-1, steps)  # (N, steps)

        # 稀疏时序（保持为列表，后续按子批按步拼 batched SparseTensor）
        latents_seq_list = [s["latents_seq"] for s in all_samples]  # 长度 N, 每项长度 steps+1

        # 可选：时间步矩阵（当前不用于前向，便于形状对齐与重组接口一致）
        t_seq = all_samples[0]["t_seq"]
        timesteps = torch.as_tensor(t_seq[:-1], device=accelerator.device, dtype=torch.float32).unsqueeze(0).expand(len(all_samples), -1)  # (N, steps)

        # 仅统计优势非零比例，不做样本丢弃（让零优势样本以零权重参与，稳定批次大小）
        num_batches_epoch = int(config.sample.num_batches_per_epoch)  # 标量
        mask = (advantages.abs().sum(dim=1) != 0)  # (N,)
        valid_samples_ratio = float(mask.sum().item() / max(1, advantages.shape[0])) if advantages.shape[0] > 0 else 0.0

        # 保持原始样本顺序，仅封装为字典以统一接口
        total_batch_size = old_log_probs.shape[0]
        pos_cond_batched = {"cond": pos_cond_batched}
        neg_cond_batched = {"neg_cond": neg_cond_batched}

        # 保持时间维顺序，不做打乱

        # Step-3: 自适应切分为子批（最后一个子批可小；当总样本少于子批数时，丢弃 0 大小子批）
        nb = max(1, num_batches_epoch)  # 标量
        base = total_batch_size // nb  # 标量
        rem = total_batch_size % nb  # 标量
        split_sizes = [base + 1 if i < rem else base for i in range(nb)]  # 长度 nb
        split_sizes = [int(s) for s in split_sizes if s > 0]  # 过滤掉0

        samples_batched = []
        offset = 0  # 标量
        for sz in split_sizes:
            start = offset  # 标量
            end = offset + sz  # 标量
            sub_dict = {
                "positive_image_cond": {k: v[start:end] for k, v in pos_cond_batched.items()},  # (sz, P, C)
                "negative_image_cond": {k: v[start:end] for k, v in neg_cond_batched.items()},  # (sz, P, C)
                "old_log_probs": old_log_probs[start:end],  # (sz, T)
                "advantages": advantages[start:end],  # (sz, T)
                "timesteps": timesteps[start:end],  # (sz, T)
                # 稀疏时序使用列表切片
                "latents_seq": latents_seq_list[start:end],  # 长度 sz
            }
            samples_batched.append(sub_dict)
            offset = end  # 标量

        run_logger.log_sampling_stats(
            epoch=epoch,
            actual_batch_size=(split_sizes[0] if len(split_sizes) > 0 else 0),
            num_sub_batches=len(samples_batched),
            valid_ratio=float(valid_samples_ratio),
        )

        # ===== 训练阶段：外层 inner-epoch × 内层子批循环 × 时间步循环（对齐 Hunyuan3D） =====
        accelerator.print(f"[Epoch {epoch}] Training...")
        slat_model.train()

        global_step = 0
        # 按 epoch 聚合训练指标（已在 epoch 开始处创建）
        steps_to_train = max(1, int(float(config.train.timestep_fraction) * steps))  # 标量
        train_step_indices = np.linspace(0, steps - 1, steps_to_train, dtype=int)  # 形状 (steps_to_train,)
        autocast_ctx = accelerator.autocast

        num_inner_epochs = int(config.train.num_inner_epochs)  # 标量
        for inner_epoch in range(num_inner_epochs):
            # 内层 epoch 不再打乱子批顺序，保持稳定顺序
            samples_batched_shuffled = samples_batched

            for batch_idx_sub, sample in enumerate(tqdm(samples_batched_shuffled, disable=not accelerator.is_main_process)):
                # 将 batched 条件拆成 per-sample 列表，与稀疏时序一一对应
                B_sub = sample["old_log_probs"].shape[0]  # 标量
                image_conds_list = []  # 长度 B_sub
                for bi in range(B_sub):
                    image_conds_list.append({
                        "cond": sample["positive_image_cond"]["cond"][bi:bi+1],  # 形状 (1, P, C)
                        "neg_cond": sample["negative_image_cond"]["neg_cond"][bi:bi+1],  # 形状 (1, P, C)
                    })

                # 构造与 compute_log_prob_trellis_stage2_batched 接口一致的样本列表（仅包含稀疏时序）
                batch_samples_list = [
                    {"latents_seq": sample["latents_seq"][bi], "t_seq": t_seq} for bi in range(B_sub)
                ]  # 长度 B_sub

                for j in train_step_indices:
                    j = int(j)  # 标量
                    with accelerator.accumulate(slat_model):
                        with autocast_ctx():
                            # SD3 风格：直接小批量前向，通过配置控制显存而非微分批
                            log_prob_vec, kl_vec = compute_log_prob_trellis_stage2_batched(
                                pipeline=pipeline,
                                samples=batch_samples_list,
                                j=j,
                                image_conds_list=image_conds_list,
                                config=ml_collections.FrozenConfigDict({
                                    "guidance_scale": float(config.sample.guidance_scale),
                                    "num_inference_steps": int(config.sample.num_steps),
                                    "sigma_min": float(config.slat_sampler_params.sigma_min),
                                    "rescale_t": float(config.slat_sampler_params.rescale_t),
                                    "deterministic": False,
                                    "kl_reward": float(config.sample.kl_reward),
                                }),
                            )  # log_prob_vec: (B_sub,), kl_vec: (B_sub,)

                            # GRPO 计算（全量子批）
                            adv_vec = torch.clamp(sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max)  # 形状 (B_sub,)
                            old_lp_vec = sample["old_log_probs"][:, j]  # 形状 (B_sub,)
                            ratio_vec = torch.exp(log_prob_vec - old_lp_vec)  # 形状 (B_sub,)
                            unclipped = -adv_vec * ratio_vec  # 形状 (B_sub,)
                            # 非对称裁剪：使用 clip_range_low / clip_range_high
                            clipped = -adv_vec * torch.clamp(
                                ratio_vec,
                                1.0 - float(config.train.clip_range_low),
                                1.0 + float(config.train.clip_range_high),
                            )  # 形状 (B_sub,)
                            policy_loss_vec = torch.maximum(unclipped, clipped)  # 形状 (B_sub,)
                            loss_vec = policy_loss_vec  # 形状 (B_sub,)
                            if float(config.train.beta) > 0.0:
                                loss_vec = loss_vec + float(config.train.beta) * kl_vec  # 形状 (B_sub,)
                            loss = loss_vec.mean()  # 标量 ()

                            accelerator.backward(loss)

                    # 仅在同步梯度步执行优化器 step/zero_grad（与梯度累积严格对齐）
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            slat_model.parameters(), config.train.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    global_step += 1  # 标量
                    if bool(config.train.ema) and ema is not None:
                        ema.step([p for p in slat_model.parameters() if p.requires_grad], global_step)

                    # PPO 关键诊断指标
                    # 这里用 log_prob 差值近似 KL：approx_kl = mean((log_prob - old_log_prob)^2 / 2)
                    delta_vec = (log_prob_vec - old_lp_vec)  # 形状 (B_sub,)
                    approx_kl = 0.5 * torch.mean(delta_vec * delta_vec)  # 标量
                    # 非对称 clipfrac：分别统计超出上下界的比例
                    lower_bound = 1.0 - float(config.train.clip_range_low)
                    upper_bound = 1.0 + float(config.train.clip_range_high)
                    clipfrac_low = torch.mean((ratio_vec < lower_bound).float())   # 标量
                    clipfrac_high = torch.mean((ratio_vec > upper_bound).float())  # 标量
                    policy_loss = policy_loss_vec.mean()  # 标量

                    # 累积本 epoch 的训练统计（含 PPO 诊断指标）
                    epoch_logger.update(loss, kl_vec, adv_vec, ratio_vec, batch_size=ratio_vec.shape[0])
                    # 传入当前子批样本数用于 clipfrac 的计数加权
                    epoch_logger.update_ppo_metrics(approx_kl, clipfrac_low, clipfrac_high, policy_loss, batch_size=ratio_vec.shape[0])

        # 本 epoch 结束：按调度记录一次到 W&B（步数用 epoch）
        if epoch_logger.num_steps > 0 and (epoch + 1) % max(1, schedule.log_every_epochs) == 0:
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
                all_rewards_np = eval_trellis(pipeline, eval_loader, config, accelerator, epoch, mesh_scorer, generator=gen)
                ema.copy_temp_to(trainable)
            else:
                all_rewards_np = eval_trellis(pipeline, eval_loader, config, accelerator, epoch, mesh_scorer, generator=gen)
            accelerator.wait_for_everyone()
            run_logger.log_eval_rewards(epoch, all_rewards_np)

        # 保存节奏对齐：每 epoch 末保存（频率由调度控制）
        if (epoch + 1) % int(schedule.save_every_epochs) == 0 and accelerator.is_main_process:
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
                )
                run_logger.log_mesh_previews(epoch, preview_files, viz.image_paths)

            if viz.camera_pairs_best is not None and len(viz.camera_pairs_best) > 0:
                run_logger.log_normal_pairs(epoch, viz.camera_pairs_best, prefix="normal_similarity/best", max_pairs=4)
            if viz.camera_pairs_worst is not None and len(viz.camera_pairs_worst) > 0:
                run_logger.log_normal_pairs(epoch, viz.camera_pairs_worst, prefix="normal_similarity/worst", max_pairs=4)


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
        run_name_dir = config.run_name if isinstance(config.run_name, str) and len(config.run_name) > 0 else "trellis_stage2"
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
    camera_pairs_best: Optional[list] = None
    camera_pairs_worst: Optional[list] = None

    def update_from_batch(self, meshes, image_paths, rewards, camera_pairs_best, camera_pairs_worst=None):
        self.meshes = meshes
        self.image_paths = image_paths
        self.rewards = rewards
        if camera_pairs_best is not None and len(camera_pairs_best) > 0:
            self.camera_pairs_best = camera_pairs_best
        if camera_pairs_worst is not None and len(camera_pairs_worst) > 0:
            self.camera_pairs_worst = camera_pairs_worst


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

    def log_normal_pairs(self, epoch: int, pairs: list, prefix: str = "normal_similarity", max_pairs: int = 4):
        log_normal_similarity_pairs(self.accelerator, pairs, step=epoch + 1, prefix=prefix, max_pairs=max_pairs)


class CheckpointSaver:
    def __init__(self, accelerator: Accelerator, dirs: RunDirs):
        self.accelerator = accelerator
        self.dirs = dirs

    def save_epoch(self, epoch: int, slat_model: nn.Module, optimizer: optim.Optimizer, config: ml_collections.ConfigDict, ema: Optional[Any] = None, use_lora: bool = False):
        save_dir = self.dirs.ckpt_dir / f"ckpt_{epoch+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        if use_lora:
            unwrapped = self.accelerator.unwrap_model(slat_model)
            lora_dir = save_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(str(lora_dir))
            torch.save(optimizer.state_dict(), str(save_dir / "optimizer.bin"))
            torch.save(dict(config), str(save_dir / "config.bin"))
            self.accelerator.print(f"💾 Saved LoRA adapter: {str(lora_dir)}")
        else:
            to_save: Dict[str, Any] = {
                "model": self.accelerator.get_state_dict(slat_model),
                "optimizer": optimizer.state_dict(),
                "config": dict(config),
            }
            if ema is not None:
                to_save["ema_state"] = ema.state_dict()
            torch.save(to_save, str(save_dir / "pytorch_model.bin"))
            self.accelerator.print(f"💾 Saved: {str(save_dir)}")


if __name__ == "__main__":
    app.run(main) 