#!/usr/bin/env python3
"""
TRELLIS Stage 2 Pipeline with Log Probability for GRPO Training

基于 flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py 的模式，
适配 TRELLIS 的两阶段架构和 SparseTensor 数据结构。

核心设计：
- Stage 1 (稀疏结构): 预训练权重冻结，在线推理生成坐标
- Stage 2 (SLAT生成): 使用 GRPO 进行强化学习训练
- SparseTensor LogProb: 适配 coords + feats 的稀疏张量格式
- CFG 处理: 支持正负条件的稀疏张量拼接/分离

参考路径:
- TRELLIS官方: `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py`
- Hunyuan3D GRPO: `flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py`
- LogProb计算: `scripts/train_hunyuan3d.py:181-232`
- SD3 Pipeline (对等实现): `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:12-462`
  - Denoising loop: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:294-352`
  - Guidance merge: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`
  - sde_step_with_logprob 调用: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:341-347`
- SD3 SDE/LogProb (单步对等): `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:17-80`
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import time

import torch
import torch.nn as nn
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

# 导入 TRELLIS 内置门面
from generators.trellis import sparse as sp

# 导入项目模块
from generators.trellis.pipeline import TrellisStage2Pipeline
# 训练路径不依赖兼容工具，解码时使用 pipeline.core_pipeline.decode_slat
from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat
from generators.trellis.utils.compat import convert_trellis_to_trimesh

 

@torch.no_grad()
def trellis_stage1_with_logprob(
    pipeline: TrellisStage2Pipeline,
    *,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    sparse_structure_sampler_params: Optional[Dict] = None,
    stage1_cond_dict: Optional[Dict[str, torch.Tensor]] = None,
    num_candidates: int = 1,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Stage1 显式接口：返回 coords_list、每步结构 logprob（占位）与 t_seq。

    - coords_list: 长度 BK 的 List[Tensor(N_i,4)]（若 K>1，复用同一 coords 重复 K 次）
    - log_prob_seq_sparse: 长度 steps 的列表，每项形状 (BK,)。此处无结构流的逐步 logprob 实现，返回全 0 占位。
    - t_seq: 形状 (steps+1,)
    """
    assert stage1_cond_dict is not None and 'cond' in stage1_cond_dict and 'neg_cond' in stage1_cond_dict
    device = pipeline.device  # 标量设备
    B = int(stage1_cond_dict['cond'].shape[0])  # 标量
    K = int(num_candidates)  # 标量

    stage1_params = sparse_structure_sampler_params or {}

    coords_list_per_image: List[torch.Tensor] = []  # 长度 B
    for i in range(B):
        cond_patches = stage1_cond_dict['cond'][i:i+1]  # 形状: (1,P,C)
        neg_patches = stage1_cond_dict['neg_cond'][i:i+1]  # 形状: (1,P,C)
        coords = pipeline.forward_stage1(image_cond={'cond': cond_patches, 'neg_cond': neg_patches}, **stage1_params)  # 形状: (N_i,4)
        coords_list_per_image.append(coords)

    # 展开到 BK
    coords_list: List[torch.Tensor] = []  # 长度 BK
    for i in range(B):
        for _ in range(K):
            coords_list.append(coords_list_per_image[i])  # 形状: (N_i,4)

    # 生成 t_seq（与 SLAT 共用 rescale_t 时间表）
    steps = int(num_inference_steps)  # 标量
    rescale_t = float((sparse_structure_sampler_params or {}).get('rescale_t', 1.0))  # 标量
    t_seq_np = np.linspace(1.0, 0.0, steps + 1) * 1000.0  # 形状: (steps+1,)
    t_seq_np = rescale_t * t_seq_np / (1.0 + (rescale_t - 1.0) * t_seq_np / 1000.0)  # 形状: (steps+1,)
    t_seq = torch.as_tensor(t_seq_np, device=device, dtype=torch.float32)  # 形状: (steps+1,)

    # 占位结构 logprob（无结构流逐步 logprob 实现时返回 0）
    BK = B * K  # 标量
    log_prob_seq_sparse: List[torch.Tensor] = [torch.zeros((BK,), device=device, dtype=torch.float32) for _ in range(steps)]  # 长度 steps

    return coords_list, log_prob_seq_sparse, t_seq



@contextmanager
def gpu_timer(name, enabled: bool = False):
    """GPU计时器，用于性能监控（受 verbose 控制）"""
    start_time = time.time()
    if enabled:
        print(f"🕐 开始: {name}")
    torch.cuda.synchronize()
    start_cuda = time.time()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    if enabled:
        print(f"✅ 完成: {name} - 耗时: {end_time - start_cuda:.2f}秒 (总时间: {end_time - start_time:.2f}秒)")


@torch.no_grad()
def trellis_stage2_with_logprob(
    pipeline: TrellisStage2Pipeline,
    image_conds: Dict[str, torch.Tensor] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    generator: Optional[torch.Generator] = None,
    output_type: str = "trimesh",
    kl_reward: float = 0.0,
    deterministic: bool = False,
    sparse_structure_sampler_params: Optional[Dict] = None,
    slat_sampler_params: Optional[Dict] = None,
    stage1_cond_dict: Optional[Dict[str, torch.Tensor]] = None,
    num_candidates: int = 1,
    verbose: bool = False,
    coords_list: Optional[List[torch.Tensor]] = None,
    **kwargs
) -> Tuple[List, List, List, torch.Tensor]:
    """
    使用官方风格的 patch 级 cond/neg_cond:
    - Stage 1: 从 stage1_cond_dict 取 {'cond': (B,P,C), 'neg_cond': (B,P,C)}
    - Stage 2: 同样使用每样本的 patch 级 cond/neg_cond，CFG 通过分别推理再线性合并
    """
    device = pipeline.device
    dtype = pipeline.dtype
    # 分布式 rank（用于打印定位）
    is_dist = (torch.distributed.is_available() and torch.distributed.is_initialized())
    rank = torch.distributed.get_rank() if is_dist else 0  # 标量

    assert stage1_cond_dict is not None, "必须提供 stage1_cond_dict（来自 pipeline.get_cond）"
    assert 'cond' in stage1_cond_dict and 'neg_cond' in stage1_cond_dict
    batch_size = stage1_cond_dict['cond'].shape[0]

    do_classifier_free_guidance = guidance_scale > 1.0

    if verbose:
        print(f"🎯 TRELLIS Stage 2 推理开始: batch_size={batch_size}, CFG={do_classifier_free_guidance}, guidance_scale={guidance_scale}")

    # 要求外部提供 coords_list（不在此处生成）
    assert coords_list is not None and isinstance(coords_list, list) and len(coords_list) > 0, "必须提供非空 coords_list"
    expected_BK = int(batch_size) * int(num_candidates)
    if len(coords_list) == batch_size:
        coords_list = [c for i, c in enumerate(coords_list) for _ in range(int(num_candidates))]
    assert len(coords_list) == expected_BK, f"coords_list 长度应为 {expected_BK}，当前为 {len(coords_list)}"

    # ===========================================
    # Stage 2: SLAT Flow 采样 + LogProb 计算
    # ===========================================
    with gpu_timer("[GRPO][Sample] Stage2 SLAT 生成+LogProb", enabled=verbose):
        stage2_params = slat_sampler_params or {}
        stage2_params.update({
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'generator': generator,
            'deterministic': deterministic,
        })

        all_latents: List[sp.SparseTensor] = []     # 按 sample 展开的稀疏序列，长度 B*k*(steps+1)
        all_log_probs: List[torch.Tensor] = []      # 按 sample 展开的每步对数概率，长度 B*k*steps
        all_kl: List[torch.Tensor] = []             # 按 sample 展开的每步KL，长度 B*k*steps（不再作为返回值）
        final_slats: List[sp.SparseTensor] = []     # 最终每个候选的 SLAT，长度 B*k

        from .trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
        slat_flow_model = pipeline.get_trainable_model()
        base_model = slat_flow_model.module if hasattr(slat_flow_model, "module") else slat_flow_model
        in_channels = int(base_model.in_channels)  # 标量

        # —— 并行修复 ——
        # 将每个样本的 coords 复制 num_candidates 份，拼成 batched SparseTensor 一次性采样
        batched_noises: List[sp.SparseTensor] = []   # 合并前单样本 SparseTensor 列表
        batched_pos_conds: List[torch.Tensor] = []   # 合并前的 (1, P, C) 条件列表
        batched_neg_conds: List[torch.Tensor] = []   # 合并前的 (1, P, C) 负条件列表（CFG时）

        for i in range(batch_size):
            coords = coords_list[i]
            cond_patches = stage1_cond_dict['cond'][i:i+1]  # (1, P, C)
            neg_patches = stage1_cond_dict['neg_cond'][i:i+1]  # (1, P, C)
            for _ in range(int(num_candidates)):
                if generator is None:
                    noise_feats = torch.randn(coords.shape[0], in_channels, device=device, dtype=dtype)  # 形状 (N_i, C)
                else:
                    noise_feats = torch.randn((coords.shape[0], in_channels), device=device, dtype=dtype, generator=generator)  # 形状 (N_i, C)
                initial_noise = sp.SparseTensor(coords=coords, feats=noise_feats)
                batched_noises.append(initial_noise)
                batched_pos_conds.append(cond_patches)
                if do_classifier_free_guidance:
                    batched_neg_conds.append(neg_patches)

        # 拼接为 batched SparseTensor 与 batched cond
        batched_noise = sparse_tensor_cat(batched_noises)  # 形状: coords(sum(N_i*rep),4), feats(sum(N_i*rep),C)，batch 维为 B*num_candidates
        Bk = len(batched_pos_conds)
        pos_cond_batched = torch.cat(batched_pos_conds, dim=0)  # 形状 (B*k, P, C)
        neg_cond_batched = torch.cat(batched_neg_conds, dim=0) if do_classifier_free_guidance else None  # 形状 (B*k, P, C)
        # 快速验证：统计每个批的点数并打印峰值与总量（受 verbose 控制）
        counts_per_batch = torch.bincount(batched_noise.coords[:, 0].to(torch.long), minlength=Bk)  # 形状 (Bk,)
        total_points = int(batched_noise.coords.shape[0])  # 标量
        max_points = int(counts_per_batch.max().item()) if counts_per_batch.numel() > 0 else 0  # 标量
        channels = int(batched_noise.feats.shape[1])  # 标量
        if verbose:
            print(f"[Rank {rank}] Batched Sparse (B*k={Bk}) total_N={total_points}, max_N_per_sample={max_points}, C={channels}")

        final_slat_batched, sample_latents_flat, sample_log_probs_flat, sample_kl_flat = trellis_flow_euler_sampler_with_logprob(
            model=slat_flow_model,
            noise=batched_noise,
            cond=pos_cond_batched,
            steps=num_inference_steps,
            sigma_min=stage2_params.get('sigma_min', 0.002),
            rescale_t=stage2_params.get('rescale_t', 1.0),
            generator=generator,
            deterministic=deterministic,
            guidance_scale=guidance_scale,
            neg_cond=neg_cond_batched,
            kl_reward=kl_reward,
            verbose=False,
        )

        # 将 batched 输出按样本拆分并填充至列表
        # 注意：解码到网格时，spconv 在巨大批量上会触发 int32 上限断言。
        # 因此此处按 batch 维将 SparseTensor 拆分为单样本 SLAT，逐个解码，避免激活点过大。
        def split_batched_sparse(sparse_tensor: sp.SparseTensor, batch_count: int) -> List[sp.SparseTensor]:
            coords = sparse_tensor.coords  # 形状 (N, 4)
            feats = sparse_tensor.feats   # 形状 (N, C)
            slats: List[sp.SparseTensor] = []
            for b in range(int(batch_count)):
                mask = (coords[:, 0] == b)  # 形状 (N,)
                coords_b = coords[mask].clone()  # 形状 (N_b, 4)
                coords_b[:, 0] = 0  # 重置为单样本 batch 索引
                feats_b = feats[mask]  # 形状 (N_b, C)
                slats.append(sp.SparseTensor(coords=coords_b, feats=feats_b))
            return slats

        # 按 Bk 拆分
        final_slats_per_sample = split_batched_sparse(final_slat_batched, Bk)  # 长度 Bk，每项 feats 形状 (N_i, C)
        final_slats.extend(final_slats_per_sample)
        all_latents.extend(sample_latents_flat)
        all_log_probs.extend(sample_log_probs_flat)
        all_kl.extend(sample_kl_flat)

        if verbose:
            print(f"🎯 Stage 2 完成: 生成了 {len(final_slats_per_sample)} 个 SLAT")

    # ===========================================
    # 输出处理 + 返回时间表 t_seq（对齐 Direct3D-S2）
    # ===========================================
    if output_type == "latent":
        meshes = final_slats
    elif output_type == "kiui":
        with gpu_timer("[GRPO][Sample] 解码为 KiuiMesh", enabled=verbose):
            from kiui.mesh import Mesh as KiuiMesh
            meshes = []
            def _fallback_kiui(device: torch.device) -> KiuiMesh:
                # 以极小三角形替代空网格，避免渲染器报错
                v_fb = torch.tensor([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]], dtype=torch.float32, device=device)
                f_fb = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
                return KiuiMesh(v=v_fb, f=f_fb, device=device)
            for slat in final_slats:
                decoded = pipeline.core_pipeline.decode_slat(slat, formats=['mesh'])
                mesh_data = decoded['mesh']
                mesh_list = mesh_data if isinstance(mesh_data, list) else [mesh_data]
                for m in mesh_list:
                    if isinstance(m, KiuiMesh):
                        v = m.v
                        f = m.f
                        if isinstance(v, torch.Tensor) and isinstance(f, torch.Tensor) and (v.numel() == 0 or f.numel() == 0):
                            meshes.append(_fallback_kiui(device=pipeline.device))
                        else:
                            meshes.append(m)
                    elif isinstance(m, trimesh.Trimesh):
                        v = m.vertices
                        f = m.faces
                        if torch.is_tensor(v):
                            v = v.detach().float()
                        else:
                            v = torch.tensor(v, dtype=torch.float32)
                        if torch.is_tensor(f):
                            f = f.detach().int()
                        else:
                            f = torch.tensor(f, dtype=torch.int32)
                        if v.numel() == 0 or f.numel() == 0:
                            meshes.append(_fallback_kiui(device=pipeline.device))
                        else:
                            meshes.append(KiuiMesh(v=v, f=f, device=v.device))
                    else:
                        # 兼容具有 vertices/faces 属性的自定义网格表示
                        v_attr = getattr(m, 'vertices', None)
                        f_attr = getattr(m, 'faces', None)
                        if v_attr is not None and f_attr is not None:
                            v = v_attr.detach().float() if torch.is_tensor(v_attr) else torch.tensor(v_attr, dtype=torch.float32)
                            f = f_attr.detach().int() if torch.is_tensor(f_attr) else torch.tensor(f_attr, dtype=torch.int32)
                            if v.numel() == 0 or f.numel() == 0:
                                meshes.append(_fallback_kiui(device=pipeline.device))
                            else:
                                meshes.append(KiuiMesh(v=v, f=f, device=v.device))
                        else:
                            # 退化为最小三角形，保证训练流程可继续
                            meshes.append(_fallback_kiui(device=pipeline.device))
            if verbose:
                print(f"🏆 KiuiMesh 解码完成: 生成了 {len(meshes)} 个 mesh")
    else:
        with gpu_timer("[GRPO][Sample] 解码为 Mesh", enabled=verbose):
            meshes = []
            for slat in final_slats:
                decoded = pipeline.core_pipeline.decode_slat(slat, formats=['mesh'])
                mesh_list = convert_trellis_to_trimesh(decoded)
                meshes.extend(mesh_list)
            if verbose:
                print(f"🏆 网格解码完成: 生成了 {len(meshes)} 个 mesh")

    # print(f"✅ TRELLIS Stage 2 管道完成:")
    # print(f"   - 输出类型: {output_type}")
    # print(f"   - 每图候选数: {int(num_candidates)}")
    # print(f"   - 总 latents: {len(all_latents)}")
    # print(f"   - 总 log_probs: {len(all_log_probs)}")
    # print(f"   - 总 KL: {len(all_kl)}")

    # 统一使用 Stage2 的时间表（rescale_t）
    steps = int(num_inference_steps)  # 标量
    rescale_t = float((slat_sampler_params or {}).get('rescale_t', 1.0))  # 标量
    t_seq_np = np.linspace(1.0, 0.0, steps + 1) * 1000.0  # 形状: (steps+1,)
    t_seq_np = rescale_t * t_seq_np / (1.0 + (rescale_t - 1.0) * t_seq_np / 1000.0)  # 形状: (steps+1,)
    t_seq = torch.as_tensor(t_seq_np, device=pipeline.device, dtype=torch.float32)  # 形状: (steps+1,)

    return meshes, all_latents, all_log_probs, t_seq


def decode_slat_to_mesh(
    pipeline: TrellisStage2Pipeline, 
    slat: sp.SparseTensor,
    **decode_params
) -> List[trimesh.Trimesh]:
    """
    将 SLAT 解码为 mesh 格式
    
    参考:
    - TRELLIS: `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py:200-217` (decode_slat)
    - SD3: 无直接对等（SD3 为 VAE 图像解码），可参考 `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:435-439`
    """
    # 调用工具函数进行转换
    return convert_trellis_to_trimesh([slat], **decode_params)


def sparse_tensor_cfg_cat(
    positive_sparse: sp.SparseTensor,
    negative_sparse: sp.SparseTensor
) -> sp.SparseTensor:
    """
    SparseTensor 的 CFG 拼接操作，用于分类器引导
    
    参考:
    - TRELLIS 稀疏拼接: `generators/trellis/patches/sparse_tensor_utils.py`
    - SD3 对应逻辑: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:315-318`
      (noise_pred_uncond + w * (noise_pred_text - noise_pred_uncond))
    
    返回: negative 在前，positive 在后
    """
    return sparse_tensor_cat([negative_sparse, positive_sparse])


def sparse_tensor_cfg_split(
    cfg_sparse: sp.SparseTensor,
    batch_size: int
) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
    """
    SparseTensor 的 CFG 分离操作，用于分类器引导后处理
    
    说明:
    - SD3 中通过张量 chunk 拆分正负分支（参见 `sd3_pipeline_with_logprob.py:315-318` 处的两分支合并），
      此处在 SparseTensor 上以坐标/特征切分实现等价效果
    """
    # 这里需要实现 SparseTensor 的分块操作
    # 暂时使用简单的特征分割，后续需要完善
    total_points = cfg_sparse.coords.shape[0]
    half_points = total_points // 2
    
    # 分离坐标和特征
    neg_coords = cfg_sparse.coords[:half_points]  # shape: (N/2, 4)
    pos_coords = cfg_sparse.coords[half_points:]  # shape: (N/2, 4)
    
    neg_feats = cfg_sparse.feats[:half_points]    # shape: (N/2, C)
    pos_feats = cfg_sparse.feats[half_points:]    # shape: (N/2, C)
    
    # 调整坐标的批次索引
    pos_coords = pos_coords.clone()
    pos_coords[:, 0] -= batch_size  # 将 batch 索引从 [batch_size, 2*batch_size) 调整到 [0, batch_size)
    
    negative_sparse = sp.SparseTensor(coords=neg_coords, feats=neg_feats)
    positive_sparse = sp.SparseTensor(coords=pos_coords, feats=pos_feats)
    
    return negative_sparse, positive_sparse 


 


class TrellisPipelineWithLogProb:
    """Trellis 最小 GRPO 包装（类式接口）。"""

    def __init__(self, ref_pipeline: TrellisStage2Pipeline):
        self.ref = ref_pipeline
        self.device = self.ref.device  # 形状: 标量设备
        try:
            self.dtype = self.ref.dtype  # 形状: 标量 dtype
        except Exception:
            self.dtype = torch.float32  # 形状: 标量 dtype

    # --- 构建/设备迁移 ---
    @classmethod
    def from_pretrained(cls, model_path: str, verbose: bool = False) -> "TrellisPipelineWithLogProb":
        ref = TrellisStage2Pipeline(model_path=model_path, verbose=bool(verbose))
        return cls(ref)

    def to(self, device: Union[str, torch.device]) -> None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device  # 形状: 标量设备
        self.ref.to(dev)  # 形状: ()
        self.device = self.ref.device  # 形状: 标量设备
        try:
            self.dtype = self.ref.dtype  # 形状: 标量 dtype
        except Exception:
            pass

    # --- Direct3D 风格 API ---
    def prepare_image_conditions(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cond, neg_cond = self.ref.prepare_image_conditions(images)  # 形状: (B,P,C), (B,P,C)
        return cond, neg_cond

    # 仅保留基础接口：prepare_image_conditions（返回二元组）

    def get_trainable_model(self) -> nn.Module:
        return self.ref.get_trainable_model()

    def forward_stage1(self, image_cond: Dict[str, torch.Tensor], **sampler_params) -> torch.Tensor:
        return self.ref.forward_stage1(image_cond=image_cond, **sampler_params)  # 形状: (N,4)

    # --- 计时器（静态） ---
    @staticmethod
    @contextmanager
    def _gpu_timer(name: str, enabled: bool = False):
        start_time = time.time()
        if enabled:
            print(f"🕐 开始: {name}")
        torch.cuda.synchronize()
        start_cuda = time.time()
        yield
        torch.cuda.synchronize()
        end_time = time.time()
        if enabled:
            print(f"✅ 完成: {name} - 耗时: {end_time - start_cuda:.2f}秒 (总时间: {end_time - start_time:.2f}秒)")

    # --- Stage1 with logprob（占位 logprob） ---
    @torch.no_grad()
    def stage1_with_logprob(
        self,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        sparse_structure_sampler_params: Optional[Dict] = None,
        stage1_cond_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_candidates: int = 1,
        verbose: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        assert stage1_cond_dict is not None and 'cond' in stage1_cond_dict and 'neg_cond' in stage1_cond_dict
        device = self.device  # 形状: 标量设备
        B = int(stage1_cond_dict['cond'].shape[0])  # 形状: 标量
        K = int(num_candidates)  # 形状: 标量

        stage1_params = sparse_structure_sampler_params or {}

        coords_list_per_image: List[torch.Tensor] = []  # 长度 B
        for i in range(B):
            cond_patches = stage1_cond_dict['cond'][i:i+1]  # 形状: (1,P,C)
            neg_patches = stage1_cond_dict['neg_cond'][i:i+1]  # 形状: (1,P,C)
            coords = self.forward_stage1(image_cond={'cond': cond_patches, 'neg_cond': neg_patches}, **stage1_params)  # 形状: (N_i,4)
            coords_list_per_image.append(coords)

        # 展开到 BK
        coords_list: List[torch.Tensor] = []  # 长度 BK
        for i in range(B):
            for _ in range(K):
                coords_list.append(coords_list_per_image[i])  # 形状: (N_i,4)

        # 生成 t_seq（与 SLAT 共用 rescale_t 时间表）
        steps = int(num_inference_steps)  # 形状: 标量
        rescale_t = float((sparse_structure_sampler_params or {}).get('rescale_t', 1.0))  # 形状: 标量
        t_seq_np = np.linspace(1.0, 0.0, steps + 1) * 1000.0  # 形状: (steps+1,)
        t_seq_np = rescale_t * t_seq_np / (1.0 + (rescale_t - 1.0) * t_seq_np / 1000.0)  # 形状: (steps+1,)
        t_seq = torch.as_tensor(t_seq_np, device=device, dtype=torch.float32)  # 形状: (steps+1,)

        # 占位结构 logprob（无结构流逐步 logprob 实现时返回 0）
        BK = B * K  # 形状: 标量
        log_prob_seq_sparse: List[torch.Tensor] = [torch.zeros((BK,), device=device, dtype=torch.float32) for _ in range(steps)]  # 长度 steps

        return coords_list, log_prob_seq_sparse, t_seq

    # --- Stage2 with logprob ---
    @torch.no_grad()
    def stage2_with_logprob(
        self,
        image_conds: Dict[str, torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "trimesh",
        kl_reward: float = 0.0,
        deterministic: bool = False,
        sparse_structure_sampler_params: Optional[Dict] = None,
        slat_sampler_params: Optional[Dict] = None,
        stage1_cond_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_candidates: int = 1,
        verbose: bool = False,
        coords_list: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[List, List, List, torch.Tensor]:
        device = self.device  # 形状: 标量设备
        dtype = self.dtype  # 形状: 标量 dtype
        is_dist = (torch.distributed.is_available() and torch.distributed.is_initialized())
        rank = torch.distributed.get_rank() if is_dist else 0  # 形状: 标量

        assert stage1_cond_dict is not None and 'cond' in stage1_cond_dict and 'neg_cond' in stage1_cond_dict
        batch_size = int(stage1_cond_dict['cond'].shape[0])  # 形状: 标量

        do_classifier_free_guidance = bool(guidance_scale > 1.0)  # 形状: 标量

        if verbose:
            print(f"🎯 TRELLIS Stage 2 推理开始: batch_size={batch_size}, CFG={do_classifier_free_guidance}, guidance_scale={guidance_scale}")

        # Stage 1: 在线结构坐标
        with self._gpu_timer("[GRPO][Sample] Stage1 稀疏结构生成", enabled=verbose):
            stage1_params = sparse_structure_sampler_params or {}
            if coords_list is None:
                coords_list_internal: List[torch.Tensor] = []  # 长度 B
                for i in range(batch_size):
                    cond_patches = stage1_cond_dict['cond'][i:i+1]  # 形状: (1,P,C)
                    neg_patches = stage1_cond_dict['neg_cond'][i:i+1]  # 形状: (1,P,C)
                    coords = self.forward_stage1(
                        image_cond={'cond': cond_patches, 'neg_cond': neg_patches},
                        **stage1_params
                    )  # 形状: (N_i,4)
                    coords_list_internal.append(coords)
                coords_list = []  # 长度 BK
                for i in range(batch_size):
                    for _ in range(int(num_candidates)):
                        coords_list.append(coords_list_internal[i])
            else:
                expected_BK = int(batch_size) * int(num_candidates)  # 形状: 标量
                if len(coords_list) == batch_size:
                    coords_list = [c for i, c in enumerate(coords_list) for _ in range(int(num_candidates))]
                assert len(coords_list) == expected_BK, f"coords_list 长度应为 {expected_BK}，当前为 {len(coords_list)}"
            if verbose:
                print(f"🏗️  Stage 1 完成: 生成/接收了 {len(coords_list)} 个候选的稀疏结构")

        # Stage 2: SLAT 生成 + LogProb
        with self._gpu_timer("[GRPO][Sample] Stage2 SLAT 生成+LogProb", enabled=verbose):
            stage2_params = slat_sampler_params or {}
            stage2_params.update({
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'generator': generator,
                'deterministic': deterministic,
            })

            all_latents: List[sp.SparseTensor] = []     # 形状: 列表(len=B*k*(steps+1))
            all_log_probs: List[torch.Tensor] = []      # 形状: 列表(len=B*k*steps), 每项 (,) 标量
            all_kl: List[torch.Tensor] = []             # 形状: 列表(len=B*k*steps)
            final_slats: List[sp.SparseTensor] = []     # 形状: 列表(len=B*k)

            from .trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
            slat_flow_model = self.get_trainable_model()  # 形状: 模型
            base_model = slat_flow_model.module if hasattr(slat_flow_model, "module") else slat_flow_model  # 形状: 模型
            in_channels = int(base_model.in_channels)  # 形状: 标量

            # 拼 batched 初始噪声与条件
            batched_noises: List[sp.SparseTensor] = []  # 形状: 列表
            batched_pos_conds: List[torch.Tensor] = []  # 形状: 列表
            batched_neg_conds: List[torch.Tensor] = []  # 形状: 列表

            for i in range(batch_size):
                coords_i = coords_list[i]  # 形状: (N_i,4)
                cond_patches = stage1_cond_dict['cond'][i:i+1]  # 形状: (1,P,C)
                neg_patches = stage1_cond_dict['neg_cond'][i:i+1]  # 形状: (1,P,C)
                for _ in range(int(num_candidates)):
                    if generator is None:
                        noise_feats = torch.randn(coords_i.shape[0], in_channels, device=device, dtype=dtype)  # 形状: (N_i,C)
                    else:
                        noise_feats = torch.randn((coords_i.shape[0], in_channels), device=device, dtype=dtype, generator=generator)  # 形状: (N_i,C)
                    initial_noise = sp.SparseTensor(coords=coords_i, feats=noise_feats)  # 形状: 稀疏(N_i,C)
                    batched_noises.append(initial_noise)  # 形状: 列表累加
                    batched_pos_conds.append(cond_patches)  # 形状: 列表累加
                    if do_classifier_free_guidance:
                        batched_neg_conds.append(neg_patches)  # 形状: 列表累加

            batched_noise = sparse_tensor_cat(batched_noises)  # 形状: batched 稀疏(合并)
            Bk = len(batched_pos_conds)  # 形状: 标量
            pos_cond_batched = torch.cat(batched_pos_conds, dim=0)  # 形状: (B*k,P,C)
            neg_cond_batched = torch.cat(batched_neg_conds, dim=0) if do_classifier_free_guidance else None  # 形状: (B*k,P,C) 或 None
            counts_per_batch = torch.bincount(batched_noise.coords[:, 0].to(torch.long), minlength=Bk)  # 形状: (Bk,)
            total_points = int(batched_noise.coords.shape[0])  # 形状: 标量
            max_points = int(counts_per_batch.max().item()) if counts_per_batch.numel() > 0 else 0  # 形状: 标量
            channels = int(batched_noise.feats.shape[1])  # 形状: 标量
            if verbose:
                print(f"[Rank {rank}] Batched Sparse (B*k={Bk}) total_N={total_points}, max_N_per_sample={max_points}, C={channels}")

            final_slat_batched, sample_latents_flat, sample_log_probs_flat, sample_kl_flat = trellis_flow_euler_sampler_with_logprob(
                model=slat_flow_model,
                noise=batched_noise,
                cond=pos_cond_batched,
                steps=num_inference_steps,
                sigma_min=stage2_params.get('sigma_min', 0.002),
                rescale_t=stage2_params.get('rescale_t', 1.0),
                generator=generator,
                deterministic=deterministic,
                guidance_scale=guidance_scale,
                neg_cond=neg_cond_batched,
                kl_reward=kl_reward,
                verbose=False,
            )  # 形状: (稀疏, 列表, 列表, 列表)

            def split_batched_sparse(sparse_tensor: sp.SparseTensor, batch_count: int) -> List[sp.SparseTensor]:
                coords = sparse_tensor.coords  # 形状 (N,4)
                feats = sparse_tensor.feats    # 形状 (N,C)
                slats: List[sp.SparseTensor] = []
                for b in range(int(batch_count)):
                    mask = (coords[:, 0] == b)  # 形状: (N,)
                    coords_b = coords[mask].clone()  # 形状: (N_b,4)
                    coords_b[:, 0] = 0  # 形状: (N_b,4)
                    feats_b = feats[mask]  # 形状: (N_b,C)
                    slats.append(sp.SparseTensor(coords=coords_b, feats=feats_b))  # 形状: 稀疏(N_b,C)
                return slats

            final_slats_per_sample = split_batched_sparse(final_slat_batched, Bk)  # 形状: 列表(Bk)
            final_slats.extend(final_slats_per_sample)  # 形状: 列表累加
            all_latents.extend(sample_latents_flat)  # 形状: 列表累加
            all_log_probs.extend(sample_log_probs_flat)  # 形状: 列表累加
            all_kl.extend(sample_kl_flat)  # 形状: 列表累加

            if verbose:
                print(f"🎯 Stage 2 完成: 生成了 {len(final_slats_per_sample)} 个 SLAT")

        # 输出与 t_seq
        if output_type == "latent":
            meshes = final_slats  # 形状: 列表(Bk)
        elif output_type == "kiui":
            with self._gpu_timer("[GRPO][Sample] 解码为 KiuiMesh", enabled=verbose):
                from kiui.mesh import Mesh as KiuiMesh
                meshes = []
                def _fallback_kiui(device: torch.device) -> KiuiMesh:
                    v_fb = torch.tensor([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]], dtype=torch.float32, device=device)  # 形状: (3,3)
                    f_fb = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)  # 形状: (1,3)
                    return KiuiMesh(v=v_fb, f=f_fb, device=device)
                for slat in final_slats:
                    decoded = self.ref.core_pipeline.decode_slat(slat, formats=['mesh'])  # 形状: dict
                    mesh_data = decoded['mesh']  # 形状: List|KiuiMesh|trimesh
                    mesh_list = mesh_data if isinstance(mesh_data, list) else [mesh_data]  # 形状: 列表
                    for m in mesh_list:
                        if isinstance(m, KiuiMesh):
                            v = m.v  # 形状: (V,3)
                            f = m.f  # 形状: (F,3)
                            if isinstance(v, torch.Tensor) and isinstance(f, torch.Tensor) and (v.numel() == 0 or f.numel() == 0):
                                meshes.append(_fallback_kiui(device=self.device))
                            else:
                                meshes.append(m)
                        elif isinstance(m, trimesh.Trimesh):
                            v = m.vertices  # 形状: (V,3) 或 ndarray
                            f = m.faces     # 形状: (F,3) 或 ndarray
                            if torch.is_tensor(v):
                                v = v.detach().float()
                            else:
                                v = torch.tensor(v, dtype=torch.float32)
                            if torch.is_tensor(f):
                                f = f.detach().int()
                            else:
                                f = torch.tensor(f, dtype=torch.int32)
                            if v.numel() == 0 or f.numel() == 0:
                                meshes.append(_fallback_kiui(device=self.device))
                            else:
                                meshes.append(KiuiMesh(v=v, f=f, device=v.device))
                        else:
                            v_attr = getattr(m, 'vertices', None)
                            f_attr = getattr(m, 'faces', None)
                            if v_attr is not None and f_attr is not None:
                                v = v_attr.detach().float() if torch.is_tensor(v_attr) else torch.tensor(v_attr, dtype=torch.float32)
                                f = f_attr.detach().int() if torch.is_tensor(f_attr) else torch.tensor(f_attr, dtype=torch.int32)
                                if v.numel() == 0 or f.numel() == 0:
                                    meshes.append(_fallback_kiui(device=self.device))
                                else:
                                    meshes.append(KiuiMesh(v=v, f=f, device=v.device))
                            else:
                                meshes.append(_fallback_kiui(device=self.device))
                if verbose:
                    print(f"🏆 KiuiMesh 解码完成: 生成了 {len(meshes)} 个 mesh")
        else:
            with self._gpu_timer("[GRPO][Sample] 解码为 Mesh", enabled=verbose):
                meshes = []
                for slat in final_slats:
                    decoded = self.ref.core_pipeline.decode_slat(slat, formats=['mesh'])  # 形状: dict
                    mesh_list = convert_trellis_to_trimesh(decoded)  # 形状: 列表
                    meshes.extend(mesh_list)
                if verbose:
                    print(f"🏆 网格解码完成: 生成了 {len(meshes)} 个 mesh")

        steps = int(num_inference_steps)  # 形状: 标量
        rescale_t = float((slat_sampler_params or {}).get('rescale_t', 1.0))  # 形状: 标量
        t_seq_np = np.linspace(1.0, 0.0, steps + 1) * 1000.0  # 形状: (steps+1,)
        t_seq_np = rescale_t * t_seq_np / (1.0 + (rescale_t - 1.0) * t_seq_np / 1000.0)  # 形状: (steps+1,)
        t_seq = torch.as_tensor(t_seq_np, device=self.device, dtype=torch.float32)  # 形状: (steps+1,)

        return meshes, all_latents, all_log_probs, t_seq

    # --- 解码 ---
    def decode_slat_to_mesh(self, slat: sp.SparseTensor, **decode_params) -> List[trimesh.Trimesh]:
        return convert_trellis_to_trimesh([slat], **decode_params)

    # --- 稀疏 CFG 工具（静态） ---
    @staticmethod
    def sparse_tensor_cfg_cat(positive_sparse: sp.SparseTensor, negative_sparse: sp.SparseTensor) -> sp.SparseTensor:
        return sparse_tensor_cat([negative_sparse, positive_sparse])

    @staticmethod
    def sparse_tensor_cfg_split(cfg_sparse: sp.SparseTensor, batch_size: int) -> Tuple[sp.SparseTensor, sp.SparseTensor]:
        total_points = cfg_sparse.coords.shape[0]  # 形状: 标量
        half_points = total_points // 2  # 形状: 标量
        neg_coords = cfg_sparse.coords[:half_points]  # 形状: (N/2,4)
        pos_coords = cfg_sparse.coords[half_points:]  # 形状: (N/2,4)
        neg_feats = cfg_sparse.feats[:half_points]    # 形状: (N/2,C)
        pos_feats = cfg_sparse.feats[half_points:]    # 形状: (N/2,C)
        pos_coords = pos_coords.clone()  # 形状: (N/2,4)
        pos_coords[:, 0] -= batch_size  # 形状: (N/2,4)
        negative_sparse = sp.SparseTensor(coords=neg_coords, feats=neg_feats)  # 形状: 稀疏
        positive_sparse = sp.SparseTensor(coords=pos_coords, feats=pos_feats)  # 形状: 稀疏
        return negative_sparse, positive_sparse
