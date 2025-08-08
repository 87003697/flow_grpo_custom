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
import sys
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

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入 TRELLIS 相关模块
reference_path = project_root / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))
import trellis.modules.sparse as sp

# 导入项目模块
from generators.trellis.pipeline import TrellisStage2Pipeline
from generators.trellis.utils import convert_trellis_to_trimesh
from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat


@contextmanager
def gpu_timer(name):
    """GPU计时器，用于性能监控"""
    start_time = time.time()
    print(f"🕐 开始: {name}")
    torch.cuda.synchronize()
    start_cuda = time.time()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"✅ 完成: {name} - 耗时: {end_time - start_cuda:.2f}秒 (总时间: {end_time - start_time:.2f}秒)")


@torch.no_grad()
def trellis_stage2_with_logprob(
    pipeline: TrellisStage2Pipeline,
    image_conds: Dict[str, torch.Tensor],
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    generator: Optional[torch.Generator] = None,
    output_type: str = "trimesh",
    kl_reward: float = 0.0,
    deterministic: bool = False,
    sparse_structure_sampler_params: Optional[Dict] = None,
    slat_sampler_params: Optional[Dict] = None,
    **kwargs
) -> Tuple[List, List, List, List]:
    """
    TRELLIS Stage 2完整推理+LogProb计算，返回 (meshes, all_latents, all_log_probs, all_kl)
    
    这个函数遵循与 hunyuan3d_pipeline_with_logprob 相同的模式：
    - 第一个参数是 pipeline 实例
    - 使用 pipeline.device 获取设备信息
    - 返回 (output, all_latents, all_log_probs, all_kl) 格式
    
    参考:
    - Hunyuan3D: `flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py`
    - SD3 对应实现: `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:12-462`
      - Denoising loop: `294-352`，Guidance 合并: `315-318`，SDE 步调用: `341-347`
    """
    # 获取设备信息（遵循 hunyuan3d 模式）
    device = pipeline.device
    dtype = pipeline.dtype
    
    # 验证输入条件
    assert 'positive' in image_conds, "必须提供 positive 图像条件"
    positive_image_cond = image_conds['positive']  # shape: (B, C)
    batch_size = positive_image_cond.shape[0]
    
    # CFG 设置
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        assert 'negative' in image_conds, "CFG 模式下必须提供 negative 图像条件"
        negative_image_cond = image_conds['negative']  # shape: (B, C)
        assert negative_image_cond.shape[0] == batch_size, "正负条件的批量大小必须相同"
        
        # 🔧 修复: 统一为先 negative 后 positive 的顺序（与 hunyuan3d 一致）
        cond_for_generation = torch.cat([negative_image_cond, positive_image_cond], dim=0)  # shape: (2B, C)
    else:
        cond_for_generation = positive_image_cond  # shape: (B, C)
    
    print(f"🎯 TRELLIS Stage 2 推理开始: batch_size={batch_size}, CFG={do_classifier_free_guidance}, guidance_scale={guidance_scale}")
    
    # ===========================================
    # Stage 1: 在线推理生成稀疏结构坐标（冻结权重）
    # ===========================================
    with gpu_timer("Stage 1 - 稀疏结构生成"):
        # 准备 Stage 1 采样参数
        stage1_params = sparse_structure_sampler_params or {}
        
        # 🔧 Stage 1 在线推理（使用正面条件，因为结构生成不需要CFG）
        coords_list = []
        for i in range(batch_size):
            # 单个样本的图像条件 shape: (1, C)
            single_image_cond = positive_image_cond[i:i+1]
            
            # 调用 pipeline 的 Stage 1 推理
            coords = pipeline.forward_stage1(
                image_cond={'main': single_image_cond},
                **stage1_params
            )  # 返回稀疏结构坐标 shape: (N_i, 4)
            coords_list.append(coords)
            
        print(f"🏗️  Stage 1 完成: 生成了 {len(coords_list)} 个稀疏结构")
        for i, coords in enumerate(coords_list):
            print(f"   样本 {i}: 坐标数量 {coords.shape[0]}")
    
    # ===========================================  
    # Stage 2: SLAT Flow 采样 + LogProb 计算
    # ===========================================
    with gpu_timer("Stage 2 - SLAT 生成 + LogProb"):
        # 准备 Stage 2 采样参数
        stage2_params = slat_sampler_params or {}
        stage2_params.update({
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'generator': generator,
            'deterministic': deterministic,
        })
        
        # 存储返回结果
        all_latents = []  # List[sp.SparseTensor]
        all_log_probs = []  # List[torch.Tensor] 
        all_kl = []  # List[torch.Tensor]
        final_slats = []  # List[sp.SparseTensor]
        
        # 逐个样本进行 Stage 2 推理（因为每个样本的坐标结构不同）
        for i in range(batch_size):
            print(f"🔄 处理样本 {i+1}/{batch_size}")
            
            # 当前样本的坐标和条件
            coords = coords_list[i]  # shape: (N_i, 4)
            
            # 准备当前样本的图像条件
            if do_classifier_free_guidance:
                # CFG 模式：拼接负面和正面条件
                sample_image_cond = torch.cat([
                    negative_image_cond[i:i+1],  # shape: (1, C)
                    positive_image_cond[i:i+1]   # shape: (1, C)
                ], dim=0)  # shape: (2, C)
            else:
                sample_image_cond = positive_image_cond[i:i+1]  # shape: (1, C)
            
            # 🔧 使用 TRELLIS Flow Euler 采样器进行 LogProb 计算
            from .trellis_flow_with_logprob import trellis_flow_euler_sampler_with_logprob
            
            # 获取 SLatFlowModel（可训练模型）
            slat_flow_model = pipeline.get_trainable_model()
            
            # 准备初始噪声 SparseTensor
            noise_feats = torch.randn(coords.shape[0], slat_flow_model.in_channels, device=device, dtype=dtype)
            initial_noise = sp.SparseTensor(
                coords=coords,  # shape: (N_i, 4)
                feats=noise_feats  # shape: (N_i, C)
            )
            
            # 准备条件
            if do_classifier_free_guidance:
                pos_cond = {'main': positive_image_cond[i:i+1]}
                neg_cond = {'main': negative_image_cond[i:i+1]}
            else:
                pos_cond = {'main': positive_image_cond[i:i+1]}
                neg_cond = None
            
            # Flow Euler 采样 + LogProb 计算
            final_slat, sample_latents, sample_log_probs, sample_kl = trellis_flow_euler_sampler_with_logprob(
                model=slat_flow_model,
                noise=initial_noise,
                cond=pos_cond,
                neg_cond=neg_cond,
                steps=num_inference_steps,
                sigma_min=stage2_params.get('sigma_min', 0.002),
                rescale_t=stage2_params.get('rescale_t', 1.0),
                generator=generator,
                deterministic=deterministic,
                guidance_scale=guidance_scale,
                verbose=False,  # 避免重复进度条
            )
            
            # KL 奖励计算（如果需要）
            if kl_reward > 0 and not deterministic:
                print(f"   计算 KL 奖励: kl_reward={kl_reward}")
                # 重新计算参考策略的 LogProb
                # 这里可以使用 LoRA 禁用的方式，暂时简化为零
                ref_kl = [torch.zeros_like(lp) for lp in sample_log_probs]
                sample_kl = ref_kl  # 替换为实际 KL 计算
            
            # 存储结果
            final_slats.append(final_slat)
            all_latents.extend(sample_latents)  # 展平所有步骤
            all_log_probs.extend(sample_log_probs)
            all_kl.extend(sample_kl)
        
        print(f"🎯 Stage 2 完成: 生成了 {len(final_slats)} 个 SLAT")
    
    # ===========================================
    # 输出处理：SLAT 解码为 Mesh
    # ===========================================
    if output_type == "latent":
        meshes = final_slats
    else:
        with gpu_timer("SLAT 解码为 Mesh"):
            meshes = []
            for slat in final_slats:
                # 使用 pipeline 的解码方法
                mesh_list = convert_trellis_to_trimesh([slat])
                meshes.extend(mesh_list)
            
            print(f"🏆 网格解码完成: 生成了 {len(meshes)} 个 mesh")
    
    print(f"✅ TRELLIS Stage 2 管道完成:")
    print(f"   - 输出类型: {output_type}")
    print(f"   - 总 latents: {len(all_latents)}")
    print(f"   - 总 log_probs: {len(all_log_probs)}")
    print(f"   - 总 KL: {len(all_kl)}")
    
    # 返回与 hunyuan3d_pipeline_with_logprob 相同的格式
    return meshes, all_latents, all_log_probs, all_kl


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