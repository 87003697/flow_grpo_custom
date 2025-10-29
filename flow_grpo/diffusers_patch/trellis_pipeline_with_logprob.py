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
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
from kiui.mesh import Mesh as KiuiMesh
from flow_grpo.diffusers_patch.trellis_sparse_tensor import create_trellis_scheduler, trellis_flow_step_with_logprob

# 导入 TRELLIS 内置门面
from generators.trellis import sparse as sp

# 导入项目模块
from generators.trellis.pipeline import TrellisStage2Pipeline
# 训练路径不依赖兼容工具，解码时使用 pipeline.core_pipeline.decode_slat
from generators.trellis.patches.sparse_tensor_utils import sparse_tensor_cat
from generators.trellis.utils.compat import convert_trellis_to_trimesh
from flow_grpo.diffusers_patch.trellis_sparse_tensor import sparse_tensor_cfg_guidance

 

 



 
class TrellisPipelineWithLogProb:
    """Trellis 最小 GRPO 包装（类式接口）。"""

    def __init__(self, ref_pipeline: TrellisStage2Pipeline):
        self.ref = ref_pipeline
        self.device = self.ref.device  # 形状: 标量设备
        self.dtype = getattr(self.ref, 'dtype', torch.float32)  # 形状: 标量 dtype

    # --- 构建/设备迁移 ---
    @classmethod
    def from_pretrained(cls, model_path: str, verbose: bool = False) -> "TrellisPipelineWithLogProb":
        ref = TrellisStage2Pipeline(model_path=model_path, verbose=bool(verbose))
        return cls(ref)

    def to(self, device: Union[str, torch.device]) -> None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device  # 形状: 标量设备
        self.ref.to(dev)  # 形状: ()
        self.device = self.ref.device  # 形状: 标量设备
        ref_dtype = getattr(self.ref, 'dtype', None)
        if ref_dtype is not None:
            self.dtype = ref_dtype  # 形状: 标量 dtype

    def get_trainable_model(self) -> nn.Module:
        return self.ref.get_trainable_model()

    def get_structure_flow_model(self) -> nn.Module:
        model = self.ref.core_pipeline.models.sparse_structure_flow_model  # 形状: 模型
        return model  # 形状: 模型

    # --- Direct3D 风格 API ---
    def prepare_image_conditions(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cond, neg_cond = self.ref.prepare_image_conditions(images)  # 形状: (B,P,C), (B,P,C)
        return cond, neg_cond

    # 仅保留基础接口：prepare_image_conditions（返回二元组）


    def forward_stage1(self, image_cond: Dict[str, torch.Tensor], **sampler_params) -> torch.Tensor:
        return self.ref.forward_stage1(image_cond=image_cond, **sampler_params)  # 形状: (N,4)

    

    # --- 解码 ---
    def decode_slat_to_mesh(self, slat: sp.SparseTensor, **decode_params) -> List[trimesh.Trimesh]:
        return convert_trellis_to_trimesh([slat], **decode_params)

    def decode_structure_sparse_to_coords(self, structure_sparse: sp.SparseTensor, **kwargs) -> torch.Tensor:
        # 若官方结构流提供专用解码，请在此封装；默认返回 coords
        return structure_sparse.coords  # 形状: (N,4)

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

    def _decode_sparse_mesh(self, slat: sp.SparseTensor) -> List[object]:
        """将 SLAT 稀疏张量解码为 KiuiMesh 列表。"""
        decoded = self.ref.core_pipeline.decode_slat(slat, formats=['mesh'])  # 形状: dict
        mesh_data = decoded['mesh']
        mesh_list = mesh_data if isinstance(mesh_data, list) else [mesh_data]
        return [self._ensure_kiui_mesh(m) for m in mesh_list]

    # --- helpers to simplify CFG/model outputs (对齐 direct3d_s2) ---
    @staticmethod
    def _apply_cfg(vel_pos: sp.SparseTensor, vel_neg: Optional[sp.SparseTensor], guidance_scale: float) -> sp.SparseTensor:
        """当 vel_neg 为 None 或 guidance_scale <= 1.0 时，直接返回 vel_pos。否则做线性组合（稀疏 CFG）。"""
        if (vel_neg is None) or (float(guidance_scale) <= 1.0):
            return vel_pos
        return sparse_tensor_cfg_guidance(
            positive_sparse=vel_pos,
            negative_sparse=vel_neg,
            guidance_scale=float(guidance_scale),
        )

    @classmethod
    def _model_output(
        cls,
        slat_flow_module: torch.nn.Module,
        x_sp: sp.SparseTensor,
        t_tensor: torch.Tensor,
        cond_batched: torch.Tensor,
        neg_batched: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> sp.SparseTensor:
        """统一的模型输出（含可选 CFG）。"""
        vel_pos = slat_flow_module(x_sp, t_tensor, cond_batched)
        vel_neg = None
        if (neg_batched is not None) and (float(guidance_scale) > 1.0):
            vel_neg = slat_flow_module(x_sp, t_tensor, neg_batched)
        return cls._apply_cfg(vel_pos, vel_neg, guidance_scale)

    # ------------------------------
    # Minimal helpers (对齐 direct3d_s2)
    # ------------------------------
    def _ensure_kiui_mesh(self, mesh_obj: object):
        # 直接是 KiuiMesh
        if isinstance(mesh_obj, KiuiMesh):
            return mesh_obj.to(self.device)
        # trimesh.Trimesh -> KiuiMesh
        if isinstance(mesh_obj, trimesh.Trimesh):
            v = mesh_obj.vertices
            f = mesh_obj.faces
            v_t = torch.as_tensor(v, dtype=torch.float32, device=self.device)  # 形状: (V,3)
            f_t = torch.as_tensor(f, dtype=torch.int32, device=self.device)    # 形状: (F,3)
            return KiuiMesh(v=v_t, f=f_t, device=self.device)
        # 通用对象（带 vertices/faces）
        v_attr = getattr(mesh_obj, 'vertices', None)
        f_attr = getattr(mesh_obj, 'faces', None)
        if v_attr is not None and f_attr is not None:
            v_t = torch.as_tensor(v_attr, dtype=torch.float32, device=self.device)  # 形状: (V,3)
            f_t = torch.as_tensor(f_attr, dtype=torch.int32, device=self.device)    # 形状: (F,3)
            return KiuiMesh(v=v_t, f=f_t, device=self.device)
        # 回退到最小三角形，确保渲染不崩
        v_fb = torch.tensor([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]], dtype=torch.float32, device=self.device)
        f_fb = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=self.device)
        return KiuiMesh(v=v_fb, f=f_fb, device=self.device)

    # --- Stage1 with logprob（占位 logprob） ---
    @torch.no_grad()

    # --- Stage2 with logprob ---
    @torch.no_grad()
    def stage2_with_logprob(
        self,
        stage1_cond_dict: Optional[Union[dict, List[dict]]] = None,
        slat_sampler_params: Optional[dict] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> Tuple[List, List, List, torch.Tensor]:
        device = self.device  # 形状: 标量设备
        dtype = self.dtype  # 形状: 标量 dtype
        is_dist = (torch.distributed.is_available() and torch.distributed.is_initialized())
        rank = torch.distributed.get_rank() if is_dist else 0  # 形状: 标量

        assert stage1_cond_dict is not None and 'cond' in stage1_cond_dict and 'neg_cond' in stage1_cond_dict and 'coords' in stage1_cond_dict
        cond_b = stage1_cond_dict['cond']  # 形状: (BK,P,C)
        neg_b = stage1_cond_dict['neg_cond']  # 形状: (BK,P,C) 或 None
        coords_st: sp.SparseTensor = stage1_cond_dict['coords']  # 形状: batched 稀疏(仅用 coords+layout)
        BK = int(cond_b.shape[0])  # 形状: 标量

        do_classifier_free_guidance = bool(guidance_scale > 1.0)  # 形状: 标量

        # 直用上游提供的 batched coords（候选级 layout 已内联）

        # Stage 2: SLAT 生成 + LogProb
        with nullcontext():
            stage2_params = slat_sampler_params or {}

            all_latents: List[sp.SparseTensor] = []     # 形状: 列表(len=B*k*(steps+1))
            all_log_probs: List[torch.Tensor] = []      # 形状: 列表(len=B*k*steps), 每项 (,) 标量
            all_kl: List[torch.Tensor] = []             # 形状: 列表(len=B*k*steps)
            final_slats: List[sp.SparseTensor] = []     # 形状: 列表(len=B*k)

            slat_flow_model = self.get_trainable_model()  # 形状: 模型
            base_model = slat_flow_model.module if hasattr(slat_flow_model, "module") else slat_flow_model  # 形状: 模型
            in_channels = int(base_model.in_channels)  # 形状: 标量

            # 使用 batched coords 直接初始化噪声
            coords_batched = coords_st.coords.to(device).int()  # 形状: (sum N,4)
            layouts: List[slice] = list(getattr(coords_st, 'layout', []))  # 形状: 长度 BK
            total_points = int(coords_batched.shape[0])  # 形状: 标量
            noise_feats = torch.randn((total_points, in_channels), device=device, dtype=dtype, generator=generator)  # 形状: (sum N, C)
            batched_noise = sp.SparseTensor(coords=coords_batched, feats=noise_feats, layout=layouts)  # 形状: batched 稀疏

            # 条件按 BK 对齐
            pos_cond_batched = cond_b.to(device, dtype=dtype)  # 形状: (BK,P,C)
            neg_cond_batched = (None if (neg_b is None) else neg_b.to(device, dtype=dtype))  # 形状: (BK,P,C) 或 None
            Bk = int(pos_cond_batched.shape[0])  # 形状: 标量

            # 构建 scheduler 与时间对
            steps = int(num_inference_steps)  # 形状: 标量
            rescale_t = float(stage2_params.get('rescale_t', 1.0))  # 形状: 标量
            scheduler = create_trellis_scheduler(steps=steps, device=device, rescale_t=rescale_t)
            t_seq = (scheduler.timesteps.cpu().numpy()).astype(np.float32)  # 形状: (steps+1,)
            t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]  # 长度 steps

            sample = batched_noise  # 形状: batched 稀疏
            all_latents_batched = [sample]  # 长度 steps+1
            all_log_probs_batched: List[torch.Tensor] = []  # 每步 (BK,)

            do_cfg = bool(guidance_scale > 1.0) and (neg_cond_batched is not None)  # 形状: 标量

            for t, t_prev in t_pairs:
                Bk_loop = int(sample.shape[0])  # 形状: 标量
                t_tensor = torch.tensor([t] * Bk_loop, device=sample.coords.device, dtype=torch.float32)  # 形状: (BK,)

                if do_cfg:
                    with torch.no_grad():
                        neg_out = slat_flow_model(sample, t_tensor, neg_cond_batched)
                    with torch.no_grad():
                        pos_out = slat_flow_model(sample, t_tensor, pos_cond_batched)
                    cfg_feats = neg_out.feats + float(guidance_scale) * (pos_out.feats - neg_out.feats)  # 形状: (sumN, C)
                    model_output = sp.SparseTensor(coords=sample.coords, feats=cfg_feats)
                else:
                    with torch.no_grad():
                        model_output = slat_flow_model(sample, t_tensor, pos_cond_batched)

                sample, log_prob, sample_mean, std_dev = trellis_flow_step_with_logprob(
                    scheduler=scheduler,
                    sample=sample,
                    model_output=model_output,
                    timestep=float(t),
                    prev_timestep=float(t_prev),
                    generator=generator,
                    deterministic=bool(deterministic),
                )

                all_latents_batched.append(sample)
                all_log_probs_batched.append(log_prob)

            # 拆分 batched 稀疏为每样本 SLAT
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

            final_slat_batched = sample
            final_slats_per_sample = split_batched_sparse(final_slat_batched, Bk)  # 形状: 列表(Bk)

            # 展平成 per-sample 列表
            sample_latents_flat: List[sp.SparseTensor] = []
            sample_log_probs_flat: List[torch.Tensor] = []
            for b in range(Bk):
                for step_idx in range(len(all_latents_batched)):
                    sample_latents_flat.append(all_latents_batched[step_idx][b])
                for step_idx in range(len(all_log_probs_batched)):
                    sample_log_probs_flat.append(all_log_probs_batched[step_idx][b])

            sample_kl_flat: List[torch.Tensor] = [torch.zeros_like(all_log_probs_batched[0]) for _ in range(len(all_log_probs_batched))] if len(all_log_probs_batched) > 0 else []

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

        # 输出与 t_seq
        # 统一解码为 KiuiMesh 列表
        meshes = []
        for slat in final_slats:
            meshes.extend(self._decode_sparse_mesh(slat))

        # 返回 t_seq 与 scheduler 一致
        t_seq = scheduler.timesteps.to(device=self.device, dtype=torch.float32)  # 形状: (steps+1,)

        return meshes, all_latents, all_log_probs, t_seq


    def stage1_with_logprob(
        self,
        cond_dict: Dict[str, torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        assert cond_dict is not None and 'cond' in cond_dict and 'neg_cond' in cond_dict
        device = self.device  # 形状: 标量设备
        cond_batched = cond_dict['cond']  # 形状: (BK,P,C)
        neg_batched = cond_dict['neg_cond']  # 形状: (BK,P,C) 或 None
        BK = int(cond_batched.shape[0])  # 形状: 标量

        # 并行生成稀疏结构坐标：一次性前向
        stage1_params: Dict = {}
        coords_batched = self.forward_stage1(image_cond={'cond': cond_batched, 'neg_cond': neg_batched}, **stage1_params)  # 形状: (sum N,4)
        # 拆分为每样本坐标，并将 batch 维归零
        coords_list: List[torch.Tensor] = []
        for b in range(BK):
            mask = (coords_batched[:, 0] == b)
            coords_b = coords_batched[mask].clone()
            if coords_b.numel() == 0:
                coords_b = torch.zeros((0, 4), dtype=coords_batched.dtype, device=coords_batched.device)
            coords_b[:, 0] = 0
            coords_list.append(coords_b)

        # 时间表（占位，与 Stage2 一致的 rescale_t）
        steps = int(num_inference_steps)  # 形状: 标量
        rescale_t = 1.0  # 形状: 标量
        t_seq_np = np.linspace(1.0, 0.0, steps + 1) * 1000.0  # 形状: (steps+1,)
        t_seq_np = rescale_t * t_seq_np / (1.0 + (rescale_t - 1.0) * t_seq_np / 1000.0)  # 形状: (steps+1,)
        t_seq = torch.as_tensor(t_seq_np, device=device, dtype=torch.float32)  # 形状: (steps+1,)

        # 占位结构 logprob（当前无结构流逐步 logprob 实现，返回 0 向量）
        log_prob_seq_sparse: List[torch.Tensor] = [torch.zeros((BK,), device=device, dtype=torch.float32) for _ in range(steps)]  # 长度 steps

        return coords_list, log_prob_seq_sparse, t_seq
