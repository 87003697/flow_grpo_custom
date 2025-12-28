#!/usr/bin/env python3
"""
TRELLIS Stage 2 Pipeline with Log Probability for GRPO Training

精简为两层封装：
- 外层：本文件 `TrellisPipelineWithLogProb`
- 内层：官方 `_reference_codes/TRELLIS` 提供的 `trellis.pipelines.trellis_image_to_3d.TrellisImageTo3DPipeline`
"""
import os
import sys
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
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from flow_grpo.diffusers_patch.trellis_sparse_tensor import (
    create_trellis_scheduler,
    set_trellis_timesteps,
    trellis_flow_step_with_logprob,
    trellis_flow_step_with_logprob_dense,
    extract_sparse_tensor_from_batch,
    sparse_tensor_cat,
    sparse_tensor_cfg_guidance,
)
from dataclasses import dataclass

# 注入 TRELLIS 官方代码路径，直接使用官方 Pipeline
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_TRELLIS_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "TRELLIS")
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

from trellis.modules import sparse as sp  # type: ignore
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline as RefPipeline  # type: ignore


# ============================================================================
# Mesh 转换工具：从 generators/trellis/utils/compat.py 迁移
# ============================================================================
def _to_trimesh(vertices, faces) -> trimesh.Trimesh:
    """将 (vertices, faces) 转为 trimesh.Trimesh。
    - 支持 torch.Tensor 或 numpy 数组输入
    - 不做兜底，仅负责类型转换与构造
    """
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def convert_trellis_to_trimesh(decoded: Union[Dict, List, trimesh.Trimesh, object]) -> List[trimesh.Trimesh]:
    """将 TRELLIS decode_slat 输出转换为 trimesh.Trimesh 列表"""
    meshes: List[trimesh.Trimesh] = []
    if isinstance(decoded, dict):
        mesh_data = decoded.get('mesh')
        if mesh_data is None:
            raise ValueError("decode_slat 输出缺少 'mesh' 键")
        if isinstance(mesh_data, list):
            for m in mesh_data:
                if isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
                else:
                    v = getattr(m, 'vertices', None)
                    f = getattr(m, 'faces', None)
                    if v is None or f is None:
                        raise TypeError("mesh对象缺少 vertices/faces 属性")
                    meshes.append(_to_trimesh(v, f))
        else:
            m = mesh_data
            if isinstance(m, trimesh.Trimesh):
                meshes.append(m)
            else:
                v = getattr(m, 'vertices', None)
                f = getattr(m, 'faces', None)
                if v is None or f is None:
                    raise TypeError("mesh对象缺少 vertices/faces 属性")
                meshes.append(_to_trimesh(v, f))
        return meshes

    if isinstance(decoded, list):
        if all(isinstance(x, trimesh.Trimesh) for x in decoded):
            return decoded
        out: List[trimesh.Trimesh] = []
        for m in decoded:
            v = getattr(m, 'vertices', None)
            f = getattr(m, 'faces', None)
            if v is None or f is None:
                raise TypeError("列表中的元素不是可识别的 mesh 表示")
            out.append(_to_trimesh(v, f))
        return out

    if isinstance(decoded, sp.SparseTensor):
        raise TypeError("收到 SparseTensor。请先调用 decode_slat(slat, formats=['mesh'])")

    if isinstance(decoded, trimesh.Trimesh):
        return [decoded]

    v = getattr(decoded, 'vertices', None)
    f = getattr(decoded, 'faces', None)
    if v is not None and f is not None:
        return [_to_trimesh(v, f)]
    raise TypeError("未知的 mesh 表示类型，无法转换为 trimesh.Trimesh")


def convert_trellis_to_kiuimesh(decoded: Union[Dict, List, trimesh.Trimesh]) -> List[KiuiMesh]:
    """将 TRELLIS decode_slat 输出转换为 KiuiMesh 列表"""
    meshes_trimesh = convert_trellis_to_trimesh(decoded)
    out: List[KiuiMesh] = []
    for m in meshes_trimesh:
        v = torch.tensor(m.vertices, dtype=torch.float32)
        f = torch.tensor(m.faces, dtype=torch.int32)
        out.append(KiuiMesh(v=v, f=f, device=v.device))
    return out


@dataclass
class SlatSamplerParams:
    mc_threshold: float = 0.2
    rescale_t: float = 1.0


class TrellisPipelineWithLogProb:
    """Trellis 最小 GRPO 包装（两层：本包装 + 官方 Pipeline）。"""

    def __init__(self, ref_pipeline: RefPipeline):
        self.ref = ref_pipeline
        self.device = self.ref.device  # 形状: 标量设备
        self.dtype = getattr(self.ref, 'dtype', torch.float32)  # 形状: 标量 dtype
        
        # Stage 1: 从 params 读取 steps 和 rescale_t
        self.stage1_scheduler = create_trellis_scheduler(
            steps=self.ref.sparse_structure_sampler_params['steps'],
            device=self.device,
            rescale_t=self.ref.sparse_structure_sampler_params['rescale_t']
        )

        # Stage 2: 直接从 params 读取 steps 和 rescale_t
        self.stage2_scheduler = create_trellis_scheduler(
            steps=self.ref.slat_sampler_params['steps'],
            device=self.device,
            rescale_t=self.ref.slat_sampler_params['rescale_t'],
        )

    @property
    def stage2_params(self):
        return self.ref.slat_sampler_params

    # --- 构建/设备迁移 ---
    @classmethod
    def from_pretrained(cls, model_path: str, verbose: bool = False) -> "TrellisPipelineWithLogProb":
        ref = RefPipeline.from_pretrained(model_path)
        return cls(ref)

    def to(self, device: Union[str, torch.device]) -> None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device  # 形状: 标量设备
        self.ref.to(dev)  # 形状: ()
        self.device = self.ref.device  # 形状: 标量设备
        ref_dtype = getattr(self.ref, 'dtype', None)
        if ref_dtype is not None:
            self.dtype = ref_dtype  # 形状: 标量 dtype

    def get_slat_flow_model(self) -> nn.Module:
        return self.ref.models['slat_flow_model']

    def get_structure_flow_model(self) -> nn.Module:
        return self.ref.models['sparse_structure_flow_model']
    
    def get_trainable_model_stage2(self) -> nn.Module:
        """Direct3D 对齐别名：返回 Stage 2 可训练模型。"""
        return self.get_slat_flow_model()
    
    def get_trainable_model_stage1(self) -> nn.Module:
        """Direct3D 对齐别名：返回 Stage 1 可训练模型。"""
        return self.get_structure_flow_model()
    
    def get_flow_module(self, kind: str, unwrap_ddp: bool = True) -> nn.Module:
        """
        统一获取 flow 模块，多卡时可解包 .module。
        
        Args:
            kind: "structure" | "shape_slat"
            unwrap_ddp: 是否返回内部模块（DDP/FSDP 包裹时）
        """
        if kind == "shape_slat":
            module = self.get_slat_flow_model()
        elif kind == "structure":
            module = self.get_structure_flow_model()
        else:
            raise ValueError(f"unknown flow kind: {kind}")

        if module is None:
            raise KeyError(f"flow module not found for kind={kind}")
        return module.module if (unwrap_ddp and hasattr(module, "module")) else module

    def _offload_sparse_tensor(self, sparse: sp.SparseTensor) -> sp.SparseTensor:
        feats_cpu = sparse.feats.detach().cpu()
        coords_cpu = sparse.coords.detach().cpu()
        layout = list(getattr(sparse, "layout", []))
        return sp.SparseTensor(feats=feats_cpu, coords=coords_cpu, layout=layout)

    # --- Direct3D 风格 API ---
    def prepare_image_conditions(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cond_dict = self.ref.get_cond(images)  # 形状: dict(cond:(B,P,C), neg_cond:(B,P,C))
        return cond_dict['cond'], cond_dict['neg_cond']

    # 仅保留基础接口：prepare_image_conditions（返回二元组）


    def forward_stage1(self, image_cond: Dict[str, torch.Tensor], **sampler_params) -> torch.Tensor:
        # 使用官方采样接口生成稀疏结构坐标
        return self.ref.sample_sparse_structure(cond=image_cond, num_samples=1, sampler_params=sampler_params)  # 形状: (N,4)

    

    # --- 解码 ---
    def decode_slat_to_mesh(self, slat: sp.SparseTensor, **decode_params) -> List:
        """解码 SLAT 稀疏张量为 mesh 对象列表。"""
        return self._decode_sparse_mesh(slat)

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
        decoded = self.ref.decode_slat(slat, formats=['mesh'])  # 形状: dict
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
    def stage2_with_logprob(
        self,
        stage1_cond_dict: Optional[Union[dict, List[dict]]] = None,
        slat_sampler_params: Optional[Union[SlatSamplerParams, Dict[str, float]]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        noise_level: float = 0.7,
    ) -> Tuple[List[KiuiMesh], List[sp.SparseTensor], torch.Tensor, torch.Tensor]:
        assert stage1_cond_dict is not None, "stage1 条件不能为空"
        cond_b = stage1_cond_dict["cond"]
        neg_b = stage1_cond_dict["neg_cond"]
        coords_st: sp.SparseTensor = stage1_cond_dict["coords"]
        BK = int(cond_b.shape[0])

        if isinstance(slat_sampler_params, SlatSamplerParams):
            sampler_params = slat_sampler_params
        elif isinstance(slat_sampler_params, dict):
            sampler_params = SlatSamplerParams(
                mc_threshold=float(slat_sampler_params.get("mc_threshold", 0.2)),
                rescale_t=float(slat_sampler_params.get("rescale_t", 1.0)),
            )
        elif slat_sampler_params is None:
            sampler_params = SlatSamplerParams()
        else:
            raise TypeError("slat_sampler_params 必须为 SlatSamplerParams 或 dict")

        scheduler = self.stage2_scheduler
        slat_flow_module = self.get_flow_module("shape_slat")
        in_channels = int(getattr(slat_flow_module, "in_channels"))

        coords = coords_st.coords.to(self.device).int()
        layouts: List[slice] = list(getattr(coords_st, "layout", []))
        total_points = int(coords.shape[0])
        feats0 = torch.randn(
            (total_points, in_channels),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        batched_current = sp.SparseTensor(coords=coords, feats=feats0, layout=layouts)

        cond_batched = cond_b.to(self.device, dtype=self.dtype)
        neg_batched = None if (neg_b is None) else neg_b.to(self.device, dtype=self.dtype)
        do_cfg = bool(guidance_scale > 1.0) and (neg_batched is not None)

        latents_seq: List[sp.SparseTensor] = [self._offload_sparse_tensor(batched_current)]
        log_prob_rows: List[torch.Tensor] = []

        for idx_t, t in enumerate(scheduler.timesteps[:-1]):
            t_tensor = torch.full((BK,), float(t), device=self.device, dtype=torch.float32)
            model_output = self._model_output(
                slat_flow_module=slat_flow_module,
                x_sp=batched_current,
                t_tensor=t_tensor,
                cond_batched=cond_batched,
                neg_batched=neg_batched,
                guidance_scale=float(guidance_scale),
            )
            t_prev = scheduler.timesteps[idx_t + 1]
            gen = (generator if (not bool(deterministic)) else None)
            prev_batched, log_prob_vec, _, _ = trellis_flow_step_with_logprob(
                scheduler=scheduler,
                sample=batched_current,
                model_output=model_output,
                timestep=float(t),
                prev_timestep=float(t_prev),
                generator=gen,
                deterministic=bool(deterministic),
                noise_level=float(noise_level),
            )
            batched_current = prev_batched
            latents_seq.append(self._offload_sparse_tensor(prev_batched))
            log_prob_rows.append(log_prob_vec.detach().cpu())

        final_batched = latents_seq[-1]
        meshes_all: List[KiuiMesh] = []
        mc_value = float(sampler_params.mc_threshold)
        for i in range(BK):
            single_sp = extract_sparse_tensor_from_batch(final_batched, i)
            single_sp = sp.SparseTensor(
                coords=single_sp.coords.to(self.device).int(),
                feats=single_sp.feats.to(self.device, dtype=self.dtype),
            )
            decoded = self._decode_sparse_mesh(single_sp)
            mesh_obj = decoded[0] if len(decoded) > 0 else None
            meshes_all.append(self._ensure_kiui_mesh(mesh_obj))

        log_prob_seq = torch.stack(log_prob_rows, dim=0) if len(log_prob_rows) > 0 else torch.empty((0, BK))
        t_seq_all = torch.cat([scheduler.timesteps[:-1], scheduler.timesteps[-1:]]).to(dtype=torch.float32).cpu()
        return meshes_all, latents_seq, log_prob_seq, t_seq_all


    def stage1_with_logprob(
        self,
        cond_dict: Dict[str, torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        noise_level: float = 0.7,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        稠密结构流（Stage1）带 logprob 的批量回放，与 Direct3D‑S2 实现方式对齐：
        - 以稠密噪声为初始状态，按 scheduler 时序进行 SDE/ODE 步进
        - 每步计算 (BK,) 的对数概率
        - 末步用结构解码器阈值化得到稀疏结构坐标
        返回：
        - coords_list: List[Tensor(N_i,4)]
        - log_prob_seq_dense_rows: List[Tensor(BK,)]（长度 steps）
        - t_seq: Tensor(steps+1,)
        """
        assert cond_dict is not None and 'cond' in cond_dict and 'neg_cond' in cond_dict
        device = self.device  # 形状: 标量设备
        dtype = self.dtype  # 形状: 标量 dtype
        cond_batched = cond_dict['cond']  # 形状: (BK,P,C)
        neg_batched = cond_dict['neg_cond']  # 形状: (BK,P,C) 或 None
        BK = int(cond_batched.shape[0])  # 形状: 标量

        # 调度器（与官方一致）
        steps = int(num_inference_steps)  # 形状: 标量
        scheduler = self.stage1_scheduler  # 形状: 调度器

        # 稠密结构流模型与解码器
        flow_model = self.ref.models['sparse_structure_flow_model']  # 形状: 模型/DDP
        decoder = self.ref.models['sparse_structure_decoder']  # 形状: 模型
        flow_model_attr = flow_model.module if hasattr(flow_model, 'module') else flow_model  # 形状: 模型本体
        reso = int(getattr(flow_model_attr, 'resolution'))  # 形状: 标量
        in_channels = int(getattr(flow_model_attr, 'in_channels'))  # 形状: 标量

        # 初始化稠密 latent
        init_shape = (BK, in_channels, reso, reso, reso)  # 形状: (BK,C,R,R,R)
        latents_cur = torch.randn(init_shape, dtype=dtype, device=device, generator=generator)  # 形状: (BK,C,R,R,R)

        # 条件
        cond_b = cond_batched.to(device=device, dtype=dtype)  # 形状: (BK,P,C)
        neg_b = None if (neg_batched is None) else neg_batched.to(device=device, dtype=dtype)  # 形状: (BK,P,C) 或 None

        # 序列记录
        log_prob_rows: List[torch.Tensor] = []
        latents_seq_dense: List[torch.Tensor] = [latents_cur.detach().cpu()]

        # 主循环
        for idx_t, t in enumerate(scheduler.timesteps[:-1]):  # 形状: ()
            t_tensor = torch.full((BK,), float(t), device=device, dtype=torch.float32)  # 形状: (BK,)
            # 模型输出（含 CFG）
            if (neg_b is not None) and (float(guidance_scale) > 1.0):
                vel_neg = flow_model(latents_cur, t_tensor, neg_b)  # 形状: (BK,C,R,R,R)
                vel_pos = flow_model(latents_cur, t_tensor, cond_b)  # 形状: (BK,C,R,R,R)
                model_out = vel_neg + float(guidance_scale) * (vel_pos - vel_neg)  # 形状: (BK,C,R,R,R)
            else:
                model_out = flow_model(latents_cur, t_tensor, cond_b)  # 形状: (BK,C,R,R,R)

            t_prev = scheduler.timesteps[idx_t + 1]  # 形状: ()
            gen = (generator if (not bool(deterministic)) else None)  # 形状: 可能为 None
            deterministic_step = bool(deterministic)  # 形状: ()

            latents_next, log_prob_vec, prev_mean, std_vec = trellis_flow_step_with_logprob_dense(
                scheduler=scheduler,
                sample=latents_cur,
                model_output=model_out,
                timestep=float(t),
                prev_timestep=float(t_prev),
                generator=gen,
                deterministic=deterministic_step,
                noise_level=float(noise_level),
            )  # 形状: ((BK,C,R,R,R),(BK,), (BK,C,R,R,R), (BK,))

            latents_cur = latents_next  # 形状: (BK,C,R,R,R)
            latents_seq_dense.append(latents_cur.detach().cpu())
            log_prob_rows.append(log_prob_vec.detach().cpu())

        # 稠密 -> 稀疏结构坐标（阈值化）
        with torch.no_grad():
            z_s = latents_cur  # 形状: (BK,C,R,R,R)
            occ = decoder(z_s)  # 形状: (BK, 1 or C_out, R,R,R)
            coords_all = torch.argwhere(occ > 0)[:, [0, 2, 3, 4]].int()  # 形状: (N,4)
        # 拆分为每样本
        coords_list: List[torch.Tensor] = []
        for b in range(BK):
            mask = (coords_all[:, 0] == b)  # 形状: (N,)
            coords_b = coords_all[mask].clone()  # 形状: (N_b,4)
            if coords_b.numel() == 0:
                coords_b = torch.zeros((0, 4), dtype=coords_all.dtype, device=coords_all.device)  # 形状: (0,4)
            coords_b[:, 0] = 0  # 形状: (N_b,4)
            coords_list.append(coords_b)

        # 时间序列（与 scheduler 一致）
        t_seq = scheduler.timesteps.to(device=device, dtype=torch.float32)  # 形状: (steps+1,)
        log_prob_seq_dense = torch.stack(log_prob_rows, dim=0) if len(log_prob_rows) > 0 else torch.empty((0, BK))
        return coords_list, latents_seq_dense, log_prob_seq_dense, t_seq
