"""
Trellis reference pipeline 适配器（统一使用 SparseTensor）。

仅依赖 _reference_codes/TRELLIS 下的 TrellisImageTo3DPipeline，
并对齐 edit4shape/systems/trellis.py 期望的接口：
- dense_sampling: 生成稀疏结构 coords，返回形状 (T,4)，外部可扩 batch。
- init_sparse_latents: 生成初始 SparseTensor latent（feats 形状 (N,C)）。
 - scheduler: 提供 set_timesteps/step，基于 FlowEuler 的公式，输入输出均为 SparseTensor。
- sparse_sampling_step: 单步预测 v（SparseTensor），支持 CFG。
- prepare_image_conditions: 预处理图像并生成 cond/neg_cond。
- backend.tokens_to_sparse: 直接返回 SparseTensor。
- precompute_cache: 占位直接回传。

注意：所有张量操作行均按用户要求添加形状注释。
"""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.modules.sparse import SparseTensor
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler

from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler



def build_pipeline_from_reference(cfg: Any, accelerator: Any, device: Optional[torch.device] = None) -> Any:
    """
    构建参考 Trellis pipeline 的适配器实例。
    
    Args:
        cfg: 配置对象
        accelerator: Accelerate 加速器
        device: 可选，指定模型加载的设备。如果不指定，使用 accelerator.device
    """
    project_root = torch.__file__  # 占位以便 mypy，实际下方重置
    # 将 _reference_codes/TRELLIS 加入 sys.path
    project_root = sys.argv[0]  # 仅占位防静态检查告警
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
    if trellis_ref_root not in sys.path:
        sys.path.insert(0, trellis_ref_root)
    triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
    if triposf_ref_root not in sys.path:
        sys.path.insert(0, triposf_ref_root)


    # 设置默认 CUDA 设备（支持传入自定义设备用于流水线并行）
    if device is None:
        device = accelerator.device
    if device.type == "cuda":
        # 确保设备有具体索引
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    
    pipe_raw = TrellisImageTo3DPipeline.from_pretrained(cfg.pretrained.model)
    pipe_raw.to(device)
    # 注意：不再调用 pipe_raw.cuda()，因为它会覆盖设备设置为 GPU 0
    os.environ["TRELLIS_VERBOSE"] = "1" if bool(getattr(cfg, "verbose", False)) else "0"

    return TrellisRefAdapter(pipe_raw, FlowEulerSampler=FlowEulerSampler)


# =====================================================================
# 共享工具
# =====================================================================

def _scale_timesteps(
    timesteps: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将 [0,1] 范围的时间步缩放为模型期望的 [0,1000] 范围。

    支持标量 float / 0-dim Tensor / (B,) Tensor 输入。
    """
    if torch.is_tensor(timesteps):
        if timesteps.dim() == 0:
            return torch.full(
                (batch_size,), float(timesteps.item()) * 1000,
                device=device, dtype=torch.float32,
            )  # (B,)
        return timesteps * 1000  # (B,)
    return torch.full(
        (batch_size,), float(timesteps) * 1000,
        device=device, dtype=torch.float32,
    )  # (B,)


# =====================================================================
# DenseStage — Stage 1 (sparse_structure_flow_model)
# =====================================================================

class DenseStage:
    """Stage 1 操作：dense Tensor (B, C, R, R, R)。"""

    def __init__(self, adapter: "TrellisRefAdapter"):
        self._adapter = adapter

    @property
    def pipe(self):
        return self._adapter.pipe

    def get_runtime_params(self) -> tuple[int, float, float, float, float]:
        """
        返回 (steps, guidance, rescale_t, cfg_min, cfg_max)。
        """
        ss_params = self.pipe.sparse_structure_sampler_params
        return (
            int(ss_params["steps"]),
            float(ss_params["cfg_strength"]),
            float(ss_params["rescale_t"]),
            float(ss_params["cfg_interval"][0]),
            float(ss_params["cfg_interval"][1]),
        )

    def resolve_flow_module(self) -> Any:
        """获取 sparse_structure_flow_model 原始模型（去除 DDP 包装）。"""
        model = self.pipe.models["sparse_structure_flow_model"]
        return model.module if hasattr(model, "module") else model

    def decode_to_coords(
        self,
        z_s: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        将 Dense latent 解码为占位坐标。

        z_s (B, C, R, R, R) → decoder → threshold > 0 → coords (B*T, 4) int32

        Args:
            z_s: Dense rollout 输出的 latent (B, C, R, R, R)
            batch_size: 用于 batch 索引扩展

        Returns:
            coords: (B*T, 4) int32，列为 [batch_idx, x, y, z]
        """
        decoder = self.pipe.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()  # (T, 4)

        if batch_size <= 1:
            return coords

        coords_list = []
        for b in range(batch_size):
            cb = coords.clone()  # (T, 4)
            cb[:, 0] = b
            coords_list.append(cb)
        return torch.cat(coords_list, dim=0)  # (B*T, 4)

    def init_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """生成 Dense 初始噪声 (B, C, R, R, R)。

        与原生 sample_sparse_structure 保持一致：先在 CPU 上生成噪声，
        再 .to(device)，以确保相同种子下数值完全相同。
        """
        flow_model = self.resolve_flow_module()
        reso = flow_model.resolution  # 16
        in_ch = flow_model.in_channels  # 8
        noise = torch.randn(
            batch_size, in_ch, reso, reso, reso,
            dtype=torch.float32,
            generator=generator,
        )  # CPU
        return noise.to(self.pipe.device)  # (B, C, R, R, R)

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: Any,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """Dense 单步 v 预测，返回 (B, C, R, R, R)。"""
        model = self.pipe.models["sparse_structure_flow_model"]
        t_scaled = _scale_timesteps(timesteps, x_t.shape[0], x_t.device)
        return model(x_t, t_scaled, cond_embeddings)  # (B, C, R, R, R)

    def scheduler(
        self,
        steps: int,
        rescale_t: float,
    ) -> Tuple[np.ndarray, list]:
        """
        时间步序列生成。

        Returns:
            (t_seq, t_pairs): t_seq 为 (steps+1,)，t_pairs 为 [(t, t_prev), ...]
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
        return t_seq, t_pairs


# =====================================================================
# SparseStage — Stage 2 (slat_flow_model)
# =====================================================================

class SparseStage:
    """Stage 2 操作：SparseTensor (feats: (N, C), coords: (N, 4))。"""

    def __init__(self, adapter: "TrellisRefAdapter"):
        self._adapter = adapter

    @property
    def pipe(self):
        return self._adapter.pipe

    def get_runtime_params(self) -> tuple[int, float, float, float, float, float]:
        """
        返回 (steps, guidance, rescale_t, cfg_min, cfg_max, mc_threshold)。
        """
        slat_params = self.pipe.slat_sampler_params
        return (
            int(slat_params["steps"]),
            float(slat_params["cfg_strength"]),
            float(slat_params["rescale_t"]),
            float(slat_params["cfg_interval"][0]),
            float(slat_params["cfg_interval"][1]),
            float(slat_params.get("mc_threshold", 0.0)),
        )

    def resolve_flow_module(self) -> Any:
        """获取 slat_flow_model 原始模型（去除 DDP 包装）。"""
        model = self.pipe.models["slat_flow_model"]
        return model.module if hasattr(model, "module") else model

    def init_latents(
        self,
        coords: torch.Tensor,
        in_channels: int,
        generator: Optional[torch.Generator] = None,
    ) -> Any:
        """生成初始 SparseTensor latent (feats: (N, C))。"""
        feats = torch.randn(
            coords.shape[0],
            int(in_channels),
            device=coords.device,
            dtype=torch.float32,
            generator=generator,
        )  # (N, C)
        return SparseTensor(coords=coords, feats=feats)

    def sampling_step(
        self,
        x_t_sparse: Any,
        timesteps: Any,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> Any:
        """
        Sparse 单步 v 预测，返回 SparseTensor。

        模型期望 t ∈ [0, 1000]，外部传入 [0, 1]，内部缩放。
        CFG 由外部处理。
        """
        model = self.pipe.models["slat_flow_model"]
        t_scaled = _scale_timesteps(timesteps, cond_embeddings.shape[0], x_t_sparse.device)
        return model(x_t_sparse, t_scaled, cond_embeddings)

    def scheduler(self) -> TrellisFlowScheduler:
        """返回 Trellis Flow Matching 调度器。"""
        return TrellisFlowScheduler()

    def decode(
        self,
        latents: Any,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """统一封装 decode_slat。"""
        fmt = formats if formats is not None else ["mesh"]
        return self.pipe.decode_slat(latents, formats=fmt)


# =====================================================================
# TrellisRefAdapter — 组合 + 通用工具
# =====================================================================

class TrellisRefAdapter:
    """
    适配 TrellisImageTo3DPipeline，提供 .dense 和 .sparse 子对象。

    用法:
        pipeline.dense.sampling_step(...)   # Stage 1
        pipeline.sparse.sampling_step(...)  # Stage 2
        pipeline.prepare_image_conditions(...)  # 通用
    """

    def __init__(self, pipe_raw: Any, FlowEulerSampler: Any):
        self.pipe = pipe_raw
        self.FlowEulerSampler = FlowEulerSampler
        self.dense = DenseStage(self)
        self.sparse = SparseStage(self)

    # === 兼容入口 ===

    def get_sampler_runtime_params(self) -> tuple[int, float, int, float, float, float]:
        """
        兼容入口：返回 (ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold)。
        """
        ss_steps, ss_guidance, _, _, _ = self.dense.get_runtime_params()
        slat_steps, slat_guidance, slat_rescale_t, _, _, slat_mc_threshold = self.sparse.get_runtime_params()
        return ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold

    # === 条件 / 图像 ===

    def prepare_image_conditions(self, images: List[Any]) -> Dict[str, Any]:
        """预处理图像并生成 cond/neg_cond。"""
        images_proc = [self.pipe.preprocess_image(img) for img in images]
        cond_dict = self.pipe.get_cond(images_proc)

        cond = cond_dict.get("cond")
        if cond is None:
            raise ValueError("prepare_image_conditions: get_cond 返回的 cond 为空。")
        neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))  # (B, S, C)

        return {"cond": cond, "neg_cond": neg_cond}

    # === Mesh 导出 ===

    def export_mesh_obj(self, mesh: Any, out_path: str) -> None:
        """导出 MeshExtractResult 为 OBJ。"""
        if mesh is None:
            return
        mesh_np = trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh_np.export(out_path)

    # === LoRA 控制 ===

    @contextmanager
    def disable_lora_context(self, model_key: str = "slat_flow_model"):
        """临时禁用指定模型的 LoRA 适配器。"""
        raw = self.pipe.models[model_key]
        model = raw.module if hasattr(raw, "module") else raw
        if hasattr(model, 'disable_adapters'):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            yield
