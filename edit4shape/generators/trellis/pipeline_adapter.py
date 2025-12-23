"""
Trellis reference pipeline 适配器（统一使用 SparseTensor）。

仅依赖 _reference_codes/TRELLIS 下的 TrellisImageTo3DPipeline，
并对齐 edit4shape/systems/trellis.py 期望的接口：
- dense_sampling: 生成稀疏结构 coords，返回形状 (T,4)，外部可扩 batch。
- init_latents: 生成初始 SparseTensor latent（feats 形状 (N,C)）。
 - scheduler: 提供 set_timesteps/step，基于 FlowEuler 的公式，输入输出均为 SparseTensor。
- sparse_sampling_step: 单步预测 v（SparseTensor），支持 CFG。
- prepare_image_conditions: 预处理图像并生成 cond/neg_cond。
- backend.tokens_to_sparse: 直接返回 SparseTensor。
- precompute_cache: 占位直接回传。

注意：所有张量操作行均按用户要求添加形状注释。
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import os
import torch
import trimesh

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.modules.sparse import SparseTensor
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler



def build_pipeline_from_reference(cfg: Any, accelerator: Any) -> Any:
    """
    构建参考 Trellis pipeline 的适配器实例。
    """
    project_root = torch.__file__  # 占位以便 mypy，实际下方重置
    # 将 _reference_codes/TRELLIS 加入 sys.path
    project_root = sys.argv[0]  # 仅占位防静态检查告警
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
    if trellis_ref_root not in sys.path:
        sys.path.insert(0, trellis_ref_root)


    pipe_raw = TrellisImageTo3DPipeline.from_pretrained(cfg.pretrained.model)
    pipe_raw.to(accelerator.device)
    if accelerator.device.type == "cuda":
        pipe_raw.cuda()
    os.environ["TRELLIS_VERBOSE"] = "1" if bool(getattr(cfg, "verbose", False)) else "0"

    return TrellisRefAdapter(pipe_raw, FlowEulerSampler=FlowEulerSampler)


class TrellisRefAdapter:
    """
    适配 _reference_codes/TRELLIS 的 TrellisImageTo3DPipeline。
    """

    def __init__(self, pipe_raw: Any, FlowEulerSampler: Any):
        self.pipe = pipe_raw
        self.FlowEulerSampler = FlowEulerSampler

    # === Sampler 参数（直接使用 pipeline 内置配置） ===
    def get_sampler_runtime_params(self) -> tuple[int, float, int, float, float, float]:
        """
        返回 (ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold)。
        """
        ss_params = self.pipe.sparse_structure_sampler_params
        slat_params = self.pipe.slat_sampler_params
        ss_steps = int(ss_params["steps"])
        ss_guidance = float(ss_params["cfg_strength"])
        slat_steps = int(slat_params["steps"])
        slat_guidance = float(slat_params["cfg_strength"])
        slat_rescale_t = float(slat_params["rescale_t"])
        slat_mc_threshold = float(slat_params.get("mc_threshold", 0.0))
        return ss_steps, ss_guidance, slat_steps, slat_guidance, slat_rescale_t, slat_mc_threshold

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

    # === 条件准备 ===
    def prepare_image_conditions(self, images: List[Any]) -> Dict[str, Any]:
        """
        预处理图像并生成 cond/neg_cond。
        """
        images_proc = [self.pipe.preprocess_image(img) for img in images]  # images_proc: List[PIL]
        cond_dict = self.pipe.get_cond(images_proc)  # cond_dict: {"cond": (B,S,C), "neg_cond": (B,S,C)}
        return cond_dict

    # === 稀疏结构采样 ===
    def dense_sampling(self, condition_utils: Dict[str, Any], steps: Optional[int] = None) -> torch.Tensor:
        """
        生成稀疏结构 coords，并按 batch 写入 coords[:,0]，返回形状 (B*T,4) int32。
        """
        # 推断 batch_size
        cond = condition_utils.get("cond")
        if isinstance(cond, list):
            cond = torch.cat(cond, dim=0)  # cond: (B, ..., ...)
        assert isinstance(cond, torch.Tensor), "condition_utils['cond'] 必须为 Tensor 或 list[Tensor]"
        batch_size = int(cond.shape[0])  # ()

        ss_steps, _, _, _, _, _ = self.get_sampler_runtime_params()
        steps_val = int(ss_steps if steps is None else steps)  # 形状: 标量
        sampler_params = {**self.pipe.sparse_structure_sampler_params, "steps": steps_val}
        coords = self.pipe.sample_sparse_structure(
            cond=condition_utils,
            num_samples=1,
            sampler_params=sampler_params,
        )  # coords: (T,4)，coords[:,0] 默认为 0
        coords = coords.to(device=self.pipe.device, dtype=torch.int32)  # coords: (T,4)

        # 为每个 batch 样本写入 batch 索引并拼接
        coords_list = []
        for b in range(batch_size):
            cb = coords.clone()  # cb: (T,4)
            cb[:, 0] = b  # 写入 batch 维
            coords_list.append(cb)
        coords_batched = torch.cat(coords_list, dim=0)  # coords_batched: (B*T,4)
        return coords_batched

    # === latent 初始化 ===
    def init_latents(
        self,
        coords: torch.Tensor,
        in_channels: int,
        generator: Optional[torch.Generator] = None,
    ) -> Any:
        """
        根据输入的 coords 生成初始 SparseTensor latent，feats 形状 (N,C)。
        """
        coords_batched = coords
        feats = torch.randn(
            coords_batched.shape[0],
            int(in_channels),
            device=coords_batched.device,
            dtype=torch.float32,
            generator=generator,
        )  # feats: (N,C)
        return SparseTensor(coords=coords_batched, feats=feats)

    # === Scheduler 适配（基于 FlowEuler 公式） ===
    def scheduler(self) -> Any:
        sampler = self.pipe.slat_sampler  # FlowEulerSampler 或其变体

        class _Scheduler:
            def __init__(self, sampler_ref):
                self.sampler = sampler_ref
                self.timesteps: List[torch.Tensor] = []

            def set_timesteps(self, num_steps: int, device: torch.device, rescale_t: float = 1.0) -> None:
                # timesteps: 递减序列，含首尾（长度 num_steps+1）
                # 官方逻辑：t_seq = np.linspace(1, 0, steps + 1)
                # t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
                timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
                self.timesteps = rescale_t * timesteps / (1 + (rescale_t - 1) * timesteps)

            def step(self, noise_pred: Any, t: torch.Tensor, latents: Any) -> Any:
                """
                Euler 公式：x_{t-1} = x_t - (t - t_prev) * v，输入/输出均为 SparseTensor。
                """
                # noise_pred: SparseTensor
                # latents: SparseTensor
                # t: 标量
                t_val = float(t)
                # 查找 t_prev（要求 t 必须命中 timesteps 且存在后继）
                match_idx = (torch.isclose(self.timesteps, torch.tensor(t_val, device=self.timesteps.device, dtype=self.timesteps.dtype))).nonzero(as_tuple=False)
                assert match_idx.numel() > 0 and int(match_idx[0]) + 1 < self.timesteps.numel(), "t 必须匹配 timesteps 且有后继步"
                idx = int(match_idx[0])
                t_prev = float(self.timesteps[idx + 1].item())
                delta = (t_val - t_prev)  # 标量，保持与 FlowEuler sample_once 一致，不再 /1000
                pred_feats = latents.feats - delta * noise_pred.feats  # pred_feats: (N,C)
                prev_sample = SparseTensor(coords=latents.coords, feats=pred_feats)
                return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=None)

        return _Scheduler(sampler)

    # === 单步预测 v（原 denoise） ===
    def sparse_sampling_step(
        self,
        x_t_sparse: Any,
        timesteps: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> Any:
        """
        使用 slat_sampler 的 _get_model_prediction 预测 v，支持 CFG。
        输入/输出均为 SparseTensor，coords[:,0] 表示 batch 索引。
        """
        model = self.pipe.models["slat_flow_model"]
        t = timesteps  # t: 标量/[0,1]
        sampler_params = self.pipe.slat_sampler_params  # dict
        cfg_interval = sampler_params["cfg_interval"]  # cfg_interval: List[float]

        def _pred_v(cond, neg_cond=None):
            extra_args = {
                "neg_cond": neg_cond,
                "cfg_strength": float(guidance_scale),  # 标量
                "cfg_interval": cfg_interval,  # List[float]
            }

            pred_x0, pred_eps, pred_v = self.pipe.slat_sampler._get_model_prediction(
                model=model,
                x_t=x_t_sparse,
                t=t,
                cond=cond,
                **extra_args
            )  # pred_v: SparseTensor，feats: (N,C)
            return pred_v

        neg_cond = uncond_embeddings if uncond_embeddings is not None else None  # neg_cond: (B,S,C) 或 None
        pred_v = _pred_v(cond_embeddings, neg_cond=neg_cond)  # pred_v.feats: (N,C)
        return pred_v

    # === 预计算缓存（占位） ===
    def precompute_cache(self, sparse_latent: Any) -> Any:
        """
        占位：直接返回输入。
        """
        return sparse_latent

    # === tokens -> SparseTensor ===
