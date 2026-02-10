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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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


class TrellisRefAdapter:
    """
    适配 _reference_codes/TRELLIS 的 TrellisImageTo3DPipeline。
    """

    def __init__(self, pipe_raw: Any, FlowEulerSampler: Any):
        self.pipe = pipe_raw
        self.FlowEulerSampler = FlowEulerSampler

    def _resolve_slat_flow_module(self) -> Any:
        """获取 slat_flow_model 的原始模型（去除 DDP 包装），用于属性访问。"""
        model = self.pipe.models["slat_flow_model"]
        return model.module if hasattr(model, "module") else model

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
        cond_dict = self.pipe.get_cond(images_proc)  # cond_dict: {"cond": (B,S,C), "neg_cond": (B,S,C)} 或缺少 neg_cond

        cond = cond_dict.get("cond")
        if cond is None:
            raise ValueError("prepare_image_conditions: get_cond 返回的 cond 为空，无法继续。")
        neg_cond = cond_dict.get("neg_cond", torch.zeros_like(cond))  # neg_cond: (B,S,C)

        return {"cond": cond, "neg_cond": neg_cond}

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
    def scheduler(self) -> TrellisFlowScheduler:
        """返回 Trellis 专用 Flow Matching 调度器"""
        return TrellisFlowScheduler()

    # === 单步预测 v（原 denoise） ===
    def sparse_sampling_step(
        self,
        x_t_sparse: Any,
        timesteps: torch.Tensor,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
    ) -> Any:
        """
        简化版：始终直连 slat_flow_model forward，不走 _get_model_prediction / GuidanceInterval。
        CFG 由外部（如 trellis.rollout_sparse 的 mix_cfg）负责。
        输入/输出均为 SparseTensor，coords[:,0] 表示 batch 索引。
        
        注意：模型期望的 t 范围是 [0, 1000]，而非 [0, 1]，需要缩放。
        参考：_reference_codes/TRELLIS/trellis/pipelines/samplers/flow_euler.py:39
        """
        model = self.pipe.models["slat_flow_model"]
        
        # 时间步缩放：模型期望 t * 1000（与源代码 FlowEulerSampler._inference_model 对齐）
        if torch.is_tensor(timesteps):
            if timesteps.dim() == 0:
                # 标量 tensor，扩展为 batch 形状
                batch_size = cond_embeddings.shape[0]  # ()
                t_scaled = torch.full(
                    (batch_size,), float(timesteps.item()) * 1000,
                    device=x_t_sparse.device, dtype=torch.float32
                )  # (B,)
            else:
                t_scaled = timesteps * 1000  # (B,)
        else:
            batch_size = cond_embeddings.shape[0]  # ()
            t_scaled = torch.full(
                (batch_size,), float(timesteps) * 1000,
                device=x_t_sparse.device, dtype=torch.float32
            )  # (B,)

        # 仅 cond 前向，feats: (N,C)
        cond_pred_v = model(x_t_sparse, t_scaled, cond_embeddings)
        return cond_pred_v

    # === 预计算缓存（占位） ===
    def precompute_cache(self, sparse_latent: Any) -> Any:
        """
        占位：直接返回输入。
        """
        return sparse_latent

    # === Decode ===
    def decode(self, latents: Any, formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        统一封装 decode_slat，便于上层选择 mesh / gaussian 等输出。
        """
        fmt = formats if formats is not None else ["mesh"]
        outputs = self.pipe.decode_slat(latents, formats=fmt)
        return outputs

    # === LoRA 控制 ===
    @contextmanager
    def disable_lora_context(self):
        """
        临时禁用 LoRA 适配器的上下文管理器。
        用于正则化时获取教师（原始模型）的预测。
        注意：使用 _resolve 获取原始模型，因为 DDP 不暴露 disable_adapters。
        """
        model = self._resolve_slat_flow_module()
        if hasattr(model, 'disable_adapters'):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            # 无 LoRA 时直接透过
            yield
