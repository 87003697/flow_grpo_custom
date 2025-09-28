"""
Direct3D‑S2 Pipeline Wrapper with LogProb for GRPO (sparse512)

参考文档：`DEV.md`
- 本文件实现 dense -> sparse512（符合参考流程：dense -> sparse512 -> refiner -> sparse1024 -> refiner）。本实现聚焦于 sparse512 阶段与 logprob。
- 主要参考：
    - 参考管线：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py`
        - 类与构造/加载：1-172 行
        - 设备迁移：54-66 行
        - 条件编码：194-217 行
        - 采样主循环（CFG + scheduler.step + 可选 SDE）：260-314 行
        - 解码与后处理：320-341 行
    - 现有 GRPO 管线：
        - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`（采样与 logprob 框架）
        - `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`（单步 SDE / logprob 风格）

约束：
- 不使用 try/except 或 fallback。
- 每行张量运算附形状注释。
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Union

import torch
from kiui.mesh import Mesh as KiuiMesh

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DIRECT3D_S2_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "Direct3D-S2")
if _DIRECT3D_S2_ROOT not in sys.path:
        sys.path.append(_DIRECT3D_S2_ROOT)

from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import sparse_tensor_cfg_guidance
from direct3d_s2.utils import sort_block  # type: ignore
from direct3d_s2.pipeline import Direct3DS2Pipeline as _RefPipeline  # type: ignore

from .direct3d_s2_sde_with_logprob import sde_step_with_logprob
@dataclass
class PipelineOptions:
    """最小配置，保持与 Trellis Stage2 行为一致。"""
    use_refiner: bool = False  # 是否加载与使用 refiner（默认 False）


@dataclass
class SparseStageConfig:
    steps: int = 30
    guidance_scale: float = 0.0
    use_sde: bool = True
    sigma_min: float = 0.002
    rescale_t: float = 0.5
    mc_threshold: float = 0.2


@dataclass
class SlatSamplerParams:
    sigma_min: float
    rescale_t: float
    mc_threshold: float
    use_sde: bool


class Direct3DS2PipelineWithLogProb:
    """Direct3D‑S2 最小 GRPO 包装。"""

    def __init__(self, ref_pipeline: _RefPipeline, opts: Optional[PipelineOptions] = None):
        """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:23-53`（参考管线的构造与成员挂载）"""
        self.ref = ref_pipeline
        self.dtype = getattr(ref_pipeline, "dtype", torch.float16)
        self.device = getattr(ref_pipeline, "device", torch.device("cpu"))
        if opts is None:
            opts = PipelineOptions(
                use_refiner = os.environ.get("DIRECT3D_USE_REFINER", "0") == "1",
            )
        self.opts = opts

    @staticmethod
    def _clear_cuda_cache(*objs: Optional[Any]) -> None:
        any_obj = False
        for obj in objs:
            if obj is None:
                continue
            del obj
            any_obj = True
        if any_obj and torch.cuda.is_available():
            torch.cuda.empty_cache()


    @staticmethod
    def _offload_sparse_tensor(feats: torch.Tensor, coords: torch.Tensor) -> sp.SparseTensor:
        feats_cpu = feats.detach().cpu()  # shape: (N_total, C)
        coords_cpu = coords.detach().cpu()  # shape: (N_total, 4)
        return sp.SparseTensor(feats=feats_cpu, coords=coords_cpu)


    @classmethod
    def from_pretrained(
        cls,
        pipeline_path: str,
        subfolder: str = "direct3d-s2-v-1-1",
        dtype: torch.dtype = torch.float16,
        minimal_512_only: bool = True,
        use_refiner: bool = False,
        opts: Optional[PipelineOptions] = None,
    ) -> "Direct3DS2PipelineWithLogProb":
        """构建 Direct3D‑S2 参考管线包装（512-only）。参数 `minimal_512_only` 保留仅为兼容，不影响行为。"""
        if opts is None:
            opts = PipelineOptions(use_refiner=bool(use_refiner))
        ref = cls._custom_load(pipeline_path, dtype, opts=opts)
        return cls(ref, opts=opts)

    @staticmethod
    def _custom_load(pipeline_path: str, dtype: torch.dtype, opts: Optional[PipelineOptions] = None) -> _RefPipeline:
        """自定义加载：构建 dense + sparse512（禁用 1024 分支），对齐参考实现风格。"""
        from omegaconf import OmegaConf
        from direct3d_s2.utils import instantiate_from_config  # type: ignore

        cfg = OmegaConf.load(os.path.join(pipeline_path, 'config.yaml'))

        def load_ckpt(path: str):
            """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:117,125,133,141,146`（torch.load 用法）"""
            return torch.load(path, map_location='cpu', weights_only=True)

        def build(node):
            """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:118,121,126,129,142,147,151,154-156`（instantiate_from_config）"""
            return instantiate_from_config(node)

        def load_pair(vae_cfg, dit_cfg, ckpt_filename: str):
            """加载 (vae, dit) 配对权重。"""
            sd = load_ckpt(os.path.join(pipeline_path, ckpt_filename))
            vae = build(vae_cfg); vae.load_state_dict(sd['vae'], strict=True); vae.eval()
            dit = build(dit_cfg); dit.load_state_dict(sd['dit'], strict=True); dit.eval()
            return vae, dit

        def load_single(cfg_node, ckpt_filename: str, key: str):
            """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:141-149`（refiner/refiner_1024 加载流程）"""
            sd = load_ckpt(os.path.join(pipeline_path, ckpt_filename))
            mod = build(cfg_node); mod.load_state_dict(sd[key], strict=True); mod.eval()
            return mod

        dense_vae, dense_dit = load_pair(cfg.dense_vae, cfg.dense_dit, 'model_dense.ckpt')
        sparse_vae_512, sparse_dit_512 = load_pair(cfg.sparse_vae_512, cfg.sparse_dit_512, 'model_sparse_512.ckpt')

        use_refiner_flag = (opts.use_refiner if opts else False)
        if not use_refiner_flag:
            class _RefinerStub(torch.nn.Module):
                def run(self, *a, **k):
                    return a
            refiner = _RefinerStub()
        else:
            refiner = load_single(cfg.refiner, 'model_refiner.ckpt', 'refiner')

        dense_image_encoder = build(cfg.dense_image_encoder).eval()
        sparse_image_encoder = build(cfg.sparse_image_encoder).eval()

        # 调度器
        dense_scheduler = build(cfg.dense_scheduler)
        sparse_scheduler_512 = build(cfg.sparse_scheduler_512)

        class _NullModule(torch.nn.Module):
            def forward(self, *a, **k):
                raise RuntimeError('1024-stage disabled in 512-only mode')

        # 统一 dtype 与归一化层处理的简化助手
        def _set_module_dtype_flags(module: torch.nn.Module, target: torch.dtype) -> None:
            if hasattr(module, 'dtype'):
                module.dtype = target
            if hasattr(module, 'use_fp16'):
                module.use_fp16 = (target == torch.float16)

        def _cast_and_fix_norms(module: torch.nn.Module, target: torch.dtype, keep_norm_fp32: bool, include_groupnorm: bool) -> None:
            if hasattr(module, 'to'):
                module.to(dtype=target)
            if keep_norm_fp32:
                norm_names = {"LayerNorm32", "GroupNorm32", "ChannelLayerNorm32"}
                for sub in module.modules():
                    name = sub.__class__.__name__
                    if (name in norm_names) or (include_groupnorm and isinstance(sub, torch.nn.GroupNorm)):
                        sub.to(dtype=torch.float32)

        # 批量应用到主要模块
        main_modules = [dense_vae, dense_dit, sparse_vae_512, sparse_dit_512, dense_image_encoder, sparse_image_encoder]
        for m in main_modules:
            _cast_and_fix_norms(m, dtype, keep_norm_fp32=True, include_groupnorm=False)
        if use_refiner_flag:
            _cast_and_fix_norms(refiner, dtype, keep_norm_fp32=True, include_groupnorm=True)

        # 设置模块上的 dtype/use_fp16 标志（仅对需要者）
        _set_module_dtype_flags(dense_dit, dtype)
        _set_module_dtype_flags(sparse_dit_512, dtype)

        ref = _RefPipeline(
            dense_vae=dense_vae,
            dense_dit=dense_dit,
            sparse_vae_512=sparse_vae_512,
            sparse_dit_512=sparse_dit_512,
            sparse_vae_1024=_NullModule(),
            sparse_dit_1024=_NullModule(),
            refiner=refiner,
            refiner_1024=_NullModule(),
            dense_image_encoder=dense_image_encoder,
            sparse_image_encoder=sparse_image_encoder,
            dense_scheduler=dense_scheduler,
            sparse_scheduler_512=sparse_scheduler_512,
            sparse_scheduler_1024=_NullModule(),
            dtype=dtype,
        )
        ref.dtype = dtype
        ref._use_refiner = bool(use_refiner_flag)
        return ref

    def to(self, device: str) -> None:
        """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:54-66`（将各子模块迁移到 device）"""
        self.ref.to(device)
        self.device = torch.device(device)

    # --- Trellis 兼容接口：返回可训练模型（Stage2 的 sparse_dit_512） ---
    def get_trainable_model(self):
        """提供与 Trellis 一致的接口，返回用于训练的稀疏分支模型。

        返回：nn.Module（通常为 sparse_dit_512）
        """
        return self.ref.sparse_dit_512

    # --- Public helpers for Trellis-style calling ---
    def prepare_image_conditions(self, image: Any, do_classifier_free_guidance: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """将图像编码为稀疏分支的 cond/neg_cond（patch 级）。

        返回:
        - cond: torch.Tensor  # 形状依赖于模型实现，通常为 (B, P, C)
        - neg_cond: Optional[torch.Tensor]  # 同形状；当 do_classifier_free_guidance=False 时为 None
        
        参考：
        - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:185-193`（prepare_image）
        - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:194-217`（encode_image）
        """
        with torch.no_grad():
            image_tensor = self.ref.prepare_image(image)  # (B,C,H,W)
            image_tensor = image_tensor.to(dtype=self.dtype)  # (B,C,H,W)
            sparse_conditions_flag = self._resolve_sparse_conditions_flag()
            cond, uncond = self.ref.encode_image(
                image_tensor,
                self.ref.sparse_image_encoder,
                do_classifier_free_guidance=bool(do_classifier_free_guidance),
                use_mask=sparse_conditions_flag,
            )
        return cond, (uncond if do_classifier_free_guidance else None)

    # Trellis 风格：显式的 preprocess 与 batch 版条件编码
    def preprocess_image(self, image: Any) -> torch.Tensor:
        """对单张图像执行与 Trellis 一致的预处理（返回 (1,C,H,W) 张量）。"""
        with torch.no_grad():
            image_tensor = self.ref.prepare_image(image)  # (B,C,H,W)
        return image_tensor.to(dtype=self.dtype)

    def preprocess_images(self, images: List[Any]) -> torch.Tensor:
        """批量预处理，拼接为 (B,C,H,W)。"""
        tensors = [self.preprocess_image(img) for img in images]
        return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

    def prepare_image_conditions_batch(self, images_tensor: torch.Tensor, do_classifier_free_guidance: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """对预处理后的 (B,C,H,W) 批量编码 patch 条件，返回 (B,P,C) cond/neg。"""
        images_tensor = images_tensor.to(dtype=self.dtype)
        sparse_conditions_flag = self._resolve_sparse_conditions_flag()
        cond, uncond = self.ref.encode_image(
            images_tensor,
            self.ref.sparse_image_encoder,
            do_classifier_free_guidance=bool(do_classifier_free_guidance),
            use_mask=sparse_conditions_flag,
        )
        return cond, (uncond if do_classifier_free_guidance else None)

    def _resolve_sparse_conditions_flag(self) -> bool:
        sparse_dit = self.ref.sparse_dit_512
        has_flag = hasattr(sparse_dit, "sparse_conditions")
        module_has_flag = hasattr(sparse_dit, "module") and hasattr(sparse_dit.module, "sparse_conditions")
        if has_flag:
            return sparse_dit.sparse_conditions
        if module_has_flag:
            return sparse_dit.module.sparse_conditions
        raise AttributeError("sparse_dit_512 missing sparse_conditions attribute in both wrapper and module")

    def _resolve_sparse_dit_module(self):
        sparse_dit = self.ref.sparse_dit_512
        if hasattr(sparse_dit, "module"):
            return sparse_dit.module
        return sparse_dit

    # Trellis 风格别名：forward_stage1（对齐命名与职责）
    def forward_stage1(
        self,
        image: Any,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """参考：直接调用参考管线的 dense `inference`，获取 64^3 索引（不进行 128^3 上采样）。
        - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:219-341`（inference 逻辑）
        - `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:359-363`（dense 索引获取与 sort_block）
        """
        image_tensor = self.ref.prepare_image(image)  # (B,C,H,W)
        image_tensor = image_tensor.to(dtype=self.dtype)  # (B,C,H,W)
        with torch.no_grad():
            dense_indices = self.ref.inference(  # list/tuple(len=B)
                image=image_tensor,  # (B, C, H, W)
                vae=self.ref.dense_vae,
                dit=self.ref.dense_dit,
                conditioner=self.ref.dense_image_encoder,
                scheduler=self.ref.dense_scheduler,
                num_inference_steps=int(num_inference_steps),  # 标量
                guidance_scale=float(guidance_scale),  # 标量
                generator=generator,
                mode='dense',
                mc_threshold=0.1,
            )
        latent_index_64 = dense_indices[0].to(dtype=torch.int64)  # (N,4)
        # 直接使用 64^3 索引
        latent_index_64[:, 1:] = torch.clamp(latent_index_64[:, 1:], 0, 63)  # (N,4)
        latent_index_64 = torch.unique(latent_index_64, dim=0)  # (N',4)
        sparse_dit_module = self._resolve_sparse_dit_module()
        latent_index_64 = sort_block(latent_index_64, sparse_dit_module.selection_block_size)  # (N',4)
        self._clear_cuda_cache(image_tensor, dense_indices)
        return latent_index_64

    # ------------------------------
    # Minimal helpers
    # ------------------------------
    # （简化）移除额外包装：直接在调用处使用参考调度器与解码器

    def _decode_sparse_mesh(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mc_threshold: float = 0.2,
        remove_interior: bool = False,
    ):
        """将稀疏潜变量解码为 mesh（测试脚本可能直接调用）。"""
        coords_int = coords.int()  # (N,4)
        latents_scaled = 1.0 / self.ref.sparse_vae_512.latents_scale * feats + self.ref.sparse_vae_512.latents_shift  # (N,C)
        latents_scaled = latents_scaled.to(self.dtype)  # (N,C) 确保与权重半精度一致
        lat_sp = sp.SparseTensor(latents_scaled, coords_int)  # (N,C)+(N,4)
        if bool(getattr(self.ref, '_use_refiner', False)) and bool(remove_interior):
            reconst_feat = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp, return_feat=True)
            mesh_out = self.ref.refiner.run(*reconst_feat, mc_threshold=float(mc_threshold) * 2.0)
        else:
            mesh_out = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp, mc_threshold=float(mc_threshold))
        if isinstance(mesh_out, list):
            mesh_candidate = (mesh_out[0] if len(mesh_out) > 0 else None)
        else:
            mesh_candidate = mesh_out
        return self._ensure_kiui_mesh(mesh_candidate)

    def _ensure_kiui_mesh(self, mesh_obj: Any) -> KiuiMesh:
        if isinstance(mesh_obj, KiuiMesh):
            return mesh_obj.to(self.device)
        if hasattr(mesh_obj, "v") and hasattr(mesh_obj, "f"):
            verts_src = mesh_obj.v
            faces_src = mesh_obj.f
        elif hasattr(mesh_obj, "vertices") and hasattr(mesh_obj, "faces"):
            verts_src = mesh_obj.vertices
            faces_src = mesh_obj.faces
        else:
            raise TypeError("mesh 对象缺少 vertices/faces 或 v/f 字段，无法转换为 KiuiMesh")
        vertices_tensor = torch.as_tensor(verts_src, dtype=torch.float32, device=self.device)  # (N,3)
        faces_tensor = torch.as_tensor(faces_src, dtype=torch.int32, device=self.device)  # (M,3)
        return KiuiMesh(v=vertices_tensor, f=faces_tensor, device=self.device)

    def _sample_sparse_candidates(self, *args, **kwargs):
        raise RuntimeError("_sample_sparse_candidates 已移除：SDE 采样内联至 stage2_with_logprob")

    # 移除旧接口：sample_candidates_with_logprob（改用 stage2_with_logprob）
    
    # --- Trellis-style alias: stage2_with_logprob ---
    def stage2_with_logprob(
        self,
        num_inference_steps: int = 30,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        kl_reward: float = 0.0,
        deterministic: bool = False,
        sparse_structure_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        stage1_cond_dict: Optional[Union[dict, List[dict]]] = None,
        num_candidates: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[Any], List[sp.SparseTensor], List[torch.Tensor], List[torch.Tensor]]:
        """Stage2 采样。支持传入单个或多个 stage1 条目，内部逐条处理并展平输出。"""

        if stage1_cond_dict is None:
            raise ValueError("stage2_with_logprob 需要提供 stage1_cond_dict")

        entries = stage1_cond_dict if isinstance(stage1_cond_dict, list) else [stage1_cond_dict]
        if len(entries) == 0:
            raise ValueError("stage1_cond_dict 列表不能为空")

        # Sparse 阶段参数
        if slat_sampler_params is None:
            raise ValueError("stage2_with_logprob 需要提供 slat_sampler_params")
        if not isinstance(slat_sampler_params, SlatSamplerParams):
            raise TypeError("slat_sampler_params 必须为 SlatSamplerParams")
        sampler_params = slat_sampler_params

        sched = self.ref.sparse_scheduler_512
        sparse_dit_module = self._resolve_sparse_dit_module()

        meshes_all: List[Any] = []
        latents_seq_all: List[sp.SparseTensor] = []
        step_log_probs_all: List[torch.Tensor] = []
        step_kl_all: List[torch.Tensor] = []

        for entry in entries:
            if ('cond' not in entry) or ('neg_cond' not in entry):
                raise ValueError("stage1 条目缺少 'cond' 或 'neg_cond'")
            name_coords = 'coords' if ('coords' in entry) else 'latent_index'
            coords_tensor = entry[name_coords]
            if not isinstance(coords_tensor, torch.Tensor):
                raise ValueError("'coords' 必须为 torch.Tensor")

            cond = entry['cond'].to(self.device)
            uncond = entry['neg_cond']
            if uncond is not None:
                uncond = uncond.to(self.device)
            coords_int = coords_tensor.to(self.device).int()

            with torch.no_grad():
                for _ in range(int(num_candidates)):
                    sched.set_timesteps(int(num_inference_steps), device=self.device)
                    latent_shape = (int(coords_int.shape[0]), int(sparse_dit_module.out_channels))
                    latents = torch.randn(latent_shape, dtype=self.dtype, device=self.device, generator=generator)

                    # 记录初始 latent
                    latents_seq_all.append(self._offload_sparse_tensor(latents, coords_int))  # (N,C)+(N,4)

                    for idx_t, t in enumerate(sched.timesteps):
                        t_tensor = latents.new_tensor([t])  # (1)
                        x_sp = sp.SparseTensor(latents, coords_int)  # (N,C)+(N,4)
                        noise_cond = sparse_dit_module(x_sp, t_tensor, cond)  # (N,C)

                        if uncond is not None:
                            noise_uncond = sparse_dit_module(x_sp, t_tensor, uncond)  # (N,C)
                            noise_sparse = sparse_tensor_cfg_guidance(
                                positive_sparse=noise_cond,
                                negative_sparse=noise_uncond,
                                guidance_scale=float(guidance_scale),
                            )  # (N,C)+(N,4)
                            noise = noise_sparse.feats  # (N,C)
                        else:
                            noise = noise_cond.feats  # (N,C)

                        prev_mean = sched.step(noise, float(t_tensor.item()), latents, generator=generator).prev_sample  # (N,C)

                        t_prev = sched.timesteps[idx_t + 1] if idx_t + 1 < len(sched.timesteps) else t  # 标量
                        t_cur_f32 = latents.new_tensor(float(t), dtype=torch.float32)  # (1)
                        t_prev_f32 = latents.new_tensor(float(t_prev), dtype=torch.float32)  # (1)
                        prev_sample_tensor = None
                        if sampler_params.use_sde:
                            prev_sample_tensor, log_prob_step_tensor, _, _, _ = sde_step_with_logprob(
                                prev_mean=prev_mean,
                                t_cur=t_cur_f32,
                                t_prev=t_prev_f32,
                                rescale_t=sampler_params.rescale_t,
                                sigma_min=sampler_params.sigma_min,
                                generator=generator,
                                deterministic=bool(deterministic),
                                prev_sample=None,
                            )
                            latents = prev_sample_tensor
                            log_prob_val = log_prob_step_tensor.view(-1)
                        else:
                            latents = prev_mean
                            log_prob_val = torch.zeros(1, device=self.device, dtype=torch.float32)

                        latents_seq_all.append(self._offload_sparse_tensor(latents, coords_int))  # (N,C)+(N,4)
                        log_prob_cpu = log_prob_val.detach().cpu()  # shape: (1,)
                        step_log_probs_all.append(log_prob_cpu)  # (1)
                        step_kl_all.append(torch.zeros_like(log_prob_cpu))  # (1)
                        self._clear_cuda_cache(noise_sparse, noise_cond, noise_uncond, prev_mean, prev_sample_tensor, x_sp, t_tensor, t_cur_f32, t_prev_f32)

                    latents_scaled = 1.0 / self.ref.sparse_vae_512.latents_scale * latents + self.ref.sparse_vae_512.latents_shift
                    latents_scaled = latents_scaled.to(self.dtype)
                    lat_sp = sp.SparseTensor(latents_scaled, coords_int.clone())
                    mc_value = sampler_params.mc_threshold
                    
                    reconst_feat = None  # shape: None 表示未使用重建特征
                    if getattr(self.ref, '_use_refiner', False):
                        reconst_feat = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp, return_feat=True)
                        mesh_out = self.ref.refiner.run(*reconst_feat, mc_threshold=mc_value * 2.0)
                    else:
                        mesh_out = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp, mc_threshold=mc_value)
                    mesh_single = (
                        mesh_out[0]
                        if isinstance(mesh_out, list) and len(mesh_out) > 0
                        else (mesh_out if not isinstance(mesh_out, list) else None)
                    )
                    mesh_kiui = self._ensure_kiui_mesh(mesh_single)
                    meshes_all.append(mesh_kiui)
                    self._clear_cuda_cache(latents_scaled, lat_sp, mesh_out, reconst_feat, latents)

            self._clear_cuda_cache(coords_int, cond, uncond)

        return meshes_all, latents_seq_all, step_log_probs_all, step_kl_all


def direct3d_s2_stage2_with_logprob(
    pipeline: Direct3DS2PipelineWithLogProb,
    image_path: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    sparse_structure_sampler_params: Optional[Dict] = None,
    slat_sampler_params: Optional[Dict] = None,
    num_candidates: int = 1,
    verbose: bool = False,
    **kwargs,
):
    """
    与 Trellis 对齐的 Direct3D‑S2 Stage2 封装：在函数内部执行 Stage1，然后调用 pipeline.stage2_with_logprob。

    参数对齐 Trellis（精简版）：
    - image_path: 输入图像路径（用于 Stage1 与 条件编码）
    - num_inference_steps: 稀疏阶段步数（传入 Stage2）
    - guidance_scale: 分类器引导系数（同时决定是否计算 neg_cond）
    - generator/deterministic/num_candidates 等均向下传递

    参考：无直接对应；封装自仓库内 `Direct3DS2PipelineWithLogProb.stage2_with_logprob`，其核心逻辑参考
    `_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:253-291, 320-341`。
    """
    # ==== Stage 1: 生成 latent_index（稀疏坐标索引）与 patch 级 cond/neg_cond ====
    st1_params = sparse_structure_sampler_params or {}
    st1_steps = int(st1_params.get("num_inference_steps", 50))  # 标量
    st1_guidance = float(st1_params.get("guidance_scale", 0.0))  # 标量

    latent_index = pipeline.forward_stage1(
        image=image_path,
        num_inference_steps=st1_steps,
        guidance_scale=st1_guidance,
        generator=generator,
    )  # 形状 (N,4)

    do_cfg = guidance_scale > 0.0  # 标量
    cond, neg_cond = pipeline.prepare_image_conditions(
        image=image_path,
        do_classifier_free_guidance=bool(do_cfg),
    )  # cond: (B,P,C), neg_cond: (B,P,C) 或 None

    stage1_cond_dict = {
        'cond': cond,               # (B,P,C)
        'neg_cond': neg_cond,       # (B,P,C) 或 None
        'coords': latent_index,     # (N,4)
        'image_path': image_path,   # 字符串路径
    }

    # ==== Stage 2: 稀疏阶段采样 + LogProb ====
    meshes, all_latents, all_log_probs, all_kl = pipeline.stage2_with_logprob(
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        deterministic=bool(deterministic),
        sparse_structure_sampler_params=sparse_structure_sampler_params,
        slat_sampler_params=slat_sampler_params,
        stage1_cond_dict=stage1_cond_dict,
        num_candidates=int(num_candidates),
        verbose=bool(verbose),
    )

    return meshes, all_latents, all_log_probs, all_kl


# （移除）与 Trellis 完全对齐的自由函数包装，为简化接口不再提供
