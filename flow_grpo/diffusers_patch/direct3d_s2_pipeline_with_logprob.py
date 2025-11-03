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

# 强制 torch.hub 离线加载，避免多进程在构建编码器时访问网络
os.environ.setdefault("TORCH_HUB_DISABLE_NETWORK", "1")
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DIRECT3D_S2_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "Direct3D-S2")
if _DIRECT3D_S2_ROOT not in sys.path:
        sys.path.append(_DIRECT3D_S2_ROOT)

from flow_grpo.diffusers_patch import direct3d_s2_sparse_tensor as sp
from flow_grpo.diffusers_patch.direct3d_s2_sparse_tensor import sparse_tensor_cfg_guidance, Stage1RuntimeConfig
from direct3d_s2.utils import sort_block  # type: ignore
from direct3d_s2.pipeline import Direct3DS2Pipeline as _RefPipeline  # type: ignore

from .direct3d_s2_sparse_tensor import direct3d_flow_step_with_logprob, direct3d_flow_step_with_logprob_dense, compute_log_prob_direct3d_stage1


@dataclass
class PipelineOptions:
    """最小配置，保持与 Trellis Stage2 行为一致。"""
    use_refiner: bool = False  # 是否加载与使用 refiner（默认 False）


    


@dataclass
class SlatSamplerParams:
    mc_threshold: float


class Direct3DS2PipelineWithLogProb:
    """Direct3D‑S2 最小 GRPO 包装。"""

    def __init__(self, ref_pipeline: _RefPipeline, opts: Optional[PipelineOptions] = None):
        """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:23-53`（参考管线的构造与成员挂载）"""
        self.ref = ref_pipeline
        self.dtype = getattr(ref_pipeline, "dtype", torch.float16)
        self.device = getattr(ref_pipeline, "device", torch.device("cpu"))
        if opts is None:
            opts = PipelineOptions(use_refiner=False)
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


    def _offload_sparse_tensor(self, sparse: sp.SparseTensor) -> sp.SparseTensor:
        feats_cpu = sparse.feats.detach().cpu()  # shape: (N_total, C)
        coords_cpu = sparse.coords.detach().cpu()  # shape: (N_total, 4)
        return sp.SparseTensor(feats=feats_cpu, coords=coords_cpu, layout=list(sparse.layout))


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

        # 强制要求本地 TorchHub 已预热 dinov2，若不存在则直接报错；并将编码器 model 字段重定向到本地目录
        torch_home_dir = os.path.expanduser(os.environ.get('TORCH_HOME', '~/.cache/torch'))
        local_hub_repo = os.path.join(torch_home_dir, 'hub', 'facebookresearch_dinov2_main')
        assert os.path.isdir(local_hub_repo), f"本地 dinov2 TorchHub 缓存不存在: {local_hub_repo}. 请先运行 scripts/download/download_dinov2.py"

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

    # --- 训练模型接口 ---
    def get_trainable_model_stage2(self):
        """返回 Stage2（sparse_512）的可训练模型。"""
        return self.ref.sparse_dit_512

    def get_trainable_model_stage1(self):
        """返回 Stage1（dense）的可训练模型。"""
        return self.ref.dense_dit

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

    def _resolve_dense_dit_module(self):
        dense_dit = self.ref.dense_dit
        if hasattr(dense_dit, "module"):
            return dense_dit.module
        return dense_dit

    # --- helpers to simplify CFG/model outputs ---
    @staticmethod
    def _apply_cfg(vel_pos: sp.SparseTensor, vel_neg: Optional[sp.SparseTensor], guidance_scale: float) -> sp.SparseTensor:
        """当 vel_neg 为 None 或 guidance_scale <= 1.0 时，直接返回 vel_pos。否则做线性组合（稀疏 CFG）。"""
        if (vel_neg is None) or (float(guidance_scale) <= 1.0):
            return vel_pos  # 形状: 稀疏
        return sparse_tensor_cfg_guidance(
            positive_sparse=vel_pos,
            negative_sparse=vel_neg,
            guidance_scale=float(guidance_scale),
        )  # 形状: 稀疏

    @classmethod
    def _model_output(
        cls,
        sparse_dit_module: torch.nn.Module,
        x_sp: sp.SparseTensor,
        t_tensor: torch.Tensor,
        cond_batched: torch.Tensor,
        neg_batched: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> sp.SparseTensor:
        """统一的模型输出（含可选 CFG）。"""
        vel_pos = sparse_dit_module(x_sp, t_tensor, cond_batched)  # 形状: 稀疏（flow 速度）
        vel_neg = None
        if (neg_batched is not None) and (float(guidance_scale) > 1.0):
            vel_neg = sparse_dit_module(x_sp, t_tensor, neg_batched)  # 形状: 稀疏
        return cls._apply_cfg(vel_pos, vel_neg, guidance_scale)  # 形状: 稀疏

    # Trellis 风格别名：forward_stage1（批量版，对齐命名与职责）
    def forward_stage1(
        self,
        images: List[Any],
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        images_tensor = self.preprocess_images(images)  # (B,C,H,W)
        images_tensor = images_tensor.to(dtype=self.dtype)  # (B,C,H,W)
        B = int(images_tensor.shape[0])  # 标量
        with torch.no_grad():
            dense_indices = self.ref.inference(  # list/tuple(len=B)
                image=images_tensor,  # (B, C, H, W)
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
        coords_list: List[torch.Tensor] = []
        sparse_dit_module = self._resolve_sparse_dit_module()
        for i in range(B):
            latent_index_64 = dense_indices[i].to(dtype=torch.int64)  # (N_i,4)
            latent_index_64[:, 1:] = torch.clamp(latent_index_64[:, 1:], 0, 63)  # (N_i,4)
            latent_index_64 = torch.unique(latent_index_64, dim=0)  # (N'_i,4)
            latent_index_64 = sort_block(latent_index_64, sparse_dit_module.selection_block_size)  # (N'_i,4)
            coords_list.append(latent_index_64)  # 追加 (N'_i,4)
        self._clear_cuda_cache(images_tensor, dense_indices)
        return coords_list

    

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
        lat_sp = sp.SparseTensor(latents_scaled, coords_int, layout=[slice(0, latents_scaled.shape[0])])  # (N,C)+(N,4)
        if bool(self.ref._use_refiner) and bool(remove_interior):
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
        stage1_cond_dict: Optional[Union[dict, List[dict]]] = None,
        slat_sampler_params: Optional[dict] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        noise_level: float = 0.7,
    ) -> Tuple[List[Any], List[sp.SparseTensor], torch.Tensor, torch.Tensor]:
        """Stage2 采样。支持传入单个或多个 stage1 条目，内部逐条处理并展平输出。"""

        # 新格式：单个批字典 {'cond': (BK,P,C), 'neg_cond': (BK,P,C)|None, 'coords': SparseTensor(候选级layout)}
        cond_b = stage1_cond_dict["cond"]  # 形状: (BK, P, C)
        neg_b = stage1_cond_dict["neg_cond"]  # 形状: (BK, P, C) 或 None
        coords_st: sp.SparseTensor = stage1_cond_dict["coords"]  # 形状: 稀疏(合批，候选级layout)
        BK = int(cond_b.shape[0])  # 形状: 标量
        # 省略一致性检查（由上游保证）

        # Sparse 阶段参数
        # 省略 slat_sampler_params 校验（由上游保证）
        if not isinstance(slat_sampler_params, SlatSamplerParams):
            raise TypeError("slat_sampler_params 必须为 SlatSamplerParams")
        sampler_params = slat_sampler_params

        # ==== 批处理实现（B × K 一次性进行 SDE 步进）====
        sched = self.ref.sparse_scheduler_512
        sched.set_timesteps(int(num_inference_steps), device=self.device)
        sparse_dit_module = self._resolve_sparse_dit_module()

        meshes_all: List[Any] = []
        # 改为整批输出：
        # - latents_seq: List[batched SparseTensor]，长度 steps+1
        # - log_prob_seq: List[torch.Tensor]（步序列），stack后即为 (steps, BK)
        latents_seq: List[sp.SparseTensor] = []
        log_prob_seq: List[torch.Tensor] = []  # 每步一行，形状 (BK,)

        # 直用上游提供的候选级 batched coords 与 layout，初始化整批 latent 特征
        coords = coords_st.coords.to(self.device).int()  # 形状: (sum N, 4)
        layouts: List[slice] = list(coords_st.layout)  # 形状: 长度 BK
        total_points = int(coords.shape[0])  # 形状: 标量
        C_out = int(sparse_dit_module.out_channels)  # 形状: 标量
        feats0 = torch.randn((total_points, C_out), dtype=self.dtype, device=self.device, generator=generator)  # 形状: (sum N, C)
        batched_current = sp.SparseTensor(feats=feats0, coords=coords, layout=layouts)  # 形状: batched 稀疏

        # 批条件已为 (BK,P,C)
        cond_batched = cond_b.to(self.device, dtype=self.dtype)  # 形状: (BK, P, C)
        neg_batched = (None if (neg_b is None) else neg_b.to(self.device, dtype=self.dtype))  # 形状: (BK,P,C) 或 None

        # 记录初始 batched latent（直接整批 offload，带候选级 layout）
        latents_seq.append(self._offload_sparse_tensor(batched_current))  # 形状: batched 稀疏(CPU)

        # 时间步循环（一次性批处理 BK 个候选）
        for idx_t, t in enumerate(sched.timesteps[:-1]):
            t_tensor = torch.full((BK,), float(t), device=self.device, dtype=torch.float32)  # 形状: (BK,)
            x_sp = batched_current  # 形状: batched 稀疏
            model_output_sparse = self._model_output(
                sparse_dit_module=sparse_dit_module,
                x_sp=x_sp,
                t_tensor=t_tensor,
                cond_batched=cond_batched,
                neg_batched=neg_batched,
                guidance_scale=float(guidance_scale),
            )  # 形状: 稀疏

            t_prev = sched.timesteps[idx_t + 1]  # 形状: 标量
            gen = (generator if (not bool(deterministic)) else None)  # 形状: 可为 None
            deterministic_step = bool(deterministic)  # 形状: 标量

            prev_batched, log_prob_vec, _, _ = direct3d_flow_step_with_logprob(
                scheduler=sched,
                sample=x_sp,
                model_output=model_output_sparse,
                timestep=float(t),
                prev_timestep=float(t_prev),
                generator=gen,
                deterministic=deterministic_step,
                noise_level=float(noise_level),
            )  # 形状: (稀疏, (BK,), 稀疏均值, (BK,))

            batched_current = prev_batched  # 形状: 稀疏

            # 追加本步的 batched latent 与每候选 log_prob 标量（已是 BK 长度）
            latents_seq.append(self._offload_sparse_tensor(prev_batched))  # 形状: batched 稀疏(CPU)
            log_prob_seq.append(log_prob_vec.detach().cpu())  # 形状: (BK,)
            self._clear_cuda_cache(model_output_sparse, prev_batched)

        # 串行解码每个候选，使用通用拆分工具提取单候选稀疏张量
        final_batched = latents_seq[-1]
        mc_value = sampler_params.mc_threshold  # 形状: 标量
        with torch.no_grad():
            for i in range(BK):  # 形状: 标量循环
                single_sp = sp.extract_sparse_tensor_from_batch(final_batched, i)  # 形状: 稀疏(单候选)
                feats_c = single_sp.feats.to(self.device, dtype=self.dtype)  # 形状: (N_i, C)
                coords_c = single_sp.coords.to(self.device).int()  # 形状: (N_i, 4)
                lat_sp_single = sp.SparseTensor(
                    1.0 / self.ref.sparse_vae_512.latents_scale * feats_c + self.ref.sparse_vae_512.latents_shift,  # 形状: (N_i, C)
                    coords_c,  # 形状: (N_i, 4)
                    layout=[slice(0, feats_c.shape[0])],  # 形状: 单候选 layout
                )
                if self.ref._use_refiner:
                    reconst_feat = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp_single, return_feat=True)
                    mesh_single = self.ref.refiner.run(*reconst_feat, mc_threshold=mc_value * 2.0)
                    if isinstance(mesh_single, list):
                        mesh_single = (mesh_single[0] if len(mesh_single) > 0 else None)
                else:
                    mesh_single = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp_single, mc_threshold=mc_value)
                    if isinstance(mesh_single, list):
                        mesh_single = (mesh_single[0] if len(mesh_single) > 0 else None)
                meshes_all.append(self._ensure_kiui_mesh(mesh_single))
                self._clear_cuda_cache(single_sp, feats_c, coords_c, lat_sp_single)

        # 返回有效步数+1 的时间序列（layout 已内联为候选级）
        t_seq_all = torch.cat([sched.timesteps[:-1], sched.timesteps[-1:]]).to(dtype=torch.float32).cpu()
        return meshes_all, latents_seq, torch.stack(log_prob_seq, dim=0), t_seq_all

# ------------------------------
# Stage1 with logprob (dense SDE rollout)
# ------------------------------
    def stage1_with_logprob(
        self,
        cond_dict: Dict[str, torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        noise_level: float = 0.7,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """稠密分支批量 SDE/ODE 采样与 logprob 记录。

        输入：
        - cond_batched: (BK, P, C)
        - neg_batched: Optional[(BK, P, C)]
        返回：
        - coords_list: List[Tensor(N_i,4)]
        - latents_seq_dense: List[Tensor(BK,C,R,R,R)]
        - log_prob_seq_dense: Tensor(steps, BK)
        - t_seq: Tensor(steps+1,)
        """
        cond_batched = cond_dict["cond"]  # 形状: (BK,P,C)
        neg_batched = cond_dict.get("neg_cond")  # 形状: (BK,P,C) 或 None
        BK = int(cond_batched.shape[0])  # 形状: ()

        # 调度器
        sched = self.ref.dense_scheduler  # 形状: ()
        sched.set_timesteps(int(num_inference_steps), device=self.device)  # 形状: ()

        dense_dit = self.ref.dense_dit  # 形状: 模型（可能为 DDP 包裹）
        # 与稀疏分支一致：通过解包后的模块读取属性
        latent_shape = self._resolve_dense_dit_module().latent_shape  # 形状: 可能为 (C,R,R,R)

        # 初始化稠密 latent
        init_shape = (BK, *latent_shape)  # 形状: (BK,C,R,R,R)
        latents_cur = torch.randn(init_shape, dtype=self.dtype, device=self.device, generator=generator)  # 形状: (BK,C,R,R,R)

        # 条件准备（与 Stage2 接口一致，直接传入模型，CFG 逐元线性合成）
        cond_b = cond_batched.to(self.device, dtype=self.dtype)  # 形状: (BK,P,C)
        neg_b = None if (neg_batched is None) else neg_batched.to(self.device, dtype=self.dtype)  # 形状: (BK,P,C) 或 None

        # 记录序列
        latents_seq_dense: List[torch.Tensor] = []  # 形状: 列表(len=steps+1)
        log_prob_seq_dense_rows: List[torch.Tensor] = []  # 形状: 列表(len=steps，每项 (BK,))
        latents_seq_dense.append(latents_cur.detach().cpu())  # 形状: (BK,C,R,R,R)

        for idx_t, t in enumerate(sched.timesteps[:-1]):  # 形状: ()
            t_tensor = torch.full((BK,), float(t), device=self.device, dtype=torch.float32)  # 形状: (BK,)
            # 模型输出（含 CFG）
            if (neg_b is not None) and (float(guidance_scale) > 1.0):
                vel_neg = dense_dit(latents_cur, t_tensor, neg_b)  # 形状: (BK,C,R,R,R)
                vel_pos = dense_dit(latents_cur, t_tensor, cond_b)  # 形状: (BK,C,R,R,R)
                model_out = vel_neg + float(guidance_scale) * (vel_pos - vel_neg)  # 形状: (BK,C,R,R,R)
            else:
                model_out = dense_dit(latents_cur, t_tensor, cond_b)  # 形状: (BK,C,R,R,R)

            # 单步 SDE/ODE
            t_prev = sched.timesteps[idx_t + 1]  # 形状: ()
            gen = (generator if (not bool(deterministic)) else None)  # 形状: 可能为 None
            deterministic_step = bool(deterministic)  # 形状: ()

            latents_next, log_prob_vec, prev_mean, std_vec = direct3d_flow_step_with_logprob_dense(
                scheduler=sched,
                sample=latents_cur,
                model_output=model_out,
                timestep=float(t),
                prev_timestep=float(t_prev),
                generator=gen,
                deterministic=deterministic_step,
                noise_level=float(noise_level),
            )  # 形状: ((BK,C,R,R,R),(BK,), (BK,C,R,R,R), (BK,))

            latents_cur = latents_next  # 形状: (BK,C,R,R,R)
            latents_seq_dense.append(latents_cur.detach().cpu())  # 形状: (BK,C,R,R,R)
            log_prob_seq_dense_rows.append(log_prob_vec.detach().cpu())  # 形状: (BK,)

        # 最终 latent → 稀疏坐标（参考管线 dense decode 返回 index）
        latents_scaled = 1.0 / self.ref.dense_vae.latents_scale * latents_cur + self.ref.dense_vae.latents_shift  # 形状: (BK,C,R,R,R)
        with torch.no_grad():
            indices = self.ref.dense_vae.decode_mesh(latents=latents_scaled, return_index=True)  # 形状: List(len=BK)
        coords_list: List[torch.Tensor] = []  # 形状: 列表(len=BK)
        sparse_dit_module = self._resolve_sparse_dit_module()  # 形状: ()
        for i in range(BK):  # 形状: ()
            latent_index_64 = torch.as_tensor(indices[i]).to(dtype=torch.int64)  # 形状: (N_i,4)
            latent_index_64[:, 1:] = torch.clamp(latent_index_64[:, 1:], 0, 63)  # 形状: (N_i,4)
            latent_index_64 = torch.unique(latent_index_64, dim=0)  # 形状: (N'_i,4)
            latent_index_64 = sort_block(latent_index_64, sparse_dit_module.selection_block_size)  # 形状: (N'_i,4)
            coords_list.append(latent_index_64)  # 形状: 追加 (N'_i,4)

        t_seq_all = torch.cat([sched.timesteps[:-1], sched.timesteps[-1:]]).to(dtype=torch.float32).cpu()  # 形状: (steps+1,)
        log_prob_seq_dense = torch.stack(log_prob_seq_dense_rows, dim=0) if len(log_prob_seq_dense_rows) > 0 else torch.empty((0, BK))  # 形状: (steps, BK)
        return coords_list, latents_seq_dense, log_prob_seq_dense, t_seq_all
