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
from typing import Any, List, Optional, Tuple, Dict

import torch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DIRECT3D_S2_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "Direct3D-S2")
if _DIRECT3D_S2_ROOT not in sys.path:
        sys.path.append(_DIRECT3D_S2_ROOT)

from direct3d_s2.modules import sparse as sp  # 稀疏张量结构
from direct3d_s2.utils import (
        sort_block,
)
from direct3d_s2.pipeline import Direct3DS2Pipeline as _RefPipeline

from .direct3d_s2_sde_with_logprob import sde_step_with_logprob
@dataclass
class PipelineOptions:
    """最小配置，保持与 Trellis Stage2 行为一致。"""
    skip_refiner: bool = False  # 是否跳过 refiner


@dataclass
class SparseStageConfig:
    steps: int = 30
    guidance_scale: float = 0.0
    use_sde: bool = True
    sigma_min: float = 0.002
    rescale_t: float = 0.5
    mc_threshold: float = 0.2


class Direct3DS2PipelineWithLogProb:
    """Direct3D‑S2 最小 GRPO 包装。"""

    def __init__(self, ref_pipeline: _RefPipeline, opts: Optional[PipelineOptions] = None):
        """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:23-53`（参考管线的构造与成员挂载）"""
        self.ref = ref_pipeline
        self.dtype = getattr(ref_pipeline, "dtype", torch.float16)
        self.device = getattr(ref_pipeline, "device", torch.device("cpu"))
        if opts is None:
            opts = PipelineOptions(
                skip_refiner = os.environ.get("DIRECT3D_SKIP_REFINER", "0") == "1",
            )
        self.opts = opts


    @classmethod
    def from_pretrained(
        cls,
        pipeline_path: str,
        subfolder: str = "direct3d-s2-v-1-1",
        dtype: torch.dtype = torch.float16,
        minimal_512_only: bool = True,
        opts: Optional[PipelineOptions] = None,
    ) -> "Direct3DS2PipelineWithLogProb":
        """构建 Direct3D‑S2 参考管线包装（512-only）。参数 `minimal_512_only` 保留仅为兼容，不影响行为。"""
        ref = cls._build_1024_only(pipeline_path, dtype, opts=opts)
        return cls(ref, opts=opts)

    @staticmethod
    def _build_1024_only(pipeline_path: str, dtype: torch.dtype, opts: Optional[PipelineOptions] = None) -> _RefPipeline:
        """构建 dense + sparse512（禁用 1024 分支）。"""
        from omegaconf import OmegaConf
        from direct3d_s2.utils import instantiate_from_config

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

        skip_refiner_flag = (opts.skip_refiner if opts else (os.environ.get('DIRECT3D_SKIP_REFINER', '0') == '1'))
        if skip_refiner_flag:
            class _RefinerStub(torch.nn.Module):
                def run(self, *a, **k):
                    return a
            refiner = _RefinerStub()
        else:
            refiner = load_single(cfg.refiner, 'model_refiner.ckpt', 'refiner')

        dense_image_encoder = build(cfg.dense_image_encoder).eval()
        sparse_image_encoder = build(cfg.sparse_image_encoder).eval()

        # 统一权重 dtype，避免 Half/BFloat16 混用
        modules_to_cast = [
            dense_vae,
            dense_dit,
            sparse_vae_512,
            sparse_dit_512,
            dense_image_encoder,
            sparse_image_encoder,
        ]
        dense_scheduler = build(cfg.dense_scheduler)
        sparse_scheduler_512 = build(cfg.sparse_scheduler_512)

        class _NullModule(torch.nn.Module):
            def forward(self, *a, **k):
                raise RuntimeError('1024-stage disabled in 512-only mode')

        # 按目标 dtype 将模块转换，保持一致
        for m in modules_to_cast:
            if hasattr(m, 'to'):
                m.to(dtype=dtype)
        if not skip_refiner_flag and hasattr(refiner, 'to'):
            refiner.to(dtype=dtype)

        # 将归一化层保持在 float32，避免 LayerNorm32 中 x.float() 触发权重/输入 dtype 冲突
        _norm_names = {"LayerNorm32", "GroupNorm32", "ChannelLayerNorm32"}
        def _recast_norms(mod):
            for sub in mod.modules():
                if sub.__class__.__name__ in _norm_names:
                    sub.to(dtype=torch.float32)
        for mm in modules_to_cast:
            _recast_norms(mm)
        if not skip_refiner_flag:
            _recast_norms(refiner)
            # 仅对 refiner 精确放宽：将 GroupNorm 强制为 float32，避免 CUDA group_norm half 权重报错
            for sub in refiner.modules():
                if isinstance(sub, torch.nn.GroupNorm):
                    sub.to(dtype=torch.float32)

        # 覆盖参考模块中的 dtype 标志，避免内部将张量强制转换为半精度
        for mm in (dense_dit, sparse_dit_512):
            if hasattr(mm, 'dtype'):
                mm.dtype = dtype
            if hasattr(mm, 'use_fp16'):
                mm.use_fp16 = (dtype == torch.float16)

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
        ref._skip_refiner = skip_refiner_flag
        return ref

    def to(self, device: str) -> None:
        """参考：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:54-66`（将各子模块迁移到 device）"""
        self.ref.to(device)
        self.device = torch.device(device)

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
        image_tensor = self.ref.prepare_image(image)  # (B,C,H,W)
        image_tensor = image_tensor.to(dtype=self.dtype)  # (B,C,H,W)
        cond, uncond = self.ref.encode_image(
            image_tensor,
            self.ref.sparse_image_encoder,
            do_classifier_free_guidance=bool(do_classifier_free_guidance),
            use_mask=self.ref.sparse_dit_512.sparse_conditions,
        )
        return cond, (uncond if do_classifier_free_guidance else None)

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
        latent_index_64 = sort_block(latent_index_64, self.ref.sparse_dit_512.selection_block_size)  # (N',4)
        return latent_index_64

    # ------------------------------
    # Minimal helpers
    # ------------------------------
    # （简化）移除额外包装：直接在调用处使用参考调度器与解码器

    def _sample_sparse_candidates(
        self,
        num_candidates: int,
        sparse_cfg: SparseStageConfig,
        generator: Optional[torch.Generator],
        cond: torch.Tensor,
        uncond: Optional[torch.Tensor],
        latent_index_override: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """最小化稀疏候选采样：显式传入 latent_index 与 cond/uncond。"""

        # 直接使用外部提供的 latent_index
        if latent_index_override is None:
            raise ValueError("_sample_sparse_candidates 需要外部提供 latent_index_override")
        latent_index = latent_index_override  # (N,4)
        coords_int = latent_index.int()  # (N,4)

        meshes: List[Any] = []
        logps_all: List[torch.Tensor] = []
        kl_all: List[torch.Tensor] = []

        # 绑定调度器并设置时间步
        sched = self.ref.sparse_scheduler_512  # 调度器对象
        sched.set_timesteps(int(sparse_cfg.steps), device=self.device)
        timesteps = sched.timesteps  # (T,)

        # 采样（无梯度图，省显存）
        with torch.no_grad():
            for k in range(int(num_candidates)):
                latent_shape = (int(latent_index.shape[0]), int(self.ref.sparse_dit_512.out_channels))  # (N,C)
                latents = torch.randn(latent_shape, dtype=self.dtype, device=self.device, generator=generator)  # (N,C)
                step_log_probs = []
                gs = float(sparse_cfg.guidance_scale)  # 标量
                for i, t in enumerate(timesteps):
                    t_tensor = latents.new_tensor([t])  # (1,)
                    x_sp = sp.SparseTensor(latents, coords_int)  # (N,C) feats + (N,4) coords
                    noise_cond = self.ref.sparse_dit_512(x_sp, t_tensor, cond).feats  # (N,C)
                    noise_uncond = (self.ref.sparse_dit_512(x_sp, t_tensor, uncond).feats if uncond is not None else None)  # (N,C) 或 None
                    noise = (noise_cond if noise_uncond is None else noise_uncond + gs * (noise_cond - noise_uncond))  # (N,C)
                    prev_mean = sched.step(noise, float(t_tensor.item()), latents, generator=generator).prev_sample  # (N,C)
                    if sparse_cfg.use_sde:
                        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else t  # 标量
                        t_cur_f32 = latents.new_tensor(float(t), dtype=torch.float32)  # 标量 ()
                        t_prev_f32 = latents.new_tensor(float(t_prev), dtype=torch.float32)  # 标量 ()
                        prev_sample, log_prob_1, _, _, _ = sde_step_with_logprob(
                            prev_mean=prev_mean,  # (N,C)
                            t_cur=t_cur_f32,  # 标量 ()
                            t_prev=t_prev_f32,  # 标量 ()
                            rescale_t=float(sparse_cfg.rescale_t),  # 标量 ()
                            sigma_min=float(sparse_cfg.sigma_min),  # 标量 ()
                            generator=generator,
                            deterministic=False,
                            prev_sample=None,
                        )
                        latents_next = prev_sample  # (N,C)
                        step_logp = log_prob_1[0].to(torch.float32)  # 标量 ()
                    else:
                        latents_next = prev_mean  # (N,C)
                        step_logp = torch.zeros((), device=self.device, dtype=torch.float32)  # 标量 ()
                    step_log_probs.append(step_logp)  # 标量 ()
                    latents = latents_next  # (N,C)

                latents_scaled = 1.0 / self.ref.sparse_vae_512.latents_scale * latents + self.ref.sparse_vae_512.latents_shift  # (N,C)
                lat_sp = sp.SparseTensor(latents_scaled, coords_int)  # (N,C)
                with torch.no_grad():
                    # 按官方流程：先获取重建特征，再经 refiner 生成 mesh
                    reconst_feat = self.ref.sparse_vae_512.decode_mesh(latents=lat_sp, return_feat=True)  # SparseTensor/tuple
                    mesh_out = self.ref.refiner.run(*reconst_feat, mc_threshold=float(sparse_cfg.mc_threshold) * 2.0)
                if isinstance(mesh_out, list):
                    mesh = (mesh_out[0] if len(mesh_out) > 0 else None)
                else:
                    mesh = mesh_out
                meshes.append(mesh)
                logps_all.extend(step_log_probs)
                kl_all.extend([torch.zeros_like(x) for x in step_log_probs])

        return meshes, [], logps_all, kl_all

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
        stage1_cond_dict: Optional[dict] = None,
        num_candidates: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Trellis 对齐的 Stage2 入口：外部显式传入 patch 级 cond/neg_cond 与 latent_index。

        要求（与 Trellis 拆分粒度一致）：
        - 必须提供 'cond' 与 'neg_cond'（patch 级条件）。
        - 必须提供 'latent_index'（由 forward_stage1 产出）。

        参考：无同名函数；本函数等价于将参考 `inference(sparse512)` 的核心采样/解码逻辑拆分复用。
        关键片段：`_reference_codes/Direct3D-S2/direct3d_s2/pipeline.py:253-291, 320-341`。
        """
        if not (isinstance(stage1_cond_dict, dict) and ('cond' in stage1_cond_dict) and ('neg_cond' in stage1_cond_dict)):
            raise ValueError("stage2_with_logprob 需要 stage1_cond_dict 包含 'cond' 与 'neg_cond'")
        if 'latent_index' not in stage1_cond_dict:
            raise ValueError("stage2_with_logprob 需要 stage1_cond_dict 包含 'latent_index'（由 forward_stage1 计算）")

        # Sparse 阶段参数（Trellis 的 Stage2 对齐）
        sigma_min = 0.002
        rescale_t = 1000.0
        mc_threshold = 0.2
        use_sde = True
        if isinstance(slat_sampler_params, dict):
            sigma_min = float(slat_sampler_params.get("sigma_min", sigma_min))  # 标量
            rescale_t = float(slat_sampler_params.get("rescale_t", rescale_t))  # 标量
            mc_threshold = float(slat_sampler_params.get("mc_threshold", mc_threshold))  # 标量
            use_sde = bool(slat_sampler_params.get("use_sde", use_sde))  # 标量
        sparse_cfg = SparseStageConfig(
            steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            use_sde=bool(use_sde),
            sigma_min=float(sigma_min),
            rescale_t=float(rescale_t),
            mc_threshold=float(mc_threshold),
        )

        # Dense latent_index：仅使用外部提供，保持拆分粒度一致
        latent_index_override = stage1_cond_dict['latent_index']
        if not isinstance(latent_index_override, torch.Tensor):
            raise ValueError("'latent_index' 必须为 torch.Tensor")

        # 当禁用 SDE 时，直接复用参考管线进行稀疏阶段推理，进一步化简
        if not sparse_cfg.use_sde:
            image_tensor = self.ref.prepare_image(stage1_cond_dict['image_path'])  # (B, C, H, W)
            image_tensor = image_tensor.to(dtype=self.dtype)  # (B, C, H, W)
            meshes: List[Any] = []
            zeros: List[torch.Tensor] = []
            for k in range(int(num_candidates)):
                with torch.no_grad():
                    outs = self.ref.inference(
                        image=image_tensor,  # (B,C,H,W)
                        vae=self.ref.sparse_vae_512,
                        dit=self.ref.sparse_dit_512,
                        conditioner=self.ref.sparse_image_encoder,
                        scheduler=self.ref.sparse_scheduler_512,
                        num_inference_steps=int(sparse_cfg.steps),  # 标量
                        guidance_scale=float(sparse_cfg.guidance_scale),  # 标量
                        generator=generator,
                        latent_index=latent_index_override,  # (N,4)
                        mode='sparse512',
                        mc_threshold=float(sparse_cfg.mc_threshold),  # 标量
                        remove_interior=True,
                    )
                # inference 内部已在 remove_interior=True 情况下调用 refiner，不再重复调用
                meshes.append(outs[0])
                zeros.extend([torch.zeros((), device=self.device, dtype=torch.float32) for _ in range(int(sparse_cfg.steps))])
            return meshes, [], zeros, [torch.zeros_like(z) for z in zeros]

        # 启用 SDE：执行候选采样（显式传入 cond/uncond 与 latent_index）
        meshes, latents_seq_flat, step_log_probs_flat, step_kl_flat = self._sample_sparse_candidates(
            num_candidates=int(num_candidates),
            sparse_cfg=sparse_cfg,
            generator=generator,
            cond=stage1_cond_dict['cond'],
            uncond=stage1_cond_dict['neg_cond'],
            latent_index_override=latent_index_override,
        )

        # 与 Trellis 输出四元组对齐
        return meshes, latents_seq_flat, step_log_probs_flat, step_kl_flat


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
        'latent_index': latent_index,  # (N,4)
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
