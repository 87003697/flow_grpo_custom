# =====================================================================
# Imports
# =====================================================================
import contextlib
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from edit4shape.systems.base import BaseState
from .stage_latent import DenseStageLatent, ShapeStageLatent, TexStageLatent

# =====================================================================
# Trellis2State - Trellis2 专用状态类
# =====================================================================
# =====================================================================
# Trellis2State - Trellis2 专用状态类
# =====================================================================

@dataclass
class Trellis2State(BaseState):
    """
    Trellis2 生成过程的状态容器。
    
    扩展 BaseState 以支持 TRELLIS.2 的多阶段生成：
    - dense: Dense Sampling 产物（coords）
    - shape: Shape 阶段 latent + decode 中间产物
    - tex:   Tex 阶段 latent
    
    属性说明:
        dense (DenseStageLatent): Dense Sampling 阶段容器。
            - coords (torch.Tensor): (N, 4) 稀疏坐标
        
        shape (ShapeStageLatent): Shape 阶段容器。
            - z0 (SparseTensor): Shape 阶段输出（denormalized）
            - z0_norm (SparseTensor): Shape latent（normalized, detached）
            - subs (List[SparseTensor]): Shape 解码中间结果
            - meshes (List[Mesh]): Shape 解码输出的 mesh
            - reg_loss: 正则化 loss
            - reg_metric: 正则化 metric
        
        tex (TexStageLatent): Tex 阶段容器。
            - z0 (SparseTensor): Tex 阶段输出（denormalized）
            - z0_norm (SparseTensor): Tex latent（normalized, detached）
            - reg_loss: 正则化 loss
            - reg_metric: 正则化 metric
        
        cameras (BaseState.Cameras): 相机参数容器。
        views_conditioned (Trellis2State.ViewsConditioned): 条件信息容器（双分辨率）。
        views_generated (Trellis2State.ViewsGenerated): 双阶段生成结果容器。
        views_edited (BaseState.ViewsEdited): 编辑结果容器。
        guidance (Trellis2State.Guidance): Guidance 结果容器。
    """
    
    @dataclass
    class Guidance:
        """Guidance 结果容器。存储各 Guidance 类型的 loss。"""
        loss: Any = None                  # 主 loss（可直接 backward）
        loss_dict: Dict[str, Any] = None  # {loss_name: loss_tensor}，统一存放所有 guidance loss
    
    @dataclass
    class ViewsGenerated:
        """双阶段生成结果容器。"""
        shape_tensor: Any = None  # (B, V, H, W, C) Shape 阶段 Normal 图
        pbr_tensor: Any = None    # (B, V, H, W, C) Tex 阶段 PBR shaded 图
    
    @dataclass
    class ViewsConditioned:
        """
        条件视角缓存（覆盖基类，支持双分辨率条件编码）。
        
        对齐 TRELLIS.2 参考实现：
        - Dense Sampling 始终使用 512 分辨率条件编码
        - Shape/Tex Rollout 使用对应 pipeline 分辨率的条件编码
        """
        image_pils: Any = None          # list[len=B] of PIL.Image
        paths: Any = None               # list[len=B] of str
        # 512 分辨率条件编码（Dense Sampling 始终使用）
        cond_512_embed: Any = None      # (B, S, C) 512 分辨率条件嵌入
        uncond_512_embed: Any = None    # (B, S, C) 512 分辨率无条件嵌入
        # 1024 分辨率条件编码（Shape/Tex Rollout 使用）
        cond_1024_embed: Any = None     # (B, S, C) 1024 分辨率条件嵌入
        uncond_1024_embed: Any = None   # (B, S, C) 1024 分辨率无条件嵌入
    
    # batch key -> state 属性的映射（类常量）
    _CAMERA_KEYS: ClassVar[List[str]] = ["c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions"]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]
    
    # ============== Trellis2 专用子状态容器 ==============
    dense: DenseStageLatent = field(default_factory=DenseStageLatent)
    shape: ShapeStageLatent = field(default_factory=ShapeStageLatent)
    tex: TexStageLatent = field(default_factory=TexStageLatent)
    guidance: Guidance = field(default_factory=Guidance)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)  # 覆盖 BaseState
    views_conditioned: ViewsConditioned = field(default_factory=ViewsConditioned)  # 覆盖 BaseState

    def attach_batch(self, batch: Dict[str, Any], pipeline: Any = None, resolution: int = 1024) -> "Trellis2State":
        """
        从数据批次中提取并挂载所有数据到 state。
        
        Args:
            batch: DataLoader 返回的批次数据，包含图像、相机参数等。
            pipeline: 必须提供，用于调用 prepare_image_conditions 从 image_pils 生成条件嵌入。
            resolution: 条件编码分辨率（512 或 1024）
        
        Returns:
            self: 支持链式调用
        """
        # ---- 1. views_conditioned（图像、路径、嵌入） ----
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])
        
        # 从 image_pils 生成双分辨率条件编码（对齐 TRELLIS.2 参考实现）
        if "image_pils" in batch and pipeline is not None:
            # 始终生成 512 分辨率的条件编码（Dense Sampling 始终需要）
            cond_512 = pipeline.prepare_image_conditions(batch["image_pils"], resolution=512)
            self.views_conditioned.cond_512_embed = cond_512["cond"]  # (B, S, C)
            self.views_conditioned.uncond_512_embed = cond_512["neg_cond"]  # (B, S, C)
            
            # 生成目标分辨率的条件编码（用于 Shape/Tex Rollout）
            if resolution == 512:
                # 复用 512 的结果
                self.views_conditioned.cond_1024_embed = self.views_conditioned.cond_512_embed
                self.views_conditioned.uncond_1024_embed = self.views_conditioned.uncond_512_embed
            else:
                # 生成 1024 分辨率
                cond_1024 = pipeline.prepare_image_conditions(batch["image_pils"], resolution=resolution)
                self.views_conditioned.cond_1024_embed = cond_1024["cond"]  # (B, S, C)
                self.views_conditioned.uncond_1024_embed = cond_1024["neg_cond"]  # (B, S, C)
        
        # ---- 2. 指导信号 (Guidance 数据) ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]
        
        # ---- 3. 相机参数 ----
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])
        
        return self
    
    def extract_embeddings(self, resolution: int = 1024) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        提取指定分辨率的条件和无条件嵌入（覆盖基类方法）。
        
        Args:
            resolution: 条件编码分辨率，512 或 1024
        
        Returns:
            tuple: (cond_embed, uncond_embed)
        """
        if resolution == 512:
            cond = self.views_conditioned.cond_512_embed
            uncond = self.views_conditioned.uncond_512_embed
        else:
            cond = self.views_conditioned.cond_1024_embed
            uncond = self.views_conditioned.uncond_1024_embed
        
        if cond is None:
            raise ValueError(f"views_conditioned.cond_{resolution}_embed 未设置，请先调用 attach_batch")
        
        return cond, uncond

    def attach_guidance_result(self, guidance_result: Any) -> "Trellis2State":
        """
        将 GuidanceResult 挂载到 state。
        
        Args:
            guidance_result: GuidanceResult 对象，包含 loss 和可选的 edited_imgs。
        
        Returns:
            self: 支持链式调用
        """
        # Loss 挂载到 guidance
        self.guidance.loss = guidance_result.loss
        self.guidance.loss_dict = guidance_result.loss_dict
        # 编辑后图像和 trackers 挂载到 views_edited（FlowEdit 专用）
        self.views_edited.image_tensor = guidance_result.edited_imgs
        self.views_edited.trackers = guidance_result.trackers
        return self

    def simplify_meshes(self, max_faces: int = 16777216) -> "Trellis2State":
        """
        简化 state 中的 meshes，避免 nvdiffrast 的面片数量限制。
        
        nvdiffrast 的 CUDA 光栅化器最多支持 2^24 = 16,777,216 个三角面片。
        当 mesh 面片数超过限制时，调用 mesh.simplify() 进行简化。
        
        注意：simplify() 是不可微的操作，使用 torch.no_grad() 包裹。
        
        Args:
            max_faces: 最大面片数量，默认 16777216（nvdiffrast 限制）
        
        Returns:
            self: 支持链式调用
        """
        if self.shape.meshes is None:
            return self
        
        for mesh in self.shape.meshes:
            if mesh.faces.shape[0] > max_faces:
                with torch.no_grad():
                    mesh.simplify(max_faces)
        
        return self

    def release_shape_decode_cache(self) -> "Trellis2State":
        """
        释放 Shape 解码阶段的中间产物，回收显存。
        
        Shape decode 产生的 subs / meshes / shape_slat_norm 仅供 Tex 阶段使用：
        - subs: List[SparseTensor], decode_tex 的 guide_subs 参数
        - meshes: List[Mesh], decode_and_render_pbr 的输入
        - shape_slat_norm: SparseTensor, Tex rollout 的条件输入
        
        在 Shape-only 训练中，Phase 2 backward 之后这些数据不再被使用，
        可以安全释放以降低后续阶段（Phase 3）的显存压力。
        
        ⚠️ 调用后 Tex 阶段将不可用。仅在 Shape-only 训练中使用。
        
        Returns:
            self: 支持链式调用
        """
        self.shape.subs = None
        self.shape.meshes = None
        self.shape.z0_norm = None
        return self

    def release_shape_spatial_cache(self) -> "Trellis2State":
        """
        释放 decoder 在 shape_slat._spatial_cache 中累积的 spatial cache。
        
        Decoder 的 sparse conv / spatial2channel 等操作会在 shape.z0._spatial_cache
        中注册 neighbor maps、上下采样索引、subdivision masks 等。
        由于 SparseTensor.replace() 共享同一个 dict 引用，这些缓存全部累积在
        shape.z0._spatial_cache 中，可达 ~20-40 GiB。
        
        VJP 阶段只使用 SLatFlowModel（纯 transformer），不需要这些 decoder 缓存。
        layout / spatial_shape 等属性会在需要时从 coords 惰性重算，开销可忽略。
        
        ⚠️ 调用后 decoder 若再次 forward，需要重新构建 neighbor maps。
        
        ★ 使用 dict.clear() 原地清空，而非 clear_spatial_cache() 的 rebind（= {}）。
          replace() / SparseDownsample / SparseSpatial2Channel 等操作使所有派生
          SparseTensor 共享同一个 _spatial_cache dict 引用。rebind 只影响自身，
          in-place clear 才能让所有共享者同时看到空 dict，真正释放 neighbor maps。
        
        Returns:
            self: 支持链式调用
        """
        if self.shape.z0 is not None:
            self.shape.z0._spatial_cache.clear()
        return self

    def release_tex_spatial_cache(self) -> "Trellis2State":
        """
        释放 tex decoder 在 tex.z0._spatial_cache 中累积的 spatial cache。
        
        与 release_shape_spatial_cache 对称，清理 tex decoder 的 neighbor maps、
        上下采样索引等。VJP 阶段只使用 SLatFlowModel（纯 transformer），
        不需要这些 decoder 缓存。
        
        ★ 使用 dict.clear() 原地清空（同 release_shape_spatial_cache 的原因）。
        
        Returns:
            self: 支持链式调用
        """
        if self.tex.z0 is not None:
            self.tex.z0._spatial_cache.clear()
        return self

    def prepare_for_shape_vjp(self, *, keep_decode_cache: bool = False) -> "Trellis2State":
        """
        Shape P2→P3 过渡：释放 VJP 不需要的 decode 缓存，降低显存水位。
        
        P3 VJP 只需 shape.z0.coords（通过 .replace() 构建 x_t）+ cond_embeds。
        spatial_cache（neighbor maps，~20-40 GiB）和 decode 产物（subs/meshes/z0_norm）
        在 P3 期间完全不被访问，可以安全释放。
        
        Args:
            keep_decode_cache: True 时保留 subs/meshes/shape_slat_norm（供后续 Tex 阶段使用）。
                               shape-only 训练默认 False；shape_tex 双阶段训练传 True。
        
        Returns:
            self: 支持链式调用
        """
        self.release_shape_spatial_cache()
        if not keep_decode_cache:
            self.release_shape_decode_cache()
        torch.cuda.empty_cache()
        return self

    def prepare_for_tex_vjp(self) -> "Trellis2State":
        """
        Tex P2→P3 过渡：释放 VJP 不需要的 decode 缓存和几何，降低显存水位。
        
        P3 VJP 只需 tex.z0.coords（通过 .replace() 构建 x_t）
        + cond_embeds + shape.z0_norm（作为 tex flow model 的 concat_cond）。
        tex spatial_cache、subs、meshes 在 P3 期间完全不被访问。
        
        Returns:
            self: 支持链式调用
        """
        self.release_tex_spatial_cache()
        self.shape.subs = None
        self.shape.meshes = None
        torch.cuda.empty_cache()
        return self

    def detach_features(self) -> "Trellis2State":
        """切断 shape/tex z0 上的 autograd proxy chain（就地 detach）。"""
        if self.shape.z0 is not None:
            self.shape.z0 = self.shape.z0.detach()
        if self.tex.z0 is not None:
            self.tex.z0 = self.tex.z0.detach()
        return self

    def release_uncond_embeddings(self) -> "Trellis2State":
        """释放无条件嵌入（CFG 完成后不再需要）。"""
        self.views_conditioned.uncond_512_embed = None
        self.views_conditioned.uncond_1024_embed = None
        return self

    def offload_decode_cache_to_cpu(self) -> "Trellis2State":
        """
        将 subs/meshes 搬到 CPU，降低 Shape VJP 期间的 GPU 显存水位。

        ★ 前提：shape spatial_cache 已 clear（否则 SparseTensor.to() 会遗留跨设备缓存）
        ★ Tex P2-grad 使用前必须调用 reload_decode_cache_to_gpu() 搬回。

        显存节省估算：subs ~50-200 MiB, meshes ~50-100 MiB → 总计 ~100-300 MiB。
        CPU↔GPU 传输耗时 ~4-12 ms（PCIe Gen4），相对 VJP ~30s 可忽略。
        """
        if self.shape.subs is not None:
            self.shape.subs = [sub.to("cpu") for sub in self.shape.subs]
        if self.shape.meshes is not None:
            self.shape.meshes = [mesh.to("cpu") for mesh in self.shape.meshes]
        return self

    def reload_decode_cache_to_gpu(self, device: torch.device) -> "Trellis2State":
        """
        将 CPU 上的 subs/meshes 搬回 GPU，供 Tex P2-grad 使用。

        ★ 与 offload_decode_cache_to_cpu() 配对使用。
        ★ 如果 subs/meshes 已被释放（None），则跳过。
        """
        if self.shape.subs is not None:
            self.shape.subs = [sub.to(device) for sub in self.shape.subs]
        if self.shape.meshes is not None:
            self.shape.meshes = [mesh.to(device) for mesh in self.shape.meshes]
        return self

    def offload_vis_to_cpu(self) -> "Trellis2State":
        """将可视化 tensor 搬到 CPU（save_shape_train 在 CPU 也能工作）。"""
        if self.views_generated.shape_tensor is not None:
            self.views_generated.shape_tensor = self.views_generated.shape_tensor.cpu()
        if self.views_generated.pbr_tensor is not None:
            self.views_generated.pbr_tensor = self.views_generated.pbr_tensor.cpu()
        if self.views_edited.image_tensor is not None:
            self.views_edited.image_tensor = self.views_edited.image_tensor.cpu()
        return self

    @contextlib.contextmanager
    def override_embeddings(self, cond_embed: torch.Tensor, resolution: int = 1024):
        """
        临时替换指定分辨率的 cond embed，yield 后恢复原值。

        contrastive 训练中，teacher 需要用 tgt_embed（编辑后图片的 DINOv3
        编码）做去噪，而 student velocity 用的是原 src_embed。本上下文管理器
        临时替换 cond，调用 predict_cfg_velocity_teacher 后自动恢复。

        ★ uncond 不变 — teacher denoise 的 CFG 仍用原 uncond。

        Args:
            cond_embed: 替换用的条件嵌入 (B, S, C)
            resolution: 512 或 1024
        """
        vc = self.views_conditioned
        if resolution == 512:
            orig = vc.cond_512_embed
            vc.cond_512_embed = cond_embed
            try:
                yield
            finally:
                vc.cond_512_embed = orig
        else:
            orig = vc.cond_1024_embed
            vc.cond_1024_embed = cond_embed
            try:
                yield
            finally:
                vc.cond_1024_embed = orig

    @contextlib.contextmanager
    def disable_uncond_embeddings(self, disable: bool = True):
        """临时清空 uncond embed，跳过 uncond forward pass。

        disable=True 时清空 512/1024 双路 uncond；disable=False 时 no-op。
        用于 student_denoise_cfg=False 或 teacher_cfg=False 场景。
        """
        if not disable:
            yield
            return
        vc = self.views_conditioned
        orig_512 = vc.uncond_512_embed
        orig_1024 = vc.uncond_1024_embed
        vc.uncond_512_embed = None
        vc.uncond_1024_embed = None
        try:
            yield
        finally:
            vc.uncond_512_embed = orig_512
            vc.uncond_1024_embed = orig_1024
