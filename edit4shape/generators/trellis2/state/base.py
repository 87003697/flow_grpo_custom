# =====================================================================
# Imports
# =====================================================================
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from edit4shape.systems.base import BaseState

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
    
    扩展 BaseState 以支持 TRELLIS.2 的双阶段生成：
    - shape_slat: 几何阶段的稀疏潜变量
    - tex_slat: 纹理阶段的稀疏潜变量
    - subs: 解码中间结果（用于 tex 解码）
    
    属性说明:
        coords (torch.Tensor): 稀疏结构坐标，形状 (N, 4)。
                               N 为总点数 (batch_size * num_points)。
                               第 0 列为 batch 索引，后 3 列为 (x, y, z) 坐标。
        
        features (Trellis2State.Features): 特征容器。
            - shape_slat (SparseTensor): Shape 阶段输出的稀疏特征
            - tex_slat (SparseTensor): Tex 阶段输出的稀疏特征
            - subs (List[SparseTensor]): Shape 解码中间结果
        
        cameras (BaseState.Cameras): 相机参数容器。
            - c2w (torch.Tensor): (B, V, 4, 4) 相机到世界变换矩阵。
            - w2c (torch.Tensor): (B, V, 4, 4) 世界到相机变换矩阵。
            - intrinsics (torch.Tensor): (B, V, 3, 3) 内参矩阵。
            
        views_conditioned (Trellis2State.ViewsConditioned): 条件信息容器（覆盖基类，支持双分辨率）。
            - image_pils (List[PIL.Image]): 输入的条件图像列表。
            - cond_512_embed (torch.Tensor): (B, S, C) 512 分辨率条件嵌入（Dense Sampling 使用）。
            - uncond_512_embed (torch.Tensor): (B, S, C) 512 分辨率无条件嵌入。
            - cond_1024_embed (torch.Tensor): (B, S, C) 1024 分辨率条件嵌入（Shape/Tex Rollout 使用）。
            - uncond_1024_embed (torch.Tensor): (B, S, C) 1024 分辨率无条件嵌入。
            
        views_generated (Trellis2State.ViewsGenerated): 双阶段生成结果容器。
            - shape_tensor (torch.Tensor): (B, V, H, W, C) Shape 阶段 Normal 图
            - pbr_tensor (torch.Tensor): (B, V, H, W, C) Tex 阶段 PBR shaded 图
            
        views_edited (BaseState.ViewsEdited): 编辑结果容器。
            - image_tensor (torch.Tensor): (B, V, C, H, W) 经过 Guidance 编辑后的图像。
            
        regularization (Trellis2State.Regularization): 正则化信息容器。
            - reg_loss: 正则化 loss（用于反向传播）
            - reg_metric: 正则化 metric（用于日志记录）
            
        guidance (Trellis2State.Guidance): Guidance 结果容器。
            - loss: 主 loss（可直接 backward）
            - loss_dict: 细分 loss 字典（用于日志）
    """
    
    @dataclass
    class Features:
        """特征容器。存储 Shape 和 Tex 阶段的稀疏特征。"""
        # Denormalized 版本（用于 decode）
        shape_slat: Any = None      # SparseTensor, Shape 阶段输出（denormalized）
        tex_slat: Any = None        # SparseTensor, Tex 阶段输出（denormalized）
        # Normalized 版本（用于作为条件输入其他模型）
        shape_slat_norm: Any = None # SparseTensor, Shape latent（normalized）
        tex_slat_norm: Any = None   # SparseTensor, Tex latent（normalized）
        # 解码中间结果
        subs: Any = None            # List[SparseTensor], Shape 解码中间结果
        meshes: Any = None          # List[Mesh], Shape 解码输出的 mesh
    
    @dataclass
    class Regularization:
        """正则化信息容器。存储 VSD/KL 正则化的 loss 和 metric。"""
        reg_loss: Any = None    # 正则化 loss（用于反向传播）
        reg_metric: Any = None  # 正则化 metric（用于日志记录）
    
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
    features: Features = field(default_factory=Features)
    regularization: Regularization = field(default_factory=Regularization)
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
        if self.features.meshes is None:
            return self
        
        for mesh in self.features.meshes:
            if mesh.faces.shape[0] > max_faces:
                with torch.no_grad():
                    mesh.simplify(max_faces)
        
        return self

