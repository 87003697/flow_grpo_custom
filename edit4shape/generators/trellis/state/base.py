"""
TrellisState - Trellis 生成过程的状态容器

存储整个生成流程中的所有中间状态，支持:
- 稀疏结构坐标 (coords)
- 稀疏特征 (features)
- 相机参数 (cameras)
- 条件信息 (views_conditioned)
- 生成结果 (views_generated)
- 编辑结果 (views_edited)
- 正则化 (regularization)
- 指导信号 (guidance)
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional
import torch

from edit4shape.systems.base import BaseState


@dataclass
class TrellisState(BaseState):
    """
    Trellis 生成过程的状态容器。
    
    存储整个生成流程中的所有中间状态，包括：
    - 稀疏结构坐标 (coords)
    - 稀疏特征 (features)
    - 相机参数 (cameras)
    - 条件信息 (views_conditioned)
    - 生成结果 (views_generated)
    - 编辑结果 (views_edited)
    - 正则化 (regularization)
    - 指导信号 (guidance)
    
    属性说明:
        coords (torch.Tensor): 稀疏结构坐标，形状 (N, 4)。
                               N 为总点数 (batch_size * num_points)。
                               第 0 列为 batch 索引，后 3 列为 (x, y, z) 坐标。
        
        features (TrellisState.Features): 特征容器。
            - slat (SparseTensor): SLAT 阶段输出的稀疏特征，形状 (N, C)。
        
        cameras (BaseState.Cameras): 相机参数容器。
            - c2w (torch.Tensor): (B, V, 4, 4) 相机到世界变换矩阵。
            - w2c (torch.Tensor): (B, V, 4, 4) 世界到相机变换矩阵。
            - intrinsics (torch.Tensor): (B, V, 3, 3) 内参矩阵。
            
        views_conditioned (BaseState.ViewsConditioned): 条件信息容器。
            - image_pils (List[PIL.Image]): 输入的条件图像列表。
            - cond_embed (torch.Tensor): (B, S, C) 条件嵌入 (CLIP/DINOv2)。
            - uncond_embed (torch.Tensor): (B, S, C) 无条件嵌入 (用于 CFG)。
            
        views_generated (TrellisState.ViewsGenerated): 生成结果容器。
            - image_tensor (torch.Tensor): (B, V, H, W, C) GS Color 渲染结果
            - normal_tensor (torch.Tensor): (B, V, H, W, C) Mesh Normal 渲染（Hybrid 用）
            
        views_edited (TrellisState.ViewsEdited): 编辑结果容器（覆盖基类）。
            - color_tensor (torch.Tensor): (B, V, C, H, W) GS Color edit
            - color_trackers: FlowEdit trackers（GS Color 路）
            - normal_tensor (torch.Tensor): (B, V, C, H, W) Mesh Normal edit（Hybrid 用）
            - normal_trackers: FlowEdit trackers（Mesh Normal 路，Hybrid 用）
            
        regularization (TrellisState.Regularization): 正则化信息容器。
            - reg_loss: 正则化 loss（用于反向传播和日志记录）
            
        guidance (TrellisState.Guidance): Guidance 结果容器。
            - loss: 主 loss（可直接 backward）
            - loss_dict: 细分 loss 字典（用于日志）
    """
    
    @dataclass
    class Features:
        """特征容器。存储各阶段的稀疏特征。"""
        slat: Any = None  # SparseTensor, SLAT 阶段输出的稀疏特征
    
    @dataclass
    class Regularization:
        """正则化信息容器。存储 VSD/KL 正则化的 loss。"""
        reg_loss: Any = None    # 正则化 loss（用于反向传播和日志记录）
    
    @dataclass
    class Guidance:
        """Guidance 结果容器。"""
        loss: Any = None                  # 主 loss（可直接 backward）
        loss_dict: Any = None             # 细分 loss 字典（用于日志）
    
    @dataclass
    class ViewsEdited:
        """编辑结果容器（覆盖基类，支持 Hybrid 双路渲染）。

        - color_tensor / color_trackers: GS Color edit + trackers
        - normal_tensor / normal_trackers: Mesh Normal edit + trackers（Hybrid 用）

        单路模式下只使用 color_tensor / color_trackers。
        attach_guidance_result 默认写入 color_* 字段。
        """
        color_tensor: Any = None         # (B, V, C, H, W) GS Color edit
        color_trackers: Any = None       # List[StateTracker]
        normal_tensor: Any = None        # (B, V, C, H, W) Mesh Normal edit（Hybrid 用）
        normal_trackers: Any = None      # List[StateTracker]（Hybrid 用）

    @dataclass
    class ViewsGenerated:
        """生成视角缓存（覆盖基类，支持 Hybrid 双路渲染）。

        - image_tensor: 单路 GS Color 渲染结果，或 Hybrid 模式下的 GS Color。
        - normal_tensor: Hybrid 模式下的 Mesh Normal 渲染结果。
        """
        image_tensor: Any = None   # (B, V, H, W, C) GS Color 渲染结果
        normal_tensor: Any = None  # (B, V, H, W, C) Mesh Normal 渲染结果（Hybrid 用）
    
    # batch key -> state 属性的映射（类常量）
    _CAMERA_KEYS: ClassVar[List[str]] = ["c2w", "w2c", "mvp", "positions", "intrinsics", "light_positions"]
    _VIEWS_COND_KEYS: ClassVar[List[str]] = ["image_pils", "paths"]
    
    # ============== Trellis 专用子状态容器 ==============
    features: Features = field(default_factory=Features)
    regularization: Regularization = field(default_factory=Regularization)
    guidance: Guidance = field(default_factory=Guidance)
    views_generated: ViewsGenerated = field(default_factory=ViewsGenerated)  # 覆盖 BaseState
    views_edited: ViewsEdited = field(default_factory=ViewsEdited)           # 覆盖 BaseState

    def attach_batch(self, batch: Dict[str, Any], pipeline: Any = None) -> "TrellisState":
        """
        从数据批次中提取并挂载所有数据到 state。
        
        Args:
            batch: DataLoader 返回的批次数据，包含图像、相机参数等。
            pipeline: 必须提供，用于调用 prepare_image_conditions 从 image_pils 生成条件嵌入。
        
        Returns:
            self: 支持链式调用
        """
        # ---- 1. views_conditioned（图像、路径、嵌入） ----
        for key in self._VIEWS_COND_KEYS:
            if key in batch:
                setattr(self.views_conditioned, key, batch[key])
        
        # 从 image_pils 生成条件编码
        if "image_pils" in batch and pipeline is not None:
            # pipeline.prepare_image_conditions 负责预处理图像（如去背、缩放）并计算 Embedding
            cond = pipeline.prepare_image_conditions(batch["image_pils"])
            self.views_conditioned.cond_embed = cond["cond"]
            # 如果没有 neg_cond，使用全零张量作为无条件嵌入
            self.views_conditioned.uncond_embed = cond.get("neg_cond", torch.zeros_like(cond["cond"]))
        
        # ---- 2. 指导信号 (Guidance 数据) ----
        if "Guidances" in batch:
            self.guidances_data = batch["Guidances"]
        
        # ---- 3. 相机参数 ----
        for key in self._CAMERA_KEYS:
            if key in batch:
                setattr(self.cameras, key, batch[key])
        
        return self

    def attach_guidance_result(self, guidance_result: Any) -> "TrellisState":
        """
        将 GuidanceResult 挂载到 state。

        默认写入 color_tensor / color_trackers（GS Color 路）。
        Hybrid 模式下，mesh pass 的 edit 由调用方搬运到 normal_tensor / normal_trackers。

        Args:
            guidance_result: GuidanceResult 对象，包含 loss 和可选的 edited_imgs。
        
        Returns:
            self: 支持链式调用
        """
        # Loss 挂载到 guidance
        self.guidance.loss = guidance_result.loss
        self.guidance.loss_dict = guidance_result.loss_dict
        # 编辑后图像和 trackers 写入 color_* 缓冲区（默认）
        self.views_edited.color_tensor = guidance_result.edited_imgs
        self.views_edited.color_trackers = guidance_result.trackers
        return self

    def prepare_for_vjp(self) -> "TrellisState":
        """
        Phase 2→3 过渡清理：释放 decode/render 中间产物，降低 VJP 阶段显存水位。

        VJP 只需要：
        - features.slat.coords（通过 .replace() 构建 x_t）
        - views_conditioned（条件编码）
        其余 decode 产物、spatial_cache 均可释放。

        Returns:
            self: 支持链式调用
        """
        # SparseTensor._spatial_cache 存储 neighbor maps / window partition indices 等
        # 使用 .clear() 做 in-place 清空：replace() 会共享同一个 dict 引用，
        # in-place 才能让所有共享方都释放缓存
        if self.features.slat is not None:
            self.features.slat._spatial_cache.clear()
        torch.cuda.empty_cache()
        return self