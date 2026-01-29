"""
Guidance 模块基础设施。

提供：
- GuidanceResult: 统一的返回格式
- BaseGuidance: 抽象基类，包含共享逻辑
- create_guidance(): 工厂函数，根据配置创建 Guidance 实例

设备分配：
- Guidance 模型默认运行在 训练设备 + 1 的 GPU 上
- 例如：训练在 cuda:0，则 Guidance 在 cuda:1

统一框架（真 Loss 模式）：
- 所有 Guidance 子类遵循相同的 compute_guidance 流程：
    1. 格式转换
    2. 编码到 latent（一次，有梯度）
    3. 调用 Pipeline（无梯度）
    4. 通过 Tracker.loss() 计算真 loss（可直接 backward）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any, List, Dict, Tuple
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from edit4shape.systems.base import compute_guidance_device

if TYPE_CHECKING:
    from edit4shape.guidance.pipelines.qwen_image_edit.trackers import FlowEditStateTracker


# =====================================================================
# GuidanceResult - 统一返回格式
# =====================================================================

@dataclass
class GuidanceResult:
    """
    Guidance 计算结果（通用格式）。
    
    Attributes:
        loss: 主 loss（可直接 backward）
        edited_imgs: 编辑后的图像 (B,V,C,H,W)，FlowEdit 专用
        loss_dict: 细分 loss 字典，用于日志记录
        trackers: FlowEdit 中间状态跟踪器列表（用于多步监督）
    """
    loss: torch.Tensor                                                  # 主 loss（必须）
    edited_imgs: Optional[torch.Tensor] = None                          # (B,V,C,H,W) FlowEdit 专用
    loss_dict: Optional[Dict[str, torch.Tensor]] = field(default=None)  # 细分 loss
    trackers: Optional[List["FlowEditStateTracker"]] = None             # FlowEdit 专用


# =====================================================================
# BaseGuidance - 抽象基类（包含共享逻辑）
# =====================================================================

class BaseGuidance(ABC):
    """
    Guidance 抽象基类（真 Loss 模式）。
    
    提供共享功能：
    - 设备初始化
    - 格式转换（tensor_to_pil, pils_to_tensor）
    - 输入格式处理（_reshape_input）
    - Latent 编码（encode_to_latent）
    - 统一的 compute_guidance 模板方法
    
    子类需要：
    - 设置 loss_key 类属性
    - 初始化 self.pipe（Pipeline 实例）
    - 实现 _run_pipeline() 调用 pipeline
    - 实现 _compute_loss() 计算真 loss（通过 Tracker.loss()）
    - 可选覆盖 _build_result() 自定义返回结果
    """
    
    # 类属性：用于 loss_dict 的 key 名称（子类可覆盖）
    loss_key: str = "guidance"
    
    # 实例属性（子类初始化）
    device: torch.device
    train_device: torch.device
    pipe: Any  # Pipeline 实例
    edit_resolution: int
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        基类初始化（共享逻辑）。
        
        Args:
            cfg: 配置对象
            train_device: 训练设备
        """
        self.cfg = cfg
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        self.edit_resolution = cfg.guidance.edit_resolution
        
        # 子类需要在 __init__ 中初始化 self.pipe
    
    # =========================================================================
    # 格式转换（共享）
    # =========================================================================
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Tensor 转 PIL 图像。
        
        Args:
            tensor: [C, H, W] float [0,1]
        
        Returns:
            PIL.Image
        """
        arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr.transpose(1, 2, 0))
    
    def pils_to_tensor(self, pils: List[Image.Image], size: Tuple[int, int]) -> torch.Tensor:
        """
        PIL 图像列表转 Tensor。
        
        Args:
            pils: List[PIL.Image]
            size: (W, H) 目标尺寸
        
        Returns:
            [N, C, H, W] tensor
        """
        tensors = [TF.to_tensor(p.resize(size, Image.LANCZOS)) for p in pils]
        return torch.stack(tensors).to(self.device)
    
    # =========================================================================
    # 输入格式处理（共享）
    # =========================================================================
    
    def _reshape_input(self, comp_rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int, int, torch.device]:
        """
        统一处理输入格式。
        
        Args:
            comp_rgb: (B, V, H, W, C) 或 (B, V, C, H, W) 或 (N, C, H, W)
        
        Returns:
            (comp_rgb, B, V, C, H, W, source_device)
            其中 comp_rgb 已转换为 [N, C, H, W] 并移动到 self.device
        """
        source_device = comp_rgb.device
        
        if comp_rgb.dim() == 5:
            if comp_rgb.shape[-1] == 3:
                # (B, V, H, W, C) → (N, C, H, W)
                B, V, H, W, C = comp_rgb.shape
                comp_rgb = comp_rgb.permute(0, 1, 4, 2, 3).reshape(B * V, C, H, W)
            else:
                # (B, V, C, H, W) → (N, C, H, W)
                B, V, C, H, W = comp_rgb.shape
                comp_rgb = comp_rgb.reshape(B * V, C, H, W)
        else:
            # (N, C, H, W)
            B, V = 1, 1
            _, C, H, W = comp_rgb.shape
        
        comp_rgb = comp_rgb.to(self.device)
        
        return comp_rgb, B, V, C, H, W, source_device
    
    # =========================================================================
    # Latent 编码（共享）
    # =========================================================================
    
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 packed latent（可微分）。
        
        Args:
            images: [B, C, H, W] float [0,1]
        
        Returns:
            [B, seq, C_lat] packed latent
        """
        B = images.shape[0]
        
        # Resize 到编辑分辨率
        resized = F.interpolate(
            images,
            size=(self.edit_resolution, self.edit_resolution),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )  # [B, C, edit_res, edit_res]
        
        # [0,1] → [-1,1]
        normalized = resized * 2 - 1  # [B, C, H, W]
        images_5d = normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # [B, C, 1, H, W]
        
        # VAE encode（可微分）
        latent_5d = self.pipe._encode_vae_image_differentiable(images_5d)  # [B, C_lat, 1, H_lat, W_lat]
        
        # Pack
        _, C_lat, _, H_lat, W_lat = latent_5d.shape
        latent = self.pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)  # [B, seq, C_lat]
        
        return latent.to(dtype=images.dtype)
    
    # =========================================================================
    # 抽象方法（子类必须实现）
    # =========================================================================
    
    @abstractmethod
    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        B: int,
        V: int,
    ) -> Any:
        """
        调用 Pipeline（无梯度）。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            Pipeline 输出（SDSOutput / CSDOutput / FlowEditPipelineOutput）
        """
        pass
    
    @abstractmethod
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: Any,
        comp_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 loss（真 loss，可直接 backward）。
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: Pipeline 输出（包含 Tracker）
            comp_rgb: [N, C, H, W] 渲染图（部分子类需要）
        
        Returns:
            (loss, loss_dict)
            - loss: 标量 loss（有梯度，可 backward）
            - loss_dict: 各项 loss 字典（用于日志）
        """
        pass
    
    # =========================================================================
    # 主入口（模板方法）
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> GuidanceResult:
        """
        计算 Guidance loss（模板方法，真 Loss 模式）。
        
        统一流程：
            1. 格式转换
            2. 编码到 latent（一次，有梯度）
            3. 调用 Pipeline（无梯度）
            4. 通过 Tracker.loss() 计算真 loss
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C) 或 (B,V,C,H,W)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 额外参数
        
        Returns:
            GuidanceResult: 包含 loss 和可选的 edited_imgs
        """
        # 1. 格式转换
        comp_rgb, B, V, C, H, W, source_device = self._reshape_input(comp_rgb)
        
        # 2. 编码到 latent（一次，有梯度）
        src_latent = self.encode_to_latent(comp_rgb)  # [N, seq, C_lat]
        
        # 3. 调用 Pipeline（无梯度）
        with torch.no_grad():
            pipeline_output = self._run_pipeline(
                comp_rgb,
                condition_images,
                src_latent=src_latent.detach(),
                B=B, V=V,
            )
        
        # 4. 计算 loss（真 loss，有梯度）
        loss, loss_dict = self._compute_loss(
            src_latent,
            pipeline_output,
            comp_rgb,
        )
        
        # 5. 移动到训练设备
        loss = loss.to(self.train_device)
        
        # 6. 返回结果（子类可通过 _build_result 自定义）
        return self._build_result(
            loss, loss_dict, pipeline_output, B, V, C, H, W, source_device
        )
    
    def _build_result(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        pipeline_output: Any,
        B: int, V: int, C: int, H: int, W: int,
        source_device: torch.device,
    ) -> GuidanceResult:
        """
        构建返回结果。子类可覆盖以添加额外字段。
        
        Args:
            loss: 主 loss
            loss_dict: 各项 loss 字典
            pipeline_output: Pipeline 输出
            B, V, C, H, W: 维度信息
            source_device: 原始设备
        
        Returns:
            GuidanceResult
        """
        return GuidanceResult(
            loss=loss,
            edited_imgs=None,
            loss_dict={self.loss_key: loss.detach(), **loss_dict},
            trackers=None,
        )
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        获取各项 loss 的权重。
        
        默认返回 {loss_key: 1.0}，适用于 SDS/CSD/CSD-Rev。
        FlowEdit 需要覆盖此方法，返回细分 loss 权重。
        
        Returns:
            Dict[str, float]: {loss_name: weight}
        """
        return {self.loss_key: 1.0}
    
    def cleanup(self) -> None:
        """释放资源。"""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()


# =====================================================================
# 工厂函数
# =====================================================================

def create_guidance(cfg: Any, train_device: torch.device, use_pp: bool = False) -> BaseGuidance:
    """
    创建 Guidance 实例。
    
    根据 cfg.guidance.type 选择不同的 Guidance 范式：
    - "flowedit": FlowEdit（编辑图像 → 计算相似度 loss）
    - "csd": CSD（Classifier-free Score Distillation）
    - "sds": SDS（Score Distillation Sampling）
    
    Args:
        cfg: 配置对象，需包含 guidance 子配置
        train_device: 训练使用的设备（如 cuda:0）
        use_pp: 是否使用流水线并行版本
    
    Returns:
        BaseGuidance: Guidance 实例
    
    Example:
        >>> guidance = create_guidance(cfg, accelerator.device)
        >>> result = guidance.compute_guidance(comp_rgb, condition_images)
        >>> result.loss.backward()
    """
    paradigm = cfg.guidance.get("type", "flowedit")
    
    if paradigm == "flowedit":
        if use_pp:
            from edit4shape.guidance.paradigms.flowedit import FlowEditGuidancePP
            return FlowEditGuidancePP(cfg, train_device)
        else:
            from edit4shape.guidance.paradigms.flowedit import FlowEditGuidance
            return FlowEditGuidance(cfg, train_device)
    elif paradigm == "distillation":
        from edit4shape.guidance.paradigms.distillation import DistillationGuidance
        return DistillationGuidance(cfg, train_device)
    else:
        raise ValueError(f"Unknown guidance type: {paradigm}. Choose from: flowedit, distillation")
