"""
FlowEdit Guidance 模块。

将 Qwen-Image-Edit 模型加载到 Guidance 设备上，
与 Trellis 训练进程共存于同一 Python 进程。

数据流（继承自 BaseGuidance，真 Loss 模式）：
    1. 格式转换（父类）
    2. 编码到 latent（父类，一次）
    3. 调用 FlowEdit Pipeline（_run_pipeline，多步编辑）
    4. 通过 Tracker.loss() 计算真 loss（_compute_loss）
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image

import torch
import torch.nn.functional as F

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipeline_parallel import PipelineParallelMixin
from edit4shape.guidance.pipelines.adapters import create_pipeline_adapter
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import FlowEditStateTracker
from edit4shape.guidance.metric import create_metrics


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EditOutput:
    """单张图编辑输出"""
    image: Image.Image                  # 编辑后的图像
    latent: torch.Tensor                # [1, seq, C] 编辑后的 latent
    tracker: FlowEditStateTracker       # 中间状态跟踪器


@dataclass
class FlowEditPipelineOutput:
    """FlowEdit Pipeline 输出（与 CSDOutput/SDSOutput 对齐）"""
    edited_images: List[Image.Image]    # [N] 编辑后的图像
    edited_tensor: torch.Tensor         # [N, C, H, W] 编辑后的图像 tensor
    latent_after: torch.Tensor          # [N, seq, C] 编辑后的 latent
    trackers: List[FlowEditStateTracker]  # [N] 中间状态跟踪器


# =============================================================================
# FlowEditGuidance
# =============================================================================

class FlowEditGuidance(BaseGuidance):
    """
    FlowEdit Guidance（继承 BaseGuidance，真 Loss 模式）。
    
    数据流（统一框架）：
        1. 格式转换（父类）
        2. 编码到 latent（父类，一次）
        3. 调用 FlowEdit Pipeline（_run_pipeline）
        4. 计算 loss（_compute_loss）
    
    特有功能：
        - 多步编辑过程
        - 编辑后图像可视化
        - 多种 loss（latent_mse, ssim, lpips, dino）
        - 中间状态追踪（trackers）
    """
    
    # 类属性：用于标识 Guidance 类型
    loss_key = "flowedit"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 FlowEdit Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        super().__init__(cfg, train_device)
        
        # FlowEdit 专属配置
        self.flowedit_cfg = cfg.guidance.flowedit
        self.loss_cfg = cfg.guidance.flowedit.loss
        
        # Loss 配置（分离聚合方式和归一化方式）
        self.reduce_mode = cfg.guidance.flowedit.reduce_mode
        self.ada_normalize = cfg.guidance.flowedit.ada_normalize
        self.ada_eps = cfg.guidance.flowedit.ada_eps
        
        # Loss 权重配置（统一从 loss 子配置读取）
        self.latent_csd_weight = self.loss_cfg.latent_csd
        self.latent_mse_weight = self.loss_cfg.latent_mse
        
        # 加载 Pipeline
        pipeline_type = cfg.guidance.flowedit.pipeline_type
        model_path = cfg.guidance.model_path
        
        print(f"[FlowEditGuidance] Loading pipeline on {self.device}...")
        print(f"[FlowEditGuidance] Pipeline: {pipeline_type}, Model: {model_path}")
        print(f"[FlowEditGuidance] Loss weights: latent_csd={self.latent_csd_weight}, latent_mse={self.latent_mse_weight}")
        
        self.adapter = create_pipeline_adapter(pipeline_type)
        self.adapter.load(model_path, self.device)
        self.pipe = self.adapter.pipe
        
        # 创建 Metrics (SSIM, LPIPS, DINO)
        # 注：latent_csd 和 latent_mse 由 Tracker.loss() 计算，不在此处创建
        self.metrics = create_metrics(
            self.loss_cfg,
            self.device,
            extra_kwargs={
                "dino": {
                    "model_path": cfg.guidance.get("dino_model_path", "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"),
                    "image_size": cfg.guidance.get("dino_image_size", 518),
                },
            },
        )
        print(f"[FlowEditGuidance] Metrics: {list(self.metrics.keys())}")
    
    # =========================================================================
    # FlowEdit 单张编辑（内部方法）
    # =========================================================================
    
    def _edit_single(
        self,
        rendered_pil: Image.Image,
        condition_pil: Image.Image,
        latent_before: torch.Tensor,
    ) -> EditOutput:
        """
        执行单张图的 FlowEdit 编辑。
        
        Args:
            rendered_pil: 渲染图 PIL（供 VLM 使用）
            condition_pil: 条件图 PIL
            latent_before: 渲染图的 latent [1, seq, C]
        
        Returns:
            EditOutput
        """
        condition_pil = composite_alpha_to_white(condition_pil)
        
        # 确保 latent 是 bfloat16（与 FlowEdit pipeline 一致）
        latent_before = latent_before.to(dtype=torch.bfloat16)
        
        result = self.adapter.edit(
            rendered_pil,
            condition_pil,
            self.flowedit_cfg,
            src_latent=latent_before,
        )
        
        return EditOutput(
            image=result.image,
            latent=result.latent,
            tracker=result.tracker,
        )
    
    # =========================================================================
    # Pipeline 调用（实现抽象方法）
    # =========================================================================
    
    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        B: int,
        V: int,
    ) -> FlowEditPipelineOutput:
        """
        执行 FlowEdit Pipeline（无梯度，多步编辑）。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            FlowEditPipelineOutput
        """
        N = B * V
        H, W = comp_rgb.shape[2], comp_rgb.shape[3]
        
        edited_images, latents, trackers = [], [], []
        
        for b in range(B):
            for v in range(V):
                i = b * V + v
                
                # 使用传入的 latent
                rendered_pil = self.tensor_to_pil(comp_rgb[i])
                output = self._edit_single(
                    rendered_pil,
                    condition_images[b],
                    src_latent[i:i+1],
                )
                
                edited_images.append(output.image)
                latents.append(output.latent)
                trackers.append(output.tracker)
        
        # 转换编辑后的图像为 tensor
        edited_tensor = self.pils_to_tensor(edited_images, (W, H))
        
        return FlowEditPipelineOutput(
            edited_images=edited_images,
            edited_tensor=edited_tensor,
            latent_after=torch.cat(latents, dim=0),
            trackers=trackers,
        )
    
    # =========================================================================
    # Latent MSE 计算（FlowEdit 专属，通过 Tracker.loss()）
    # =========================================================================
    
    def _compute_latent_loss(
        self,
        latent_before: torch.Tensor,
        trackers: List[FlowEditStateTracker],
    ) -> torch.Tensor:
        """
        计算 Latent Loss（通过 Tracker.loss()，支持 CSD + MSE 混合）。
        
        Loss = csd_weight * CSD_Loss + mse_weight * MSE_Loss
        
        Args:
            latent_before: [N, seq, C] 有梯度
            trackers: [N] 中间状态
        
        Returns:
            标量 loss
        """
        losses = []
        for i, tracker in enumerate(trackers):
            single = latent_before[i:i+1]
            # 使用 Tracker 的统一 loss 方法（支持 latent_csd + latent_mse）
            loss = tracker.loss(
                src=single,
                csd_weight=self.latent_csd_weight,
                mse_weight=self.latent_mse_weight,
                reduce=self.reduce_mode,
                ada=self.ada_normalize,
                eps=self.ada_eps,
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================
    
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        comp_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 FlowEdit Loss（真 loss）。
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: FlowEdit Pipeline 输出
            comp_rgb: [N, C, H, W] 渲染图（用于图像空间 loss）
        
        Returns:
            (loss, loss_dict)
        """
        loss_dict = {}
        
        # 1. Latent Loss（核心 loss，通过 Tracker.loss()，支持 CSD + MSE 混合）
        latent_loss = self._compute_latent_loss(
            src_latent,
            pipeline_output.trackers,
        )
        loss_dict["latent"] = latent_loss
        
        # 2. 图像空间 loss（辅助 loss：SSIM, LPIPS, DINO）
        for name, metric in self.metrics.items():
            loss_dict[name] = metric.compute(comp_rgb, pipeline_output.edited_tensor.detach())
        
        # 加权汇总（latent loss 权重固定为 1.0，内部已用 csd/mse 权重）
        total_loss = latent_loss
        for name, metric in self.metrics.items():
            total_loss = total_loss + metric.weight * loss_dict[name]
        
        return total_loss, {k: v.detach() for k, v in loss_dict.items()}
    
    # =========================================================================
    # 构建返回结果（覆盖父类方法以添加 FlowEdit 专属字段）
    # =========================================================================
    
    def _build_result(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        pipeline_output: FlowEditPipelineOutput,
        B: int, V: int, C: int, H: int, W: int,
        source_device: torch.device,
    ) -> GuidanceResult:
        """
        构建返回结果，添加 FlowEdit 专属字段。
        """
        edited_for_vis = pipeline_output.edited_tensor.reshape(B, V, C, H, W).to(source_device)
        
        return GuidanceResult(
            loss=loss,
            edited_imgs=edited_for_vis,  # FlowEdit 专属
            loss_dict={self.loss_key: loss.detach(), **loss_dict},
            trackers=pipeline_output.trackers,  # FlowEdit 专属
        )
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取各项 loss 的权重"""
        weights = {
            "latent_csd": self.latent_csd_weight,
            "latent_mse": self.latent_mse_weight,
        }
        for name in ["ssim", "lpips", "dino"]:
            weights[name] = self.metrics[name].weight if name in self.metrics else 0.0
        return weights
    
    def cleanup(self) -> None:
        """释放模型显存"""
        print("[FlowEditGuidance] Cleaning up...")
        del self.pipe
        for metric in self.metrics.values():
            metric.cleanup()
        self.metrics.clear()
        torch.cuda.empty_cache()
        print("[FlowEditGuidance] Cleanup done.")


# =============================================================================
# FlowEditGuidancePP - 流水线并行版本
# =============================================================================

class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
    """
    FlowEdit Guidance + 流水线并行。
    
    通过 Mixin 组合实现：
    - 继承 FlowEditGuidance 的全部功能
    - 添加 submit_async/wait_and_get 异步接口
    """
    
    def __init__(self, cfg, train_device: torch.device):
        super().__init__(cfg, train_device)
        self._init_pipeline_parallel(num_streams=2)
        print(f"[FlowEditGuidancePP] Pipeline parallelism enabled.")
