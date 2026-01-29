"""
统一蒸馏 Guidance 模块（合并 SDS + CSD）。

将 Qwen-Image-Edit 模型作为蒸馏教师，
提供单步梯度蒸馏能力。

通过 sds_weight 和 csd_weight 控制 loss 类型：
- sds_weight=1, csd_weight=0 → 纯 SDS: MSE(src, x0_high)
- sds_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_high) - MSE(src, x0_low)
- sds_weight=1, csd_weight=1 → 混合模式

数据流（继承自 BaseGuidance，真 Loss 模式）：
    1. 格式转换（父类）
    2. 编码到 latent（父类）
    3. 调用 Distillation Pipeline（_run_pipeline）
    4. 通过 Tracker.loss() 计算真 loss（_compute_loss）
"""

from typing import List, Any, Dict, Tuple

import torch
from PIL import Image

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipelines.qwen_image_edit.distillation import (
    QwenImageDistillationPipeline,
    DistillationOutput,
)


# =============================================================================
# DistillationGuidance
# =============================================================================

class DistillationGuidance(BaseGuidance):
    """
    统一蒸馏 Guidance（合并 SDS + CSD）。
    
    使用 Qwen-Image-Edit 模型进行单步蒸馏。
    
    Loss 公式：
        loss = sds_weight * MSE(src, x0_high)
             + csd_weight * (MSE(src, x0_high) - MSE(src, x0_low))
    
    其中：
        - x0_high: 高 CFG 预测的 x0（吸引目标）
        - x0_low: 低 CFG 预测的 x0（排斥目标）
    
    weight_type：
        - uniform: 不加权
        - ada: 自适应归一化（按差异大小归一化）
    """
    
    # 类属性：用于 loss_dict 的 key 名称
    loss_key = "distillation"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 Distillation Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        super().__init__(cfg, train_device)
        
        # 蒸馏专属配置
        self.distill_cfg = cfg.guidance.distillation
        self.min_step_percent = self.distill_cfg.min_step_percent
        self.max_step_percent = self.distill_cfg.max_step_percent
        self.weight_type = self.distill_cfg.weight_type
        self.weight_eps = self.distill_cfg.weight_eps
        self.true_cfg_scale = self.distill_cfg.true_cfg_scale
        self.target_prompt = self.distill_cfg.target_prompt
        self.negative_prompt = self.distill_cfg.negative_prompt
        self.seed = self.distill_cfg.seed
        
        # Loss 权重
        self.sds_weight = self.distill_cfg.sds_weight
        self.csd_weight = self.distill_cfg.csd_weight
        
        # 是否需要计算低 CFG 分支（CSD 模式需要）
        self.compute_low_cfg = self.csd_weight > 0
        
        # 加载 Distillation Pipeline
        model_path = cfg.guidance.model_path
        
        print(f"[DistillationGuidance] Loading pipeline on {self.device}...")
        print(f"[DistillationGuidance] Model: {model_path}")
        print(f"[DistillationGuidance] Mode: sds_weight={self.sds_weight}, csd_weight={self.csd_weight}")
        
        self.pipe = QwenImageDistillationPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        print(f"[DistillationGuidance] Params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
              f"weight={self.weight_type}, cfg={self.true_cfg_scale}")
    
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
    ) -> DistillationOutput:
        """
        调用 Distillation Pipeline。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            DistillationOutput: Pipeline 输出（包含 DistillationStateTracker）
        """
        # 构造 image 列表（[rendered, condition]）
        rendered_pil = self.tensor_to_pil(comp_rgb[0].cpu())
        condition_pil = composite_alpha_to_white(condition_images[0])
        image_list = [rendered_pil, condition_pil]
        
        return self.pipe(
            image=image_list,
            prompt=self.target_prompt,
            negative_prompt=self.negative_prompt,
            src_latent=src_latent.to(torch.bfloat16),
            height=self.edit_resolution,
            width=self.edit_resolution,
            min_step_percent=self.min_step_percent,
            max_step_percent=self.max_step_percent,
            true_cfg_scale=self.true_cfg_scale,
            compute_low_cfg=self.compute_low_cfg,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        )
    
    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================
    
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: DistillationOutput,
        comp_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算蒸馏 Loss（通过 Tracker.loss()）。
        
        Loss = sds_weight * MSE(src, x0_high) + csd_weight * CSD_Loss
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: Pipeline 输出（包含 DistillationStateTracker）
            comp_rgb: [N, C, H, W] 渲染图（未使用）
        
        Returns:
            (loss, loss_dict)
        """
        tracker = pipeline_output.tracker
        
        # 使用 Tracker 的统一 loss 方法
        loss = tracker.loss(
            src=src_latent,
            sds_weight=self.sds_weight,
            csd_weight=self.csd_weight,
            ada=(self.weight_type == "ada"),
            eps=self.weight_eps,
        )
        
        return loss, {}
