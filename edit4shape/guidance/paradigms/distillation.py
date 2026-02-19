"""
统一蒸馏 Guidance 模块（合并 SDS + CSD）。

将 Qwen-Image-Edit 模型作为蒸馏教师，
提供单步梯度蒸馏能力。

通过 sds_weight 和 csd_weight 控制 loss 类型：
- sds_weight=1, csd_weight=0 → 纯 SDS: MSE(src, x0_pos)
- sds_weight=0, csd_weight=1 → 纯 CSD: MSE(src, x0_pos) - MSE(src, x0_neg)
- sds_weight=1, csd_weight=1 → 混合模式

★ Init / Runtime 分离：
- __init__: 只加载 Pipeline 模型
- compute_guidance: 运行时参数（prompt / loss 权重等）通过 guidance_cfg 传入

数据流（继承自 BaseGuidance，真 Loss 模式）：
    1. 格式转换（父类）
    2. 编码到 latent（父类）
    3. 调用 Distillation Pipeline（_run_pipeline）
    4. 通过 Tracker.loss() 计算真 loss（_compute_loss）
"""

import logging
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
    
    ★ Init / Runtime 分离：
    - __init__: 加载 Distillation Pipeline
    - compute_guidance(..., guidance_cfg=...): 运行时参数通过 guidance_cfg 传入
    
    guidance_cfg 运行时参数（per-stage 不同）：
        - target_prompt, negative_prompt
        - min_step_percent, max_step_percent, true_cfg_scale
        - mse_weight, csd_weight
        - ada_normalize, ada_eps
        - num_timesteps, reduce_mode, noise_mode
        - seed
    """
    
    # 类属性：用于 loss_dict 的 key 名称
    loss_key = "distillation"
    
    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        """
        初始化 Distillation Guidance（只加载模型）。
        
        Args:
            guidance_cfg: Guidance 初始化配置（cfg.guidance），包含：
                - model_path: 模型路径
                - edit_resolution: VAE 编码分辨率
            train_device: 训练使用的设备
        """
        super().__init__(guidance_cfg, train_device)
        
        # 加载 Distillation Pipeline（重量级，一次性）
        model_path = guidance_cfg.model_path
        
        logging.info(f"[DistillationGuidance] Loading pipeline on {self.device}...")
        logging.info(f"[DistillationGuidance] Model: {model_path}")
        
        self.pipe = QwenImageDistillationPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        logging.info(f"[DistillationGuidance] Ready. Runtime params will be read from guidance_cfg at call time.")
    
    # =========================================================================
    # Pipeline 调用（实现抽象方法）
    # =========================================================================
    
    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        guidance_cfg: Any,
        B: int,
        V: int,
    ) -> DistillationOutput:
        """
        调用 Distillation Pipeline。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            guidance_cfg: 运行时配置（prompt / steps / loss 权重等）
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
            prompt=guidance_cfg.target_prompt,
            negative_prompt=guidance_cfg.negative_prompt,
            src_latent=src_latent.to(torch.bfloat16),
            height=self.edit_resolution,
            width=self.edit_resolution,
            min_step_percent=guidance_cfg.min_step_percent,
            max_step_percent=guidance_cfg.max_step_percent,
            true_cfg_scale=guidance_cfg.true_cfg_scale,
            num_timesteps=guidance_cfg.num_timesteps,
            noise_mode=guidance_cfg.noise_mode,
            generator=torch.Generator(device=self.device).manual_seed(guidance_cfg.seed),
        )
    
    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================
    
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: DistillationOutput,
        comp_rgb: torch.Tensor,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算蒸馏 Loss（通过 Tracker.loss()）。
        
        Loss = sds_weight * MSE(src, x0_pos) + csd_weight * CSD_Loss
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: Pipeline 输出（包含 DistillationStateTracker）
            comp_rgb: [N, C, H, W] 渲染图（未使用）
            guidance_cfg: 运行时配置（loss 权重 / reduce 策略等）
        
        Returns:
            (loss, loss_dict)
        """
        tracker = pipeline_output.tracker
        
        # 使用 Tracker 的统一 loss 方法（支持多时间步聚合）
        loss = tracker.loss(
            src=src_latent,
            mse_weight=guidance_cfg.mse_weight,
            csd_weight=guidance_cfg.csd_weight,
            ada=guidance_cfg.ada_normalize,
            eps=guidance_cfg.ada_eps,
            reduce=guidance_cfg.reduce_mode,
        )
        
        return loss, {}
