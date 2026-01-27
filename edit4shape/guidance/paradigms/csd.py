"""
CSD (Classifier Score Distillation) Guidance 模块。

将 Qwen-Image-Edit 模型作为 CSD 教师，
提供单步梯度蒸馏能力。

CSD 与 SDS 的区别：
    - SDS: 让 src_latent 向 x0_pred 靠拢（单次推理）
    - CSD: 让 src_latent 向 x0_high 靠拢，同时远离 x0_low（两次推理）

CSD Loss 公式：
    loss = MSE(src_latent, x0_pred_high) - MSE(src_latent, x0_pred_low)

数据流（继承自 BaseGuidance，真 Loss 模式）：
    1. 格式转换（父类）
    2. 编码到 latent（父类）
    3. 调用 CSD Pipeline（_run_pipeline）
    4. 通过 Tracker.loss() 计算真 loss（_compute_loss）
"""

from typing import List, Any, Dict, Tuple

import torch
from PIL import Image

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipelines.qwen_image_edit.csd import QwenImageCSDPipeline, CSDOutput


# =============================================================================
# CSDGuidance
# =============================================================================

class CSDGuidance(BaseGuidance):
    """
    CSD Guidance（继承 BaseGuidance，真 Loss 模式）。
    
    使用 Qwen-Image-Edit 模型进行 Classifier Score Distillation。
    
    CSD Loss 公式：
        loss = MSE(src_latent, x0_pred_high) - MSE(src_latent, x0_pred_low)
    
    其中：
        - x0_pred_high: 高 CFG 预测的 x0（吸引目标）
        - x0_pred_low: 低 CFG 预测的 x0（排斥目标）
    
    weight_type：
        - uniform: 不加权
        - ada: 自适应归一化（按差异大小归一化）
    """
    
    # 类属性：用于 loss_dict 的 key 名称
    loss_key = "csd"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 CSD Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        super().__init__(cfg, train_device)
        
        # CSD 专属配置
        self.csd_cfg = cfg.guidance.csd
        self.min_step_percent = self.csd_cfg.min_step_percent
        self.max_step_percent = self.csd_cfg.max_step_percent
        self.weight_type = self.csd_cfg.weight_type
        self.weight_eps = self.csd_cfg.weight_eps
        self.true_cfg_scale = self.csd_cfg.true_cfg_scale
        self.target_prompt = self.csd_cfg.target_prompt
        self.negative_prompt = self.csd_cfg.negative_prompt
        self.seed = self.csd_cfg.seed
        
        # 加载 CSD Pipeline
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2511")
        
        print(f"[CSDGuidance] Loading CSD pipeline on {self.device}...")
        print(f"[CSDGuidance] Model: {model_path}")
        
        self.pipe = QwenImageCSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        print(f"[CSDGuidance] CSD params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
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
    ) -> CSDOutput:
        """
        调用 CSD Pipeline。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            CSDOutput: CSD Pipeline 输出（包含 CSDStateTracker）
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
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        )
    
    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================
    
    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: CSDOutput,
        comp_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 CSD Loss（真 loss，通过 Tracker.loss()）。
        
        CSD Loss = MSE(src, x0_high) - MSE(src, x0_low)
        
        Args:
            src_latent: [N, seq, C] 有梯度的 latent
            pipeline_output: CSD Pipeline 输出（包含 CSDStateTracker）
            comp_rgb: [N, C, H, W] 渲染图（未使用）
        
        Returns:
            (loss, loss_dict)
        """
        tracker = pipeline_output.tracker
        
        # 使用 Tracker 的统一 loss 方法
        loss = tracker.loss(
            src=src_latent,
            ada=(self.weight_type == "ada"),
            eps=self.weight_eps,
        )
        
        return loss, {}
