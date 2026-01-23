"""
CSD-Rev (Classifier Score Distillation with Reverse Correction) Guidance 模块。

相比 CSD，增加逆向修正步骤（iCSD）来减少梯度方差。
继承自 CSDGuidance，只更换 Pipeline 为带逆向修正的版本。

参考：RFDS-Rev (Rectified Flow Distillation Sampling with Reverse)

逆向修正步骤:
    1. 用条件 prompt 预测 velocity（不带 CFG）
    2. 使用预测结果修正 noise
    3. 用修正后的 noise 重新加噪

计算开销：相比 CSD 多一次 transformer 前向传播
"""

from typing import Any, List, Dict, Tuple
import torch
from PIL import Image

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.paradigms.csd import CSDGuidance
from edit4shape.guidance.pipelines.qwen_image_edit.csd_rev import QwenImageCSDRevPipeline, CSDOutput


class CSDRevGuidance(CSDGuidance):
    """
    CSD-Rev Guidance（带逆向修正的 CSD）。
    
    继承自 CSDGuidance，仅更换 Pipeline 为带逆向修正的版本。
    计算开销：相比 CSD 多一次 transformer 前向传播（逆向修正步）。
    """
    
    # 覆盖父类的 loss_key
    loss_key = "csd_rev"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 CSD-Rev Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        # 调用 BaseGuidance.__init__（跳过 CSDGuidance.__init__）
        from edit4shape.guidance.base import BaseGuidance
        BaseGuidance.__init__(self, cfg, train_device)
        
        # 使用 csd_rev 专用配置
        self.csd_cfg = cfg.guidance.csd_rev
        self.min_step_percent = self.csd_cfg.min_step_percent
        self.max_step_percent = self.csd_cfg.max_step_percent
        self.weight_type = self.csd_cfg.weight_type
        self.weight_eps = self.csd_cfg.weight_eps
        self.true_cfg_scale = self.csd_cfg.true_cfg_scale
        self.target_prompt = self.csd_cfg.target_prompt
        self.negative_prompt = self.csd_cfg.negative_prompt
        self.seed = self.csd_cfg.seed
        
        # 逆向修正步配置（CSD-Rev 专用）
        self.rev_use_uncond = self.csd_cfg.rev_use_uncond
        
        # 加载 CSD-Rev Pipeline
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2511")
        
        print(f"[CSDRevGuidance] Loading CSD-Rev pipeline on {self.device}...")
        print(f"[CSDRevGuidance] Model: {model_path}")
        
        self.pipe = QwenImageCSDRevPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        print(f"[CSDRevGuidance] CSD-Rev params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
              f"weight={self.weight_type}, cfg={self.true_cfg_scale}, rev_use_uncond={self.rev_use_uncond}")
    
    # =========================================================================
    # Pipeline 调用（覆盖父类方法以传递 rev_use_uncond）
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
        调用 CSD-Rev Pipeline。
        
        覆盖父类以传递 rev_use_uncond 参数。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            B, V: batch size 和 views
        
        Returns:
            CSDOutput: CSD-Rev Pipeline 输出
        """
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
            rev_use_uncond=self.rev_use_uncond,  # CSD-Rev 专用参数
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        )
