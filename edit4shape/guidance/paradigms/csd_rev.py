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

from typing import Any, List
from PIL import Image
import torch

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult, SpecifyGradient
from edit4shape.guidance.paradigms.csd import CSDGuidance
from edit4shape.guidance.pipelines.qwen_image_edit.csd_rev import QwenImageCSDRevPipeline


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
        # 不调用 super().__init__()，手动初始化以使用不同的 Pipeline
        self.cfg = cfg
        self.csd_cfg = cfg.guidance.csd_rev  # 使用 csd_rev 专用配置
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        
        # 加载 CSD-Rev Pipeline（与 CSD 唯一的不同点）
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2511")
        
        print(f"[CSDRevGuidance] Loading CSD-Rev pipeline on {self.device}...")
        print(f"[CSDRevGuidance] Model: {model_path}")
        
        self.pipe = QwenImageCSDRevPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # 使用 csd_rev 专用配置
        self.edit_resolution = cfg.guidance.edit_resolution
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
        
        print(f"[CSDRevGuidance] CSD-Rev params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
              f"weight={self.weight_type}, cfg={self.true_cfg_scale}, rev_use_uncond={self.rev_use_uncond}")
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> GuidanceResult:
        """
        计算 CSD-Rev Guidance loss。
        
        覆盖父类以传递 rev_use_uncond 参数。
        
        Args:
            comp_rgb: 渲染图像 (B, V, H, W, C) 或 (B, V, C, H, W)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 额外参数
        
        Returns:
            GuidanceResult: 包含 CSD-Rev loss
        """
        # 展平 batch 和 view 维度
        if comp_rgb.dim() == 5:
            if comp_rgb.shape[-1] == 3:
                B, V, H, W, C = comp_rgb.shape
                comp_rgb = comp_rgb.permute(0, 1, 4, 2, 3).reshape(B * V, C, H, W)
            else:
                B, V, C, H, W = comp_rgb.shape
                comp_rgb = comp_rgb.reshape(B * V, C, H, W)
        
        comp_rgb = comp_rgb.to(self.device)
        src_latent = self.encode_to_latent(comp_rgb)  # [N, seq, C_lat]
        
        rendered_pil = self.tensor_to_pil(comp_rgb[0].cpu())
        condition_pil = composite_alpha_to_white(condition_images[0])
        image_list = [rendered_pil, condition_pil]
        
        # 调用 CSD-Rev Pipeline（传递 rev_use_uncond）
        with torch.no_grad():
            csd_output = self.pipe(
                image=image_list,
                prompt=self.target_prompt,
                negative_prompt=self.negative_prompt,
                src_latent=src_latent.to(torch.bfloat16),
                height=self.edit_resolution,
                width=self.edit_resolution,
                min_step_percent=self.min_step_percent,
                max_step_percent=self.max_step_percent,
                weight_type=self.weight_type,
                weight_eps=self.weight_eps,
                true_cfg_scale=self.true_cfg_scale,
                rev_use_uncond=self.rev_use_uncond,  # CSD-Rev 专用参数
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
            )
        
        grad = csd_output.grad  # [N, seq, C_lat]
        weight = csd_output.weight  # [N]
        weighted_grad = grad * weight.view(-1, 1, 1)  # [N, seq, C_lat]
        
        # 计算 loss 值
        loss_value = 0.5 * (weighted_grad ** 2).mean()
        
        # 通过 SpecifyGradient 注入梯度
        src_latent_train = src_latent.to(self.train_device)
        weighted_grad_train = weighted_grad.to(self.train_device, dtype=src_latent_train.dtype)
        loss_value_train = loss_value.to(self.train_device)
        
        loss = SpecifyGradient.apply(src_latent_train, weighted_grad_train, loss_value_train)
        loss = loss.mean()
        
        return GuidanceResult(
            loss=loss,
            edited_imgs=None,
            loss_dict={self.loss_key: loss.detach()},
            trackers=None,
        )
