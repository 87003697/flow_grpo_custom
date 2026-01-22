"""
CSD (Classifier Score Distillation) Guidance 模块。

将 Qwen-Image-Edit 模型作为 CSD 教师，
提供单步梯度蒸馏能力。

CSD 与 SDS 的区别：
    - SDS: grad = noise_pred - noise（单次推理）
    - CSD: grad = x0_low - x0_high（两次推理，高低 CFG 差分）

数据流:
    1. rendered → encode → clean_latent
    2. 采样 t, noise → noisy_latent = (1-t)*clean + t*noise
    3. Transformer 预测 v_pred (高 CFG + 低 CFG 两次)
    4. 计算 x0_high, x0_low → grad = x0_low - x0_high
    5. 通过 SpecifyGradient 注入梯度
"""

from typing import List, Any
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult, BaseGuidance, SpecifyGradient
from edit4shape.guidance.pipelines.qwen_image_edit.csd import QwenImageCSDPipeline


# =============================================================================
# CSDGuidance
# =============================================================================

class CSDGuidance(BaseGuidance):
    """
    CSD Guidance。
    
    使用 Qwen-Image-Edit 模型进行 Classifier Score Distillation。
    
    CSD 与 SDS 的区别：
        - SDS: grad = noise_pred - noise（单次推理）
        - CSD: grad = x0_low - x0_high（两次推理，高低 CFG 差分）
    
    数据流:
        1. rendered → VAE encode → clean_latent
        2. 采样时间步 t，加噪得到 noisy_latent
        3. Transformer 预测 v_pred（高 CFG + 低 CFG 两次）
        4. 计算 CSD 梯度: grad = x0_low - x0_high
        5. 通过 SpecifyGradient 注入梯度到 rendered
    
    与 FlowEdit 的主要区别:
        - FlowEdit: 多步编辑，生成目标图像，用 MSE loss 监督
        - CSD: 单步采样，直接计算梯度，通过 SpecifyGradient 注入
    """
    
    # 类属性：用于 loss_dict 的 key 名称（子类可覆盖）
    loss_key = "csd"
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 CSD Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        self.cfg = cfg
        self.csd_cfg = cfg.guidance.csd
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        
        # 加载 CSD Pipeline
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2511")
        
        print(f"[CSDGuidance] Loading CSD pipeline on {self.device}...")
        print(f"[CSDGuidance] Model: {model_path}")
        
        self.pipe = QwenImageCSDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # 编辑分辨率
        self.edit_resolution = cfg.guidance.edit_resolution
        
        # CSD 参数
        self.min_step_percent = self.csd_cfg.min_step_percent
        self.max_step_percent = self.csd_cfg.max_step_percent
        self.weight_type = self.csd_cfg.weight_type
        self.weight_eps = self.csd_cfg.weight_eps
        self.true_cfg_scale = self.csd_cfg.true_cfg_scale
        self.target_prompt = self.csd_cfg.target_prompt
        self.negative_prompt = self.csd_cfg.negative_prompt
        self.seed = self.csd_cfg.seed
        
        print(f"[CSDGuidance] CSD params: min_t={self.min_step_percent}, max_t={self.max_step_percent}, "
              f"weight={self.weight_type}, cfg={self.true_cfg_scale}")
    
    # =========================================================================
    # 格式转换模块
    # =========================================================================
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """[C, H, W] float [0,1] → PIL"""
        arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr.transpose(1, 2, 0))
    
    # =========================================================================
    # Latent 编码模块
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
    # 主入口
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        **kwargs,
    ) -> GuidanceResult:
        """
        计算 CSD Guidance loss。
        
        Args:
            comp_rgb: 渲染图像 (B, V, H, W, C) 或 (B, V, C, H, W)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            **kwargs: 额外参数
        
        Returns:
            GuidanceResult: 包含 CSD loss
        """
        # 展平 batch 和 view 维度
        if comp_rgb.dim() == 5:
            if comp_rgb.shape[-1] == 3:
                # (B, V, H, W, C) → (B*V, C, H, W)
                B, V, H, W, C = comp_rgb.shape
                comp_rgb = comp_rgb.permute(0, 1, 4, 2, 3).reshape(B * V, C, H, W)
            else:
                # (B, V, C, H, W) → (B*V, C, H, W)
                B, V, C, H, W = comp_rgb.shape
                comp_rgb = comp_rgb.reshape(B * V, C, H, W)
        
        N = comp_rgb.shape[0]  # B * V
        
        # 移动到 guidance 设备
        comp_rgb = comp_rgb.to(self.device)
        
        # 编码渲染图到 latent（可微分）
        src_latent = self.encode_to_latent(comp_rgb)  # [N, seq, C_lat]
        
        # 构造 image 列表（与 FlowEdit 保持一致：[rendered, condition]）
        # 目前只处理 N=1 的情况（单张渲染图）
        rendered_pil = self.tensor_to_pil(comp_rgb[0].cpu())
        condition_pil = composite_alpha_to_white(condition_images[0])
        image_list = [rendered_pil, condition_pil]  # [0]=rendered, [1]=condition
        
        # 调用 CSD Pipeline
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
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
            )
        
        # 计算加权梯度
        grad = csd_output.grad  # [N, seq, C_lat]
        weight = csd_output.weight  # [N]
        
        # 应用权重
        weighted_grad = grad * weight.view(-1, 1, 1)  # [N, seq, C_lat]
        
        # 计算有意义的 loss 值（CSD loss = 0.5 * ||grad||^2）
        loss_value = 0.5 * (weighted_grad ** 2).mean()
        
        # 通过 SpecifyGradient 注入梯度
        # 需要将梯度转回 train_device
        src_latent_train = src_latent.to(self.train_device)
        weighted_grad_train = weighted_grad.to(self.train_device, dtype=src_latent_train.dtype)
        loss_value_train = loss_value.to(self.train_device)
        
        loss = SpecifyGradient.apply(src_latent_train, weighted_grad_train, loss_value_train)
        loss = loss.mean()  # 确保标量
        
        return GuidanceResult(
            loss=loss,
            edited_imgs=None,  # CSD 不生成编辑图像
            loss_dict={self.loss_key: loss.detach()},  # 使用类属性，子类可覆盖
            trackers=None,
        )
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
