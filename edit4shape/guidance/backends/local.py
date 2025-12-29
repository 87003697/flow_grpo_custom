"""
同进程多 GPU Guidance。

将 Qwen-Image-Edit 模型加载到 训练设备+1 的 GPU 上，
与 Trellis 训练进程共存于同一 Python 进程。

优势：
- 零序列化开销：Tensor 直接跨 GPU 传输
- 自动求导：无需手动计算梯度，PyTorch autograd 自动处理
- 简单调试：单进程，断点调试容易

设备分配示例：
- 训练在 cuda:0 → FlowEdit 在 cuda:1
- 训练在 cuda:2 → FlowEdit 在 cuda:3
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim
import lpips

from edit4shape.guidance.base import GuidanceResult

# 添加 Qwen-Image-Edit 到路径
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_QWEN_EDIT_ROOT = os.path.join(_REPO_ROOT, "_reference_codes", "Qwen-Image-Edit")
if _QWEN_EDIT_ROOT not in sys.path:
    sys.path.insert(0, _QWEN_EDIT_ROOT)

from pipelines.pipeline_qwenimage_edit_plus_flowedit_v2 import QwenImageEditPlusPipeline


def _compute_guidance_device(train_device: torch.device) -> torch.device:
    """
    根据训练设备计算 Guidance 模型设备。
    
    规则：Guidance 设备 = 训练设备 + 1
    
    Args:
        train_device: 训练使用的设备
    
    Returns:
        torch.device: Guidance 模型设备
    """
    if train_device.type != "cuda":
        raise ValueError(f"训练设备必须是 CUDA，当前: {train_device}")
    
    train_idx = train_device.index if train_device.index is not None else 0
    guidance_idx = train_idx + 1
    
    # 检查设备是否存在
    if guidance_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"Guidance 需要 cuda:{guidance_idx}，但只有 {torch.cuda.device_count()} 个 GPU。"
            f"训练设备: {train_device}"
        )
    
    return torch.device(f"cuda:{guidance_idx}")


@dataclass
class PreprocessedImages:
    """预处理后的图像数据。"""
    pred: torch.Tensor       # (B*V,C,H,W) 渲染图（在 guidance 设备上）
    target: torch.Tensor     # (B*V,C,H,W) 编辑后图像（在 guidance 设备上，detached）
    edited_imgs: torch.Tensor  # (B,V,C,H,W) 用于返回的编辑图像（在原设备上）


class LocalGuidance:
    """
    同进程多 GPU Guidance。
    
    特点：
    - Qwen-Image-Edit 自动加载到 train_device + 1
    - 直接计算 loss（SSIM/LPIPS/Latent MSE）
    - PyTorch autograd 自动处理梯度
    """
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 Guidance。
        
        Args:
            cfg: guidance 配置
            train_device: 训练使用的设备（用于计算 Guidance 设备）
        """
        self.cfg = cfg
        self.flowedit_cfg = cfg.flowedit
        self.train_device = train_device
        self.device = _compute_guidance_device(train_device)
        
        # ---- 1. 加载 Pipeline ----
        print(f"[LocalGuidance] Loading Qwen-Image-Edit pipeline on {self.device}...")
        print(f"[LocalGuidance] 训练设备: {train_device}, Guidance 设备: {self.device}")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[LocalGuidance] Pipeline loaded.")
        
        # ---- 2. LPIPS 模型 (fp32) ----
        print(f"[LocalGuidance] Loading LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False  # LPIPS 只做前向
        print(f"[LocalGuidance] LPIPS model loaded.")
        
        # ---- 3. 算法参数 ----
        self.prompt = self.flowedit_cfg.prompt
        self.seed = self.flowedit_cfg.seed
        self.steps = self.flowedit_cfg.steps
        self.guidance_scale = self.flowedit_cfg.guidance_scale
        self.true_cfg_scale_tgt = self.flowedit_cfg.true_cfg_scale_tgt
        self.n_min = self.flowedit_cfg.n_min
        self.n_max = self.flowedit_cfg.n_max
        
        # ---- 4. Loss 权重 ----
        self.ssim_weight = self.flowedit_cfg.ssim_weight
        self.lpips_weight = self.flowedit_cfg.lpips_weight
        self.latent_mse_weight = self.flowedit_cfg.latent_mse_weight
        
        # FlowEdit 的工作分辨率
        self.edit_resolution = cfg.get("edit_resolution", 1024)
    
    # =========================================================================
    # 图像格式转换
    # =========================================================================
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """(C,H,W) float [0,1] -> PIL.Image"""
        arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        arr = arr.transpose(1, 2, 0)  # (H,W,C)
        return Image.fromarray(arr)
    
    def _pil_to_tensor(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        """PIL.Image -> (C,H,W) float [0,1]"""
        return TF.to_tensor(img).to(device)  # (C,H,W)
    
    # =========================================================================
    # FlowEdit 编辑
    # =========================================================================
    
    def _edit_single(self, src_pil: Image.Image, tgt_pil: Image.Image) -> Image.Image:
        """
        单张图像 FlowEdit 编辑。
        
        Args:
            src_pil: 源图像（渲染图）
            tgt_pil: 目标图像（条件图）
        
        Returns:
            编辑后的图像
        """
        # Resize 到工作分辨率
        src_resized = src_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        tgt_resized = tgt_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        
        with torch.inference_mode():
            output = self.pipe(
                image_src=src_resized,
                image_tgt=tgt_resized,
                prompt=self.prompt,
                generator=torch.manual_seed(self.seed),
                negative_prompt=" ",
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                true_cfg_scale_tgt=self.true_cfg_scale_tgt,
                n_min=self.n_min,
                n_max=self.n_max,
            )
        
        return output.images[0]
    
    # =========================================================================
    # 图像预处理
    # =========================================================================
    
    def _preprocess_images(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
    ) -> PreprocessedImages:
        """
        图像预处理：执行 FlowEdit 编辑并准备 loss 计算所需的张量。
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
        
        Returns:
            PreprocessedImages: 包含 pred、target 和 edited_imgs
        """
        B, V, H, W, C = comp_rgb.shape
        source_device = comp_rgb.device
        
        # 转换格式：(B,V,H,W,C) -> (B,V,C,H,W)
        pred_imgs = comp_rgb.permute(0, 1, 4, 2, 3)  # (B,V,C,H,W)
        
        # 收集并编辑所有图像
        edited_tensors = []
        for b in range(B):
            for v in range(V):
                src_tensor = pred_imgs[b, v]  # (C,H,W)
                src_pil = self._tensor_to_pil(src_tensor)
                
                # FlowEdit 编辑
                edited_pil = self._edit_single(src_pil, condition_images[b])
                
                # Resize 回原始分辨率并转为 Tensor
                edited_pil_resized = edited_pil.resize((W, H), Image.LANCZOS)
                edited_tensor = self._pil_to_tensor(edited_pil_resized, self.device)  # (C,H,W)
                edited_tensors.append(edited_tensor)
        
        # 堆叠为 Tensor
        edited_flat = torch.stack(edited_tensors)  # (B*V,C,H,W)
        edited_imgs = edited_flat.reshape(B, V, C, H, W)  # (B,V,C,H,W)
        
        # 准备 loss 计算所需的张量
        pred_flat = pred_imgs.reshape(B * V, C, H, W).to(self.device)  # (B*V,C,H,W)
        target_flat = edited_flat.detach()  # (B*V,C,H,W) - 无梯度
        
        # 返回结果（edited_imgs 移回原设备供输出使用）
        return PreprocessedImages(
            pred=pred_flat,
            target=target_flat,
            edited_imgs=edited_imgs.to(source_device),
        )
    
    # =========================================================================
    # Loss 计算
    # =========================================================================
    
    def _compute_ssim_loss(
        self,
        pred: torch.Tensor,    # (B*V,C,H,W)
        target: torch.Tensor,  # (B*V,C,H,W)
    ) -> Optional[torch.Tensor]:
        """
        计算 SSIM loss（返回原始值，不乘权重）。
        
        SSIM 越高越好，所以 loss = 1 - SSIM
        
        Args:
            pred: 渲染图（有梯度）
            target: 编辑后图像（无梯度）
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.ssim_weight <= 0:
            return None
        
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)  # scalar
        return 1 - ssim_val  # 原始 loss，不乘权重
    
    def _compute_lpips_loss(
        self,
        pred: torch.Tensor,    # (B*V,C,H,W)
        target: torch.Tensor,  # (B*V,C,H,W)
    ) -> Optional[torch.Tensor]:
        """
        计算 LPIPS loss（返回原始值，不乘权重）。
        
        LPIPS 越低越好，直接作为 loss。
        
        Args:
            pred: 渲染图（有梯度），[0,1] 范围
            target: 编辑后图像（无梯度），[0,1] 范围
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.lpips_weight <= 0:
            return None
        
        # LPIPS 需要 [-1, 1] 范围
        pred_normalized = pred * 2 - 1      # [0,1] → [-1,1]
        target_normalized = target * 2 - 1
        
        lpips_val = self.lpips_fn(pred_normalized, target_normalized).mean()  # scalar
        return lpips_val  # 原始 loss，不乘权重
    
    def _compute_latent_mse_loss(
        self,
        pred: torch.Tensor,    # (B*V,C,H,W)
        target: torch.Tensor,  # (B*V,C,H,W)
    ) -> Optional[torch.Tensor]:
        """
        计算 Latent MSE loss（返回原始值，不乘权重）。
        
        在 VAE latent 空间计算 MSE。
        
        Args:
            pred: 渲染图（有梯度），[0,1] 范围
            target: 编辑后图像（无梯度），[0,1] 范围
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.latent_mse_weight <= 0:
            return None
        
        # 编码到 latent 空间
        pred_latent = self._encode_to_latent(pred)        # 有梯度
        target_latent = self._encode_to_latent(target)    # 无梯度
        
        latent_mse_val = F.mse_loss(pred_latent, target_latent.detach())
        return latent_mse_val  # 原始 loss，不乘权重
    
    def _encode_to_latent(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        编码到 VAE latent 空间。
        
        Args:
            imgs: 图像张量 (B,C,H,W)，float [0,1]
        
        Returns:
            torch.Tensor: latent 张量
        """
        # VAE 期望 [-1, 1] 范围
        imgs_normalized = imgs * 2 - 1  # [0,1] → [-1,1]
        latent = self.pipe.vae.encode(imgs_normalized).latent_dist.sample()
        return latent
    
    # =========================================================================
    # 主入口
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
        rank: int = 0,  # 兼容接口，本地版本忽略
    ) -> GuidanceResult:
        """
        计算 FlowEdit Guidance。
        
        流程：
        1. 图像预处理（FlowEdit 编辑 + 格式转换）
        2. 计算各项 loss（SSIM/LPIPS/Latent MSE）
        3. 返回 GuidanceResult
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
            rank: 分布式进程 rank（本地版本忽略）
        
        Returns:
            GuidanceResult: 包含编辑后图像和可微分 loss
        """
        # 1. 图像预处理
        preprocessed = self._preprocess_images(comp_rgb, condition_images)
        
        # 2. 计算各项 loss
        loss_ssim = self._compute_ssim_loss(preprocessed.pred, preprocessed.target)
        loss_lpips = self._compute_lpips_loss(preprocessed.pred, preprocessed.target)
        loss_latent_mse = self._compute_latent_mse_loss(preprocessed.pred, preprocessed.target)
        
        # 3. 返回结果
        return GuidanceResult(
            edited_imgs=preprocessed.edited_imgs,
            loss_ssim=loss_ssim,
            loss_lpips=loss_lpips,
            loss_latent_mse=loss_latent_mse,
        )
    
    # =========================================================================
    # Loss 权重查询
    # =========================================================================
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        获取各项 loss 的权重配置。
        
        Returns:
            dict: {"ssim": float, "lpips": float, "latent_mse": float}
        """
        return {
            "ssim": self.ssim_weight,
            "lpips": self.lpips_weight,
            "latent_mse": self.latent_mse_weight,
        }
    
    # =========================================================================
    # 资源清理
    # =========================================================================
    
    def cleanup(self) -> None:
        """释放模型显存"""
        print("[LocalGuidance] Cleaning up...")
        del self.pipe
        del self.lpips_fn
        torch.cuda.empty_cache()
        print("[LocalGuidance] Cleanup done.")
