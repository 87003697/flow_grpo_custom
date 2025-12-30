"""
同进程多 GPU Guidance。

将 Qwen-Image-Edit 模型加载到 Guidance 设备上，
与 Trellis 训练进程共存于同一 Python 进程。

优势：
- 零序列化开销：Tensor 直接跨 GPU 传输
- 自动求导：无需手动计算梯度，PyTorch autograd 自动处理
- 简单调试：单进程，断点调试容易

设备分配策略（由 base.py compute_guidance_device 统一管理）：
- 前 N 张 GPU 给训练（Trellis DDP）
- 后 N 张 GPU 给 Guidance
- 例如 N=4: train=cuda:0-3, guidance=cuda:4-7
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim
import lpips

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult
from edit4shape.guidance.flowedit import QwenImageEditPlusPipeline


@dataclass
class PreprocessedImages:
    """预处理后的图像数据。"""
    rendered: torch.Tensor       # (B*V,C,H,W) 渲染图（Trellis 输出，在 guidance 设备上）
    edited: torch.Tensor         # (B*V,C,H,W) 编辑后图像（FlowEdit 输出，无梯度）
    edited_for_vis: torch.Tensor # (B,V,C,H,W) 用于可视化的编辑图像（在原设备上）
    edited_latent: torch.Tensor  # (B*V, seq_len, C) 编辑后的 packed latent


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
            cfg: 完整配置对象（需要 cfg.guidance.flowedit 和 cfg.train.loss）
            train_device: 训练使用的设备（用于计算 Guidance 设备）
        """
        self.cfg = cfg
        self.flowedit_cfg = cfg.guidance.flowedit
        self.loss_cfg = cfg.train.loss  # Loss 权重从 train.loss 读取
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        
        # ---- 1. 加载 Pipeline ----
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2509")
        print(f"[LocalGuidance] Loading Qwen-Image-Edit pipeline on {self.device}...")
        print(f"[LocalGuidance] 训练设备: {train_device}, Guidance 设备: {self.device}")
        print(f"[LocalGuidance] 模型路径: {model_path}")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[LocalGuidance] Pipeline loaded.")
        
        # ---- 2. LPIPS 模型 (fp32) ----
        print(f"[LocalGuidance] Loading LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
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
        
        # ---- 4. Loss 权重（从 cfg.train.loss 读取）----
        self.ssim_weight = self.loss_cfg.ssim
        self.lpips_weight = self.loss_cfg.lpips
        self.latent_mse_weight = self.loss_cfg.latent_mse
        
        # FlowEdit 的工作分辨率
        self.edit_resolution = cfg.guidance.get("edit_resolution", 1024)
    
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
    
    def _edit_single(
        self, 
        rendered_pil: Image.Image, 
        condition_pil: Image.Image,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        单张图像 FlowEdit 编辑。
        
        Args:
            rendered_pil: 渲染图（Trellis 输出）
            condition_pil: 条件图像（用户输入，指导编辑方向）
        
        Returns:
            (编辑后的图像, 编辑后的 packed latent)
        """
        # 处理可能存在的 Alpha 通道（变为白底 RGB，与 TRELLIS 预处理一致）
        condition_pil = composite_alpha_to_white(condition_pil)

        # Resize 到工作分辨率
        rendered_resized = rendered_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        condition_resized = condition_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        
        with torch.inference_mode():
            output = self.pipe(
                image_src=rendered_resized,
                image_tgt=condition_resized,
                prompt=self.prompt,
                generator=torch.manual_seed(self.seed),
                negative_prompt=" ",
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                true_cfg_scale_tgt=self.true_cfg_scale_tgt,
                n_min=self.n_min,
                n_max=self.n_max,
            )
        
        return output.images[0], output.latents  # 返回图像和 packed latent
    
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
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]，Trellis 输出
            condition_images: 条件图像列表 [len=B] of PIL.Image
        
        Returns:
            PreprocessedImages: 包含 rendered、edited、edited_for_vis 和 edited_latent
        """
        B, V, H, W, C = comp_rgb.shape
        source_device = comp_rgb.device
        
        # 转换格式：(B,V,H,W,C) -> (B,V,C,H,W)
        rendered_imgs = comp_rgb.permute(0, 1, 4, 2, 3)  # (B,V,C,H,W)
        
        # 收集并编辑所有图像
        edited_tensors = []
        edited_latents = []  # 收集 FlowEdit 返回的 packed latents
        for b in range(B):
            for v in range(V):
                rendered_tensor = rendered_imgs[b, v]  # (C,H,W)
                rendered_pil = self._tensor_to_pil(rendered_tensor)
                
                # FlowEdit 编辑，返回图像和 latent
                edited_pil, latent = self._edit_single(rendered_pil, condition_images[b])
                edited_latents.append(latent)  # latent shape: (1, seq_len, C)
                
                # Resize 回原始分辨率并转为 Tensor
                edited_pil_resized = edited_pil.resize((W, H), Image.LANCZOS)
                edited_tensor = self._pil_to_tensor(edited_pil_resized, self.device)  # (C,H,W)
                edited_tensors.append(edited_tensor)
        
        # 堆叠为 Tensor
        edited_flat = torch.stack(edited_tensors)  # (B*V,C,H,W)
        edited_for_vis = edited_flat.reshape(B, V, C, H, W)  # (B,V,C,H,W)
        
        # 堆叠 latents: (B*V, seq_len, C)
        edited_latent = torch.cat(edited_latents, dim=0)  # (B*V, seq_len, C)
        
        # 准备 loss 计算所需的张量
        rendered_flat = rendered_imgs.reshape(B * V, C, H, W).to(self.device)  # (B*V,C,H,W)
        edited = edited_flat.detach()  # (B*V,C,H,W) - 无梯度
        
        # 返回结果（edited_for_vis 移回原设备供可视化使用）
        return PreprocessedImages(
            rendered=rendered_flat,
            edited=edited,
            edited_for_vis=edited_for_vis.to(source_device),
            edited_latent=edited_latent,
        )
    
    # =========================================================================
    # Loss 计算
    # =========================================================================
    
    def _compute_ssim_loss(
        self,
        rendered: torch.Tensor,  # (B*V,C,H,W) 渲染图
        edited: torch.Tensor,    # (B*V,C,H,W) 编辑后图像
    ) -> Optional[torch.Tensor]:
        """
        计算 SSIM loss（返回原始值，不乘权重）。
        
        SSIM 越高越好，所以 loss = 1 - SSIM
        
        Args:
            rendered: 渲染图（Trellis 输出，有梯度）
            edited: 编辑后图像（FlowEdit 输出，无梯度）
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.ssim_weight <= 0:
            return None
        
        ssim_val = ssim(rendered, edited, data_range=1.0, size_average=True)  # scalar
        return 1 - ssim_val  # 原始 loss，不乘权重
    
    def _compute_lpips_loss(
        self,
        rendered: torch.Tensor,  # (B*V,C,H,W) 渲染图
        edited: torch.Tensor,    # (B*V,C,H,W) 编辑后图像
    ) -> Optional[torch.Tensor]:
        """
        计算 LPIPS loss（返回原始值，不乘权重）。
        
        LPIPS 越低越好，直接作为 loss。
        
        Args:
            rendered: 渲染图（Trellis 输出，有梯度），[0,1] 范围
            edited: 编辑后图像（FlowEdit 输出，无梯度），[0,1] 范围
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.lpips_weight <= 0:
            return None
        
        # LPIPS 需要 [-1, 1] 范围
        rendered_normalized = rendered * 2 - 1  # [0,1] → [-1,1]
        edited_normalized = edited * 2 - 1
        
        lpips_val = self.lpips_fn(rendered_normalized, edited_normalized).mean()  # scalar
        return lpips_val  # 原始 loss，不乘权重
    
    def _compute_latent_mse_loss(
        self,
        rendered: torch.Tensor,   # (B*V,C,H,W) 渲染图
        edited_latent: torch.Tensor,  # (B*V, seq_len, C) packed latent
    ) -> Optional[torch.Tensor]:
        """
        计算 Latent MSE loss（返回原始值，不乘权重）。
        
        在 VAE latent 空间计算 MSE。优化：直接使用 FlowEdit 返回的 packed latent，
        避免对编辑后图像的冗余编码。
        
        Args:
            rendered: 渲染图（Trellis 输出，有梯度），[0,1] 范围
            edited_latent: FlowEdit 返回的编辑后 packed latent（无梯度）
        
        Returns:
            原始标量 loss，如果 weight=0 则返回 None
        """
        if self.latent_mse_weight <= 0:
            return None
        
        # 将渲染图编码到 packed latent 格式（与 FlowEdit 输出格式一致）
        rendered_latent = self._encode_to_latent_packed(rendered)  # (B*V, seq_len, C)
        
        latent_mse_val = F.mse_loss(rendered_latent, edited_latent.detach())
        return latent_mse_val  # 原始 loss，不乘权重
    
    def _encode_to_latent_packed(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        编码到 packed latent 格式（与 FlowEdit 输出格式一致）。
        
        Args:
            imgs: 图像张量 (B,C,H,W)，float [0,1]
        
        Returns:
            torch.Tensor: packed latent 张量 (B, seq_len, C*4)
        """
        B = imgs.shape[0]
        
        # Resize 到编辑分辨率（与 FlowEdit 工作分辨率一致）
        imgs_resized = F.interpolate(
            imgs, 
            size=(self.edit_resolution, self.edit_resolution), 
            mode='bilinear', 
            align_corners=False
        )  # (B,C,edit_res,edit_res)
        
        # VAE encode
        imgs_normalized = imgs_resized * 2 - 1  # [0,1] → [-1,1]
        imgs_5d = imgs_normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # (B,C,1,H,W)
        latent_5d = self.pipe.vae.encode(imgs_5d).latent_dist.sample()   # (B,C',1,H',W')
        latent = latent_5d.squeeze(2)  # (B,C',H',W')
        
        # Pack 到与 FlowEdit 相同的格式
        _, C_lat, H_lat, W_lat = latent.shape
        latent = latent.view(B, C_lat, H_lat // 2, 2, W_lat // 2, 2)
        latent = latent.permute(0, 2, 4, 1, 3, 5)
        latent = latent.reshape(B, (H_lat // 2) * (W_lat // 2), C_lat * 4)  # (B, seq_len, C*4)
        
        return latent.to(dtype=imgs.dtype)
    
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
        loss_ssim = self._compute_ssim_loss(preprocessed.rendered, preprocessed.edited)
        loss_lpips = self._compute_lpips_loss(preprocessed.rendered, preprocessed.edited)
        # Latent MSE: 直接使用 FlowEdit 返回的 packed latent，避免冗余编码
        loss_latent_mse = self._compute_latent_mse_loss(preprocessed.rendered, preprocessed.edited_latent)
        
        # 3. 返回结果
        return GuidanceResult(
            edited_imgs=preprocessed.edited_for_vis,
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
