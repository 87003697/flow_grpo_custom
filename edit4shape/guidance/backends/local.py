import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult
from edit4shape.guidance.backends.pipeline_adapters import create_pipeline_adapter
from edit4shape.guidance.metric import create_metrics


@dataclass
class PreprocessedImages:
    rendered: torch.Tensor          # (B*V, C, H, W) [0, 1]
    edited: torch.Tensor            # (B*V, C, H, W) [0, 1], detached
    edited_latent: torch.Tensor     # (B*V, seq_len, C), detached
    edited_for_vis: torch.Tensor    # (B*V, C, H, W) [0, 1], detached, cpu


class LocalGuidance:
    """
    本地 Guidance 后端，在同一进程的另一个 GPU 上运行 FlowEdit。
    
    特点：
    - Qwen-Image-Edit 自动加载到 train_device + 1
    - 通过 metric 模块统一计算 loss（SSIM/LPIPS/Latent MSE/DINO）
    - PyTorch autograd 自动处理梯度
    """
    
    def __init__(self, cfg, train_device: torch.device):
        """
        初始化 LocalGuidance。
        
        Args:
            cfg: 全局配置对象
            train_device: 训练进程所在的设备 (cuda:N)
        """
        self.flowedit_cfg = cfg.guidance.flowedit
        self.loss_cfg = cfg.train.loss
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        
        # ---- 1. 创建 Pipeline 适配器 ----
        pipeline_type = self.flowedit_cfg.get("pipeline_type", "simple")
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2509")
        
        print(f"[LocalGuidance] Loading Qwen-Image-Edit pipeline on {self.device}...")
        print(f"[LocalGuidance] 训练设备: {train_device}, Guidance 设备: {self.device}")
        print(f"[LocalGuidance] Pipeline 类型: {pipeline_type}, 模型路径: {model_path}")
        
        self.adapter = create_pipeline_adapter(pipeline_type)
        self.adapter.load(model_path, self.device)
        self.pipe = self.adapter.pipe  # 保留 pipe 引用供 _encode_to_latent_packed 使用
        
        print(f"[LocalGuidance] Pipeline loaded.")
        
        # ---- 2. FlowEdit 工作分辨率 ----
        self.edit_resolution = cfg.guidance.get("edit_resolution", 1024)
        
        # ---- 3. 创建 Metrics（根据权重按需创建）----
        print(f"[LocalGuidance] Creating metrics...")
        self.metrics = create_metrics(
            self.loss_cfg,
            self.device,
            extra_kwargs={
                "dino": {
                    "model_path": cfg.guidance.get("dino_model_path", "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"),
                    "image_size": cfg.guidance.get("dino_image_size", 518),
                },
                "latent_mse": {
                    "encode_fn": self._encode_to_latent_packed,
                },
            },
        )
        print(f"[LocalGuidance] Created metrics: {list(self.metrics.keys())}")
    
    # =========================================================================
    # 图像格式转换
    # =========================================================================
    
    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将 Tensor 转换为 PIL Image。
        Args:
            tensor: (C, H, W) 范围 [0, 1]
        """
        return TF.to_pil_image(tensor.clamp(0, 1).cpu())
    
    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """
        将 PIL Image 转换为 Tensor。
        Args:
            pil_img: PIL Image
        Returns:
            tensor: (C, H, W) 范围 [0, 1], device=self.device
        """
        return TF.to_tensor(pil_img).to(self.device)
    
    # =========================================================================
    # 核心编辑逻辑
    # =========================================================================
    
    def _edit_images(
        self,
        rendered_pil: Image.Image,
        condition_pil: Image.Image,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        调用 Pipeline 进行编辑。
        
        Args:
            rendered_pil: 渲染图 (PIL)
            condition_pil: 条件图 (PIL)
            
        Returns:
            edited_image: (C, H, W) 编辑后图像，范围 [0, 1]
            edited_latent: (seq_len, C) 编辑后 packed latent
        """
        rendered_resized = rendered_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        condition_resized = condition_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        
        with torch.inference_mode():
            result = self.adapter.edit(rendered_resized, condition_resized, self.flowedit_cfg)
        
        return result.image, result.latent
    
    # =========================================================================
    # 图像预处理
    # =========================================================================
    
    def _preprocess_images(
        self,
        comp_rgb: torch.Tensor,     # (B, V, H, W, 3) 渲染图
        condition_images: list,     # List[PIL.Image] 条件图
    ) -> PreprocessedImages:
        """
        预处理：展平 batch -> 遍历编辑 -> 拼接 -> 调整大小。
        """
        B, V, H, W, C = comp_rgb.shape
        comp_rgb_flat = comp_rgb.view(B * V, H, W, C).permute(0, 3, 1, 2)  # (N, C, H, W)
        
        edited_list = []
        edited_latent_list = []
        
        # 逐张编辑 (TODO: 优化为 batch 编辑)
        for i in range(B * V):
            rendered_tensor = comp_rgb_flat[i]  # (3, H, W)
            rendered_pil = self._to_pil(rendered_tensor)
            condition_pil = condition_images[i]
            
            # FlowEdit 编辑
            edited_tensor, edited_latent = self._edit_images(rendered_pil, condition_pil)
            
            edited_list.append(edited_tensor)
            edited_latent_list.append(edited_latent)
            
        # 拼接结果
        edited_batch = torch.stack(edited_list)  # (N, 3, H_edit, W_edit)
        edited_latent_batch = torch.stack(edited_latent_list)  # (N, seq_len, C)
        
        # 将渲染图调整到编辑分辨率，以便计算 loss
        rendered_resized = F.interpolate(
            comp_rgb_flat,
            size=(self.edit_resolution, self.edit_resolution),
            mode='bilinear',
            align_corners=False
        )
        
        return PreprocessedImages(
            rendered=rendered_resized,
            edited=edited_batch,  # detached inside _edit_images/adapter
            edited_latent=edited_latent_batch,
            edited_for_vis=edited_batch.detach().cpu()
        )
    
    # =========================================================================
    # Latent 编码（供 LatentMSEMetric 使用）
    # =========================================================================
    
    def _encode_to_latent_packed(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        编码到 packed latent 格式（与 FlowEdit 输出格式一致）。
        使用 Qwen-Image-Edit 自带的 VAE 和 PatchEmbed。
        
        Args:
            imgs: (N, 3, H, W) [0,1]
            
        Returns:
            (N, seq_len, C) packed latent
        """
        # 1. VAE Encode
        # images needs to be [-1, 1]
        imgs_norm = imgs * 2.0 - 1.0
        
        with torch.no_grad():
            posterior = self.pipe.vae.encode(imgs_norm.to(self.device, dtype=self.pipe.vae.dtype)).latent_dist
            latents = posterior.sample() * self.pipe.vae.config.scaling_factor  # (N, 4, h, w)
            
        # 2. Patch Embedding (flatten to sequence)
        # Qwen-Image-Edit 使用 patch_embed 模块将 latent 展平为序列
        # latents: (N, 4, 128, 128) -> (N, 4096, 1152) ? 
        # 具体实现依赖 model.patch_embed
        
        # 注意：这里我们假设 pipe 是原始 FlowEditPipeline
        # 如果 self.pipe 是 patch 后的，可能需要直接调用 patch_embed
        # Qwen2VLForConditionalGeneration 的 patch_embed 在 visual.patch_embed
        
        # 实际上 FlowEditPipeline 的 patchify 逻辑如下：
        # latents (B, C, H, W) -> (B, H*W, C) or similar
        # 查看 pipeline 源码，它是在 Qwen2VL 内部处理的。
        # 为了计算 latent MSE，我们需要模拟这个过程，或者直接比较 VAE latent。
        
        # ★ 修正：为了简单起见，且 pipeline 输出的是 patch 后的 latent，
        # 我们这里也应该输出 patch 后的。
        # 但 Qwen 的 patch 逻辑比较复杂（涉及 3D 卷积等）。
        # 简单起见，我们暂时直接比较 VAE latent (flattened)，
        # 或者在 adapter 中统一 latent 格式。
        
        # 经查阅 adapter 实现，simple adapter 返回的是 VAE latent (N, 4, H, W)。
        # 原始 pipeline 返回的也是 VAE latent。
        # 所以这里直接展平即可，或者保持形状。
        
        # 为了兼容性，统一展平为 (N, L, C)
        N, C, H, W = latents.shape
        latents_packed = latents.view(N, C, -1).permute(0, 2, 1)  # (N, H*W, C)
        
        return latents_packed

    # =========================================================================
    # 主入口
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,   # (B, V, H, W, 3) 渲染图
        condition_images: list,   # List[PIL.Image] 条件图 (原图)
    ) -> GuidanceResult:
        """
        计算 Guidance Loss。
        
        流程：
        1. 图像预处理（FlowEdit 编辑 + 格式转换）
        2. 通过 metric 模块计算各项 loss
        3. 返回 GuidanceResult
        
        Args:
            comp_rgb: 渲染图 (B, V, H, W, 3)
            condition_images: 条件图列表
        """
        # 1. 图像预处理
        preprocessed = self._preprocess_images(comp_rgb, condition_images)
        
        # 2. 通过 metric 模块计算各项 loss
        losses = {}
        for name, metric in self.metrics.items():
            if name == "latent_mse":
                # LatentMSEMetric 的 target 是 latent
                losses[name] = metric.compute(preprocessed.rendered, preprocessed.edited_latent)
            else:
                # 其他 metric 的 target 是图像
                losses[name] = metric.compute(preprocessed.rendered, preprocessed.edited)
        
        # 3. 返回结果（未启用的 metric 为 None）
        return GuidanceResult(
            edited_imgs=preprocessed.edited_for_vis,
            loss_ssim=losses.get("ssim"),
            loss_lpips=losses.get("lpips"),
            loss_latent_mse=losses.get("latent_mse"),
            loss_dino=losses.get("dino"),
        )
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def get_guidance_weights(self) -> Dict[str, float]:
        """
        获取各项 loss 的权重配置。
        
        Returns:
            dict: {name: weight} 包含所有可能的 metrics（未创建的为 0）
        """
        # 返回所有可能的 metric 权重，保证向后兼容
        all_names = ["ssim", "lpips", "latent_mse", "dino"]
        return {name: self.metrics[name].weight if name in self.metrics else 0.0 for name in all_names}
    
    # =========================================================================
    # 资源清理
    # =========================================================================
    
    def cleanup(self):
        """释放模型显存"""
        print("[LocalGuidance] Cleaning up...")
        del self.pipe
        for metric in self.metrics.values():
            metric.cleanup()
        self.metrics.clear()
        torch.cuda.empty_cache()
        print("[LocalGuidance] Cleanup done.")
