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
        
        直接调用 pipeline 的 _encode_vae_image，确保 VAE encode + normalization 与 FlowEdit 完全一致。
        
        Args:
            imgs: 图像张量 (B,C,H,W)，float [0,1]
        
        Returns:
            torch.Tensor: packed latent 张量 (B, seq_len, C*4)
        """
        B = imgs.shape[0]  # scalar
        
        # Resize 到编辑分辨率（与 FlowEdit 工作分辨率一致）
        imgs_resized = F.interpolate(
            imgs, 
            size=(self.edit_resolution, self.edit_resolution), 
            mode='bilinear', 
            align_corners=False
        )  # (B,C,edit_res,edit_res)
        
        # 转换为 pipeline 期望的格式：[0,1] → [-1,1]，然后添加 frame 维度
        imgs_normalized = imgs_resized * 2 - 1  # (B,C,H,W), [0,1] → [-1,1]
        imgs_5d = imgs_normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # (B,C,1,H,W)
        
        # 使用 pipeline 的 _encode_vae_image：VAE encode + normalization
        latent_5d = self.pipe._encode_vae_image(imgs_5d, generator=None)  # (B,C',1,H',W'), normalized
        
        # Pack 到与 FlowEdit 相同的格式
        _, C_lat, _, H_lat, W_lat = latent_5d.shape  # (B,C',1,H',W')
        latent = self.pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)  # (B, seq_len, C*4)
        
        return latent  # 返回 normalized bf16 latent

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
