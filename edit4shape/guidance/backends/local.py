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

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult
from edit4shape.guidance.flowedit import FlowEditPipeline
from edit4shape.guidance.metric import create_metrics


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
    - 通过 metric 模块统一计算 loss（SSIM/LPIPS/Latent MSE/DINO）
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
        self.pipe = FlowEditPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[LocalGuidance] Pipeline loaded.")
        
        # ---- 2. 算法参数 ----
        self.prompt = self.flowedit_cfg.prompt
        self.seed = self.flowedit_cfg.seed
        self.steps = self.flowedit_cfg.steps
        self.guidance_scale = self.flowedit_cfg.guidance_scale
        self.true_cfg_scale_tgt = self.flowedit_cfg.true_cfg_scale_tgt
        self.n_min = self.flowedit_cfg.n_min
        self.n_max = self.flowedit_cfg.n_max
        self.noise_mode = self.flowedit_cfg.get("noise_mode", "random")
        
        # FlowEdit 的工作分辨率
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
                noise_mode=self.noise_mode,
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
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
        rank: int = 0,  # 兼容接口，本地版本忽略
    ) -> GuidanceResult:
        """
        计算 FlowEdit Guidance。
        
        流程：
        1. 图像预处理（FlowEdit 编辑 + 格式转换）
        2. 通过 metric 模块计算各项 loss
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
    # Loss 权重查询
    # =========================================================================
    
    def get_loss_weights(self) -> Dict[str, float]:
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
    
    def cleanup(self) -> None:
        """释放模型显存"""
        print("[LocalGuidance] Cleaning up...")
        del self.pipe
        for metric in self.metrics.values():
            metric.cleanup()
        self.metrics.clear()
        torch.cuda.empty_cache()
        print("[LocalGuidance] Cleanup done.")
