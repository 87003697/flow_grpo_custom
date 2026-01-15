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
from edit4shape.guidance.backends.pipeline_adapters import create_pipeline_adapter
from edit4shape.guidance.flowedit.state_tracker import FlowEditStateTracker
from edit4shape.guidance.metric import create_metrics


@dataclass
class PreprocessedImages:
    """
    预处理后的图像数据。
    
    Latent 格式说明:
        - packed:   [B, seq_len, C_lat]  其中 seq_len = H_lat * W_lat
        - unpacked: [B, C_lat, T, H_lat, W_lat]  标准 VAE latent 格式
    """
    rendered: torch.Tensor                   # [B*V, C, H, W] unpacked 图像（Trellis 渲染输出）
    edited: torch.Tensor                     # [B*V, C, H, W] unpacked 图像（FlowEdit 编辑输出，无梯度）
    edited_for_vis: torch.Tensor             # [B, V, C, H, W] unpacked 图像（用于可视化，在原设备上）
    edited_latent: torch.Tensor              # [B*V, seq_len, C_lat] packed latent（最终编辑结果）
    trackers: List[FlowEditStateTracker]     # len = B*V，每个 tracker 的 latents 都是 packed 格式


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
        self.latent_mse_mode = cfg.train.loss.latent_mse_mode  # "final" | "mean" | "weighted"
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
    ) -> Tuple[Image.Image, torch.Tensor, FlowEditStateTracker]:
        """
        单张图像 FlowEdit 编辑。
        
        Args:
            rendered_pil: 渲染图（Trellis 输出）
            condition_pil: 条件图像（用户输入，指导编辑方向）
        
        Returns:
            (编辑后的图像, 编辑后的 packed latent, StateTracker)
        """
        # 处理可能存在的 Alpha 通道（变为白底 RGB，与 TRELLIS 预处理一致）
        condition_pil = composite_alpha_to_white(condition_pil)

        # Resize 到工作分辨率
        rendered_resized = rendered_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        condition_resized = condition_pil.resize((self.edit_resolution, self.edit_resolution), Image.LANCZOS)
        
        with torch.inference_mode():
            result = self.adapter.edit(rendered_resized, condition_resized, self.flowedit_cfg)
        
        return result.image, result.latent, result.tracker
    
    # =========================================================================
    # 图像预处理
    # =========================================================================
    
    def _pil_to_tensor_resized(self, pil_img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
        """PIL 图像 resize 后转为 tensor (C,H,W)"""
        return self._pil_to_tensor(pil_img.resize(size, Image.LANCZOS), self.device)
    
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
            PreprocessedImages: 包含 rendered、edited、edited_for_vis、edited_latent、trackers
        """
        B, V, H, W, C = comp_rgb.shape
        source_device = comp_rgb.device
        target_size = (W, H)
        
        # (B,V,H,W,C) -> (B,V,C,H,W)
        rendered_imgs = comp_rgb.permute(0, 1, 4, 2, 3)
        
        # 收集编辑结果
        tensors, latents, trackers = [], [], []
        
        for b in range(B):
            for v in range(V):
                rendered_pil = self._tensor_to_pil(rendered_imgs[b, v])
                edited_pil, latent, tracker = self._edit_single(rendered_pil, condition_images[b])
                
                tensors.append(self._pil_to_tensor_resized(edited_pil, target_size))
                latents.append(latent)
                trackers.append(tracker)
        
        # 堆叠结果
        edited_flat = torch.stack(tensors)  # (B*V,C,H,W)
        edited_latent = torch.cat(latents, dim=0)  # (B*V,seq_len,C)
        
        return PreprocessedImages(
            rendered=rendered_imgs.reshape(B * V, C, H, W).to(self.device),
            edited=edited_flat.detach(),
            edited_for_vis=edited_flat.reshape(B, V, C, H, W).to(source_device),
            edited_latent=edited_latent,
            trackers=trackers,
        )
    
    # =========================================================================
    # Latent 编码（供 LatentMSEMetric 使用）
    # =========================================================================
    
    def _encode_to_latent_packed(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 packed latent 格式（与 FlowEdit 输出格式一致）。
        
        转换流程:
            [B, C, H, W] unpacked 图像
            -> [B, C, 1, H, W] 添加 frame 维度
            -> [B, C_lat, 1, H_lat, W_lat] VAE encode (unpacked latent)
            -> [B, seq_len, C_lat] pack (packed latent)
        
        Args:
            imgs: 图像张量, shape: [B, C, H, W], float [0,1], unpacked 图像格式
        
        Returns:
            packed latent, shape: [B, seq_len, C_lat], 其中 seq_len = H_lat * W_lat
        """
        B = imgs.shape[0]
        
        # Resize 到编辑分辨率
        imgs_resized = F.interpolate(
            imgs, 
            size=(self.edit_resolution, self.edit_resolution), 
            mode='bilinear', 
            align_corners=False
        )  # [B, C, edit_res, edit_res] unpacked 图像
        
        # [0,1] → [-1,1]，添加 frame 维度
        imgs_normalized = imgs_resized * 2 - 1  # [B, C, H, W] unpacked 图像
        imgs_5d = imgs_normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # [B, C, 1, H, W] unpacked 图像
        
        # VAE encode: unpacked 图像 -> unpacked latent
        latent_5d = self.pipe._encode_vae_image(imgs_5d, generator=None)  # [B, C_lat, 1, H_lat, W_lat] unpacked latent
        
        # Pack: unpacked latent -> packed latent
        _, C_lat, _, H_lat, W_lat = latent_5d.shape
        latent = self.pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)  # [B, seq_len, C_lat] packed latent
        
        return latent  # 返回 normalized bf16 latent
    
    # =========================================================================
    # Latent MSE Loss 计算（支持多步监督）
    # =========================================================================
    
    def _compute_latent_mse_loss(
        self,
        rendered: torch.Tensor,                      # [B*V, C, H, W] unpacked 图像格式
        edited_latent: torch.Tensor,                 # [B*V, seq_len, C_lat] packed latent
        trackers: List[FlowEditStateTracker],        # len = B*V
    ) -> torch.Tensor:
        """
        根据 latent_mse_mode 计算 Latent MSE loss。
        
        Latent 格式说明:
            - packed:   [B, seq_len, C_lat]  其中 seq_len = H_lat * W_lat
            - unpacked: [B, C_lat, T, H_lat, W_lat]  标准 VAE latent 格式
        
        Args:
            rendered: 渲染图像, shape: [B*V, C, H, W] (unpacked 图像，需要编码)
            edited_latent: 最终编辑后的 packed latent, shape: [B*V, seq_len, C_lat]
            trackers: 每个样本的 StateTracker, len = B*V
        
        Returns:
            标量 loss
        """
        # 编码渲染图到 packed latent
        # rendered: [B*V, C, H, W] unpacked 图像 -> [B*V, seq_len, C_lat] packed latent
        rendered_latent = self._encode_to_latent_packed(rendered)  # [B*V, seq_len, C_lat] packed
        
        if self.latent_mse_mode == "final":
            # 只用最终 latent（与原行为一致）
            # rendered_latent: [B*V, seq_len, C_lat] packed
            # edited_latent:   [B*V, seq_len, C_lat] packed
            loss = F.mse_loss(rendered_latent.float(), edited_latent.detach().float())  # scalar
        
        elif self.latent_mse_mode == "mean":
            # 所有中间步均匀加权
            losses = []
            for i, tracker in enumerate(trackers):
                # rendered_latent[i:i+1]: [1, seq_len, C_lat] packed，单样本
                # tracker.latents[k]:     [1, seq_len, C_lat] packed
                sample_loss = tracker.loss_mean(rendered_latent[i:i+1])  # scalar
                losses.append(sample_loss)
            loss = torch.stack(losses).mean()  # scalar
        
        elif self.latent_mse_mode == "weighted":
            # 用编辑次数的倒数加权
            losses = []
            for i, tracker in enumerate(trackers):
                # rendered_latent[i:i+1]: [1, seq_len, C_lat] packed，单样本
                sample_loss = tracker.loss_weighted(rendered_latent[i:i+1])  # scalar
                losses.append(sample_loss)
            loss = torch.stack(losses).mean()  # scalar
        
        else:
            raise ValueError(f"Unknown latent_mse_mode: {self.latent_mse_mode}")
        
        return loss  # scalar
    
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
                # 根据 latent_mse_mode 选择计算方式
                losses[name] = self._compute_latent_mse_loss(
                    preprocessed.rendered, 
                    preprocessed.edited_latent,
                    preprocessed.trackers,
                )
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
            trackers=preprocessed.trackers,
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
        weights = {name: self.metrics[name].weight if name in self.metrics else 0.0 for name in all_names}
        return weights
    
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
