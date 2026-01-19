"""
FlowEdit Guidance 模块。

将 Qwen-Image-Edit 模型加载到 Guidance 设备上，
与 Trellis 训练进程共存于同一 Python 进程。

数据流:
    Phase 1 (无梯度): rendered → encode → latent_before → FlowEdit → latent_after
    Phase 2 (有梯度): rendered → encode → latent_before → Loss(latent_after)
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.systems.base import compute_guidance_device
from edit4shape.guidance.base import GuidanceResult, BaseGuidance, SpecifyGradient
from edit4shape.guidance.pipeline_parallel import PipelineParallelMixin
from edit4shape.guidance.pipelines.adapters import create_pipeline_adapter
from edit4shape.guidance.pipelines.qwen_image_edit.state_tracker import FlowEditStateTracker
from edit4shape.guidance.metric import create_metrics


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EditOutput:
    """单张图编辑输出"""
    image: Image.Image                  # 编辑后的图像
    latent: torch.Tensor                # [1, seq, C] 编辑后的 latent
    tracker: FlowEditStateTracker       # 中间状态跟踪器


@dataclass
class FlowEditBatchResult:
    """批量编辑结果"""
    edited_images: List[Image.Image]    # [N] 编辑后的图像
    edited_tensor: torch.Tensor         # [N, C, H, W] 编辑后的图像 tensor
    latent_after: torch.Tensor          # [N, seq, C] 编辑后的 latent
    trackers: List[FlowEditStateTracker]  # [N] 中间状态跟踪器


# =============================================================================
# FlowEditGuidance
# =============================================================================

class FlowEditGuidance(BaseGuidance):
    """
    FlowEdit Guidance。
    
    数据流:
        Phase 1 (无梯度): rendered → encode → latent_before → FlowEdit → latent_after
        Phase 2 (有梯度): rendered → encode → latent_before → Loss(latent_after)
    
    模块划分:
        - 格式转换: tensor_to_pil, pils_to_tensor
        - Latent 编码: encode_to_latent
        - FlowEdit 编辑: edit_single, run_flowedit_batch
        - Loss 计算: compute_latent_mse, compute_image_losses, compute_total_loss
        - 主入口: compute_guidance
    """
    
    def __init__(self, cfg: Any, train_device: torch.device):
        """
        初始化 FlowEdit Guidance。
        
        Args:
            cfg: 完整配置对象
            train_device: 训练使用的设备
        """
        self.cfg = cfg
        self.flowedit_cfg = cfg.guidance.flowedit
        self.loss_cfg = cfg.train.loss
        self.latent_mse_mode = cfg.train.loss.latent_mse_mode
        self.train_device = train_device
        self.device = compute_guidance_device(train_device)
        
        # 加载 Pipeline
        pipeline_type = self.flowedit_cfg.get("pipeline_type", "simple")
        model_path = cfg.guidance.get("model_path", "Qwen/Qwen-Image-Edit-2509")
        
        print(f"[FlowEditGuidance] Loading pipeline on {self.device}...")
        print(f"[FlowEditGuidance] Pipeline: {pipeline_type}, Model: {model_path}")
        
        self.adapter = create_pipeline_adapter(pipeline_type)
        self.adapter.load(model_path, self.device)
        self.pipe = self.adapter.pipe
        
        # 编辑分辨率
        self.edit_resolution = cfg.guidance.edit_resolution
        
        # 是否使用 autograd 计算梯度（True: 预计算梯度 + SpecifyGradient 注入，False: 正常 autograd）
        self.enable_autograd = cfg.guidance.get("enable_autograd", True)
        
        # 创建 Metrics (SSIM, LPIPS, DINO, latent_mse)
        self.metrics = create_metrics(
            self.loss_cfg,
            self.device,
            extra_kwargs={
                "dino": {
                    "model_path": cfg.guidance.get("dino_model_path", "pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"),
                    "image_size": cfg.guidance.get("dino_image_size", 518),
                },
                "latent_mse": {
                    "encode_fn": self.encode_to_latent,
                },
            },
        )
        print(f"[FlowEditGuidance] Metrics: {list(self.metrics.keys())}")
    
    # =========================================================================
    # 格式转换模块
    # =========================================================================
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """[C, H, W] float [0,1] → PIL"""
        arr = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr.transpose(1, 2, 0))
    
    def pils_to_tensor(self, pils: List[Image.Image], size: Tuple[int, int]) -> torch.Tensor:
        """List[PIL] → [N, C, H, W]"""
        tensors = [TF.to_tensor(p.resize(size, Image.LANCZOS)) for p in pils]
        return torch.stack(tensors).to(self.device)
    
    # =========================================================================
    # Latent 编码模块
    # =========================================================================
    
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 packed latent。
        
        Args:
            images: [B, C, H, W] float [0,1]
        
        Returns:
            [B, seq, C_lat] packed latent
        
        Note:
            使用 bicubic + antialias 插值，与 FlowEdit 内部编码保持一致。
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
        
        # VAE encode
        latent_5d = self.pipe._encode_vae_image_differentiable(images_5d)  # [B, C_lat, 1, H_lat, W_lat]
        
        # Pack
        _, C_lat, _, H_lat, W_lat = latent_5d.shape
        latent = self.pipe._pack_latents(latent_5d, B, C_lat, H_lat, W_lat)  # [B, seq, C_lat]
        
        return latent.to(dtype=images.dtype)
    
    # =========================================================================
    # FlowEdit 编辑模块
    # =========================================================================
    
    def edit_single(
        self,
        rendered_pil: Image.Image,
        condition_pil: Image.Image,
        latent_before: torch.Tensor,
    ) -> EditOutput:
        """
        执行单张图的 FlowEdit 编辑。
        
        Args:
            rendered_pil: 渲染图 PIL（供 VLM 使用）
            condition_pil: 条件图 PIL
            latent_before: 渲染图的 latent [1, seq, C]
        
        Returns:
            EditOutput
        """
        condition_pil = composite_alpha_to_white(condition_pil)
        
        # 确保 latent 是 bfloat16（与 FlowEdit pipeline 一致）
        latent_before = latent_before.to(dtype=torch.bfloat16)
        
        result = self.adapter.edit(
            rendered_pil,
            condition_pil,
            self.flowedit_cfg,
            src_latent=latent_before,
        )
        
        return EditOutput(
            image=result.image,
            latent=result.latent,
            tracker=result.tracker,
        )
    
    @torch.no_grad()
    def run_flowedit_batch(
        self,
        rendered: torch.Tensor,
        condition_images: List[Image.Image],
        B: int,
        V: int,
    ) -> FlowEditBatchResult:
        """
        批量执行 FlowEdit 编辑（无梯度）。
        
        Args:
            rendered: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            B: batch size
            V: views per sample
        
        Returns:
            FlowEditBatchResult
        """
        N = B * V
        H, W = rendered.shape[2], rendered.shape[3]
        
        edited_images, latents, trackers = [], [], []
        
        for b in range(B):
            for v in range(V):
                i = b * V + v
                
                # 编码渲染图
                latent_before = self.encode_to_latent(rendered[i:i+1])
                
                # 编辑
                rendered_pil = self.tensor_to_pil(rendered[i])
                output = self.edit_single(
                    rendered_pil,
                    condition_images[b],
                    latent_before,
                )
                
                edited_images.append(output.image)
                latents.append(output.latent)
                trackers.append(output.tracker)
        
        # 转换编辑后的图像为 tensor
        edited_tensor = self.pils_to_tensor(edited_images, (W, H))
        
        return FlowEditBatchResult(
            edited_images=edited_images,
            edited_tensor=edited_tensor,
            latent_after=torch.cat(latents, dim=0),
            trackers=trackers,
        )
    
    # =========================================================================
    # Loss 计算模块
    # =========================================================================
    
    def compute_latent_mse(
        self,
        latent_before: torch.Tensor,
        latent_after: torch.Tensor,
        trackers: List[FlowEditStateTracker],
    ) -> torch.Tensor:
        """
        计算 Latent MSE Loss。
        
        Args:
            latent_before: [N, seq, C] 有梯度
            latent_after: [N, seq, C] 无梯度
            trackers: [N] 中间状态
        
        Returns:
            标量 loss
        """
        latent_after = latent_after.detach()
        
        if self.latent_mse_mode == "final":
            return F.mse_loss(latent_before.float(), latent_after.float())
        
        # mean / weighted / ada / ada_position: 使用 tracker 的方法
        losses = []
        for i, tracker in enumerate(trackers):
            single = latent_before[i:i+1]
            if self.latent_mse_mode == "mean":
                losses.append(tracker.loss_mean(single))
            elif self.latent_mse_mode == "weighted":
                losses.append(tracker.loss_weighted(single))
            elif self.latent_mse_mode == "ada":
                losses.append(tracker.loss_ada(single))
            elif self.latent_mse_mode == "ada_position":
                losses.append(tracker.loss_ada_position(single))
            else:
                raise ValueError(f"Unknown latent_mse_mode: {self.latent_mse_mode}")
        
        return torch.stack(losses).mean()
    
    def compute_image_losses(
        self,
        rendered: torch.Tensor,
        edited: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算图像空间的 loss (SSIM, LPIPS, DINO)。
        
        Args:
            rendered: [N, C, H, W] 有梯度
            edited: [N, C, H, W] 无梯度
        
        Returns:
            {name: loss}
        """
        losses = {}
        for name, metric in self.metrics.items():
            if name != "latent_mse":
                losses[name] = metric.compute(rendered, edited.detach())
        return losses
    
    def compute_total_loss(
        self,
        rendered: torch.Tensor,
        batch_result: FlowEditBatchResult,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算并汇总所有 loss（有梯度）。
        
        Args:
            rendered: [N, C, H, W] 有梯度
            batch_result: FlowEdit 编辑结果
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Latent MSE
        if "latent_mse" in self.metrics:
            latent_before = self.encode_to_latent(rendered)
            loss_dict["latent_mse"] = self.compute_latent_mse(
                latent_before,
                batch_result.latent_after,
                batch_result.trackers,
            )
        
        # 图像空间 loss
        image_losses = self.compute_image_losses(rendered, batch_result.edited_tensor)
        loss_dict.update(image_losses)
        
        # 加权汇总
        total = torch.tensor(0.0, device=self.device)
        for name, loss in loss_dict.items():
            weight = self.metrics[name].weight if name in self.metrics else 0.0
            total = total + weight * loss
        
        return total, loss_dict
    
    # =========================================================================
    # 主入口
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        rank: int = 0,
    ) -> GuidanceResult:
        """
        计算 FlowEdit Guidance。
        
        Args:
            comp_rgb: [B, V, H, W, C] Trellis 渲染图像
            condition_images: [B] 条件图像
            rank: 分布式进程 rank（本地版本忽略）
        
        Returns:
            GuidanceResult
        """
        B, V, H, W, C = comp_rgb.shape
        N = B * V
        source_device = comp_rgb.device
        
        # 格式转换: [B, V, H, W, C] → [N, C, H, W]
        rendered = comp_rgb.permute(0, 1, 4, 2, 3).reshape(N, C, H, W).to(self.device)
        
        # =====================================================================
        # Phase 1: FlowEdit 编辑（无梯度，省显存）
        # =====================================================================
        batch_result = self.run_flowedit_batch(rendered, condition_images, B, V)
        
        # =====================================================================
        # Phase 2: Loss 计算（有梯度）
        # =====================================================================
        total_loss, loss_dict = self.compute_total_loss(rendered, batch_result)
        
        # 可选：使用 SpecifyGradient 注入梯度，释放计算图
        if self.enable_autograd:
            grad = torch.autograd.grad(total_loss, rendered)[0]
            total_loss = SpecifyGradient.apply(rendered, grad.detach()).sum()
        
        # =====================================================================
        # 组装返回结果
        # =====================================================================
        edited_for_vis = batch_result.edited_tensor.reshape(B, V, C, H, W).to(source_device)
        
        return GuidanceResult(
            loss=total_loss,
            edited_imgs=edited_for_vis,
            loss_dict=loss_dict,
            trackers=batch_result.trackers,
        )
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取各项 loss 的权重"""
        all_names = ["ssim", "lpips", "latent_mse", "dino"]
        return {name: self.metrics[name].weight if name in self.metrics else 0.0 for name in all_names}
    
    def cleanup(self) -> None:
        """释放模型显存"""
        print("[FlowEditGuidance] Cleaning up...")
        del self.pipe
        for metric in self.metrics.values():
            metric.cleanup()
        self.metrics.clear()
        torch.cuda.empty_cache()
        print("[FlowEditGuidance] Cleanup done.")


# =============================================================================
# FlowEditGuidancePP - 流水线并行版本
# =============================================================================

class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
    """
    FlowEdit Guidance + 流水线并行。
    
    通过 Mixin 组合实现：
    - 继承 FlowEditGuidance 的全部功能
    - 添加 submit_async/wait_and_get 异步接口
    """
    
    def __init__(self, cfg, train_device: torch.device):
        super().__init__(cfg, train_device)
        self._init_pipeline_parallel(num_streams=2)
        print(f"[FlowEditGuidancePP] Pipeline parallelism enabled.")
