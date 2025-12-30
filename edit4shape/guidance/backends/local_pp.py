"""
同进程多 GPU Guidance（支持流水线并行）。

将 Qwen-Image-Edit 模型加载到 Guidance 设备上，
与 Trellis 训练进程共存于同一 Python 进程。

优势：
- 零序列化开销：Tensor 直接跨 GPU 传输
- 自动求导：无需手动计算梯度，PyTorch autograd 自动处理
- 简单调试：单进程，断点调试容易
- 支持异步接口实现流水线并行

设备分配策略（由 base.py compute_guidance_device 统一管理）：
- 前 N 张 GPU 给训练（Trellis DDP）
- 后 N 张 GPU 给 Guidance
- 例如 N=4: train=cuda:0-3, guidance=cuda:4-7
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim
import lpips

from edit4shape.systems.base import compute_guidance_device
from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult
from edit4shape.guidance.flowedit import QwenImageEditPlusPipeline


@dataclass
class PreprocessedImages:
    """预处理后的图像数据。"""
    pred: torch.Tensor       # (B*V,C,H,W) 渲染图（在 guidance 设备上）
    target: torch.Tensor     # (B*V,C,H,W) 编辑后图像（在 guidance 设备上，detached）
    edited_imgs: torch.Tensor  # (B,V,C,H,W) 用于返回的编辑图像（在原设备上）


class LocalGuidance:
    """
    同进程多 GPU Guidance（支持流水线并行）。
    
    特点：
    - Qwen-Image-Edit 自动加载到 train_device + 1
    - 直接计算 loss（SSIM/LPIPS/Latent MSE）
    - PyTorch autograd 自动处理梯度
    - 支持异步接口实现流水线并行
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
        
        # ---- 5. 流水线并行支持 ----
        # 两个 CUDA stream 用于双缓冲
        self._guidance_streams = [
            torch.cuda.Stream(device=self.device),
            torch.cuda.Stream(device=self.device),
        ]
        # 使用 deque 实现 FIFO 队列，支持真正的流水线重叠
        self._pending_queue: deque = deque(maxlen=2)  # 双缓冲，最多 2 个 pending 任务
        self._slot_counter = 0
    
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
        # 处理可能存在的 Alpha 通道（变为白底 RGB，与 TRELLIS 预处理一致）
        tgt_pil = composite_alpha_to_white(tgt_pil)

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
            torch.Tensor: latent 张量 (B,C,H',W')
        """
        # VAE 期望 [-1, 1] 范围
        imgs_normalized = imgs * 2 - 1  # (B,C,H,W), [0,1] → [-1,1]
        # Qwen VAE 期望 5D 输入: (B,C,num_frame,H,W)，且需要 bfloat16
        imgs_5d = imgs_normalized.unsqueeze(2).to(dtype=torch.bfloat16)  # (B,C,1,H,W)
        latent_5d = self.pipe.vae.encode(imgs_5d).latent_dist.sample()  # (B,C',1,H',W')
        latent = latent_5d.squeeze(2).to(dtype=imgs.dtype)  # (B,C',H',W'), 转回原始dtype
        return latent
    
    # =========================================================================
    # 内部计算（供同步和异步接口共用）
    # =========================================================================
    
    def _compute_guidance_internal(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
    ) -> GuidanceResult:
        """
        内部 guidance 计算（在指定 stream 上执行）。
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
        
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
    # 同步接口（兼容现有代码）
    # =========================================================================
    
    def compute_guidance(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
        rank: int = 0,  # 兼容接口，本地版本忽略
    ) -> GuidanceResult:
        """
        计算 FlowEdit Guidance（同步接口）。
        
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
        return self._compute_guidance_internal(comp_rgb, condition_images)
    
    # =========================================================================
    # 异步接口（流水线并行）
    # =========================================================================
    
    def submit_async(
        self,
        comp_rgb: torch.Tensor,            # (B,V,H,W,C)
        condition_images: List[Image.Image],
    ) -> None:
        """
        异步提交 guidance 计算（不阻塞）。
        
        当前 micro-batch 提交后，调用方可立即开始下一个 micro-batch 的 Trellis。
        使用 wait_and_get() 获取结果。
        
        流水线时序：
        - micro-batch N: Trellis 完成 → submit_async() → 开始 FlowEdit（异步）
        - micro-batch N+1: 同时开始 Trellis → submit_async() → ...
        - wait_and_get(): 等待 micro-batch N 的 FlowEdit 完成
        
        Args:
            comp_rgb: 渲染图像 (B,V,H,W,C)，float [0,1]
            condition_images: 条件图像列表 [len=B] of PIL.Image
        """
        # 选择当前 stream（双缓冲交替）
        slot_idx = self._slot_counter % 2
        self._slot_counter += 1
        stream = self._guidance_streams[slot_idx]
        
        # 记录当前默认 stream 上的事件（等待 Trellis 完成）
        trellis_done = torch.cuda.Event()
        trellis_done.record(torch.cuda.current_stream(self.train_device))
        
        # 创建完成事件
        guidance_done = torch.cuda.Event()
        
        # 在 guidance stream 上异步执行
        with torch.cuda.stream(stream):
            # 等待 Trellis 完成
            stream.wait_event(trellis_done)
            
            # 执行 guidance 计算
            result = self._compute_guidance_internal(comp_rgb, condition_images)
            
            # 记录完成事件
            guidance_done.record(stream)
        
        # 添加到队列（FIFO）
        self._pending_queue.append({
            "result": result,
            "done_event": guidance_done,
            "stream": stream,
        })
    
    def wait_and_get(self) -> GuidanceResult:
        """
        等待并获取最早提交的 submit_async 结果（FIFO）。
        
        阻塞直到 FlowEdit 计算完成，并确保训练 stream 同步。
        
        Returns:
            GuidanceResult: 最早提交的异步计算结果
        
        Raises:
            RuntimeError: 如果没有 pending 的异步提交
        """
        if not self._pending_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")
        
        # 从队列头部取出（FIFO）
        slot = self._pending_queue.popleft()
        
        # 等待 guidance stream 完成
        slot["done_event"].synchronize()
        
        # 确保训练 stream 也等待 guidance 完成，避免 backward 时的竞态条件
        torch.cuda.current_stream(self.train_device).wait_event(slot["done_event"])
        
        return slot["result"]
    
    def has_pending(self) -> bool:
        """检查是否有 pending 的异步提交。"""
        return len(self._pending_queue) > 0
    
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
