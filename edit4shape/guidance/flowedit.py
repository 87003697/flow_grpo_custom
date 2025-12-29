"""
FlowEdit Guidance 模块。

使用服务端计算的 SSIM/LPIPS 梯度，通过 SpecifyGradient 绑定到渲染图。
"""

import requests
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import List, Any, Optional, Dict
from dataclasses import dataclass
from PIL import Image

from edit4shape.guidance.utils import (
    tensor_to_base64, base64_to_tensor, pil_to_base64, base64_to_grad_tensor,
)


class SpecifyGradient(Function):
    """将预计算的梯度绑定到 tensor，反向传播时使用该梯度。"""
    
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        return gt_grad * grad_scale, None


@dataclass
class GuidanceResult:
    """Guidance 结果"""
    edited_imgs: torch.Tensor                        # (B,V,C,H,W) 编辑后图像
    loss_ssim: Optional[torch.Tensor] = None         # 标量 (SpecifyGradient 伪 loss)
    loss_lpips: Optional[torch.Tensor] = None        # 标量 (SpecifyGradient 伪 loss)
    loss_latent_mse: Optional[torch.Tensor] = None   # 标量 (SpecifyGradient 伪 loss)
    avg_ssim: Optional[float] = None                 # 平均 SSIM（用于日志）
    avg_lpips: Optional[float] = None                # 平均 LPIPS（用于日志）
    avg_latent_mse: Optional[float] = None           # 平均 Latent MSE（用于日志）


class FlowEditClient:
    """FlowEdit API 客户端。"""
    
    def __init__(self, cfg: Any, loss_cfg: Any = None):
        """
        初始化客户端。
        
        Args:
            cfg: guidance 配置，包含 service 和 flowedit 子配置
            loss_cfg: 可选，loss 权重配置（cfg.train.loss），包含 ssim/lpips/latent_mse 权重
        """
        # 服务参数
        self.base_port = cfg.service.base_port
        self.timeout = cfg.service.timeout
        
        # 算法参数
        self.prompt = cfg.flowedit.prompt
        self.seed = cfg.flowedit.seed
        self.steps = cfg.flowedit.steps
        self.guidance_scale = cfg.flowedit.guidance_scale
        self.true_cfg_scale_tgt = cfg.flowedit.true_cfg_scale_tgt
        self.n_min = cfg.flowedit.n_min
        self.n_max = cfg.flowedit.n_max
        
        # Loss 权重（优先从 loss_cfg 读取，兼容旧配置）
        if loss_cfg is not None:
            self.ssim_weight = loss_cfg.ssim
            self.lpips_weight = loss_cfg.lpips
            self.latent_mse_weight = loss_cfg.latent_mse
        else:
            # 兼容旧配置路径
            self.ssim_weight = getattr(cfg.flowedit, 'ssim_weight', 0.0)
            self.lpips_weight = getattr(cfg.flowedit, 'lpips_weight', 0.0)
            self.latent_mse_weight = getattr(cfg.flowedit, 'latent_mse_weight', 0.0)
        
        # 梯度计算开关（weight > 0 时才请求梯度）
        self.compute_ssim_grad = self.ssim_weight > 0
        self.compute_lpips_grad = self.lpips_weight > 0
        self.compute_latent_mse_grad = self.latent_mse_weight > 0
    
    def get_api_url(self, rank: int) -> str:
        """根据进程 rank 获取对应的 API 地址。"""
        return f"http://localhost:{self.base_port + (rank % 4)}"
    
    def check_health(self, api_url: str) -> bool:
        """检查服务健康状态。"""
        try:
            return requests.get(f"{api_url}/health", timeout=5.0).status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _call_edit_api(
        self, 
        api_url: str, 
        source: torch.Tensor, 
        target: Image.Image, 
        size: tuple,
        max_retries: int = 3,
    ) -> Dict:
        """
        调用编辑 API，返回原始响应 dict。
        
        Args:
            api_url: API 地址
            source: 渲染图 (C,H,W)
            target: 条件图 PIL.Image
            size: 目标尺寸 (H,W)
            max_retries: 最大重试次数
        
        Returns:
            API 响应 dict
        """
        payload = {
            "source_image": tensor_to_base64(source),
            "target_image": pil_to_base64(target, size=size),
            "prompt": self.prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale_tgt": self.true_cfg_scale_tgt,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "compute_ssim_grad": self.compute_ssim_grad,
            "compute_lpips_grad": self.compute_lpips_grad,
            "compute_latent_mse_grad": self.compute_latent_mse_grad,
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(f"{api_url}/edit", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1.0 * (attempt + 1))  # 指数退避
        
        raise RuntimeError(f"FlowEdit API 调用失败 (尝试 {max_retries} 次): {last_error}")
    
    def _aggregate_images(self, responses: List[List[Dict]], device: torch.device) -> torch.Tensor:
        """聚合必选项：编辑后图像。返回 (B,V,C,H,W)"""
        imgs = []
        for batch in responses:
            batch_imgs = [base64_to_tensor(r["image"], device) for r in batch]
            imgs.append(torch.stack(batch_imgs))  # (V,C,H,W)
        return torch.stack(imgs)  # (B,V,C,H,W)
    
    def _aggregate_ssim(self, responses: List[List[Dict]], device: torch.device) -> Dict:
        """聚合 SSIM 相关：梯度和指标值。"""
        grads, vals = [], []
        for batch in responses:
            b_grads = []
            for r in batch:
                if r.get("ssim") is not None:
                    vals.append(r["ssim"])
                if r.get("ssim_grad"):
                    b_grads.append(base64_to_grad_tensor(r["ssim_grad"], device))
            if b_grads:
                grads.append(torch.stack(b_grads))  # (V,C,H,W)
        
        return {
            "grads": torch.stack(grads) if grads else None,  # (B,V,C,H,W) or None
            "avg": sum(vals) / len(vals) if vals else None,
        }
    
    def _aggregate_lpips(self, responses: List[List[Dict]], device: torch.device) -> Dict:
        """聚合 LPIPS 相关：梯度和指标值。"""
        grads, vals = [], []
        for batch in responses:
            b_grads = []
            for r in batch:
                if r.get("lpips") is not None:
                    vals.append(r["lpips"])
                if r.get("lpips_grad"):
                    b_grads.append(base64_to_grad_tensor(r["lpips_grad"], device))
            if b_grads:
                grads.append(torch.stack(b_grads))  # (V,C,H,W)
        
        return {
            "grads": torch.stack(grads) if grads else None,  # (B,V,C,H,W) or None
            "avg": sum(vals) / len(vals) if vals else None,
        }
    
    def _aggregate_latent_mse(self, responses: List[List[Dict]], device: torch.device) -> Dict:
        """聚合 Latent MSE 相关：梯度和指标值。"""
        grads, vals = [], []
        for batch in responses:
            b_grads = []
            for r in batch:
                if r.get("latent_mse") is not None:
                    vals.append(r["latent_mse"])
                if r.get("latent_mse_grad"):
                    b_grads.append(base64_to_grad_tensor(r["latent_mse_grad"], device))
            if b_grads:
                grads.append(torch.stack(b_grads))  # (V,C,H,W)
        
        return {
            "grads": torch.stack(grads) if grads else None,  # (B,V,C,H,W) or None
            "avg": sum(vals) / len(vals) if vals else None,
        }
    
    def compute_guidance(
        self, 
        comp_rgb: torch.Tensor,          # (B,V,H,W,C)
        condition_images: List[Image.Image],
        rank: int,
    ) -> GuidanceResult:
        """
        计算 FlowEdit Guidance。
        
        流程：
        1. 遍历 B×V 调用 edit API
        2. 聚合响应
        3. 使用 SpecifyGradient 绑定梯度到渲染图
        
        Args:
            comp_rgb: 渲染图 (B,V,H,W,C)
            condition_images: 条件图像列表 [len=B] of PIL.Image
            rank: 进程 rank（用于选择 API 端口）
        
        Returns:
            GuidanceResult
        """
        api_url = self.get_api_url(rank)
        if not self.check_health(api_url):
            raise RuntimeError(f"FlowEdit 服务不可用: {api_url}")
        
        B, V, H, W, C = comp_rgb.shape
        
        # 1. 遍历调用 API
        responses = []
        for b in range(B):
            batch = []
            for v in range(V):
                src = comp_rgb[b, v].permute(2, 0, 1)  # (C,H,W)
                batch.append(self._call_edit_api(api_url, src, condition_images[b], (H, W)))
            responses.append(batch)
        
        device = comp_rgb.device
        pred_imgs = comp_rgb.permute(0, 1, 4, 2, 3)  # (B,V,C,H,W)
        
        # 2. 聚合必选项
        edited_imgs = self._aggregate_images(responses, device)  # (B,V,C,H,W)
        
        # 3. 聚合可选项 & 绑定梯度（应用权重）
        ssim = self._aggregate_ssim(responses, device)
        lpips = self._aggregate_lpips(responses, device)
        latent_mse = self._aggregate_latent_mse(responses, device)
        
        loss_ssim = None
        if ssim["grads"] is not None and self.ssim_weight > 0:
            loss_ssim = SpecifyGradient.apply(pred_imgs, ssim["grads"] * self.ssim_weight)
        
        loss_lpips = None
        if lpips["grads"] is not None and self.lpips_weight > 0:
            loss_lpips = SpecifyGradient.apply(pred_imgs, lpips["grads"] * self.lpips_weight)

        # Latent MSE: 需要 normalize（服务端用的是 [-1,1] 空间）
        loss_latent_mse = None
        if latent_mse["grads"] is not None and self.latent_mse_weight > 0:
            pred_imgs_normalized = pred_imgs * 2 - 1  # [0,1] → [-1,1]，可导反传
            loss_latent_mse = SpecifyGradient.apply(pred_imgs_normalized, latent_mse["grads"] * self.latent_mse_weight * 2)  # 2 还原 scale
        
        return GuidanceResult(
            edited_imgs=edited_imgs,
            loss_ssim=loss_ssim,
            loss_lpips=loss_lpips,
            loss_latent_mse=loss_latent_mse,
            avg_ssim=ssim["avg"],
            avg_lpips=lpips["avg"],
            avg_latent_mse=latent_mse["avg"],
        )
