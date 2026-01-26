"""
Qwen-Image Pipeline 工具模块。

包含:
- BaseStateTracker: 状态追踪器抽象基类
- FlowEditStateTracker: FlowEdit 中间状态跟踪器
- CSDStateTracker: CSD 中间状态跟踪器
- SDSStateTracker: SDS 中间状态跟踪器
- DifferentiableVAEMixin: 可微分 VAE 编码 Mixin
"""

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import List, Any, Optional, Protocol
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents


# =============================================================================
# BaseStateTracker (抽象基类)
# =============================================================================

@dataclass
class BaseStateTracker(ABC):
    """
    状态追踪器抽象基类。
    
    所有 Guidance 方法的 Tracker（FlowEdit、CSD、SDS）都继承此类。
    
    共有属性：
        - height, width: 图像尺寸（用于 decode）
    
    子类必须实现：
        - target: 目标 latent（用于 loss 计算）
        - loss(): 主要 loss 计算方法
    """
    
    height: int = 0
    width: int = 0
    
    # =========================================================================
    # 抽象属性和方法
    # =========================================================================
    
    @property
    @abstractmethod
    def target(self) -> torch.Tensor:
        """
        目标 latent [B, seq, C]。
        
        用于 loss 计算，子类根据算法返回不同的目标：
        - FlowEdit: 最终编辑后的 latent
        - CSD: x0_pred_high（高 CFG 预测）
        - SDS: x0_pred（模型预测）
        """
        pass
    
    @abstractmethod
    def loss(
        self, 
        src_latent: torch.Tensor, 
        weight_type: str = "uniform",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        计算 loss（真 loss，可直接 backward）。
        
        Args:
            src_latent: [B, seq, C] 有梯度
            weight_type: 加权类型（子类各自定义）
            eps: 数值稳定 epsilon
        
        Returns:
            标量 loss
        """
        pass
    
    # =========================================================================
    # 共有方法
    # =========================================================================
    
    def _decode_latent(self, pipe: Any, latent: torch.Tensor) -> Optional[Image.Image]:
        """
        Decode 单个 latent 为图像。
        
        Args:
            pipe: Pipeline（需要 _decode_latent_to_image 方法）
            latent: [1, seq, C]
        
        Returns:
            PIL Image
        """
        with torch.no_grad():
            images = pipe._decode_latent_to_image(
                latent,
                self.height,
                self.width,
                "pil"
            )
            return images[0] if images else None
    
    def _concat_images_horizontal(self, *images: Image.Image) -> Optional[Image.Image]:
        """
        水平拼接多张图像。
        
        Args:
            *images: PIL Image 列表
        
        Returns:
            拼接后的 PIL Image
        """
        images = [img for img in images if img is not None]
        if not images:
            return None
        
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return combined


# =============================================================================
# StepVisualizationMixin
# =============================================================================

class StepVisualizationMixin(ABC):
    """
    步骤可视化 Mixin。
    
    为多步 Tracker 提供中间步可视化能力。
    子类需要实现 `step_latents` 属性，返回要可视化的 latent 列表。
    
    要求子类定义以下字段：
        - images: List[Image.Image] 用于存储 decode 后的图像
        - height, width: int 图像尺寸
    """
    
    # 子类需要定义这些字段
    images: List[Image.Image]
    height: int
    width: int
    
    @property
    @abstractmethod
    def step_latents(self) -> List[torch.Tensor]:
        """
        返回要可视化的 latent 列表。
        
        子类根据自身数据结构返回：
        - FlowEditStateTracker: self.latents
        - ContrastStateTracker: self.z_edits
        
        Returns:
            List of [B, seq, C] tensors
        """
        pass
    
    @property
    def has_images(self) -> bool:
        """是否已 decode 图像"""
        return len(self.images) > 0
    
    def decode_uniform_samples(self, pipe: Any, n_samples: int = 4) -> None:
        """
        均匀采样 n_samples 个中间步进行并行 decode。
        
        Args:
            pipe: Pipeline（需要 _decode_latent_to_image 方法）
            n_samples: 采样步数，必须是完全平方数（4, 9, 16, ...）
        
        结果存储到 self.images。
        """
        K = len(self.step_latents)
        if K == 0:
            return
        
        # 计算均匀采样的索引
        if n_samples >= K:
            indices = list(range(K))
            n_samples = K
        else:
            indices = [int(i * (K - 1) / (n_samples - 1)) for i in range(n_samples)]
        
        # 收集要 decode 的 latents
        sampled_latents = [self.step_latents[i] for i in indices]
        
        # 固定为单张解码，避免 OOM
        max_batch = 1
        
        # 分批 decode
        decoded_images = []
        with torch.no_grad():
            for start in range(0, n_samples, max_batch):
                chunk_latents = torch.cat(  # [batch, seq_len, C]
                    sampled_latents[start : start + max_batch], 
                    dim=0
                )
                chunk_images = pipe._decode_latent_to_image(
                    chunk_latents, 
                    self.height, 
                    self.width, 
                    "pil"
                )
                decoded_images.extend(chunk_images)
        
        self.images = decoded_images
    
    def get_progress_grid(self, pipe: Any, n_samples: int = 4) -> Image.Image:
        """
        将 n_samples 张中间步图像合成一张 √n × √n 网格图。
        
        Args:
            pipe: Pipeline
            n_samples: 采样步数，必须是完全平方数（4, 9, 16, ...）
        
        Returns:
            合成的网格 PIL Image
        """
        # 检查是否为完全平方数
        grid_size = int(n_samples ** 0.5)
        assert grid_size * grid_size == n_samples, f"n_samples must be a perfect square, got {n_samples}"
        
        # Decode 中间步（如果还没 decode 或数量不对）
        if not self.has_images or len(self.images) != n_samples:
            self.decode_uniform_samples(pipe, n_samples)
        
        # 合成网格
        img_w, img_h = self.images[0].size
        grid_img = Image.new("RGB", (img_w * grid_size, img_h * grid_size), (255, 255, 255))
        
        for i, img in enumerate(self.images):
            row = i // grid_size
            col = i % grid_size
            grid_img.paste(img, (col * img_w, row * img_h))
        
        return grid_img


# =============================================================================
# SDSStateTracker
# =============================================================================

@dataclass
class SDSStateTracker(BaseStateTracker):
    """
    SDS 状态追踪器。
    
    SDS Loss = MSE(src_latent, x0_pred)
    让 src_latent 向模型预测的 x0_pred 靠拢。
    
    Attributes:
        x0_pred: 模型预测的 x0 [B, seq, C]（吸引目标）
        t: 采样的时间步 [B]
        t_normalized: 归一化时间步 [B, 1, 1]
        noise: 使用的噪声 [B, seq, C]
        image: decode 后的预测图像（按需填充）
    """
    
    x0_pred: torch.Tensor = None        # [B, seq, C] 模型预测的 x0
    t: torch.Tensor = None              # [B] 采样的时间步
    t_normalized: torch.Tensor = None   # [B, 1, 1] 归一化时间步
    noise: torch.Tensor = None          # [B, seq, C] 使用的噪声
    image: Image.Image = None           # decode 后的预测图像
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = x0_pred"""
        return self.x0_pred
    
    def loss(
        self, 
        src_latent: torch.Tensor, 
        weight_type: str = "uniform",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        SDS Loss：MSE(src_latent, x0_pred)
        
        Args:
            src_latent: [B, seq, C] 有梯度
            weight_type: "uniform" | "ada" | "t"
            eps: ada 模式的 epsilon
        
        Returns:
            标量 loss
        """
        x0_pred = self.x0_pred.detach()  # [B, seq, C]
        
        if weight_type == "uniform":
            # 标准 MSE
            return F.mse_loss(src_latent.float(), x0_pred.float())
        
        elif weight_type == "t":
            # 时间步加权 MSE
            weight = self.t_normalized.squeeze(-1).squeeze(-1)  # [B]
            diff = (src_latent - x0_pred).float()  # [B, seq, C]
            weighted_diff = diff * weight.view(-1, 1, 1)  # [B, seq, C]
            return (weighted_diff ** 2).mean()
        
        elif weight_type == "ada":
            # 自适应归一化：先计算梯度，再归一化
            # SDS 梯度 = src_latent - x0_pred
            grad_raw = (src_latent - x0_pred).detach().float()  # [B, seq, C]
            
            # 归一化因子
            normalizer = torch.abs(grad_raw).mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
            grad_normalized = grad_raw / normalizer  # [B, seq, C]
            
            # 构造 loss，使 ∂loss/∂src = grad_normalized
            return (src_latent.float() * grad_normalized).mean()
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # =========================================================================
    # 可视化
    # =========================================================================
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode x0_pred 为图像"""
        self.image = self._decode_latent(pipe, self.x0_pred)
    
    def get_comparison_image(self, pipe: Any, rendered_latent: torch.Tensor) -> Image.Image:
        """
        生成对比图：rendered | x0_pred
        
        Args:
            pipe: Pipeline
            rendered_latent: 渲染图的 latent [B, seq, C]
        
        Returns:
            并排对比图
        """
        rendered_img = self._decode_latent(pipe, rendered_latent)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(rendered_img, self.image)


# =============================================================================
# CSDStateTracker
# =============================================================================

@dataclass
class CSDStateTracker(BaseStateTracker):
    """
    CSD 状态追踪器。
    
    CSD Loss = MSE(src, x0_high) - MSE(src, x0_low)
    让 src 向 x0_high 靠拢，同时远离 x0_low。
    
    Attributes:
        x0_pred_high: 高 CFG 预测的 x0 [B, seq, C]（吸引目标）
        x0_pred_low: 低 CFG 预测的 x0 [B, seq, C]（排斥目标）
        t: 采样的时间步 [B]
        t_normalized: 归一化时间步 [B, 1, 1]
        noise: 使用的噪声 [B, seq, C]
        image: decode 后的预测图像（按需填充）
    """
    
    x0_pred_high: torch.Tensor = None   # [B, seq, C] 高 CFG 预测（吸引目标）
    x0_pred_low: torch.Tensor = None    # [B, seq, C] 低 CFG 预测（排斥目标）
    t: torch.Tensor = None              # [B] 采样的时间步
    t_normalized: torch.Tensor = None   # [B, 1, 1] 归一化时间步
    noise: torch.Tensor = None          # [B, seq, C] 使用的噪声
    image: Image.Image = None           # decode 后的预测图像
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = x0_pred_high（吸引目标）"""
        return self.x0_pred_high
    
    def loss(
        self, 
        src_latent: torch.Tensor, 
        weight_type: str = "uniform",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        CSD Loss：MSE(src, x0_high) - MSE(src, x0_low)
        
        Args:
            src_latent: [B, seq, C] 有梯度
            weight_type: "uniform" | "ada"
            eps: ada 模式的 epsilon
        
        Returns:
            标量 loss
        """
        x0_high = self.x0_pred_high.detach()  # [B, seq, C]
        x0_low = self.x0_pred_low.detach()    # [B, seq, C]
        
        if weight_type == "uniform":
            # 标准双 MSE
            loss_pos = F.mse_loss(src_latent.float(), x0_high.float())
            loss_neg = F.mse_loss(src_latent.float(), x0_low.float())
            return loss_pos - loss_neg
        
        elif weight_type == "ada":
            # 自适应归一化：先计算 CSD 梯度，再归一化
            # CSD 梯度 = x0_low - x0_high
            grad_raw = (x0_low - x0_high).detach().float()  # [B, seq, C]
            
            # 归一化因子
            normalizer = torch.abs(grad_raw).mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
            grad_normalized = grad_raw / normalizer  # [B, seq, C]
            
            # 构造 loss，使 ∂loss/∂src = grad_normalized
            return (src_latent.float() * grad_normalized).mean()
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # =========================================================================
    # 可视化
    # =========================================================================
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode x0_pred_high 为图像"""
        self.image = self._decode_latent(pipe, self.x0_pred_high)
    
    def get_comparison_image(self, pipe: Any, rendered_latent: torch.Tensor) -> Image.Image:
        """
        生成对比图：rendered | x0_pred_high
        
        Args:
            pipe: Pipeline
            rendered_latent: 渲染图的 latent [B, seq, C]
        
        Returns:
            并排对比图
        """
        rendered_img = self._decode_latent(pipe, rendered_latent)
        if self.image is None:
            self.decode_prediction(pipe)
        return self._concat_images_horizontal(rendered_img, self.image)


# =============================================================================
# FlowEditStateTracker
# =============================================================================

@dataclass
class FlowEditStateTracker(BaseStateTracker, StepVisualizationMixin):
    """
    FlowEdit 多步编辑状态追踪器。
    
    记录每步的 packed latent 和 t 值，支持多步监督和可视化。
    
    Latent 格式说明:
        - packed:   [B, seq_len, C]  其中 seq_len = H_lat * W_lat, C = latent_channels * 4
        - unpacked: [B, C, T, H_lat, W_lat]  标准 VAE latent 格式
    
    Attributes:
        latents: 每步的 packed latent，每个元素 shape: [B, seq_len, C]
        t_values: 每步的时间步 t ∈ (0, 1]
        images: decode 后的中间步图像（按需填充）
    """
    
    latents: List[torch.Tensor] = field(default_factory=list)  # List of [B, seq, C]
    t_values: List[float] = field(default_factory=list)
    images: List[Image.Image] = field(default_factory=list)
    
    def record(self, z_edit: torch.Tensor, t: float) -> None:
        """
        记录一个中间状态。
        
        Args:
            z_edit: 当前 packed latent [B, seq, C]
            t: 当前时间步
        """
        self.latents.append(z_edit.detach().clone())
        self.t_values.append(t)
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最终编辑后的 latent"""
        return self.latents[-1] if self.latents else None
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名，向后兼容"""
        return self.target
    
    def stack(self) -> torch.Tensor:
        """堆叠所有 latent [K, B, seq, C]"""
        return torch.stack(self.latents, dim=0)
    
    def __len__(self) -> int:
        return len(self.latents)
    
    @property
    def step_latents(self) -> List[torch.Tensor]:
        """实现 StepVisualizationMixin 要求的属性"""
        return self.latents
    
    def loss(
        self, 
        src_latent: torch.Tensor, 
        weight_type: str = "final",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        FlowEdit Loss。
        
        Args:
            src_latent: [B, seq, C] 有梯度
            weight_type: "final" | "mean" | "weighted" | "ada"
            eps: ada 模式的 epsilon
        
        Returns:
            标量 loss
        """
        if weight_type == "final":
            # 只用最终 latent
            return F.mse_loss(src_latent.float(), self.final.detach().float())
        
        elif weight_type == "mean":
            # 所有中间步均匀加权
            losses = []
            for lat in self.latents:
                losses.append(F.mse_loss(src_latent.float(), lat.detach().float()))
            return torch.stack(losses).mean()
        
        elif weight_type == "weighted":
            # 1/k 加权（k=1,2,...,K）
            losses = []
            for lat in self.latents:
                losses.append(F.mse_loss(src_latent.float(), lat.detach().float()))
            losses = torch.stack(losses)  # [K]
            K = len(losses)
            w = 1.0 / torch.arange(1, K + 1, device=losses.device, dtype=losses.dtype)
            w = w / w.sum()
            return (losses * w).sum()
        
        elif weight_type == "ada":
            # 自适应归一化（对最终 latent）
            grad_raw = (src_latent - self.final).detach().float()  # [B, seq, C]
            normalizer = torch.abs(grad_raw).mean(dim=(1, 2), keepdim=True) + eps
            grad_normalized = grad_raw / normalizer
            return (src_latent.float() * grad_normalized).mean()
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
    
    # =========================================================================
    # 向后兼容的 loss 方法（调用统一的 loss 方法）
    # =========================================================================
    
    def loss_final(self, rendered: torch.Tensor) -> torch.Tensor:
        """向后兼容：只用最终 latent 计算 loss"""
        return self.loss(rendered, weight_type="final")
    
    def loss_mean(self, rendered: torch.Tensor) -> torch.Tensor:
        """向后兼容：所有中间步均匀加权的 loss"""
        return self.loss(rendered, weight_type="mean")
    
    def loss_weighted(self, rendered: torch.Tensor) -> torch.Tensor:
        """向后兼容：用编辑次数的倒数加权的 loss"""
        return self.loss(rendered, weight_type="weighted")
    
    def loss_ada(self, rendered: torch.Tensor) -> torch.Tensor:
        """向后兼容：自适应归一化的 loss"""
        return self.loss(rendered, weight_type="ada")
    
    # 可视化方法由 StepVisualizationMixin 提供


# =============================================================================
# ContrastStateTracker (Multi-Step CSD)
# =============================================================================

@dataclass
class ContrastStateTracker(BaseStateTracker, StepVisualizationMixin):
    """
    Multi-Step CSD (Contrast) 状态追踪器。
    
    在每个去噪步记录高低 CFG 的 x0 预测，用于 Multi-Step CSD loss 计算。
    
    CSD Loss = MSE(src, x0_high) - MSE(src, x0_low)
    让 src 向高 CFG 预测靠拢，同时远离低 CFG 预测。
    
    Attributes:
        x0_highs: 每步的高 CFG x0 预测 [B, seq, C]
        x0_lows: 每步的低 CFG x0 预测 [B, seq, C]
        t_values: 每步的时间步
        z_edits: 每步的编辑 latent（用于可视化）
    """
    
    x0_highs: List[torch.Tensor] = field(default_factory=list)
    x0_lows: List[torch.Tensor] = field(default_factory=list)
    t_values: List[float] = field(default_factory=list)
    z_edits: List[torch.Tensor] = field(default_factory=list)
    images: List[Image.Image] = field(default_factory=list)  # decode 后的中间步图像
    
    def record(
        self, 
        x0_high: torch.Tensor,
        x0_low: torch.Tensor,
        t: float,
        z_edit: Optional[torch.Tensor] = None,
    ) -> None:
        """记录一个中间状态"""
        self.x0_highs.append(x0_high.detach().clone())
        self.x0_lows.append(x0_low.detach().clone())
        self.t_values.append(t)
        if z_edit is not None:
            self.z_edits.append(z_edit.detach().clone())
    
    @property
    def target(self) -> torch.Tensor:
        """目标 latent = 最后一步的 x0_high"""
        return self.x0_highs[-1] if self.x0_highs else None
    
    @property
    def final(self) -> torch.Tensor:
        """target 的别名，向后兼容"""
        return self.target
    
    def __len__(self) -> int:
        return len(self.x0_highs)
    
    @property
    def step_latents(self) -> List[torch.Tensor]:
        """实现 StepVisualizationMixin 要求的属性"""
        return self.z_edits
    
    def loss(
        self, 
        src_latent: torch.Tensor, 
        weight_type: str = "inv_weighted",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        计算 Multi-Step CSD Loss。
        
        Args:
            src_latent: [B, seq, C] 有梯度
            weight_type: 
                - "weighted": 1/k 加权（前期大，后期小）
                - "inv_weighted": k/K 加权（前期小，后期大）
                - "ada": 每步归一化后累加，分母为 |src - x0_high|
            eps: ada 模式的 epsilon
        
        Returns:
            标量 loss
        """
        K = len(self.x0_highs)
        if K == 0:
            return torch.tensor(0.0, device=src_latent.device, requires_grad=True)
        
        if weight_type == "weighted":
            # 1/k 加权（前期大，后期小）
            losses = []
            for x0_h, x0_l in zip(self.x0_highs, self.x0_lows):
                loss_pos = F.mse_loss(src_latent.float(), x0_h.detach().float())
                loss_neg = F.mse_loss(src_latent.float(), x0_l.detach().float())
                losses.append(loss_pos - loss_neg)
            
            losses = torch.stack(losses)  # [K]
            w = 1.0 / torch.arange(1, K + 1, device=losses.device, dtype=losses.dtype)  # [1, 1/2, ..., 1/K]
            w = w / w.sum()
            return (losses * w).sum()
        
        elif weight_type == "inv_weighted":
            # k/K 加权（前期小，后期大）
            losses = []
            for x0_h, x0_l in zip(self.x0_highs, self.x0_lows):
                loss_pos = F.mse_loss(src_latent.float(), x0_h.detach().float())
                loss_neg = F.mse_loss(src_latent.float(), x0_l.detach().float())
                losses.append(loss_pos - loss_neg)
            
            losses = torch.stack(losses)  # [K]
            w = torch.arange(1, K + 1, device=losses.device, dtype=losses.dtype) / K  # [1/K, 2/K, ..., 1]
            w = w / w.sum()
            return (losses * w).sum()
        
        elif weight_type == "ada":
            # Multi-Step Ada: 每步归一化后累加
            # 归一化分母使用 |src - x0_high|（吸引力的幅度）
            K = len(self.x0_highs)
            grad_sum = torch.zeros_like(src_latent.float())  # [B, seq, C]
            
            for x0_h, x0_l in zip(self.x0_highs, self.x0_lows):
                x0_h = x0_h.detach().float()  # [B, seq, C]
                x0_l = x0_l.detach().float()  # [B, seq, C]
                
                # 归一化分母：|src - x0_high|（当前 src 到目标的距离）
                diff_high = src_latent.float() - x0_h  # [B, seq, C]
                normalizer_k = torch.abs(diff_high).mean(dim=(1, 2), keepdim=True) + eps  # [B, 1, 1]
                
                # CSD 梯度方向 = x0_low - x0_high
                grad_k = x0_l - x0_h  # [B, seq, C]
                grad_normalized_k = grad_k / normalizer_k  # [B, seq, C]
                
                # 累加
                grad_sum = grad_sum + grad_normalized_k
            
            # 平均
            grad_avg = grad_sum / K  # [B, seq, C]
            
            # 构造 loss，使 ∂loss/∂src = grad_avg
            return (src_latent.float() * grad_avg).mean()
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}. Choose from: weighted, inv_weighted, ada")
    
    # 可视化方法 (has_images, decode_uniform_samples, get_progress_grid) 由 StepVisualizationMixin 提供
    
    def decode_prediction(self, pipe: Any) -> None:
        """Decode 最终的 x0_high 为图像"""
        if self.x0_highs:
            self.image = self._decode_latent(pipe, self.x0_highs[-1])
    
    def get_comparison_image(self, pipe: Any, rendered_latent: torch.Tensor) -> Optional[Image.Image]:
        """生成对比图：rendered | x0_high_final"""
        rendered_img = self._decode_latent(pipe, rendered_latent)
        if not hasattr(self, 'image') or self.image is None:
            if self.x0_highs:
                self.decode_prediction(pipe)
        return self._concat_images_horizontal(rendered_img, getattr(self, 'image', None))


# =============================================================================
# DifferentiableVAEMixin
# =============================================================================

class DifferentiableVAEMixin:
    """
    为 Pipeline 提供可微分 VAE encode 能力的 Mixin。
    
    要求 Pipeline 具有以下属性:
        - self.vae: VAE 模型
        - self.latent_channels: latent 通道数
    """
    
    def _encode_vae_image_differentiable(self, image: torch.Tensor) -> torch.Tensor:
        """
        可微分版本的 VAE encode（不带 @torch.no_grad）。
        
        与 _encode_vae_image 相同的逻辑，但保留梯度用于反向传播。
        
        Args:
            image: [B, C, 1, H, W] 图像，[-1, 1] 范围，bfloat16
        
        Returns:
            normalized latent [B, C_lat, 1, H_lat, W_lat]
        """
        # VAE encode，保留梯度
        image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
        
        # 标准化
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )  # [1, C_lat, 1, 1, 1]
        return (image_latents - latents_mean) / latents_std  # [B, C_lat, 1, H_lat, W_lat]
