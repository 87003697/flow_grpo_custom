"""
Pipeline 适配器模块。

为 FlowEdit Pipeline 提供统一接口。

支持的 Pipeline 类型：
- "simple": FlowEditSimplePipeline（Source branch 使用解析式，速度快）
- "full": FlowEditFullPipeline（双分支都使用模型推理，效果更好）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type, Optional
from PIL import Image
import torch

from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditSimplePipeline, FlowEditFullPipeline
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import StateTracker


@dataclass
class EditResult:
    """
    FlowEdit 编辑结果。
    
    Latent 格式说明:
        - packed:   [B, seq_len, C_lat]  其中 seq_len = H_lat * W_lat
        - unpacked: [B, C_lat, T, H_lat, W_lat]  标准 VAE latent 格式
    """
    image: Image.Image                  # 编辑后的 PIL 图像
    latent: torch.Tensor                # [B, seq_len, C_lat] packed latent（最终编辑结果）
    tracker: StateTracker       # 中间状态跟踪器（latents 都是 packed 格式）


class BasePipelineAdapter(ABC):
    """
    Pipeline 适配器基类。
    
    定义统一的加载和编辑接口，屏蔽不同 pipeline 的参数差异。
    """
    
    pipe: Any = None  # Pipeline 实例
    
    @abstractmethod
    def load(self, model_path: str, device: torch.device) -> None:
        """
        加载模型到指定设备。
        
        Args:
            model_path: 模型路径（HuggingFace ID 或本地路径）
            device: 目标设备
        """
        pass
    
    @abstractmethod
    def edit(
        self,
        rendered: Image.Image,
        condition: Image.Image,
        cfg: Any,
        src_latent: Optional[torch.Tensor] = None,
    ) -> EditResult:
        """
        执行图像编辑。
        
        Args:
            rendered: 渲染图（Trellis 输出）
            condition: 条件图像（用户输入）
            cfg: flowedit 运行时配置（cfg.train.guidance / cfg.{stage}.guidance）
            src_latent: 预编码的 src latent [B, seq_len, C]，用于可导编码。
                        如果提供，将替换 pipeline 内部编码的 x_src。
        
        Returns:
            EditResult: 包含编辑后图像和 latent
        """
        pass


class SimplePipelineAdapter(BasePipelineAdapter):
    """
    FlowEditSimplePipeline 适配器。
    
    特点：
    - Source branch 使用解析式（不需要模型推理）
    - 同时记录 z_edit 和 x0_pos/x0_neg
    - 通过 csd_weight 和 mse_weight 控制 loss 类型
    - 速度快，适合训练
    """
    
    def load(self, model_path: str, device: torch.device) -> None:
        self.pipe = FlowEditSimplePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
    
    def edit(
        self, 
        rendered: Image.Image, 
        condition: Image.Image, 
        cfg: Any,
        src_latent: Optional[torch.Tensor] = None,
    ) -> EditResult:
        device = torch.device(self.pipe._execution_device)
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        output = self.pipe(
            image=[rendered, condition],
            target_prompt=cfg.target_prompt,
            generator=generator,
            negative_prompt_tgt=cfg.negative_prompt_tgt,
            num_inference_steps=cfg.steps,
            true_cfg_scale_tgt=cfg.true_cfg_scale_tgt,
            n_max=cfg.n_max,
            noise_mode=cfg.noise_mode,
            use_mts_sampling=cfg.use_mts_sampling,
            src_latent=src_latent,
        )
        return EditResult(
            image=output.images[0],
            latent=output.latents,
            tracker=output.tracker,
        )


class FullPipelineAdapter(BasePipelineAdapter):
    """
    FlowEditFullPipeline 适配器（双分支模型推理）。
    
    特点：
    - Source 和 Target 两个分支都使用完整模型推理
    - 需要额外的 source_prompt 参数
    - 效果更好，但速度较慢
    """
    
    def load(self, model_path: str, device: torch.device) -> None:
        self.pipe = FlowEditFullPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
    
    def edit(
        self, 
        rendered: Image.Image, 
        condition: Image.Image, 
        cfg: Any,
        src_latent: Optional[torch.Tensor] = None,
    ) -> EditResult:
        device = torch.device(self.pipe._execution_device)
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        output = self.pipe(
            image=[rendered, condition],
            target_prompt=cfg.target_prompt,
            source_prompt=cfg.source_prompt,
            generator=generator,
            negative_prompt_src=cfg.negative_prompt_src,
            negative_prompt_tgt=cfg.negative_prompt_tgt,
            num_inference_steps=cfg.steps,
            true_cfg_scale_src=cfg.true_cfg_scale_src,
            true_cfg_scale_tgt=cfg.true_cfg_scale_tgt,
            n_max=cfg.n_max,
            noise_mode=cfg.noise_mode,
            use_mts_sampling=cfg.use_mts_sampling,
            src_latent=src_latent,
        )
        return EditResult(
            image=output.images[0],
            latent=output.latents,
            tracker=output.tracker,
        )


# =====================================================================
# 工厂注册表
# =====================================================================

PIPELINE_ADAPTERS: Dict[str, Type[BasePipelineAdapter]] = {
    "simple": SimplePipelineAdapter,
    "full": FullPipelineAdapter,
}


def create_pipeline_adapter(pipeline_type: str = "simple") -> BasePipelineAdapter:
    """
    根据类型创建 pipeline 适配器。
    
    Args:
        pipeline_type: Pipeline 类型，可选 "simple" | "full"
    
    Returns:
        BasePipelineAdapter: 对应的适配器实例
    
    Raises:
        ValueError: 未知的 pipeline_type
    """
    if pipeline_type not in PIPELINE_ADAPTERS:
        available = list(PIPELINE_ADAPTERS.keys())
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}. Choose from {available}")
    return PIPELINE_ADAPTERS[pipeline_type]()
