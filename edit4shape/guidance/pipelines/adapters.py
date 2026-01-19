"""
Pipeline 适配器模块。

为不同的 FlowEdit Pipeline 提供统一接口，消除 FlowEditGuidance 中的条件分支。

支持的 Pipeline 类型：
- "simple": FlowEditSimplePipeline，source branch 使用解析式（速度快）
- "full": FlowEditPipeline，双分支都使用模型推理（效果更好）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type
from PIL import Image
import torch

from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditSimplePipeline, FlowEditPipeline
from edit4shape.guidance.pipelines.qwen_image_edit.state_tracker import FlowEditStateTracker


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
    tracker: FlowEditStateTracker       # 中间状态跟踪器（latents 都是 packed 格式）


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
    ) -> EditResult:
        """
        执行图像编辑。
        
        Args:
            rendered: 渲染图（Trellis 输出）
            condition: 条件图像（用户输入）
            cfg: flowedit 配置（cfg.guidance.flowedit）
        
        Returns:
            EditResult: 包含编辑后图像和 latent
        """
        pass


class SimplePipelineAdapter(BasePipelineAdapter):
    """
    FlowEditSimplePipeline 适配器。
    
    特点：
    - Source branch 使用解析式（不需要模型推理）
    - 速度快，适合快速迭代
    - 支持多种噪声模式（noise_mode）
    """
    
    def load(self, model_path: str, device: torch.device) -> None:
        self.pipe = FlowEditSimplePipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
    
    def edit(self, rendered: Image.Image, condition: Image.Image, cfg: Any) -> EditResult:
        output = self.pipe(
            image=[rendered, condition],
            target_prompt=cfg.target_prompt,
            generator=torch.manual_seed(cfg.seed),
            negative_prompt_tgt=cfg.negative_prompt_tgt,
            num_inference_steps=cfg.steps,
            init_image_index=0,
            target_prompt_image_indices=list(cfg.target_prompt_image_indices),
            true_cfg_scale_tgt=cfg.true_cfg_scale_tgt,
            n_max=cfg.n_max,
            fixed_noise=cfg.fixed_noise,
        )
        return EditResult(
            image=output.images[0],
            latent=output.latents,
            tracker=output.tracker,
        )


class FullPipelineAdapter(BasePipelineAdapter):
    """
    FlowEditPipeline 适配器（双分支模型推理）。
    
    特点：
    - Source 和 Target 两个分支都使用完整模型推理
    - 需要额外的 source_prompt 参数
    - 效果更好，但速度较慢
    """
    
    def load(self, model_path: str, device: torch.device) -> None:
        self.pipe = FlowEditPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
    
    def edit(self, rendered: Image.Image, condition: Image.Image, cfg: Any) -> EditResult:
        output = self.pipe(
            image=[rendered, condition],
            target_prompt=cfg.target_prompt,
            source_prompt=cfg.source_prompt,
            generator=torch.manual_seed(cfg.seed),
            negative_prompt_src=cfg.negative_prompt_src,
            negative_prompt_tgt=cfg.negative_prompt_tgt,
            num_inference_steps=cfg.steps,
            init_image_index=0,
            target_prompt_image_indices=list(cfg.target_prompt_image_indices),
            source_prompt_image_indices=list(cfg.source_prompt_image_indices),
            true_cfg_scale_src=cfg.true_cfg_scale_src,
            true_cfg_scale_tgt=cfg.true_cfg_scale_tgt,
            n_max=cfg.n_max,
            fixed_noise=cfg.fixed_noise,
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


def create_pipeline_adapter(pipeline_type: str) -> BasePipelineAdapter:
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

