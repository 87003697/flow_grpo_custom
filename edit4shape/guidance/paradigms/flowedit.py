"""
FlowEdit Guidance 模块。

将 Qwen-Image-Edit 模型加载到 Guidance 设备上，
与 Trellis 训练进程共存于同一 Python 进程。

★ Init / Runtime 分离：
- __init__: 只加载 Pipeline 模型（重量级，一次性）
- compute_guidance: 每次调用传入 guidance_cfg（prompt / loss 权重等）

数据流（继承自 BaseGuidance，真 Loss 模式）：
    1. 格式转换（父类）
    2. 编码到 latent（父类，一次）
    3. 调用 FlowEdit Pipeline（_run_pipeline，多步编辑）
    4. 通过 Tracker.loss() 计算真 loss（_compute_loss）
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image

import torch

from edit4shape.systems.utils import composite_alpha_to_white
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipeline_parallel import PipelineParallelMixin
from edit4shape.guidance.pipelines.adapters import create_pipeline_adapter
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import StateTracker


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EditOutput:
    """单张图编辑输出"""
    image: Image.Image                  # 编辑后的图像
    latent: torch.Tensor                # [1, seq, C] 编辑后的 latent
    tracker: StateTracker               # 中间状态跟踪器


@dataclass
class FlowEditPipelineOutput:
    """FlowEdit Pipeline 输出"""
    edited_images: List[Image.Image]    # [N] 编辑后的图像
    edited_tensor: torch.Tensor         # [N, C, H, W] 编辑后的图像 tensor（用于可视化）
    latent_after: torch.Tensor          # [N, seq, C] 编辑后的 latent
    trackers: List[StateTracker]        # [N] 中间状态跟踪器


# =============================================================================
# FlowEditGuidance
# =============================================================================

class FlowEditGuidance(BaseGuidance):
    """
    FlowEdit Guidance（真 Loss 模式，纯 Latent Loss）。

    ★ Init / Runtime 分离：
    - __init__: 加载 Pipeline 模型
    - compute_guidance(..., guidance_cfg=...): 运行时参数通过 guidance_cfg 传入

    数据流：
        1. 格式转换（父类）
        2. 编码到 latent（父类，一次）
        3. 调用 FlowEdit Pipeline（_run_pipeline，传入 guidance_cfg）
        4. 计算 Latent Loss（_compute_loss，通过 Tracker.loss()）
    """

    loss_key = "flowedit"

    def __init__(self, guidance_cfg: Any, train_device: torch.device):
        """
        初始化 FlowEdit Guidance（只加载模型）。

        Args:
            guidance_cfg: Guidance 初始化配置（cfg.guidance），包含：
                - type: "flowedit"
                - model_path: 模型路径
                - edit_resolution: VAE 编码分辨率
                - flowedit.pipeline_type: "simple" | "full"（FlowEdit 专属）
            train_device: 训练使用的设备
        """
        super().__init__(guidance_cfg, train_device)

        flowedit_init_cfg = guidance_cfg[guidance_cfg.type]  # = guidance_cfg.flowedit
        pipeline_type = flowedit_init_cfg.pipeline_type
        model_path = guidance_cfg.model_path

        logging.info(f"[FlowEditGuidance] Loading pipeline on {self.device}...")
        logging.info(f"[FlowEditGuidance] Pipeline: {pipeline_type}, Model: {model_path}")

        self.adapter = create_pipeline_adapter(pipeline_type)
        self.adapter.load(model_path, self.device)
        self.pipe = self.adapter.pipe

        logging.info(f"[FlowEditGuidance] Ready.")

    # =========================================================================
    # FlowEdit 单张编辑
    # =========================================================================

    def _edit_single(
        self,
        rendered_pil: Image.Image,
        condition_pil: Image.Image,
        latent_before: torch.Tensor,
        flowedit_cfg: Any,
    ) -> EditOutput:
        """执行单张图的 FlowEdit 编辑。"""
        condition_pil = composite_alpha_to_white(condition_pil)
        latent_before = latent_before.to(dtype=torch.bfloat16)

        result = self.adapter.edit(
            rendered_pil,
            condition_pil,
            flowedit_cfg,
            src_latent=latent_before,
            height=self.edit_resolution,
            width=self.edit_resolution,
        )

        return EditOutput(
            image=result.image,
            latent=result.latent,
            tracker=result.tracker,
        )

    # =========================================================================
    # Pipeline 调用（实现抽象方法）
    # =========================================================================

    def _run_pipeline(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        src_latent: torch.Tensor,
        guidance_cfg: Any,
        B: int,
        V: int,
    ) -> FlowEditPipelineOutput:
        """
        执行 FlowEdit Pipeline（无梯度，多步编辑）。
        
        Args:
            comp_rgb: [N, C, H, W] 渲染图
            condition_images: [B] 条件图
            src_latent: [N, seq, C] latent（已 detach）
            guidance_cfg: 运行时配置（prompt / steps / cfg scales 等）
            B, V: batch size 和 views
        
        Returns:
            FlowEditPipelineOutput
        """
        N = B * V
        H, W = comp_rgb.shape[2], comp_rgb.shape[3]

        edited_images, latents, trackers = [], [], []

        for b in range(B):
            for v in range(V):
                i = b * V + v
                
                # 使用传入的 latent
                rendered_pil = self.tensor_to_pil(comp_rgb[i])
                output = self._edit_single(
                    rendered_pil,
                    condition_images[b],
                    src_latent[i:i+1],
                    flowedit_cfg=guidance_cfg,
                )
                
                edited_images.append(output.image)
                latents.append(output.latent)
                trackers.append(output.tracker)
        
        # 转换编辑后的图像为 tensor
        edited_tensor = self.pils_to_tensor(edited_images, (W, H))

        return FlowEditPipelineOutput(
            edited_images=edited_images,
            edited_tensor=edited_tensor,
            latent_after=torch.cat(latents, dim=0),
            trackers=trackers,
        )

    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================

    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        comp_rgb: torch.Tensor,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Latent Loss（通过 Tracker.loss()，支持 CSD + MSE 混合）。

        Loss = csd_weight * CSD_Loss + mse_weight * MSE_Loss
        """
        loss_cfg = guidance_cfg.loss

        losses = []
        for i, tracker in enumerate(pipeline_output.trackers):
            single = src_latent[i:i+1]
            loss = tracker.loss(
                src=single,
                csd_weight=loss_cfg.latent_csd,
                mse_weight=loss_cfg.latent_mse,
                reduce=guidance_cfg.reduce_mode,
                ada=guidance_cfg.ada_normalize,
                eps=guidance_cfg.ada_eps,
            )
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        return total_loss, {"latent": total_loss.detach()}

    # =========================================================================
    # 构建返回结果
    # =========================================================================

    def _build_result(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        pipeline_output: FlowEditPipelineOutput,
        B: int, V: int, C: int, H: int, W: int,
        source_device: torch.device,
    ) -> GuidanceResult:
        """
        构建返回结果，添加 FlowEdit 专属字段。
        """
        edited_for_vis = pipeline_output.edited_tensor.reshape(B, V, C, H, W).to(source_device)

        return GuidanceResult(
            loss=loss,
            edited_imgs=edited_for_vis,
            loss_dict={self.loss_key: loss.detach(), **loss_dict},
            trackers=pipeline_output.trackers,  # FlowEdit 专属
        )

    def cleanup(self) -> None:
        """释放模型显存"""
        logging.info("[FlowEditGuidance] Cleaning up...")
        del self.pipe
        torch.cuda.empty_cache()
        logging.info("[FlowEditGuidance] Cleanup done.")


# =============================================================================
# FlowEditGuidancePP - 流水线并行版本
# =============================================================================

class FlowEditGuidancePP(PipelineParallelMixin, FlowEditGuidance):
    """FlowEdit Guidance + 流水线并行。"""

    def __init__(self, guidance_cfg, train_device: torch.device):
        super().__init__(guidance_cfg, train_device)
        self._init_pipeline_parallel(num_streams=2)
        logging.info(f"[FlowEditGuidancePP] Pipeline parallelism enabled.")
