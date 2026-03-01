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
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

import torch

from edit4shape.systems.utils import composite_alpha
from edit4shape.guidance.base import GuidanceResult, BaseGuidance
from edit4shape.guidance.pipeline_parallel import PipelineParallelMixin
from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
from edit4shape.guidance.pipelines.qwen_image_edit.trackers import StateTracker


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EditOutput:
    """单张图编辑输出"""
    image: Image.Image                                  # 编辑后的图像
    latent: torch.Tensor                                # [1, seq, C] 编辑后的 latent
    tracker_tgt: Optional[StateTracker] = None          # tgt 分支跟踪器
    tracker_src: Optional[StateTracker] = None          # src 分支跟踪器


@dataclass
class FlowEditPipelineOutput:
    """FlowEdit Pipeline 输出"""
    edited_images: List[Image.Image]                    # [N] 编辑后的图像
    edited_tensor: torch.Tensor                         # [N, C, H, W] 编辑后的图像 tensor（用于可视化）
    latent_after: torch.Tensor                          # [N, seq, C] 编辑后的 latent
    trackers_tgt: Optional[List[StateTracker]] = None   # [N] tgt 分支跟踪器列表
    trackers_src: Optional[List[StateTracker]] = None   # [N] src 分支跟踪器列表


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
                - flowedit: FlowEdit 专属 init 配置
            train_device: 训练使用的设备
        """
        super().__init__(guidance_cfg, train_device)

        self.flowedit_cfg = guidance_cfg[guidance_cfg.type]  # = guidance_cfg.flowedit
        model_path = guidance_cfg.model_path

        logging.info(f"[FlowEditGuidance] Loading pipeline on {self.device}...")
        logging.info(f"[FlowEditGuidance] Model: {model_path}")

        self.pipe = FlowEditFullPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

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
        condition_pil = composite_alpha(condition_pil, self.bg_color)
        latent_before = latent_before.to(dtype=torch.bfloat16)

        device = torch.device(self.pipe._execution_device)
        generator = torch.Generator(device=device).manual_seed(flowedit_cfg.seed)
        ic = self.flowedit_cfg

        # 从 runtime loss config 推导分支启用开关
        loss_cfg = flowedit_cfg.loss
        use_tgt = loss_cfg.tgt_branch > 0
        use_src = loss_cfg.src_branch > 0

        output = self.pipe(
            image=[rendered_pil, condition_pil],
            target_prompt=flowedit_cfg.target_prompt,
            source_prompt=flowedit_cfg.source_prompt,
            generator=generator,
            negative_prompt_src=flowedit_cfg.negative_prompt_src,
            negative_prompt_tgt=flowedit_cfg.negative_prompt_tgt,
            num_inference_steps=ic.steps,
            true_cfg_scale_src=flowedit_cfg.true_cfg_scale_src,
            true_cfg_scale_tgt=flowedit_cfg.true_cfg_scale_tgt,
            n_max=ic.n_max,
            noise_mode=ic.noise_mode,
            use_tgt_record=use_tgt,
            use_src_record=use_src,
            csd_pos_mode=ic.csd_pos_mode,
            csd_neg_mode=ic.csd_neg_mode,
            remove_tgt_neg=ic.remove_tgt_neg,
            src_latent=latent_before,
            height=self.edit_resolution,
            width=self.edit_resolution,
        )

        return EditOutput(
            image=output.images[0],
            latent=output.latents,
            tracker_tgt=output.tracker_tgt,
            tracker_src=output.tracker_src,
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

        edited_images, latents = [], []
        trackers_tgt: List[StateTracker] = []
        trackers_src: List[StateTracker] = []

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
                if output.tracker_tgt is not None:
                    trackers_tgt.append(output.tracker_tgt)
                if output.tracker_src is not None:
                    trackers_src.append(output.tracker_src)
        
        # 转换编辑后的图像为 tensor
        edited_tensor = self.pils_to_tensor(edited_images, (W, H))

        return FlowEditPipelineOutput(
            edited_images=edited_images,
            edited_tensor=edited_tensor,
            latent_after=torch.cat(latents, dim=0),
            trackers_tgt=trackers_tgt if trackers_tgt else None,
            trackers_src=trackers_src if trackers_src else None,
        )

    # =========================================================================
    # Loss 计算（实现抽象方法）
    # =========================================================================

    def _compute_branch_loss(
        self,
        src_latent: torch.Tensor,
        trackers: List[StateTracker],
        guidance_cfg: Any,
    ) -> torch.Tensor:
        """计算单分支 loss（对所有 view 取平均）。"""
        loss_cfg = guidance_cfg.loss
        losses = []
        for i, tracker in enumerate(trackers):
            single = src_latent[i:i+1]  # [1, seq, C]
            loss = tracker.loss(
                src=single,
                csd_weight=loss_cfg.latent_csd,
                mse_weight=loss_cfg.latent_mse,
                reduce=guidance_cfg.reduce_mode,
                ada=guidance_cfg.ada_normalize,
                eps=guidance_cfg.ada_eps,
            )
            losses.append(loss)
        return torch.stack(losses).mean()  # scalar

    def _compute_loss(
        self,
        src_latent: torch.Tensor,
        pipeline_output: FlowEditPipelineOutput,
        comp_rgb: torch.Tensor,
        guidance_cfg: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Latent Loss（分支独立计算，按权重加和）。

        total = tgt_branch * tgt_loss + src_branch * src_loss
        """
        loss_cfg = guidance_cfg.loss
        total_loss = torch.tensor(0.0, device=src_latent.device, dtype=src_latent.dtype)
        loss_dict: Dict[str, torch.Tensor] = {}

        # ---- tgt branch ----
        if loss_cfg.tgt_branch > 0 and pipeline_output.trackers_tgt:
            loss_tgt = self._compute_branch_loss(
                src_latent, pipeline_output.trackers_tgt, guidance_cfg,
            )
            total_loss = total_loss + loss_cfg.tgt_branch * loss_tgt
            loss_dict["latent_tgt"] = loss_tgt.detach()

        # ---- src branch ----
        if loss_cfg.src_branch > 0 and pipeline_output.trackers_src:
            loss_src = self._compute_branch_loss(
                src_latent, pipeline_output.trackers_src, guidance_cfg,
            )
            total_loss = total_loss + loss_cfg.src_branch * loss_src
            loss_dict["latent_src"] = loss_src.detach()

        return total_loss, loss_dict

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
            trackers=pipeline_output.trackers_tgt,  # FlowEdit 专属（tgt 分支）
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
