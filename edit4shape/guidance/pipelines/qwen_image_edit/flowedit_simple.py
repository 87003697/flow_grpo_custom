# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlowEdit 统一 Pipeline。

支持通过 csd_weight 和 mse_weight 灵活组合 loss：
- csd_weight=1, mse_weight=0 → 纯 CSD（原 Contrast 模式）
- csd_weight=0, mse_weight=1 → 纯 MSE（原 Simple 模式）
- csd_weight=1, mse_weight=0.5 → 混合模式
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.utils import BaseOutput, is_torch_xla_available, logging
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline as BaseEditPlusPipeline,
    calculate_dimensions,
    retrieve_latents,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_timesteps

from edit4shape.guidance.pipelines.qwen_image_edit.trackers import StateTracker
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin, NoiseMode, sample_timesteps_uniform


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


# Constants
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


@dataclass
class FlowEditPipelineOutput(BaseOutput):
    """
    Output class for FlowEdit pipeline.
    
    Args:
        images: Generated images (PIL or tensor)
        latents: Edited latents in packed format [B, seq_len, C]
        tracker: StateTracker containing intermediate states
    """
    images: Any
    latents: Optional[torch.Tensor] = None
    tracker: Optional[StateTracker] = None


class FlowEditPipeline(BaseEditPlusPipeline, DifferentiableVAEMixin):
    """
    FlowEdit 统一 Pipeline。
    
    在每个去噪步同时记录：
    - z_edit: 编辑后的 latent（用于 MSE loss）
    - x0_high/x0_low: 高/低 CFG x0 预测（用于 CSD loss）
    
    通过 csd_weight 和 mse_weight 配置 loss 类型：
    - csd_weight=1, mse_weight=0 → 纯 CSD（原 Contrast 模式）
    - csd_weight=0, mse_weight=1 → 纯 MSE（原 Simple 模式）
    - csd_weight=1, mse_weight=0.5 → 混合模式
    """

    def _decode_latent_to_image(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ):
        """
        将 packed latent 解码为图像。
        
        Args:
            latent: packed latent (B, seq_len, C)
            height: 图像高度
            width: 图像宽度
            output_type: 输出类型 ("pil" 或 "pt")
        
        Returns:
            解码后的图像 (PIL.Image 列表或 tensor)
        """
        latents = self._unpack_latents(latent, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return image

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        target_prompt: Union[str, List[str]] = None,
        negative_prompt_tgt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # FlowEdit Params
        true_cfg_scale_tgt: float = 5.5,
        n_max: int = 20,
        noise_mode: NoiseMode = "random",  # 噪声模式: random/fixed/aligned
        src_latent: Optional[torch.Tensor] = None,  # 预编码的 src latent [B, seq_len, C]，用于可导编码
        use_mts_sampling: bool = False,  # 是否使用 MTS 采样（与 Distillation 一致）
    ):
        """
        FlowEdit pipeline for image editing.
        
        Args:
            image: Input image(s) for editing. Can be a single image or a list of images.
            target_prompt: Target prompt for editing.
            negative_prompt_tgt: Negative prompt for target branch CFG.
            true_cfg_scale_tgt: CFG scale for target branch.
            n_max: FlowEdit step range control.
            noise_mode: 噪声模式 (random/fixed/aligned)
            src_latent: 预编码的 src latent，用于可导编码
            use_mts_sampling: 是否使用 MTS 采样（在 [0.02, 0.98] 范围内均匀分区随机采样）
        """
        # Calculate dimensions from image
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs
        self.check_inputs(
            target_prompt,
            height,
            width,
            negative_prompt=negative_prompt_tgt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if target_prompt is not None and isinstance(target_prompt, str):
            batch_size = 1
        elif target_prompt is not None and isinstance(target_prompt, list):
            batch_size = len(target_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        has_neg_prompt_tgt = negative_prompt_tgt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale_tgt > 1 and not has_neg_prompt_tgt:
            logger.warning("true_cfg_scale_tgt > 1 but negative_prompt_tgt is not provided.")
        elif true_cfg_scale_tgt <= 1 and has_neg_prompt_tgt:
            logger.warning("negative_prompt_tgt is passed but CFG is not enabled since true_cfg_scale_tgt <= 1")

        # Prepare images for VLM encoding（固定使用条件图 index=1）
        cond_images_tgt = [condition_images[1]]

        do_true_cfg_tgt = has_neg_prompt_tgt and true_cfg_scale_tgt > 1

        # 检测 target_prompt 是否与 negative_prompt_tgt 相同（用于复用 embedding 和跳过 uncond 推理）
        tgt_neg_same = (target_prompt == negative_prompt_tgt)

        # Encode Target Prompt
        prompt_embeds_tgt, prompt_embeds_mask_tgt = self.encode_prompt(
            image=cond_images_tgt,
            prompt=target_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        txt_seq_lens_tgt = prompt_embeds_mask_tgt.sum(dim=1).tolist()

        if do_true_cfg_tgt:
            if tgt_neg_same:
                # target_prompt == negative_prompt_tgt，复用 target embedding
                negative_prompt_embeds_tgt = prompt_embeds_tgt
                negative_prompt_embeds_mask_tgt = prompt_embeds_mask_tgt
                negative_txt_seq_lens_tgt = txt_seq_lens_tgt
            else:
                negative_prompt_embeds_tgt, negative_prompt_embeds_mask_tgt = self.encode_prompt(
                    image=cond_images_tgt,
                    prompt=negative_prompt_tgt,
                    prompt_embeds=negative_prompt_embeds,
                    prompt_embeds_mask=negative_prompt_embeds_mask,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
                negative_txt_seq_lens_tgt = negative_prompt_embeds_mask_tgt.sum(dim=1).tolist()

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        # Prepare ALL latents (including noise and condition latents for all images)
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_tgt.dtype,
            device,
            generator,
            latents,
        )

        # Parse image_latents into individual tensors
        all_latents_list = []
        current_idx = 0
        for (vw, vh) in vae_image_sizes:
            h_lat = vh // (self.vae_scale_factor * 2)
            w_lat = vw // (self.vae_scale_factor * 2)
            seq_len = h_lat * w_lat
            # image_latents shape: [B, TotalSeq, C]
            img_latent = image_latents[:, current_idx : current_idx + seq_len, :]  # shape: [B, seq_len, C]
            all_latents_list.append(img_latent)
            current_idx += seq_len

        # Extract base latent for editing (clean latent, not noise)
        if src_latent is not None:
            # 使用外部传入的预编码 latent（可导版本）替换 x_src
            x_src = src_latent.clone()  # shape: [B, seq_len, C]
        else:
            # 原来的逻辑：使用 pipeline 内部编码的 latent
            x_src = all_latents_list[0].clone()  # 渲染图的 latent, shape: [B, seq_len, C]
        z_edit = x_src.clone()  # shape: [B, seq_len, C]

        # Helper to construct model inputs based on indices
        def get_latent_model_input_and_img_shapes_tgt(z_t):
            # 1. Concat condition latent（固定使用条件图 index=1）
            cond_latent = all_latents_list[1]  # 条件图的 latent
            latent_model_input = torch.cat([z_t, cond_latent], dim=1)

            # 2. Construct img_shapes
            main_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
            vw, vh = vae_image_sizes[1]
            cond_shape = (1, vh // self.vae_scale_factor // 2, vw // self.vae_scale_factor // 2)
            img_shapes = [main_shape, cond_shape]

            return latent_model_input, [img_shapes] * batch_size

        # 5. Prepare timesteps
        if use_mts_sampling:
            # MTS 采样：在 1000 步范围内均匀分区随机采样
            num_train_timesteps = 1000
            min_step_percent = 0.02  # 硬编码最小时间步百分比
            max_step_percent = 0.98  # 硬编码最大时间步百分比
            
            # 采样 num_inference_steps 个时间步（覆盖完整范围）
            # 跳过逻辑会筛选出后 n_max 步
            min_step = int(num_train_timesteps * min_step_percent)  # 20
            max_step = int(num_train_timesteps * max_step_percent)  # 980
            
            timesteps_list = sample_timesteps_uniform(
                min_step=min_step,
                max_step=max_step,
                num_steps=num_inference_steps,  # 采样完整步数
                batch_size=batch_size,
                device=device,
                generator=generator,
                ascending=False,  # FlowEdit 从大到小
            )
            # 转换为 1D Tensor（取每个 batch 的第一个值，因为 batch 内相同）
            timesteps = torch.stack([t[0:1] for t in timesteps_list]).squeeze(-1)  # (num_inference_steps,)
            self._num_timesteps = num_inference_steps
        else:
            # 原有逻辑：使用 scheduler 时间步
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            # Using main latent shape for shift calc
            image_seq_len = x_src.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            self._num_timesteps = len(timesteps)
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # 6. FlowEdit Loop
        self.scheduler.set_begin_index(0)
        
        # 初始化 StateTracker（统一记录 z_edit 和 x0_high/x0_low）
        tracker = StateTracker(height=height, width=width)
        
        # 初始化噪声管理
        tracker.init(x_src, mode=noise_mode)  # [B, seq_len, C]
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # FlowEdit Step Logic
                # Skip initial steps if n_max set（两种模式都跳过）
                if num_inference_steps - i > n_max:
                    progress_bar.update()
                    continue

                self._current_timestep = t
                t_curr = t / 1000.0
                t_prev = timesteps[i+1] / 1000.0 if i < len(timesteps) - 1 else torch.tensor(0.0, device=device, dtype=t.dtype)
                dt = t_prev - t_curr
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # ========== FlowEdit 差分采样阶段 ==========
                # Source Branch (Analytical)
                noise = tracker.get_noise(x_src)  # [B, seq_len, C]
                latents_src = (1 - t_curr) * x_src + t_curr * noise  # [B, seq_len, C]
                noise_pred_src = noise - x_src  # [B, seq_len, C]

                # Target Branch (Model Inference Required)
                latents_tgt = z_edit + latents_src - x_src  # shape: [B, seq_len, C]
                
                # Target Model Input (with condition image)
                latent_model_input_tgt, img_shapes_tgt = get_latent_model_input_and_img_shapes_tgt(latents_tgt)
                
                # Calc noise_pred_tgt with Transformer
                with self.transformer.cache_context("cond"):
                    noise_pred_tgt = self.transformer(
                        hidden_states=latent_model_input_tgt,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask_tgt,
                        encoder_hidden_states=prompt_embeds_tgt,
                        img_shapes=img_shapes_tgt,
                        txt_seq_lens=txt_seq_lens_tgt,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_tgt = noise_pred_tgt[:, :x_src.shape[1]]  # shape: [B, seq_len, C]

                if do_true_cfg_tgt and not tgt_neg_same:
                    # 仅当 target_prompt != negative_prompt_tgt 时才需要 uncond 推理
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred_tgt = self.transformer(
                            hidden_states=latent_model_input_tgt,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask_tgt,
                            encoder_hidden_states=negative_prompt_embeds_tgt,
                            img_shapes=img_shapes_tgt,
                            txt_seq_lens=negative_txt_seq_lens_tgt,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred_tgt = neg_noise_pred_tgt[:, :x_src.shape[1]]  # shape: [B, seq_len, C]

                    # 保存纯 cond 和纯 uncond 预测用于 CSD loss
                    v_cond = noise_pred_tgt  # [B, seq_len, C] 纯条件预测 (cfg=1)
                    v_uncond = neg_noise_pred_tgt  # [B, seq_len, C] 纯无条件预测 (cfg=0)

                    # CFG combine with L2 norm rescale (仅用于 z_edit 更新)
                    comb_pred = neg_noise_pred_tgt + true_cfg_scale_tgt * (noise_pred_tgt - neg_noise_pred_tgt)  # shape: [B, seq_len, C]
                    cond_norm = torch.norm(noise_pred_tgt, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    v_cfg = comb_pred * (cond_norm / noise_norm)  # shape: [B, seq_len, C] CFG后用于z_edit
                else:
                    # 无 CFG 时，三者相同
                    v_cond = noise_pred_tgt
                    v_uncond = noise_pred_tgt
                    v_cfg = noise_pred_tgt

                # Update z_edit 使用 CFG 后的结果
                v_delta = v_cfg - noise_pred_src  # [B, seq_len, C] packed
                z_edit = z_edit + dt * v_delta  # [B, seq_len, C] packed
                
                # ========== 计算 x0_high 和 x0_low ==========
                # x0_high = 纯 cond 预测的 x0 (cfg=1)
                # x0_low = 纯 uncond 预测的 x0 (cfg=0)
                x0_high = latents_tgt - t_curr * v_cond    # [B, seq_len, C]
                x0_low = latents_tgt - t_curr * v_uncond   # [B, seq_len, C]
                
                # 记录状态（x0_pred = z_edit）
                tracker.record(z_edit, float(t_curr), x0_high, x0_low)
                
                # 累积更新噪声：noise -= (v_cond - v_uncond) * (1 - t)
                tracker.update(
                    v_cond=v_cond,
                    v_uncond=v_uncond,
                    v_cfg=v_cfg,
                    t=float(t_curr),
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = z_edit
        self._current_timestep = None

        # 保存 packed latent 用于返回
        packed_latents = latents.clone()  # shape: [B, seq_len, C]

        if output_type == "latent":
            image = latents
        else:
            image = self._decode_latent_to_image(latents, height, width, output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, packed_latents, tracker)

        return FlowEditPipelineOutput(
            images=image, 
            latents=packed_latents,
            tracker=tracker,
        )
