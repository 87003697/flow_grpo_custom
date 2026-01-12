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
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


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
        images: Generated images (PIL or tensor) - 正样本
        latents: Edited latents in packed format [B, seq_len, C] (正样本)
        latents_neg: 反向一步的 packed latent [B, seq_len, C] (负样本)
        images_neg: 负样本图像 (PIL or tensor)
    """
    images: Any
    latents: Optional[torch.Tensor] = None
    latents_neg: Optional[torch.Tensor] = None
    images_neg: Any = None


class FlowEditPipeline(BaseEditPlusPipeline):
    """
    FlowEdit pipeline for image editing using differential velocity fields.
    
    Inherits from QwenImageEditPlusPipeline and overrides __call__ with FlowEdit algorithm.
    This version uses full model inference for both source and target branches.
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
        source_prompt: Union[str, List[str]] = None,
        negative_prompt_src: Union[str, List[str]] = None,
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
        init_image_index: int = 0,
        # FlowEdit Params
        source_prompt_image_indices: Optional[List[int]] = None,
        target_prompt_image_indices: Optional[List[int]] = None,
        true_cfg_scale_src: float = 1.5,
        true_cfg_scale_tgt: float = 5.5,
        n_max: int = 20,
        n_min: int = 0,
        cfg_rescale: bool = False,
        shared_noise: bool = False,  # 是否在所有 step 使用相同噪声
    ):
        """
        FlowEdit pipeline for image editing with full dual-branch model inference.

        Args:
            image: Input image(s) for editing.
            target_prompt: Target prompt for editing.
            source_prompt: Source prompt describing the original image.
            negative_prompt_src: Negative prompt for source branch CFG.
            negative_prompt_tgt: Negative prompt for target branch CFG.
            true_cfg_scale_src: CFG scale for source branch.
            true_cfg_scale_tgt: CFG scale for target branch.
            n_max: FlowEdit step range control.
            source_prompt_image_indices: Image indices for source prompt encoding.
            target_prompt_image_indices: Image indices for target prompt encoding.
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
            if init_image_index < 0 or init_image_index >= len(vae_images):
                raise ValueError(f"`init_image_index` must be in [0, {len(vae_images) - 1}], got {init_image_index}")

        has_neg_prompt_src = negative_prompt_src is not None
        has_neg_prompt_tgt = negative_prompt_tgt is not None

        if true_cfg_scale_src > 1 and not has_neg_prompt_src:
            logger.warning("true_cfg_scale_src > 1 but negative_prompt_src is not provided.")
        if true_cfg_scale_tgt > 1 and not has_neg_prompt_tgt:
            logger.warning("true_cfg_scale_tgt > 1 but negative_prompt_tgt is not provided.")

        # Handle indices defaults
        if source_prompt_image_indices is None:
            source_prompt_image_indices = [init_image_index]
        if target_prompt_image_indices is None:
            target_prompt_image_indices = [init_image_index]

        # Prepare images for VLM encoding
        cond_images_src = [condition_images[i] for i in source_prompt_image_indices]
        cond_images_tgt = [condition_images[i] for i in target_prompt_image_indices]

        do_true_cfg_src = has_neg_prompt_src and true_cfg_scale_src > 1
        do_true_cfg_tgt = has_neg_prompt_tgt and true_cfg_scale_tgt > 1

        # 检测 prompt 是否与对应的 negative_prompt 相同（用于复用 embedding 和跳过 uncond 推理）
        src_neg_same = (source_prompt == negative_prompt_src)
        tgt_neg_same = (target_prompt == negative_prompt_tgt)

        # Encode Source Prompt
        prompt_embeds_src, prompt_embeds_mask_src = self.encode_prompt(
            image=cond_images_src,
            prompt=source_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        txt_seq_lens_src = prompt_embeds_mask_src.sum(dim=1).tolist()

        if do_true_cfg_src:
            if src_neg_same:
                # source_prompt == negative_prompt_src，复用 source embedding
                negative_prompt_embeds_src = prompt_embeds_src
                negative_prompt_embeds_mask_src = prompt_embeds_mask_src
                negative_txt_seq_lens_src = txt_seq_lens_src
            else:
                negative_prompt_embeds_src, negative_prompt_embeds_mask_src = self.encode_prompt(
                    image=cond_images_src,
                    prompt=negative_prompt_src,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
                negative_txt_seq_lens_src = negative_prompt_embeds_mask_src.sum(dim=1).tolist()

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
            prompt_embeds_src.dtype,
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
        x_src = all_latents_list[init_image_index].clone()  # shape: [B, seq_len, C]
        z_edit = x_src.clone()  # shape: [B, seq_len, C]

        # Helper to construct model inputs based on indices
        def get_latent_model_input_and_img_shapes(z_t, indices):
            # 1. Concat condition latents
            conds = [all_latents_list[i] for i in indices]
            if conds:
                cond_latents = torch.cat(conds, dim=1)
                latent_model_input = torch.cat([z_t, cond_latents], dim=1)
            else:
                latent_model_input = z_t

            # 2. Construct img_shapes
            # First element is main generated image shape
            main_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
            img_shapes = [main_shape]
            for i in indices:
                vw, vh = vae_image_sizes[i]
                img_shapes.append((1, vh // self.vae_scale_factor // 2, vw // self.vae_scale_factor // 2))

            return latent_model_input, [img_shapes] * batch_size

        # 5. Prepare timesteps
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
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance
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
        z_edit_neg = None  # 用于存储反向一步的负样本 latent
        first_active_step = True  # 标记是否是第一个真正执行的步骤
        xt_tar = None  # 用于 SDEDIT 阶段的 latent
        
        # 如果 shared_noise=True，预采样噪声供所有 step 共用
        fixed_noise = torch.randn_like(x_src) if shared_noise else None  # shape: [B, seq_len, C]
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # FlowEdit Step Logic
                # Skip initial steps if n_max set
                if num_inference_steps - i > n_max:
                    progress_bar.update()
                    continue

                self._current_timestep = t
                t_curr = t / 1000.0
                t_prev = timesteps[i + 1] / 1000.0 if i < len(timesteps) - 1 else torch.tensor(0.0, device=device, dtype=t.dtype)
                dt = t_prev - t_curr
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if num_inference_steps - i > n_min:
                    # ========== FlowEdit 差分采样阶段 ==========
                    # 1. Source Branch (Full Model Inference)
                    noise = fixed_noise if shared_noise else torch.randn_like(x_src)  # shape: [B, seq_len, C]
                    latents_src = (1 - t_curr) * x_src + t_curr * noise  # shape: [B, seq_len, C]

                    # Source Model Input
                    latent_model_input_src, img_shapes_src = get_latent_model_input_and_img_shapes(
                        latents_src, source_prompt_image_indices
                    )

                    # Calc noise_pred_src
                    with self.transformer.cache_context("cond"):
                        noise_pred_src = self.transformer(
                            hidden_states=latent_model_input_src,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask_src,
                            encoder_hidden_states=prompt_embeds_src,
                            img_shapes=img_shapes_src,
                            txt_seq_lens=txt_seq_lens_src,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred_src = noise_pred_src[:, :x_src.shape[1]]  # shape: [B, seq_len, C]

                    if do_true_cfg_src and not src_neg_same:
                        # 仅当 source_prompt != negative_prompt_src 时才需要 uncond 推理
                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred_src = self.transformer(
                                hidden_states=latent_model_input_src,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask_src,
                                encoder_hidden_states=negative_prompt_embeds_src,
                                img_shapes=img_shapes_src,
                                txt_seq_lens=negative_txt_seq_lens_src,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                            neg_noise_pred_src = neg_noise_pred_src[:, :x_src.shape[1]]  # shape: [B, seq_len, C]

                        # Standard CFG combine
                        noise_pred_src = neg_noise_pred_src + true_cfg_scale_src * (noise_pred_src - neg_noise_pred_src)  # shape: [B, seq_len, C]

                    # 2. Target Branch
                    latents_tgt = z_edit + latents_src - x_src  # shape: [B, seq_len, C]

                    # Target Model Input
                    latent_model_input_tgt, img_shapes_tgt = get_latent_model_input_and_img_shapes(
                        latents_tgt, target_prompt_image_indices
                    )

                    # Calc noise_pred_tgt
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

                        # Standard CFG combine
                        noise_pred_tgt = neg_noise_pred_tgt + true_cfg_scale_tgt * (noise_pred_tgt - neg_noise_pred_tgt)  # shape: [B, seq_len, C]

                    v_delta = noise_pred_tgt - noise_pred_src  # shape: [B, seq_len, C]

                    # 3. Update z_edit (Euler step)
                    z_edit = z_edit + dt * v_delta  # shape: [B, seq_len, C]
                    
                    # 捕获第一个活跃步骤的反向 Latent 作为负样本
                    if first_active_step:
                        z_edit_neg = x_src - dt * v_delta  # shape: [B, seq_len, C]
                        first_active_step = False

                else:
                    # ========== DDIM 风格常规采样阶段 (最后 n_min 步) ==========
                    if i == num_inference_steps - n_min:
                        # 直接使用 z_edit，不加噪（DDIM 风格）
                        xt_tar = z_edit.clone()  # shape: [B, seq_len, C]
                    
                    # 常规采样：只用 target prompt
                    latent_model_input_tgt, img_shapes_tgt = get_latent_model_input_and_img_shapes(
                        xt_tar, target_prompt_image_indices
                    )
                    
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
                        # CFG（与 QwenImageEditPlusPipeline 一致，cfg_scale=4.0）
                        comb_pred = neg_noise_pred_tgt + 4.0 * (noise_pred_tgt - neg_noise_pred_tgt)  # shape: [B, seq_len, C]
                        if cfg_rescale:
                            # L2 norm rescale
                            cond_norm = torch.norm(noise_pred_tgt, dim=-1, keepdim=True)
                            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                            noise_pred_tgt = comb_pred * (cond_norm / noise_norm)  # shape: [B, seq_len, C]
                        else:
                            noise_pred_tgt = comb_pred  # shape: [B, seq_len, C]

                    # Euler 步更新
                    xt_tar = xt_tar + dt * noise_pred_tgt  # shape: [B, seq_len, C]

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

        latents = z_edit if n_min == 0 else xt_tar
        self._current_timestep = None

        # Save packed latent for return
        packed_latents = latents.clone()  # shape: [B, seq_len, C]

        if output_type == "latent":
            image = latents
            image_neg = z_edit_neg
        else:
            # 解码正样本
            image = self._decode_latent_to_image(latents, height, width, output_type)
            
            # 解码负样本
            image_neg = None
            if z_edit_neg is not None:
                image_neg = self._decode_latent_to_image(z_edit_neg, height, width, output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, packed_latents, z_edit_neg, image_neg)

        return FlowEditPipelineOutput(
            images=image, 
            latents=packed_latents,
            latents_neg=z_edit_neg,
            images_neg=image_neg,
        )
