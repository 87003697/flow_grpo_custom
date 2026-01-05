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
from typing import Any, Callable, Dict, List, Literal, Optional, Union

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
        images: Generated images (PIL or tensor)
        latents: Edited latents in packed format [B, seq_len, C]
    """
    images: Any
    latents: Optional[torch.Tensor] = None


class FlowEditStateTracker:
    """
    Records state at each step of FlowEdit trajectory and manages noise generation.
    """
    
    def __init__(self):
        self._states: Dict[int, Dict[str, Any]] = {}
        self._current_step: int = -1
        self._x_src: Optional[torch.Tensor] = None
        self._init_noise: Optional[torch.Tensor] = None
        self._velocity_fixed_noise: Optional[torch.Tensor] = None  # 缓存 velocity 反演的 noise
    
    def init(self, x_src: torch.Tensor):
        """Initialize with source latent."""
        self._x_src = x_src
        self._init_noise = torch.randn_like(x_src)  # shape: [B, seq_len, C]
        self._velocity_fixed_noise = None  # 重置缓存
    
    def reset(self):
        """Clear all states."""
        self._states = {}
        self._current_step = -1
        self._x_src = None
        self._init_noise = None
        self._velocity_fixed_noise = None  # 重置缓存
    
    def get_noise(self, t_curr: float, mode: Literal["random", "fixed", "velocity", "velocity_fixed"] = "fixed") -> torch.Tensor:
        """
        Get noise based on mode.
        
        Args:
            t_curr: Current timestep (normalized)
            mode: Noise generation mode - "random" | "fixed" | "velocity" | "velocity_fixed"
        
        Returns:
            noise: [B, seq_len, C]
        """
        if self._x_src is None:
            raise ValueError("StateTracker not initialized. Call init(x_src) first.")
        
        if mode == "random":
            return torch.randn_like(self._x_src)
        
        elif mode == "fixed":
            return self._init_noise
        
        elif mode == "velocity":
            if not self.has_prev:
                return self._init_noise
            # epsilon = z_t + (1-t) * v
            prev_latents_tgt = self.get_prev("latents_tgt")
            prev_noise_pred_tgt = self.get_prev("noise_pred_tgt")
            prev_t = self.get_prev("t")
            return prev_latents_tgt + (1 - prev_t) * prev_noise_pred_tgt
        
        elif mode == "velocity_fixed":
            # 如果已经缓存了 velocity 反演的 noise，直接返回
            if self._velocity_fixed_noise is not None:
                return self._velocity_fixed_noise
            
            # 否则计算一次并缓存
            if not self.has_prev:
                # 第一步没有 prev，用 init_noise
                self._velocity_fixed_noise = self._init_noise.clone()
            else:
                # epsilon = z_t + (1-t) * v
                prev_latents_tgt = self.get_prev("latents_tgt")
                prev_noise_pred_tgt = self.get_prev("noise_pred_tgt")
                prev_t = self.get_prev("t")
                self._velocity_fixed_noise = prev_latents_tgt + (1 - prev_t) * prev_noise_pred_tgt
            
            return self._velocity_fixed_noise
        
        else:
            raise ValueError(f"Unknown noise mode: {mode}")
    
    def record(
        self,
        step: int,
        t: float,
        noise: torch.Tensor,
        latents_src: torch.Tensor,
        latents_tgt: torch.Tensor,
        noise_pred_src: torch.Tensor,
        noise_pred_tgt: torch.Tensor,
        z_edit: torch.Tensor,
        **extra,
    ):
        """Record state for a given step."""
        self._states[step] = {
            "t": t,
            "noise": noise.clone(),
            "latents_src": latents_src.clone(),
            "latents_tgt": latents_tgt.clone(),
            "noise_pred_src": noise_pred_src.clone(),
            "noise_pred_tgt": noise_pred_tgt.clone(),
            "z_edit": z_edit.clone(),
            **{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in extra.items()},
        }
        self._current_step = step
    
    def get(self, step: int, key: str) -> Optional[Any]:
        """Get a specific value from a step."""
        if step in self._states:
            return self._states[step].get(key)
        return None
    
    def get_prev(self, key: str) -> Optional[Any]:
        """Get value from previous step."""
        if self._current_step >= 0 and (self._current_step - 1) in self._states:
            return self.get(self._current_step - 1, key)
        return None
    
    @property
    def has_prev(self) -> bool:
        return self._current_step >= 0 and (self._current_step - 1) in self._states
    
    def __len__(self):
        return len(self._states)


    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

class FlowEditPipeline(BaseEditPlusPipeline):
    """
    FlowEdit pipeline for image editing using differential velocity fields.
    
    Inherits from QwenImageEditPlusPipeline and overrides __call__ with FlowEdit algorithm.
    """

    @torch.no_grad()
    def __call__(
        self,
        image_src: PipelineImageInput = None,
        image_tgt: PipelineImageInput = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
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
        n_min: int = 0,
        n_max: int = 20,
        noise_mode: Literal["random", "fixed", "velocity", "velocity_fixed"] = "random",
    ):
        """
        FlowEdit pipeline for image editing.
        
        Args:
            image_src: Source image to be edited.
            image_tgt: Target reference image for visual guidance.
            prompt: Text prompt for editing.
            negative_prompt: Negative prompt for CFG.
            true_cfg_scale_tgt: CFG scale for target branch.
            n_min, n_max: FlowEdit step range control.
            noise_mode: Noise generation strategy - "random" | "fixed" | "velocity" | "velocity_fixed".
        """
        # Calculate dimensions from source image
        image_size = image_src.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width,
            negative_prompt=negative_prompt,
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
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        
        # 3. Preprocess images (source and target)
        img_tgt = image_tgt
        tgt_width, tgt_height = img_tgt.size
        condition_width_tgt, condition_height_tgt = calculate_dimensions(
            CONDITION_IMAGE_SIZE, tgt_width / tgt_height
        )
        vae_width_tgt, vae_height_tgt = calculate_dimensions(VAE_IMAGE_SIZE, tgt_width / tgt_height)
        condition_image_tgt = self.image_processor.resize(img_tgt, condition_height_tgt, condition_width_tgt)
        vae_image_tgt = self.image_processor.preprocess(img_tgt, vae_height_tgt, vae_width_tgt).unsqueeze(2)

        # Process source image for editing
        img_src = image_src
        src_width, src_height = img_src.size
        vae_width_src, vae_height_src = calculate_dimensions(VAE_IMAGE_SIZE, src_width / src_height)
        vae_image_src = self.image_processor.preprocess(img_src, vae_height_src, vae_width_src).unsqueeze(2)

        # Store for later use
        vae_images = [vae_image_tgt, vae_image_src]  # [target, source]
        vae_image_sizes = [(vae_width_tgt, vae_height_tgt), (vae_width_src, vae_height_src)]
        cond_images_tgt = [condition_image_tgt]  # target image for prompt encoding

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale_tgt > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale_tgt is passed as {true_cfg_scale_tgt}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale_tgt <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale_tgt <= 1"
            )

        do_true_cfg_tgt = has_neg_prompt and true_cfg_scale_tgt > 1

        # Encode Target Prompt
        prompt_embeds_tgt, prompt_embeds_mask_tgt = self.encode_prompt(
            image=cond_images_tgt,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        txt_seq_lens_tgt = prompt_embeds_mask_tgt.sum(dim=1).tolist()

        if do_true_cfg_tgt:
            negative_prompt_embeds_tgt, negative_prompt_embeds_mask_tgt = self.encode_prompt(
                image=cond_images_tgt,
                prompt=negative_prompt,
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
        # Order: [target_latent (index 0), source_latent (index 1)]
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

        # Extract source latent for editing (index 1, clean latent, not noise)
        latent_tgt = all_latents_list[0]  # shape: [B, seq_len_tgt, C]
        x_src = all_latents_list[1].clone()  # shape: [B, seq_len_src, C]
        z_edit = x_src.clone()  # shape: [B, seq_len_src, C]

        # Helper to construct model inputs for target branch
        def get_latent_model_input_and_img_shapes_tgt(z_t):
            # Concat target latent as condition
            latent_model_input = torch.cat([z_t, latent_tgt], dim=1)  # shape: [B, seq_len_src + seq_len_tgt, C]
            
            # Construct img_shapes
            # First element is main generated image shape (source)
            main_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
            # Second element is target condition image shape
            vw_tgt, vh_tgt = vae_image_sizes[0]
            tgt_shape = (1, vh_tgt // self.vae_scale_factor // 2, vw_tgt // self.vae_scale_factor // 2)
            img_shapes = [main_shape, tgt_shape]
            
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

        # handle guidance
        # If model is guidance distilled, we pass this
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
        state_tracker = FlowEditStateTracker()
        state_tracker.init(x_src)

        self.scheduler.set_begin_index(0)
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
                t_prev = timesteps[i+1] / 1000.0 if i < len(timesteps) - 1 else torch.tensor(0.0, device=device, dtype=t.dtype)
                dt = t_prev - t_curr

                # Source Branch (Analytical)
                noise = state_tracker.get_noise(t_curr, mode=noise_mode)  # shape: [B, seq_len, C]
                latents_src = (1 - t_curr) * x_src + t_curr * noise  # shape: [B, seq_len, C]
                noise_pred_src = noise - x_src  # shape: [B, seq_len, C]

                # 2. Target Branch (Model Inference Required)
                latents_tgt = z_edit + latents_src - x_src  # shape: [B, seq_len, C]
                
                # Target Model Input (with target image as condition)
                latent_model_input_tgt, img_shapes_tgt = get_latent_model_input_and_img_shapes_tgt(latents_tgt)
                
                # broadcast timestep
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
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

                if do_true_cfg_tgt:
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
                
                # Cache cond prediction for norm-preserving CFG
                cond_noise_pred_tgt = noise_pred_tgt
                
                # Standard CFG combine
                noise_pred_tgt = neg_noise_pred_tgt + true_cfg_scale_tgt * (cond_noise_pred_tgt - neg_noise_pred_tgt)  # shape: [B, seq_len, C]
                
                # Match norm to cond branch (avoid over-/under-scaling)
                cond_norm = torch.norm(cond_noise_pred_tgt, dim=-1, keepdim=True)
                comb_norm = torch.norm(noise_pred_tgt, dim=-1, keepdim=True)
                noise_pred_tgt = noise_pred_tgt * (cond_norm / comb_norm)
                
                # Record state
                state_tracker.record(
                    step=i, t=t_curr, noise=noise,
                    latents_src=latents_src, latents_tgt=latents_tgt,
                    noise_pred_src=noise_pred_src, noise_pred_tgt=noise_pred_tgt,
                    z_edit=z_edit,
                )
                
                # Update z_edit
                v_delta = noise_pred_tgt - noise_pred_src  # shape: [B, seq_len, C]
                
                # 4. Update z_edit using Euler step
                z_edit = z_edit + dt * v_delta  # shape: [B, seq_len, C]

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
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
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

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, packed_latents)

        return FlowEditPipelineOutput(images=image, latents=packed_latents)

