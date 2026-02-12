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
单步蒸馏 Pipeline（CSD）。

核心公式（Flow Matching 版本）:
    x0 = z_t - t * v_pred  # x0 预测
    
CSD Loss:
    MSE(src, x0_pos) - MSE(src, x0_neg)  # 吸引高 CFG，排斥低 CFG
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from edit4shape.guidance.pipelines.qwen_image_edit.trackers import create_distillation_tracker, DistillationTracker
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin, sample_timesteps_uniform


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# =============================================================================
# Distillation Output
# =============================================================================

@dataclass
class DistillationOutput:
    """
    蒸馏 Pipeline 输出。
    
    Attributes:
        tracker: StateTracker 或 InversionStateTracker 实例
    """
    tracker: DistillationTracker


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class QwenImageDistillationPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin, DifferentiableVAEMixin):
    r"""
    单步蒸馏 Pipeline（CSD）。
    
    计算高/低 CFG 的 x0 预测，用于 CSD Loss。
    Loss 类型由调用方通过 Tracker.loss(mse_weight, csd_weight) 控制。

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
        tokenizer (`QwenTokenizer`):
            Tokenizer for the text encoder.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        # QwenImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024

        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = self._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        # 条件图像列表（与 FlowEdit 一致：[rendered, condition]）
        image: Optional[PipelineImageInput] = None,
        # Prompt
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        # CFG
        true_cfg_scale: float = 4.0,
        # 尺寸
        height: Optional[int] = None,
        width: Optional[int] = None,
        # 蒸馏参数
        src_latent: Optional[torch.Tensor] = None,  # 渲染图的 packed latent [B, seq, C]
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
        num_timesteps: int = 1,  # 采样时间步数量（MTS）
        noise_mode: str = "fixed",  # 噪声模式: random | fixed | aligned | inversion_*
        # 其他
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ) -> DistillationOutput:
        r"""
        计算 CSD 蒸馏梯度（单步）。
        
        Flow Matching x0 预测公式:
            x0 = z_t - t * v_pred

        Args:
            image: 图像列表（[rendered, condition]）
            prompt: 目标 prompt
            negative_prompt: 负面 prompt（用于 CFG 和低 CFG 分支）
            true_cfg_scale: CFG 强度
            height, width: 图像尺寸（可选，自动从 image 推断）
            src_latent: 渲染图的 packed latent [B, seq, C]，外部可微分编码
            min_step_percent, max_step_percent: 时间步采样范围 [0, 1]
            num_timesteps: 采样时间步数量（MTS）
            noise_mode: 噪声模式
                - random: 每次随机噪声
                - fixed: 固定噪声
                - aligned: DNAEdit 风格累积补偿
                - inversion_cond: Naive Inversion（用 v_cond）
                - inversion_uncond: Naive Inversion（用 v_uncond）
                - inversion_cfg: Naive Inversion（用 v_cfg）
            generator: 随机数生成器
            prompt_embeds, prompt_embeds_mask: 预计算的 prompt embeddings
            negative_prompt_embeds, negative_prompt_embeds_mask: 预计算的 negative prompt embeddings
            attention_kwargs: transformer attention 参数
            max_sequence_length: prompt 最大长度

        Returns:
            DistillationOutput:
                - tracker: DistillationStateTracker 实例，包含 x0_pos、x0_neg 等状态
        """
        if src_latent is None:
            raise ValueError("`src_latent` is required. Please provide the packed latent of the rendered image.")

        # 1. 处理输入尺寸
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 2. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length=max_sequence_length,
        )

        self._attention_kwargs = attention_kwargs

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        # 4. Preprocess image（与原始 Pipeline 完全一致）
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

        # 5. CFG 设置（CSD 模式必须提供 negative_prompt）
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        assert has_neg_prompt, "negative_prompt or negative_prompt_embeds must be provided for CSD"

        # 6. 选择条件图用于 prompt 编码（固定使用 index=1）
        prompt_cond_images = [condition_images[1]]

        # 7. Encode prompt（条件 prompt = 图 + 文）
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_cond_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        # 8. Encode negative prompt（如果需要）
        if True:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_cond_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        # 8. Prepare latent variables（与原始 Pipeline 一致）
        num_channels_latents = self.transformer.config.in_channels // 4
        _, image_latents = self.prepare_latents(
            vae_images,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )
        
        # 9. 构建 img_shapes（与原始 Pipeline 一致，包含条件图）
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 10. 准备 src_latent（渲染图的 clean latent）
        # src_latent 应为 packed 格式 [B, seq, C*4]
        clean_latents = src_latent.to(device=device, dtype=dtype)  # (B, seq, C*4)

        # 11. Handle guidance embedding
        # Qwen-Image-Edit 不是 guidance-distilled 模型，直接设为 None
        guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # =====================================================================
        # 蒸馏核心计算（支持多时间步 MTS + 噪声对齐）
        # =====================================================================

        # 12. 采样时间步
        num_train_timesteps = 1000
        min_step = int(num_train_timesteps * min_step_percent)
        max_step = int(num_train_timesteps * max_step_percent)

        timesteps_list = sample_timesteps_uniform(
            min_step=min_step,
            max_step=max_step,
            num_steps=num_timesteps,
            batch_size=batch_size,
            device=device,
            generator=generator,
            ascending=True,  # 从小到大采样
        )  # List[Tensor(B,)]

        # 13. 创建 Tracker 并初始化噪声（工厂函数根据 noise_mode 选择）
        tracker = create_distillation_tracker(noise_mode, height=height, width=width)
        seed = generator.initial_seed() if generator is not None else None
        tracker.init(clean_latents, mode=noise_mode, seed=seed)

        # =====================================================================
        # 14. 对每个时间步计算 x0 预测
        # =====================================================================
        t_prev_scalar = 0.0  # 追踪上一步时间，用于计算 dt
        for t_step in timesteps_list:
            t = t_step.float() / num_train_timesteps  # (B,) 归一化到 [0, 1]
            t_scalar = t[0].item()  # 标量版本
            dt_scalar = t_scalar - t_prev_scalar  # 当前步的时间差
            
            # 获取噪声并手动加噪（与 FlowEdit 一致）
            noise = tracker.get_noise(clean_latents)  # (B, seq, C*4)
            latents_noisy = (1.0 - t_scalar) * clean_latents + t_scalar * noise  # (B, seq, C*4)

            # 构建 latent_model_input
            latent_model_input = latents_noisy
            if image_latents is not None:
                latent_model_input = torch.cat([latents_noisy, image_latents], dim=1)
            latent_model_input = latent_model_input.to(dtype)

            # Transformer 前向（条件分支）
            with self.transformer.cache_context("cond"):
                v_cond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.to(dtype),
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]
            v_cond = v_cond[:, :clean_latents.size(1)]  # (B, seq, C*4)

            # Transformer 前向（无条件分支，如果需要）
            v_uncond = None
            if True:
                with self.transformer.cache_context("uncond"):
                    v_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t.to(dtype),
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]
                v_uncond = v_uncond[:, :clean_latents.size(1)]  # (B, seq, C*4)

            # 计算 v_cfg（CFG with L2 norm rescale）
            comb_pred = v_uncond + true_cfg_scale * (v_cond - v_uncond)  # (B, seq, C*4)
            cond_norm = torch.norm(v_cond, dim=-1, keepdim=True)  # (B, seq, 1)
            comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)  # (B, seq, 1)
            v_cfg = comb_pred * (cond_norm / (comb_norm + 1e-8))  # (B, seq, C*4)
            
            # 计算 x0（Flow Matching 公式: x0 = z_t - t * v）
            x0_pred = latents_noisy - t_scalar * v_cfg    # (B, seq, C*4) MSE 目标
            x0_pos = latents_noisy - t_scalar * v_cond    # (B, seq, C*4) CSD 正样本（吸引）
            x0_neg = latents_noisy - t_scalar * v_uncond  # (B, seq, C*4) CSD 负样本（排斥）
            
            # 记录状态
            tracker.record(x0_pred, t_scalar, x0_pos, x0_neg)
            
            # 更新噪声（aligned / inversion_* 模式下生效）
            tracker.update(v_cond, v_uncond, v_cfg, t_scalar, dt_scalar)
            t_prev_scalar = t_scalar
        
        # =====================================================================
        # 15. 返回
        # =====================================================================
        return DistillationOutput(tracker=tracker)
