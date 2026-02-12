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
双层蒸馏 Pipeline（VSD - Variational Score Distillation）。

核心思想：
    教师-学生双层优化。教师（基础模型，LoRA 关闭）提供"真实"分布的
    速度估计，学生（LoRA 开启）拟合当前"假"分布（渲染图）的速度场。
    两者对比产生 VSD 梯度，稳定训练、降低方差。

核心公式（Flow Matching 版本）:
    x0_teacher = z_t - t * v_teacher  # 教师 x0 预测（LoRA OFF）
    x0_student = z_t - t * v_student  # 学生 x0 预测（LoRA ON）

VSD Loss（外层，优化 3D 模型）:
    复用 CSD 体系：x0_pos = x0_teacher（吸引），x0_neg = x0_student（排斥）
    loss_vsd = csd_weight * (MSE(src, x0_teacher) - MSE(src, x0_student))

Student Loss（内层，优化 LoRA）:
    loss_student = lambda_sup * MSE(v_student, noise - clean_latents)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging as diffusers_logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from peft import LoraConfig, get_peft_model

from edit4shape.guidance.pipelines.qwen_image_edit.trackers import create_distillation_tracker, DistillationTracker
from edit4shape.guidance.pipelines.utils import DifferentiableVAEMixin, sample_timesteps_uniform


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = diffusers_logging.get_logger(__name__)  # pylint: disable=invalid-name
py_logger = logging.getLogger(__name__)


# =============================================================================
# Bilevel Distillation Output
# =============================================================================

@dataclass
class BilevelDistillationOutput:
    """
    双层蒸馏 Pipeline 输出。
    
    Attributes:
        tracker: StateTracker 实例（x0_pos=教师, x0_neg=学生）
        student_loss_context: 学生带梯度前向所需的上下文字典
    """
    tracker: DistillationTracker
    student_loss_context: Optional[Dict[str, Any]] = None


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


class QwenImageBilevelDistillationPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin, DifferentiableVAEMixin):
    r"""
    双层蒸馏 Pipeline（VSD）。
    
    教师（LoRA 关闭）和学生（LoRA 开启）双前向对比。
    外层 VSD Loss 复用 CSD 体系（x0_pos=教师, x0_neg=学生），
    内层 Student Loss 通过 compute_student_loss() 单独计算。

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

        # LoRA 状态标记
        self._lora_injected = False

    # =========================================================================
    # LoRA 注入与管理
    # =========================================================================

    def init_lora(
        self,
        lora_rank: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """
        向 transformer 注入 LoRA 适配器，冻结 base 参数，仅开放 LoRA 训练。
        
        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: LoRA 目标模块列表
        """
        if self._lora_injected:
            py_logger.warning("[BilevelPipeline] LoRA already injected, skipping.")
            return

        if target_modules is None:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

        # 冻结所有 base 参数
        for p in self.transformer.parameters():
            p.requires_grad = False

        # 注入 LoRA
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, config)  # PeftModel 包装

        # 仅开放 LoRA 参数
        for name, p in self.transformer.named_parameters():
            p.requires_grad = "lora_" in name

        # 启用 per-block 梯度检查点（降低学生前向的显存占用）
        # 注意：compute_student_loss 中使用 eval 模式（关闭 dropout），
        # 避免 use_reentrant=False 下 forward/recompute tensor 数量不匹配。
        base_model = self.transformer.get_base_model()
        if hasattr(base_model, "enable_gradient_checkpointing"):
            base_model.enable_gradient_checkpointing()

        self._lora_injected = True
        self._log_lora_stats()

    def get_lora_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """返回所有可训练的 LoRA 参数列表。"""
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """导出 LoRA 权重用于保存 checkpoint。"""
        return {
            name: p.data.clone()
            for name, p in self.transformer.named_parameters()
            if "lora_" in name
        }

    def _log_lora_stats(self) -> None:
        """打印 LoRA 参数统计。"""
        total = sum(p.numel() for p in self.transformer.parameters())  # scalar
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)  # scalar
        lora_count = sum(1 for n, _ in self.transformer.named_parameters() if "lora_" in n)  # scalar
        py_logger.info(
            f"[BilevelPipeline] LoRA injected: "
            f"total_params={total:,}, trainable={trainable:,}, lora_tensors={lora_count}"
        )

    def log_lora_param_ranges(self) -> Dict[str, float]:
        """统计 LoRA 参数的数值范围（权重/梯度），用于训练监控。"""
        value_abs_mean_sum = 0.0  # scalar
        value_count = 0  # scalar
        grad_abs_mean_sum = 0.0  # scalar
        grad_count = 0  # scalar

        for name, p in self.transformer.named_parameters():
            if "lora_" in name:
                value_abs_mean_sum += float(p.data.abs().mean().item())  # scalar
                value_count += 1  # scalar
                if p.grad is not None:
                    grad_abs_mean_sum += float(p.grad.abs().mean().item())  # scalar
                    grad_count += 1  # scalar

        return {
            "lora_value_abs_mean": value_abs_mean_sum / max(value_count, 1),  # scalar
            "lora_grad_abs_mean": grad_abs_mean_sum / max(grad_count, 1),  # scalar
            "lora_tensor_count": value_count,  # scalar
        }

    # =========================================================================
    # 学生带梯度前向
    # =========================================================================

    def compute_student_loss(
        self,
        context: Dict[str, Any],
        lambda_sup: float = 1.0,
    ) -> torch.Tensor:
        """
        学生带梯度前向，计算速度场监督 loss。
        
        必须在 torch.no_grad() 之外调用。
        显存优化依赖 init_lora() 中 enable_gradient_checkpointing()（per-block 级别）。
        
        Args:
            context: __call__ 返回的 student_loss_context 字典
            lambda_sup: 学生监督 loss 权重
        
        Returns:
            loss_student: 标量 loss（float32），梯度流向 LoRA 参数
        """
        latent_model_input = context["latent_model_input"]  # (B, seq_all, C*4)
        t = context["t"]  # (B,)
        prompt_embeds = context["prompt_embeds"]  # (B, T, D)
        prompt_embeds_mask = context["prompt_embeds_mask"]  # (B, T)
        guidance = context["guidance"]  # None
        img_shapes = context["img_shapes"]  # list
        attention_kwargs = context["attention_kwargs"]  # dict
        noise = context["noise"]  # (B, seq, C*4)
        clean_latents = context["clean_latents"]  # (B, seq, C*4)
        seq_len = context["seq_len"]  # int
        dtype = context["dtype"]  # torch.dtype

        # 保持 eval 模式（关闭 dropout），确保 gradient_checkpointing
        # 的 forward/recompute 确定性一致。LoRA 参数仍有 requires_grad=True，梯度照常流动。
        self.transformer.eval()

        # 前向（显存优化由模型内部 per-block gradient_checkpointing 处理）
        v_pred_student = self.transformer(
            hidden_states=latent_model_input.detach().clone(),  # (B, seq_all, C*4)
            timestep=t.to(dtype),  # (B,)
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,  # (B, T, D)
            encoder_hidden_states_mask=prompt_embeds_mask,  # (B, T)
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]  # (B, seq_all, C*4)
        v_pred_student = v_pred_student[:, :seq_len]  # (B, seq, C*4)

        # Flow Matching 目标：v_target = noise - clean_latents（线性路径常速度场）
        v_target = (noise - clean_latents).to(v_pred_student.dtype)  # (B, seq, C*4)

        # 学生监督 loss
        loss_student = lambda_sup * F.mse_loss(v_pred_student, v_target)  # scalar

        # 确保保持 eval 模式（此处为冗余保护）
        self.transformer.eval()

        return loss_student.to(torch.float32)  # scalar, float32

    # =========================================================================
    # Prompt 编码（与 Distillation Pipeline 完全一致）
    # =========================================================================

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

    # =========================================================================
    # Latent 操作（与 Distillation Pipeline 完全一致）
    # =========================================================================

    @staticmethod
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
    def interrupt(self):
        return self._interrupt

    # =========================================================================
    # 主入口：教师/学生双前向（无梯度）
    # =========================================================================

    @torch.no_grad()
    def __call__(
        self,
        # 条件图像列表（[rendered, condition]）
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
    ) -> BilevelDistillationOutput:
        r"""
        VSD 双层蒸馏：教师（LoRA OFF, CFG）+ 学生（LoRA ON, CFG=1）无梯度前向。
        
        对每个时间步（共 3 次 Transformer 前向）：
            1. 教师 cond 前向（LoRA 关闭）
            2. 教师 uncond 前向（LoRA 关闭）→ CFG 合成 → x0_teacher
            3. 学生 cond 前向（LoRA 开启，无梯度，CFG=1）→ x0_student
            4. 记录到 Tracker：x0_pos=x0_teacher, x0_neg=x0_student
        
        学生带梯度前向通过 compute_student_loss() 单独调用。
        
        Returns:
            BilevelDistillationOutput:
                - tracker: 包含 x0_pos(teacher)、x0_neg(student) 的 Tracker
                - student_loss_context: 学生带梯度前向所需的上下文
        """
        if src_latent is None:
            raise ValueError("`src_latent` is required.")
        if not self._lora_injected:
            raise RuntimeError("LoRA not injected. Call init_lora() before __call__().")

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
            prompt, height, width,
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

        # 5. CFG 设置
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        assert has_neg_prompt, "negative_prompt or negative_prompt_embeds must be provided"

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

        # 8. Encode negative prompt
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            image=prompt_cond_images,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        # 9. Prepare latent variables
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

        # 10. 构建 img_shapes
        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 11. 准备 src_latent
        clean_latents = src_latent.to(device=device, dtype=dtype)  # (B, seq, C*4)

        # 12. Guidance embedding
        guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # =====================================================================
        # VSD 核心计算（教师/学生双前向）
        # =====================================================================

        # 13. 采样时间步
        num_train_timesteps = 1000
        min_step = int(num_train_timesteps * min_step_percent)  # scalar
        max_step = int(num_train_timesteps * max_step_percent)  # scalar

        timesteps_list = sample_timesteps_uniform(
            min_step=min_step,
            max_step=max_step,
            num_steps=num_timesteps,
            batch_size=batch_size,
            device=device,
            generator=generator,
            ascending=True,
        )  # List[Tensor(B,)]

        # 14. 创建 Tracker 并初始化噪声
        tracker = create_distillation_tracker(noise_mode, height=height, width=width)
        seed = generator.initial_seed() if generator is not None else None
        tracker.init(clean_latents, mode=noise_mode, seed=seed)

        # =====================================================================
        # 15. 对每个时间步计算教师/学生 x0 预测
        # =====================================================================
        student_loss_context = None  # 最后一步的上下文

        t_prev_scalar = 0.0  # 追踪上一步时间，用于计算 dt
        for t_step in timesteps_list:
            t = t_step.float() / num_train_timesteps  # (B,) 归一化到 [0, 1]
            t_scalar = t[0].item()  # scalar
            dt_scalar = t_scalar - t_prev_scalar  # 当前步的时间差

            # 获取噪声并手动加噪
            noise = tracker.get_noise(clean_latents)  # (B, seq, C*4)
            latents_noisy = (1.0 - t_scalar) * clean_latents + t_scalar * noise  # (B, seq, C*4)

            # 构建 latent_model_input
            latent_model_input = latents_noisy  # (B, seq, C*4)
            if image_latents is not None:
                latent_model_input = torch.cat([latents_noisy, image_latents], dim=1)  # (B, seq_all, C*4)
            latent_model_input = latent_model_input.to(dtype)  # (B, seq_all, C*4)

            # === 教师前向（LoRA 关闭） ===
            self.transformer.eval()
            with self.transformer.disable_adapter():
                with self.transformer.cache_context("teacher_cond"):
                    v_cond_teacher = self.transformer(
                        hidden_states=latent_model_input,  # (B, seq_all, C*4)
                        timestep=t.to(dtype),  # (B,)
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,  # (B, T, D)
                        encoder_hidden_states_mask=prompt_embeds_mask,  # (B, T)
                        img_shapes=img_shapes,
                        attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]  # (B, seq_all, C*4)
                v_cond_teacher = v_cond_teacher[:, :clean_latents.size(1)]  # (B, seq, C*4)

                with self.transformer.cache_context("teacher_uncond"):
                    v_uncond_teacher = self.transformer(
                        hidden_states=latent_model_input,  # (B, seq_all, C*4)
                        timestep=t.to(dtype),  # (B,)
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,  # (B, T, D)
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,  # (B, T)
                        img_shapes=img_shapes,
                        attention_kwargs=self._attention_kwargs,
                        return_dict=False,
                    )[0]  # (B, seq_all, C*4)
                v_uncond_teacher = v_uncond_teacher[:, :clean_latents.size(1)]  # (B, seq, C*4)

            # 教师 CFG（L2 norm rescale）
            comb_teacher = v_uncond_teacher + true_cfg_scale * (v_cond_teacher - v_uncond_teacher)  # (B, seq, C*4)
            cond_norm_t = torch.norm(v_cond_teacher, dim=-1, keepdim=True)  # (B, seq, 1)
            comb_norm_t = torch.norm(comb_teacher, dim=-1, keepdim=True)  # (B, seq, 1)
            v_cfg_teacher = comb_teacher * (cond_norm_t / (comb_norm_t + 1e-8))  # (B, seq, C*4)
            x0_teacher = latents_noisy - t_scalar * v_cfg_teacher  # (B, seq, C*4)

            # === 学生前向（LoRA 开启，无梯度，CFG=1 仅 cond） ===
            self.transformer.eval()
            with self.transformer.cache_context("student_cond"):
                v_cond_student = self.transformer(
                    hidden_states=latent_model_input,  # (B, seq_all, C*4)
                    timestep=t.to(dtype),  # (B,)
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,  # (B, T, D)
                    encoder_hidden_states_mask=prompt_embeds_mask,  # (B, T)
                    img_shapes=img_shapes,
                    attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]  # (B, seq_all, C*4)
            v_cond_student = v_cond_student[:, :clean_latents.size(1)]  # (B, seq, C*4)

            # 学生 CFG=1，直接用 cond 预测（无需 uncond 前向）
            x0_student = latents_noisy - t_scalar * v_cond_student  # (B, seq, C*4)

            # === VSD 映射到 CSD 体系：pos=教师（吸引），neg=学生（排斥） ===
            tracker.record(
                x0_pred=x0_teacher,   # (B, seq, C*4) MSE 目标
                t=t_scalar,
                x0_pos=x0_teacher,    # (B, seq, C*4) 吸引（教师）
                x0_neg=x0_student,    # (B, seq, C*4) 排斥（学生）
            )

            # 更新噪声（使用教师的速度，aligned / inversion 模式下生效）
            tracker.update(v_cond_teacher, v_uncond_teacher, v_cfg_teacher, t_scalar, dt_scalar)
            t_prev_scalar = t_scalar

            # 保存当前步上下文（每步覆盖，最终保留最后一步）
            student_loss_context = {
                "latent_model_input": latent_model_input.detach().clone(),  # (B, seq_all, C*4)
                "t": t.clone(),  # (B,)
                "prompt_embeds": prompt_embeds,  # (B, T, D)
                "prompt_embeds_mask": prompt_embeds_mask,  # (B, T)
                "guidance": guidance,  # None
                "img_shapes": img_shapes,  # list
                "attention_kwargs": self._attention_kwargs,  # dict
                "noise": noise.detach().clone(),  # (B, seq, C*4)
                "clean_latents": clean_latents.detach().clone(),  # (B, seq, C*4)
                "seq_len": clean_latents.size(1),  # int
                "dtype": dtype,  # torch.dtype
            }

        # =====================================================================
        # 16. 返回
        # =====================================================================
        return BilevelDistillationOutput(
            tracker=tracker,
            student_loss_context=student_loss_context,
        )
