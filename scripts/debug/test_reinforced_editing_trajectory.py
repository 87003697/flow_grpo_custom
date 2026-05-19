#!/usr/bin/env python3
"""
Reinforced Editing Trajectory 可视化对比脚本。

对比 FlowEdit 中有/无 negative source guidance 的中间步编辑轨迹：
  - With negative source guidance（标准 Reinforced Editing）:
      v_delta = v_cfg_tgt - v_cfg_src
  - Without negative source guidance（ablation baseline）:
      v_delta = v_cfg_tgt

两种设置共享相同的 seed / noise / guidance scale / steps / prompt embedding，
唯一差异是 v_delta 计算中是否减去 v_cfg_src。

输出：
  - 每种设置各 step 的 decoded 中间图像
  - 一张 grid 对比图（2 行 x N 列）

模式 A（手动指定图像）：
  python scripts/debug/test_reinforced_editing_trajectory.py \
    --condition_image dataset/alphaimages_v3/test/test_00.png \
    --rendered_image outputs/renders/000_view0.png \
    --output_dir outputs/reinforced_editing_trajectory

模式 B（TRELLIS 自动渲染 source view）：
  python scripts/debug/test_reinforced_editing_trajectory.py \
    --condition_image dataset/alphaimages_v3/test/test_00.png \
    --trellis_model pretrained_weights/TRELLIS-image-large \
    --output_dir outputs/reinforced_editing_trajectory
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRELLIS_ROOT = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS")
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)


# =========================================================================
# FlowEdit 核心循环（支持 ablation 开关 + 中间步保存）
# =========================================================================

def run_flowedit_trajectory(
    pipe,
    # 预计算的 latent / embeddings（两次运行共享）
    x_src: torch.Tensor,
    all_latents_list: List[torch.Tensor],
    prompt_embeds_src: torch.Tensor,
    prompt_embeds_mask_src: torch.Tensor,
    negative_prompt_embeds_src: torch.Tensor,
    negative_prompt_embeds_mask_src: torch.Tensor,
    prompt_embeds_tgt: torch.Tensor,
    prompt_embeds_mask_tgt: torch.Tensor,
    negative_prompt_embeds_tgt: torch.Tensor,
    negative_prompt_embeds_mask_tgt: torch.Tensor,
    # FlowEdit 参数
    timesteps: torch.Tensor,
    num_inference_steps: int,
    n_max: int,
    true_cfg_scale_src: float,
    true_cfg_scale_tgt: float,
    guidance: Optional[torch.Tensor],
    noise_init: torch.Tensor,
    height: int,
    width: int,
    vae_image_sizes: List[Tuple[int, int]],
    # prompt 相等性标记（控制是否跳过 uncond 推理）
    src_neg_same: bool,
    tgt_neg_same: bool,
    # Ablation 开关
    use_neg_src_guidance: bool = True,
    noise_mode: str = "aligned",
    # 中间步保存
    save_active_steps: Optional[Set[int]] = None,
    label: str = "",
) -> Tuple[torch.Tensor, Dict[int, Image.Image]]:
    """
    运行单次 FlowEdit 轨迹，返回最终 z_edit 和中间步 decoded 图像。

    Args:
        save_active_steps: 在这些 active step index 处 decode z_edit。
                           active step 从 0 开始计数（跳过的步不计）。
        use_neg_src_guidance: True = v_delta = v_cfg_tgt - v_cfg_src;
                              False = v_delta = v_cfg_tgt.
    Returns:
        (z_edit, step_images) — step_images[active_step_idx] = PIL.Image
    """
    device = pipe._execution_device
    batch_size = x_src.shape[0]

    if save_active_steps is None:
        save_active_steps = set()

    z_edit = x_src.clone()  # [B, seq_len, C]
    noise = noise_init.clone()  # [B, seq_len, C]

    def get_latent_model_input_and_img_shapes(z_t):
        cond_latent = all_latents_list[1]
        latent_model_input = torch.cat([z_t, cond_latent], dim=1)
        main_shape = (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)
        vw, vh = vae_image_sizes[1]
        cond_shape = (1, vh // pipe.vae_scale_factor // 2, vw // pipe.vae_scale_factor // 2)
        img_shapes = [main_shape, cond_shape]
        return latent_model_input, [img_shapes] * batch_size

    step_images: Dict[int, Image.Image] = {}
    active_step_idx = 0
    n_active = min(n_max, num_inference_steps)
    desc = f"FlowEdit ({'w/ neg src' if use_neg_src_guidance else 'w/o neg src'})"

    for i, t in enumerate(tqdm(timesteps, desc=desc, leave=False)):
        if num_inference_steps - i > n_max:
            continue

        t_curr = t / 1000.0  # []
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1] / 1000.0  # []
        else:
            t_prev = torch.tensor(0.0, device=device, dtype=t.dtype)  # []
        dt = t_prev - t_curr  # []
        timestep = t.expand(batch_size).to(torch.bfloat16)  # [B]

        # ========== Source Branch ==========
        latents_src = (1 - t_curr) * x_src + t_curr * noise  # [B, seq_len, C]
        latent_model_input_src, img_shapes_src = get_latent_model_input_and_img_shapes(latents_src)

        with pipe.transformer.cache_context("cond"):
            v_cond_src = pipe.transformer(
                hidden_states=latent_model_input_src,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask_src,
                encoder_hidden_states=prompt_embeds_src,
                img_shapes=img_shapes_src,
                attention_kwargs={},
                return_dict=False,
            )[0]
            v_cond_src = v_cond_src[:, :x_src.shape[1]]  # [B, seq_len, C]

        if not src_neg_same:
            with pipe.transformer.cache_context("uncond"):
                v_uncond_src = pipe.transformer(
                    hidden_states=latent_model_input_src,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask_src,
                    encoder_hidden_states=negative_prompt_embeds_src,
                    img_shapes=img_shapes_src,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                v_uncond_src = v_uncond_src[:, :x_src.shape[1]]  # [B, seq_len, C]

            comb_pred_src = v_uncond_src + true_cfg_scale_src * (v_cond_src - v_uncond_src)  # [B, seq_len, C]
            cond_norm_src = torch.norm(v_cond_src, dim=-1, keepdim=True)  # [B, seq_len, 1]
            cfg_norm_src = torch.norm(comb_pred_src, dim=-1, keepdim=True)  # [B, seq_len, 1]
            v_cfg_src = comb_pred_src * (cond_norm_src / (cfg_norm_src + 1e-8))  # [B, seq_len, C]
        else:
            v_uncond_src = v_cond_src
            v_cfg_src = v_cond_src

        # ========== Target Branch ==========
        latents_tgt = z_edit + latents_src - x_src  # [B, seq_len, C]
        latent_model_input_tgt, img_shapes_tgt = get_latent_model_input_and_img_shapes(latents_tgt)

        with pipe.transformer.cache_context("cond"):
            v_cond_tgt = pipe.transformer(
                hidden_states=latent_model_input_tgt,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask_tgt,
                encoder_hidden_states=prompt_embeds_tgt,
                img_shapes=img_shapes_tgt,
                attention_kwargs={},
                return_dict=False,
            )[0]
            v_cond_tgt = v_cond_tgt[:, :x_src.shape[1]]  # [B, seq_len, C]

        if not tgt_neg_same:
            with pipe.transformer.cache_context("uncond"):
                v_uncond_tgt = pipe.transformer(
                    hidden_states=latent_model_input_tgt,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask_tgt,
                    encoder_hidden_states=negative_prompt_embeds_tgt,
                    img_shapes=img_shapes_tgt,
                    attention_kwargs={},
                    return_dict=False,
                )[0]
                v_uncond_tgt = v_uncond_tgt[:, :x_src.shape[1]]  # [B, seq_len, C]

            comb_pred_tgt = v_uncond_tgt + true_cfg_scale_tgt * (v_cond_tgt - v_uncond_tgt)  # [B, seq_len, C]
            cond_norm_tgt = torch.norm(v_cond_tgt, dim=-1, keepdim=True)  # [B, seq_len, 1]
            cfg_norm_tgt = torch.norm(comb_pred_tgt, dim=-1, keepdim=True)  # [B, seq_len, 1]
            v_cfg_tgt = comb_pred_tgt * (cond_norm_tgt / (cfg_norm_tgt + 1e-8))  # [B, seq_len, C]
        else:
            v_uncond_tgt = v_cond_tgt
            v_cfg_tgt = v_cond_tgt

        # ========== Euler Step (ablation point) ==========
        if use_neg_src_guidance:
            v_delta = v_cfg_tgt - v_cfg_src  # [B, seq_len, C]
        else:
            v_delta = v_cfg_tgt  # [B, seq_len, C]

        z_edit = z_edit + dt * v_delta  # [B, seq_len, C]

        # ========== Noise Update ==========
        if noise_mode == "aligned":
            noise = noise - (v_cond_tgt - v_uncond_tgt) * (1.0 - float(t_curr))  # [B, seq_len, C]
        elif noise_mode == "random":
            step_gen = torch.Generator(device=device).manual_seed(42 + i)
            noise = torch.randn(noise.shape, generator=step_gen, device=device, dtype=noise.dtype)  # [B, seq_len, C]

        # ========== 保存中间步 ==========
        is_final = (i == len(timesteps) - 1)
        if active_step_idx in save_active_steps or is_final:
            key = active_step_idx if not is_final else -1
            with torch.no_grad():
                imgs = pipe._decode_latent_to_image(z_edit.clone(), height, width, "pil")
            step_images[active_step_idx] = imgs[0]
            if is_final and active_step_idx not in save_active_steps:
                step_images[-1] = imgs[0]

        active_step_idx += 1

    return z_edit, step_images


# =========================================================================
# Pipeline Setup（共享，只跑一次）
# =========================================================================

def setup_pipeline(pipe, rendered_pil, condition_pil, args, device):
    """
    复制 FlowEditFullPipeline.__call__ 中的 setup 部分，
    返回两次 trajectory 运行所需的所有预计算张量。
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions,
        retrieve_timesteps,
    )

    image = [rendered_pil, condition_pil]
    height = args.height
    width = args.width

    multiple_of = pipe.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    target_prompt = args.prompt
    source_prompt = args.prompt
    negative_prompt_src = args.negative_prompt
    negative_prompt_tgt = args.negative_prompt
    true_cfg_scale_src = -args.cfg_scale
    true_cfg_scale_tgt = args.cfg_scale
    guidance_scale = args.guidance_scale

    batch_size = 1
    num_images_per_prompt = 1
    max_sequence_length = 512

    CONDITION_IMAGE_SIZE = 384 * 384
    VAE_IMAGE_SIZE = 1024 * 1024

    condition_images = []
    vae_image_sizes = []
    vae_images = []
    for img in image:
        image_width, image_height = img.size
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
        vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
        vae_image_sizes.append((vw, vh))
        condition_images.append(pipe.image_processor.resize(img, ch, cw))
        vae_images.append(pipe.image_processor.preprocess(img, vh, vw).unsqueeze(2))

    cond_images = [condition_images[1]]

    src_neg_same = (source_prompt == negative_prompt_src)
    tgt_neg_same = (target_prompt == negative_prompt_tgt)

    # Encode prompts
    prompt_embeds_src, prompt_embeds_mask_src = pipe.encode_prompt(
        image=cond_images, prompt=source_prompt,
        device=device, num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if src_neg_same:
        negative_prompt_embeds_src = prompt_embeds_src
        negative_prompt_embeds_mask_src = prompt_embeds_mask_src
    else:
        negative_prompt_embeds_src, negative_prompt_embeds_mask_src = pipe.encode_prompt(
            image=cond_images, prompt=negative_prompt_src,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    prompt_embeds_tgt, prompt_embeds_mask_tgt = pipe.encode_prompt(
        image=cond_images, prompt=target_prompt,
        device=device, num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if tgt_neg_same:
        negative_prompt_embeds_tgt = prompt_embeds_tgt
        negative_prompt_embeds_mask_tgt = prompt_embeds_mask_tgt
    else:
        negative_prompt_embeds_tgt, negative_prompt_embeds_mask_tgt = pipe.encode_prompt(
            image=cond_images, prompt=negative_prompt_tgt,
            device=device, num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # Prepare latents
    num_channels_latents = pipe.transformer.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(args.seed)

    latents, image_latents = pipe.prepare_latents(
        vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height, width,
        prompt_embeds_src.dtype,
        device, generator, None,
    )

    all_latents_list = []
    current_idx = 0
    for (vw, vh) in vae_image_sizes:
        h_lat = vh // (pipe.vae_scale_factor * 2)
        w_lat = vw // (pipe.vae_scale_factor * 2)
        seq_len = h_lat * w_lat
        img_latent = image_latents[:, current_idx: current_idx + seq_len, :]  # [B, seq_len, C]
        all_latents_list.append(img_latent)
        current_idx += seq_len

    x_src = all_latents_list[0].clone()  # [B, seq_len, C]

    # Noise（用固定 seed 生成，两次运行共享）
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    noise_init = torch.randn(
        x_src.shape, generator=noise_gen, device=device, dtype=x_src.dtype,
    )  # [B, seq_len, C]

    # Timesteps
    sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
    image_seq_len = x_src.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, args.num_inference_steps, device, sigmas=sigmas, mu=mu,
    )

    # Guidance
    guidance = None
    if pipe.transformer.config.guidance_embeds:
        if guidance_scale is None:
            raise ValueError("guidance_scale required for guidance-distilled model")
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)  # [1]
        guidance = guidance.expand(latents.shape[0])  # [B]

    return {
        "x_src": x_src,
        "all_latents_list": all_latents_list,
        "noise_init": noise_init,
        "timesteps": timesteps,
        "num_inference_steps": num_inference_steps,
        "guidance": guidance,
        "height": height,
        "width": width,
        "vae_image_sizes": vae_image_sizes,
        "true_cfg_scale_src": true_cfg_scale_src,
        "true_cfg_scale_tgt": true_cfg_scale_tgt,
        "prompt_embeds_src": prompt_embeds_src,
        "prompt_embeds_mask_src": prompt_embeds_mask_src,
        "negative_prompt_embeds_src": negative_prompt_embeds_src,
        "negative_prompt_embeds_mask_src": negative_prompt_embeds_mask_src,
        "prompt_embeds_tgt": prompt_embeds_tgt,
        "prompt_embeds_mask_tgt": prompt_embeds_mask_tgt,
        "negative_prompt_embeds_tgt": negative_prompt_embeds_tgt,
        "negative_prompt_embeds_mask_tgt": negative_prompt_embeds_mask_tgt,
        "src_neg_same": src_neg_same,
        "tgt_neg_same": tgt_neg_same,
    }


# =========================================================================
# Grid 对比图生成
# =========================================================================

def add_label(img: Image.Image, text: str, font_size: int = 16) -> Image.Image:
    """在图像左上角添加文字标签。"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 4
    draw.rectangle([0, 0, tw + 2 * padding, th + 2 * padding], fill=(0, 0, 0, 200))
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return img


def _render_latex_label(tex: str, fontsize: int = 18, color: str = "white",
                        bg_color: str = "#282828") -> Image.Image:
    """用 matplotlib 渲染一段 LaTeX 公式/文字，返回紧凑 PIL Image。"""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.1, 0.1), dpi=150)
    ax.set_axis_off()
    fig.patch.set_facecolor(bg_color)
    text_obj = ax.text(
        0.5, 0.5, tex, fontsize=fontsize, color=color,
        ha="center", va="center", transform=ax.transAxes,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text_obj.get_window_extent(renderer=renderer)
    pad = 6
    bbox = bbox.expanded(1.0 + pad / bbox.width, 1.0 + pad / bbox.height)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches=bbox_inches, facecolor=bg_color, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_comparison_grid(
    source_render: Image.Image,
    reference_image: Image.Image,
    steps_without: Dict[int, Image.Image],
    steps_with: Dict[int, Image.Image],
    save_step_keys: List[int],
    cell_size: int = 512,
) -> Image.Image:
    """
    构建对比图。

    布局：
      - 左侧 row label 列 (LaTeX)
      - source / reference 各 1 列，跨两行居中贴一次
      - step 列：2 行 x N 列（w/o 和 w/ neg src guidance）
      - 顶部 column header (LaTeX)
    """
    BG = (255, 255, 255)
    BG_HEX = "#FFFFFF"

    def resize(img):
        return img.resize((cell_size, cell_size), Image.LANCZOS)

    n_step_cols = len(save_step_keys) + 1
    n_shared_cols = 2
    n_cols = n_shared_cols + n_step_cols
    n_rows = 2

    header_h = 40
    row_label_w = 260
    grid_w = row_label_w + n_cols * cell_size
    grid_h = header_h + n_rows * cell_size

    grid = Image.new("RGB", (grid_w, grid_h), BG)

    # ---- Column header labels (LaTeX) ----
    col_tex = [r"$x^{\rm src}$", r"$x^{\rm ref}$"]
    for k in save_step_keys:
        col_tex.append(rf"$t = {k}$")
    col_tex.append(r"$t_{\rm final}$")

    for c, tex in enumerate(col_tex):
        label_img = _render_latex_label(tex, fontsize=16, color="black", bg_color=BG_HEX)
        lw, lh = label_img.size
        cx = row_label_w + c * cell_size + cell_size // 2
        cy = header_h // 2
        grid.paste(label_img, (cx - lw // 2, cy - lh // 2))

    # ---- Row labels (two-line / three-line, large font) ----
    row_tex = [
        (r"$\bf{w/o}$", "neg src guidance", None),
        (r"$\bf{w/}$", "neg src guidance", "(Ours)"),
    ]
    for r, (line1, line2, line3) in enumerate(row_tex):
        img1 = _render_latex_label(line1, fontsize=16, color="black", bg_color=BG_HEX)
        img2 = _render_latex_label(line2, fontsize=12, color="black", bg_color=BG_HEX)
        imgs = [img1, img2]
        if line3 is not None:
            img3 = _render_latex_label(line3, fontsize=12, color="black", bg_color=BG_HEX)
            imgs.append(img3)
        gap = 4
        total_h = sum(im.size[1] for im in imgs) + gap * (len(imgs) - 1)
        cy = header_h + r * cell_size + cell_size // 2
        y = cy - total_h // 2
        for im in imgs:
            grid.paste(im, (row_label_w // 2 - im.size[0] // 2, y))
            y += im.size[1] + gap

    def paste_cell(img, row, col):
        x = row_label_w + col * cell_size
        y = header_h + row * cell_size
        grid.paste(resize(img), (x, y))

    # ---- Source / Reference: 居中跨两行贴一次 ----
    for col_idx, img in enumerate([source_render, reference_image]):
        resized = resize(img)
        x = row_label_w + col_idx * cell_size
        y_center = header_h + n_rows * cell_size // 2 - cell_size // 2
        grid.paste(resized, (x, y_center))

    # ---- Step columns: 逐行贴 ----
    for row_idx, steps_dict in enumerate([steps_without, steps_with]):
        for col_offset, step_key in enumerate(save_step_keys):
            if step_key in steps_dict:
                paste_cell(steps_dict[step_key], row_idx, n_shared_cols + col_offset)

        final_idx = max(k for k in steps_dict.keys() if k >= 0)
        if final_idx in steps_dict:
            paste_cell(steps_dict[final_idx], row_idx, n_shared_cols + len(save_step_keys))

    return grid


# =========================================================================
# TRELLIS Source Render 生成
# =========================================================================

def render_source_view_with_trellis(
    condition_pil: Image.Image,
    trellis_model_path: str,
    device: torch.device,
    seed: int = 42,
    render_resolution: int = 512,
    yaw: float = 3.141592653589793,
    pitch: float = 0.3,
    camera_r: float = 2.0,
    camera_fov: float = 40.0,
    adaptive_distance: bool = True,
    fill_ratio: float = 0.9,
) -> Image.Image:
    """
    用 TRELLIS 从 condition image 生成 3D 并渲染一个 source view。

    流程: condition image → TRELLIS 3D → GS render → source view PIL
    """
    import math
    import ml_collections
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler
    from trellis.modules.sparse import SparseTensor
    from trellis.utils.render_utils import (
        yaw_pitch_r_fov_to_extrinsics_intrinsics,
        render_frames,
    )

    if adaptive_distance:
        fov_rad = math.radians(camera_fov)
        camera_r = 0.5 / (fill_ratio * math.tan(fov_rad / 2))
        print(f"  Adaptive distance: fov={camera_fov}°, fill_ratio={fill_ratio} → r={camera_r:.4f}")

    print(f"  Loading TRELLIS: {trellis_model_path}")
    cfg = ml_collections.ConfigDict()
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = trellis_model_path
    cfg.verbose = False

    class _MockAccelerator:
        pass
    mock_acc = _MockAccelerator()
    mock_acc.device = device

    adapter = build_pipeline_from_reference(cfg, mock_acc, device=device)
    pipe = adapter.pipe

    # Condition encoding
    cond_dict = adapter.prepare_image_conditions([condition_pil])
    cond_emb = cond_dict['cond'].to(device)  # (1, S, C)
    uncond_emb = cond_dict.get('neg_cond', torch.zeros_like(cond_emb)).to(device)  # (1, S, C)

    # Stage 1: Sparse structure
    print("  Stage 1: Generating sparse structure...")
    torch.manual_seed(seed)
    coords = pipe.sample_sparse_structure(cond_dict, num_samples=1)
    print(f"    Coords: {coords.shape[0]} points")

    # Stage 2: ODE rollout
    print("  Stage 2: Sparse ODE rollout...")
    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = adapter.sparse.get_runtime_params()
    in_channels = pipe.models['slat_flow_model'].in_channels
    B = cond_emb.shape[0]

    g_ode = torch.Generator(device=device).manual_seed(seed)
    x_t = adapter.sparse.init_latents(coords=coords, in_channels=in_channels, generator=g_ode)

    scheduler = TrellisFlowScheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    model = pipe.models["slat_flow_model"]

    ode_steps = list(scheduler.timesteps)[:-1]
    for t_step in tqdm(ode_steps, desc="  ODE Rollout", leave=False):
        t_val = float(t_step.item())
        t_scaled = torch.full((B,), t_val * 1000, device=device, dtype=torch.float32)  # (B,)
        with torch.no_grad():
            uncond_v = model(x_t, t_scaled, uncond_emb)
            cond_v = model(x_t, t_scaled, cond_emb)
            cfg_w = slat_guidance if cfg_min <= t_val <= cfg_max else 0.0
            guided_feats = (1 + cfg_w) * cond_v.feats - cfg_w * uncond_v.feats  # (N, C)
            velocity = SparseTensor(coords=x_t.coords, feats=guided_feats)
        x_t = scheduler.step(velocity, t_val, x_t).prev_sample

    # Denormalize + decode
    norm = pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats = x_t.feats * std + mean  # (N, C)
    x_0_denorm = SparseTensor(coords=x_t.coords, feats=denorm_feats)

    decoded = adapter.sparse.decode(x_0_denorm, formats=['gaussian'])
    gs = decoded['gaussian'][0]

    # Render one view
    print("  Rendering source view...")
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], camera_r, camera_fov,
    )
    render_out = render_frames(
        gs, extrinsics, intrinsics,
        options={'resolution': render_resolution, 'bg_color': (1.0, 1.0, 1.0)},
        verbose=False,
    )
    color_np = render_out['color'][0]  # (H, W, 3) uint8
    rendered_pil = Image.fromarray(color_np)

    # Cleanup
    del adapter, pipe, model, gs, x_t, x_0_denorm
    torch.cuda.empty_cache()

    return rendered_pil


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reinforced Editing Trajectory 可视化对比"
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")

    # 输入模式（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--condition_image", type=str, default=None,
                             help="单张 reference image 路径")
    input_group.add_argument("--image_dir", type=str, default=None,
                             help="批量处理：包含图片的目录")

    parser.add_argument("--rendered_image", type=str, default=None,
                        help="Source render 路径（单图模式可用）")

    # TRELLIS 自动渲染模式
    parser.add_argument("--trellis_model", type=str, default=None,
                        help="TRELLIS 模型路径（指定后自动生成 source render）")
    parser.add_argument("--render_resolution", type=int, default=512,
                        help="TRELLIS source view 渲染分辨率")
    parser.add_argument("--camera_yaw", type=float, default=3.141592653589793,
                        help="Source view 相机 yaw (弧度), π=正面")
    parser.add_argument("--camera_pitch", type=float, default=0.3,
                        help="Source view 相机 pitch (弧度)")

    parser.add_argument("--prompt", type=str,
                        default="Rotate the camera.")
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/reinforced_editing_trajectory")

    parser.add_argument("--num_inference_steps", type=int, default=12)
    parser.add_argument("--n_max", type=int, default=9,
                        help="Active editing steps (config default: 9)")
    parser.add_argument("--noise_mode", type=str, default="aligned",
                        choices=["random", "fixed", "aligned"],
                        help="Noise update mode (pipeline default: aligned, config default: random)")
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="CFG scale (tgt=+cfg, src=-cfg)")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Guidance-distilled model guidance scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--save_steps", type=str, default="0,3,6,9",
                        help="Active step indices to save (comma-separated)")
    parser.add_argument("--cell_size", type=int, default=512,
                        help="Grid cell size in pixels")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


# =========================================================================
# 单张图片处理
# =========================================================================

def process_single_image(
    condition_path: str,
    rendered_pil: Optional[Image.Image],
    pipe,
    trellis_adapter,
    args,
    device: torch.device,
    save_step_keys: List[int],
    sample_out_dir: str,
):
    """处理单张图片：TRELLIS 渲染 → FlowEdit trajectory × 2 → grid。"""
    from edit4shape.systems.utils import composite_alpha_to_white

    sample_name = os.path.splitext(os.path.basename(condition_path))[0]
    out_dir = os.path.join(sample_out_dir, sample_name)
    os.makedirs(out_dir, exist_ok=True)

    condition_pil = composite_alpha_to_white(Image.open(condition_path))

    # ---- Source render ----
    if rendered_pil is not None:
        src_pil = rendered_pil
    elif trellis_adapter is not None:
        src_pil = render_single_view_from_adapter(
            trellis_adapter, condition_pil, device,
            seed=args.seed, render_resolution=args.render_resolution,
            yaw=args.camera_yaw, pitch=args.camera_pitch,
            camera_fov=40.0, adaptive_distance=True,
        )
    else:
        src_pil = condition_pil.copy()

    condition_pil.save(os.path.join(out_dir, "reference_image.png"))
    src_pil.save(os.path.join(out_dir, "source_render.png"))

    # ---- Setup shared state ----
    with torch.no_grad():
        shared = setup_pipeline(pipe, src_pil, condition_pil, args, device)

    save_set = set(save_step_keys)
    common_kwargs = dict(
        pipe=pipe,
        x_src=shared["x_src"],
        all_latents_list=shared["all_latents_list"],
        prompt_embeds_src=shared["prompt_embeds_src"],
        prompt_embeds_mask_src=shared["prompt_embeds_mask_src"],
        negative_prompt_embeds_src=shared["negative_prompt_embeds_src"],
        negative_prompt_embeds_mask_src=shared["negative_prompt_embeds_mask_src"],
        prompt_embeds_tgt=shared["prompt_embeds_tgt"],
        prompt_embeds_mask_tgt=shared["prompt_embeds_mask_tgt"],
        negative_prompt_embeds_tgt=shared["negative_prompt_embeds_tgt"],
        negative_prompt_embeds_mask_tgt=shared["negative_prompt_embeds_mask_tgt"],
        timesteps=shared["timesteps"],
        num_inference_steps=shared["num_inference_steps"],
        n_max=args.n_max,
        true_cfg_scale_src=shared["true_cfg_scale_src"],
        true_cfg_scale_tgt=shared["true_cfg_scale_tgt"],
        guidance=shared["guidance"],
        noise_init=shared["noise_init"],
        height=shared["height"],
        width=shared["width"],
        vae_image_sizes=shared["vae_image_sizes"],
        src_neg_same=shared["src_neg_same"],
        tgt_neg_same=shared["tgt_neg_same"],
        noise_mode=args.noise_mode,
        save_active_steps=save_set,
    )

    # ---- Two trajectories ----
    with torch.no_grad():
        _, steps_without = run_flowedit_trajectory(
            **common_kwargs, use_neg_src_guidance=False,
        )
        _, steps_with = run_flowedit_trajectory(
            **common_kwargs, use_neg_src_guidance=True,
        )

    # ---- Save ----
    wo_dir = os.path.join(out_dir, "without_neg_guidance")
    w_dir = os.path.join(out_dir, "with_neg_guidance")
    os.makedirs(wo_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)
    for step_idx, img in steps_without.items():
        name = f"step_{step_idx:02d}.png" if step_idx >= 0 else "step_final.png"
        img.save(os.path.join(wo_dir, name))
    for step_idx, img in steps_with.items():
        name = f"step_{step_idx:02d}.png" if step_idx >= 0 else "step_final.png"
        img.save(os.path.join(w_dir, name))

    grid = build_comparison_grid(
        source_render=src_pil,
        reference_image=condition_pil,
        steps_without=steps_without,
        steps_with=steps_with,
        save_step_keys=save_step_keys,
        cell_size=args.cell_size,
    )
    grid_path = os.path.join(out_dir, "comparison_grid.png")
    grid.save(grid_path)

    params_path = os.path.join(out_dir, "parameters.txt")
    with open(params_path, "w") as f:
        f.write(f"sample: {sample_name}\n")
        f.write("=" * 60 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        f.write(f"\nSteps saved (without): {sorted(steps_without.keys())}\n")
        f.write(f"Steps saved (with):    {sorted(steps_with.keys())}\n")

    del shared, steps_without, steps_with
    torch.cuda.empty_cache()

    return grid_path


# =========================================================================
# TRELLIS adapter 级别的单 view 渲染（复用已加载的 adapter）
# =========================================================================

def render_single_view_from_adapter(
    adapter, condition_pil: Image.Image, device: torch.device,
    seed: int = 42, render_resolution: int = 512,
    yaw: float = 3.141592653589793, pitch: float = 0.3,
    camera_fov: float = 40.0, adaptive_distance: bool = True,
    fill_ratio: float = 0.9,
) -> Image.Image:
    """用已加载的 TRELLIS adapter 渲染单个 view（不重复加载模型）。"""
    import math
    from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler
    from trellis.modules.sparse import SparseTensor
    from trellis.utils.render_utils import (
        yaw_pitch_r_fov_to_extrinsics_intrinsics,
        render_frames,
    )

    camera_r = 2.0
    if adaptive_distance:
        fov_rad = math.radians(camera_fov)
        camera_r = 0.5 / (fill_ratio * math.tan(fov_rad / 2))

    pipe_trellis = adapter.pipe

    cond_dict = adapter.prepare_image_conditions([condition_pil])
    cond_emb = cond_dict['cond'].to(device)  # (1, S, C)
    uncond_emb = cond_dict.get('neg_cond', torch.zeros_like(cond_emb)).to(device)  # (1, S, C)

    torch.manual_seed(seed)
    coords = pipe_trellis.sample_sparse_structure(cond_dict, num_samples=1)

    slat_steps, slat_guidance, slat_rescale_t, cfg_min, cfg_max, _ = adapter.sparse.get_runtime_params()
    in_channels = pipe_trellis.models['slat_flow_model'].in_channels
    B = cond_emb.shape[0]

    g_ode = torch.Generator(device=device).manual_seed(seed)
    x_t = adapter.sparse.init_latents(coords=coords, in_channels=in_channels, generator=g_ode)

    scheduler = TrellisFlowScheduler()
    scheduler.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    model = pipe_trellis.models["slat_flow_model"]

    for t_step in list(scheduler.timesteps)[:-1]:
        t_val = float(t_step.item())
        t_scaled = torch.full((B,), t_val * 1000, device=device, dtype=torch.float32)  # (B,)
        with torch.no_grad():
            uncond_v = model(x_t, t_scaled, uncond_emb)
            cond_v = model(x_t, t_scaled, cond_emb)
            cfg_w = slat_guidance if cfg_min <= t_val <= cfg_max else 0.0
            guided_feats = (1 + cfg_w) * cond_v.feats - cfg_w * uncond_v.feats  # (N, C)
            velocity = SparseTensor(coords=x_t.coords, feats=guided_feats)
        x_t = scheduler.step(velocity, t_val, x_t).prev_sample

    norm = pipe_trellis.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats = x_t.feats * std + mean  # (N, C)
    x_0_denorm = SparseTensor(coords=x_t.coords, feats=denorm_feats)

    decoded = adapter.sparse.decode(x_0_denorm, formats=['gaussian'])
    gs = decoded['gaussian'][0]

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], camera_r, camera_fov,
    )
    render_out = render_frames(
        gs, extrinsics, intrinsics,
        options={'resolution': render_resolution, 'bg_color': (1.0, 1.0, 1.0)},
        verbose=False,
    )
    rendered_pil = Image.fromarray(render_out['color'][0])

    del gs, x_t, x_0_denorm, decoded
    torch.cuda.empty_cache()

    return rendered_pil


# =========================================================================
# Main
# =========================================================================

def _collect_image_paths(image_dir: str) -> List[str]:
    """收集目录下所有图片文件，排序返回。"""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = []
    for f in sorted(os.listdir(image_dir)):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(os.path.join(image_dir, f))
    return paths


def _shard_list(lst: list, rank: int, world_size: int) -> list:
    """将列表按 rank 均匀分片（interleaved）。"""
    return lst[rank::world_size]


def main():
    args = parse_args()

    # ---- DDP / 单卡 自适应 ----
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    save_step_keys = [int(s.strip()) for s in args.save_steps.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 收集待处理图片 ----
    if args.image_dir:
        all_image_paths = _collect_image_paths(args.image_dir)
        if not all_image_paths:
            print(f"[ERROR] No images found in {args.image_dir}")
            return
    else:
        all_image_paths = [args.condition_image]

    image_paths = _shard_list(all_image_paths, rank, world_size)

    src_mode = "trellis" if args.trellis_model else ("file" if args.rendered_image else "copy")

    if is_main:
        print("=" * 70)
        print("Reinforced Editing Trajectory Visualization")
        print("=" * 70)
        print(f"  FlowEdit:    {args.model_path}")
        print(f"  Input:       {args.image_dir or args.condition_image} ({len(all_image_paths)} images)")
        print(f"  Source mode: {src_mode}")
        print(f"  DDP:         {world_size} GPU(s)")
        print(f"  Steps:       {args.num_inference_steps}, n_max: {args.n_max}")
        print(f"  Noise mode:  {args.noise_mode}")
        print(f"  CFG scale:   tgt=+{args.cfg_scale}, src=-{args.cfg_scale}")
        print(f"  Prompt:      {args.prompt!r}")
        print(f"  Neg prompt:  {args.negative_prompt!r}")
        print(f"  Seed:        {args.seed}")
        print(f"  Save steps:  {save_step_keys}")
        print(f"  Output:      {args.output_dir}")
        print("=" * 70)

    print(f"[Rank {rank}/{world_size}] Processing {len(image_paths)} images on {device}")

    # ---- 加载 TRELLIS adapter（如果需要，只加载一次）----
    trellis_adapter = None
    if args.trellis_model:
        import ml_collections
        from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference

        print(f"[Rank {rank}] Loading TRELLIS adapter...")
        t0 = time.time()
        cfg_t = ml_collections.ConfigDict()
        cfg_t.pretrained = ml_collections.ConfigDict()
        cfg_t.pretrained.model = args.trellis_model
        cfg_t.verbose = False

        class _MockAcc:
            pass
        mock_acc = _MockAcc()
        mock_acc.device = device
        trellis_adapter = build_pipeline_from_reference(cfg_t, mock_acc, device=device)
        print(f"[Rank {rank}] TRELLIS loaded in {time.time() - t0:.1f}s")

    # ---- 加载 FlowEdit pipeline（只加载一次）----
    print(f"[Rank {rank}] Loading FlowEdit pipeline...")
    t0 = time.time()
    from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
    pipe = FlowEditFullPipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"[Rank {rank}] FlowEdit loaded in {time.time() - t0:.1f}s")

    # ---- 预渲染的 source（单图模式才用）----
    rendered_pil = None
    if args.rendered_image:
        rendered_pil = Image.open(args.rendered_image).convert("RGB")

    # ---- 逐张处理 ----
    for idx, img_path in enumerate(image_paths):
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[Rank {rank}] [{idx+1}/{len(image_paths)}] Processing: {sample_name}")
        t0 = time.time()

        grid_path = process_single_image(
            condition_path=img_path,
            rendered_pil=rendered_pil,
            pipe=pipe,
            trellis_adapter=trellis_adapter,
            args=args,
            device=device,
            save_step_keys=save_step_keys,
            sample_out_dir=args.output_dir,
        )
        print(f"[Rank {rank}]   Done in {time.time() - t0:.1f}s → {grid_path}")

    # ---- 清理 ----
    del pipe
    if trellis_adapter is not None:
        del trellis_adapter
    torch.cuda.empty_cache()

    if use_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        print(f"\n{'='*70}")
        print(f"[DONE] All {len(all_image_paths)} images processed ({world_size} GPU(s)).")
        print(f"  Results: {args.output_dir}/")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
