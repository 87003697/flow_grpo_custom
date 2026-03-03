"""
Test-Time 3D Editing Pipeline
=============================

使用 TRELLIS 生成 3D 资产（GS + Mesh），然后用 FlowEdit 编辑 GS 渲染的多视角图像，
最后将编辑后的图像烘焙为 Mesh 纹理并导出 GLB。

Pipeline:
    Phase 1: 加载模型（TRELLIS + FlowEdit）
    Phase 2: TRELLIS 推理 → GS + Mesh
    Phase 3: 从 GS 渲染多视角图像
    Phase 4: FlowEdit 逐视角编辑
    Phase 5: Mesh 后处理 + 纹理烘焙
    Phase 6: 组装 GLB 并导出

用法:
    python -m edit4shape.experimental.test_time_edit \\
        --input_image assets/cat.png \\
        --flowedit_model /path/to/qwen-image-edit \\
        --source_prompt "a 3D model of a cat" \\
        --target_prompt "a 3D model of a golden cat wearing a crown" \\
        --nviews 16 \\
        --output_dir outputs/cat_golden
"""

import os
import sys
import argparse
import logging
from typing import List

# ---- 将 _reference_codes/TRELLIS 加入 sys.path，使得 `import trellis` 可用 ----
_repo_root = os.path.abspath(os.getcwd())
_trellis_ref_root = os.path.join(_repo_root, "_reference_codes", "TRELLIS")
if _trellis_ref_root not in sys.path:
    sys.path.insert(0, _trellis_ref_root)

import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Argument Parsing
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-Time 3D Editing: TRELLIS → FlowEdit → Texture Bake → GLB",
    )

    # ---- I/O ----
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="输出目录（默认: ./outputs）",
    )
    parser.add_argument(
        "--output_name", type=str, default="edited",
        help="输出文件名前缀（默认: edited）",
    )

    # ---- 模型 ----
    parser.add_argument(
        "--trellis_model", type=str, default="microsoft/TRELLIS-image-large",
        help="TRELLIS 模型路径或 HuggingFace 名称",
    )
    parser.add_argument(
        "--flowedit_model", type=str, required=True,
        help="FlowEdit（QwenImage-Edit）模型路径",
    )

    # ---- TRELLIS 推理 ----
    parser.add_argument(
        "--trellis_seed", type=int, default=1,
        help="TRELLIS 随机种子（默认: 1）",
    )

    # ---- 多视角渲染 ----
    parser.add_argument(
        "--render_resolution", type=int, default=1024,
        help="GS 渲染分辨率（默认: 1024）",
    )
    parser.add_argument(
        "--nviews", type=int, default=16,
        help="渲染 & 编辑的视角数（默认: 16）",
    )

    # ---- FlowEdit 编辑 ----
    parser.add_argument(
        "--source_prompt", type=str, required=True,
        help="源 prompt（描述原始 3D 物体）",
    )
    parser.add_argument(
        "--target_prompt", type=str, required=True,
        help="目标 prompt（描述编辑后的 3D 物体）",
    )
    parser.add_argument(
        "--negative_prompt_src", type=str, default="",
        help="源分支负 prompt（默认: 空字符串）",
    )
    parser.add_argument(
        "--negative_prompt_tgt", type=str, default="",
        help="目标分支负 prompt（默认: 空字符串）",
    )
    parser.add_argument(
        "--edit_steps", type=int, default=50,
        help="FlowEdit 采样步数（默认: 50）",
    )
    parser.add_argument(
        "--cfg_scale_src", type=float, default=1.5,
        help="源分支 CFG 强度（默认: 1.5）",
    )
    parser.add_argument(
        "--cfg_scale_tgt", type=float, default=5.5,
        help="目标分支 CFG 强度（默认: 5.5）",
    )
    parser.add_argument(
        "--n_max", type=int, default=20,
        help="FlowEdit 生效步数范围（默认: 20）",
    )
    parser.add_argument(
        "--noise_mode", type=str, default="aligned",
        choices=["random", "fixed", "aligned"],
        help="FlowEdit 噪声模式（默认: aligned）",
    )
    parser.add_argument(
        "--edit_seed", type=int, default=42,
        help="FlowEdit 随机种子（默认: 42）",
    )
    parser.add_argument(
        "--edit_resolution", type=int, default=None,
        help="FlowEdit 编辑分辨率（默认: None，自动从图像计算）",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=None,
        help="Guidance embedding scale（默认: None，仅 guidance-distilled 模型需要）",
    )
    parser.add_argument(
        "--bg_color", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="条件图背景色 R G B [0,1]（默认: 1 1 1，白色，与 GS renderer 一致）",
    )

    # ---- 纹理烘焙 ----
    parser.add_argument(
        "--simplify", type=float, default=0.95,
        help="Mesh 简化比例，0 表示不简化（默认: 0.95）",
    )
    parser.add_argument(
        "--texture_size", type=int, default=1024,
        help="纹理贴图尺寸（默认: 1024）",
    )
    parser.add_argument(
        "--bake_mode", type=str, default="opt", choices=["opt", "fast"],
        help="纹理烘焙模式（默认: opt）",
    )

    # ---- 可选输出 ----
    parser.add_argument(
        "--save_video", action="store_true",
        help="保存编辑前后对比视频",
    )
    parser.add_argument(
        "--save_intermediate", action="store_true",
        help="保存中间产物（渲染图、编辑图、纹理）",
    )

    return parser.parse_args()


# =========================================================================
# Phase Functions
# =========================================================================

def phase1_load_trellis(args) -> "TrellisImageTo3DPipeline":
    """Phase 1a: 加载 TRELLIS Pipeline"""
    from trellis.pipelines import TrellisImageTo3DPipeline

    logger.info(f"[Phase 1] Loading TRELLIS from: {args.trellis_model}")
    trellis_pipe = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model)
    trellis_pipe.cuda()
    logger.info("[Phase 1] TRELLIS loaded.")
    return trellis_pipe


def phase2_trellis_inference(
    trellis_pipe,
    input_image: Image.Image,
    seed: int,
) -> dict:
    """
    Phase 2: TRELLIS 推理，生成 GS + Mesh。

    Returns:
        dict with keys:
            - 'gs': Gaussian 表征
            - 'mesh': MeshExtractResult
            - 'condition_pil': 预处理后的条件图
    """
    logger.info("[Phase 2] Running TRELLIS inference...")

    # 预处理输入图（去背景、裁剪、resize 到 518×518）
    condition_pil = trellis_pipe.preprocess_image(input_image)

    # 推理（只需要 gaussian 和 mesh，不需要 radiance_field）
    outputs = trellis_pipe.run(
        input_image,
        seed=seed,
        formats=["gaussian", "mesh"],
    )
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    logger.info(f"[Phase 2] GS: {gs.get_xyz.shape[0]} gaussians")
    logger.info(f"[Phase 2] Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    return {
        "gs": gs,
        "mesh": mesh,
        "condition_pil": condition_pil,
    }


def phase3_render_multiview(
    gs,
    resolution: int,
    nviews: int,
) -> dict:
    """
    Phase 3: 从 GS 渲染多视角图像。

    Returns:
        dict with keys:
            - 'observations': List[np.ndarray]  — N 张 (H, W, 3) uint8 RGB
            - 'extrinsics': List[torch.Tensor]  — N 个 (4, 4) 外参
            - 'intrinsics': List[torch.Tensor]  — N 个 (3, 3) 内参
    """
    from trellis.utils import render_utils

    logger.info(f"[Phase 3] Rendering {nviews} views at {resolution}×{resolution}...")

    observations, extrinsics, intrinsics = render_utils.render_multiview(
        gs, resolution=resolution, nviews=nviews,
    )

    logger.info(f"[Phase 3] Rendered {len(observations)} views, shape: {observations[0].shape}")

    return {
        "observations": observations,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def phase4_flowedit(
    observations: List[np.ndarray],
    condition_pil: Image.Image,
    args,
) -> List[np.ndarray]:
    """
    Phase 4: 用 FlowEdit 逐视角编辑。

    Args:
        observations: 原始 GS 渲染的多视角图像
        condition_pil: 条件图（TRELLIS 预处理后的输入图）
        args: 命令行参数

    Returns:
        edited_observations: 编辑后的多视角图像 List[np.ndarray] (H, W, 3) uint8
    """
    from edit4shape.guidance.pipelines.qwen_image_edit import FlowEditFullPipeline
    from edit4shape.systems.utils import composite_alpha

    # 加载 FlowEdit
    logger.info(f"[Phase 4] Loading FlowEdit from: {args.flowedit_model}")
    flowedit_pipe = FlowEditFullPipeline.from_pretrained(
        args.flowedit_model, torch_dtype=torch.bfloat16,
    ).to("cuda")
    flowedit_pipe.set_progress_bar_config(disable=True)
    logger.info("[Phase 4] FlowEdit loaded.")

    # 准备条件图（合成到指定背景色）
    condition_edit = composite_alpha(condition_pil, tuple(args.bg_color))

    logger.info(f"[Phase 4] Editing {len(observations)} views...")
    logger.info(f"  Source prompt: '{args.source_prompt}'")
    logger.info(f"  Target prompt: '{args.target_prompt}'")

    edited_observations = []
    for i, obs in enumerate(tqdm(observations, desc="FlowEdit Editing")):
        rendered_pil = Image.fromarray(obs)

        # 每张图使用相同的种子，保证可复现性
        device = torch.device(flowedit_pipe._execution_device)
        generator = torch.Generator(device=device).manual_seed(args.edit_seed)

        # 构建 FlowEdit 调用参数
        pipe_kwargs = dict(
            image=[rendered_pil, condition_edit],
            target_prompt=args.target_prompt,
            source_prompt=args.source_prompt,
            negative_prompt_src=args.negative_prompt_src,
            negative_prompt_tgt=args.negative_prompt_tgt,
            num_inference_steps=args.edit_steps,
            true_cfg_scale_src=args.cfg_scale_src,
            true_cfg_scale_tgt=args.cfg_scale_tgt,
            n_max=args.n_max,
            noise_mode=args.noise_mode,
            generator=generator,
            # test-time 不需要记录 tracker
            use_tgt_record=False,
            use_src_record=False,
            output_type="pil",
        )

        # 可选参数
        if args.edit_resolution is not None:
            pipe_kwargs["height"] = args.edit_resolution
            pipe_kwargs["width"] = args.edit_resolution
        if args.guidance_scale is not None:
            pipe_kwargs["guidance_scale"] = args.guidance_scale

        with torch.no_grad():
            output = flowedit_pipe(**pipe_kwargs)

        edited_pil = output.images[0]

        # 如果编辑后的分辨率与渲染分辨率不一致，resize 回来
        render_size = (obs.shape[1], obs.shape[0])  # (W, H)
        if edited_pil.size != render_size:
            edited_pil = edited_pil.resize(render_size, Image.LANCZOS)

        edited_observations.append(np.array(edited_pil))

    # 释放 FlowEdit 显存
    logger.info("[Phase 4] Unloading FlowEdit...")
    del flowedit_pipe
    torch.cuda.empty_cache()

    return edited_observations


def phase5_bake_texture(
    mesh,
    edited_observations: List[np.ndarray],
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    args,
) -> dict:
    """
    Phase 5: Mesh 后处理 + UV 展开 + 纹理烘焙。

    Returns:
        dict with keys:
            - 'vertices': np.ndarray (V, 3)
            - 'faces': np.ndarray (F, 3)
            - 'uvs': np.ndarray (V, 2)
            - 'texture': np.ndarray (texture_size, texture_size, 3) uint8
    """
    from trellis.utils import postprocessing_utils

    logger.info("[Phase 5] Postprocessing mesh...")

    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    # Mesh 后处理（简化 + 去不可见面 + 填洞）
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices, faces,
        simplify=args.simplify > 0,
        simplify_ratio=args.simplify,
        fill_holes=True,
        fill_holes_max_hole_size=0.04,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(max(1 - args.simplify, 0.01))),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        verbose=True,
    )
    logger.info(f"[Phase 5] After postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # UV 展开（xatlas）
    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
    logger.info(f"[Phase 5] After UV parametrize: {vertices.shape[0]} vertices, {uvs.shape[0]} UVs")

    # 纹理烘焙
    logger.info(f"[Phase 5] Baking texture ({args.bake_mode} mode, {args.texture_size}×{args.texture_size})...")
    masks = [np.any(obs > 0, axis=-1) for obs in edited_observations]
    extrinsics_np = [ext.cpu().numpy() for ext in extrinsics]
    intrinsics_np = [intr.cpu().numpy() for intr in intrinsics]

    texture = postprocessing_utils.bake_texture(
        vertices, faces, uvs,
        edited_observations, masks,
        extrinsics_np, intrinsics_np,
        texture_size=args.texture_size,
        mode=args.bake_mode,
        verbose=True,
    )
    logger.info(f"[Phase 5] Texture baked: {texture.shape}")

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "texture": texture,
    }


def phase6_export(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
    args,
) -> str:
    """
    Phase 6: 组装 GLB 并导出。

    Returns:
        output_path: GLB 文件路径
    """
    logger.info("[Phase 6] Assembling and exporting GLB...")

    texture_img = Image.fromarray(texture)

    # z-up → y-up 坐标变换
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    glb_mesh = trimesh.Trimesh(
        vertices, faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
    )

    output_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
    glb_mesh.export(output_path)
    logger.info(f"[Phase 6] Exported GLB: {output_path}")

    return output_path


# =========================================================================
# 辅助函数
# =========================================================================

def save_intermediate(
    observations: List[np.ndarray],
    edited_observations: List[np.ndarray],
    texture: np.ndarray,
    output_dir: str,
):
    """保存中间产物"""
    inter_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(inter_dir, exist_ok=True)

    for i, obs in enumerate(observations):
        Image.fromarray(obs).save(os.path.join(inter_dir, f"original_{i:03d}.png"))

    for i, edited in enumerate(edited_observations):
        Image.fromarray(edited).save(os.path.join(inter_dir, f"edited_{i:03d}.png"))

    Image.fromarray(texture).save(os.path.join(inter_dir, "texture.png"))

    logger.info(f"[Save] Intermediate results saved to: {inter_dir}")


def save_comparison_video(
    observations: List[np.ndarray],
    edited_observations: List[np.ndarray],
    output_dir: str,
    output_name: str,
):
    """保存编辑前后对比视频（左右拼接）"""
    import imageio

    frames = []
    for orig, edited in zip(observations, edited_observations):
        # 左右拼接：原始 | 编辑后
        frame = np.concatenate([orig, edited], axis=1)  # (H, W*2, 3)
        frames.append(frame)

    video_path = os.path.join(output_dir, f"{output_name}_comparison.mp4")
    imageio.mimsave(video_path, frames, fps=2)
    logger.info(f"[Save] Comparison video saved to: {video_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Test-Time 3D Editing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_image}")
    logger.info(f"Output: {args.output_dir}/{args.output_name}.glb")
    logger.info(f"Edit:   '{args.source_prompt}' → '{args.target_prompt}'")
    logger.info("=" * 60)

    input_image = Image.open(args.input_image)

    # ===== Phase 1 + 2: TRELLIS =====
    trellis_pipe = phase1_load_trellis(args)
    trellis_output = phase2_trellis_inference(
        trellis_pipe, input_image, seed=args.trellis_seed,
    )
    gs = trellis_output["gs"]
    mesh = trellis_output["mesh"]
    condition_pil = trellis_output["condition_pil"]

    # ===== Phase 3: 渲染多视角 =====
    render_output = phase3_render_multiview(
        gs,
        resolution=args.render_resolution,
        nviews=args.nviews,
    )
    observations = render_output["observations"]
    extrinsics = render_output["extrinsics"]
    intrinsics = render_output["intrinsics"]

    # 卸载 TRELLIS（GS/Mesh 数据量小，留在显存/内存中）
    logger.info("[Memory] Unloading TRELLIS to free GPU memory...")
    del trellis_pipe
    torch.cuda.empty_cache()

    # ===== Phase 4: FlowEdit 编辑 =====
    edited_observations = phase4_flowedit(
        observations, condition_pil, args,
    )

    # ===== Phase 5: 纹理烘焙 =====
    bake_output = phase5_bake_texture(
        mesh, edited_observations, extrinsics, intrinsics, args,
    )

    # ===== Phase 6: 导出 GLB =====
    output_path = phase6_export(
        bake_output["vertices"],
        bake_output["faces"],
        bake_output["uvs"],
        bake_output["texture"],
        args,
    )

    # ===== 可选输出 =====
    if args.save_intermediate:
        save_intermediate(
            observations, edited_observations,
            bake_output["texture"], args.output_dir,
        )

    if args.save_video:
        save_comparison_video(
            observations, edited_observations,
            args.output_dir, args.output_name,
        )

    logger.info("=" * 60)
    logger.info(f"Done! Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
