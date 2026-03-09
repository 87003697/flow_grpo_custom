#!/usr/bin/env python3
"""
TRELLIS.2 FlowEdit Refine 推理脚本。

两阶段流程：
  Stage 1: 标准 TRELLIS.2 推理 → 得到 clean latent (x_src)
  Stage 2: FlowEdit 差分采样 → 对 x_src 做 refine

FlowEdit 配置约定（对齐 2D FlowEdit 方案）：
  - source 和 target 使用相同的图像条件（DINOv2 embedding）
  - Target CFG: positive guidance_strength（拉向条件）
  - Source CFG: negative guidance_strength（推离条件，核心创新点）
  - 差分更新: z_edit += dt * (v_cfg_tgt - v_cfg_src)

用法:
  python scripts/debug/test_trellis2_flowedit.py \
    --model_path /path/to/trellis2 \
    --input_image path/to/image.png \
    --refine_stages shape tex \
    --output_dir outputs/trellis2_flowedit
"""

import argparse
import os
import sys
import time

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# Path Setup
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
TRELLIS2_REF = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS.2")
for p in [PROJECT_ROOT, TRELLIS2_REF]:
    if p not in sys.path:
        sys.path.insert(0, p)

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
from edit4shape.generators.trellis2.pipeline_adapter import Trellis2RefAdapter
from edit4shape.generators.trellis2.scheduler import FlowEulerScheduler


# ============================================================
# 工具函数
# ============================================================

def seed_everything(seed: int):
    """设置所有随机种子，确保可复现。"""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CFG 辅助函数
# ============================================================

def sparse_cfg_l2_rescale(
    cond_pred: SparseTensor,
    uncond_pred: SparseTensor,
    guidance_strength: float,
    eps: float = 1e-8,
) -> SparseTensor:
    """
    CFG with per-point L2 norm rescale（对齐 2D FlowEdit）。

    公式:
      combined = g * cond + (1-g) * uncond
      rescaled = combined * (||cond||_2 / ||combined||_2)  # 逐点

    Args:
        cond_pred: SparseTensor, 条件 velocity 预测
        uncond_pred: SparseTensor, 无条件 velocity 预测
        guidance_strength: CFG 强度（可为负数）
        eps: 防止除零

    Returns:
        SparseTensor: CFG + L2 rescale 后的 velocity
    """
    # CFG combine: combined = g * cond + (1-g) * uncond
    combined = (
        guidance_strength * cond_pred.feats
        + (1 - guidance_strength) * uncond_pred.feats
    )  # (N, C)

    # L2 norm rescale (per point)
    cond_norm = torch.norm(cond_pred.feats, dim=-1, keepdim=True).clamp(min=eps)  # (N, 1)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=eps)  # (N, 1)
    rescaled = combined * (cond_norm / combined_norm)  # (N, C)

    return cond_pred.replace(rescaled)


def sparse_cfg_standard(
    cond_pred: SparseTensor,
    uncond_pred: SparseTensor,
    guidance_strength: float,
) -> SparseTensor:
    """
    Standard CFG（无 rescale）。

    公式: pred = g * cond + (1-g) * uncond
    """
    combined = (
        guidance_strength * cond_pred.feats
        + (1 - guidance_strength) * uncond_pred.feats
    )  # (N, C)
    return cond_pred.replace(combined)


def _apply_cfg(
    cond_pred: SparseTensor,
    uncond_pred: SparseTensor,
    guidance_strength: float,
    rescale_mode: str = "l2_norm",
) -> SparseTensor:
    """
    Apply CFG，支持多种 rescale 模式。

    Args:
        rescale_mode: "l2_norm" (2D FlowEdit 方案) | "none" (标准 CFG)
    """
    if guidance_strength == 1.0:
        return cond_pred
    if guidance_strength == 0.0:
        return uncond_pred

    if rescale_mode == "l2_norm":
        return sparse_cfg_l2_rescale(cond_pred, uncond_pred, guidance_strength)
    else:
        return sparse_cfg_standard(cond_pred, uncond_pred, guidance_strength)


# ============================================================
# 归一化辅助函数
# ============================================================

def normalize_slat(pipe, slat: SparseTensor, stage: str) -> SparseTensor:
    """反归一化域 → 归一化域（flow model 操作域）。"""
    if stage == "shape":
        norm = pipe.shape_slat_normalization
    else:
        norm = pipe.tex_slat_normalization
    std = torch.tensor(norm["std"])[None].to(slat.device)  # (1, C)
    mean = torch.tensor(norm["mean"])[None].to(slat.device)  # (1, C)
    feats = (slat.feats - mean) / std  # (N, C)
    return slat.replace(feats)


def denormalize_slat(pipe, slat: SparseTensor, stage: str) -> SparseTensor:
    """归一化域 → 反归一化域。"""
    if stage == "shape":
        norm = pipe.shape_slat_normalization
    else:
        norm = pipe.tex_slat_normalization
    std = torch.tensor(norm["std"])[None].to(slat.device)  # (1, C)
    mean = torch.tensor(norm["mean"])[None].to(slat.device)  # (1, C)
    feats = slat.feats * std + mean  # (N, C)
    return slat.replace(feats)


def get_sigma_min(pipe, stage: str) -> float:
    """获取指定阶段的 sigma_min。"""
    if stage == "shape":
        return pipe.shape_slat_sampler.sigma_min
    else:
        return pipe.tex_slat_sampler.sigma_min


# ============================================================
# 多视角渲染
# ============================================================

def load_envmap(envmap_path: str, device: torch.device) -> EnvMap:
    """加载 HDR 环境贴图。"""
    img = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(img, dtype=torch.float32, device=device))


def render_multiview(
    meshes,
    envmap: EnvMap,
    resolution: int = 512,
    nviews: int = 4,
    r: float = 2.0,
    fov: float = 36.0,
    yaw_offset_deg: float = -16.0,
    pitch_deg: float = 20.0,
):
    """
    渲染多视角图片（PBR shaded + normal）。

    Args:
        meshes: List[MeshWithVoxel]，decode_latent 返回的结果
        envmap: EnvMap，环境光贴图
        resolution: 渲染分辨率
        nviews: 视角数量
        r: 相机距离
        fov: 视场角 (度)
        yaw_offset_deg: yaw 偏移 (度)
        pitch_deg: pitch 角度 (度)

    Returns:
        List[dict]: 每个 mesh 一个 dict，包含 "shaded" 和 "normal" 等 key，
                    每个 key 对应 List[np.ndarray] (H, W, 3) uint8
    """
    all_results = []
    for mesh in meshes:
        result = render_utils.render_snapshot(
            mesh,
            resolution=resolution,
            r=r,
            fov=fov,
            nviews=nviews,
            offset=(yaw_offset_deg / 180.0 * np.pi, pitch_deg / 180.0 * np.pi),
            envmap=envmap,
        )
        all_results.append(result)
    return all_results


def save_multiview_images(
    render_result: dict,
    output_dir: str,
    prefix: str,
    cond_image: Image.Image = None,
    channels: list = None,
):
    """
    保存多视角渲染结果为图片文件。

    输出:
      - {prefix}_{channel}_v{i}.png: 每个视角的单独图片
      - {prefix}_{channel}_grid.png: 所有视角拼接为一行 + 可选条件图

    Args:
        render_result: render_multiview 返回的单个 mesh 的 dict
        output_dir: 输出目录
        prefix: 文件名前缀（如 "baseline" 或 "refined"）
        cond_image: 条件图像（可选，拼在 grid 最左侧）
        channels: 要保存的通道列表（默认 ["shaded", "normal"]）
    """
    os.makedirs(output_dir, exist_ok=True)
    if channels is None:
        channels = ["shaded", "normal"]

    for ch in channels:
        if ch not in render_result:
            continue
        views = render_result[ch]  # List[np.ndarray] (H, W, 3) uint8

        # 保存每个视角
        for i, view in enumerate(views):
            pil = Image.fromarray(view)
            pil.save(os.path.join(output_dir, f"{prefix}_{ch}_v{i}.png"))

        # 拼接 grid: [cond | v0 | v1 | ... | vN]
        margin = 12
        pil_views = [Image.fromarray(v) for v in views]

        imgs = []
        if cond_image is not None:
            # 缩放条件图到与渲染分辨率相同的高度
            h = pil_views[0].height
            w_scaled = max(1, int(cond_image.width * h / cond_image.height))
            imgs.append(cond_image.resize((w_scaled, h), Image.LANCZOS))
        imgs.extend(pil_views)

        total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
        total_h = max(im.height for im in imgs) + margin * 2
        grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x = margin
        for im in imgs:
            grid.paste(im, (x, margin))
            x += im.width + margin
        grid.save(os.path.join(output_dir, f"{prefix}_{ch}_grid.png"))

    print(f"  多视角渲染已保存: {output_dir}/{prefix}_*")


# ============================================================
# FlowEdit 核心循环
# ============================================================

@torch.no_grad()
def flowedit_refine_sparse(
    adapter: Trellis2RefAdapter,
    x_src_norm: SparseTensor,
    cond: torch.Tensor,
    neg_cond: torch.Tensor,
    stage: str,
    resolution: int,
    steps: int = 50,
    n_max: int = 50,
    cfg_strength_tgt: float = 3.0,
    cfg_strength_src: float = -3.0,
    sigma_min: float = 0.0,
    cfg_interval: tuple = (0.0, 1.0),
    shape_cond: SparseTensor = None,
    device: torch.device = None,
    seed: int = 42,
    rescale_mode: str = "l2_norm",
) -> SparseTensor:
    """
    FlowEdit 差分采样循环（单阶段，SparseTensor 版本）。

    核心算法（对齐 2D FlowEdit 方案）：
      z_edit = x_src  （从 clean source latent 启动）
      for each step t (从 t=1 到 t=0):
        if 跳过（不在 n_max 范围内）: continue

        # Source branch: 加噪 + 双分支 CFG
        z_src_t = (1-t) * x_src + noise_coeff(t) * noise
        v_cfg_src = CFG(model(z_src_t, t, cond), model(z_src_t, t, neg_cond), cfg_src)

        # Target branch: 共享噪声偏移 + 双分支 CFG
        z_tgt_t = z_edit + z_src_t - x_src
        v_cfg_tgt = CFG(model(z_tgt_t, t, cond), model(z_tgt_t, t, neg_cond), cfg_tgt)

        # 差分更新
        z_edit += dt * (v_cfg_tgt - v_cfg_src)

    每步需 4 次模型前向（source cond/uncond + target cond/uncond）。

    Args:
        adapter: Trellis2RefAdapter, 封装了 sampling_step 等推理接口
        x_src_norm: SparseTensor, 标准 rollout 输出的 clean latent（归一化域）
        cond: (B, S, C) DINOv2 图像条件嵌入
        neg_cond: (B, S, C) 零向量（无条件嵌入）
        stage: "shape" 或 "tex"
        resolution: flow model 分辨率（512 或 1024）
        steps: 扩散步数
        n_max: FlowEdit 生效步数（仅最后 n_max 步执行差分更新）
        cfg_strength_tgt: target 分支 CFG 强度（正值，拉向条件）
        cfg_strength_src: source 分支 CFG 强度（负值，推离条件）
        sigma_min: flow matching 噪声下限参数
        cfg_interval: CFG 生效的时间区间 (min, max)
        shape_cond: SparseTensor, tex 阶段需要的 shape 条件（已归一化）
        device: 计算设备
        seed: 噪声随机种子
        rescale_mode: CFG rescale 模式 ("l2_norm" | "none")

    Returns:
        SparseTensor: refine 后的 latent（归一化域）
    """
    # ---- Scheduler 初始化 ----
    sampler_params = adapter.get_sampler_params(stage)
    rescale_t = float(sampler_params["rescale_t"])
    scheduler = FlowEulerScheduler(rescale_t=rescale_t)
    scheduler.set_timesteps(steps, device=device)

    # ---- 初始化 z_edit ----
    z_edit = x_src_norm  # SparseTensor（从 clean source 启动）

    # ---- 生成固定噪声（全步共享，对齐 2D FlowEdit "fixed" 模式）----
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        x_src_norm.feats.shape, device=device,
        dtype=x_src_norm.feats.dtype, generator=gen,
    )  # (N, C)

    step_indices = scheduler.get_timesteps_for_loop()  # [0, 1, ..., steps-1]
    total_steps = len(step_indices)
    B = cond.shape[0]  # ()

    active_count = 0
    for step_idx in tqdm(step_indices, desc=f"FlowEdit [{stage}]"):
        # 跳过不在 n_max 范围内的步骤
        remaining = total_steps - step_idx
        if remaining > n_max:
            continue
        active_count += 1

        t_val = scheduler.get_precise_t(step_idx)  # float64 精度
        t_prev_val = scheduler.get_precise_t(step_idx + 1)  # float64 精度
        dt = t_prev_val - t_val  # 负值（时间从 1→0 递减）

        t_batch = torch.full(
            (B,), t_val, device=device, dtype=torch.float32
        )  # (B,)

        # 判断是否启用 CFG
        use_cfg = cfg_interval[0] <= t_val <= cfg_interval[1]

        # ===== 1. Source Branch =====
        # 加噪: z_src_t = (1-t)*x_src + (sigma_min + (1-sigma_min)*t) * noise
        noise_coeff = sigma_min + (1 - sigma_min) * t_val  # scalar
        src_noisy_feats = (
            (1 - t_val) * x_src_norm.feats + noise_coeff * noise
        )  # (N, C)
        z_src_t = x_src_norm.replace(src_noisy_feats)

        # Source cond 预测
        v_cond_src = adapter.sampling_step(
            z_src_t, t_batch, cond, stage, resolution,
            shape_cond=shape_cond,
        )  # SparseTensor

        # Source CFG
        if use_cfg and cfg_strength_src != 1.0:
            v_uncond_src = adapter.sampling_step(
                z_src_t, t_batch, neg_cond, stage, resolution,
                shape_cond=shape_cond,
            )  # SparseTensor
            v_cfg_src = _apply_cfg(
                v_cond_src, v_uncond_src, cfg_strength_src, rescale_mode,
            )
        else:
            v_cfg_src = v_cond_src

        # ===== 2. Target Branch =====
        # 共享噪声偏移: z_tgt_t = z_edit + z_src_t - x_src
        tgt_feats = z_edit.feats + z_src_t.feats - x_src_norm.feats  # (N, C)
        z_tgt_t = x_src_norm.replace(tgt_feats)

        # Target cond 预测
        v_cond_tgt = adapter.sampling_step(
            z_tgt_t, t_batch, cond, stage, resolution,
            shape_cond=shape_cond,
        )  # SparseTensor

        # Target CFG
        if use_cfg and cfg_strength_tgt != 1.0:
            v_uncond_tgt = adapter.sampling_step(
                z_tgt_t, t_batch, neg_cond, stage, resolution,
                shape_cond=shape_cond,
            )  # SparseTensor
            v_cfg_tgt = _apply_cfg(
                v_cond_tgt, v_uncond_tgt, cfg_strength_tgt, rescale_mode,
            )
        else:
            v_cfg_tgt = v_cond_tgt

        # ===== 3. 差分更新 z_edit =====
        v_delta_feats = v_cfg_tgt.feats - v_cfg_src.feats  # (N, C)
        z_edit_feats = z_edit.feats + dt * v_delta_feats  # (N, C)
        z_edit = z_edit.replace(z_edit_feats)

    print(f"  [{stage}] FlowEdit 完成: {active_count} active steps "
          f"(n_max={n_max}, total={total_steps})")
    return z_edit


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 FlowEdit Refine 推理脚本"
    )

    # ---- 模型 ----
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="TRELLIS.2 预训练模型路径 (如 ./pretrained_weights/TRELLIS.2-4B)",
    )
    parser.add_argument(
        "--dino_local_path", type=str, default=None,
        help="DINOv2 本地模型路径（可选）",
    )
    parser.add_argument(
        "--pipeline_type", type=str, default="1024",
        choices=["512", "1024", "1024_cascade"],
        help="Pipeline 类型 (default: 1024)",
    )

    # ---- 输入输出 ----
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="输入图像路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/trellis2_flowedit",
        help="输出目录",
    )
    parser.add_argument(
        "--no_preprocess", action="store_true",
        help="跳过图像预处理（去背等）",
    )

    # ---- FlowEdit 参数 ----
    parser.add_argument(
        "--refine_stages", nargs="+", default=["shape", "tex"],
        choices=["shape", "tex"],
        help="要 refine 的阶段 (default: shape tex)",
    )
    parser.add_argument(
        "--refine_steps", type=int, default=None,
        help="FlowEdit 扩散步数 (default: 与标准 rollout 相同)",
    )
    parser.add_argument(
        "--refine_n_max", type=int, default=None,
        help="FlowEdit 生效步数 (default: 全部步骤)",
    )
    parser.add_argument(
        "--cfg_strength_tgt", type=float, default=None,
        help="Target CFG 强度（正值，default: 与标准 rollout 相同）",
    )
    parser.add_argument(
        "--cfg_strength_src", type=float, default=None,
        help="Source CFG 强度（负值，default: -cfg_tgt）",
    )
    parser.add_argument(
        "--rescale_mode", type=str, default="l2_norm",
        choices=["l2_norm", "none"],
        help="CFG rescale 模式 (default: l2_norm)",
    )
    parser.add_argument(
        "--num_refine_rounds", type=int, default=1,
        help="FlowEdit 迭代轮数 (default: 1)",
    )

    # ---- 渲染输出 ----
    parser.add_argument(
        "--render_resolution", type=int, default=512,
        help="多视角渲染分辨率 (default: 512)",
    )
    parser.add_argument(
        "--num_views", type=int, default=4,
        help="多视角渲染的视角数量 (default: 4)",
    )
    parser.add_argument(
        "--envmap_path", type=str, default=None,
        help="HDR 环境贴图路径 (default: _reference_codes/TRELLIS.2/assets/hdri/forest.exr)",
    )
    parser.add_argument(
        "--render_channels", nargs="+", default=["shaded", "normal"],
        help="要保存的渲染通道 (default: shaded normal)",
    )

    # ---- 通用 ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--low_vram", action="store_true",
        help="低显存模式（模型按需加载到 GPU）",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定 flow_resolution（FlowEdit 使用的模型分辨率）
    flow_resolution = 1024 if args.pipeline_type in ["1024", "1024_cascade"] else 512

    # 解析 envmap 路径
    if args.envmap_path is None:
        args.envmap_path = os.path.join(
            PROJECT_ROOT, "_reference_codes", "TRELLIS.2",
            "assets", "hdri", "forest.exr",
        )

    print("=" * 60)
    print("TRELLIS.2 FlowEdit Refine")
    print("=" * 60)
    print(f"  Model:           {args.model_path}")
    print(f"  Pipeline:        {args.pipeline_type}")
    print(f"  Flow Resolution: {flow_resolution}")
    print(f"  Input:           {args.input_image}")
    print(f"  Refine stages:   {args.refine_stages}")
    print(f"  Refine steps:    {args.refine_steps or 'auto'}")
    print(f"  Refine n_max:    {args.refine_n_max or 'auto (=steps)'}")
    print(f"  CFG tgt:         {args.cfg_strength_tgt or 'auto'}")
    print(f"  CFG src:         {args.cfg_strength_src or 'auto (-tgt)'}")
    print(f"  Rescale mode:    {args.rescale_mode}")
    print(f"  Rounds:          {args.num_refine_rounds}")
    print(f"  Render:          {args.render_resolution}px x {args.num_views} views")
    print(f"  Channels:        {args.render_channels}")
    print(f"  Seed:            {args.seed}")
    print(f"  Low VRAM:        {args.low_vram}")
    print("=" * 60)

    # ================================================================
    # 1. 加载 Pipeline
    # ================================================================
    print("\n[1/5] 加载 TRELLIS.2 pipeline...")
    t0 = time.time()

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        args.model_path, dino_local_path=args.dino_local_path,
    )
    pipe.low_vram = args.low_vram
    pipe.to(device)

    adapter = Trellis2RefAdapter(pipe, pipeline_type=args.pipeline_type)

    # 加载环境贴图
    print(f"  加载 EnvMap: {args.envmap_path}")
    envmap = load_envmap(args.envmap_path, device)

    print(f"  Pipeline 加载完成，用时: {time.time() - t0:.1f}s")

    # ================================================================
    # 2. 预处理图像
    # ================================================================
    print("\n[2/5] 预处理图像...")
    input_image = Image.open(args.input_image).convert("RGB")

    if not args.no_preprocess:
        image = pipe.preprocess_image(input_image)
    else:
        image = input_image

    image.save(os.path.join(args.output_dir, "input_preprocessed.png"))
    print(f"  图像尺寸: {image.size[0]}x{image.size[1]}")

    # ================================================================
    # 3. Stage 1 — 标准 TRELLIS.2 Rollout
    # ================================================================
    print("\n[3/5] Stage 1: 标准 TRELLIS.2 rollout...")
    t1 = time.time()

    seed_everything(args.seed)
    result = pipe.run(
        image,
        seed=args.seed,
        preprocess_image=False,  # 已预处理
        return_latent=True,
        pipeline_type=args.pipeline_type,
    )

    out_mesh_list, (shape_slat, tex_slat, res) = result

    rollout_time = time.time() - t1
    print(f"  Rollout 完成，用时: {rollout_time:.1f}s")
    print(f"  Shape SLat: feats {shape_slat.feats.shape}, "
          f"coords {shape_slat.coords.shape}")
    print(f"  Tex SLat:   feats {tex_slat.feats.shape}, "
          f"coords {tex_slat.coords.shape}")
    print(f"  Resolution: {res}")

    # 解码 baseline 并渲染多视角
    print("  解码 baseline mesh 并渲染多视角...")
    baseline_meshes = pipe.decode_latent(shape_slat, tex_slat, res)
    for i, m in enumerate(baseline_meshes):
        nv = m.vertices.shape[0] if hasattr(m, 'vertices') else '?'
        nf = m.faces.shape[0] if hasattr(m, 'faces') else '?'
        print(f"  Baseline mesh {i}: {nv} vertices, {nf} faces")
        if hasattr(m, 'simplify'):
            m.simplify(16777216)
    baseline_renders = render_multiview(
        baseline_meshes, envmap,
        resolution=args.render_resolution,
        nviews=args.num_views,
    )
    # 准备条件图（用于 grid 拼接）
    cond_pil = image.copy()
    if cond_pil.mode == "RGBA":
        bg = Image.new("RGBA", cond_pil.size, (255, 255, 255, 255))
        cond_pil = Image.alpha_composite(bg, cond_pil).convert("RGB")
    for i, r_dict in enumerate(baseline_renders):
        save_multiview_images(
            r_dict, args.output_dir, f"baseline_{i}",
            cond_image=cond_pil,
            channels=args.render_channels,
        )

    # ================================================================
    # 4. Stage 2 — FlowEdit Refine
    # ================================================================
    print("\n[4/5] Stage 2: FlowEdit Refine...")
    t2 = time.time()

    # 获取图像条件编码（cond + neg_cond）
    cond_resolution = flow_resolution  # 与 flow model 分辨率对齐
    cond_dict = pipe.get_cond([image], cond_resolution)
    cond = cond_dict["cond"].to(device)  # (B, S, C)
    neg_cond = cond_dict["neg_cond"].to(device)  # (B, S, C)
    print(f"  条件编码: cond {cond.shape}, neg_cond {neg_cond.shape}")

    # 当前 latent（FlowEdit 迭代更新）
    current_shape_slat = shape_slat
    current_tex_slat = tex_slat

    for round_idx in range(args.num_refine_rounds):
        round_label = (
            f" (Round {round_idx + 1}/{args.num_refine_rounds})"
            if args.num_refine_rounds > 1
            else ""
        )
        print(f"\n  === FlowEdit Refine{round_label} ===")

        # ---- Shape 阶段 ----
        if "shape" in args.refine_stages:
            print(f"\n  [Shape] 开始 FlowEdit...")

            # 获取 shape 阶段参数
            shape_params = adapter.get_sampler_params("shape")
            shape_steps = args.refine_steps or int(shape_params["steps"])
            shape_n_max = args.refine_n_max if args.refine_n_max is not None else shape_steps
            shape_cfg_tgt = (
                args.cfg_strength_tgt
                or float(shape_params["guidance_strength"])
            )
            shape_cfg_src = (
                args.cfg_strength_src
                if args.cfg_strength_src is not None
                else -shape_cfg_tgt
            )
            shape_sigma_min = get_sigma_min(pipe, "shape")
            shape_cfg_interval = adapter.get_cfg_interval("shape")

            print(f"    Steps: {shape_steps}, n_max: {shape_n_max}")
            print(f"    CFG tgt: +{shape_cfg_tgt}, CFG src: {shape_cfg_src}")
            print(f"    sigma_min: {shape_sigma_min}")
            print(f"    CFG interval: {shape_cfg_interval}")

            # 归一化 shape_slat → flow model 操作域
            shape_norm = normalize_slat(pipe, current_shape_slat, "shape")

            # low_vram: 将 flow model 加载到 GPU
            model_key = f"shape_slat_flow_model_{flow_resolution}"
            if pipe.low_vram:
                pipe.models[model_key].to(device)

            # FlowEdit refine
            shape_refined_norm = flowedit_refine_sparse(
                adapter=adapter,
                x_src_norm=shape_norm,
                cond=cond,
                neg_cond=neg_cond,
                stage="shape",
                resolution=flow_resolution,
                steps=shape_steps,
                n_max=shape_n_max,
                cfg_strength_tgt=shape_cfg_tgt,
                cfg_strength_src=shape_cfg_src,
                sigma_min=shape_sigma_min,
                cfg_interval=shape_cfg_interval,
                shape_cond=None,  # shape 阶段不需要 shape_cond
                device=device,
                seed=args.seed + round_idx,
                rescale_mode=args.rescale_mode,
            )

            # low_vram: 释放 flow model
            if pipe.low_vram:
                pipe.models[model_key].cpu()
                torch.cuda.empty_cache()

            # 反归一化
            current_shape_slat = denormalize_slat(
                pipe, shape_refined_norm, "shape"
            )

        # ---- Tex 阶段 ----
        if "tex" in args.refine_stages:
            print(f"\n  [Tex] 开始 FlowEdit...")

            # 获取 tex 阶段参数
            tex_params = adapter.get_sampler_params("tex")
            tex_steps = args.refine_steps or int(tex_params["steps"])
            tex_n_max = args.refine_n_max if args.refine_n_max is not None else tex_steps
            tex_cfg_tgt = (
                args.cfg_strength_tgt
                or float(tex_params["guidance_strength"])
            )
            tex_cfg_src = (
                args.cfg_strength_src
                if args.cfg_strength_src is not None
                else -tex_cfg_tgt
            )
            tex_sigma_min = get_sigma_min(pipe, "tex")
            tex_cfg_interval = adapter.get_cfg_interval("tex")

            print(f"    Steps: {tex_steps}, n_max: {tex_n_max}")
            print(f"    CFG tgt: +{tex_cfg_tgt}, CFG src: {tex_cfg_src}")
            print(f"    sigma_min: {tex_sigma_min}")
            print(f"    CFG interval: {tex_cfg_interval}")

            # 归一化 tex_slat
            tex_norm = normalize_slat(pipe, current_tex_slat, "tex")

            # 准备 shape_cond（归一化的 shape_slat，作为 tex model 的 concat 输入）
            tex_shape_cond = normalize_slat(pipe, current_shape_slat, "shape")

            # low_vram: 将 flow model 加载到 GPU
            model_key = f"tex_slat_flow_model_{flow_resolution}"
            if pipe.low_vram:
                pipe.models[model_key].to(device)

            # FlowEdit refine
            tex_refined_norm = flowedit_refine_sparse(
                adapter=adapter,
                x_src_norm=tex_norm,
                cond=cond,
                neg_cond=neg_cond,
                stage="tex",
                resolution=flow_resolution,
                steps=tex_steps,
                n_max=tex_n_max,
                cfg_strength_tgt=tex_cfg_tgt,
                cfg_strength_src=tex_cfg_src,
                sigma_min=tex_sigma_min,
                cfg_interval=tex_cfg_interval,
                shape_cond=tex_shape_cond,
                device=device,
                seed=args.seed + round_idx + 1000,
                rescale_mode=args.rescale_mode,
            )

            # low_vram: 释放 flow model
            if pipe.low_vram:
                pipe.models[model_key].cpu()
                torch.cuda.empty_cache()

            # 反归一化
            current_tex_slat = denormalize_slat(
                pipe, tex_refined_norm, "tex"
            )

        # 多轮时保存中间结果
        if args.num_refine_rounds > 1:
            print(f"\n  解码 Round {round_idx + 1} 并渲染多视角...")
            torch.cuda.empty_cache()
            round_meshes = pipe.decode_latent(
                current_shape_slat, current_tex_slat, res,
            )
            round_renders = render_multiview(
                round_meshes, envmap,
                resolution=args.render_resolution,
                nviews=args.num_views,
            )
            for i, r_dict in enumerate(round_renders):
                save_multiview_images(
                    r_dict, args.output_dir,
                    f"refined_round{round_idx + 1}_{i}",
                    cond_image=cond_pil,
                    channels=args.render_channels,
                )

    refine_time = time.time() - t2
    print(f"\n  FlowEdit 总用时: {refine_time:.1f}s")

    # ================================================================
    # 5. 解码最终结果并渲染多视角
    # ================================================================
    print("\n[5/5] 解码 refined 结果并渲染多视角...")
    t3 = time.time()

    # 释放 FlowEdit 循环占用的 CUDA 缓存，避免 nvdiffrast 显存不足
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    refined_meshes = pipe.decode_latent(
        current_shape_slat, current_tex_slat, res,
    )

    # 诊断: 打印 refined mesh 信息
    for i, m in enumerate(refined_meshes):
        nv = m.vertices.shape[0] if hasattr(m, 'vertices') else '?'
        nf = m.faces.shape[0] if hasattr(m, 'faces') else '?'
        print(f"  Refined mesh {i}: {nv} vertices, {nf} faces")
        # 对齐参考实现: nvdiffrast 有顶点数上限
        if hasattr(m, 'simplify'):
            m.simplify(16777216)

    refined_renders = render_multiview(
        refined_meshes, envmap,
        resolution=args.render_resolution,
        nviews=args.num_views,
    )
    for i, r_dict in enumerate(refined_renders):
        save_multiview_images(
            r_dict, args.output_dir, f"refined_{i}",
            cond_image=cond_pil,
            channels=args.render_channels,
        )

    decode_time = time.time() - t3
    print(f"  Decode + 渲染完成，用时: {decode_time:.1f}s")

    # ================================================================
    # 保存参数
    # ================================================================
    total_time = rollout_time + refine_time + decode_time
    with open(os.path.join(args.output_dir, "parameters.txt"), "w") as f:
        f.write("TRELLIS.2 FlowEdit Refine 参数:\n")
        f.write("=" * 60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nResolution: {res}\n")
        f.write(f"Flow Resolution: {flow_resolution}\n")
        f.write(f"Rollout time: {rollout_time:.1f}s\n")
        f.write(f"Render resolution: {args.render_resolution}\n")
        f.write(f"Num views: {args.num_views}\n")
        f.write(f"Render channels: {args.render_channels}\n")
        f.write(f"Refine time: {refine_time:.1f}s\n")
        f.write(f"Decode time: {decode_time:.1f}s\n")
        f.write(f"Total time: {total_time:.1f}s\n")

    print(f"\n{'=' * 60}")
    print(f"[SUCCESS] 所有结果保存到: {args.output_dir}")
    print(f"  Rollout:   {rollout_time:.1f}s")
    print(f"  FlowEdit:  {refine_time:.1f}s")
    print(f"  Decode:    {decode_time:.1f}s")
    print(f"  Total:     {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
