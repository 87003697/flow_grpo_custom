#!/usr/bin/env python3
"""
TRELLIS SDE vs ODE 对比脚本
=================================================

只使用 edit4shape/generators/trellis 中的代码进行对比测试：
- ODE: scheduler.step() (Euler)
- SDE: scheduler.sde_step() (带噪声)

功能：
- 共享 Stage1 坐标生成
- 分别使用 SDE 和 ODE 进行 Stage2 采样
- 渲染统一视角的多视图图像
- 生成对比拼图并保存

示例：
  python scripts/debug/test_trellis_SDE-vs-ODE.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/sde_vs_ode \
    --steps 50 --guidance 3.0 --seed 777
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
import math
import numpy as np

# 设置环境变量避免加载问题
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加 TRELLIS 参考代码路径
TRELLIS_ROOT = os.path.join(PROJECT_ROOT, "_reference_codes", "TRELLIS")
if TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS_ROOT)

# 使用 edit4shape/generators 的代码
from edit4shape.generators.trellis.pipeline_adapter import TrellisRefAdapter, build_pipeline_from_reference
from edit4shape.generators.trellis.scheduler import TrellisFlowScheduler

# 使用 trellis 参考代码的稀疏结构和渲染
from trellis.modules.sparse import SparseTensor
from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="TRELLIS 预训练模型目录")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径")
    ap.add_argument("--out", type=str, default="outputs/test_runs/sde_vs_ode", help="输出目录")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=50, help="Stage2 采样步数")
    ap.add_argument("--guidance", type=float, default=3.0, help="CFG 引导系数")
    ap.add_argument("--rescale_t", type=float, default=1.0, help="时间重标")
    ap.add_argument("--seed", type=int, default=777, help="随机种子")
    ap.add_argument("--render_resolution", type=int, default=512, help="渲染分辨率")
    ap.add_argument("--num_views", type=int, default=4, help="统一视角数量")
    ap.add_argument("--camera_r", type=float, default=2.0, help="相机距离")
    ap.add_argument("--camera_fov", type=float, default=40.0, help="视场角 (度)")
    ap.add_argument("--camera_pitch", type=float, default=0.3, help="相机俯仰角 (弧度)")
    ap.add_argument("--noise_level", type=float, default=0.7, help="SDE 噪声强度")
    ap.add_argument("--sde_type", type=str, default="sde", choices=["sde", "cps"], help="SDE 类型")
    return ap.parse_args()


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def generate_uniform_cameras(
    num_views: int,
    r: float = 2.0,
    fov: float = 40.0,
    pitch: float = 0.3,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成均匀分布的相机视角。"""
    yaws = [2 * math.pi * i / num_views for i in range(num_views)]  # [V]
    pitchs = [pitch] * num_views  # [V]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    return extrinsics, intrinsics


def predict_velocity_with_cfg(
    adapter: TrellisRefAdapter,
    x_t: SparseTensor,
    t_val: float,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """预测速度场，支持 CFG 引导"""
    model = adapter.pipe.models["slat_flow_model"]
    batch_size = cond_emb.shape[0]  # ()
    t_scaled = torch.full((batch_size,), t_val * 1000, device=device, dtype=torch.float32)  # (B,)
    
    # 无条件预测
    uncond_v = model(x_t, t_scaled, uncond_emb)  # SparseTensor
    
    # 条件预测
    cond_v = model(x_t, t_scaled, cond_emb)  # SparseTensor
    
    # CFG 混合
    t_norm = t_val
    if cfg_min <= t_norm <= cfg_max:
        cfg_weight = guidance
    else:
        cfg_weight = 1.0
    
    guided_feats = uncond_v.feats + cfg_weight * (cond_v.feats - uncond_v.feats)  # (N, C)
    return SparseTensor(coords=x_t.coords, feats=guided_feats)


def rollout_ode(
    adapter: TrellisRefAdapter,
    x_t: SparseTensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    scheduler: TrellisFlowScheduler,
    guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
) -> SparseTensor:
    """ODE 采样循环"""
    steps = list(scheduler.timesteps)[:-1]
    
    for t in tqdm(steps, desc="ODE Rollout", leave=False):
        t_val = float(t.item())
        
        with torch.no_grad():
            velocity = predict_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                guidance, cfg_min, cfg_max, device,
            )  # SparseTensor
        
        # ODE 步进
        x_t = scheduler.step(velocity, t_val, x_t).prev_sample  # SparseTensor
    
    return x_t


def rollout_sde(
    adapter: TrellisRefAdapter,
    x_t: SparseTensor,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    scheduler: TrellisFlowScheduler,
    guidance: float,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
    generator: torch.Generator,
    noise_level: float = 0.7,
    sde_type: str = 'sde',
) -> SparseTensor:
    """SDE 采样循环"""
    steps = list(scheduler.timesteps)[:-1]
    
    for t in tqdm(steps, desc="SDE Rollout", leave=False):
        t_val = float(t.item())
        
        with torch.no_grad():
            velocity = predict_velocity_with_cfg(
                adapter, x_t, t_val, cond_emb, uncond_emb,
                guidance, cfg_min, cfg_max, device,
            )  # SparseTensor
        
        # SDE 步进
        x_t, log_prob, _, _, _ = scheduler.sde_step(
            noise_pred=velocity,
            t=t_val,
            latents=x_t,
            noise_level=noise_level,
            generator=generator,
            sde_type=sde_type,
            return_sqrt_dt=True,
        )  # x_t: SparseTensor
    
    return x_t


def render_mesh_multiview(
    mesh: Any,
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    resolution: int = 512,
) -> Dict[str, List[np.ndarray]]:
    """渲染 mesh 的多视角图像。"""
    render_out = render_frames(
        mesh, extrinsics, intrinsics,
        options={'resolution': resolution, 'bg_color': (0, 0, 0)},
        verbose=False,
    )
    return render_out


def concat_images_horizontally(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)  # (H, W*V, C)


def concat_images_vertically(images: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)  # (H*N, W, C)


def add_label_to_image(image: np.ndarray, label: str) -> np.ndarray:
    """在图像左上角添加标签文字。"""
    from PIL import ImageDraw, ImageFont
    
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((10, 10), label, font=font)
    padding = 5
    draw.rectangle(
        [text_bbox[0] - padding, text_bbox[1] - padding,
         text_bbox[2] + padding, text_bbox[3] + padding],
        fill=(0, 0, 0)
    )
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    assert os.path.isdir(args.model_path), f"模型目录不存在: {args.model_path}"
    assert os.path.isfile(args.image), f"图像文件不存在: {args.image}"
    
    os.makedirs(args.out, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # ========== 加载模型（使用 edit4shape adapter）==========
    print(f"[INFO] 加载模型: {args.model_path}")
    
    # 构造一个简单的配置对象用于加载
    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = args.model_path
    cfg.verbose = True
    
    # 使用 MockAccelerator
    class MockAccelerator:
        pass
    mock_accelerator = MockAccelerator()
    mock_accelerator.device = device
    
    adapter = build_pipeline_from_reference(cfg, mock_accelerator, device=device)
    pipe = adapter.pipe  # 底层的 TrellisImageTo3DPipeline
    
    # ========== 读取输入图像 ==========
    print(f"[INFO] 读取输入图像: {args.image}")
    img = load_image(args.image)
    img.save(os.path.join(args.out, "input.png"))
    
    # ========== 条件编码（使用 adapter）==========
    print(f"[INFO] 条件编码...")
    cond_dict = adapter.prepare_image_conditions([img])  # {'cond': (B,P,C), 'neg_cond': (B,P,C)}
    cond_emb = cond_dict['cond'].to(device)  # (B, P, C)
    uncond_emb = cond_dict.get('neg_cond', torch.zeros_like(cond_emb)).to(device)  # (B, P, C)
    
    # ========== Stage1: 稀疏结构生成（使用 adapter）==========
    print(f"[INFO] Stage1: 生成稀疏结构...")
    coords = adapter.dense_sampling(cond_dict, steps=None)  # (N, 4)
    print(f"  生成坐标数量: {coords.shape[0]}")
    
    # ========== 获取 Stage2 参数 ==========
    _, _, slat_steps_default, slat_guidance_default, slat_rescale_t_default, _ = adapter.get_sampler_runtime_params()
    slat_steps = args.steps
    slat_guidance = args.guidance
    slat_rescale_t = args.rescale_t
    slat_params = pipe.slat_sampler_params
    cfg_interval = slat_params.get("cfg_interval", (0.0, 1.0))
    cfg_min, cfg_max = cfg_interval
    
    in_channels = pipe.models['slat_flow_model'].in_channels  # int
    
    # ========== 生成统一的相机视角 ==========
    print(f"[INFO] 生成 {args.num_views} 个统一相机视角...")
    extrinsics, intrinsics = generate_uniform_cameras(
        num_views=args.num_views,
        r=args.camera_r,
        fov=args.camera_fov,
        pitch=args.camera_pitch,
    )
    
    # ========== ODE 采样 ==========
    print(f"[INFO] Stage2 ODE 采样 (deterministic)...")
    g_ode = torch.Generator(device=device)
    g_ode.manual_seed(args.seed)
    
    # 初始化 latent（使用 adapter）
    x_t_ode = adapter.init_sparse_latents(
        coords=coords,
        in_channels=in_channels,
        generator=g_ode,
    )  # SparseTensor
    
    # 初始化 scheduler（使用 adapter）
    scheduler_ode = adapter.scheduler()
    scheduler_ode.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    
    # ODE rollout
    x_0_ode = rollout_ode(
        adapter, x_t_ode, cond_emb, uncond_emb,
        scheduler_ode, slat_guidance, cfg_min, cfg_max, device,
    )
    
    # 反归一化
    norm = pipe.slat_normalization
    std = torch.tensor(norm['std'])[None].to(device)  # (1, C)
    mean = torch.tensor(norm['mean'])[None].to(device)  # (1, C)
    denorm_feats_ode = x_0_ode.feats * std + mean  # (N, C)
    x_0_ode_denorm = SparseTensor(coords=x_0_ode.coords, feats=denorm_feats_ode)
    
    # 解码（使用 adapter）
    decoded_ode = adapter.decode(x_0_ode_denorm, formats=['mesh'])
    meshes_ode = decoded_ode['mesh']
    print(f"  ODE: 生成 {len(meshes_ode)} 个 mesh")
    
    # ========== SDE 采样 ==========
    print(f"[INFO] Stage2 SDE 采样 (stochastic, noise_level={args.noise_level})...")
    g_sde = torch.Generator(device=device)
    g_sde.manual_seed(args.seed)
    
    # 初始化 latent（使用相同种子保证初始噪声一致）
    g_sde_init = torch.Generator(device=device)
    g_sde_init.manual_seed(args.seed)
    x_t_sde = adapter.init_sparse_latents(
        coords=coords,
        in_channels=in_channels,
        generator=g_sde_init,
    )  # SparseTensor
    
    # 初始化 scheduler
    scheduler_sde = adapter.scheduler()
    scheduler_sde.set_timesteps(slat_steps, device=device, rescale_t=slat_rescale_t)
    
    # SDE rollout
    x_0_sde = rollout_sde(
        adapter, x_t_sde, cond_emb, uncond_emb,
        scheduler_sde, slat_guidance, cfg_min, cfg_max, device,
        generator=g_sde,
        noise_level=args.noise_level,
        sde_type=args.sde_type,
    )
    
    # 反归一化
    denorm_feats_sde = x_0_sde.feats * std + mean  # (N, C)
    x_0_sde_denorm = SparseTensor(coords=x_0_sde.coords, feats=denorm_feats_sde)
    
    # 解码
    decoded_sde = adapter.decode(x_0_sde_denorm, formats=['mesh'])
    meshes_sde = decoded_sde['mesh']
    print(f"  SDE: 生成 {len(meshes_sde)} 个 mesh")
    
    # ========== 渲染对比图 ==========
    print(f"[INFO] 渲染统一视角对比图...")
    
    for mesh_idx in range(len(meshes_ode)):
        mesh_ode = meshes_ode[mesh_idx]
        mesh_sde = meshes_sde[mesh_idx]
        
        # 渲染 ODE mesh
        render_ode = render_mesh_multiview(
            mesh_ode, extrinsics, intrinsics, resolution=args.render_resolution
        )  # dict with 'normal': list[V] of (H,W,3)
        
        # 渲染 SDE mesh
        render_sde = render_mesh_multiview(
            mesh_sde, extrinsics, intrinsics, resolution=args.render_resolution
        )  # dict with 'normal': list[V] of (H,W,3)
        
        # 拼接每个视角的 ODE 和 SDE 对比
        comparison_views = []
        for v_idx in range(args.num_views):
            img_ode = render_ode['normal'][v_idx]  # (H,W,3)
            img_sde = render_sde['normal'][v_idx]  # (H,W,3)
            
            # 添加标签
            img_ode_labeled = add_label_to_image(img_ode, f"ODE View-{v_idx}")
            img_sde_labeled = add_label_to_image(img_sde, f"SDE View-{v_idx}")
            
            # 垂直拼接 ODE 和 SDE
            view_comparison = concat_images_vertically([img_ode_labeled, img_sde_labeled])
            comparison_views.append(view_comparison)
        
        # 水平拼接所有视角
        full_comparison = concat_images_horizontally(comparison_views)
        
        # 保存对比图
        comparison_path = os.path.join(args.out, f"comparison_mesh_{mesh_idx}.png")
        Image.fromarray(full_comparison).save(comparison_path)
        print(f"  保存对比图: {comparison_path}")
        
        # 单独保存每个视角
        for v_idx in range(args.num_views):
            ode_view_path = os.path.join(args.out, f"ode_mesh_{mesh_idx}_view_{v_idx}.png")
            Image.fromarray(render_ode['normal'][v_idx]).save(ode_view_path)
            
            sde_view_path = os.path.join(args.out, f"sde_mesh_{mesh_idx}_view_{v_idx}.png")
            Image.fromarray(render_sde['normal'][v_idx]).save(sde_view_path)
        
        # 保存 ODE 和 SDE 的水平拼接图
        ode_row = concat_images_horizontally(render_ode['normal'])
        sde_row = concat_images_horizontally(render_sde['normal'])
        
        ode_row_labeled = add_label_to_image(ode_row, "ODE (Deterministic)")
        sde_row_labeled = add_label_to_image(sde_row, "SDE (Stochastic)")
        
        Image.fromarray(ode_row_labeled).save(os.path.join(args.out, f"ode_mesh_{mesh_idx}_all_views.png"))
        Image.fromarray(sde_row_labeled).save(os.path.join(args.out, f"sde_mesh_{mesh_idx}_all_views.png"))
        
        # 导出 mesh（使用 adapter）
        adapter.export_mesh_obj(mesh_ode, os.path.join(args.out, f"ode_mesh_{mesh_idx}.obj"))
        adapter.export_mesh_obj(mesh_sde, os.path.join(args.out, f"sde_mesh_{mesh_idx}.obj"))
    
    # ========== 汇总统计 ==========
    print(f"\n{'='*60}")
    print(f"[完成] SDE vs ODE 对比测试")
    print(f"{'='*60}")
    print(f"  输入图像: {args.image}")
    print(f"  输出目录: {args.out}")
    print(f"  采样步数: {args.steps}")
    print(f"  引导系数: {args.guidance}")
    print(f"  随机种子: {args.seed}")
    print(f"  视角数量: {args.num_views}")
    print(f"  ODE mesh 数量: {len(meshes_ode)}")
    print(f"  SDE mesh 数量: {len(meshes_sde)}")
    print(f"  SDE noise_level: {args.noise_level}")
    print(f"  SDE type: {args.sde_type}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
