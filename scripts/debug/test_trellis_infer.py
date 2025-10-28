#!/usr/bin/env python3
"""
TRELLIS 推理与导出脚本（工作路径版本）
=================================================

功能：
- 从单张图像出发，使用仓库内 TRELLIS Stage2 管线完成：
  1) 图像条件编码（patch 级 cond/neg_cond）
  2) Stage1 稀疏结构在线生成
  3) Stage2 SLAT Flow 采样（支持 SDE/ODE）
  4) 网格导出（.ply）

约束：
- 使用当前工作路径下的实现（generators/trellis 与 flow_grpo/diffusers_patch）。
- 不使用 try/except 或任何 fallback。
- 每行张量运算附形状注释。

示例：
  python scripts/debug/test_trellis_infer.py \
    --model_path pretrained_weights/TRELLIS-image-large \
    --image dataset/eval3d_hunyuan3d/images/004.png \
    --out outputs/test_runs/trellis_validation \
    --steps 50 --guidance 3.0 --sigma_min 0.002 --rescale_t 1.0 \
    --candidates 2 --seed 777 --sde
"""

import os
import argparse
from typing import List, Tuple

import torch
from PIL import Image

from generators.trellis.pipeline import TrellisStage2Pipeline
from flow_grpo.diffusers_patch.trellis_pipeline_with_logprob import TrellisPipelineWithLogProb


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="TRELLIS 预训练模型目录（包含权重与配置）")
    ap.add_argument("--image", type=str, required=True, help="输入图像路径")
    ap.add_argument("--out", type=str, default="outputs/test_runs/trellis_validation", help="输出目录")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=50, help="Stage2 采样步数")
    ap.add_argument("--guidance", type=float, default=3.0, help="CFG 引导系数 (>1.0 启用)" )
    ap.add_argument("--sigma_min", type=float, default=0.002, help="最小噪声尺度（SDE sigma 域）")
    ap.add_argument("--rescale_t", type=float, default=1.0, help="时间重标（与 TRELLIS t∈[0,1000] 配合）")
    ap.add_argument("--candidates", type=int, default=1, help="每图候选数量")
    ap.add_argument("--seed", type=int, default=777, help="主随机种子")
    ap.add_argument("--sde", action="store_true", help="使用 SDE（不加则 ODE）")
    ap.add_argument("--output_type", type=str, default="trimesh", choices=["trimesh", "kiui", "latent"], help="输出类型")
    ap.add_argument("--deterministic", action="store_true", help="使后端确定性（不影响 SDE/ODE 选择）")
    return ap.parse_args()


def build_pipeline(model_path: str, device: torch.device) -> TrellisStage2Pipeline:
    pipe = TrellisStage2Pipeline(model_path=model_path, verbose=False)
    pipe.to(device)
    return pipe


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def run_infer(
    pipe: TrellisStage2Pipeline,
    image_path: str,
    out_dir: str,
    steps: int,
    guidance: float,
    sigma_min: float,
    rescale_t: float,
    candidates: int,
    seed: int,
    use_sde: bool,
    output_type: str,
) -> Tuple[List, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    os.makedirs(out_dir, exist_ok=True)

    # 读图
    img = load_image(image_path)

    # 条件编码（patch 级）
    cond_dict = pipe.prepare_image_conditions([img])  # cond: (B,P,C), neg_cond: (B,P,C)

    # 主生成器
    g = torch.Generator(device=pipe.device)
    g.manual_seed(int(seed))

    # Stage2 采样参数（含我们已改为 sigma 域的 SDE）
    slat_sampler_params = {
        "sigma_min": float(sigma_min),  # 标量
        "rescale_t": float(rescale_t),  # 标量（仍用于 TRELLIS 时间序列重标）
    }

    # ODE/SDE 切换：通过 deterministic 标志
    deterministic = (not use_sde)

    wrapper = TrellisPipelineWithLogProb(pipe)
    coords_list_eval, _, _ = wrapper.stage1_with_logprob(
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g,
        deterministic=bool(deterministic),
        sparse_structure_sampler_params={},
        stage1_cond_dict=cond_dict,
        num_candidates=int(candidates),
        verbose=True,
    )
    meshes, all_latents, all_log_probs, all_kl = wrapper.stage2_with_logprob(
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g,
        output_type=str(output_type),
        kl_reward=0.0,
        deterministic=bool(deterministic),
        sparse_structure_sampler_params={},
        slat_sampler_params=slat_sampler_params,
        stage1_cond_dict=cond_dict,
        num_candidates=int(candidates),
        verbose=True,
        coords_list=coords_list_eval,
    )

    # 简要校验（张量形状）
    if len(all_log_probs) > 0:
        lp_cat = torch.stack(all_log_probs)  # (K*T,1) 或 (K*T,) 取决于上游聚合
        _ = lp_cat.view(-1)  # (K*T,)
    if len(all_latents) > 0:
        x0 = all_latents[0]
        if isinstance(x0, torch.Tensor):
            _ = x0.shape  # (N,C) 或其他

    # 导出网格（Trimesh 或 KiuiMesh，output_type 已选择）
    for i, m in enumerate(meshes):
        if output_type == "trimesh":
            import trimesh
            if isinstance(m, trimesh.Trimesh):
                out_path = os.path.join(out_dir, f"mesh_{i}.ply")
                m.export(out_path)
        elif output_type == "kiui":
            # KiuiMesh: 直接转 Trimesh 存盘（点面张量转 numpy）
            from kiui.mesh import Mesh as KiuiMesh
            if isinstance(m, KiuiMesh):
                v = m.v  # (V,3)
                f = m.f  # (F,3)
                v_np = v.detach().cpu().numpy()  # (V,3)
                f_np = f.detach().cpu().numpy()  # (F,3)
                import trimesh
                tri = trimesh.Trimesh(vertices=v_np, faces=f_np)
                out_path = os.path.join(out_dir, f"mesh_{i}.ply")
                tri.export(out_path)
        else:
            # latent：不导出
            pass

    return meshes, all_latents, all_log_probs, all_kl


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    assert os.path.isdir(args.model_path), f"模型目录不存在: {args.model_path}"
    assert os.path.isfile(args.image), f"图像文件不存在: {args.image}"

    pipe = build_pipeline(args.model_path, device)

    meshes, latents, log_probs, kl = run_infer(
        pipe=pipe,
        image_path=args.image,
        out_dir=args.out,
        steps=int(args.steps),
        guidance=float(args.guidance),
        sigma_min=float(args.sigma_min),
        rescale_t=float(args.rescale_t),
        candidates=int(args.candidates),
        seed=int(args.seed),
        use_sde=bool(args.sde),
        output_type=str(args.output_type),
    )

    # 简要统计
    n_mesh = len(meshes)
    n_lp = len(log_probs)
    n_lat = len(latents)
    print(f"[DONE] TRELLIS 推理完成: meshes={n_mesh} latents={n_lat} log_probs={n_lp} 输出目录={args.out}")


if __name__ == "__main__":
    main()


