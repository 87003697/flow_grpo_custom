#!/usr/bin/env python3
"""
验证 TRELLIS 阶段二（SLAT Flow）在 SDE 与 ODE 模式下的输出一致性（可视化对比）。

流程:
- 加载 TrellisStage2Pipeline（Stage1 冻结，仅训练 Stage2 SLatFlowModel）
- 预处理图像 → 编码图像条件
- Stage1 在线推理 → 稀疏结构坐标 coords ∈ ℤ^{N×4}
- Stage2 两次采样：
  - SDE: deterministic=False，记录对数概率轨迹
  - ODE: deterministic=True，确定性推进
- 均使用相同的初始噪声 SparseTensor（相同 coords 与 feats），确保对比公平
- 解码为 mesh，并分别渲染保存，供人工检查

注意:
- 代码不包含 try/except 或任何 fallback
- 关键 Tensor 运算处标注 shape 注释
- 默认输入图片: dataset/eval3d/images/feeding_squirrel.png
- 输出目录: outputs/trellis_sde_vs_ode/
"""

import sys
from pathlib import Path
from typing import Tuple, List

import torch
from PIL import Image

# 项目根路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 TRELLIS 封装与工具
from generators.trellis.pipeline import TrellisStage2Pipeline
from generators.trellis.utils import trellis_preprocess_image, normalize_slat_tensor

# 导入渲染器（用于保存对比图）
from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import (
    render_mesh_multiple_views,
)

# 导入 Flow 采样器（包含对数概率记录）
from flow_grpo.diffusers_patch.trellis_flow_with_logprob import (
    trellis_flow_euler_sampler_with_logprob,
)

# 引入 TRELLIS 稀疏张量类型
reference_path = PROJECT_ROOT / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(reference_path))
import trellis.modules.sparse as sp  # noqa: E402


def prepare_inputs(image_path: Path) -> Tuple[TrellisStage2Pipeline, torch.Tensor, dict]:
    """加载 pipeline、读图、预处理并编码条件，返回 (pipeline, coords, image_conds)。

    - coords: shape (N, 4)
    - image_conds: dict，键与官方 get_cond 一致（如 'cond', 'neg_cond'），将直接传入 Stage2
    """
    # 1) 加载 pipeline（Stage1 冻结，Stage2 训练）
    pipeline = TrellisStage2Pipeline()

    # 2) 迁移到 GPU（若可用）
    if torch.cuda.is_available():
        pipeline.cuda()

    # 3) 读图与预处理
    #    test_image: PIL.Image.Image  (H, W, 3)
    test_image = Image.open(str(image_path))
    preprocessed_image = trellis_preprocess_image(test_image)  # shape: (518, 518, 3)

    # 4) 图像条件编码（官方接口）
    image_conds = pipeline.prepare_image_conditions([preprocessed_image])

    # 5) Stage1 在线推理，生成稀疏结构坐标
    #    coords: torch.IntTensor, shape: (N, 4) = (batch_idx, x, y, z)
    coords = pipeline.forward_stage1(image_conds)

    return pipeline, coords, image_conds


def build_initial_noise(coords: torch.Tensor, in_channels: int, device: torch.device) -> sp.SparseTensor:
    """基于 coords 创建初始噪声 SparseTensor。

    - coords: torch.IntTensor, shape (N, 4)
    - feats: torch.FloatTensor, shape (N, C)
    """
    # feats: 随机正态噪声 (N, C)
    noise_feats = torch.randn(coords.shape[0], in_channels, device=device)  # (N, C)
    # initial_noise: 稀疏张量 (coords: (N, 4), feats: (N, C))
    initial_noise = sp.SparseTensor(coords=coords, feats=noise_feats)
    return initial_noise


def run_stage2(
    pipeline: TrellisStage2Pipeline,
    initial_noise: sp.SparseTensor,
    image_conds: dict,
    steps: int,
    deterministic: bool,
    guidance_scale: float = 1.0,
    seed: int = 42,
):
    """运行 Stage2（SLAT Flow）采样，返回 (final_slat, all_latents, all_log_probs)。

    - deterministic=True 为 ODE；False 为 SDE
    - SDE/ODE 共享相同 initial_noise（由外部构建并传入）
    - guidance_scale 默认 1.0（关闭 CFG）
    - 条件传参与官方一致：使用 {'cond': (B,P,C), 'neg_cond': (B,P,C)} 字典
    """
    slat_flow_model = pipeline.get_trainable_model()
    device = initial_noise.coords.device

    # 生成器仅用于 SDE 的随机过程复现（ODE 不使用）
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # 采样（记录每步 log_prob）
    cond_dict = image_conds  # 官方 get_cond 的原始输出

    final_slat, all_latents, all_log_probs, _ = trellis_flow_euler_sampler_with_logprob(
        model=slat_flow_model,
        noise=initial_noise,
        cond=cond_dict,
        steps=steps,
        sigma_min=0.002,
        rescale_t=1.0,
        generator=rng if not deterministic else None,
        deterministic=deterministic,
        guidance_scale=guidance_scale,
        neg_cond=None,
        verbose=True,
    )

    return final_slat, all_latents, all_log_probs


def decode_and_render(
    pipeline: TrellisStage2Pipeline,
    slat: sp.SparseTensor,
    save_dir: Path,
    tag: str,
    preset: str = "turntable",
) -> Path:
    """解码 SLAT → mesh 并渲染保存多视角图片，返回保存路径。

    - slat: SparseTensor，feats: (N, C)，coords: (N, 4)
    - 输出文件名: sde_{preset}.png / ode_{preset}.png
    """
    # 解码为 list[mesh]
    meshes = pipeline.decode_slat_to_mesh(slat)

    # 仅渲染第一个 mesh（通常返回单个）
    mesh = meshes[0]

    # 保存路径
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{tag}_{preset}.png"

    # 渲染
    result_path = render_mesh_multiple_views(
        mesh_trimesh=mesh,
        save_path=str(save_path),
        preset=preset,
        device=pipeline.device.type,
    )
    return Path(result_path)


def main():
    # 配置
    image_path = PROJECT_ROOT / "dataset/eval3d/images/feeding_squirrel.png"
    output_dir = PROJECT_ROOT / "outputs/trellis_sde_vs_ode"
    steps = 20

    # 数据准备
    pipeline, coords, image_conds = prepare_inputs(image_path=image_path)

    # 固定 coords 在当前设备
    coords = coords.to(device=pipeline.device)

    # 构建一次性的初始噪声，并在 SDE/ODE 之间复用
    slat_flow_model = pipeline.get_trainable_model()
    in_channels = getattr(slat_flow_model, "in_channels")
    initial_noise = build_initial_noise(coords=coords, in_channels=in_channels, device=pipeline.device)

    # SDE（随机）
    sde_slat, sde_latents, sde_log_probs = run_stage2(
        pipeline=pipeline,
        initial_noise=initial_noise,
        image_conds=image_conds,
        steps=steps,
        deterministic=False,
        guidance_scale=1.0,
        seed=42,
    )

    # ODE（确定性）
    ode_slat, ode_latents, ode_log_probs = run_stage2(
        pipeline=pipeline,
        initial_noise=initial_noise,
        image_conds=image_conds,
        steps=steps,
        deterministic=True,
        guidance_scale=1.0,
        seed=42,
    )

    # 规范化到官方尺度（与 sample_slat 后处理一致）
    normalization = pipeline.core_pipeline.slat_normalization
    sde_slat = normalize_slat_tensor(sde_slat, normalization)  # feats: (N, C)
    ode_slat = normalize_slat_tensor(ode_slat, normalization)

    # 解码与渲染
    sde_img = decode_and_render(
        pipeline=pipeline,
        slat=sde_slat,
        save_dir=output_dir,
        tag="sde",
        preset="turntable",
    )
    ode_img = decode_and_render(
        pipeline=pipeline,
        slat=ode_slat,
        save_dir=output_dir,
        tag="ode",
        preset="turntable",
    )

    print("\n对比图片已保存。请人工比对 SDE 与 ODE 是否一致：")
    print(f"SDE 图: {sde_img}")
    print(f"ODE 图: {ode_img}")


if __name__ == "__main__":
    main() 