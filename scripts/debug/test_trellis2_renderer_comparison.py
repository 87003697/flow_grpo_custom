"""
对比 PbrMeshRenderer 和 PbrVoxelRenderer 的渲染效果

输出:
  - comparison.mp4: 对比视频
  - comparison_frame0.png: 第一帧静态对比图

布局:
  | Mesh PBR Shaded | Voxel PBR Shaded |
  | Mesh Normal     | Voxel Normal     |
"""
import os
import sys
from pathlib import Path
import cv2
import imageio
import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ref_root = ROOT / "_reference_codes" / "TRELLIS.2"
sys.path.append(str(ref_root))              # 使 trellis2 可被直接 import
sys.path.append(str(ref_root / "o-voxel"))  # 确保 o_voxel 包可见

# 环境变量需在导入 trellis2 之前设置
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap, PbrMeshRenderer
import o_voxel

# 导入我们的 PbrVoxelRenderer
sys.path.append(str(ROOT))
from edit4shape.renderers.ovoxel_trellis2 import PbrVoxelRenderer


def load_envmap(hdri_path: Path) -> EnvMap:
    env_bgr = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)          # [H, W, 3]
    env_tensor = torch.tensor(env_rgb, dtype=torch.float32, device="cuda")  # [H, W, 3]
    return EnvMap(env_tensor)


def build_pipeline() -> Trellis2ImageTo3DPipeline:
    dino_local = ROOT / "pretrained_weights" / "dinov3-vitl16-pretrain-lvd1689m" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        dino_local_path=str(dino_local),
    )
    pipe.cuda()
    return pipe


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to uint8 image, handles various input shapes"""
    tensor = tensor.detach().cpu()
    
    # 移除多余的 batch 维度
    while tensor.dim() > 3:
        tensor = tensor.squeeze(0)
    
    if tensor.dim() == 2:
        # [H, W] -> [H, W, 3]
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        img = tensor.numpy()
    elif tensor.dim() == 3:
        if tensor.shape[0] in [1, 3, 4]:  # [C, H, W] 格式
            img = tensor.numpy().transpose(1, 2, 0)  # -> [H, W, C]
        else:  # [H, W, C] 格式
            img = tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    # 处理 NaN 值
    img = np.nan_to_num(img, nan=0.0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def render_comparison(
    mesh,  # MeshWithVoxel
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    envmap: EnvMap,
    resolution: int = 512,
):
    """
    对比两种渲染器的结果
    
    Returns:
        dict with 'mesh_pbr' and 'voxel_pbr' results
    """
    # 1. 原始 PbrMeshRenderer
    mesh_renderer = PbrMeshRenderer({
        'resolution': resolution,
        'near': 1,
        'far': 100,
        'ssaa': 2,
    })
    mesh_result = mesh_renderer.render(mesh, extrinsics, intrinsics, envmap)
    
    # 2. 新的 PbrVoxelRenderer
    voxel_renderer = PbrVoxelRenderer({
        'resolution': resolution,
        'near': 1,
        'far': 100,
        'ssaa': 1,
    })
    voxel_result = voxel_renderer.render(mesh, extrinsics, intrinsics, envmap)
    
    return {
        'mesh_pbr': mesh_result,
        'voxel_pbr': voxel_result,
    }


def create_comparison_image(mesh_result, voxel_result):
    """
    创建对比图像：
    | Mesh PBR Shaded | Voxel PBR Shaded |
    | Mesh Normal     | Voxel Normal     |
    """
    # 获取图像
    mesh_shaded = tensor_to_image(mesh_result['shaded'])
    voxel_shaded = tensor_to_image(voxel_result['shaded'])
    mesh_normal = tensor_to_image(mesh_result['normal'])
    voxel_normal = tensor_to_image(voxel_result['normal'])
    
    # 拼接
    row1 = np.concatenate([mesh_shaded, voxel_shaded], axis=1)
    row2 = np.concatenate([mesh_normal, voxel_normal], axis=1)
    comparison = np.concatenate([row1, row2], axis=0)
    
    return comparison


def main():
    hdri_path = ref_root / "assets" / "hdri" / "forest.exr"
    image_path = ref_root / "assets" / "example_image" / "T.png"
    out_dir = ROOT / "outputs" / "trellis2_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading environment map...")
    envmap = load_envmap(hdri_path)
    
    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Generating mesh from image...")
    image = Image.open(image_path)
    mesh = pipeline.run(image, num_samples=1, seed=42, pipeline_type='1024')[0]
    mesh.simplify(16_777_216)

    # 生成相机参数
    num_frames = 60
    yaws = -torch.linspace(0, 2 * 3.1415, num_frames) + np.pi/2
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    
    extrinsics_list, intrinsics_list = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws.tolist(), pitch.tolist(), rs=2, fovs=40
    )
    
    # 渲染对比视频
    print("Rendering comparison frames...")
    comparison_frames = []
    for i, (extr, intr) in enumerate(zip(extrinsics_list, intrinsics_list)):
        print(f"  Frame {i+1}/{num_frames}", end='\r')
        results = render_comparison(mesh, extr, intr, envmap, resolution=512)
        comparison = create_comparison_image(results['mesh_pbr'], results['voxel_pbr'])
        comparison_frames.append(comparison)
    print()
    
    # 保存视频
    print("Saving video...")
    imageio.mimsave(
        str(out_dir / "comparison.mp4"),
        comparison_frames,
        format="FFMPEG",
        fps=15
    )
    
    # 保存单帧对比图
    Image.fromarray(comparison_frames[0]).save(out_dir / "comparison_frame0.png")
    
    print(f"Results saved to {out_dir}")
    print("  - comparison.mp4: 对比视频")
    print("  - comparison_frame0.png: 第一帧静态图")


if __name__ == "__main__":
    main()

