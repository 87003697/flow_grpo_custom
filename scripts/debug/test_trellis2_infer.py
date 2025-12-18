import os
import sys
from pathlib import Path
import cv2
import imageio
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ref_root = ROOT / "_reference_codes" / "TRELLIS.2"
sys.path.append(str(ref_root))              # 使 trellis2/o_voxel 可被直接 import
sys.path.append(str(ref_root / "o-voxel"))  # 形状同上，确保 o_voxel 包可见

# 环境变量需在导入 trellis2 之前设置，保证 cumesh 等依赖能找到系统 libstdc++
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

def load_envmap(hdri_path: Path) -> EnvMap:
    env_bgr = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)  # shape: (H, W, 3)
    env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)          # shape: (H, W, 3)
    env_tensor = torch.tensor(env_rgb, dtype=torch.float32, device="cuda")  # shape: (H, W, 3)
    return EnvMap(env_tensor)

def build_pipeline() -> Trellis2ImageTo3DPipeline:
    dino_local = ref_root / "pretrained_weights" / "dinov3-vitl16-pretrain-lvd1689m" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        dino_local_path=str(dino_local),
    )
    pipe.cuda()
    return pipe

def main():
    hdri_path = ref_root / "assets" / "hdri" / "forest.exr"
    image_path = ref_root / "assets" / "example_image" / "T.png"
    out_dir = ROOT / "outputs" / "trellis2"
    out_dir.mkdir(parents=True, exist_ok=True)

    envmap = load_envmap(hdri_path)
    pipeline = build_pipeline()

    image = Image.open(image_path)
    mesh = pipeline.run(image, num_samples=1, seed=42)[0]
    mesh.simplify(16_777_216)

    video_frames = render_utils.render_video(mesh, envmap=envmap)               # shape: (T, H, W, 3)
    video_frames = render_utils.make_pbr_vis_frames(video_frames)              # shape: (T, H, W, 3)
    imageio.mimsave(out_dir / "sample.mp4", video_frames, format="FFMPEG", fps=15)

if __name__ == "__main__":
    main()