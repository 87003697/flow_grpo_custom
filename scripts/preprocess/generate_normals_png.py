#!/usr/bin/env python3
import os
import argparse
import importlib
import sys
from typing import List

import torch
import numpy as np
from tqdm import tqdm

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # 形状: 项目根
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)  # 形状: 添加到模块路径

from PIL import Image


def list_images(img_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")  # 形状: 元组
    names = [n for n in os.listdir(img_dir) if n.lower().endswith(exts)]  # 形状: 列表
    names.sort()  # 形状: 就地排序
    return [os.path.join(img_dir, n) for n in names]  # 形状: 列表


def load_predictor(module_path: str, model_dir: str, device: torch.device) -> torch.nn.Module:
    mod = importlib.import_module(module_path)  # 形状: 模块
    predictor = mod.create_predictor(model_dir, device)  # 形状: 模型（前向: (S,3,H,W)->(S,3,H,W), ∈[-1,1]）
    return predictor  # 形状: 模型


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Stable Normal 预测器批量生成法线 PNG 缓存 ([-1,1]→[0,255])")
    parser.add_argument("--input_dir", required=True, help="输入图像目录（直接包含若干 .png/.jpg/... 文件）")
    parser.add_argument("--output_dir", required=True, help="输出缓存根目录（脚本会写入 <output_dir>/R{R}/*.png）")
    parser.add_argument("--resolution", type=int, default=512, help="输出分辨率 R")
    parser.add_argument("--device", default="cuda", help="设备 cuda/cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小 S")
    parser.add_argument("--predictor_module", default="reward_models.camera_normal_scorer.normal_io.stable_normal_predictor", help="可导入模块路径，需提供 create_predictor(model_dir, device)")
    parser.add_argument("--model_dir", required=True, help="Stable Normal 权重目录（如 pretrained_weights/stable-normal）")
    parser.add_argument("--ts_path", default=None, help="可选：直接指定 TorchScript 文件路径，优先于 --model_dir")
    args = parser.parse_args()

    img_dir = str(args.input_dir)  # 形状: 标量
    assert os.path.isdir(img_dir), f"未找到输入图像目录: {img_dir}"

    cache_dir = str(args.output_dir)  # 形状: 标量
    R = int(args.resolution)  # 形状: 标量
    device = torch.device(args.device)  # 形状: 设备
    bs = int(args.batch_size)  # 形状: 标量

    out_dir = os.path.join(cache_dir, f"R{R}")  # 形状: 标量
    os.makedirs(out_dir, exist_ok=True)

    img_paths = list_images(img_dir)  # 形状: 长度 N
    assert len(img_paths) > 0, f"未在 {img_dir} 找到图像"

    if args.ts_path is not None:
        mod = importlib.import_module(args.predictor_module)  # 形状: 模块
        predictor = mod.create_predictor_from_ts_path(args.ts_path, device)  # 形状: 模型
    else:
        predictor = load_predictor(args.predictor_module, args.model_dir, device)  # 形状: 模型
    predictor.eval()  # 形状: 无返回

    for s in tqdm(range(0, len(img_paths), bs), desc="Generating normals"):
        e = min(len(img_paths), s + bs)  # 形状: 标量
        batch_paths = img_paths[s:e]  # 形状: 长度 b

        for p in tqdm(batch_paths, desc="Processing images"):
            img = Image.open(p)  # 形状: PIL.Image
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))  # 形状: PIL.Image
                img = Image.alpha_composite(bg, img).convert("RGB")  # 形状: PIL.Image
            else:
                img = img.convert("RGB")  # 形状: PIL.Image

            arr = np.array(img)  # 形状: (H,W,3)
            ten = torch.from_numpy(arr).to(torch.float32).permute(2, 0, 1)  # 形状: (3,H,W)
            img01 = (ten / 255.0).clamp(0.0, 1.0)  # 形状: (3,H,W)
            img01_b = img01.unsqueeze(0).to(device, non_blocking=True)  # 形状: (1,3,H,W)

            normals = predictor(img01_b)  # 形状: (1,3,H,W)，值域期望 ∈[-1,1]
            normals = torch.nn.functional.interpolate(  # 形状: (1,3,R,R)
                normals, size=(R, R), mode="bilinear", align_corners=False
            )
            normals = normals.clamp(-1.0, 1.0)  # 形状: (1,3,R,R)

            n = normals[0].detach().cpu()  # 形状: (3,R,R)
            img01_out = ((n.clamp(-1, 1) + 1.0) * 0.5).clamp(0.0, 1.0)  # 形状: (3,R,R)
            img8 = (img01_out * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()  # 形状: (R,R,3)

            stem = os.path.splitext(os.path.basename(p))[0]  # 形状: 标量
            out_path = os.path.join(out_dir, f"{stem}.png")  # 形状: 标量
            Image.fromarray(img8).save(out_path)

    print(f"✅ 生成完成: {len(img_paths)} files -> {out_dir}")


if __name__ == "__main__":
    main()


