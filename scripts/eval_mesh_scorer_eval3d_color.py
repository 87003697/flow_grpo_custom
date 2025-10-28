import os
import json
import csv
import glob
from typing import List, Dict, Any

import torch
import numpy as np
import trimesh
from PIL import Image

from reward_models.camera_rgb_scorer.scorer import CameraRGBScorer


def load_glb_mesh_as_obj(path: str) -> Any:
    mesh = trimesh.load(path, force='mesh')  # 形状: trimesh.Trimesh
    v = torch.from_numpy(np.asarray(mesh.vertices)).float()  # 形状: (V,3)
    f = torch.from_numpy(np.asarray(mesh.faces)).long()  # 形状: (F,3)
    return type('SimpleMesh', (), {'vertices': v, 'faces': f})  # 形状: 简单对象


def _cache_path_from_image(image_path_or_name: str, cache_dir: str, normal_resolution: int) -> str:
    stem = os.path.splitext(os.path.basename(image_path_or_name))[0]  # 形状: 标量
    dir_r = os.path.join(cache_dir, f"R{int(normal_resolution)}")  # 形状: 标量
    return os.path.join(dir_r, f"{stem}.png")  # 形状: 标量


def load_normal_pil_from_cache(image_path: str, cache_dir: str, normal_resolution: int) -> Image.Image:
    """从缓存目录读取法线 PNG（[0,255] 编码），返回 PIL。

    注: scorer 内部会将 PIL->Tensor 后再映射到 [-1,1]，与缓存编码一致。
    法线仍然用于相机搜索（几何对齐）。
    """
    p = _cache_path_from_image(image_path, cache_dir, normal_resolution)  # 形状: 标量
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到法线缓存: {p}")
    return Image.open(p).convert("RGB")  # 形状: PIL(R,R,3)


def _rotate_meshes_by_source_front(meshes: List[Any], source_front: str) -> None:
    if len(meshes) == 0:
        return
    src = str(source_front)  # 形状: 字符串
    if src == "+z":
        return

    suffix = 0  # 形状: 标量
    if len(src) > 0 and src[-1] in ("1", "2", "3"):
        suffix = int(src[-1])  # 形状: 标量
        base = src[:-1]  # 形状: 字符串
    else:
        base = src  # 形状: 字符串

    first_vertices = getattr(meshes[0], 'vertices', None)
    if not isinstance(first_vertices, torch.Tensor):
        if first_vertices is None and hasattr(meshes[0], 'v'):
            first_vertices = getattr(meshes[0], 'v')
        if not isinstance(first_vertices, torch.Tensor):
            raise TypeError("mesh.vertices 必须为 torch.Tensor")
    device = first_vertices.device  # 形状: 标量
    dtype = first_vertices.dtype  # 形状: 标量

    if base == "-z":
        T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=device, dtype=dtype)  # 形状: (3,3)
    elif base == "+x":
        T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)  # 形状: (3,3)
    elif base == "-x":
        T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=device, dtype=dtype)  # 形状: (3,3)
    elif base == "+y":
        T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=device, dtype=dtype)  # 形状: (3,3)
    elif base == "-y":
        T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=device, dtype=dtype)  # 形状: (3,3)
    else:
        T = torch.eye(3, device=device, dtype=dtype)  # 形状: (3,3)

    if suffix == 1:
        T = T @ torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=dtype)  # 形状: (3,3)
    elif suffix == 2:
        T = T @ torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device, dtype=dtype)  # 形状: (3,3)
    elif suffix == 3:
        T = T @ torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device, dtype=dtype)  # 形状: (3,3)

    for mesh in meshes:
        verts = getattr(mesh, 'vertices', None)
        if verts is None and hasattr(mesh, 'v'):
            verts = getattr(mesh, 'v')
        if not isinstance(verts, torch.Tensor):
            continue
        rotated = verts @ T  # 形状: (V,3)
        mesh.vertices = rotated  # 形状: (V,3)
        if hasattr(mesh, 'v'):
            mesh.v = rotated  # 形状: (V,3)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估 Camera RGB Scorer（基于 RGB 外观相似度）')
    parser.add_argument('--data_root', type=str, default='dataset/eval3d_hi3dgen')
    parser.add_argument('--rgb_resolution', type=int, default=256, help='RGB 渲染分辨率')
    parser.add_argument('--normal_resolution', type=int, default=518, help='法线分辨率（用于相机搜索）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder', type=str, default='dino_v2')
    parser.add_argument('--dino_v2_path', type=str, default='pretrained_weights/dinov2-giant')
    parser.add_argument('--dino_v3_path', type=str, default='pretrained_weights/dinov3-vitb14')
    parser.add_argument('--cache_dir', type=str, default='dataset/eval3d_hi3dgen/normals', help='法线缓存目录（用于相机搜索）')
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--vis_dir', type=str, default='logs/dino_vis_rgb')
    parser.add_argument('--cam_batch_size', type=int, default=64)
    parser.add_argument('--render_batch_size', type=int, default=8)
    parser.add_argument('--dino_batch_size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--output_csv', type=str, default='logs/eval3d_mesh_scores_rgb.csv')
    parser.add_argument('--camera_config', type=str, default='_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py')
    parser.add_argument('--camera_ckpt', type=str, default='')
    parser.add_argument('--source_front', type=str, default='+z')
    args = parser.parse_args()

    os.environ['FLOW_GRPO_DATA_DIR'] = args.data_root

    device = torch.device(args.device)
    cfg = {
        'rgb_resolution': args.rgb_resolution,  # RGB 渲染分辨率
        'cache_dir': args.cache_dir,
        'encoder': args.encoder,
        'dino_v2_path': args.dino_v2_path,
        'dino_v3_path': args.dino_v3_path,
        'save_vis': args.save_vis,
        'vis_dir': args.vis_dir,
        'cam_batch_size': args.cam_batch_size,
        'render_batch_size': args.render_batch_size,
        'dino_batch_size': args.dino_batch_size,
        'camera_config_py': args.camera_config,
        'camera_ckpt': args.camera_ckpt,
        'img_size': 518,  # VGGT 相机搜索固定尺寸
    }

    scorer = CameraRGBScorer(device=device, cfg=cfg)  # 形状: scorer

    # 构建同名文件列表
    img_dir = os.path.join(args.data_root, 'images')
    mesh_dir = os.path.join(args.data_root, 'meshes')
    names = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')])
    pairs = []
    for n in names:
        img_path = os.path.join(img_dir, f'{n}.png')
        mesh_candidates = [
            os.path.join(mesh_dir, f'{n}_textured_frame_000000.glb'),
            os.path.join(mesh_dir, f'{n}.glb'),
            os.path.join(mesh_dir, f'{n}.obj'),
            os.path.join(mesh_dir, f'{n}.ply'),
        ]
        mesh_path = next((p for p in mesh_candidates if os.path.exists(p)), None)
        if mesh_path is None:
            ply_matches = sorted(glob.glob(os.path.join(mesh_dir, f'{n}_*.ply')))
            if len(ply_matches) > 0:
                mesh_path = ply_matches[0]
        if mesh_path is None:
            continue
        pairs.append((n, img_path, mesh_path))

    if args.limit > 0:
        pairs = pairs[:args.limit]

    meshes: List[Any] = []
    images: List[Image.Image] = []  # 关键差异：加载原始 RGB 图像
    metadata: List[Dict[str, Any]] = []
    for name, img_path, mesh_path in pairs:
        m = load_glb_mesh_as_obj(mesh_path)  # 形状: 简单对象
        meshes.append(m)  # 形状: 追加
        
        # 关键差异：加载原始 RGB 图像（而不是 None）
        rgb_pil = Image.open(img_path).convert("RGB")  # 形状: PIL
        images.append(rgb_pil)  # 形状: 追加
        
        # 仍然需要法线（用于相机搜索）
        normal_pil = load_normal_pil_from_cache(img_path, args.cache_dir, args.normal_resolution)  # 形状: PIL
        metadata.append({
            'image_path': img_path,
            'image_name': f'{name}.png',
            'normal_pil': normal_pil,  # 用于相机搜索
        })  # 形状: 元数据

    _rotate_meshes_by_source_front(meshes, args.source_front)

    result = scorer.compute_scores(meshes, images, metadata)
    if isinstance(result, tuple):
        scores, _grouped_meta = result
    else:
        scores = result

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'image', 'mesh', 'rgb_score'])
        for (name, img_path, mesh_path), sc in zip(pairs, scores):
            writer.writerow([name, img_path, mesh_path, f'{float(sc):.6f}'])
        if len(scores) > 0:
            mean_score = float(np.mean(scores))  # 形状: 标量
            writer.writerow(['mean', '', '', f'{mean_score:.6f}'])

    print(json.dumps({
        'scorer_type': 'camera_rgb_scorer',  # 标识使用的评分器
        'num_samples': len(scores),  # 形状: 标量
        'mean_score': float(np.mean(scores)) if len(scores) > 0 else None,  # 形状: 可空标量
        'output_csv': args.output_csv,  # 形状: 字符串
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
