import os
import json
import csv
import glob
from typing import List, Dict, Any

import torch
import numpy as np

from reward_models.camera_normal_scorer.scorer import CameraNormalScorer
from scripts.eval.utils_camera_normal import (
    load_glb_mesh_as_obj,
    load_normal_pil_from_cache,
    _rotate_meshes_by_source_front,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset/eval3d_hi3dgen')
    parser.add_argument('--normal_resolution', type=int, default=518)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder', type=str, default='dino_v2')
    parser.add_argument('--dino_v2_path', type=str, default='pretrained_weights/dinov2-base')
    parser.add_argument('--dino_v3_path', type=str, default='pretrained_weights/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--cache_dir', type=str, default='dataset/eval3d_hi3dgen/normals')
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--vis_dir', type=str, default='logs/dino_vis')
    parser.add_argument('--batch_size', type=int, default=16, help='统一批大小，作用于所有阶段')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--output_csv', type=str, default='logs/eval3d_mesh_scores.csv')
    parser.add_argument('--camera_config', type=str, default='_reference_codes/VGGTObj/training/config/camera_search_seven_view_fixed.py')
    parser.add_argument('--camera_ckpt', type=str, default='')
    parser.add_argument('--source_front', type=str, default='+z')
    args = parser.parse_args()

    os.environ['FLOW_GRPO_DATA_DIR'] = args.data_root

    device = torch.device(args.device)
    cfg = {
        'normal_resolution': args.normal_resolution,
        'cache_dir': args.cache_dir,
        'encoder': args.encoder,
        'dino_v2_path': args.dino_v2_path,
        'dino_v3_path': args.dino_v3_path,
        'save_vis': args.save_vis,
        'vis_dir': args.vis_dir,
        'batch_size': args.batch_size,
        'camera_config_py': args.camera_config,
        'camera_ckpt': args.camera_ckpt,
    }

    scorer = CameraNormalScorer(device=device, cfg=cfg)  # 形状: scorer

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
    images: List[Any] = []
    metadata: List[Dict[str, Any]] = []
    for name, img_path, mesh_path in pairs:
        m = load_glb_mesh_as_obj(mesh_path)  # 形状: 简单对象
        meshes.append(m)  # 形状: 追加
        images.append(None)  # 形状: 占位
        normal_pil = load_normal_pil_from_cache(img_path, args.cache_dir, args.normal_resolution)  # 形状: PIL
        metadata.append({'image_path': img_path, 'image_name': f'{name}.png', 'normal_pil': normal_pil})  # 形状: 元数据

    _rotate_meshes_by_source_front(meshes, args.source_front)

    result = scorer.compute_scores(meshes, images, metadata)
    if isinstance(result, tuple):
        scores, _grouped_meta = result
    else:
        scores = result

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'image', 'mesh', 'score'])
        for (name, img_path, mesh_path), sc in zip(pairs, scores):
            writer.writerow([name, img_path, mesh_path, f'{float(sc):.6f}'])
        if len(scores) > 0:
            mean_score = float(np.mean(scores))  # 形状: 标量
            writer.writerow(['mean', '', '', f'{mean_score:.6f}'])

    print(json.dumps({
        'num_samples': len(scores),  # 形状: 标量
        'mean_score': float(np.mean(scores)) if len(scores) > 0 else None,  # 形状: 可空标量
        'output_csv': args.output_csv,  # 形状: 字符串
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()


