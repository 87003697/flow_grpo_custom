"""CameraNormalScorer 相关复用工具。"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import torch
import trimesh
from PIL import Image

__all__ = [
    "load_glb_mesh_as_obj",
    "load_normal_pil_from_cache",
    "_rotate_meshes_by_source_front",
    "_cache_path_from_image",
]


def load_glb_mesh_as_obj(path: str) -> Any:
    mesh = trimesh.load(path, force="mesh")  # 形状: trimesh.Trimesh
    v = torch.from_numpy(np.asarray(mesh.vertices)).float()  # 形状: (V,3)
    f = torch.from_numpy(np.asarray(mesh.faces)).long()  # 形状: (F,3)
    return type("SimpleMesh", (), {"vertices": v, "faces": f})  # 形状: 简单对象


def _cache_path_from_image(image_path_or_name: str, cache_dir: str, normal_resolution: int) -> str:
    stem = os.path.splitext(os.path.basename(image_path_or_name))[0]  # 形状: 标量
    dir_r = os.path.join(cache_dir, f"R{int(normal_resolution)}")  # 形状: 标量
    return os.path.join(dir_r, f"{stem}.png")  # 形状: 标量


def load_normal_pil_from_cache(image_path: str, cache_dir: str, normal_resolution: int) -> Image.Image:
    """从缓存目录读取法线 PNG（[0,255] 编码），返回 PIL。"""
    cache_png = _cache_path_from_image(image_path, cache_dir, normal_resolution)  # 形状: 标量
    if not os.path.isfile(cache_png):
        raise FileNotFoundError(f"未找到法线缓存: {cache_png}")
    return Image.open(cache_png).convert("RGB")  # 形状: PIL(R,R,3)


def _rotate_meshes_by_source_front(meshes: List[Any], source_front: str) -> None:
    if len(meshes) == 0:
        return
    src = str(source_front)
    if src == "+z":
        return

    suffix = 0
    if len(src) > 0 and src[-1] in ("1", "2", "3"):
        suffix = int(src[-1])
        base = src[:-1]
    else:
        base = src

    first_vertices = getattr(meshes[0], "vertices", None)
    if not isinstance(first_vertices, torch.Tensor):
        if first_vertices is None and hasattr(meshes[0], "v"):
            first_vertices = getattr(meshes[0], "v")
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
        verts = getattr(mesh, "vertices", None)
        if verts is None and hasattr(mesh, "v"):
            verts = getattr(mesh, "v")
        if not isinstance(verts, torch.Tensor):
            continue
        rotated = verts @ T  # 形状: (V,3)
        mesh.vertices = rotated  # 形状: (V,3)
        if hasattr(mesh, "v"):
            mesh.v = rotated  # 形状: (V,3)


