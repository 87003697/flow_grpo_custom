from typing import Any
import torch
from generators.trellis.representations.mesh import MeshExtractResult


def to_mesh_extract(mesh: Any, device: torch.device) -> MeshExtractResult:
    """将任意 mesh 对象适配为 MeshExtractResult。
    输入: 可能含属性 v/f 或 vertices/faces
    输出: MeshExtractResult
    """
    if isinstance(mesh, MeshExtractResult):
        return mesh  # 形状: MeshExtractResult
    v_src = getattr(mesh, "v", getattr(mesh, "vertices", None))  # 形状: (V,3)
    f_src = getattr(mesh, "f", getattr(mesh, "faces", None))  # 形状: (F,3)
    v = torch.as_tensor(v_src, device=device, dtype=torch.float32)  # 形状: (V,3)
    f = torch.as_tensor(f_src, device=device, dtype=torch.int64)  # 形状: (F,3)
    return MeshExtractResult(vertices=v, faces=f)  # 形状: MeshExtractResult


def compose_white_background(normal01: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
    """以白色为背景进行合成，返回 (3,R,R) 的 0-1 范围法线图像。"""
    white = torch.ones_like(normal01)  # 形状: (3,R,R)
    return white * (1.0 - mask01.unsqueeze(0)) + normal01 * mask01.unsqueeze(0)  # 形状: (3,R,R)


