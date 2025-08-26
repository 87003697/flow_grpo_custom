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

class KiuiMeshLike:
    """简单的 kiui 兼容 mesh 适配器，提供 .v/.f（可选 .vn）。

    输入:
        v: (V,3) float32 顶点
        f: (F,3) int64 面
    """
    def __init__(self, v: torch.Tensor, f: torch.Tensor) -> None:
        self.v = v  # 形状: (V,3)
        self.f = f  # 形状: (F,3)
        self.vn = None

