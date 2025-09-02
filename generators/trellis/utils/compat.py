from typing import List, Dict, Union
import torch
import trimesh
from PIL import Image
from ..modules import sparse as sp


def _to_trimesh(vertices, faces) -> trimesh.Trimesh:
    """将 (vertices, faces) 转为 trimesh.Trimesh。
    - 支持 torch.Tensor 或 numpy 数组输入
    - 不做兜底，仅负责类型转换与构造
    """
    if torch.is_tensor(vertices):
        vertices = vertices.cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.cpu().numpy()
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def convert_trellis_to_trimesh(decoded: Union[Dict, List, trimesh.Trimesh, object]) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []
    if isinstance(decoded, dict):
        mesh_data = decoded.get('mesh')
        if mesh_data is None:
            raise ValueError("decode_slat 输出缺少 'mesh' 键")
        if isinstance(mesh_data, list):
            for m in mesh_data:
                if isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
                else:
                    v = getattr(m, 'vertices', None)
                    f = getattr(m, 'faces', None)
                    if v is None or f is None:
                        raise TypeError("mesh对象缺少 vertices/faces 属性")
                    meshes.append(_to_trimesh(v, f))
        else:
            m = mesh_data
            if isinstance(m, trimesh.Trimesh):
                meshes.append(m)
            else:
                v = getattr(m, 'vertices', None)
                f = getattr(m, 'faces', None)
                if v is None or f is None:
                    raise TypeError("mesh对象缺少 vertices/faces 属性")
                meshes.append(_to_trimesh(v, f))
        return meshes

    if isinstance(decoded, list):
        if all(isinstance(x, trimesh.Trimesh) for x in decoded):
            return decoded
        out: List[trimesh.Trimesh] = []
        for m in decoded:
            v = getattr(m, 'vertices', None)
            f = getattr(m, 'faces', None)
            if v is None or f is None:
                raise TypeError("列表中的元素不是可识别的 mesh 表示")
            out.append(_to_trimesh(v, f))
        return out

    if isinstance(decoded, sp.SparseTensor):
        raise TypeError("收到 SparseTensor。请先调用 decode_slat(slat, formats=['mesh'])")

    if isinstance(decoded, trimesh.Trimesh):
        return [decoded]

    v = getattr(decoded, 'vertices', None)
    f = getattr(decoded, 'faces', None)
    if v is not None and f is not None:
        return [_to_trimesh(v, f)]
    raise TypeError("未知的 mesh 表示类型，无法转换为 trimesh.Trimesh")


def convert_trellis_to_kiuimesh(decoded: Union[Dict, List, trimesh.Trimesh]):
    try:
        from kiui.mesh import Mesh as KiuiMesh
    except Exception as e:
        raise RuntimeError("需要安装 kiui 以使用 convert_trellis_to_kiuimesh") from e

    meshes_trimesh = convert_trellis_to_trimesh(decoded)
    out: List[KiuiMesh] = []
    for m in meshes_trimesh:
        v = torch.tensor(m.vertices, dtype=torch.float32)
        f = torch.tensor(m.faces, dtype=torch.int32)
        out.append(KiuiMesh(v=v, f=f, device=v.device))
    return out


