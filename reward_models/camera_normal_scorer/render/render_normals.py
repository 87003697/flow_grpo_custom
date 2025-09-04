from typing import Any, List
import os
import sys
import torch

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)
from generators.trellis.renderers.mesh_renderer import MeshRenderer as GenMeshRenderer
from ..camera.estimate_utils import scale_intrinsics_to_square

from .adapter import to_mesh_extract, KiuiMeshLike


def render_normals_batched(meshes: List[Any], idxs: List[int], extri_all: torch.Tensor, intr_pix_all: torch.Tensor, R: int, W: int, device: torch.device) -> torch.Tensor:
    """使用参考渲染器批量渲染法线并映射到[-1,1]。

    功能:
        - 将 VGGT 输出的 OpenCV W2C(4x4) 转换为 OpenGL C2W(4x4)。
        - 将像素内参从 (H,W) 重标到 (R,R)。
        - 用参考渲染器渲染 normals，并线性映射到 [-1,1]。

    输入:
        meshes: mesh 列表。
        idxs: 当前分组的索引序列。
        extri_all: (K,4,4) OpenCV W2C。
        intr_pix_all: (K,3,3) 像素内参（基于 H×W）。
        R: 渲染分辨率；W: 原图宽。
        device: 设备。
    输出:
        n_mesh_all: (K,3,R,R)，范围 [-1,1]。

    参考:
        - 坐标系转换: `_reference_codes/VGGTObj/training/utils/coordinate_conversion.py` L21-L69
        - 渲染接口: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L179-L215
    """
    gen_renderer = GenMeshRenderer(  # 形状: 渲染器
        rendering_options={
            "resolution": int(R),
            "near": 0.01,
            "far": 10.0,
            "ssaa": 1,
        },
        device=str(device),
    )
    K = extri_all.shape[0]  # 形状: 标量
    n_mesh_list = []
    for j in range(K):
        mesh_ex = to_mesh_extract(meshes[idxs[j]], device)  # 形状: MeshExtractResult

        w2c34 = extri_all[j][:3, :]  # 形状: (3,4)
        bottom = torch.tensor([0, 0, 0, 1], dtype=w2c34.dtype, device=w2c34.device).view(1, 4)  # 形状: (1,4)
        w2c44 = torch.cat([w2c34, bottom], dim=0)  # 形状: (4,4)

        K_pix = scale_intrinsics_to_square(intr_pix_all[j], R, W)  # 形状: (3,3)

        out = gen_renderer.render(
            mesh=mesh_ex,  # 形状: MeshExtractResult
            extrinsics=w2c44,  # 形状: (4,4)
            intrinsics=K_pix,  # 形状: (3,3)
            return_types=["normal"],  # 形状: 列表
        )  # 形状: 字典
        img01 = out["normal"]  # 形状: (3,R,R) in [0,1]
        n_mesh = (img01 * 2.0 - 1.0).clamp(-1, 1)  # 形状: (3,R,R)
        n_mesh_list.append(n_mesh)

    n_mesh_all = torch.stack(n_mesh_list, dim=0)  # 形状: (K,3,R,R)
    return n_mesh_all


