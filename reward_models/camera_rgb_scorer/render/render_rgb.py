from typing import Any, List
import os
import sys
import torch
import torch.nn.functional as F

# 复用 camera_normal_scorer 的渲染基础设施和辅助函数
from reward_models.camera_normal_scorer.render.render_normals import create_camera_from_c2w_K
from reward_models.camera_normal_scorer.render.adapter import to_mesh_extract, KiuiMeshLike

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from _reference_codes.VGGTObj.training.utils.coordinate_conversion import CoordinateConverter


def render_rgb_batched(
    meshes: List[Any], 
    idxs: List[int], 
    extri_all: torch.Tensor, 
    intr_pix_all: torch.Tensor, 
    img_size_for_K: int, 
    R: int, 
    device: torch.device,
    renderer: RefMeshRenderer
) -> torch.Tensor:
    """使用参考渲染器批量渲染 RGB 图像并映射到 [0,1]（使用像素内参）。

    功能:
        - 将 VGGT 输出的 OpenCV W2C(4x4) 取逆得到 C2W(4x4)。
        - 直接使用像素内参 K_pix；参考渲染器内部根据 img_size 构建投影矩阵。
        - 渲染返回 RGB 图像（值域 [0,1]），无需映射。

    输入:
        meshes: mesh 列表。
        idxs: 当前分组的索引序列。
        extri_all: (K,4,4) OpenCV W2C。
        intr_pix_all: (K,3,3) 像素内参（H×W 基准）。
        img_size_for_K: 渲染器使用的图像尺寸（需与 K 像素坐标系一致）。
        R: 目标 RGB 分辨率（如需与 img_size_for_K 不同则重采样）。
        device: 设备。
    输出:
        rgb_mesh_all: (K,3,R,R)，范围 [0,1]。

    参考:
        - 基于 camera_normal_scorer/render/render_normals.py
        - 关键差异：return_normals=False，获取 'images' 而非 'normals'
    """
    ref_renderer = renderer  # 形状: 参考渲染器（外部必须传入）
    K = extri_all.shape[0]  # 形状: 标量
    rgb_mesh_list = []
    for j in range(K):
        mesh_ex = to_mesh_extract(meshes[idxs[j]], device)  # 形状: MeshExtractResult
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: 适配到 kiui 接口

        w2c34 = extri_all[j][:3, :]  # 形状: (3,4)
        # 官方函数：OpenCV W2C(3x4) -> OpenGL C2W(4x4)
        w2c_bv = w2c34.unsqueeze(0).unsqueeze(0)  # 形状: (1,1,3,4)
        c2w_bv = CoordinateConverter.opencv_w2c_to_opengl_c2w(w2c_bv)  # 形状: (1,1,4,4)
        c2w44 = c2w_bv[0, 0]  # 形状: (4,4)

        K_pix = intr_pix_all[j]  # 形状: (3,3)
        
        # 创建 OrbitCamera（复用 render_normals 中的辅助函数）
        camera = create_camera_from_c2w_K(
            c2w=c2w44.cpu().numpy(),  # 形状: (4,4)
            K=K_pix.cpu().numpy(),    # 形状: (3,3)
            H=int(img_size_for_K),
            W=int(img_size_for_K)
        )  # 形状: OrbitCamera

        out = ref_renderer.render_mesh(
            mesh=mesh_kiui,  # 形状: KiuiMeshLike
            cameras=[camera],  # 形状: List[OrbitCamera]
            return_depth=False,
            return_normals=False,  # 关键差异：不返回法线
            return_positions=False,
            return_masks=False,
        )
        rgb01 = out['images'][0]  # 形状: (3,H,H) in [0,1]，关键差异：获取 'images'
        # 无需映射，已经是 [0,1]
        if int(img_size_for_K) != int(R):
            rgb01 = F.interpolate(rgb01.unsqueeze(0), size=(int(R), int(R)), mode='bilinear', align_corners=False).squeeze(0)  # 形状: (3,R,R)
        rgb_mesh_list.append(rgb01)

    rgb_mesh_all = torch.stack(rgb_mesh_list, dim=0)  # 形状: (K,3,R,R)
    return rgb_mesh_all
