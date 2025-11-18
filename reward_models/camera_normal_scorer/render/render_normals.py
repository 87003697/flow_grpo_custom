from typing import Any, List, Dict
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)
from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from _reference_codes.VGGTObj.training.utils.coordinate_conversion import CoordinateConverter
from kiui.cam import OrbitCamera

from .adapter import to_mesh_extract, KiuiMeshLike


def create_camera_from_c2w_K(c2w: np.ndarray, K: np.ndarray, H: int, W: int) -> OrbitCamera:
    """从 c2w 矩阵和内参矩阵创建 OrbitCamera。
    
    参数:
        c2w: (4,4) Camera-to-World 矩阵
        K: (3,3) 内参矩阵
        H, W: 图像高度和宽度
    返回:
        OrbitCamera 对象
    """
    # 从 K 提取焦距并计算 fovy
    # OrbitCamera.intrinsics 定义: focal = H / (2 * tan(fovy / 2))
    # 所以: fovy = 2 * arctan(H / (2 * focal))
    fy = float(K[1, 1])  # 焦距 fy
    fovy_rad = 2.0 * np.arctan(H / (2.0 * fy))  # 计算 fovy（弧度）
    fovy_deg = np.rad2deg(fovy_rad)  # 转为角度
    
    # 创建基础相机（使用计算出的 fovy）
    camera = OrbitCamera(W=W, H=H, fovy=float(fovy_deg))
    
    # 从 c2w 提取旋转和平移
    # c2w[:3, :3] 是旋转矩阵, c2w[:3, 3] 是平移向量
    rot_matrix = c2w[:3, :3]  # (3,3)
    translation = c2w[:3, 3]  # (3,)
    
    # 设置 OrbitCamera 的内部状态
    # 注意：OrbitCamera 假设相机在原点看向 -Z 方向
    # pose = T(-center) @ R @ T([0,0,radius])
    # 所以 c2w = T(-center) @ R @ T([0,0,radius])
    # 我们需要反推 center, R, radius
    
    # 简化：直接设置旋转和中心
    camera.rot = Rotation.from_matrix(rot_matrix)  # 设置旋转
    camera.center = np.array([0, 0, 0], dtype=np.float32)  # 简化：假设看向原点
    
    # 从平移推算 radius（相机到中心的距离）
    # 在 OrbitCamera 中，pose[:3, 3] = R @ [0, 0, radius] - center
    # 所以 radius ≈ ||translation||
    camera.radius = float(np.linalg.norm(translation))  # 设置距离
    
    return camera


def render_normals_batched(meshes: List[Any], idxs: List[int], extri_all: torch.Tensor, intr_pix_all: torch.Tensor, img_size_for_K: int, R: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """使用参考渲染器批量渲染相机坐标法线并映射到[-1,1]，并返回前景掩码（使用像素内参）。

    功能:
        - 将 VGGT 输出的 OpenCV W2C(4x4) 取逆得到 C2W(4x4)。
        - 直接使用像素内参 K_pix；参考渲染器内部根据 img_size 构建投影矩阵。
        - 渲染返回相机坐标法线（带白背景），再映射到 [-1,1]。

    输入:
        meshes: mesh 列表。
        idxs: 当前分组的索引序列。
        extri_all: (K,4,4) OpenCV W2C。
        intr_pix_all: (K,3,3) 像素内参（H×W 基准）。
        img_size_for_K: 渲染器使用的图像尺寸（需与 K 像素坐标系一致）。
        R: 目标法线分辨率（如需与 img_size_for_K 不同则重采样）。
        device: 设备。
    输出:
        n_mesh_all: (K,3,R,R)，范围 [-1,1]。
        mask_all:   (K,R,R)  ，bool 前景掩码。

    参考:
        - 坐标系转换: `_reference_codes/VGGTObj/training/utils/coordinate_conversion.py` L21-L69
        - 渲染接口: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L179-L215
    """
    ref_renderer = RefMeshRenderer(img_size=int(img_size_for_K), device=str(device))  # 形状: 参考渲染器
    K = extri_all.shape[0]  # 形状: 标量
    n_mesh_list = []
    mask_list = []
    for j in range(K):
        mesh_ex = to_mesh_extract(meshes[idxs[j]], device)  # 形状: MeshExtractResult
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: 适配到 kiui 接口

        w2c34 = extri_all[j][:3, :]  # 形状: (3,4)
        # 官方函数：OpenCV W2C(3x4) -> OpenGL C2W(4x4)
        w2c_bv = w2c34.unsqueeze(0).unsqueeze(0)  # 形状: (1,1,3,4)
        c2w_bv = CoordinateConverter.opencv_w2c_to_opengl_c2w(w2c_bv)  # 形状: (1,1,4,4)
        c2w44 = c2w_bv[0, 0]  # 形状: (4,4)

        K_pix = intr_pix_all[j]  # 形状: (3,3)
        
        # 创建 OrbitCamera
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
            return_normals=True,
            return_positions=False,
            return_masks=True,
        )
        n01 = out['normals'][0]  # 形状: (3,H,H) in [0,1]
        mB = out['masks'][0].to(torch.bool)  # 形状: (H,H)
        n11 = (n01 * 2.0 - 1.0).clamp(-1, 1)  # 形状: (3,H,H)
        n_mesh_list.append(n11)
        mask_list.append(mB)

    n_mesh_all = torch.stack(n_mesh_list, dim=0)  # 形状: (K,3,R,R)
    mask_all = torch.stack(mask_list, dim=0)      # 形状: (K,R,R)
    return n_mesh_all, mask_all


def render_normals_predefined(
    meshes: List[Any],
    idxs: List[int],
    pose_list: List[Dict[str, float]],
    img_size: int,
    R: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    """使用固定相机参数渲染法线。"""
    renderer = RefMeshRenderer(img_size=int(img_size), device=str(device))
    cams = renderer.sample_camera_poses(num_random_views=0, predefined_poses=pose_list)
    normals_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    mesh_indices: List[int] = []
    for mesh_idx in idxs:
        mesh_ex = to_mesh_extract(meshes[mesh_idx], device)
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)
        out = renderer.render_mesh(
            mesh=mesh_kiui,
            cameras=cams,
            return_depth=False,
            return_normals=True,
            return_positions=False,
            return_masks=True,
        )
        n01 = out["normals"].to(device)
        n11 = (n01 * 2.0 - 1.0).clamp(-1, 1)
        masks = out["masks"].to(device).to(torch.bool)
        normals_list.append(n11)
        masks_list.append(masks)
        mesh_indices.extend([mesh_idx] * len(cams))
    return torch.cat(normals_list, dim=0), torch.cat(masks_list, dim=0), mesh_indices


