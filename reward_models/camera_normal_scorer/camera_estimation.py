from typing import Tuple
import torch


def extrinsics34_to44(extri_3x4: torch.Tensor) -> torch.Tensor:
    """将 (B, 3, 4) 扩展为 (B, 4, 4) 的 W2C 矩阵，最后一行固定为 [0,0,0,1]。

    输入:
        extri_3x4: (B,3,4) OpenCV W2C。
    输出:
        extri_4x4: (B,4,4) OpenCV W2C。
    参考: 无（基础线性代数）。
    """
    B = extri_3x4.shape[0]  # 形状: 标量
    bottom = torch.tensor([0, 0, 0, 1], dtype=extri_3x4.dtype, device=extri_3x4.device).view(1, 1, 4)  # 形状: (1,1,4)
    extri_4x4 = torch.cat([extri_3x4, bottom.expand(B, -1, -1)], dim=-2)  # 形状: (B,4,4)
    return extri_4x4  # 形状: (B,4,4)


def normalize_intrinsics_to_R(intr_3x3: torch.Tensor, H: int, W: int, R: int) -> torch.Tensor:
    """将像素坐标内参 (B, 3, 3) 归一化到 R×R 渲染分辨率，返回 (B, 3, 3)。

    输入:
        intr_3x3: (B,3,3) 像素内参。
        H, W: 原图尺寸。
        R: 目标渲染分辨率（正方形）。
    输出:
        归一化内参 (B,3,3)。
    参考: 无（常规相机内参缩放）。
    """
    fx = intr_3x3[:, 0, 0] / W  # 形状: (B,)
    fy = intr_3x3[:, 1, 1] / H  # 形状: (B,)
    cx = intr_3x3[:, 0, 2] / W  # 形状: (B,)
    cy = intr_3x3[:, 1, 2] / H  # 形状: (B,)
    intr_norm = intr_3x3.clone()  # 形状: (B,3,3)
    intr_norm[:, 0, 0] = fx  # 形状: (B,)
    intr_norm[:, 1, 1] = fy  # 形状: (B,)
    intr_norm[:, 0, 2] = cx  # 形状: (B,)
    intr_norm[:, 1, 2] = cy  # 形状: (B,)
    return intr_norm  # 形状: (B,3,3)


@torch.no_grad()
def estimate_camera(
    images_tensor: torch.Tensor,
    support_cameras: torch.Tensor,
    model: torch.nn.Module,
    image_hw: Tuple[int, int],
):
    """使用 VGGT camera-search 估计 query 视角相机并转换矩阵。

    输入:
        images_tensor: (B,S,3,H,W)
        support_cameras: (B,S-1,D)（D=9 pose encoding 或 12 展平外参）。
        model: VGGT camera-search 模型。
        image_hw: (H,W) 像素尺寸。
    输出:
        extri_4x4: (B,4,4) OpenCV W2C。
        intr_3x3: (B,3,3) 像素内参（基于 H×W）。
    参考:
        - 姿态反解: `_reference_codes/VGGTObj/vggt/utils/pose_enc.py` L62-L125
    """
    from _reference_codes.VGGTObj.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    H, W = image_hw  # 形状: 标量, 标量
    preds = model(images_tensor, support_cameras)  # 形状: dict，含 'pose_enc'
    pose_enc_q = preds["pose_enc"][:, -1:, :]  # 形状: (B,1,D)
    extri_b1, intr_b1 = pose_encoding_to_extri_intri(pose_enc_q, (H, W))  # 形状: (B,1,3,4),(B,1,3,3)
    extri_3x4 = extri_b1[:, 0]  # 形状: (B,3,4)
    intr_3x3 = intr_b1[:, 0]   # 形状: (B,3,3)
    extri_4x4 = extrinsics34_to44(extri_3x4)  # 形状: (B,4,4)
    return extri_4x4, intr_3x3  # 形状: (B,4,4),(B,3,3)


def spherical_to_w2c_opencv(distance: float, elevation_deg: float, azimuth_deg: float) -> torch.Tensor:
    """从球坐标生成 OpenCV W2C 3x4 矩阵。

    输入:
        distance/elevation_deg/azimuth_deg: 球坐标参数。
    输出:
        extri: (3,4) OpenCV W2C。
    参考: 无（标准视角生成）。
    """
    r = float(distance)  # 形状: 标量
    ele = float(elevation_deg) * torch.pi / 180.0  # 形状: 标量
    azi = float(azimuth_deg) * torch.pi / 180.0  # 形状: 标量
    ce, se = torch.cos(torch.tensor(ele)), torch.sin(torch.tensor(ele))  # 形状: 标量, 标量
    ca, sa = torch.cos(torch.tensor(azi)), torch.sin(torch.tensor(azi))  # 形状: 标量, 标量
    # 目标相机位置（看向原点）
    cx = r * ce * sa  # 形状: 标量
    cy = r * se  # 形状: 标量
    cz = r * ce * ca  # 形状: 标量
    C = torch.stack([cx, cy, cz], dim=0)  # 形状: (3,)
    # z 轴（前向，指向世界原点）
    z = torch.nn.functional.normalize(-C, dim=0)  # 形状: (3,)
    up = torch.tensor([0.0, 1.0, 0.0])  # 形状: (3,)
    x = torch.nn.functional.normalize(torch.cross(up, z), dim=0)  # 形状: (3,)
    y = torch.cross(z, x)  # 形状: (3,)
    R = torch.stack([x, y, z], dim=0)  # 形状: (3,3)
    t = (-R @ C.view(3, 1)).view(3)  # 形状: (3,)
    extri = torch.cat([R, t.view(3, 1)], dim=1)  # 形状: (3,4)
    return extri  # 形状: (3,4)


def fovy_to_intrinsics(fovy_deg: float, H: int, W: int) -> torch.Tensor:
    """给定垂直视场角，返回像素内参 (3,3)。

    输入:
        fovy_deg: 垂直视场角（度）。
        H, W: 图像尺寸。
    输出:
        K: (3,3) 像素内参。
    参考: 无（基础相机模型）。
    """
    fovy_rad = float(fovy_deg) * 3.141592653589793 / 180.0  # 形状: 标量
    fy = (float(H) / 2.0) / (torch.tan(torch.tensor(fovy_rad / 2.0, dtype=torch.float32)))  # 形状: 标量
    fx = fy  # 形状: 标量（假定正方形像素，保持纵横比1）
    cx = float(W) / 2.0  # 形状: 标量
    cy = float(H) / 2.0  # 形状: 标量
    K = torch.zeros(3, 3, dtype=torch.float32)
    K[0, 0] = fx  # 形状: 标量
    K[1, 1] = fy  # 形状: 标量
    K[0, 2] = cx  # 形状: 标量
    K[1, 2] = cy  # 形状: 标量
    K[2, 2] = 1.0  # 形状: 标量
    return K  # 形状: (3,3)


