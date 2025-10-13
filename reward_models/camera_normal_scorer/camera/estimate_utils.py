import torch
from typing import Tuple

def normalize_intrinsics_to_R(intr_3x3: torch.Tensor, H: int, W: int, R: int) -> torch.Tensor:
    """将像素坐标内参 (B, 3, 3) 归一化到 R×R 渲染分辨率，返回 (B, 3, 3)。

    输入:
        intr_3x3: (B,3,3) 像素内参。
        H, W: 原图尺寸。
        R: 目标渲染分辨率（正方形）。
    输出:
        归一化内参 (B,3,3)。
    参考: 本文件函数。
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


def batch_estimate_camera(camera_estimator: torch.nn.Module, images_batched: torch.Tensor, support: torch.Tensor, H: int, W: int, R: int, cam_bs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """分批进行相机估计并聚合输出。

    输入:
        camera_estimator: 兼容 `VGGTSearchEstimator` 的估计器。
        images_batched: (K,S,3,H,W)
        support: (K,S-1,D)
        H, W: 图像尺寸。
        R: 渲染分辨率（用于内参归一化）。
        cam_bs: 相机估计分批大小。
    输出:
        extri_all: (K,4,4)
        intr_all: (K,3,3)
        intr_pix_all: (K,3,3)

    参考:
        - 归一化内参: 本文件 `normalize_intrinsics_to_R`
    """
    K = images_batched.shape[0]  # 形状: 标量
    extri_list, intr_list, intr_pix_list = [], [], []
    for s in range(0, K, int(cam_bs)):
        e = min(K, s + int(cam_bs))  # 形状: 标量
        extri_4x4, intr_3x3 = camera_estimator.estimate(images_batched[s:e], support[s:e], (H, W))  # 形状: (b,4,4),(b,3,3)
        intr_R = normalize_intrinsics_to_R(intr_3x3, H, W, R)  # 形状: (b,3,3)
        extri_list.append(extri_4x4)  # 形状: 追加
        intr_list.append(intr_R)      # 形状: 追加
        intr_pix_list.append(intr_3x3)  # 形状: 追加
    extri_all = torch.cat(extri_list, dim=0)  # 形状: (K,4,4)
    intr_all = torch.cat(intr_list, dim=0)    # 形状: (K,3,3)
    intr_pix_all = torch.cat(intr_pix_list, dim=0)  # 形状: (K,3,3)
    return extri_all, intr_all, intr_pix_all

 

