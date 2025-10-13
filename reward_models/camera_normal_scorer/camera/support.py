from typing import Any, List, Tuple
import importlib
import os
import sys
import torch

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)

from _reference_codes.VGGTObj.training.utils.mesh_renderer import MeshRenderer as RefMeshRenderer
from _reference_codes.VGGTObj.vggt.utils.pose_enc import extri_intri_to_pose_encoding

from ..render.adapter import to_mesh_extract, KiuiMeshLike


def load_fixed_poses_and_renderer(camera_config_py: str, img_size: int, device: torch.device):
    """加载固定视角配置，并返回参考渲染器。

    功能:
        - 动态导入 `camera_config_py`，读取 `get_camera_search_seven_view_config()` 中的 `predefined_poses`。
        - 创建参考渲染器 `MeshRenderer` 用于批量渲染 support。

    输入:
        camera_config_py: 相机预设配置脚本路径（.py）。
        img_size: 渲染尺寸（与 VGGT 训练一致的 518）。
        device: 渲染设备。
    输出:
        fixed_poses: 预定义视角参数列表。
        ref_renderer: 参考渲染器实例。

    参考:
        - 参考渲染器: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L38-L70, L102-L157
    """
    mod = importlib.import_module(camera_config_py.replace('/', '.').replace('.py', ''))  # 形状: 模块
    if hasattr(mod, 'get_camera_search_seven_view_config'):
        cfg_ref = mod.get_camera_search_seven_view_config()  # 形状: 配置
        fixed_poses = getattr(cfg_ref.render, 'predefined_poses', [])
    else:
        raise ValueError(f"未找到 get_camera_search_seven_view_config 于 {camera_config_py}")
    ref_renderer = RefMeshRenderer(img_size=int(img_size), device=str(device))  # 形状: 渲染器
    return fixed_poses, ref_renderer


# 合并到 build_support_batches：删除单样本函数，避免重复相机加载


def build_support_batches(meshes: List[Any], idxs: List[int], imgs_query: torch.Tensor, H: int, W: int, camera_config_py: str, camera_param_dim: int, img_size: int, device: torch.device):
    """批量构建 support。

    输入:
        meshes, idxs: mesh 列表与索引列表。
        imgs_query: (1,3,H,W)
        H, W: 图像尺寸。
        camera_config_py: 相机预设配置脚本。
        camera_param_dim: 9 或 12。
        img_size: 渲染尺寸。
        device: 设备。
    输出:
        images_batched: (K,S,3,H,W)
        support: (K,S-1,D)

    参考:
        - 参考渲染器: `_reference_codes/VGGTObj/training/utils/mesh_renderer.py` L102-L157, L179-L215
    """
    fixed_poses, ref_renderer = load_fixed_poses_and_renderer(camera_config_py, img_size, device)
    images_seqs = []  # 形状: 列表
    supports = []     # 形状: 列表
    # 预先采样固定相机，避免重复调用
    cams_fixed = ref_renderer.sample_camera_poses(num_random_views=0, predefined_poses=fixed_poses)  # 形状: 列表
    for j in idxs:
        mesh_ex = to_mesh_extract(meshes[j], device)  # 形状: MeshExtractResult
        mesh_kiui = KiuiMeshLike(mesh_ex.vertices, mesh_ex.faces)  # 形状: MeshLike
        sup_out = ref_renderer.render_mesh(
            mesh=mesh_kiui,
            cameras=cams_fixed,
            return_depth=False,
            return_normals=False,
            return_positions=False,
            return_masks=False,
        )
        images_s = sup_out['images'].to(device)  # 形状: (S-1,3,H,W)
        extr_s = sup_out['extrinsics'].to(device)  # 形状: (S-1,3,4)
        intr_s = sup_out['intrinsics'].to(device)  # 形状: (S-1,4)

        Ssup = images_s.shape[0]  # 形状: 标量
        intr_33 = torch.zeros(1, Ssup, 3, 3, device=device, dtype=intr_s.dtype)  # 形状: (1,S-1,3,3)
        intr_33[:, :, 0, 0] = intr_s[:, 0].unsqueeze(0)  # 形状: (1,S-1)
        intr_33[:, :, 1, 1] = intr_s[:, 1].unsqueeze(0)  # 形状: (1,S-1)
        intr_33[:, :, 0, 2] = intr_s[:, 2].unsqueeze(0)  # 形状: (1,S-1)
        intr_33[:, :, 1, 2] = intr_s[:, 3].unsqueeze(0)  # 形状: (1,S-1)
        intr_33[:, :, 2, 2] = 1.0  # 形状: (1,S-1)

        if int(camera_param_dim) == 9:
            pose_sup = extri_intri_to_pose_encoding(
                extr_s.unsqueeze(0),  # 形状: (1,S-1,3,4)
                intr_33,              # 形状: (1,S-1,3,3)
                image_size_hw=(H, W),
            )[0]  # 形状: (S-1,9)
        else:
            pose_sup = extr_s.reshape(Ssup, -1)  # 形状: (S-1,12)

        images_seq = torch.cat([images_s.unsqueeze(0), imgs_query.unsqueeze(0)], dim=1)  # 形状: (1,S,3,H,W)
        images_seqs.append(images_seq)  # 形状: 追加 (1,S,3,H,W)
        supports.append(pose_sup)      # 形状: 追加 (S-1,D)
    images_batched = torch.cat(images_seqs, dim=0).to(device)  # 形状: (K,S,3,H,W)
    support = torch.stack(supports, dim=0).to(device)  # 形状: (K,S-1,D)
    return images_batched, support


