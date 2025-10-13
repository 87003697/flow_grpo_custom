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
from functools import lru_cache
from _reference_codes.VGGTObj.vggt.utils.pose_enc import extri_intri_to_pose_encoding

from ..render.adapter import to_mesh_extract, KiuiMeshLike


@lru_cache(maxsize=None)
def _get_cached_renderer(img_size: int, device_str: str):
    return RefMeshRenderer(img_size=int(img_size), device=device_str)


def load_fixed_poses_and_renderer(camera_config_py: str, img_size: int, device: torch.device):
    """仅支持 "module.path:function_name"，无检查/回退。"""
    module_path, fn_name = camera_config_py.split(':', 1)
    mod = importlib.import_module(module_path.replace('/', '.').replace('.py', ''))
    cfg = mod.__dict__[fn_name]()
    fixed_poses = cfg.render.predefined_poses
    return fixed_poses, _get_cached_renderer(int(img_size), str(device))


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
        # 使用 OrbitCamera 相机分支（与远端一致）
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


