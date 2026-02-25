"""
Sparse FlexiCubes mesh extractor for Trellis v1.
底层算法直接从 TripoSF import，避免代码重复和出错。
仅调整 SDF 预处理以匹配 Trellis decoder 输出。
"""
import torch

# 从 TripoSF 直接 import 底层算法（纯 torch 操作，无类型冲突）
from triposf.representations.mesh.flexicubes.flexicubes import FlexiCubes as TripoSFFlexiCubes
from triposf.representations.mesh.utils_cube import (
    sparse_cube2verts,
    get_sparse_attrs,
    get_defomed_verts,
    cube_corners,
)
# MeshExtractResult 也可以直接用 TripoSF 的
from triposf.representations.mesh import MeshExtractResult

from easydict import EasyDict as edict


class SparseFeatures2Mesh:
    """
    Trellis v1 专用稀疏 mesh 提取器。
    
    与 TripoSF 的差异：去掉 sdf * (4/res) 缩放，
    因为 Trellis decoder 训练时未做此缩放。
    """
    def __init__(self, device="cuda", res=64, use_color=False):
        self.device = device
        self.res = res
        self.use_color = use_color
        self.mesh_extractor = TripoSFFlexiCubes(device=device)  # 直接用 TripoSF 的稀疏 FlexiCubes
        self.sdf_bias = -1.0 / res
        self._calc_layout()

    def _calc_layout(self):
        layouts = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21},
        }
        if self.use_color:
            layouts['color'] = {'shape': (8, 6), 'size': 8 * 6}
        self.layouts = edict(layouts)
        start = 0
        for info in self.layouts.values():
            info['range'] = (start, start + info['size'])
            start += info['size']
        self.feats_channels = start

    def get_layout(self, feats, name):
        if name not in self.layouts:
            return None
        s, e = self.layouts[name]['range']
        return feats[:, s:e].reshape(-1, *self.layouts[name]['shape'])

    def __call__(self, cubefeats, training=False):
        assert not self.use_color, "Sparse version does not support color"

        coords = cubefeats.coords[:, 1:]  # (N, 3) 去掉 batch 维
        feats = cubefeats.feats            # (N, C)

        sdf, deform, color, weights = [
            self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']
        ]
        # ⚠️ 关键：不做 sdf * (4/res) 缩放，仅加 bias
        sdf += self.sdf_bias

        v_attrs = [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(  # 直接用 TripoSF 的函数
            coords, torch.cat(v_attrs, dim=-1), training=training
        )

        res_v = self.res + 1
        v_attrs_d_sparse, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d_sparse, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        sdf_d, deform_d = v_attrs_d_sparse[..., 0], v_attrs_d_sparse[..., 1:4]

        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)  # (V, 3)
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=x_nx3.device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1,), dtype=sdf_d.dtype, device=sdf_d.device)))

        # 构建稀疏 cube 索引
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]
        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).cuda()).reshape(-1)
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        exact_match_mask = mask_reg_c_sparse[reg_c] == reg_c_value
        reg_c[exact_match_mask == 0] = len(mask_reg_c_sparse)
        reg_c = reg_c.reshape(-1, 8)

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d_sparse[:, :12],
            alpha=weights_d_sparse[:, 12:20],
            gamma_f=weights_d_sparse[:, 20],
            cube_index_map=coords_dilate,
            training=training,
        )

        return MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)