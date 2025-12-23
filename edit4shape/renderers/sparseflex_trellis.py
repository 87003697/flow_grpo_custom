"""
Trellis Mesh Renderer using nvdiffrast.
Independent implementation without threestudio base dependency.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


@dataclass
class TrellisRendererConfig:
    """渲染器配置"""
    resolution: int = 512
    near: float = 0.01
    far: float = 100.0
    ssaa: int = 1
    bg_color: float = 0.0


class TrellisMeshRasterizer:
    """
    Trellis 专用光栅化器 (nvdiffrast)。
    不依赖 geometry.isosurface，直接渲染 MeshExtractResult。
    """

    def __init__(self, cfg: Optional[TrellisRendererConfig] = None, device: str = "cuda"):
        self.cfg = cfg or TrellisRendererConfig()
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)

    def render(
        self,
        mesh: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types: List[str] = ["mask", "depth", "normal", "color"],
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个 Mesh 的单视角。
        
        Args:
            mesh: MeshExtractResult
            extrinsics: (4, 4) W2C Matrix (OpenCV)
            intrinsics: (3, 3) Camera Intrinsics
        Returns:
            Dict[str, Tensor]: (H, W, C)
        """
        resolution = self.cfg.resolution
        ssaa = self.cfg.ssaa

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return self._get_empty_output(resolution, return_types)

        # 1. 构建矩阵 (OpenCV -> OpenGL)
        proj = self._get_projection_matrix(intrinsics, resolution, resolution)  # (4,4)
        mvp = proj @ extrinsics  # (4,4)

        # 2. 顶点变换
        vertices = mesh.vertices.unsqueeze(0)  # (1,Nv,3)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)  # (1,Nv,4)

        # Clip Space: v * MVP.T
        vertices_clip = torch.bmm(vertices_homo, mvp.unsqueeze(0).transpose(-1, -2))  # (1,Nv,4)
        
        # Camera Space (for depth): v * Ext.T
        vertices_cam = torch.bmm(vertices_homo, extrinsics.unsqueeze(0).transpose(-1, -2))  # (1,Nv,4)

        # 3. 光栅化
        faces_int = mesh.faces.int()
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa)
        )  # (1, H*ssaa, W*ssaa, 4)

        # 4. 插值与后处理
        out = {}
        for k in return_types:
            img = self._process_channel(k, rast, vertices_clip, vertices_cam, mesh, faces_int)
            
            # SSAA 下采样
            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                ).permute(0, 2, 3, 1)
            
            out[k] = img[0]  # (H,W,C)

        return out

    def _get_projection_matrix(self, K, h, w):
        """
        归一化 Intrinsics -> OpenGL Projection
        与 TRELLIS mesh_renderer.py 中的 intrinsics_to_projection 对齐。
        假设 K 来自 utils3d.torch.intrinsics_from_fov_xy，是归一化的 intrinsics。
        """
        fx, fy = K[0, 0], K[1, 1]  # [], []
        cx, cy = K[0, 2], K[1, 2]  # [], []
        n, f = self.cfg.near, self.cfg.far  # [], []

        ret = torch.zeros((4, 4), device=K.device, dtype=K.dtype)  # [4,4]
        ret[0, 0] = 2 * fx  # []
        ret[1, 1] = 2 * fy  # []
        ret[0, 2] = 2 * cx - 1  # []
        ret[1, 2] = -2 * cy + 1  # []
        ret[2, 2] = f / (f - n)  # []
        ret[2, 3] = n * f / (n - f)  # []
        ret[3, 2] = 1.0  # []
        return ret  # [4,4]

    def _process_channel(self, type_name, rast, v_clip, v_cam, mesh, faces):
        if type_name == "mask":
            return dr.antialias((rast[..., -1:] > 0).float(), rast, v_clip, faces)
            
        elif type_name == "depth":
            # TRELLIS 约定: 直接使用相机空间的 Z 坐标作为深度
            depth = v_cam[..., 2:3].contiguous()  # (1,Nv,1)
            img = dr.interpolate(depth, rast, faces)[0]  # (1,H,W,1)
            return dr.antialias(img, rast, v_clip, faces)  # (1,H,W,1)

        elif type_name == "color":
            # 优先使用 vertex_attrs 的前3通道 (RGB)
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 3:
                color = mesh.vertex_attrs[:, :3].contiguous()
                img = dr.interpolate(color, rast, faces)[0]
                return dr.antialias(img, rast, v_clip, faces)
            return torch.ones_like(rast[..., :3]) * 0.8

        elif type_name == "normal":
            # 简化：返回全0，如需真实法线需插值 mesh.vertex_normals
            return torch.zeros_like(rast[..., :3])

        return torch.zeros_like(rast[..., :1])

    def _get_empty_output(self, res, types):
        out = {}
        for t in types:
            c = 3 if t in ['color', 'normal'] else 1
            out[t] = torch.zeros((res, res, c), device=self.device)
        return out

